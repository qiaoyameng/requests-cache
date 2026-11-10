"""Tests for consumption-driven caching of streamed responses (``stream_cache=True``)"""

import gzip
from collections import deque

import pytest
from requests import Response
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict
from urllib3.response import HTTPResponse

from requests_cache import CachedSession

STREAM_URL = 'http://count.local/stream'
CHUNKS = [b'chunk-0|', b'chunk-1|', b'chunk-2|']
BODY = b''.join(CHUNKS)


class CountingStream:
    """File-like network source that counts read() calls and can fail mid-stream"""

    def __init__(self, chunks, fail_after=None):
        self._chunks = deque(chunks)
        self._buffer = b''
        self._fail_after = fail_after
        self.reads = 0
        self.closed = False

    def read(self, amt=-1):
        self.reads += 1
        if self._fail_after is not None and self.reads > self._fail_after:
            raise OSError('simulated network failure')
        if amt is None or amt < 0:
            data = self._buffer + b''.join(self._chunks)
            self._buffer = b''
            self._chunks.clear()
            return data
        while len(self._buffer) < amt and self._chunks:
            self._buffer += self._chunks.popleft()
        data, self._buffer = self._buffer[:amt], self._buffer[amt:]
        return data

    def close(self):
        self.closed = True


class CountingAdapter(BaseAdapter):
    """Adapter that serves a chunked stream from a countable network source"""

    def __init__(self, chunks=CHUNKS, headers=None, fail_after=None, **raw_kwargs):
        super().__init__()
        self._chunks = chunks
        self._headers = headers or {}
        self._fail_after = fail_after
        self._raw_kwargs = raw_kwargs
        self.calls = 0
        self.streams = []

    def send(self, request, **kwargs):
        self.calls += 1
        stream = CountingStream(self._chunks, fail_after=self._fail_after)
        self.streams.append(stream)
        headers = {
            'Content-Type': 'text/plain',
            'Content-Length': str(sum(len(c) for c in self._chunks)),
            **self._headers,
        }
        raw = HTTPResponse(
            body=stream,
            status=200,
            headers=headers,
            preload_content=False,
            **self._raw_kwargs,
        )
        response = Response()
        response.status_code = 200
        response.headers = CaseInsensitiveDict(headers)
        response.raw = raw
        response.url = request.url
        response.request = request
        response.encoding = 'utf-8'
        return response

    def close(self):
        pass

    @property
    def total_reads(self):
        return sum(s.reads for s in self.streams)


@pytest.fixture(params=['memory', 'sqlite'])
def stream_session(request, tmp_path):
    """CachedSession with stream_cache enabled, backed by a countable network source"""
    kwargs = (
        {'backend': 'memory'}
        if request.param == 'memory'
        else {'cache_name': str(tmp_path / 'test.db'), 'backend': 'sqlite'}
    )
    session = CachedSession(stream_cache=True, **kwargs)
    adapter = CountingAdapter()
    session.mount('http://', adapter)
    session.counting_adapter = adapter
    yield session
    session.close()


def test_headers_returned_immediately_without_reading_body(stream_session):
    """The caller gets response headers right away; the library must not read ahead"""
    response = stream_session.get(STREAM_URL, stream=True)

    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'text/plain'
    assert response._content_consumed is False
    assert stream_session.counting_adapter.total_reads == 0
    response.close()


def test_read_one_chunk_then_close(stream_session):
    """Consuming a single chunk and then closing must not produce a cache entry"""
    response = stream_session.get(STREAM_URL, stream=True)
    chunk = next(response.iter_content(chunk_size=8))
    assert chunk == CHUNKS[0]
    assert stream_session.counting_adapter.total_reads == 1
    response.close()

    assert len(stream_session.cache.responses) == 0

    # Second request goes back to the network
    response = stream_session.get(STREAM_URL, stream=True)
    assert response.from_cache is False
    response.close()
    assert stream_session.counting_adapter.calls == 2
    assert len(stream_session.cache.responses) == 0


def test_read_all_chunks(stream_session):
    """Fully consuming the response publishes a cacheable entry"""
    response = stream_session.get(STREAM_URL, stream=True)
    assert b''.join(response.iter_content(chunk_size=8)) == BODY
    # One network read per chunk; no eager or duplicate reads
    assert stream_session.counting_adapter.total_reads == len(CHUNKS)
    assert len(stream_session.cache.responses) == 1

    # Second request is served from the cache, with no further network activity
    response = stream_session.get(STREAM_URL, stream=True)
    assert response.from_cache is True
    assert response.content == BODY
    assert stream_session.counting_adapter.calls == 1


def test_read_via_content_property(stream_session):
    """response.content is recorded exactly once and cached"""
    response = stream_session.get(STREAM_URL, stream=True)
    assert response.content == BODY
    assert len(stream_session.cache.responses) == 1

    cached_response = stream_session.get(STREAM_URL, stream=True)
    assert cached_response.from_cache is True
    assert cached_response.content == BODY
    assert stream_session.counting_adapter.calls == 1


def test_read_via_raw(stream_session):
    """A single raw.read() to end of stream is recorded and cached"""
    response = stream_session.get(STREAM_URL, stream=True)
    assert response.raw.read() == BODY
    assert len(stream_session.cache.responses) == 1

    cached_response = stream_session.get(STREAM_URL, stream=True)
    assert cached_response.from_cache is True
    assert cached_response.content == BODY
    assert stream_session.counting_adapter.calls == 1


def test_read_error_midway(stream_session):
    """A read error partway through the body must not produce a cache entry"""
    stream_session.mount('http://', CountingAdapter(fail_after=2))
    adapter = stream_session.counting_adapter = stream_session.adapters['http://']

    response = stream_session.get(STREAM_URL, stream=True)
    with pytest.raises(Exception):
        b''.join(response.iter_content(chunk_size=8))

    assert len(stream_session.cache.responses) == 0
    assert adapter.total_reads == 3  # 2 successful chunks + the failed read

    # Second request misses the cache and hits the network again
    response = stream_session.get(STREAM_URL, stream=True)
    assert response.from_cache is False
    response.close()
    assert adapter.calls == 2
    assert len(stream_session.cache.responses) == 0


def test_content_length_mismatch(stream_session):
    """A body shorter than the declared Content-Length must not be cached"""
    stream_session.mount(
        'http://',
        CountingAdapter(
            headers={'Content-Length': str(len(BODY) + 10)},
            enforce_content_length=False,
        ),
    )
    adapter = stream_session.counting_adapter = stream_session.adapters['http://']

    response = stream_session.get(STREAM_URL, stream=True)
    assert b''.join(response.iter_content(chunk_size=8)) == BODY
    assert len(stream_session.cache.responses) == 0

    response = stream_session.get(STREAM_URL, stream=True)
    assert response.from_cache is False
    response.close()
    assert adapter.calls == 2


def test_old_cache_entry_retained_until_new_response_completed(stream_session):
    """A previously cached response stays in place while its replacement is still streaming"""
    response = stream_session.get(STREAM_URL)
    assert response.content == BODY
    cache_key = response.cache_key

    stream_session.mount('http://', CountingAdapter(chunks=[b'updated-0|', b'updated-1|']))
    adapter = stream_session.counting_adapter = stream_session.adapters['http://']

    # Force-refresh with a streamed response; the old entry is still there mid-stream
    response = stream_session.get(STREAM_URL, stream=True, force_refresh=True)
    assert response.from_cache is False
    assert stream_session.cache.get_response(cache_key).content == BODY

    # Only after the new response is fully consumed is the old entry replaced
    assert b''.join(response.iter_content(chunk_size=8)) == b'updated-0|updated-1|'
    assert stream_session.cache.get_response(cache_key).content == b'updated-0|updated-1|'
    assert adapter.calls == 1


def test_large_body_spills_to_temp_file(stream_session):
    """Bodies larger than the memory threshold are staged in a temp file, then cached"""
    stream_session.settings.stream_cache_memory_threshold = 4
    response = stream_session.get(STREAM_URL, stream=True)
    recorder = response.raw

    assert b''.join(response.iter_content(chunk_size=8)) == BODY
    assert recorder._spool._rolled  # Spilled over from memory to a temp file
    assert len(stream_session.cache.responses) == 1

    cached_response = stream_session.get(STREAM_URL, stream=True)
    assert cached_response.from_cache is True
    assert cached_response.content == BODY


def test_compressed_response(stream_session):
    """Compressed responses are cached at the same (decoded) byte layer that
    CachedResponse replays"""
    compressed = gzip.compress(BODY)
    chunks = [compressed[i : i + 7] for i in range(0, len(compressed), 7)]
    stream_session.mount(
        'http://',
        CountingAdapter(
            chunks=chunks,
            headers={'Content-Encoding': 'gzip', 'Content-Length': str(len(compressed))},
        ),
    )
    adapter = stream_session.counting_adapter = stream_session.adapters['http://']

    response = stream_session.get(STREAM_URL, stream=True)
    assert b''.join(response.iter_content(chunk_size=8)) == BODY
    assert len(stream_session.cache.responses) == 1

    cached_response = stream_session.get(STREAM_URL, stream=True)
    assert cached_response.from_cache is True
    assert cached_response.content == BODY
    assert cached_response.text == BODY.decode('utf-8')
    assert adapter.calls == 1


def test_undecoded_read_of_compressed_response_not_cached(stream_session):
    """Reading raw compressed bytes can't be replayed consistently; don't cache"""
    compressed = gzip.compress(BODY)
    stream_session.mount(
        'http://',
        CountingAdapter(
            chunks=[compressed],
            headers={'Content-Encoding': 'gzip', 'Content-Length': str(len(compressed))},
        ),
    )
    adapter = stream_session.counting_adapter = stream_session.adapters['http://']

    response = stream_session.get(STREAM_URL, stream=True)
    assert response.raw.read(decode_content=False) == compressed
    assert len(stream_session.cache.responses) == 0

    response = stream_session.get(STREAM_URL, stream=True)
    assert response.from_cache is False
    response.close()
    assert adapter.calls == 2


def test_stream_cache_disabled_by_default():
    """Without stream_cache, streamed responses are buffered up front (existing behavior)"""
    session = CachedSession(backend='memory')
    adapter = CountingAdapter()
    session.mount('http://', adapter)

    response = session.get(STREAM_URL, stream=True)
    # The cache layer eagerly reads the full body, even though the caller hasn't
    assert adapter.total_reads > 0
    assert len(session.cache.responses) == 1

    cached_response = session.get(STREAM_URL, stream=True)
    assert cached_response.from_cache is True
    assert cached_response.content == BODY
    session.close()
