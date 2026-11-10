"""Consumption-driven caching for streamed (``stream=True``) responses"""

from __future__ import annotations

from logging import getLogger
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Optional

from requests import Response

from ..policy.settings import DEFAULT_STREAM_CACHE_MEMORY_THRESHOLD
from .raw_response import _reset_fp

if TYPE_CHECKING:
    from ..backends import BaseCache

logger = getLogger(__name__)


class ConsumedStreamRecorder:
    """Wraps the raw urllib3 response of a streaming response to enable consumption-driven
    caching.

    Instead of reading the full response body up front, this records bytes as the caller reads
    them, via one of the explicitly supported read methods:

    * :py:meth:`requests.Response.iter_content`
    * :py:attr:`requests.Response.content`
    * :py:meth:`urllib3.response.HTTPResponse.read` and
      :py:meth:`~urllib3.response.HTTPResponse.stream` (via ``response.raw``)

    All of these funnel through :py:meth:`read` here, so each chunk is recorded exactly once.
    A cacheable response is published only after the body has been fully consumed. Closing the
    response early, a read error, or a body that doesn't match its declared ``Content-Length``
    discards the staged data, leaving any previously cached response intact.

    Recorded bytes are stored at the same layer as :py:class:`.CachedResponse` replay semantics:
    decoded content, matching ``response.content``. For compressed responses, undecoded reads
    (``decode_content=False``) therefore cannot be cached and invalidate the staged data.
    """

    def __init__(
        self,
        response: Response,
        cache: 'BaseCache',
        cache_key: str,
        expires=None,
        memory_threshold: int = DEFAULT_STREAM_CACHE_MEMORY_THRESHOLD,
    ):
        self._response = response
        self._raw = response.raw
        self._cache = cache
        self._cache_key = cache_key
        self._expires = expires
        # Buffer in memory up to the threshold, then spill over to a temp file
        self._spool = SpooledTemporaryFile(max_size=memory_threshold, mode='w+b')
        self._bytes_recorded = 0
        self._completed = False
        self._invalid = False
        self._encoded = 'content-encoding' in response.headers
        self._expected_length = self._get_expected_length(response)

    def _get_expected_length(self, response: Response) -> Optional[int]:
        # For encoded responses, Content-Length refers to compressed bytes, while decoded bytes
        # are recorded here; the length check only applies to unencoded responses.
        if self._encoded:
            return None
        try:
            return int(response.headers.get('Content-Length', ''))
        except ValueError:
            return None

    def __getattr__(self, name: str):
        """Delegate all other attributes to the wrapped urllib3 response"""
        return getattr(self._raw, name)

    @property
    def completed(self) -> bool:
        return self._completed

    def read(self, amt=None, decode_content=None, **kwargs) -> bytes:
        if self._completed or self._invalid:
            return self._raw.read(amt=amt, decode_content=decode_content, **kwargs)
        if self._encoded and not self._decode_requested(decode_content):
            self._invalidate('undecoded read of a compressed response cannot be cached')
            return self._raw.read(amt=amt, decode_content=decode_content, **kwargs)

        try:
            data = self._raw.read(amt=amt, decode_content=decode_content, **kwargs)
        except Exception:
            self._invalidate('error while reading streamed response')
            raise

        full_read = amt is None or amt < 0
        if data:
            self._record(data)
            # read() with no amount reads to end of stream
            if full_read:
                self._finalize()
        elif amt != 0:
            # Empty read: end of stream
            self._finalize()
        return data

    def stream(self, amt=2**16, decode_content=None, **kwargs):
        """Generator that emulates :py:meth:`urllib3.response.HTTPResponse.stream`, recording
        each chunk via :py:meth:`read`
        """
        while True:
            if self._completed:
                break
            data = self.read(amt=amt, decode_content=decode_content, **kwargs)
            if not data:
                break
            yield data

    def close(self):
        if not self._completed:
            self._invalidate('streamed response closed before being fully consumed')
        self._raw.close()

    def _decode_requested(self, decode_content) -> bool:
        if decode_content is not None:
            return decode_content
        return bool(getattr(self._raw, 'decode_content', False))

    def _record(self, data: bytes):
        self._spool.write(data)
        self._bytes_recorded += len(data)
        if self._expected_length is not None and self._bytes_recorded >= self._expected_length:
            self._finalize()

    def _invalidate(self, reason: str = ''):
        if not self._invalid and not self._completed:
            logger.debug(f'Not caching streamed response: {reason}')
        self._invalid = True
        self._spool.close()

    def _finalize(self):
        """Publish the staged response to the cache, if the body was consumed completely and
        matches its declared length
        """
        if self._completed or self._invalid:
            return
        if self._expected_length is not None and self._bytes_recorded != self._expected_length:
            self._invalidate(
                f'response body ({self._bytes_recorded} bytes) does not match declared '
                f'Content-Length ({self._expected_length} bytes)'
            )
            return

        self._completed = True
        self._spool.seek(0)
        body = self._spool.read()
        self._spool.close()

        # Restore the response to a fully-consumed state, with the recorded body available for
        # both response.content and further raw reads; no further network reads are needed
        response = self._response
        response._content = body
        response._content_consumed = True
        response.raw = self._raw
        _reset_fp(self._raw, body)

        self._cache.save_response(response, self._cache_key, self._expires)
        logger.debug(f'Streamed response fully consumed; wrote to cache: {response.url}')
