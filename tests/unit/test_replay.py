"""Acceptance tests for portable replay packages (``requests_cache.replay``)

Covers the full workflow: build a package from a source cache (small paginated JSON, a
redirect, two Accept-Language Vary variants, and a binary body), import it into an empty
directory, replay it with the network send entry points rigged to fail, and verify that a
corrupted body checksum makes the whole import fail without leaving any records behind.
"""

import base64
import json
import pickle
from datetime import timedelta
from unittest.mock import patch

import pytest
import requests
from requests import Request
from requests_mock import ANY as ANY_METHOD
from requests_mock import Adapter

from requests_cache import CachedSession, utcnow
from requests_cache.replay import (
    FORMAT_VERSION,
    ReplayConfigError,
    ReplayError,
    ReplayExportError,
    ReplayMiss,
    ReplayPackageError,
    ReplaySession,
    export_replay_package,
    import_replay_package,
)
from requests_cache.serializers import CattrStage, SerializerPipeline, Stage

BASE = 'http+mock://replay.example.com'
ITEMS_URL = f'{BASE}/items'
SEARCH_URL = f'{BASE}/search?tag=a&tag=b'
REDIRECT_URL = f'{BASE}/redirect'
REDIRECT_TARGET_URL = f'{BASE}/redirect-target'
VARY_URL = f'{BASE}/vary'
IMAGE_URL = f'{BASE}/image'
OTHER_URL = f'{BASE}/other-task'
MISSING_URL = f'{BASE}/missing'

PAGE_1 = {'page': 1, 'items': ['a', 'b'], 'next': f'{ITEMS_URL}?page=2'}
PAGE_2 = {'page': 2, 'items': ['c'], 'next': None}
BINARY_BODY = bytes(range(256)) + b'\x00\xffbinary-body'


def fail_on_network(*args, **kwargs):
    raise AssertionError('Network access attempted during replay')


@pytest.fixture
def no_network():
    """Replace the network send entry point with a function that fails on any call.
    The requests-mock adapter (a BaseAdapter, not an HTTPAdapter) is unaffected, so the
    source cache can still be populated with mock responses."""
    with patch.object(requests.adapters.HTTPAdapter, 'send', fail_on_network):
        yield


def _vary_text(request, context):
    return f"lang={request.headers.get('Accept-Language', 'unknown')}"


def build_source_session(cache_path) -> CachedSession:
    """A CachedSession backed by a mock adapter, populated with the test collection"""
    session = CachedSession(str(cache_path), backend='sqlite')
    adapter = Adapter()
    adapter.register_uri(
        ANY_METHOD, f'{ITEMS_URL}?page=1', json=PAGE_1, headers={'Content-Type': 'application/json'}
    )
    adapter.register_uri(
        ANY_METHOD, f'{ITEMS_URL}?page=2', json=PAGE_2, headers={'Content-Type': 'application/json'}
    )
    adapter.register_uri(ANY_METHOD, f'{BASE}/search', json={'results': ['x']})
    adapter.register_uri(
        ANY_METHOD, REDIRECT_URL, status_code=302, headers={'Location': REDIRECT_TARGET_URL}
    )
    adapter.register_uri(ANY_METHOD, REDIRECT_TARGET_URL, text='redirect target body')
    adapter.register_uri(
        ANY_METHOD,
        VARY_URL,
        text=_vary_text,
        headers={'Vary': 'Accept-Language', 'Content-Type': 'text/plain'},
    )
    adapter.register_uri(
        ANY_METHOD,
        IMAGE_URL,
        content=BINARY_BODY,
        headers={'Content-Type': 'application/octet-stream'},
    )
    adapter.register_uri(ANY_METHOD, OTHER_URL, text='unrelated task response')
    session.mount('http+mock://', adapter)

    # Populate the cache
    session.get(f'{ITEMS_URL}?page=1')
    session.get(f'{ITEMS_URL}?page=2')
    session.get(SEARCH_URL)
    session.get(REDIRECT_URL)
    session.get(VARY_URL, headers={'Accept-Language': 'en'})
    session.get(VARY_URL, headers={'Accept-Language': 'zh'})
    session.get(IMAGE_URL)
    session.get(OTHER_URL)

    # Make the page-2 response expired; replay sessions must still serve it
    page_2_key = session.cache.create_key(Request('GET', f'{ITEMS_URL}?page=2').prepare())
    page_2_response = session.cache.responses[page_2_key]
    page_2_response.expires = utcnow() - timedelta(days=1)
    session.cache.responses[page_2_key] = page_2_response
    return session


def export_selection(session, package_path):
    """Export everything except the unrelated 'other task' response"""
    other_key = session.cache.create_key(Request('GET', OTHER_URL).prepare())
    keys = [k for k in session.cache.responses.keys() if k != other_key]
    export_replay_package(session, keys, package_path)
    return keys


def test_replay_roundtrip(tmp_path, no_network):
    package_path = tmp_path / 'collection.replay.json'
    source = build_source_session(tmp_path / 'source.db')
    export_selection(source, package_path)
    source.close()

    # The package is versioned, non-pickle JSON that preserves the required request data
    package_text = package_path.read_text()
    package = json.loads(package_text)
    assert package['format_version'] == FORMAT_VERSION
    assert 'tag=a&tag=b' in package_text  # duplicate request params preserved
    assert base64.b64encode(BINARY_BODY).decode() in package_text  # binary body preserved
    assert len(package['redirects']) == 1  # redirect index included
    assert OTHER_URL not in package_text  # caller-selected subset only

    # Import into a brand-new cache in an empty directory
    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    target_path = str(empty_dir / 'replay.db')
    import_session = CachedSession(target_path, backend='sqlite')
    n_imported = import_replay_package(package_path, import_session)
    assert n_imported == len(package['responses'])
    import_session.close()

    # Replay read-only from the imported cache
    replay = ReplaySession(target_path, backend='sqlite', package=package_path)
    n_records = len(replay.cache.responses)

    assert replay.get(f'{ITEMS_URL}?page=1').json() == PAGE_1

    page_2 = replay.get(f'{ITEMS_URL}?page=2')
    assert page_2.json() == PAGE_2
    assert page_2.is_expired  # expired responses are still served

    assert replay.get(SEARCH_URL).json() == {'results': ['x']}

    redirect = replay.get(REDIRECT_URL)
    assert redirect.url == REDIRECT_TARGET_URL  # final URL preserved
    assert redirect.text == 'redirect target body'
    assert len(redirect.history) == 1  # response history preserved
    assert redirect.history[0].status_code == 302

    # Both Vary: Accept-Language variants are served distinctly
    assert replay.get(VARY_URL, headers={'Accept-Language': 'en'}).text == 'lang=en'
    assert replay.get(VARY_URL, headers={'Accept-Language': 'zh'}).text == 'lang=zh'

    assert replay.get(IMAGE_URL).content == BINARY_BODY

    # Misses are identifiable, including unexported and unknown Vary variants
    with pytest.raises(ReplayMiss):
        replay.get(OTHER_URL)
    with pytest.raises(ReplayMiss):
        replay.get(MISSING_URL)
    with pytest.raises(ReplayMiss):
        replay.get(VARY_URL, headers={'Accept-Language': 'fr'})

    # Replay never writes to the cache
    assert len(replay.cache.responses) == n_records
    replay.close()


def test_corrupt_checksum_leaves_nothing(tmp_path, no_network):
    package_path = tmp_path / 'collection.replay.json'
    source = build_source_session(tmp_path / 'source.db')
    export_selection(source, package_path)
    source.close()

    package = json.loads(package_path.read_text())
    some_key = sorted(package['responses'])[0]
    package['responses'][some_key]['sha256'] = '0' * 64
    package_path.write_text(json.dumps(package))

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    target = CachedSession(str(empty_dir / 'replay.db'), backend='sqlite')
    with pytest.raises(ReplayPackageError, match='checksum'):
        import_replay_package(package_path, target)
    assert len(target.cache.responses) == 0
    assert len(target.cache.redirects) == 0
    target.close()


def test_import_rejects_bad_version(tmp_path):
    package_path = tmp_path / 'collection.replay.json'
    source = build_source_session(tmp_path / 'source.db')
    export_selection(source, package_path)
    source.close()

    package = json.loads(package_path.read_text())
    package['format_version'] = FORMAT_VERSION + 1
    package_path.write_text(json.dumps(package))

    target = CachedSession(str(tmp_path / 'replay.db'), backend='sqlite')
    with pytest.raises(ReplayPackageError, match='version'):
        import_replay_package(package_path, target)
    assert len(target.cache.responses) == 0
    target.close()


def test_import_rejects_dangling_redirect(tmp_path):
    package_path = tmp_path / 'collection.replay.json'
    source = build_source_session(tmp_path / 'source.db')
    export_selection(source, package_path)
    source.close()

    package = json.loads(package_path.read_text())
    package['redirects']['deadbeef'] = 'not-in-the-package'
    package_path.write_text(json.dumps(package))

    target = CachedSession(str(tmp_path / 'replay.db'), backend='sqlite')
    with pytest.raises(ReplayPackageError, match='[Rr]edirect'):
        import_replay_package(package_path, target)
    assert len(target.cache.responses) == 0
    assert len(target.cache.redirects) == 0
    target.close()


def test_import_refuses_to_overwrite_by_default(tmp_path, no_network):
    package_path = tmp_path / 'collection.replay.json'
    source = build_source_session(tmp_path / 'source.db')
    export_selection(source, package_path)
    source.close()

    target = CachedSession(str(tmp_path / 'replay.db'), backend='sqlite')
    import_replay_package(package_path, target)
    with pytest.raises(ReplayPackageError, match='[Oo]verwrite'):
        import_replay_package(package_path, target)
    # Explicit opt-in allows overwriting
    import_replay_package(package_path, target, overwrite=True)
    target.close()


def test_config_mismatch_reported_when_opening_package(tmp_path):
    package_path = tmp_path / 'collection.replay.json'
    source = build_source_session(tmp_path / 'source.db')
    export_selection(source, package_path)
    source.close()

    target_path = str(tmp_path / 'replay.db')
    import_session = CachedSession(target_path, backend='sqlite')
    import_replay_package(package_path, import_session)
    import_session.close()

    # A mismatched session fails at import time...
    mismatched = CachedSession(
        str(tmp_path / 'other.db'), backend='sqlite', match_headers=['Accept']
    )
    with pytest.raises(ReplayConfigError, match='match_headers'):
        import_replay_package(package_path, mismatched)
    assert len(mismatched.cache.responses) == 0
    mismatched.close()

    # ...and when opening a replay session, not on the first missed request
    with pytest.raises(ReplayConfigError, match='match_headers'):
        ReplaySession(target_path, backend='sqlite', match_headers=['Accept'], package=package_path)


def test_export_refuses_custom_key_fn(tmp_path):
    session = CachedSession(backend='memory', key_fn=lambda *args, **kwargs: 'key')
    with pytest.raises(ReplayExportError, match='key_fn'):
        export_replay_package(session, [], tmp_path / 'pkg.json')


def test_export_refuses_custom_serializer(tmp_path):
    serializer = SerializerPipeline([CattrStage(), Stage(pickle)], name='custom-pipeline')
    session = CachedSession(str(tmp_path / 'source.db'), backend='sqlite', serializer=serializer)
    with pytest.raises(ReplayExportError, match='serializer'):
        export_replay_package(session, [], tmp_path / 'pkg.json')
    session.close()


def test_export_refuses_unsupported_backend(tmp_path):
    session = CachedSession(str(tmp_path / 'files'), backend='filesystem')
    with pytest.raises(ReplayError, match='memory and SQLite'):
        export_replay_package(session, [], tmp_path / 'pkg.json')
    session.close()


def test_export_missing_key(tmp_path):
    session = CachedSession(backend='memory')
    with pytest.raises(ReplayExportError, match='No cached response'):
        export_replay_package(session, ['nonexistent'], tmp_path / 'pkg.json')


def test_memory_backend_roundtrip(tmp_path, no_network):
    source = CachedSession(backend='memory')
    adapter = Adapter()
    adapter.register_uri(ANY_METHOD, f'{ITEMS_URL}?page=1', json=PAGE_1)
    source.mount('http+mock://', adapter)
    source.get(f'{ITEMS_URL}?page=1')

    package_path = tmp_path / 'pkg.json'
    export_replay_package(source, list(source.cache.responses.keys()), package_path)

    target = CachedSession(backend='memory')
    import_replay_package(package_path, target)
    replay = ReplaySession(backend=target.cache, package=package_path)
    assert replay.get(f'{ITEMS_URL}?page=1').json() == PAGE_1
    with pytest.raises(ReplayMiss):
        replay.get(MISSING_URL)
