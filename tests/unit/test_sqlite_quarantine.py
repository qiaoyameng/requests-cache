"""Tests for SQLite backend quarantine mode for corrupt cache items"""

import pickle
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest
import requests
from cattrs.errors import ClassValidationError
from requests_mock import ANY as ANY_METHOD

from requests_cache import CachedSession
from requests_cache.backends import SQLiteCache
from tests.conftest import MOCKED_URL, MOCKED_URL_JSON, MOCKED_URL_REDIRECT, mount_mock_adapter


@pytest.fixture(scope='function')
def db_path(tmpdir) -> str:
    """Get a unique tempfile path for a SQLite db"""
    return str(tmpdir / f'{uuid4()}.sqlite')


@pytest.fixture(scope='function')
def quarantine_session(db_path) -> CachedSession:
    """A mock session backed by a SQLite cache with quarantine mode enabled"""
    session = mount_mock_adapter(CachedSession(backend=SQLiteCache(db_path, quarantine=True)))
    yield session
    session.close()


def _corrupt_value(db_path: str, key: str, new_value: bytes):
    """Directly overwrite a serialized response value in the db, bypassing the cache"""
    with sqlite3.connect(db_path) as con:
        con.execute('UPDATE responses SET value=? WHERE key=?', (new_value, key))


def _response_key_for_url(session: CachedSession, url: str) -> str:
    for key in session.cache.responses.keys():
        if session.cache.responses[key].url == url:
            return key
    raise AssertionError(f'No cached response for {url}')


def test_quarantine_corrupt_serialized_value(quarantine_session, db_path):
    """A response with a corrupted serialized value should be moved to the quarantine area and
    treated as a cache miss, without affecting healthy items
    """
    session = quarantine_session
    assert session.get(MOCKED_URL).from_cache is False
    assert session.get(MOCKED_URL_JSON).from_cache is False
    assert session.get(MOCKED_URL).from_cache is True
    assert session.get(MOCKED_URL_JSON).from_cache is True

    key = _response_key_for_url(session, MOCKED_URL)
    _corrupt_value(db_path, key, b'\x00not a pickle')

    # The corrupt item is quarantined and treated as a miss; the refetch succeeds
    assert session.get(MOCKED_URL).from_cache is False

    quarantined = session.cache.list_quarantined()
    assert len(quarantined) == 1
    item = quarantined[0]
    assert item.key == key
    assert item.table_name == 'responses'
    # The raw value is preserved as-is, and never deserialized
    assert item.value == b'\x00not a pickle'
    assert item.format == 'pickle'
    assert 'UnpicklingError' in item.error
    assert item.quarantined_at > 0

    # The fresh response is cached again, and healthy items are unaffected
    assert session.get(MOCKED_URL).from_cache is True
    assert session.get(MOCKED_URL_JSON).from_cache is True
    assert len(session.cache.list_quarantined()) == 1


def test_quarantine_invalid_response_structure(quarantine_session, db_path):
    """A value that deserializes but fails response structure validation should also be
    quarantined instead of failing the whole request
    """
    session = quarantine_session
    session.get(MOCKED_URL)
    key = _response_key_for_url(session, MOCKED_URL)

    # Valid pickle, but not a valid CachedResponse structure
    _corrupt_value(db_path, key, pickle.dumps({'request': None}))

    assert session.get(MOCKED_URL).from_cache is False
    quarantined = session.cache.list_quarantined()
    assert len(quarantined) == 1
    assert 'ClassValidationError' in quarantined[0].error
    assert session.get(MOCKED_URL).from_cache is True


def test_quarantine_cleans_up_redirect_alias(quarantine_session, db_path):
    """Quarantining a corrupt response should atomically remove redirect aliases pointing to it"""
    session = quarantine_session
    session.get(MOCKED_URL_REDIRECT)
    redirect_key = list(session.cache.redirects.keys())[0]
    target_key = session.cache.redirects[redirect_key]

    _corrupt_value(db_path, target_key, b'\x00corrupt')

    # Access via the alias: corrupt target is quarantined and the alias is removed
    assert session.cache.get_response(redirect_key) is None
    assert redirect_key not in session.cache.redirects
    assert target_key not in session.cache.responses
    quarantined = session.cache.list_quarantined()
    assert len(quarantined) == 1
    assert quarantined[0].key == target_key

    # A fresh request re-caches both the response row and the alias. The (still healthy)
    # redirect target response is served from the cache.
    assert session.get(MOCKED_URL_REDIRECT).from_cache is True
    assert session.cache.redirects[redirect_key] == target_key
    assert session.get(MOCKED_URL_REDIRECT).from_cache is True
    assert len(session.cache.list_quarantined()) == 1


def test_network_error_not_recorded_as_corruption(quarantine_session, db_path):
    """If the network fallback after quarantining fails, the network error should be raised,
    and not recorded as a second cache corruption
    """
    session = quarantine_session
    session.get(MOCKED_URL)
    key = _response_key_for_url(session, MOCKED_URL)
    _corrupt_value(db_path, key, b'\x00corrupt')

    session.mock_adapter.register_uri(ANY_METHOD, MOCKED_URL, exc=requests.ConnectTimeout)
    with pytest.raises(requests.ConnectTimeout):
        session.get(MOCKED_URL)
    quarantined = session.cache.list_quarantined()
    assert len(quarantined) == 1
    assert quarantined[0].key == key

    # The corrupt row was already moved, so repeated failed accesses don't add records
    with pytest.raises(requests.ConnectTimeout):
        session.get(MOCKED_URL)
    assert len(session.cache.list_quarantined()) == 1


def test_expiration_cleanup_preserves_quarantine(quarantine_session, db_path):
    """Normal expiration cleanup must not remove quarantine records; they are only removed via
    the explicit deletion interfaces
    """
    session = quarantine_session
    session.get(MOCKED_URL)
    key = _response_key_for_url(session, MOCKED_URL)
    _corrupt_value(db_path, key, b'\x00corrupt')
    session.get(MOCKED_URL)
    assert len(session.cache.list_quarantined()) == 1

    session.cache.delete(expired=True)
    assert len(session.cache.list_quarantined()) == 1

    # Explicit deletion by key
    session.cache.delete_quarantined(key)
    assert session.cache.list_quarantined() == []

    # Explicitly clearing all quarantine records
    _corrupt_value(db_path, key, b'\x00corrupt again')
    session.get(MOCKED_URL)
    assert len(session.cache.list_quarantined()) == 1
    session.cache.clear_quarantined()
    assert session.cache.list_quarantined() == []


def test_quarantine_records_persist_after_reopen(quarantine_session, db_path):
    """Quarantine records are stored in the cache file, so evidence is still available after
    reopening it in a new process
    """
    session = quarantine_session
    session.get(MOCKED_URL)
    key = _response_key_for_url(session, MOCKED_URL)
    _corrupt_value(db_path, key, b'\x00corrupt')
    session.get(MOCKED_URL)
    assert len(session.cache.list_quarantined()) == 1
    session.close()

    code = (
        'from requests_cache.backends import SQLiteCache;'
        f'cache = SQLiteCache({db_path!r}, quarantine=True);'
        'items = cache.list_quarantined();'
        'print(len(items));'
        'print(items[0].key);'
        'print(items[0].value);'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True
    )
    lines = result.stdout.splitlines()
    assert lines[0] == '1'
    assert lines[1] == key
    assert lines[2] == "b'\\x00corrupt'"


def test_quarantine_disabled_by_default(db_path):
    """Default behavior is unchanged: no quarantine table is created, and deserialization
    errors are treated as a cache miss without quarantining
    """
    session = mount_mock_adapter(CachedSession(backend=SQLiteCache(db_path)))
    session.get(MOCKED_URL)
    key = _response_key_for_url(session, MOCKED_URL)

    assert session.cache.list_quarantined() == []
    with sqlite3.connect(db_path) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master")}
    assert 'responses_quarantine' not in tables

    _corrupt_value(db_path, key, b'\x00not a pickle')
    assert session.get(MOCKED_URL).from_cache is False
    assert session.cache.list_quarantined() == []
    session.close()


def test_structure_error_raises_by_default(db_path):
    """Without quarantine mode, response structure validation errors still propagate"""
    session = mount_mock_adapter(CachedSession(backend=SQLiteCache(db_path)))
    session.get(MOCKED_URL)
    key = _response_key_for_url(session, MOCKED_URL)

    _corrupt_value(db_path, key, pickle.dumps({'request': None}))
    with pytest.raises(ClassValidationError):
        session.get(MOCKED_URL)
    session.close()
