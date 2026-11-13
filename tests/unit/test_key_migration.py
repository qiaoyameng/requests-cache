from importlib.metadata import version

import sqlite3

import pytest

from requests_cache import CachedResponse, CachedSession, SQLiteCache
from requests_cache.cache_keys import create_key
from requests_cache.key_migration import (
    KeyMigrationError,
    apply_key_migration,
    plan_key_migration,
)
from requests_cache.policy import DEFAULT_IGNORED_PARAMS
from tests.conftest import mount_mock_adapter


BASE_URL = 'http+mock://requests-cache.test'


@pytest.fixture(params=['memory', 'sqlite'])
def session(request, tmp_path):
    if request.param == 'memory':
        session_kwargs = {
            'cache_name': 'migration-test',
            'backend': 'memory',
        }
    else:
        backend = SQLiteCache(str(tmp_path / 'cache.sqlite'))
        session_kwargs = {'backend': backend}

    cached_session = CachedSession(
        ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
        allowable_codes=(200, 302),
        **session_kwargs,
    )
    yield mount_mock_adapter(cached_session)
    cached_session.close()


def register_mock(session):
    adapter = session.mock_adapter
    adapter.register_uri('GET', f'{BASE_URL}/items', text='same response')
    adapter.register_uri(
        'GET',
        f'{BASE_URL}/items/conflict',
        [{'text': 'first response'}, {'text': 'second response'}],
    )
    adapter.register_uri(
        'GET',
        f'{BASE_URL}/vary',
        headers={'Content-Type': 'text/plain', 'Vary': 'Accept-Language'},
        text='language response',
    )
    adapter.register_uri(
        'GET',
        f'{BASE_URL}/redirect',
        headers={'Location': f'{BASE_URL}/target'},
        status_code=302,
    )
    adapter.register_uri('GET', f'{BASE_URL}/target', text='redirect target')


def test_plan_and_apply_key_migration(session):
    register_mock(session)

    session.get(f'{BASE_URL}/items?tracking_id=1&b=2')
    session.get(f'{BASE_URL}/items?b=2&tracking_id=2')
    session.get(f'{BASE_URL}/items?tracking_id=1&b=2&b=3')
    session.get(f'{BASE_URL}/items?other=1&tracking_id=9')
    session.get(f'{BASE_URL}/vary', headers={'Accept-Language': 'en'})
    session.get(f'{BASE_URL}/vary', headers={'Accept-Language': 'fr'})
    session.get(f'{BASE_URL}/redirect?tracking_id=1')

    old_response_keys = set(session.cache.responses)
    old_redirect_keys = set(session.cache.redirects)
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']

    plan = session.cache.plan_key_migration(
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
        old_key_fn=create_key,
    )

    assert plan.applicable is True
    assert plan.version == version('requests-cache')
    assert len(plan.response_entries) == len(old_response_keys)
    assert len(plan.redirect_entries) == len(old_redirect_keys)
    assert any(entry.old_key != entry.new_key for entry in plan.response_entries)
    assert all(entry.old_key != entry.new_key for entry in plan.redirect_entries)
    assert {entry.request_url for entry in plan.redirect_entries} == {
        f'{BASE_URL}/redirect?tracking_id=1'
    }

    vary_entries = [entry for entry in plan.response_entries if entry.old_vary_key]
    assert len(vary_entries) == 2
    assert {tuple(entry.vary_headers) for entry in vary_entries} == {('accept-language',)}
    assert {entry.new_key for entry in vary_entries} != {
        entry.new_base_key for entry in vary_entries
    }

    session.cache.apply_key_migration(plan)

    assert set(session.cache.responses) != old_response_keys
    assert set(session.cache.redirects) != old_redirect_keys
    assert len(session.cache.responses) == 7
    assert len(session.cache.redirects) == 1
    assert session.get(f'{BASE_URL}/items?tracking_id=99&b=2').from_cache is True
    assert session.get(f'{BASE_URL}/items?b=2&tracking_id=100').from_cache is True
    assert session.get(f'{BASE_URL}/items?b=2&b=3&tracking_id=100').from_cache is True
    assert session.get(f'{BASE_URL}/items?other=1&tracking_id=100').from_cache is True
    assert session.get(f'{BASE_URL}/vary', headers={'Accept-Language': 'en'}).from_cache
    assert session.get(f'{BASE_URL}/vary', headers={'Accept-Language': 'fr'}).from_cache
    assert session.get(f'{BASE_URL}/redirect?tracking_id=42').from_cache is True


def test_conflict_is_reported_and_migration_does_not_overwrite(session):
    register_mock(session)
    session.get(f'{BASE_URL}/items/conflict?tracking_id=1')
    session.get(f'{BASE_URL}/items/conflict?tracking_id=2')
    old_keys = set(session.cache.responses)
    old_responses = {key: value._content for key, value in session.cache.responses.items()}
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']

    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
        old_key_fn=create_key,
    )

    assert plan.applicable is False
    assert len(plan.conflicts) == 1
    assert plan.conflicts[0].old_keys == sorted(old_keys)
    assert plan.conflicts[0].reason == 'response representations differ'

    with pytest.raises(KeyMigrationError):
        apply_key_migration(session.cache, plan)

    assert set(session.cache.responses) == old_keys
    assert {key: value._content for key, value in session.cache.responses.items()} == old_responses
    session.settings.ignored_parameters = list(DEFAULT_IGNORED_PARAMS)
    assert session.get(f'{BASE_URL}/items/conflict?tracking_id=1').from_cache is True


def test_request_semantics_conflict_is_reported(session):
    register_mock(session)

    def coarse_key(**kwargs):
        request = kwargs['request']
        return f'coarse:{request.method}:{request.url.split("?")[0]}'

    session.get(f'{BASE_URL}/items?tracking_id=1&b=2')
    session.get(f'{BASE_URL}/items?tracking_id=2&b=2&other=1')
    old_keys = set(session.cache.responses)
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']
    session.settings.key_fn = coarse_key

    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
        old_key_fn=create_key,
    )

    assert plan.applicable is False
    assert plan.conflicts[0].old_keys == sorted(old_keys)
    assert plan.conflicts[0].reason == 'requests have different normalized semantics'
    with pytest.raises(KeyMigrationError):
        apply_key_migration(session.cache, plan)
    assert set(session.cache.responses) == old_keys


def test_unrecoverable_request_blocks_migration(session):
    register_mock(session)
    response = session.get(f'{BASE_URL}/items?tracking_id=1')
    old_keys = set(session.cache.responses)
    cached_response = CachedResponse.from_response(response)
    cached_response.request.method = None
    cached_response.request.url = None
    session.cache.responses[response.cache_key] = cached_response
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']

    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
    )

    assert plan.applicable is False
    assert plan.errors
    with pytest.raises(KeyMigrationError):
        apply_key_migration(session.cache, plan)
    assert set(session.cache.responses) == old_keys


def test_unknown_custom_key_fn_blocks_migration(session):
    def custom_key(*args, **kwargs):
        return 'custom-key'

    session.settings.key_fn = custom_key
    plan = plan_key_migration(session.cache)

    assert plan.applicable is False
    assert 'custom key_fn' in plan.errors[0]
    with pytest.raises(KeyMigrationError):
        apply_key_migration(session.cache, plan)


def test_known_custom_key_fn_can_migrate(session):
    def custom_key(**kwargs):
        return create_key(**kwargs) + '_suffix'

    register_mock(session)
    session.settings.key_fn = custom_key
    response = session.get(f'{BASE_URL}/items?tracking_id=1')
    old_key = response.cache_key
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']

    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
        old_key_fn=custom_key,
    )

    assert plan.applicable is True
    assert plan.response_entries[0].old_key == old_key
    apply_key_migration(session.cache, plan)
    assert session.get(f'{BASE_URL}/items?tracking_id=2').from_cache is True


def test_vary_cookie_derivation_is_listed(session):
    url = f'{BASE_URL}/cookie-vary'
    session.mock_adapter.register_uri(
        'GET',
        url,
        headers={'Vary': 'Cookie'},
        text='cookie response',
    )
    session.get(url, cookies={'locale': 'en'})
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']

    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
        old_key_fn=create_key,
    )

    entry = plan.response_entries[0]
    assert entry.vary_headers == ['cookie']
    assert entry.old_vary_key is None
    assert entry.new_vary_key is None


def test_stale_plan_version_blocks_migration(session):
    register_mock(session)
    session.get(f'{BASE_URL}/items?tracking_id=1')
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']
    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
    )
    plan.version = '0.0.0'

    with pytest.raises(KeyMigrationError, match='version changed'):
        apply_key_migration(session.cache, plan)


def test_sqlite_write_failure_rolls_back(tmp_path, monkeypatch):
    session = CachedSession(
        backend=SQLiteCache(str(tmp_path / 'cache.sqlite')),
        ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
    )
    session = mount_mock_adapter(session)
    register_mock(session)
    session.get(f'{BASE_URL}/items?tracking_id=1')
    session.get(f'{BASE_URL}/redirect?tracking_id=1')
    old_response_keys = set(session.cache.responses)
    old_redirect_keys = set(session.cache.redirects)
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']
    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
    )

    def failing_dumps(value):
        raise sqlite3.OperationalError('simulated write failure')

    monkeypatch.setattr(session.cache.responses.serializer, 'dumps', failing_dumps)

    with pytest.raises(sqlite3.OperationalError, match='simulated write failure'):
        apply_key_migration(session.cache, plan)

    assert set(session.cache.responses) == old_response_keys
    assert set(session.cache.redirects) == old_redirect_keys
    session.settings.ignored_parameters = list(DEFAULT_IGNORED_PARAMS)
    assert session.get(f'{BASE_URL}/items?tracking_id=1').from_cache is True
    assert session.get(f'{BASE_URL}/redirect?tracking_id=1').from_cache is True
    session.close()


@pytest.mark.parametrize('table_name', ['responses', 'redirects'])
def test_missing_sqlite_index_blocks_migration(tmp_path, table_name):
    db_path = tmp_path / 'cache.sqlite'
    session = CachedSession(
        backend=SQLiteCache(str(db_path)),
        ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
    )
    session = mount_mock_adapter(session)
    register_mock(session)
    session.get(f'{BASE_URL}/items?tracking_id=1')
    session.get(f'{BASE_URL}/redirect?tracking_id=1')
    session.settings.ignored_parameters = [*DEFAULT_IGNORED_PARAMS, 'tracking_id']

    connection = session.cache.responses._connection
    index_name = (
        'expires_idx'
        if table_name == 'responses'
        else f'sqlite_autoindex_{table_name}_1'
    )
    if table_name == 'responses':
        connection.execute('DROP INDEX expires_idx')
    else:
        connection.execute('DROP TABLE redirects')
        connection.execute('CREATE TABLE redirects (key TEXT, value TEXT, expires INTEGER)')

    plan = plan_key_migration(
        session.cache,
        old_ignored_parameters=list(DEFAULT_IGNORED_PARAMS),
    )

    assert plan.applicable is False
    assert any(table_name in error and index_name in error for error in plan.errors)
    session.close()
