"""Tests for RFC 9111 section 4.4 unsafe-method cache invalidation"""

from unittest.mock import patch

import pytest
import requests
from requests_mock import Adapter

from requests_cache import CachedSession
from requests_cache.policy import (
    get_invalidation_urls,
    is_successful_invalidation_status,
    is_unsafe_method,
    same_origin,
)
from requests_cache.policy.invalidation import UNSAFE_METHODS

ORIGIN = 'http+mock://example.com'
OTHER_ORIGIN = 'https+mock://other.example'
RESOURCE_URL = f'{ORIGIN}/resource'
REDIRECT_URL = f'{ORIGIN}/redirect'
TARGET_URL = f'{ORIGIN}/target'
OTHER_URL = f'{ORIGIN}/unrelated'
CROSS_ORIGIN_URL = f'{OTHER_ORIGIN}/resource'


def mount(session) -> CachedSession:
    adapter = Adapter()
    for protocol in ('mock://', 'http+mock://', 'https+mock://'):
        session.mount(protocol, adapter)
    session.mock_adapter = adapter
    return session


@pytest.fixture
def session(request) -> CachedSession:
    backend = getattr(request, 'param', 'memory')
    session = CachedSession(
        backend=backend,
        cache_name=':memory:' if backend == 'sqlite' else 'test_cache',
        use_memory=True,
        unsafe_method_invalidation=True,
    )
    yield mount(session)
    session.close()


def _register_redirect(adapter, redirect_url, target_url):
    adapter.register_uri(
        'GET',
        redirect_url,
        status_code=302,
        headers={'Location': target_url},
        text='redirect',
    )
    adapter.register_uri('GET', target_url, status_code=200, text='target body')


# RFC 9111 section 4.4 policy helpers
# -----------------------------------------------------


@pytest.mark.parametrize('method', sorted(UNSAFE_METHODS))
def test_is_unsafe_method(method):
    prepared = requests.Request(method, RESOURCE_URL).prepare()
    assert is_unsafe_method(prepared) is True


@pytest.mark.parametrize('method', ['GET', 'HEAD', 'OPTIONS', 'TRACE', 'CONNECT'])
def test_is_unsafe_method__safe_methods(method):
    prepared = requests.Request(method, RESOURCE_URL).prepare()
    assert is_unsafe_method(prepared) is False


@pytest.mark.parametrize(
    ('status_code', 'expected'),
    [(199, False), (200, True), (201, True), (204, True), (206, True), (299, True), (300, False),
     (302, False), (400, False), (404, False), (500, False)],
)
def test_successful_status_range(status_code, expected):
    assert is_successful_invalidation_status(status_code) is expected


@pytest.mark.parametrize(
    ('url_1', 'url_2', 'expected'),
    [
        ('http://example.com/a', 'http://example.com/b', True),
        ('http://example.com', 'http://example.com:80/x', True),
        ('https://example.com', 'https://example.com:443/x', True),
        ('http://example.com', 'https://example.com/x', False),
        ('http://example.com', 'http://other.com/x', False),
        ('http://example.com:8080', 'http://example.com/x', False),
        ('http://example.com', 'http://user@example.com/x', True),
    ],
)
def test_same_origin(url_1, url_2, expected):
    assert same_origin(url_1, url_2) is expected


def test_invalidation_urls__relative_and_content_location():
    origin = 'http://example.com'
    request = requests.Request('POST', f'{origin}/api/items').prepare()
    response = requests.Response()
    response.url = f'{origin}/api/items'
    response.headers['Location'] = '../items/42'
    response.headers['Content-Location'] = f'{origin}/api/items/42'

    urls = get_invalidation_urls(request, response)
    assert urls == [
        f'{origin}/api/items',
        f'{origin}/items/42',
        f'{origin}/api/items/42',
    ]


def test_invalidation_urls__cross_origin_excluded():
    origin = 'http://example.com'
    other_origin = 'https://other.example'
    request = requests.Request('POST', f'{origin}/items').prepare()
    response = requests.Response()
    response.url = f'{origin}/items'
    response.headers['Location'] = f'{other_origin}/items/1'
    response.headers['Content-Location'] = '/items/1'

    assert get_invalidation_urls(request, response) == [
        f'{origin}/items',
        f'{origin}/items/1',
    ]


def test_invalidation_urls__redirect_base():
    # Relative Location is resolved against the response's effective URI, which follows redirects
    origin = 'http://example.com'
    request = requests.Request('POST', f'{origin}/start').prepare()
    response = requests.Response()
    response.url = f'{origin}/nested/final'
    response.headers['Location'] = 'created'

    assert get_invalidation_urls(request, response) == [
        f'{origin}/start',
        f'{origin}/nested/created',
    ]


def test_invalidation_urls__network_failure_only_request_uri():
    origin = 'http://example.com'
    resource_url = f'{origin}/resource'
    request = requests.Request('POST', resource_url).prepare()
    assert get_invalidation_urls(request, None) == [resource_url]


# End-to-end session behavior
# -----------------------------------------------------


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_post_invalidates_request_uri_and_redirect_alias(session):
    adapter = session.mock_adapter
    _register_redirect(adapter, REDIRECT_URL, TARGET_URL)
    adapter.register_uri(
        'POST',
        RESOURCE_URL,
        status_code=201,
        json={'id': 1},
        headers={'Location': TARGET_URL},
    )
    adapter.register_uri('GET', RESOURCE_URL, text='stale body')

    # Populate the cache, including a redirect alias pointing at the target
    assert session.get(REDIRECT_URL).from_cache is False
    assert session.get(REDIRECT_URL).from_cache is True
    assert session.get(RESOURCE_URL).from_cache is False
    assert session.get(RESOURCE_URL).from_cache is True

    # Successful POST with a Location pointing at the cached redirect target
    post_response = session.post(RESOURCE_URL)
    assert post_response.status_code == 201
    assert not post_response.from_cache

    # The invalidated representations must be fetched from the origin again
    assert session.get(RESOURCE_URL).from_cache is False
    # Both the target representation and the redirect alias pointing at it were removed
    assert not session.cache.contains(url=TARGET_URL)
    assert not session.cache.contains(url=REDIRECT_URL)
    refreshed = session.get(REDIRECT_URL)
    # The target representation was invalidated, so the whole redirect chain is refetched
    assert refreshed.from_cache is False
    assert refreshed.url == TARGET_URL


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_post_invalidates_vary_variants(session):
    adapter = session.mock_adapter
    adapter.register_uri(
        'GET',
        RESOURCE_URL,
        status_code=200,
        headers={'Vary': 'Accept'},
        text='variant',
    )
    adapter.register_uri('POST', RESOURCE_URL, status_code=200)

    session.get(RESOURCE_URL, headers={'Accept': 'text/plain'})
    session.get(RESOURCE_URL, headers={'Accept': 'application/json'})
    assert session.get(RESOURCE_URL, headers={'Accept': 'text/plain'}).from_cache is True
    assert session.get(RESOURCE_URL, headers={'Accept': 'application/json'}).from_cache is True

    session.post(RESOURCE_URL)

    # Every Vary variant for the URI must have been removed
    assert session.get(RESOURCE_URL, headers={'Accept': 'text/plain'}).from_cache is False
    assert session.get(RESOURCE_URL, headers={'Accept': 'application/json'}).from_cache is False


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_post_invalidates_location_and_content_location(session):
    adapter = session.mock_adapter
    location_url = f'{ORIGIN}/created'
    adapter.register_uri('GET', location_url, text='old location resource')
    adapter.register_uri('GET', TARGET_URL, text='old content-location resource')
    adapter.register_uri(
        'POST',
        RESOURCE_URL,
        status_code=201,
        headers={'Location': location_url, 'Content-Location': TARGET_URL},
    )

    session.get(location_url)
    session.get(TARGET_URL)
    assert session.get(location_url).from_cache is True
    assert session.get(TARGET_URL).from_cache is True

    session.post(RESOURCE_URL)

    assert session.get(location_url).from_cache is False
    assert session.get(TARGET_URL).from_cache is False


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_cross_origin_and_unrelated_urls_remain_cached(session):
    adapter = session.mock_adapter
    adapter.register_uri('GET', CROSS_ORIGIN_URL, text='other origin')
    adapter.register_uri('GET', OTHER_URL, text='unrelated')
    adapter.register_uri(
        'POST',
        RESOURCE_URL,
        status_code=201,
        headers={'Location': CROSS_ORIGIN_URL},
    )

    session.get(CROSS_ORIGIN_URL)
    session.get(OTHER_URL)
    assert session.get(CROSS_ORIGIN_URL).from_cache is True
    assert session.get(OTHER_URL).from_cache is True

    session.post(RESOURCE_URL)

    assert session.get(CROSS_ORIGIN_URL).from_cache is True
    assert session.get(OTHER_URL).from_cache is True


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_unsuccessful_status_does_not_invalidate(session):
    adapter = session.mock_adapter
    for status_code in (302, 400, 404, 500):
        url = f'{ORIGIN}/resource-{status_code}'
        adapter.register_uri('GET', url, text='still cached')
        adapter.register_uri('POST', url, status_code=status_code)
        session.get(url)
        assert session.get(url).from_cache is True

        response = session.post(url)
        assert response.status_code == status_code
        assert session.get(url).from_cache is True


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_network_failure_invalidates_only_request_uri(session):
    adapter = session.mock_adapter
    adapter.register_uri('GET', RESOURCE_URL, text='stale')
    adapter.register_uri('GET', OTHER_URL, text='unrelated')
    adapter.register_uri('POST', RESOURCE_URL, exc=requests.ConnectionError('boom'))

    session.get(RESOURCE_URL)
    session.get(OTHER_URL)

    with pytest.raises(requests.ConnectionError):
        session.post(RESOURCE_URL)

    assert session.get(RESOURCE_URL).from_cache is False
    # The rest of the cache is preserved; no unconditional clear() happens
    assert session.get(OTHER_URL).from_cache is True


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_put_no_store_still_invalidates(session):
    """A successful PUT whose response is not cached must still invalidate old representations"""
    adapter = session.mock_adapter
    adapter.register_uri('GET', RESOURCE_URL, text='old version')
    adapter.register_uri(
        'PUT',
        RESOURCE_URL,
        status_code=200,
        headers={'Cache-Control': 'no-store'},
        text='updated',
    )

    assert session.get(RESOURCE_URL).from_cache is False
    assert session.get(RESOURCE_URL).from_cache is True

    put_response = session.put(RESOURCE_URL)
    assert put_response.status_code == 200

    # The PUT response itself was not stored, but the old GET representation was invalidated
    assert not session.cache.contains(url=RESOURCE_URL)
    assert session.get(RESOURCE_URL).from_cache is False


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_disabled_setting_preserves_legacy_behavior(session):
    session.settings.unsafe_method_invalidation = False
    adapter = session.mock_adapter
    adapter.register_uri('GET', RESOURCE_URL, text='stale')
    adapter.register_uri('POST', RESOURCE_URL, status_code=201)

    session.get(RESOURCE_URL)
    assert session.get(RESOURCE_URL).from_cache is True
    session.post(RESOURCE_URL)
    assert session.get(RESOURCE_URL).from_cache is True


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_get_request_never_invalidates(session):
    adapter = session.mock_adapter
    adapter.register_uri('GET', RESOURCE_URL, text='body')

    session.get(RESOURCE_URL)
    session.get(RESOURCE_URL)
    assert session.get(RESOURCE_URL).from_cache is True


@pytest.mark.parametrize('session', ['memory', 'sqlite'], indirect=True)
def test_invalidation_failure_is_propagated(session):
    """A failed invalidation must not be silently reported as a completed update"""
    adapter = session.mock_adapter
    adapter.register_uri('GET', RESOURCE_URL, text='body')
    adapter.register_uri('POST', RESOURCE_URL, status_code=201)
    session.get(RESOURCE_URL)

    with patch.object(session.cache, 'invalidate_urls', side_effect=RuntimeError('backend down')):
        with pytest.raises(RuntimeError, match='backend down'):
            session.post(RESOURCE_URL)
