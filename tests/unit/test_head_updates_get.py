"""Tests for opt-in HEAD-driven updates of cached complete GET representations.

Covers RFC 9111, sections 3.4 and 4.3.5:
* A 200 HEAD response may update header fields and freshness of a cached GET representation for
  the same target URI and matching Vary request headers, but never its payload.
* Conflicting evidence (validators or Content-Length) invalidates the cached GET response.
* Insufficient evidence and non-200 statuses are handled conservatively.
* Behavior is consistent across in-memory and SQLite backends, including response bodies,
  redirect aliases, and expiration metadata.
"""

from datetime import timedelta
from unittest.mock import patch

import pytest
from requests_mock import Adapter

from requests_cache import CachedSession
from requests_cache.backends import SQLiteCache
from tests.conftest import START_DT, time_travel

URL = 'http+mock://head-updates.example.com/resource'
BODY = b'complete get representation'
ETAG_V1 = '"etag-v1"'
ETAG_V2 = '"etag-v2"'

# requests_mock needs URL normalization disabled to retain the custom protocol
NORMALIZE_PATCH = patch(
    'requests_cache.cache_keys.normalize_url', side_effect=lambda url, ignored: url
)


@pytest.fixture(params=['memory', 'sqlite'])
def head_session(request, tmp_path):
    """A CachedSession with HEAD-updates-GET enabled, backed by memory or SQLite storage"""
    if request.param == 'memory':
        backend = 'memory'
        cache_name = 'head_updates_get'
    else:
        backend = SQLiteCache(str(tmp_path / 'head_updates_get.sqlite'))
        cache_name = None
    with NORMALIZE_PATCH:
        session_kwargs = dict(
            backend=backend, expire_after=100, head_updates_get=True
        )
        if cache_name is not None:
            session_kwargs['cache_name'] = cache_name
        session = CachedSession(**session_kwargs)
        adapter = Adapter()
        for protocol in ('mock://', 'http+mock://', 'https+mock://'):
            session.mount(protocol, adapter)
        yield session
    session.close()


def test_head_with_same_etag_updates_headers_and_freshness(head_session):
    """A matching 200 HEAD response merges updatable headers, recomputes freshness, and keeps
    the complete GET body; the subsequent GET is served from the cache without going to origin.
    """
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri(
        'GET',
        URL,
        headers={'ETag': ETAG_V1, 'X-Old': '1', 'Content-Length': str(len(BODY))},
        content=BODY,
    )
    adapter.register_uri(
        'HEAD',
        URL,
        headers={'ETag': ETAG_V1, 'X-New': '2', 'Content-Length': str(len(BODY))},
    )

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key
        assert first_get.content == BODY and not first_get.from_cache

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        # A HEAD response never carries a business body
        assert head_response.content == b'' and head_response.text == ''

        stored = session.cache.get_response(get_key)
        assert stored.content == BODY
        assert stored.headers['X-New'] == '2'
        assert stored.headers['X-Old'] == '1'
        # Content-Length describing the GET body is retained, not replaced
        assert stored.headers['Content-Length'] == str(len(BODY))
        # revalidated is a non-serialized runtime flag; assert the observable freshness instead
        assert abs((stored.expires - (START_DT + timedelta(seconds=150))).total_seconds()) < 1
        # Freshness was recomputed from the HEAD response time
        assert not stored.is_expired
        if head_session.cache.__class__.__name__ == 'BaseCache':
            assert stored.revalidated is True

    with time_travel(START_DT + timedelta(seconds=90)):
        second_get = session.get(URL)
        assert second_get.from_cache is True
        assert second_get.content == BODY
        assert second_get.headers['X-New'] == '2'

    # Exactly one GET origin call; the second GET was satisfied from the refreshed cache entry
    get_calls = [request for request in adapter.request_history if request.method == 'GET']
    head_calls = [request for request in adapter.request_history if request.method == 'HEAD']
    assert len(get_calls) == 1
    assert len(head_calls) == 1


def test_head_with_changed_etag_conflicts_and_expires_get(head_session):
    """A changed ETag is conflicting evidence: the GET entry expires but keeps its full body,
    and the next GET must go back to the origin instead of serving the stale representation.
    """
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri(
        'GET',
        URL,
        [
            {
                'headers': {'ETag': ETAG_V1, 'Content-Length': str(len(BODY))},
                'content': BODY,
            },
            {
                'headers': {'ETag': ETAG_V2, 'Content-Length': str(len(BODY) + 1)},
                'content': BODY + b'2',
            },
        ],
    )
    adapter.register_uri(
        'HEAD',
        URL,
        headers={'ETag': ETAG_V2, 'Content-Length': str(len(BODY))},
    )

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key
        assert first_get.content == BODY

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.is_expired is True
        # The empty HEAD body must never replace the cached GET payload
        assert stored.content == BODY
        # Conflicting HEAD headers are not merged
        assert stored.headers.get('ETag') == ETAG_V1

    with time_travel(START_DT + timedelta(seconds=90)):
        second_get = session.get(URL)
        assert second_get.from_cache is False
        assert second_get.content == BODY + b'2'


def test_head_with_conflicting_content_length_expires_get(head_session):
    """A Content-Length mismatch against the cached body length invalidates the GET entry while
    preserving its body, even when the ETag still matches."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri('GET', URL, headers={'ETag': ETAG_V1}, content=BODY)
    adapter.register_uri(
        'HEAD', URL, headers={'ETag': ETAG_V1, 'Content-Length': str(len(BODY) + 100)}
    )

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.is_expired is True
        assert stored.content == BODY


def test_head_updates_only_matching_vary_variant(head_session):
    """HEAD associations only apply to representations with matching Vary request headers; other
    variants keep their own bodies and freshness metadata."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri(
        'GET',
        URL,
        headers={'Vary': 'Accept', 'ETag': '"json-v1"'},
        content=b'json body',
        request_headers={'Accept': 'application/json'},
    )
    adapter.register_uri(
        'GET',
        URL,
        headers={'Vary': 'Accept', 'ETag': '"html-v1"'},
        content=b'html body',
        request_headers={'Accept': 'text/html'},
    )
    adapter.register_uri(
        'HEAD',
        URL,
        headers={'Vary': 'Accept', 'ETag': '"json-v2"'},
        request_headers={'Accept': 'application/json'},
    )

    with time_travel(START_DT):
        json_get = session.get(URL, headers={'Accept': 'application/json'})
        html_get = session.get(URL, headers={'Accept': 'text/html'})
        assert json_get.content == b'json body' and html_get.content == b'html body'
        json_key, html_key = json_get.cache_key, html_get.cache_key
        assert json_key != html_key

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL, headers={'Accept': 'application/json'})
        assert head_response.content == b''

        json_stored = session.cache.get_response(json_key)
        html_stored = session.cache.get_response(html_key)
        assert json_stored.is_expired is True
        assert json_stored.content == b'json body'
        # The non-matching Vary variant is untouched and still fresh with its own body
        assert html_stored.is_expired is False
        assert html_stored.content == b'html body'


def test_head_with_unmatched_vary_does_not_touch_other_variant(head_session):
    """A HEAD whose Vary selection has no cached GET representation is a conservative no-op for
    every existing variant."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri(
        'GET',
        URL,
        headers={'Vary': 'Accept', 'ETag': ETAG_V1},
        content=BODY,
        request_headers={'Accept': 'text/html'},
    )
    adapter.register_uri(
        'HEAD',
        URL,
        headers={'Vary': 'Accept', 'ETag': ETAG_V1},
        request_headers={'Accept': 'application/json'},
    )

    with time_travel(START_DT):
        html_get = session.get(URL, headers={'Accept': 'text/html'})
        get_key = html_get.cache_key
        original_expires = session.cache.get_response(get_key).expires

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL, headers={'Accept': 'application/json'})
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.expires == original_expires
        assert stored.content == BODY


def test_head_without_validators_is_conservative_noop(head_session):
    """Without agreeing validators there is not enough evidence: headers and freshness of the
    cached GET response are left unchanged even if Content-Length happens to match."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri(
        'GET', URL, headers={'Content-Length': str(len(BODY))}, content=BODY
    )
    adapter.register_uri(
        'HEAD',
        URL,
        headers={'Content-Length': str(len(BODY)), 'X-New': 'ignored'},
    )

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key
        original_expires = session.cache.get_response(get_key).expires

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.expires == original_expires
        assert 'X-New' not in stored.headers
        assert stored.content == BODY


def test_non_200_head_does_not_fake_successful_validation(head_session):
    """Error responses to HEAD must not update or validate the cached GET representation."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri('GET', URL, headers={'ETag': ETAG_V1}, content=BODY)
    adapter.register_uri('HEAD', URL, status_code=500)

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        assert head_response.status_code == 500
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.is_expired is False
        assert stored.revalidated is False
        assert stored.content == BODY


def test_feature_disabled_leaves_cached_get_untouched():
    """With head_updates_get disabled (the default), HEAD and GET stay cached separately and
    the GET entry is not modified."""
    with NORMALIZE_PATCH:
        session = CachedSession(backend='memory', expire_after=100, head_updates_get=False)
        adapter = Adapter()
        for protocol in ('mock://', 'http+mock://', 'https+mock://'):
            session.mount(protocol, adapter)
        adapter.register_uri('GET', URL, headers={'ETag': ETAG_V1}, content=BODY)
        adapter.register_uri('HEAD', URL, headers={'ETag': ETAG_V1, 'X-New': 'ignored'})

        with time_travel(START_DT):
            first_get = session.get(URL)
            get_key = first_get.cache_key

        with time_travel(START_DT + timedelta(seconds=50)):
            head_response = session.head(URL)
            assert head_response.content == b''
            stored = session.cache.get_response(get_key)
            assert stored.is_expired is False
            assert 'X-New' not in stored.headers
            assert stored.content == BODY
    session.close()


def test_redirect_alias_still_resolves_after_head_update(head_session):
    """Updating the target GET entry in place keeps redirect aliases pointing at the same
    representation, with its body and expiration metadata intact."""
    session = head_session
    redirect_url = 'http+mock://head-updates.example.com/redirect'
    target_url = 'http+mock://head-updates.example.com/target'
    adapter = session.get_adapter(target_url)
    adapter.register_uri(
        'GET', redirect_url, status_code=302, headers={'Location': target_url}
    )
    adapter.register_uri('GET', target_url, headers={'ETag': ETAG_V1}, content=BODY)
    adapter.register_uri(
        'HEAD', target_url, headers={'ETag': ETAG_V1, 'X-New': 'refreshed'}
    )

    with time_travel(START_DT):
        redirected = session.get(redirect_url)
        assert redirected.content == BODY
        alias_key = session.cache.create_key(redirected.history[0].request)
        target_key = session.cache.redirects[alias_key]

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(target_url)
        assert head_response.content == b''

    # The alias still maps to the same storage key, whose body and headers are consistent
    assert session.cache.redirects[alias_key] == target_key
    target_stored = session.cache.responses[target_key]
    assert target_stored.content == BODY
    assert target_stored.headers['X-New'] == 'refreshed'
    assert session.cache.get_response(alias_key).content == BODY

    with time_travel(START_DT + timedelta(seconds=90)):
        assert session.get(redirect_url).from_cache is True


def test_head_entry_never_holds_the_get_body(head_session):
    """The separately stored HEAD entry (when HEAD is an allowable method) always has an empty
    body and can never masquerade as the complete GET representation."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri(
        'GET', URL, headers={'ETag': ETAG_V1, 'Content-Length': str(len(BODY))}, content=BODY
    )
    adapter.register_uri(
        'HEAD', URL, headers={'ETag': ETAG_V1, 'Content-Length': str(len(BODY))}
    )

    with time_travel(START_DT):
        session.get(URL)
    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        head_key = head_response.cache_key
        stored_head = session.cache.get_response(head_key)
        assert stored_head.content == b''
        assert stored_head.size == 0


def test_head_update_respects_vary_wildcard(head_session):
    """A cached response with `Vary: *` can never be associated with a later HEAD selection, so
    the update is conservatively skipped."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri('GET', URL, headers={'Vary': '*', 'ETag': ETAG_V1}, content=BODY)
    adapter.register_uri('HEAD', URL, headers={'Vary': '*', 'ETag': ETAG_V1, 'X-New': 'ignored'})

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key
        original_expires = session.cache.get_response(get_key).expires

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.expires == original_expires
        assert 'X-New' not in stored.headers
        assert stored.content == BODY


def test_head_update_skips_hop_by_hop_headers(head_session):
    """Hop-by-hop transport headers from the HEAD response must not be merged into the cached
    GET representation."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri('GET', URL, headers={'ETag': ETAG_V1}, content=BODY)
    adapter.register_uri(
        'HEAD',
        URL,
        headers={
            'ETag': ETAG_V1,
            'Transfer-Encoding': 'chunked',
            'Connection': 'keep-alive, X-Connection-Only',
            'X-New': 'kept',
        },
    )

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL)
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.headers['X-New'] == 'kept'
        assert 'Transfer-Encoding' not in stored.headers
        assert 'X-Connection-Only' not in stored.headers
        assert stored.content == BODY


def test_method_override_header_triggers_head_update():
    """A non-HEAD request that becomes HEAD via a method-override header still updates the cached
    GET representation."""
    with NORMALIZE_PATCH:
        session = CachedSession(backend='memory', expire_after=100, head_updates_get=True)
        adapter = Adapter()
        for protocol in ('mock://', 'http+mock://', 'https+mock://'):
            session.mount(protocol, adapter)
        adapter.register_uri('GET', URL, headers={'ETag': ETAG_V1}, content=BODY)
        adapter.register_uri('POST', URL, headers={'ETag': ETAG_V1, 'X-New': 'from-override'})

        with time_travel(START_DT):
            first_get = session.get(URL)
            get_key = first_get.cache_key

        with time_travel(START_DT + timedelta(seconds=50)):
            overridden = session.post(
                URL, headers={'X-HTTP-Method-Override': 'HEAD'}
            )
            assert overridden.content == b''
            stored = session.cache.get_response(get_key)
            assert stored.headers['X-New'] == 'from-override'
            assert stored.content == BODY
    session.close()


def test_no_store_head_does_not_modify_cached_get(head_session):
    """A HEAD request carrying Cache-Control: no-store skips all cache writes, including the
    RFC 9111 section 4.3.5 GET update."""
    session = head_session
    adapter = session.get_adapter(URL)
    adapter.register_uri('GET', URL, headers={'ETag': ETAG_V1}, content=BODY)
    adapter.register_uri('HEAD', URL, headers={'ETag': ETAG_V1, 'X-New': 'ignored'})

    with time_travel(START_DT):
        first_get = session.get(URL)
        get_key = first_get.cache_key
        original_expires = session.cache.get_response(get_key).expires

    with time_travel(START_DT + timedelta(seconds=50)):
        head_response = session.head(URL, headers={'Cache-Control': 'no-store'})
        assert head_response.content == b''
        stored = session.cache.get_response(get_key)
        assert stored.expires == original_expires
        assert 'X-New' not in stored.headers
        assert stored.content == BODY
