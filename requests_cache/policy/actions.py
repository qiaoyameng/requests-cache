from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Dict, List, Mapping, MutableMapping, Optional, Union

from attrs import define, field
from requests import PreparedRequest, Response
from requests.structures import CaseInsensitiveDict

from .._utils import coalesce
from ..cache_keys import normalize_headers
from ..models import RichMixin
from . import (
    DO_NOT_CACHE,
    EXPIRE_IMMEDIATELY,
    NEVER_EXPIRE,
    CacheDirectives,
    ExpirationTime,
    KeyCallback,
    get_expiration_datetime,
    get_expiration_seconds,
    get_url_expiration,
    utcnow,
)
from .settings import CacheSettings

if TYPE_CHECKING:
    from ..models import CachedRequest, CachedResponse

# Nonstandard headers that can be used to override the request method
METHOD_OVERRIDE_HEADERS = [
    'X-HTTP-Method-Override',
    'X-HTTP-Method',
    'X-Method-Override',
]

logger = getLogger(__name__)

# Result values for CacheActions.update_cached_get_from_head()
HEAD_UPDATED = 'updated'
HEAD_CONFLICT = 'conflict'
HEAD_NO_OP = 'no-op'

# Hop-by-hop headers (RFC 9111, section 7.1.3); these describe the transport of the HEAD
# response rather than the representation, so they must not be merged into a cached GET response
HOP_BY_HOP_HEADERS = frozenset(
    [
        'connection',
        'keep-alive',
        'proxy-authenticate',
        'proxy-authorization',
        'te',
        'trailer',
        'transfer-encoding',
        'upgrade',
    ]
)


@define(repr=False)
class CacheActions(RichMixin):
    """Translates cache settings and headers into specific actions to take for a given cache item.
     The resulting actions are then handled in :py:meth:`CachedSession.send`.

    .. rubric:: Notes

    * See :ref:`precedence` for behavior if multiple sources provide an expiration
    * See :ref:`headers` for more details about header behavior
    * The following arguments/properties are the outputs of this class:

    Args:
        cache_key: The cache key created based on the initial request
        error_504: Indicates the request cannot be fulfilled based on cache settings
        expire_after: User or header-provided expiration value
        send_request: Send a new request
        resend_request: Send a new request to refresh a stale cache item
        resend_async: Return a stale cache item, and send a non-blocking request to refresh it
        skip_read: Skip reading from the cache
        skip_write: Skip writing to the cache
    """

    # Outputs
    cache_key: str = field(default=None, repr=False)
    error_504: bool = field(default=False)
    expire_after: ExpirationTime = field(default=None)
    send_request: bool = field(default=False)
    resend_request: bool = field(default=False)
    resend_async: bool = field(default=False)
    skip_read: bool = field(default=False)
    skip_write: bool = field(default=False)
    vary_cache_key: Optional[str] = field(default=None)

    # Inputs
    _directives: CacheDirectives = field(default=None, repr=False)
    _settings: CacheSettings = field(default=None, repr=False)

    # Temporary attributes
    _only_if_cached: bool = field(default=False, repr=False)
    _refresh: bool = field(default=False, repr=False)
    _request: PreparedRequest = field(default=None, repr=False)
    _stale_if_error: Union[bool, ExpirationTime] = field(default=None, repr=False)
    _stale_while_revalidate: Union[bool, ExpirationTime] = field(default=None, repr=False)
    _validation_headers: Dict[str, str] = field(factory=dict, repr=False)
    _vary_matched: bool = field(default=True, repr=False)

    # Get session-level setting for stale_if_error and stale_while_revalidate, kept separate
    # from the above so a non-matching Vary variant's headers don't affect a later matching one
    _base_stale_if_error: Union[bool, ExpirationTime] = field(default=None, repr=False)
    _base_stale_while_revalidate: Union[bool, ExpirationTime] = field(default=None, repr=False)

    @classmethod
    def from_request(
        cls,
        cache_key: str,
        request: PreparedRequest,
        settings: Optional[CacheSettings] = None,
    ):
        """Initialize from request info and cache settings.

        Note on refreshing: `must-revalidate` isn't a standard request header, but is used here to
        indicate a user-requested refresh. Typically that's only used in response headers, and
        `max-age=0` would be used by a client to request a refresh. However, this would conflict
        with the `expire_after` option provided in :py:meth:`.CachedSession.request`.

        Args:
            cache_key: The cache key created based on the initial request
            request: The outgoing request
            settings: Session-level cache settings
        """
        settings = settings or CacheSettings()
        directives = CacheDirectives.from_headers(request.headers)
        logger.debug(f'Cache directives from request headers: {directives}')

        # Check expiration values in order of precedence
        expire_after = coalesce(
            directives.max_age,
            get_url_expiration(request.url, settings.urls_expire_after),
            settings.expire_after,
        )

        # Merge values that may come from either settings (requests-cache-specific) or headers (standard)
        actual_no_cache = directives.actual_no_cache or expire_after == DO_NOT_CACHE
        kinda_no_cache = directives.no_cache or directives.no_store
        only_if_cached = settings.only_if_cached or directives.only_if_cached
        refresh = directives.max_age == EXPIRE_IMMEDIATELY or directives.must_revalidate
        stale_if_error = settings.stale_if_error or directives.stale_if_error
        stale_while_revalidate = (
            settings.stale_while_revalidate or directives.stale_while_revalidate
        )

        # Check and log conditions for reading from the cache
        read_criteria = {
            'disabled cache': settings.disabled,
            'disabled method': not _is_method_allowed(request, settings),
            'disabled by headers or refresh': kinda_no_cache,
            'disabled by expiration': actual_no_cache,
        }
        _log_cache_criteria('read', read_criteria)

        actions = cls(
            cache_key=cache_key,
            directives=directives,
            expire_after=expire_after,
            only_if_cached=only_if_cached,
            refresh=refresh,
            request=request,
            settings=settings,
            skip_read=any(read_criteria.values()),
            skip_write=directives.no_store,
            stale_if_error=stale_if_error,
            stale_while_revalidate=stale_while_revalidate,
            base_stale_if_error=stale_if_error,
            base_stale_while_revalidate=stale_while_revalidate,
        )
        return actions

    @property
    def expires(self) -> Optional[datetime]:
        """Convert the user/header-provided expiration value to a datetime. Applies to new cached
        responses, and previously cached responses that are being revalidated.
        """
        return get_expiration_datetime(self.expire_after)

    def is_usable(self, cached_response: Optional['CachedResponse'], error: bool = False):
        """Determine whether a given cached response is "fresh enough" to satisfy the request,
        based on:

        * min-fresh
        * max-stale
        * stale-if-error (if an error has occurred)
        * stale-while-revalidate
        """
        if cached_response is None:
            return False
        elif (
            cached_response.expires is None
            or (cached_response.is_expired and self._stale_while_revalidate is True)
            or (error and self._stale_if_error is True)
        ):
            return True
        # Handle stale_if_error as a time value
        elif error and self._stale_if_error:
            offset = timedelta(seconds=get_expiration_seconds(self._stale_if_error))
        # Handle stale_while_revalidate as a time value
        elif cached_response.is_expired and self._stale_while_revalidate:
            offset = timedelta(seconds=get_expiration_seconds(self._stale_while_revalidate))
        # Handle min-fresh and max-stale
        else:
            offset = self._directives.get_expire_offset()

        return utcnow() < cached_response.expires + offset

    def update_from_cached_response(
        self,
        cached_response: Optional['CachedResponse'],
        create_key: Optional[KeyCallback] = None,
        **key_kwargs,
    ):
        """Determine if we can reuse a cached response, or set headers for a conditional request
        if possible.

        Used after fetching a cached response, but before potentially sending a new request.

        Args:
            cached_response: Cached response to examine
            create_key: Cache key function, used for validating ``Vary`` headers
            key_kwargs: Additional keyword arguments for ``create_key``.
        """
        self._vary_matched = True
        if cached_response is not None and self._settings.cache_control:
            self._update_from_cached_response_headers(cached_response)

        usable_response = self.is_usable(cached_response)
        usable_if_error = self.is_usable(cached_response, error=True)

        # Can't satisfy the request
        if not usable_response and self._only_if_cached and not usable_if_error:
            self.error_504 = True
        # Send the request for the first time
        elif cached_response is None:
            self.send_request = True
        # If response contains Vary and doesn't match, consider it a cache miss
        elif create_key and not self._validate_vary(cached_response, create_key, **key_kwargs):
            self._vary_matched = False
            self.send_request = True
        # Resend the request, unless settings permit a stale response
        elif not usable_response and not (self._only_if_cached and usable_if_error):
            self.resend_request = True
        # Resend the request in the background; meanwhile return stale response
        elif (
            cached_response.is_expired
            and usable_response
            and self._stale_while_revalidate
            and not self._only_if_cached
        ):
            self.resend_async = True

        if cached_response is not None and not self._only_if_cached:
            self._update_validation_headers(cached_response)
        logger.debug(f'Post-read cache actions: {self}')

    def update_from_response(self, response: Response):
        """Update expiration + actions based on headers and other details from a new response.

        Used after receiving a new response, but before saving it to the cache.
        """
        directives = CacheDirectives.from_headers(response.headers)
        if self._settings.cache_control:
            self._update_from_response_headers(directives)

        # If "expired" but there's a validator, save it to the cache and revalidate on use
        skip_stale = self.expire_after == EXPIRE_IMMEDIATELY and not directives.has_validator
        do_not_cache = self.expire_after == DO_NOT_CACHE

        # Apply filter callback, if any
        callback = self._settings.filter_fn
        filtered_out = callback is not None and not callback(response)

        # Check and log conditions for writing to the cache
        write_criteria = {
            'disabled cache': self._settings.disabled,
            'disabled method': not _is_method_allowed(response.request, self._settings),
            'disabled status': response.status_code not in self._settings.allowable_codes,
            'disabled by filter': filtered_out,
            'disabled by headers': self.skip_write,
            'disabled by expiration': do_not_cache or skip_stale,
            'disabled by read-only': self._settings.read_only,
        }
        self.skip_write = any(write_criteria.values())
        _log_cache_criteria('write', write_criteria)

    def update_request(self, request: PreparedRequest) -> PreparedRequest:
        """Apply validation headers (if any) before sending a request"""
        request.headers.update(self._validation_headers)
        return request

    def update_revalidated_response(
        self, response: Response, cached_response: 'CachedResponse'
    ) -> 'CachedResponse':
        """After revalidation, update the cached response's expiration and headers"""
        logger.debug(f'Response for URL {response.request.url} has not been modified')

        # Skip updating the cached response if expiration and headers are unchanged
        # OR if session/request settings skip it.
        # Ignore validators missing from new response, since they may be omitted
        headers_changed = any(
            cached_response.headers.get(k) != v for k, v in response.headers.items()
        )
        expiration_changed = self.expires != cached_response.expires
        permanent_skip = (
            self._settings.read_only or self._settings.disabled or self._directives.no_store
        )
        self.skip_write = permanent_skip or not (expiration_changed or headers_changed)

        cached_response.expires = self.expires

        # Update headers;
        # Account for Transfer-Encoding in original response and Content-Length in 304 response
        cached_response.headers.update(response.headers)
        if (
            'Content-Length' in cached_response.headers
            and 'Transfer-Encoding' in cached_response.headers
        ):
            del cached_response.headers['Content-Length']

        cached_response.revalidated = True
        cached_response.has_content_changed = False
        return cached_response

    def update_cached_get_from_head(
        self,
        head_response: Response,
        cached_get: 'CachedResponse',
        create_key: Optional[KeyCallback] = None,
        **key_kwargs,
    ) -> str:
        """Update a cached full GET representation from a 200 HEAD response (RFC 9111, sections
        3.4 and 4.3.5).

        The cached GET response must already be associated with the same target URI and a
        matching set of request headers selected by ``Vary``. A 200 HEAD response may update the
        cached response's header fields and freshness, but its (empty) payload must never replace
        the cached GET body.

        Returns one of:

        * ``HEAD_UPDATED``: Validators and any ``Content-Length`` agree; updatable response
          headers were merged and freshness was recomputed.
        * ``HEAD_CONFLICT``: The evidence is inconsistent (a mismatched validator or
          ``Content-Length``); the cached GET response was marked expired and left otherwise
          untouched.
        * ``HEAD_NO_OP``: Not enough evidence to act on (for example, no validators), or the
          responses do not represent the same selection. No state was changed.
        """
        if head_response.status_code != 200 or cached_get is None:
            return HEAD_NO_OP

        # Only representations selected by the same target URI and Vary request headers may be
        # associated; a non-matching Vary variant must not affect the cached GET response
        if create_key and not self._vary_headers_match(cached_get, create_key, **key_kwargs)[0]:
            logger.debug(
                f'HEAD response for {head_response.request.url} does not match cached GET '
                'Vary selection; not updating'
            )
            return HEAD_NO_OP

        head_headers = CaseInsensitiveDict(head_response.headers)
        cached_headers = cached_get.headers

        if self._head_evidence_conflicts(head_headers, cached_get):
            logger.info(
                f'HEAD response for {head_response.request.url} conflicts with cached GET '
                'representation; marking cached response expired'
            )
            # Invalidate the stale representation without touching its body or other metadata
            cached_get.expires = utcnow()
            return HEAD_CONFLICT

        # A successful validation requires at least one validator that agrees with the cached
        # representation; without such evidence the conservative behavior is to do nothing
        if not self._head_validators_agree(head_headers, cached_headers):
            logger.debug(
                f'HEAD response for {head_response.request.url} lacks matching validators; '
                'not updating cached GET response'
            )
            return HEAD_NO_OP

        self._merge_head_headers(head_headers, cached_headers)
        cached_get.expires = self.expires
        cached_get.revalidated = True
        cached_get.has_content_changed = False
        logger.debug(
            f'Updated headers and freshness of cached GET response for '
            f'{head_response.request.url} from HEAD response'
        )
        return HEAD_UPDATED

    @staticmethod
    def _head_evidence_conflicts(
        head_headers: CaseInsensitiveDict, cached_get: 'CachedResponse'
    ) -> bool:
        """Check whether the HEAD 200 response contains evidence contradicting the cached GET
        representation. Only evidence present on both sides can produce a conflict; missing
        values are treated conservatively as inconclusive.
        """
        cached_headers = cached_get.headers

        head_etag = head_headers.get('ETag')
        cached_etag = cached_headers.get('ETag')
        if head_etag and cached_etag and head_etag != cached_etag:
            logger.debug(
                f'HEAD ETag {head_etag!r} does not match cached ETag {cached_etag!r}'
            )
            return True

        head_last_modified = _parse_http_datetime(head_headers.get('Last-Modified'))
        cached_last_modified = _parse_http_datetime(cached_headers.get('Last-Modified'))
        if (
            head_last_modified is not None
            and cached_last_modified is not None
            and head_last_modified != cached_last_modified
        ):
            logger.debug(
                'HEAD Last-Modified does not match cached Last-Modified'
            )
            return True

        # Content-Length describes the representation body, which must remain the GET body.
        # Compare against the cached Content-Length when present, otherwise the actual body size.
        head_length = _parse_content_length(head_headers.get('Content-Length'))
        if head_length is not None:
            cached_length = _parse_content_length(cached_headers.get('Content-Length'))
            if cached_length is None and cached_get._content is not None:
                cached_length = len(cached_get._content)
            if cached_length is not None and head_length != cached_length:
                logger.debug(
                    f'HEAD Content-Length {head_length} does not match cached body length '
                    f'{cached_length}'
                )
                return True

        return False

    @staticmethod
    def _head_validators_agree(
        head_headers: CaseInsensitiveDict, cached_headers: CaseInsensitiveDict
    ) -> bool:
        """A HEAD response can validate a cached representation only when at least one validator
        is present on both sides and has the same value."""
        head_etag = head_headers.get('ETag')
        if head_etag and head_etag == cached_headers.get('ETag'):
            return True

        head_last_modified = _parse_http_datetime(head_headers.get('Last-Modified'))
        cached_last_modified = _parse_http_datetime(cached_headers.get('Last-Modified'))
        return (
            head_last_modified is not None
            and cached_last_modified is not None
            and head_last_modified == cached_last_modified
        )

    @staticmethod
    def _merge_head_headers(
        head_headers: CaseInsensitiveDict, cached_headers: CaseInsensitiveDict
    ):
        """Merge response header fields allowed to be updated from a HEAD response into the
        cached GET response headers. Representation body metadata and hop-by-hop headers are
        retained from the cached response so its body stays consistent.
        """
        excluded = set(HOP_BY_HOP_HEADERS)
        excluded.update(['content-length'])
        # Headers listed in the Connection header are hop-by-hop for this connection
        connection = head_headers.get('Connection')
        if connection:
            excluded.update(token.strip().lower() for token in connection.split(','))

        for key, value in head_headers.items():
            if key.lower() not in excluded:
                cached_headers[key] = value

    def _update_from_cached_response_headers(self, cached_response: 'CachedResponse'):
        """Recover stale-if-error and stale-while-revalidate directives from a cached response"""
        directives = CacheDirectives.from_headers(cached_response.headers)
        self._stale_if_error = self._base_stale_if_error or directives.stale_if_error
        self._stale_while_revalidate = (
            self._base_stale_while_revalidate or directives.stale_while_revalidate
        )

    def _update_from_response_headers(self, directives: CacheDirectives):
        """Check response headers for expiration and other cache directives"""
        logger.debug(f'Cache directives from response headers: {directives}')

        self._stale_if_error = self._stale_if_error or directives.stale_if_error
        if directives.immutable:
            self.expire_after = NEVER_EXPIRE
        else:
            self.expire_after = coalesce(
                directives.max_age,
                _validate_expires_header(directives.expires),
                self.expire_after,
            )
        self.skip_write = self.skip_write or directives.no_store

    def _update_validation_headers(self, cached_response: 'CachedResponse'):
        """If needed, get validation headers based on a cached response. Revalidation may be
        triggered by a stale response, request headers, or cached response headers.
        """
        directives = CacheDirectives.from_headers(cached_response.headers)
        # These conditions always apply
        revalidate = directives.has_validator and (
            cached_response.is_expired or self._refresh or self._settings.always_revalidate
        )
        # These conditions only apply if cache_control=True
        cc_revalidate = self._settings.cache_control and (
            directives.no_cache or directives.must_revalidate
        )

        # Add the appropriate validation headers, if needed
        if revalidate or cc_revalidate:
            if directives.etag:
                self._validation_headers['If-None-Match'] = directives.etag
            if directives.last_modified:
                self._validation_headers['If-Modified-Since'] = directives.last_modified
            self.send_request = False
            self.resend_request = True

    def _validate_vary(
        self, cached_response: 'CachedResponse', create_key: KeyCallback, **key_kwargs
    ) -> bool:
        """If the cached response contains Vary, check that the specified request headers match"""
        headers_match, match_headers = self._vary_headers_match(
            cached_response, create_key, **key_kwargs
        )
        if not headers_match and match_headers is not None:
            # Compute secondary cache key with merged Vary + user match_headers
            key_kwargs['match_headers'] = self._merge_match_headers(match_headers)
            self.vary_cache_key = create_key(self._request, **key_kwargs)
        return headers_match

    def _vary_headers_match(
        self, cached_response: 'CachedResponse', create_key: KeyCallback, **key_kwargs
    ) -> tuple:
        """Check Vary request-header selection without mutating any action state.

        Returns a tuple of ``(headers_match, match_headers)``; ``match_headers`` is the list of
        request header names (minus Cookie) used for the comparison, or ``None`` when no header
        comparison was needed.
        """
        vary = cached_response.headers.get('Vary')
        if not vary:
            return True, None
        elif vary == '*':
            return False, None

        # Generate a secondary cache key based on Vary for both the cached request and new request.
        # If there are redirects, compare the new request against the last request in the chain.
        match_headers = [k.strip().lower() for k in vary.split(',')]
        vary_request = (
            cached_response.history[-1].request
            if cached_response.history
            else cached_response.request
        )

        # Handle 'Vary: Cookie' separately since requests doesn't store cookies with other headers
        if 'cookie' in match_headers:
            if not self._cookies_match(vary_request):
                logger.debug('Failed Vary check: cookies do not match')
                return False, None
            match_headers.remove('cookie')

        if not match_headers:
            return True, None

        # If a Vary header is in ignored_parameters and on the request, treat it as a cache miss
        # rather than potentially returning an invalid response
        ignored = {h.lower() for h in (self._settings.ignored_parameters or [])}
        ignore_overlap = ignored & set(match_headers)
        if ignore_overlap and any(h in self._request.headers for h in ignore_overlap):
            logger.debug(
                f'Failed Vary check: Vary header(s) are in ignored_parameters: {ignore_overlap}'
            )
            return False, None

        key_kwargs['match_headers'] = match_headers
        vary_key_cached = create_key(vary_request, **key_kwargs)
        # A HEAD request associates with representations of the equivalent GET request (RFC 9111,
        # section 4.3.5), so compare on the GET target rather than on the request method itself
        vary_key_current = create_key(self._vary_selection_request(), **key_kwargs)
        headers_match = vary_key_current == vary_key_cached
        if not headers_match:
            _log_vary_diff(
                self._request.headers,
                vary_request.headers,
                key_kwargs['match_headers'],
            )
        return headers_match, match_headers

    def _vary_selection_request(self) -> PreparedRequest:
        """Return the request used for Vary selection-key comparison. A HEAD request selects the
        same representations as the equivalent GET on the same target URI.
        """
        if (self._request.method or '').upper() == 'HEAD':
            get_request = self._request.copy()
            get_request.method = 'GET'
            for override_header in METHOD_OVERRIDE_HEADERS:
                get_request.headers.pop(override_header, None)
            return get_request
        return self._request

    def _cookies_match(self, cached_request: 'CachedRequest') -> bool:
        """Compare cookies between the current request and cached request for Vary: Cookie"""

        def normalize_cookies(cookies: Optional[Mapping]) -> str:
            cookie_list = [f'{k}={v}' for k, v in (cookies or {}).items()]
            return '; '.join(sorted(cookie_list))

        cached_cookies = normalize_cookies(cached_request.cookies)
        request_cookies = normalize_cookies(getattr(self._request, '_cookies', None))
        return request_cookies == cached_cookies

    def _merge_match_headers(self, vary_headers: List[str]) -> Union[List[str], bool]:
        """Merge Vary headers with user-configured match_headers to build a complete
        set of headers for the secondary cache key."""
        user_headers = self._settings.match_headers
        if user_headers is True:
            return True
        if not user_headers:
            return vary_headers
        return sorted(set(vary_headers + [h.lower() for h in user_headers]))


def _validate_expires_header(expires: Optional[str]) -> ExpirationTime:
    """An ``Expires`` header with an invalid date must be treated as 'already expired'
    (RFC 9111, section 5.3), rather than raising an error for the whole request"""
    if expires is None:
        return None
    try:
        get_expiration_datetime(expires)
    except ValueError:
        logger.debug(f'Invalid Expires header: {expires!r}; treating as expired')
        return EXPIRE_IMMEDIATELY
    return expires


def _is_method_allowed(request: PreparedRequest, settings: CacheSettings) -> bool:
    """Check request method as well as method override headers"""
    headers = request.headers or CaseInsensitiveDict()
    methods = [headers.get(k) for k in METHOD_OVERRIDE_HEADERS]
    methods += [request.method]
    return any(m is not None and m in settings.allowable_methods for m in methods)


def _parse_http_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an HTTP-date header value (e.g. Last-Modified), returning None when absent/invalid"""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def _parse_content_length(value: Optional[str]) -> Optional[int]:
    """Parse a Content-Length header value, returning None when absent or malformed"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _log_vary_diff(
    headers_1: MutableMapping[str, str],
    headers_2: MutableMapping[str, str],
    vary: List[str],
):
    """Log which specific headers specified by Vary did not match"""
    if logger.level > DEBUG:
        return
    headers_1 = normalize_headers(headers_1)
    headers_2 = normalize_headers(headers_2)
    nonmatching = [k for k in vary if headers_1.get(k) != headers_2.get(k)]
    logger.debug(f'Failed Vary check. Non-matching headers: {", ".join(nonmatching)}')


def _log_cache_criteria(operation: str, criteria: Dict):
    """Log details on any failed checks for cache read or write"""
    if logger.level > DEBUG:
        return
    if any(criteria.values()):
        status = ', '.join([k for k, v in criteria.items() if v])
    else:
        status = 'Passed'
    logger.debug(f'Pre-{operation} cache checks: {status}')
