"""Cache invalidation following RFC 9111, section 4.4 ("Invalidation")

A response to an unsafe request method (for example, POST, PUT, DELETE, or PATCH) can change the
state of one or more resources on the origin server. After a successful unsafe-method response,
the cache needs to invalidate the affected stored representations so that a subsequent request is
forwarded to the origin instead of returning stale data.

The set of URIs to invalidate consists of:

* The effective request URI of the unsafe request
* The URIs in the response's ``Location`` and ``Content-Location`` headers, but only when they are
  same-origin with the effective request URI

Relative URI references are resolved against the response's effective URI (which, after redirects,
may differ from the request URI). Cross-origin references must never be used to remove another
origin's cached entries.
"""

from logging import getLogger
from typing import List, Optional
from urllib.parse import urljoin, urlsplit

from requests import PreparedRequest, Response

__all__ = [
    'UNSAFE_METHODS',
    'get_invalidation_urls',
    'is_successful_invalidation_status',
    'is_unsafe_method',
    'same_origin',
]

# RFC 9111 section 4.4 references the "safe methods" defined by RFC 9110; every other registered
# method is unsafe. These are the commonly used unsafe methods.
UNSAFE_METHODS = frozenset(('POST', 'PUT', 'PATCH', 'DELETE'))

# Header fields whose URI values can identify additional representations that were modified
INVALIDATION_HEADERS = ('Location', 'Content-Location')

logger = getLogger(__name__)


def is_unsafe_method(request: PreparedRequest) -> bool:
    """Check whether the request uses an unsafe method, per RFC 9110"""
    method = (request.method or '').upper()
    return bool(method) and method not in ('GET', 'HEAD', 'OPTIONS', 'TRACE', 'CONNECT')


def is_successful_invalidation_status(status_code: int) -> bool:
    """Check whether the response status code indicates a successful unsafe-method operation.

    RFC 9111 section 4.4 triggers invalidation on a "successful response", which corresponds to a
    2xx (Successful) status code. Responses with any other status leave existing stored
    representations untouched.
    """
    return 200 <= status_code < 300


def same_origin(url_1: str, url_2: str) -> bool:
    """Check whether two URLs share the same origin (scheme, host, and port)

    Default ports for the scheme are normalized so that, for example, ``http://example.com`` and
    ``http://example.com:80`` are considered the same origin.
    """
    try:
        first = urlsplit(url_1)
        second = urlsplit(url_2)
    except ValueError:
        return False
    if not (first.scheme and first.hostname and second.scheme and second.hostname):
        return False
    return (
        first.scheme.lower() == second.scheme.lower()
        and first.hostname.lower() == second.hostname.lower()
        and _normalized_port(first) == _normalized_port(second)
    )


def get_invalidation_urls(request: PreparedRequest, response: Optional[Response]) -> List[str]:
    """Collect the same-origin URIs to invalidate after a successful unsafe-method response.

    Args:
        request: The unsafe request that was sent
        response: The response received from the origin (may be unavailable when handling a
            network failure)

    Returns:
        A de-duplicated list of absolute URIs, starting with the effective request URI
    """
    request_url = request.url or ''
    urls: List[str] = [request_url]

    if response is not None:
        # The base for resolving relative references is the response's effective request URI,
        # which accounts for any redirects that were followed
        base_url = response.url or request_url
        for header_name in INVALIDATION_HEADERS:
            header_value = response.headers.get(header_name)
            if not header_value:
                continue
            resolved_url = urljoin(base_url, header_value)
            if same_origin(request_url, resolved_url):
                urls.append(resolved_url)
            else:
                logger.debug(
                    f'Skipping cross-origin invalidation URI {resolved_url!r} '
                    f'from {header_name} header (request URI: {request_url!r})'
                )

    # Preserve order while removing duplicates
    return list(dict.fromkeys(url for url in urls if url))


def _normalized_port(url_tokens) -> Optional[int]:
    if url_tokens.port is not None:
        return url_tokens.port
    return {'http': 80, 'https': 443}.get(url_tokens.scheme.lower())
