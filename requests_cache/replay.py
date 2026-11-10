"""Portable replay packages: export a selected set of cached responses to a versioned,
non-pickle file, import them into another cache, and replay them offline.

Supported backends: in-memory (``memory``) and ``sqlite``.

Typical usage::

    # On the machine that ran the original collection
    export_replay_package(session, keys, 'collection.replay.json')

    # On a colleague's machine
    session = CachedSession('replay_cache', backend='sqlite')
    import_replay_package('collection.replay.json', session)
    replay = ReplaySession('replay_cache', backend='sqlite', package='collection.replay.json')
    replay.get('https://api.example.com/items?page=1')

.. automodsumm:: requests_cache.replay
   :classes-only:
   :nosignatures:

.. automodsumm:: requests_cache.replay
   :functions-only:
   :nosignatures:
"""

from __future__ import annotations

import json
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Union

from requests.hooks import dispatch_hook

from .backends import BaseCache, SQLiteCache
from .models import CachedResponse
from .policy import CacheActions, utcnow
from .serializers import SERIALIZERS, json_serializer
from .session import CachedSession

if TYPE_CHECKING:
    from requests import PreparedRequest

__all__ = [
    'FORMAT_NAME',
    'FORMAT_VERSION',
    'ReplayError',
    'ReplayMiss',
    'ReplayPackageError',
    'ReplayConfigError',
    'ReplayExportError',
    'ReplaySession',
    'export_replay_package',
    'import_replay_package',
    'load_replay_manifest',
]

FORMAT_NAME = 'requests-cache-replay'
FORMAT_VERSION = 1

logger = getLogger(__name__)

CacheOrSession = Union[BaseCache, CachedSession]
StrOrPath = Union[str, Path]


class ReplayError(Exception):
    """Base class for replay package errors"""


class ReplayMiss(ReplayError, LookupError):
    """A replay session could not satisfy a request from the cache. Raised instead of sending
    a request over the network."""


class ReplayPackageError(ReplayError, ValueError):
    """A replay package is invalid: unsupported version, corrupted content, broken redirect
    references, or conflicting cache keys"""


class ReplayConfigError(ReplayPackageError):
    """Key-related configuration differs between the package and the target cache/session"""


class ReplayExportError(ReplayError):
    """A cache cannot be exported to a portable replay package"""


def export_replay_package(
    source: CacheOrSession,
    keys: Iterable[str],
    path: StrOrPath,
) -> Dict:
    """Export a caller-selected set of cached responses to a portable replay package.

    The package contains the selected responses (including binary bodies, duplicate request
    parameters, Vary variants, final URLs, and response history), any redirect aliases pointing
    at them, and the key-related settings needed to reproduce request matching.

    Args:
        source: Cache or session to export from (memory or SQLite backend)
        keys: Cache keys of the responses to export
        path: File path to write the package to

    Raises:
        ReplayExportError: If the backend is unsupported, a custom ``key_fn`` or
            non-portable serializer is in use, or a requested key is not cached
    """
    cache = _as_cache(source)
    _check_supported(cache)
    key_config = _export_key_config(cache)

    responses = {}
    for key in keys:
        response = cache.responses.get(key)
        if response is None:
            raise ReplayExportError(f'No cached response for key: {key}')
        responses[key] = {
            'response': json_serializer.dumps(response),
            'sha256': _body_checksum(response),
        }

    # Include only redirect aliases that point at an exported response
    redirects = {k: v for k, v in cache.redirects.items() if v in responses}

    package = {
        'format': FORMAT_NAME,
        'format_version': FORMAT_VERSION,
        'created_at': utcnow().isoformat(),
        'key_config': key_config,
        'responses': responses,
        'redirects': redirects,
    }
    with open(path, 'w') as f:
        json.dump(package, f, indent=2)
    logger.info(f'Exported {len(responses)} responses and {len(redirects)} redirects to {path}')
    return package


def import_replay_package(
    path: StrOrPath,
    target: CacheOrSession,
    overwrite: bool = False,
) -> int:
    """Import a replay package into a cache. The import is all-or-nothing: the package is
    fully validated before anything is written, and a failure during writing is rolled back,
    so a failed import never leaves a partial batch of records behind.

    Args:
        path: Path to a replay package created by :py:func:`export_replay_package`
        target: Cache or session to import into (memory or SQLite backend)
        overwrite: Allow replacing existing cache entries; by default, existing keys are
            rejected

    Returns:
        Number of imported responses

    Raises:
        ReplayPackageError: If the package is invalid or any key already exists
        ReplayConfigError: If key-related settings differ between the package and the target
    """
    cache = _as_cache(target)
    _check_supported(cache)
    package = load_replay_manifest(path)
    _check_key_config(package['key_config'], cache)

    # Deserialize and verify all response bodies before writing anything
    responses: Dict[str, CachedResponse] = {}
    for key, entry in package['responses'].items():
        responses[key] = _deserialize(key, entry)

    # Redirect aliases must reference responses contained in the package
    redirects: Dict[str, str] = dict(package['redirects'])
    dangling = sorted(set(redirects.values()) - set(responses))
    if dangling:
        raise ReplayPackageError(
            f'Redirect aliases reference responses missing from the package: {dangling}'
        )

    if not overwrite:
        conflicts = sorted(
            [k for k in responses if k in cache.responses or k in cache.redirects]
            + [k for k in redirects if k in cache.responses or k in cache.redirects]
        )
        if conflicts:
            raise ReplayPackageError(
                f'Refusing to overwrite {len(conflicts)} existing cache key(s): {conflicts}'
            )

    written = []
    try:
        for key, response in responses.items():
            cache.responses[key] = response
            written.append((cache.responses, key))
        for key, target_key in redirects.items():
            cache.redirects[key] = target_key
            written.append((cache.redirects, key))
    except Exception as e:
        for storage, key in reversed(written):
            try:
                del storage[key]
            except Exception:
                logger.warning(f'Failed to roll back cache key: {key}')
        raise ReplayPackageError(f'Import failed and was rolled back: {e}') from e

    logger.info(f'Imported {len(responses)} responses and {len(redirects)} redirects')
    return len(responses)


def load_replay_manifest(path: StrOrPath) -> Dict:
    """Load a replay package and validate its format and version, without importing it"""
    try:
        with open(path) as f:
            package = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ReplayPackageError(f'Cannot read replay package {path}: {e}') from e

    if not isinstance(package, dict) or package.get('format') != FORMAT_NAME:
        raise ReplayPackageError(f'Not a {FORMAT_NAME} package: {path}')
    if package.get('format_version') != FORMAT_VERSION:
        raise ReplayPackageError(
            f'Unsupported replay package version: {package.get("format_version")} '
            f'(expected {FORMAT_VERSION})'
        )
    for section in ('key_config', 'responses', 'redirects'):
        if not isinstance(package.get(section), dict):
            raise ReplayPackageError(f'Malformed replay package: missing section {section!r}')
    for field in ('ignored_parameters', 'match_headers', 'content_root_key', 'serializer_str'):
        if field not in package['key_config']:
            raise ReplayPackageError(
                f'Malformed replay package: missing key_config field {field!r}'
            )
    return package


class ReplaySession(CachedSession):
    """A read-only session that serves requests exclusively from a previously populated cache
    (typically one filled with :py:func:`import_replay_package`). Behavior differs from
    :py:class:`.CachedSession` as follows:

    * Expired responses are still used
    * A cache miss raises :py:class:`ReplayMiss` instead of sending a request
    * No code path sends a network request or writes to the cache

    Args:
        package: Optional path to a replay package. If provided, key-related settings are
            validated against the package when the session is opened, so configuration
            mismatches are reported immediately instead of causing cache misses later.
    """

    def __init__(self, *args, package: Optional[StrOrPath] = None, **kwargs):
        super().__init__(*args, **kwargs)
        _check_supported(self.cache)
        self.settings.read_only = True
        if package is not None:
            manifest = load_replay_manifest(package)
            _check_key_config(manifest['key_config'], self.cache)

    def send(self, request: 'PreparedRequest', **kwargs) -> CachedResponse:  # type: ignore
        """Look up a cached response without ever sending a request. Cache keys and Vary
        handling match :py:meth:`CachedSession.send`, but expiration is ignored and a miss
        raises :py:class:`ReplayMiss`.
        """
        # Work on a copy so validation headers added below don't mutate the caller's request
        request = request.copy()
        actions = CacheActions.from_request(
            self.cache.create_key(request, **kwargs), request, self.settings
        )
        cached_response = self.cache.get_response(actions.cache_key)
        actions.update_from_cached_response(cached_response, self.cache.create_key, **kwargs)

        # Secondary Vary lookup, as in CachedSession.send
        if actions.vary_cache_key is not None:
            cached_response = self.cache.get_response(actions.vary_cache_key)

        if cached_response is None:
            raise ReplayMiss(
                f'No cached response for {request.method} {request.url} '
                '(replay sessions never send requests)'
            )
        return dispatch_hook('response', request.hooks, cached_response, **kwargs)

    def _send_and_cache(self, request, actions, cached_response=None, **kwargs):
        raise ReplayMiss(f'Replay sessions never send requests: {request.url}')

    def _resend(self, request, actions, cached_response, **kwargs):
        raise ReplayMiss(f'Replay sessions never send requests: {request.url}')

    def _resend_async(self, request, actions, cached_response, **kwargs):
        raise ReplayMiss(f'Replay sessions never send requests: {request.url}')


def _as_cache(obj: CacheOrSession) -> BaseCache:
    if isinstance(obj, BaseCache):
        return obj
    cache = getattr(obj, 'cache', None)
    if isinstance(cache, BaseCache):
        return cache
    raise TypeError(f'Expected a BaseCache or CachedSession, got {type(obj).__name__}')


def _check_supported(cache: BaseCache):
    if not (type(cache) is BaseCache or isinstance(cache, SQLiteCache)):
        raise ReplayError(
            'Replay packages only support memory and SQLite backends, '
            f'not {type(cache).__name__}'
        )


def _normalize_match_headers(match_headers) -> Union[bool, list]:
    if isinstance(match_headers, bool):
        return match_headers
    return sorted(h.lower() for h in (match_headers or []))


def _export_key_config(cache: BaseCache) -> Dict:
    """Capture the settings that determine cache keys, so they can be replayed elsewhere"""
    settings = cache._settings
    if settings.key_fn is not None:
        raise ReplayExportError(
            'Cannot export a cache that uses a custom key_fn; '
            'it cannot be represented in a portable package'
        )

    serializer = cache.responses.serializer
    serializer_name = getattr(serializer, 'name', None)
    if serializer is not None and serializer_name not in SERIALIZERS:
        raise ReplayExportError(
            f'Cannot export a cache that uses a custom serializer ({serializer}); '
            'only built-in named serializers are portable'
        )

    return {
        'ignored_parameters': sorted(settings.ignored_parameters or []),
        'match_headers': _normalize_match_headers(settings.match_headers),
        'content_root_key': settings.content_root_key,
        'serializer': serializer_name,
        'serializer_str': str(serializer),
    }


def _check_key_config(key_config: Dict, cache: BaseCache):
    """Fail if the target cache/session can't reproduce the package's cache keys"""
    settings = cache._settings
    problems = []
    if settings.key_fn is not None:
        problems.append('target uses a custom key_fn')

    serializer_str = str(cache.responses.serializer)
    if serializer_str != key_config['serializer_str']:
        problems.append(
            f'serializer: package has {key_config["serializer_str"]!r}, '
            f'target has {serializer_str!r}'
        )

    ignored = sorted(settings.ignored_parameters or [])
    if ignored != key_config['ignored_parameters']:
        problems.append(
            f'ignored_parameters: package has {key_config["ignored_parameters"]}, '
            f'target has {ignored}'
        )

    match_headers = _normalize_match_headers(settings.match_headers)
    if match_headers != key_config['match_headers']:
        problems.append(
            f'match_headers: package has {key_config["match_headers"]}, '
            f'target has {match_headers}'
        )

    if settings.content_root_key != key_config['content_root_key']:
        problems.append(
            f'content_root_key: package has {key_config["content_root_key"]!r}, '
            f'target has {settings.content_root_key!r}'
        )

    if problems:
        raise ReplayConfigError(
            'Key configuration does not match the replay package; cache keys computed '
            'from new requests would not match the packaged ones:\n  - ' + '\n  - '.join(problems)
        )


def _deserialize(key: str, entry: Dict) -> CachedResponse:
    try:
        response = json_serializer.loads(entry['response'])
    except Exception as e:
        raise ReplayPackageError(f'Cannot deserialize response for key {key}: {e}') from e
    checksum = _body_checksum(response)
    if checksum != entry.get('sha256'):
        raise ReplayPackageError(
            f'Body checksum mismatch for key {key}: '
            f'expected {entry.get("sha256")}, got {checksum}'
        )
    return response


def _body_checksum(response: CachedResponse) -> str:
    return sha256(response._content or b'').hexdigest()
