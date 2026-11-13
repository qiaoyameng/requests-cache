"""Utilities for migrating cache keys when request matching rules change."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from attrs import evolve

from .cache_keys import create_key, normalize_request, normalize_url
from .models import CachedRequest, CachedResponse
from .policy import KeyCallback
from .policy.settings import CacheSettings

if TYPE_CHECKING:
    from .backends.base import BaseCache


class KeyMigrationError(RuntimeError):
    """Raised when a key migration cannot be safely planned or applied."""


@dataclass
class ResponseKeyEntry:
    """A planned key migration for one response record."""

    old_key: str
    new_key: str
    old_base_key: str
    new_base_key: str
    old_vary_key: Optional[str]
    new_vary_key: Optional[str]
    vary_headers: List[str]
    entry_type: str
    request_method: str
    request_url: str


@dataclass
class RedirectKeyEntry:
    """A planned key migration for one redirect alias."""

    old_key: str
    new_key: str
    old_target_key: str
    new_target_key: str
    request_method: str
    request_url: str


@dataclass
class KeyMigrationConflict:
    """Two or more records that would use the same new key but are not equivalent."""

    new_key: str
    old_keys: List[str]
    reason: str


@dataclass
class KeyMigrationPlan:
    """A complete, validated description of a key rule migration."""

    version: str
    applicable: bool
    response_entries: List[ResponseKeyEntry] = field(default_factory=list)
    redirect_entries: List[RedirectKeyEntry] = field(default_factory=list)
    conflicts: List[KeyMigrationConflict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    old_ignored_parameters: Optional[List[str]] = None
    old_match_headers: Any = None
    old_content_root_key: Optional[str] = None
    old_key_fn: Optional[Callable[..., str]] = field(default=None, compare=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation of the plan."""
        data = asdict(self)
        data.pop('old_key_fn', None)
        return data


def plan_key_migration(
    cache: 'BaseCache',
    old_ignored_parameters: Optional[List[str]] = None,
    old_key_fn: Optional[KeyCallback] = None,
    old_match_headers: Any = None,
    old_content_root_key: Optional[str] = None,
) -> KeyMigrationPlan:
    """Build a migration plan without modifying the cache.

    The current cache settings represent the new key rules. The optional ``old_*`` arguments
    describe the rules that were active when the existing records were written.
    """
    current_settings = cache._settings
    if current_settings.key_fn is not None and old_key_fn is None:
        unknown_custom_key = True
    else:
        unknown_custom_key = False

    old_settings = evolve(
        current_settings,
        ignored_parameters=old_ignored_parameters
        if old_ignored_parameters is not None
        else current_settings.ignored_parameters,
        key_fn=old_key_fn or create_key,
        match_headers=old_match_headers
        if old_match_headers is not None
        else current_settings.match_headers,
        content_root_key=old_content_root_key
        if old_content_root_key is not None
        else current_settings.content_root_key,
    )

    plan = KeyMigrationPlan(
        version=_package_version(),
        applicable=True,
        old_ignored_parameters=list(old_settings.ignored_parameters or []),
        old_match_headers=old_settings.match_headers,
        old_content_root_key=old_settings.content_root_key,
        old_key_fn=old_key_fn or create_key,
    )
    if unknown_custom_key:
        plan.applicable = False
        plan.errors.append(
            'Cannot migrate a cache that uses a custom key_fn without the original key function'
        )
        return plan

    cache.validate_key_migration_storage(plan)
    _build_response_entries(cache, old_settings, plan)
    _build_redirect_entries(cache, old_settings, plan)
    _detect_conflicts(cache, plan)
    plan.applicable = not plan.errors and not plan.conflicts
    return plan


def apply_key_migration(cache: 'BaseCache', plan: KeyMigrationPlan) -> KeyMigrationPlan:
    """Apply a migration plan atomically.

    If validation or any storage operation fails, the original keys and records remain available
    under the previous cache configuration.
    """
    if plan.version != _package_version():
        raise KeyMigrationError(
            f'Cache library version changed: plan was created with {plan.version!r}, '
            f'current version is {_package_version()!r}'
        )
    if not plan.applicable:
        details = '; '.join(plan.errors + [conflict.reason for conflict in plan.conflicts])
        raise KeyMigrationError(f'Key migration plan is not applicable: {details}')

    cache.validate_key_migration_storage(plan)
    _validate_plan_is_current(cache, plan)
    cache.apply_key_migration_storage(plan)
    return plan


def _build_response_entries(
    cache: 'BaseCache', old_settings: CacheSettings, plan: KeyMigrationPlan
):
    response_entries: Dict[str, ResponseKeyEntry] = {}

    for stored_key in list(cache.responses.keys()):
        response = cache.responses.get(stored_key)
        if response is None:
            plan.errors.append(f'Response {stored_key!r} could not be deserialized')
            continue
        if not _has_recoverable_request(response.request):
            plan.errors.append(f'Response {stored_key!r} does not contain a recoverable request')
            continue

        try:
            key_request = _record_request(response)
            old_base_key = _create_key(cache, key_request, old_settings)
            new_base_key = _create_key(cache, key_request, cache._settings)
            vary_headers, cookie_vary = _parse_vary_headers(response)
            vary_request = response.history[-1].request if response.history else response.request
            old_vary_key = _create_vary_key(cache, vary_request, old_settings, vary_headers)
            new_vary_key = _create_vary_key(cache, vary_request, cache._settings, vary_headers)
        except Exception as error:
            plan.errors.append(f'Unable to recreate keys for response {stored_key!r}: {error}')
            continue

        if stored_key == old_base_key:
            entry_type = 'response'
            old_key, new_key = old_base_key, new_base_key
        elif old_vary_key is not None and stored_key == old_vary_key:
            entry_type = 'vary'
            old_key, new_key = old_vary_key, new_vary_key
        else:
            plan.errors.append(
                f'Stored response key {stored_key!r} does not match a key derived from its request'
            )
            continue

        entry = ResponseKeyEntry(
            old_key=old_key,
            new_key=new_key,
            old_base_key=old_base_key,
            new_base_key=new_base_key,
            old_vary_key=old_vary_key,
            new_vary_key=new_vary_key,
            vary_headers=vary_headers + (['cookie'] if cookie_vary else []),
            entry_type=entry_type,
            request_method=key_request.method or '',
            request_url=key_request.url or '',
        )
        response_entries[stored_key] = entry

    plan.response_entries = list(response_entries.values())


def _build_redirect_entries(
    cache: 'BaseCache', old_settings: CacheSettings, plan: KeyMigrationPlan
):
    expected_redirects: Dict[str, RedirectKeyEntry] = {}

    for entry in plan.response_entries:
        response = cache.responses[entry.old_key]
        if response.is_redirect:
            continue
        for redirect in response.history:
            if not _has_recoverable_request(redirect.request):
                plan.errors.append(
                    f'Redirect history for response {entry.old_key!r} contains an unrecoverable '
                    'request'
                )
                continue
            old_key = _create_key(cache, redirect.request, old_settings)
            new_key = _create_key(cache, redirect.request, cache._settings)
            redirect_entry = RedirectKeyEntry(
                old_key=old_key,
                new_key=new_key,
                old_target_key=entry.old_key,
                new_target_key=entry.new_key,
                request_method=redirect.request.method or '',
                request_url=redirect.request.url or '',
            )
            existing = expected_redirects.get(old_key)
            if existing is not None and existing.old_target_key != entry.old_key:
                plan.errors.append(f'Redirect alias {old_key!r} points to multiple responses')
            if existing is None:
                expected_redirects[old_key] = redirect_entry

    for old_key, target_key in cache.redirects.items():
        expected = expected_redirects.get(old_key)
        if expected is None:
            plan.errors.append(f'Redirect alias {old_key!r} has no matching redirect request')
            continue
        if target_key != expected.old_target_key:
            plan.errors.append(f'Redirect alias {old_key!r} points to an unexpected response')

    for expected in expected_redirects.values():
        if expected.old_key not in cache.redirects:
            plan.errors.append(
                f'Missing redirect alias {expected.old_key!r} for {expected.request_url!r}'
            )

    new_alias_targets = {}
    for expected in expected_redirects.values():
        existing_target = new_alias_targets.get(expected.new_key)
        if existing_target is not None and existing_target != expected.new_target_key:
            plan.errors.append(
                f'New redirect alias {expected.new_key!r} points to multiple responses'
            )
        new_alias_targets[expected.new_key] = expected.new_target_key

    plan.redirect_entries = list(expected_redirects.values())


def _validate_plan_is_current(cache: 'BaseCache', plan: KeyMigrationPlan):
    response_keys = set(cache.responses.keys())
    redirect_keys = set(cache.redirects.keys())
    planned_response_keys = {entry.old_key for entry in plan.response_entries}
    planned_redirect_keys = {entry.old_key for entry in plan.redirect_entries}
    if response_keys != planned_response_keys:
        raise KeyMigrationError('Response records changed after the migration plan was created')
    if redirect_keys != planned_redirect_keys:
        raise KeyMigrationError('Redirect records changed after the migration plan was created')

    old_settings = evolve(
        cache._settings,
        ignored_parameters=plan.old_ignored_parameters,
        key_fn=plan.old_key_fn or create_key,
        match_headers=plan.old_match_headers,
        content_root_key=plan.old_content_root_key,
    )
    for entry in plan.response_entries:
        response = cache.responses[entry.old_key]
        if entry.entry_type == 'vary':
            record_request = (
                response.history[-1].request if response.history else response.request
            )
            vary_headers = [header for header in entry.vary_headers if header != 'cookie']
            old_key = _create_vary_key(cache, record_request, old_settings, vary_headers)
            new_key = _create_vary_key(cache, record_request, cache._settings, vary_headers)
        else:
            record_request = _record_request(response)
            old_key = _create_key(cache, record_request, old_settings)
            new_key = _create_key(cache, record_request, cache._settings)
        if old_key != entry.old_key or new_key != entry.new_key:
            raise KeyMigrationError('Cache key settings changed after the plan was created')

    for entry in plan.redirect_entries:
        response = cache.responses[entry.old_target_key]
        redirect = next(
            item
            for item in response.history
            if _has_recoverable_request(item.request)
            and _create_key(cache, item.request, old_settings) == entry.old_key
        )
        old_key = _create_key(cache, redirect.request, old_settings)
        new_key = _create_key(cache, redirect.request, cache._settings)
        if old_key != entry.old_key or new_key != entry.new_key:
            raise KeyMigrationError('Redirect key settings changed after the plan was created')


def _detect_conflicts(cache: 'BaseCache', plan: KeyMigrationPlan):
    groups: Dict[str, List[ResponseKeyEntry]] = {}
    for entry in plan.response_entries:
        groups.setdefault(entry.new_key, []).append(entry)

    for new_key, entries in groups.items():
        if len(entries) == 1:
            continue
        responses = [cache.responses[entry.old_key] for entry in entries]
        request_signatures = {
            _request_signature(response.request, cache._settings.ignored_parameters)
            for response in responses
        }
        if len(request_signatures) != 1:
            reason = 'requests have different normalized semantics'
        else:
            response_signatures = {
                _response_signature(response, cache._settings.ignored_parameters)
                for response in responses
            }
            reason = '' if len(response_signatures) == 1 else 'response representations differ'
        if reason:
            plan.conflicts.append(
                KeyMigrationConflict(
                    new_key=new_key,
                    old_keys=sorted(entry.old_key for entry in entries),
                    reason=reason,
                )
            )


def _create_key(cache: 'BaseCache', request: CachedRequest, settings: CacheSettings) -> str:
    key_fn = settings.key_fn or create_key
    return key_fn(
        request=request,
        ignored_parameters=settings.ignored_parameters,
        content_root_key=settings.content_root_key,
        match_headers=settings.match_headers,
        serializer=cache.responses.serializer,
    )


def _create_vary_key(
    cache: 'BaseCache',
    request: CachedRequest,
    settings: CacheSettings,
    vary_headers: List[str],
) -> Optional[str]:
    if not vary_headers:
        return None
    match_headers = _merge_match_headers(settings.match_headers, vary_headers)
    return _create_key(cache, request, evolve(settings, match_headers=match_headers))


def _merge_match_headers(match_headers: Any, vary_headers: List[str]) -> Any:
    if match_headers is True:
        return True
    headers = [header.lower() for header in (match_headers or [])]
    return sorted(set(headers + vary_headers))


def _parse_vary_headers(response: CachedResponse) -> Tuple[List[str], bool]:
    vary = response.headers.get('Vary')
    if not vary or vary == '*':
        return [], False
    headers = [header.strip().lower() for header in vary.split(',') if header.strip()]
    cookie_vary = 'cookie' in headers
    return [header for header in headers if header != 'cookie'], cookie_vary


def _has_recoverable_request(request: CachedRequest) -> bool:
    return bool(request is not None and request.method and request.url)


def _record_request(response: CachedResponse) -> CachedRequest:
    return response.history[0].request if response.history else response.request


def _request_signature(request: CachedRequest, ignored_parameters: Any) -> Tuple[Any, ...]:
    normalized = normalize_request(request, ignored_parameters)
    return (
        normalized.method,
        normalized.url,
        tuple(sorted(normalized.headers.items())),
        normalized.body,
        tuple(sorted(dict(normalized.cookies).items())),
    )


def _response_signature(response: CachedResponse, ignored_parameters: Any) -> Tuple[Any, ...]:
    return (
        response._content,
        response.status_code,
        tuple(sorted(response.headers.items())),
        normalize_url(response.url or '', ignored_parameters),
        response.encoding,
        response.reason,
        tuple(sorted(dict(response.cookies).items())),
        _request_signature(response.request, ignored_parameters),
        _optional_request_signature(response._next, ignored_parameters),
        tuple(
            _response_signature(item, ignored_parameters) for item in response.history
        ),
    )


def _optional_request_signature(
    request: Optional[CachedRequest], ignored_parameters: Any
) -> Optional[Tuple[Any, ...]]:
    return (
        _request_signature(request, ignored_parameters) if request is not None else None
    )


def _package_version() -> str:
    try:
        return package_version('requests-cache')
    except PackageNotFoundError:
        return 'unknown'
