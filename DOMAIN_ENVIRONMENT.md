# Reproducible authoring environment

Upstream: https://github.com/requests-cache/requests-cache
Upstream commit: 757803a161b926d9d3f25f6bc0e8b9d0f63ab599
Verified runtime: 3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]
Verified platform: linux

Use an isolated virtual environment. Install the pinned dependencies with
`python -m pip install -r requirements-domain.lock`, then `python -m pip install -e .`.
Set the following environment variables before installation and testing:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

Core baseline:

```text
python -m pytest -q -o addopts= tests/unit/test_cache_keys.py tests/unit/test_base_cache.py tests/unit/test_session.py
```

Measured pytest process time: 6.224 seconds on the authoring host.
Verified in Ubuntu 26.04 / WSL. The time-machine test extension could not load under this Windows host policy; use the documented Linux environment. No OS policy settings were changed.

All documented checks use in-memory fixtures or local temporary files. No production
accounts, data services, credentials or external network endpoints are required after setup.
Reset by making a fresh checkout of this seed and creating a new virtual environment.
Test fixtures remain under their upstream repository license.
