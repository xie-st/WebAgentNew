"""
API key availability checking for effGen providers.

``check_keys()`` returns a mapping of provider → key-available so the CLI
``effgen doctor`` command and user code can inspect which providers are
ready without attempting any network call.
"""

from __future__ import annotations

import os
from typing import Any


def check_keys(providers: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Check which provider API keys are present in the environment.

    Reads the ProviderRegistry (lazily, to avoid circular imports) and
    checks each provider's ``env_keys`` list.  The *first* key found counts
    as present.

    Args:
        providers: Optional list of provider names to check.  If ``None``
                   (default) all registered providers are checked.

    Returns:
        Dict ``{provider_name: {"available": bool, "env_key": str|None}}``.
        ``env_key`` is the name of the variable that was found, or ``None``
        if none were set.
    """
    from effgen.models.registry import ProviderRegistry  # lazy to avoid circular

    registered = ProviderRegistry.list_providers()
    targets = providers if providers is not None else registered

    result: dict[str, dict[str, Any]] = {}
    for prov in targets:
        if prov not in registered:
            result[prov] = {"available": False, "env_key": None, "error": "not registered"}
            continue

        info = ProviderRegistry.get_provider_info(prov)
        env_keys = info.get("env_keys", [])
        found_key: str | None = None
        for k in env_keys:
            if os.environ.get(k):
                found_key = k
                break

        result[prov] = {
            "available": found_key is not None,
            "env_key": found_key,
            "env_keys_checked": env_keys,
        }

    return result
