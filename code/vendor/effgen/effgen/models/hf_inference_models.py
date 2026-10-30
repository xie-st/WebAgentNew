"""
HuggingFace Inference API model registry for effGen — dynamic catalog.

The HuggingFace Inference Providers API (a.k.a. "HF Router") fronts a rotating
set of third-party providers (Together, Novita, Sambanova, Fireworks, Cerebras,
Groq, Cohere, Featherless, Nebius, ...) and serves them under a single
OpenAI-compatible surface at ``https://router.huggingface.co/v1``.

Because availability and pricing change frequently, effGen ships:

1. A bundled snapshot of the full router catalog (``_data/hf_inference_catalog.json``)
   so the package works offline / without an extra network call on import.
2. :func:`refresh_models` — re-fetch from the live router endpoint, write a
   local cache (``~/.effgen/cache/hf_inference_catalog.json``), and update the
   in-process registry.
3. :func:`check_drift` — compare bundled vs. live and warn the user when the
   shipped snapshot is significantly out of date.

Each model entry is normalised to the same dict shape used by the rest of
effGen (``context``, ``max_output``, ``supports_native_tools``,
``pricing_per_1m_input``, ``pricing_per_1m_output``, ...) plus the raw HF
provider list under the ``providers`` key for power users.

Snapshot metadata is exposed via :data:`REGISTRY_FETCH_DATE` and
:data:`CATALOG_SOURCE`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

_BUNDLED_PATH = Path(__file__).parent / "_data" / "hf_inference_catalog.json"
_USER_CACHE_PATH = Path.home() / ".effgen" / "cache" / "hf_inference_catalog.json"
CATALOG_SOURCE = "https://router.huggingface.co/v1/models"


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load HF catalog snapshot %s: %s", path, exc)
        return None


def _pick_best_provider(providers: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the provider entry to use for normalised fields.

    Preference order:
      1. ``status == "live"`` over anything else.
      2. Has explicit ``context_length``.
      3. Has explicit ``pricing``.
      4. Cheapest input price.
    """
    if not providers:
        return {}

    live = [p for p in providers if p.get("status") == "live"] or providers

    def _score(p: dict[str, Any]) -> tuple[int, int, float]:
        has_ctx = 1 if p.get("context_length") else 0
        has_price = 1 if p.get("pricing") else 0
        price_in = float((p.get("pricing") or {}).get("input", 1e9) or 1e9)
        return (-has_ctx, -has_price, price_in)

    return sorted(live, key=_score)[0]


def _normalise_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one router /v1/models entry to the effGen registry shape."""
    arch = raw.get("architecture") or {}
    providers = raw.get("providers") or []
    best = _pick_best_provider(providers)
    pricing = best.get("pricing") or {}

    supports_native_tools = any(p.get("supports_tools") for p in providers)
    supports_structured = any(p.get("supports_structured_output") for p in providers)

    context = best.get("context_length") or 0
    if not context:
        # fall back to any provider that exposes one
        for p in providers:
            if p.get("context_length"):
                context = p["context_length"]
                break
    if not context:
        context = 8_192

    input_mods = list(arch.get("input_modalities") or ["text"])
    return {
        "display_name": raw.get("id", ""),
        "family": _family_of(raw.get("id", "")),
        "organization": raw.get("owned_by", ""),
        "context": context,
        "max_output": min(context, 16_384),
        "supports_native_tools": supports_native_tools,
        "supports_structured_output": supports_structured,
        "supports_vision": "image" in input_mods,
        "supports_audio": "audio" in input_mods,
        "input_modalities": input_mods,
        "output_modalities": list(arch.get("output_modalities") or ["text"]),
        "requires_endpoint": False,
        "use_text_generation": False,
        "pricing_per_1m_input": float(pricing.get("input", 0.0) or 0.0),
        "pricing_per_1m_output": float(pricing.get("output", 0.0) or 0.0),
        "providers": providers,
        "preferred_provider": best.get("provider", "auto"),
        "provider": "auto",  # adapter routes via HF Router by default
    }


def _family_of(model_id: str) -> str:
    lower = model_id.lower()
    for tag in (
        "qwen3", "qwen2.5", "qwen2", "qwen", "llama-3.3", "llama-3.1", "llama-3",
        "llama", "mixtral", "mistral", "gemma-3", "gemma-2", "gemma",
        "deepseek-v3", "deepseek-r1", "deepseek", "phi-4", "phi-3", "phi",
        "gpt-oss", "kimi", "command", "zephyr", "minimax", "glm",
    ):
        if tag in lower:
            return tag
    org, _, _ = model_id.partition("/")
    return org or "unknown"


def _build_registry(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in snapshot.get("data", []):
        mid = raw.get("id")
        if not mid:
            continue
        out[mid] = _normalise_entry(raw)
    return out


# ---------------------------------------------------------------------------
# Initial load: prefer user cache (newest), fall back to bundled snapshot.
# ---------------------------------------------------------------------------

_user_snap = _load_snapshot(_USER_CACHE_PATH)
_bundled_snap = _load_snapshot(_BUNDLED_PATH)
_active_snap = _user_snap or _bundled_snap or {"data": [], "_fetched_at": "unknown"}

HF_MODELS: dict[str, dict[str, Any]] = _build_registry(_active_snap)
REGISTRY_FETCH_DATE: str = (_active_snap.get("_fetched_at") or "unknown")[:10]
HF_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
if HF_DEFAULT_MODEL not in HF_MODELS and HF_MODELS:
    # Pick a sensible default if Qwen2.5-7B isn't in the snapshot for some reason.
    for candidate in (
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ):
        if candidate in HF_MODELS:
            HF_DEFAULT_MODEL = candidate
            break
    else:
        HF_DEFAULT_MODEL = next(iter(HF_MODELS))

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def available_models() -> list[str]:
    """All registered model IDs in the active snapshot."""
    return list(HF_MODELS.keys())


def chat_models() -> list[str]:
    """Models that use the chat_completion() interface (not text_generation)."""
    return [m for m, info in HF_MODELS.items() if not info.get("use_text_generation")]


def tool_capable_models() -> list[str]:
    """Models advertising native function-calling on at least one provider."""
    return [m for m, info in HF_MODELS.items() if info.get("supports_native_tools")]


def serverless_models() -> list[str]:
    """Models reachable via the HF Router (no dedicated endpoint required)."""
    return [m for m, info in HF_MODELS.items() if not info.get("requires_endpoint")]


def endpoint_only_models() -> list[str]:
    """Models that require a dedicated Inference Endpoint."""
    return [m for m, info in HF_MODELS.items() if info.get("requires_endpoint")]


def get_model_info(model_id: str) -> dict[str, Any] | None:
    """Return registry entry for *model_id*, or None if not registered."""
    return HF_MODELS.get(model_id)


def list_providers_for(model_id: str) -> list[dict[str, Any]]:
    """Return the raw HF Router provider list for *model_id*, or []."""
    info = HF_MODELS.get(model_id) or {}
    return list(info.get("providers") or [])


def cheapest_provider(model_id: str) -> str | None:
    """Return the cheapest live provider for *model_id*, or None."""
    providers = list_providers_for(model_id)
    live = [p for p in providers if p.get("status") == "live"] or providers
    if not live:
        return None
    live.sort(key=lambda p: float((p.get("pricing") or {}).get("input", 1e9) or 1e9))
    return live[0].get("provider")


def suggest_alternatives(model_id: str, n: int = 3) -> list[str]:
    """Return up to *n* suggestions when *model_id* is unavailable."""
    info = HF_MODELS.get(model_id, {})
    family = info.get("family", "")

    same_family = [
        m
        for m in serverless_models()
        if m != model_id and HF_MODELS[m].get("family") == family
    ]
    others = [
        m
        for m in serverless_models()
        if m != model_id and m not in same_family
    ]
    candidates = same_family + others
    return candidates[:n]


# ---------------------------------------------------------------------------
# Dynamic refresh + drift detection
# ---------------------------------------------------------------------------


def _fetch_live(token: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Hit ``router.huggingface.co/v1/models`` and return parsed JSON."""
    import datetime

    import requests

    headers = {}
    tok = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    resp = requests.get(CATALOG_SOURCE, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    data["_fetched_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    data["_source"] = CATALOG_SOURCE
    return data


def refresh_models(
    token: str | None = None,
    timeout: float = 30.0,
    write_cache: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fetch the live HF Router catalog and update :data:`HF_MODELS` in place.

    Writes the snapshot to ``~/.effgen/cache/hf_inference_catalog.json`` so
    the next process start picks it up without a network round-trip.

    Args:
        token: Optional HF token; falls back to ``HF_TOKEN`` /
            ``HUGGINGFACE_API_KEY`` env vars.  Anonymous requests are
            accepted by the router but rate-limited.
        timeout: HTTP timeout in seconds.
        write_cache: Persist the fetched snapshot to disk.

    Returns:
        The updated registry dict (same object as :data:`HF_MODELS`).
    """
    global REGISTRY_FETCH_DATE

    snap = _fetch_live(token=token, timeout=timeout)
    new_registry = _build_registry(snap)

    HF_MODELS.clear()
    HF_MODELS.update(new_registry)
    REGISTRY_FETCH_DATE = (snap.get("_fetched_at") or "unknown")[:10]

    if write_cache:
        try:
            _USER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _USER_CACHE_PATH.open("w") as f:
                json.dump(snap, f, indent=2)
            logger.info(
                "HF Inference catalog refreshed: %d models written to %s",
                len(new_registry),
                _USER_CACHE_PATH,
            )
        except OSError as exc:
            logger.warning("Could not persist HF catalog cache: %s", exc)

    return HF_MODELS


def check_drift(
    token: str | None = None,
    timeout: float = 30.0,
    warn: bool = True,
) -> dict[str, Any]:
    """Compare the bundled / in-process snapshot against the live router.

    Returns a dict with ``added``, ``removed``, ``pricing_changed``,
    ``bundled_count``, ``live_count``, ``bundled_fetched_at``,
    ``live_fetched_at``.

    If *warn* is True and any drift is detected, emits a single warning via
    the module logger so users know to call :func:`refresh_models`.

    Args:
        token: Credential for the live listing, otherwise read from the
            environment.
        timeout: Seconds to wait for the router before giving up.
        warn: Whether to emit the one-time refresh suggestion on drift.
    """
    live_snap = _fetch_live(token=token, timeout=timeout)
    live_registry = _build_registry(live_snap)
    bundled_keys = set(HF_MODELS.keys())
    live_keys = set(live_registry.keys())

    added = sorted(live_keys - bundled_keys)
    removed = sorted(bundled_keys - live_keys)

    pricing_changed: list[str] = []
    for k in bundled_keys & live_keys:
        old = HF_MODELS[k]
        new = live_registry[k]
        if (
            old.get("pricing_per_1m_input") != new.get("pricing_per_1m_input")
            or old.get("pricing_per_1m_output") != new.get("pricing_per_1m_output")
        ):
            pricing_changed.append(k)

    report = {
        "added": added,
        "removed": removed,
        "pricing_changed": sorted(pricing_changed),
        "bundled_count": len(bundled_keys),
        "live_count": len(live_keys),
        "bundled_fetched_at": REGISTRY_FETCH_DATE,
        "live_fetched_at": (live_snap.get("_fetched_at") or "")[:10],
    }

    if warn and (added or removed or pricing_changed):
        logger.warning(
            "HF Inference catalog drift detected vs bundled snapshot "
            "(fetched %s): +%d models, -%d models, %d pricing changes. "
            "Call effgen.models.hf_inference_models.refresh_models() to update.",
            REGISTRY_FETCH_DATE,
            len(added),
            len(removed),
            len(pricing_changed),
        )

    return report


def catalog_summary() -> dict[str, Any]:
    """Return a quick summary of the active catalog (counts + fetch date)."""
    return {
        "fetched_at": REGISTRY_FETCH_DATE,
        "source": CATALOG_SOURCE,
        "total_models": len(HF_MODELS),
        "tool_capable": len(tool_capable_models()),
        "structured_output": sum(
            1 for i in HF_MODELS.values() if i.get("supports_structured_output")
        ),
        "multimodal": sum(
            1 for i in HF_MODELS.values()
            if set(i.get("input_modalities") or []) - {"text"}
        ),
        "providers": sorted({
            p["provider"]
            for i in HF_MODELS.values()
            for p in (i.get("providers") or [])
        }),
    }
