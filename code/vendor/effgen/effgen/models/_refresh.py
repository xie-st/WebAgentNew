"""Live catalog refresh + drift detection for every provider.

The network-free :mod:`effgen.models._catalog` spine defines the normalized
:class:`~effgen.models._catalog.ModelRecord`, the snapshot store under
``models/_data/<provider>.json`` and the diff/staleness helpers.  This module
adds the one thing that genuinely needs the network: *asking the live provider
API what models exist today* and reconciling that against what effGen ships.

Two uniform entry points cover every provider:

* :func:`refresh_models` — fetch the live catalog for one provider, return the
  normalized records, and (optionally) persist them to the bundled snapshot so a
  later process sees the fresh list offline.
* :func:`check_drift` — fetch the live catalog and diff it against the persisted
  snapshot, returning ``{added, removed, changed, ...}`` and warning once when it
  has drifted.

Per-provider fetchers are intentionally small and defensive: each returns
``[(model_id, live_raw_dict), ...]`` from the provider's *list-models* endpoint.
For an id effGen already curates we carry the hand-verified fields (pricing,
context window, capabilities) and only stamp ``verified_on`` with today's date —
the live API confirmed the id exists.  For a genuinely new id we keep whatever
the live endpoint returned and mark ``price_source = "live-api"`` with pricing
left ``None`` rather than fabricating a number (cost data is curated separately).
"""

from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Callable
from typing import Any

from effgen.models._catalog import (
    PRICE_SOURCE_LIVE,
    ModelRecord,
    _guess_family,
    _load_models_dict,
    build_records,
    diff_records,
    is_chat_model,
    is_finetune,
    known_providers,
    load_snapshot_records,
    normalize_record,
    save_snapshot,
    warn_if_stale,
)

logger = logging.getLogger(__name__)

# Env var(s) each provider's key may live under (checked in order).
_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "cerebras": ("CEREBRAS_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "replicate": ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY"),
    "hf": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
}


def _resolve_key(provider: str, api_key: str | None) -> str | None:
    if api_key:
        return api_key
    for env in _KEY_ENVS.get(provider, ()):  # first present wins
        val = os.environ.get(env)
        if val:
            return val
    return None


def has_credentials(provider: str) -> bool:
    """True if a key/token for *provider* is available in the environment."""
    return _resolve_key(provider, None) is not None


# ---------------------------------------------------------------------------
# Per-provider live fetchers — each returns [(model_id, live_raw_dict), ...].
# They never raise for an empty result; they raise only on a real API/auth error
# so refresh_models() can report it accurately.
# ---------------------------------------------------------------------------


def _fetch_openai(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    from openai import OpenAI

    from effgen.models._base_url import openai_client_base_url, resolve_base_url

    # Explicit for the same reason the adapter is: the SDK would otherwise read
    # OPENAI_BASE_URL itself and send the catalog request to a blank address.
    client = OpenAI(api_key=api_key, base_url=openai_client_base_url(resolve_base_url()))
    out: list[tuple[str, dict[str, Any]]] = []
    for m in client.models.list():
        out.append((m.id, {"owned_by": getattr(m, "owned_by", None)}))
    return out


def _fetch_gemini(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    out: list[tuple[str, dict[str, Any]]] = []
    for m in client.models.list():
        mid = (m.name or "").replace("models/", "")
        if not mid:
            continue
        actions = set(getattr(m, "supported_actions", None) or [])
        raw: dict[str, Any] = {
            "display_name": getattr(m, "display_name", "") or mid,
            "context": getattr(m, "input_token_limit", None),
            "max_output": getattr(m, "output_token_limit", None),
        }
        if actions:
            raw["supports_native_tools"] = "generateContent" in actions
        out.append((mid, raw))
    return out


def _fetch_cerebras(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)
    return [(m.id, {}) for m in client.models.list().data]


def _fetch_groq(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    from groq import Groq

    client = Groq(api_key=api_key)
    out: list[tuple[str, dict[str, Any]]] = []
    for m in client.models.list().data:
        raw: dict[str, Any] = {}
        ctx = getattr(m, "context_window", None)
        if ctx:
            raw["context"] = ctx
        if getattr(m, "active", None) is False:
            raw["active"] = False
        out.append((m.id, raw))
    return out


def _fetch_anthropic(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    return [
        (m.id, {"display_name": getattr(m, "display_name", "") or m.id})
        for m in client.models.list()
    ]


def _fetch_together(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    from together import Together

    client = Together(api_key=api_key)
    out: list[tuple[str, dict[str, Any]]] = []
    for m in client.models.list():
        mid = getattr(m, "id", None)
        if not mid:
            continue
        raw: dict[str, Any] = {}
        ctx = getattr(m, "context_length", None)
        if ctx:
            raw["context"] = ctx
        # Together types every id (chat / language / image / audio / ...).  Carry
        # it so the chat filter can use the provider's own classification.
        mtype = getattr(m, "type", None)
        if mtype:
            raw["live_type"] = str(mtype)
        out.append((mid, raw))
    return out


def _fetch_fireworks(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    import requests

    resp = requests.get(
        "https://api.fireworks.ai/inference/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    out: list[tuple[str, dict[str, Any]]] = []
    for m in resp.json().get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        # accounts/fireworks/routers/* are routing aliases for models the same
        # listing already returns; the curated catalog tracks the models.
        if mid.startswith("accounts/fireworks/routers/"):
            continue
        raw: dict[str, Any] = {}
        ctx = m.get("context_length")
        if ctx:
            raw["context"] = ctx
        if m.get("supports_tools") is not None:
            raw["supports_native_tools"] = bool(m["supports_tools"])
        if m.get("supports_image_input") is not None:
            raw["supports_vision"] = bool(m["supports_image_input"])
        out.append((mid, raw))
    return out


def _fetch_replicate(api_key: str) -> list[tuple[str, dict[str, Any]]]:
    # Replicate's catalog is enormous and paginated; the curated set is what
    # effGen routes to.  Confirm those ids still resolve rather than walking the
    # whole index (which would be thousands of mostly-image models).
    import replicate

    client = replicate.Client(api_token=api_key)
    out: list[tuple[str, dict[str, Any]]] = []
    for mid in _load_models_dict("replicate"):
        owner, _, name = mid.partition("/")
        if not name:
            continue
        try:
            client.models.get(owner, name)
            out.append((mid, {}))
        except Exception:  # noqa: BLE001 - a missing model is the signal, not an error
            logger.debug("Replicate model %s no longer resolves", mid, exc_info=True)
    return out


def _fetch_hf(api_key: str | None) -> list[tuple[str, dict[str, Any]]]:
    # The HF router catalog already has a dedicated, well-tested refresh path.
    from effgen.models import hf_inference_models as hm

    live = hm.refresh_models(token=api_key, write_cache=False)
    return [(mid, dict(raw)) for mid, raw in live.items()]


_FETCHERS: dict[str, Callable[[str], list[tuple[str, dict[str, Any]]]]] = {
    "openai": _fetch_openai,
    "gemini": _fetch_gemini,
    "cerebras": _fetch_cerebras,
    "groq": _fetch_groq,
    "anthropic": _fetch_anthropic,
    "together": _fetch_together,
    "fireworks": _fetch_fireworks,
    "replicate": _fetch_replicate,
    "hf": _fetch_hf,
}


# pip extra that ships each provider's SDK (defaults to the provider name).
_PROVIDER_EXTRAS: dict[str, str] = {
    "openai": "openai",
    "gemini": "gemini",
    "cerebras": "cerebras",
    "groq": "groq",
    "anthropic": "anthropic",
    "together": "together",
    "fireworks": "fireworks",
    "replicate": "replicate",
    "hf": "hf",
}


def _call_fetcher(provider: str, key: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Run a provider fetcher, turning a missing SDK into an actionable hint.

    In a lean install the provider SDK may be absent; a raw
    ``ModuleNotFoundError: No module named 'cerebras'`` is accurate but unhelpful.
    Mirror the adapter path and point at the right extra to install.
    """
    try:
        return _FETCHERS[provider](key)  # type: ignore[arg-type]
    except (ImportError, ModuleNotFoundError) as exc:
        extra = _PROVIDER_EXTRAS.get(provider, provider)
        raise RuntimeError(
            f"Refreshing '{provider}' needs its SDK, which is not installed "
            f"({exc}). Install with: pip install 'effgen[{extra}]'"
        ) from exc


def refreshable_providers() -> list[str]:
    """Providers with a live refresh fetcher."""
    return [p for p in known_providers() if p in _FETCHERS]


# ---------------------------------------------------------------------------
# Record construction from a live id list
# ---------------------------------------------------------------------------


def _filter_chat_live(
    provider: str,
    live: list[tuple[str, dict[str, Any]]],
    bundled: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Drop non-chat ids from a live listing, keeping ids effGen already curates.

    A provider's list-models endpoint returns its whole surface (embeddings,
    speech, image/video, moderation, rerankers and — for OpenAI — the caller's
    private ``ft:`` fine-tunes).  Reconciling that raw against the curated chat
    catalog would balloon a snapshot and leak private fine-tune ids, and would
    perpetually report non-chat models as "added" drift.  Keep an id when it is
    already curated (so a deliberately-listed model — e.g. a transcription or
    image endpoint a catalog ships — never shows as spurious "removed") or when
    it classifies as a chat/text-generation model.  A private ``ft:`` id is
    never kept, even if somehow already present.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for mid, raw in live:
        if is_finetune(mid):
            continue
        # A provider that classifies its own ids wins over both the id heuristic
        # and the curated table: a model retyped from chat to language upstream
        # is no longer a chat model, however it is bundled here.
        live_type = (raw or {}).get("live_type")
        if live_type is not None:
            if live_type == "chat":
                out.append((mid, raw))
            continue
        if mid in bundled or is_chat_model(provider, mid):
            out.append((mid, raw))
    return out


def _records_from_live(
    provider: str,
    live: list[tuple[str, dict[str, Any]]],
    *,
    verified_on: str,
) -> list[ModelRecord]:
    """Turn a live ``[(id, raw)]`` list into normalized records.

    The live listing is first filtered to chat/text-generation models (see
    :func:`_filter_chat_live`).  Curated fields are carried for ids effGen
    already ships (so refresh never loses hand-verified pricing/capabilities);
    new ids get a best-effort record marked ``price_source = live-api``.
    """
    bundled = _load_models_dict(provider)
    live = _filter_chat_live(provider, live, bundled)
    records: list[ModelRecord] = []
    seen: set[str] = set()
    for mid, live_raw in live:
        if mid in seen:
            continue
        seen.add(mid)
        curated = bundled.get(mid)
        if curated is not None:
            # Known id confirmed live: keep curated data, fill gaps from live.
            merged = {**live_raw, **curated}
            rec = normalize_record(provider, mid, merged, verified_on=verified_on)
        else:
            rec = normalize_record(
                provider,
                mid,
                live_raw,
                verified_on=verified_on,
                price_source=PRICE_SOURCE_LIVE,
            )
            if not rec.family:
                rec.family = _guess_family(mid)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Uniform public API
# ---------------------------------------------------------------------------


def available_models_live(provider: str, api_key: str | None = None) -> list[str]:
    """Return the live model ids *provider* currently serves.

    Raises a clear error if *provider* has no fetcher or no credentials.
    """
    if provider not in _FETCHERS:
        raise KeyError(f"No live refresh available for provider {provider!r}.")
    key = _resolve_key(provider, api_key)
    if key is None and provider != "hf":  # HF router allows anonymous (rate-limited)
        raise PermissionError(
            f"No API key for '{provider}'. Set {_KEY_ENVS.get(provider, ('<KEY>',))[0]}."
        )
    bundled = _load_models_dict(provider)
    live = _filter_chat_live(provider, _call_fetcher(provider, key), bundled)
    return sorted(mid for mid, _ in live)


def refresh_models(
    provider: str,
    *,
    api_key: str | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    """Fetch the live catalog for *provider* and reconcile it against effGen.

    Args:
        provider: One of :func:`refreshable_providers`.
        api_key:  Override the environment key/token.
        persist:  When True, write the fresh records to the bundled snapshot
                  (``models/_data/<provider>.json``) with today's ``verified_on``.

    Returns:
        ``{provider, verified_on, live_count, records, diff}`` where ``diff`` is
        the :func:`~effgen.models._catalog.diff_records` of the previous snapshot
        vs the fresh records (added/removed/changed/price_changed).

    Raises:
        KeyError:        Unknown / non-refreshable provider.
        PermissionError: No credentials available.
    """
    if provider not in _FETCHERS:
        raise KeyError(
            f"No live refresh available for provider {provider!r}. "
            f"Refreshable: {refreshable_providers()}"
        )
    key = _resolve_key(provider, api_key)
    if key is None and provider != "hf":
        raise PermissionError(
            f"No API key for '{provider}'. Set {_KEY_ENVS.get(provider, ('<KEY>',))[0]} "
            f"or pass api_key=."
        )

    today = datetime.date.today().isoformat()
    live = _call_fetcher(provider, key)
    records = _records_from_live(provider, live, verified_on=today)

    previous = load_snapshot_records(provider)
    diff = diff_records(previous, records)

    if persist:
        save_snapshot(provider, records, verified_on=today, source=PRICE_SOURCE_LIVE)
        logger.info(
            "Refreshed %s catalog: %d live models (+%d / -%d vs previous snapshot)",
            provider,
            len(records),
            len(diff["added"]),
            len(diff["removed"]),
        )

    return {
        "provider": provider,
        "verified_on": today,
        "live_count": len(records),
        "records": records,
        "diff": diff,
    }


def check_drift(
    provider: str,
    *,
    api_key: str | None = None,
    warn: bool = True,
) -> dict[str, Any]:
    """Diff the live catalog for *provider* against its bundled snapshot.

    Unlike :func:`refresh_models` this never writes anything.  When *warn* is
    True and drift is detected it emits one non-spammy warning suggesting
    ``effgen models refresh``.

    Returns the :func:`~effgen.models._catalog.diff_records` report plus
    ``provider`` and ``checked_on``.  Falls back to the offline catalog-vs-snapshot
    check when no credentials are available.

    Args:
        provider: The provider to check.
        api_key: Credential for the live listing, otherwise read from the
            environment.
        warn: Whether to emit the one-time refresh suggestion on drift.
    """
    today = datetime.date.today().isoformat()
    snapshot = load_snapshot_records(provider)

    key = _resolve_key(provider, api_key)
    if (key is None and provider != "hf") or provider not in _FETCHERS:
        # No way to ask the live API — fall back to comparing the in-package
        # catalog against the snapshot (purely offline self-check).
        live_records = build_records(provider)
        source = "offline-catalog"
    else:
        live = _call_fetcher(provider, key)
        live_records = _records_from_live(provider, live, verified_on=today)
        source = "live-api"

    diff = diff_records(snapshot, live_records)
    drifted = bool(diff["added"] or diff["removed"] or diff["changed"])
    if warn and drifted:
        warn_if_stale(provider)
        logger.warning(
            "Model catalog for '%s' has drifted (%s): +%d / -%d / ~%d changed. "
            "Run `effgen models refresh --provider %s` to update.",
            provider,
            source,
            len(diff["added"]),
            len(diff["removed"]),
            len(diff["changed"]),
            provider,
        )

    diff["provider"] = provider
    diff["checked_on"] = today
    diff["source"] = source
    diff["drifted"] = drifted
    return diff


__all__ = [
    "has_credentials",
    "refreshable_providers",
    "available_models_live",
    "refresh_models",
    "check_drift",
]
