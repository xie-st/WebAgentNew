"""
Together AI model registry for effGen.

Chat-model catalog last verified against the Together API on 2026-08-07 (168
chat models; the language and embedding registries are bundled as well). The
registry tables themselves live in :mod:`effgen.models.together_models_data`
and are re-exported here, so ``effgen.models.together_models`` remains the
import path for every public name.

Pricing from Together AI pricing page (https://www.together.ai/pricing).
Rate limits: Together AI applies per-account limits; serverless tier ~100 RPM typical.

serverless=True  — accessible via standard API without a dedicated endpoint.
serverless=False — requires a dedicated endpoint started in Together console.

Tool-calling support verified by live API tests.
Together uses an OpenAI-compatible function-calling API.

The snapshot date is exposed as :data:`REGISTRY_FETCH_DATE`.
Use refresh_models() to fetch the live catalog and detect drift.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from effgen.models.together_models_data import (  # noqa: F401  re-exported for import/patch parity
    REGISTRY_FETCH_DATE,
    TOGETHER_EMBEDDING_MODELS,
    TOGETHER_LANGUAGE_MODELS,
    TOGETHER_MODELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults and convenience helpers
# ---------------------------------------------------------------------------

# Default model: cheapest confirmed-serverless model with tool support. It is a
# reasoning model, so its catalog entry is flagged "reasoning": True — that is
# what earns it the larger default output budget; on a small max_tokens it
# would otherwise spend the whole allowance thinking and return empty text.
TOGETHER_DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

# Serverless-accessible (no dedicated endpoint needed)
TOGETHER_SERVERLESS_MODELS = {
    k: v for k, v in TOGETHER_MODELS.items() if v.get("serverless", False)
}

# Chat-capable models (all in TOGETHER_MODELS)
TOGETHER_CHAT_MODELS = TOGETHER_MODELS


def available_models() -> list[str]:
    """Return all registered Together chat model IDs."""
    return list(TOGETHER_MODELS.keys())


def serverless_models() -> list[str]:
    """Return chat model IDs accessible without a dedicated endpoint."""
    return list(TOGETHER_SERVERLESS_MODELS.keys())


def chat_models() -> list[str]:
    """Return model IDs usable via chat completions."""
    return list(TOGETHER_CHAT_MODELS.keys())


def tool_capable_models() -> list[str]:
    """Return chat models that support native function calling."""
    return [k for k, v in TOGETHER_CHAT_MODELS.items() if v.get("supports_native_tools")]


def model_info(model_name: str) -> dict:
    """Return registry entry for *model_name*, or raise KeyError."""
    if model_name in TOGETHER_MODELS:
        return TOGETHER_MODELS[model_name]
    if model_name in TOGETHER_LANGUAGE_MODELS:
        return TOGETHER_LANGUAGE_MODELS[model_name]
    raise KeyError(
        f"Unknown Together model {model_name!r}. "
        f"Available chat models: {available_models()}"
    )


def pricing_table() -> list[dict]:
    """Return pricing details for all registered chat models, sorted by input price."""
    rows = []
    for mid, info in TOGETHER_MODELS.items():
        rows.append({
            "model_id": mid,
            "organization": info.get("organization", ""),
            "family": info.get("family", ""),
            "context_length": info.get("context", 0),
            "input_per_1m_usd": info.get("pricing_per_1m_input", 0),
            "output_per_1m_usd": info.get("pricing_per_1m_output", 0),
            "supports_tools": info.get("supports_native_tools", False),
            "serverless": info.get("serverless", False),
        })
    rows.sort(key=lambda r: r["input_per_1m_usd"])
    return rows


def refresh_models(api_key: str | None = None) -> dict[str, Any]:
    """Fetch the live Together AI model catalog and compare against the bundled registry.

    Prints a drift summary so users know if the bundled registry is stale.
    Does NOT mutate :data:`TOGETHER_MODELS` — the bundled registry is the
    offline fallback.  Returns a dict with the comparison results.

    Args:
        api_key: Together API key. Falls back to ``TOGETHER_API_KEY`` env var.

    Returns:
        Dict with keys:
            ``live_count``   — total models returned by the live API.
            ``new_models``   — list of chat model IDs present live but not in bundle.
            ``removed_models`` — list of chat model IDs in bundle but not live.
            ``pricing_changes`` — list of dicts describing pricing drift.
            ``registry_fetch_date`` — date of the bundled snapshot.

    Example::

        from effgen.models.together_models import refresh_models
        drift = refresh_models()
        if drift["new_models"]:
            print("New models available:", drift["new_models"])
    """
    key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise ValueError(
            "Together API key not found. Set TOGETHER_API_KEY or pass api_key=."
        )

    try:
        from together import Together
    except ImportError as exc:
        raise RuntimeError(
            "together SDK not installed. Install with: pip install 'effgen[together]'"
        ) from exc

    client = Together(api_key=key)
    live_all = list(client.models.list())
    live_chat = {m.id: m for m in live_all if getattr(m, "type", "") == "chat"}

    bundled_ids = set(TOGETHER_MODELS.keys())
    live_ids = set(live_chat.keys())

    new_models = sorted(live_ids - bundled_ids)
    removed_models = sorted(bundled_ids - live_ids)

    pricing_changes: list[dict] = []
    for mid in bundled_ids & live_ids:
        lm = live_chat[mid]
        lp = getattr(lm, "pricing", None)
        if lp is None:
            continue
        bundled_in = TOGETHER_MODELS[mid].get("pricing_per_1m_input", 0)
        bundled_out = TOGETHER_MODELS[mid].get("pricing_per_1m_output", 0)
        live_in = getattr(lp, "input", 0) or 0
        live_out = getattr(lp, "output", 0) or 0
        if abs(bundled_in - live_in) > 0.0001 or abs(bundled_out - live_out) > 0.0001:
            pricing_changes.append({
                "model_id": mid,
                "bundled_input": bundled_in,
                "live_input": live_in,
                "bundled_output": bundled_out,
                "live_output": live_out,
            })

    # Print human-readable drift summary
    total_live = len(live_all)
    if new_models or removed_models or pricing_changes:
        logger.warning(
            "Together registry may be outdated (snapshot: %s). "
            "new=%d, removed=%d, pricing_changes=%d",
            REGISTRY_FETCH_DATE, len(new_models), len(removed_models), len(pricing_changes),
        )
        print(
            f"\n[effgen] Together model registry drift detected "
            f"(bundled snapshot: {REGISTRY_FETCH_DATE}):"
        )
        if new_models:
            print(f"  + {len(new_models)} new model(s) not in bundled registry:")
            for m in new_models[:10]:
                lm = live_chat[m]
                lp = getattr(lm, "pricing", None)
                inp = getattr(lp, "input", 0) if lp else 0
                ctx = getattr(lm, "context_length", 0)
                print(f"      {m}  (ctx={ctx}, in=${inp:.4f}/1M)")
            if len(new_models) > 10:
                print(f"      ... and {len(new_models) - 10} more")
        if removed_models:
            print(f"  - {len(removed_models)} model(s) in registry but no longer live:")
            for m in removed_models[:10]:
                print(f"      {m}")
            if len(removed_models) > 10:
                print(f"      ... and {len(removed_models) - 10} more")
        if pricing_changes:
            print(f"  ~ {len(pricing_changes)} model(s) with pricing changes:")
            for pc in pricing_changes[:5]:
                print(
                    f"      {pc['model_id']}: "
                    f"input ${pc['bundled_input']:.4f} → ${pc['live_input']:.4f}/1M, "
                    f"output ${pc['bundled_output']:.4f} → ${pc['live_output']:.4f}/1M"
                )
        print(
            "  Run effgen.models.together_models.refresh_models() to see full diff.\n"
            "  The bundled registry is still used as the offline fallback.\n"
        )
    else:
        print(
            f"[effgen] Together registry is up-to-date "
            f"({len(bundled_ids)} chat models match live API, snapshot: {REGISTRY_FETCH_DATE})"
        )

    return {
        "live_count": total_live,
        "live_chat_count": len(live_chat),
        "bundled_chat_count": len(bundled_ids),
        "new_models": new_models,
        "removed_models": removed_models,
        "pricing_changes": pricing_changes,
        "registry_fetch_date": REGISTRY_FETCH_DATE,
    }

