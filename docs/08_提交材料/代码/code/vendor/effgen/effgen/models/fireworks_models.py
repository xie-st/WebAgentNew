"""
Fireworks AI model registry for effGen.

Catalog reconciled with the Fireworks serverless API on 2026-08-07. Only models
returned by the live List Models API with ``filter=supports_serverless=true``
are kept here, so the ``accounts/fireworks/routers/*`` aliases the inference
endpoint also serves are deliberately absent — listing them here would make
``refresh_models()`` report them as removed on every run. Use ``refresh_models()``
to detect drift between this snapshot and the live serverless catalog.

Context windows, tool support and image-input support come from that listing.
``max_output`` is a curated default: Fireworks does not publish a per-model
output cap. Pricing is from https://docs.fireworks.ai/serverless/pricing; an
entry with no pricing keys is reported as unpriced rather than as free.

Model IDs must be given in full format: ``accounts/fireworks/models/<id>``.

``reasoning: True`` marks a model that spends output budget on a hidden chain
before it writes anything visible; the adapter reads it into
``_is_reasoning_model``, which earns the larger first token budget from
``default_max_output_tokens()``. Every chat- and vision-modality entry here
carries it: each was called once on 2026-08-13 and every one returned a
``reasoning_content`` chain. The two embedding entries do not.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FETCH_DATE = "2026-08-07"
_FIREWORKS_PREFIX = "accounts/fireworks/models/"

# ---------------------------------------------------------------------------
# Main registry — keyed by SHORT model ID (without accounts/fireworks/models/)
# ---------------------------------------------------------------------------
_REGISTRY_SHORT: dict[str, dict] = {
    "deepseek-v4-flash": {
        "display_name": "DeepSeek-V4-Flash",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "deepseek-v4-flash-0731": {
        "display_name": "DeepSeek-V4-Flash-0731",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "deepseek-v4-pro": {
        "display_name": "DeepSeek-V4-Pro",
        "family": "deepseek",
        "organization": "DeepSeek",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 1.74,
        "pricing_per_1m_output": 3.48,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "glm-5p2": {
        "display_name": "GLM 5.2",
        "family": "glm",
        "organization": "Zhipu AI",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "gpt-oss-120b": {
        "display_name": "OpenAI gpt-oss-120b",
        "family": "gpt-oss",
        "organization": "OpenAI",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 0.60,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "gpt-oss-20b": {
        "display_name": "OpenAI gpt-oss-20b",
        "family": "gpt-oss",
        "organization": "OpenAI",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.07,
        "pricing_per_1m_output": 0.30,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "inkling": {
        "display_name": "Inkling",
        "family": "inkling",
        "organization": "Thinking Machines",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "vision",
        "reasoning": True,
    },
    "kimi-k2p6": {
        "display_name": "Kimi K2.6",
        "family": "kimi",
        "organization": "Moonshot AI",
        "context": 262_144,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "pricing_per_1m_input": 0.95,
        "pricing_per_1m_output": 4.00,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "vision",
        "reasoning": True,
    },
    "kimi-k2p7-code": {
        "display_name": "Kimi K2.7 Code",
        "family": "kimi",
        "organization": "Moonshot AI",
        "context": 262_144,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "vision",
        "reasoning": True,
    },
    "kimi-k3": {
        "display_name": "Kimi K3",
        "family": "kimi",
        "organization": "Moonshot AI",
        "context": 1_048_576,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "vision",
        "reasoning": True,
    },
    "minimax-m2p7": {
        "display_name": "MiniMax M2.7",
        "family": "minimax",
        "organization": "MiniMax",
        "context": 196_608,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.30,
        "pricing_per_1m_output": 1.20,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "minimax-m3": {
        "display_name": "MiniMax M3",
        "family": "minimax",
        "organization": "MiniMax",
        "context": 512_000,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "nemotron-3-ultra-nvfp4": {
        "display_name": "NVIDIA Nemotron 3 Ultra NVFP4",
        "family": "nemotron",
        "organization": "NVIDIA",
        "context": 262_144,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "chat",
        "reasoning": True,
    },
    "qwen3p7-plus": {
        "display_name": "Qwen3.7 Plus",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 0,
        # Fireworks returns no context length for this model.
        "context_unpublished": True,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "vision",
        "reasoning": True,
    },
    "qwen3-embedding-8b": {
        "display_name": "Qwen3 Embedding 8B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 40_960,
        "max_output": 0,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.00,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "embedding",
    },
    "qwen3-reranker-8b": {
        "display_name": "Qwen3 Reranker 8B",
        "family": "qwen3",
        "organization": "Alibaba",
        "context": 40_960,
        "max_output": 0,
        "supports_native_tools": False,
        "supports_streaming": True,
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.00,
        "rpm": 10,
        "tpm": 40_000,
        "modality": "embedding",
    },
}

# Build the full registry with accounts/fireworks/models/ prefix
FIREWORKS_MODELS: dict[str, dict] = {
    f"{_FIREWORKS_PREFIX}{k}": v for k, v in _REGISTRY_SHORT.items()
}

FIREWORKS_DEFAULT_MODEL = f"{_FIREWORKS_PREFIX}gpt-oss-120b"

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return all model IDs in the registry (full path format)."""
    return sorted(FIREWORKS_MODELS.keys())


def chat_models() -> list[str]:
    """Return IDs of models usable via chat completions.

    Selection is by modality, so a chat model whose context window Fireworks does
    not publish (recorded as 0) is still listed.
    """
    return [
        mid for mid, info in FIREWORKS_MODELS.items()
        if info.get("modality", "chat") in ("chat", "vision")
    ]


def tool_capable_models() -> list[str]:
    """Return IDs of models that support native function calling."""
    return [
        mid for mid, info in FIREWORKS_MODELS.items()
        if info.get("supports_native_tools", False)
        and info.get("modality", "chat") in ("chat", "vision")
    ]


def streaming_models() -> list[str]:
    """Return IDs of models that support streaming."""
    return [
        mid for mid, info in FIREWORKS_MODELS.items()
        if info.get("supports_streaming", True)
    ]


def pricing_table() -> list[dict]:
    """Return a list of dicts with model pricing info, sorted by input cost."""
    rows = []
    for mid, info in FIREWORKS_MODELS.items():
        rows.append({
            "model": mid,
            "display_name": info.get("display_name", ""),
            "family": info.get("family", ""),
            "context": info.get("context", 0),
            "input_per_1m_usd": info.get("pricing_per_1m_input", 0),
            "output_per_1m_usd": info.get("pricing_per_1m_output", 0),
            "supports_tools": info.get("supports_native_tools", False),
        })
    return sorted(rows, key=lambda r: r["input_per_1m_usd"])


# ---------------------------------------------------------------------------
# Live catalog refresh + drift detection
# ---------------------------------------------------------------------------

def refresh_models(
    api_key: str | None = None,
    *,
    warn_on_drift: bool = True,
) -> dict[str, Any]:
    """Fetch the live Fireworks model catalog and compare against the bundled registry.

    Uses the ``/v1/accounts/fireworks/models`` API endpoint with Fireworks'
    documented ``filter=supports_serverless=true`` query (no cost, no tokens).

    Args:
        api_key: Fireworks API key. Reads ``FIREWORKS_API_KEY`` env var if omitted.
        warn_on_drift: Emit :mod:`logging` warnings when the live catalog differs
            from the bundled snapshot.

    Returns:
        A summary dict::

            {
                "fetch_date": "2026-05-15",
                "live_total": int,
                "bundled_total": int,
                "new_models": [str, ...],      # in live but not in bundled
                "removed_models": [str, ...],  # in bundled but not in live
                "live_models": {<id>: {...}},  # full live catalog
            }
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for refresh_models(). "
            "Install with: pip install requests"
        ) from exc

    key = api_key or os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise ValueError("FIREWORKS_API_KEY not set; cannot refresh model catalog.")

    # Paginate through all models
    live_raw: list[dict] = []
    page_token: str | None = None
    while True:
        url = (
            "https://api.fireworks.ai/v1/accounts/fireworks/models?"
            "pageSize=200&filter=supports_serverless%3Dtrue"
        )
        if page_token:
            url += f"&pageToken={page_token}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(
                f"Fireworks catalog fetch failed: HTTP {resp.status_code} — {resp.text[:200]}"
            )
        data = resp.json()
        live_raw.extend(data.get("models", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    # Build live dict for active Fireworks models. Image models can have
    # contextLength=0, so keep them in drift detection instead of reporting
    # bundled image models as removed.
    live_models: dict[str, dict] = {}
    for m in live_raw:
        if m.get("deprecationDate"):
            continue
        name = m.get("name", "")
        if not name.startswith(_FIREWORKS_PREFIX):
            continue
        ctx = m.get("contextLength", 0)
        live_models[name] = {
            "display_name": m.get("displayName", ""),
            "context": ctx,
            "model_type": (m.get("baseModelDetails") or {}).get("modelType", ""),
            "supports_serverless": m.get("supportsServerless", False),
            "supports_tools": m.get("supportsTools", False),
            "kind": m.get("kind", ""),
            "status": (m.get("status") or {}).get("code", ""),
        }

    bundled_ids = set(FIREWORKS_MODELS.keys())
    live_ids = set(live_models.keys())

    new_models = sorted(live_ids - bundled_ids)
    removed_models = sorted(bundled_ids - live_ids)

    if warn_on_drift:
        if new_models:
            logger.warning(
                "Fireworks catalog drift: %d new model(s) not in bundled registry:\n  %s\n"
                "Consider upgrading effGen or calling refresh_models() to get the latest.",
                len(new_models), "\n  ".join(new_models),
            )
        if removed_models:
            logger.warning(
                "Fireworks catalog drift: %d model(s) in bundled registry no longer listed:\n  %s",
                len(removed_models), "\n  ".join(removed_models),
            )

    return {
        "fetch_date": REGISTRY_FETCH_DATE,
        "live_total": len(live_models),
        "bundled_total": len(FIREWORKS_MODELS),
        "new_models": new_models,
        "removed_models": removed_models,
        "live_models": live_models,
    }
