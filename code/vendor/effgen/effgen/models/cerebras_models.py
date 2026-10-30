"""
Cerebras model registry for effGen.

Rate limits and context/max-output values fetched from:
  https://inference-docs.cerebras.ai/support/rate-limits.md
  https://inference-docs.cerebras.ai/models/<model>.md
Fetch date: 2026-04-24

Free-tier limits apply per API key.  All limits are sliding-window (token
bucketing).  Whichever metric (RPM/RPH/RPD or TPM/TPH/TPD) is hit first
triggers throttling.

Context/max-output values differ between free and paid tiers.  The default
``context`` and ``max_output`` exposed by ``model_info()`` / ``get_context_length()``
reflect the **free-tier** values, since users on the paid tier can override
via the ``context`` / ``max_output`` kwargs of :class:`CerebrasAdapter` if
needed.  ``context_paid`` and ``max_output_paid`` are included for reference.

Catalog scope:
  The live Cerebras inference API serves exactly two chat models —
  ``gpt-oss-120b`` and ``zai-glm-4.7`` (verified against
  ``client.models.list()``).  The earlier ``llama3.1-8b`` and
  ``qwen-3-235b-a22b-instruct-2507`` reached end-of-life on 2026-05-27 and now
  return 404 ``model_not_found``; they have been removed so the catalog reflects
  what is actually callable.  Run ``effgen models refresh --provider cerebras``
  to re-sync from the live endpoint.

Access notes:
  - Both models are reasoning-style models: a response's text lands in
    ``message.content`` but reasoning tokens are billed and can dominate a low
    ``max_completion_tokens`` budget, so give them generous output room.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry has:
#   context         — context window (free tier) in tokens
#   context_paid    — context window (paid tier) in tokens
#   max_output      — max completion tokens (free tier)
#   max_output_paid — max completion tokens (paid tier)
#   rpm/rph/rpd     — requests per minute / hour / day  (free tier)
#   tpm/tph/tpd     — tokens   per minute / hour / day  (free tier)
#   free_tier       — True if reliably callable on the free tier today
#   deprecated      — ISO date if scheduled for deprecation, else None
#   family          — model family label
#   reasoning       — True if the model spends output budget on a hidden
#                     reasoning chain before any visible token
# ---------------------------------------------------------------------------
CEREBRAS_MODELS: dict[str, dict] = {
    "gpt-oss-120b": {
        "family": "gpt-oss",
        "context": 65_536,          # free tier: 65k
        "context_paid": 131_072,    # paid tier: 131k
        "max_output": 32_768,       # free tier: 32k
        "max_output_paid": 40_960,  # paid tier: 40k
        "rpm": 30,
        "rph": 900,
        "rpd": 14_400,
        "tpm": 64_000,
        "tph": 1_000_000,
        "tpd": 1_000_000,
        # Verified callable via client.models.list() + a live generation
        # (gpt-oss-120b returned "4" for 2+2 on 2026-06-08).
        "free_tier": True,
        "deprecated": None,
        # Supports OpenAI-compatible function calling (empirically verified 2026-04-24)
        "supports_native_tools": True,
        "reasoning": True,
    },
    "zai-glm-4.7": {
        "family": "zai-glm",
        "context": 65_536,          # free tier: 64k (rounded to 65536)
        "context_paid": 131_072,    # paid tier: 131k
        "max_output": 40_960,       # free tier: 40k
        "max_output_paid": 40_960,
        # Stricter per-minute/hour/day request limits on free tier
        "rpm": 10,
        "rph": 100,
        "rpd": 100,
        "tpm": 60_000,
        "tph": 1_000_000,
        "tpd": 1_000_000,
        # Verified callable via client.models.list() on 2026-06-08.
        "free_tier": True,
        "deprecated": None,
        # GLM-4.7: tool calling not reliably surfaced; marked False so the agent
        # falls back to ReAct.  Paid-tier users can override.
        "supports_native_tools": False,
        "reasoning": True,
    },
}

# Default model for live testing — one of the two models the live API serves.
CEREBRAS_DEFAULT_MODEL = "gpt-oss-120b"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return list of all registered Cerebras model IDs.

    Example::

        >>> from effgen.models.cerebras_models import available_models
        >>> available_models()
        ['gpt-oss-120b', 'zai-glm-4.7']
    """
    return list(CEREBRAS_MODELS.keys())


def free_tier_models() -> list[str]:
    """Return models reliably callable on the Cerebras free tier today."""
    return [mid for mid, info in CEREBRAS_MODELS.items() if info.get("free_tier", False)]


def deprecated_models() -> dict[str, str]:
    """Return a mapping of deprecated model_id -> deprecation date (ISO)."""
    return {
        mid: info["deprecated"]
        for mid, info in CEREBRAS_MODELS.items()
        if info.get("deprecated")
    }


def model_info(model_id: str) -> dict:
    """Return rate-limit and context metadata for *model_id*.

    Args:
        model_id: A key from :data:`CEREBRAS_MODELS`.

    Returns:
        A dict with keys: ``family``, ``context``, ``context_paid``,
        ``max_output``, ``max_output_paid``, ``rpm``, ``rph``, ``rpd``,
        ``tpm``, ``tph``, ``tpd``, ``free_tier``, ``deprecated``.

    Raises:
        KeyError: If *model_id* is not registered.

    Example::

        >>> from effgen.models.cerebras_models import model_info
        >>> info = model_info("gpt-oss-120b")
        >>> info["rpm"]
        30
    """
    if model_id not in CEREBRAS_MODELS:
        raise KeyError(
            f"Unknown Cerebras model '{model_id}'. "
            f"Available: {available_models()}"
        )
    return dict(CEREBRAS_MODELS[model_id])
