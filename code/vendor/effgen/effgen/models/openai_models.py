"""
OpenAI model registry for effGen.

Context windows and max output tokens verified against:
  https://platform.openai.com/docs/models
Fetch date: 2026-04-24

Pricing verified against:
  https://platform.openai.com/docs/pricing
Fetch date: 2026-04-24

Families:
  chat      — GPT-4o / GPT-4.1 / GPT-5 / GPT-5.4 series
  reasoning — o1, o1-mini, o3, o3-mini, o4-mini

Notes:
  - Reasoning models use ``reasoning_effort`` as a top-level API parameter.
  - ``supports_prompt_caching`` means the model exposes
    ``usage.prompt_tokens_details.cached_tokens`` on responses.
  - Prices are USD per 1M tokens (input / cached_input / output).
  - None means "not publicly listed" for that price point.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry has:
#   family               — "chat" or "reasoning"
#   context              — context window in tokens
#   max_output           — maximum output tokens
#   supports_reasoning   — True if model accepts reasoning_effort param
#   supports_native_tools — True if model supports OpenAI function-calling
#   supports_prompt_caching — True if cached_tokens is surfaced in usage
#   input_price_per_1m   — USD per 1M input tokens (None = not listed)
#   cached_input_price_per_1m — USD per 1M cached input tokens (None = not listed)
#   output_price_per_1m  — USD per 1M output tokens (None = not listed)
# ---------------------------------------------------------------------------
OPENAI_MODELS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # GPT-5.4 series (as of 2026-04-24)
    # ------------------------------------------------------------------
    "gpt-5.4": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 2.50,
        "cached_input_price_per_1m": 0.25,
        "output_price_per_1m": 15.00,
    },
    "gpt-5.4-mini": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.75,
        "cached_input_price_per_1m": 0.075,
        "output_price_per_1m": 4.50,
    },
    "gpt-5.4-nano": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.20,
        "cached_input_price_per_1m": 0.02,
        "output_price_per_1m": 1.25,
    },
    "gpt-5.4-pro": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 30.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 180.00,
    },
    # ------------------------------------------------------------------
    # GPT-5 series
    # ------------------------------------------------------------------
    "gpt-5": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 1.25,
        "cached_input_price_per_1m": 0.125,
        "output_price_per_1m": 10.00,
    },
    "gpt-5-mini": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.25,
        "cached_input_price_per_1m": 0.025,
        "output_price_per_1m": 2.00,
    },
    "gpt-5-nano": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.05,
        "cached_input_price_per_1m": 0.005,
        "output_price_per_1m": 0.40,
    },
    "gpt-5-pro": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 15.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 120.00,
    },
    "gpt-5.2": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 1.75,
        "cached_input_price_per_1m": 0.175,
        "output_price_per_1m": 14.00,
    },
    "gpt-5.1": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 1.25,
        "cached_input_price_per_1m": 0.125,
        "output_price_per_1m": 10.00,
    },
    # ------------------------------------------------------------------
    # GPT-4.1 series
    # ------------------------------------------------------------------
    "gpt-4.1": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 2.00,
        "cached_input_price_per_1m": 0.50,
        "output_price_per_1m": 8.00,
    },
    "gpt-4.1-mini": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.40,
        "cached_input_price_per_1m": 0.10,
        "output_price_per_1m": 1.60,
    },
    "gpt-4.1-nano": {
        "family": "chat",
        "context": 1_047_576,
        "max_output": 32_768,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.10,
        "cached_input_price_per_1m": 0.025,
        "output_price_per_1m": 0.40,
    },
    # ------------------------------------------------------------------
    # GPT-4o series
    # ------------------------------------------------------------------
    "gpt-4o": {
        "family": "chat",
        "context": 128_000,
        "max_output": 16_384,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 2.50,
        "cached_input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
    },
    "gpt-4o-mini": {
        "family": "chat",
        "context": 128_000,
        "max_output": 16_384,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 0.15,
        "cached_input_price_per_1m": 0.075,
        "output_price_per_1m": 0.60,
    },
    "gpt-4o-2024-11-20": {
        "family": "chat",
        "context": 128_000,
        "max_output": 16_384,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 2.50,
        "cached_input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
    },
    "gpt-4o-2024-05-13": {
        "family": "chat",
        "context": 128_000,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 5.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 15.00,
    },
    # ------------------------------------------------------------------
    # Legacy GPT-4 (kept for back-compat)
    # ------------------------------------------------------------------
    "gpt-4-turbo": {
        "family": "chat",
        "context": 128_000,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 10.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 30.00,
    },
    "gpt-4-turbo-2024-04-09": {
        "family": "chat",
        "context": 128_000,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 10.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 30.00,
    },
    "gpt-4": {
        "family": "chat",
        "context": 8_192,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 30.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 60.00,
    },
    "gpt-4-0613": {
        "family": "chat",
        "context": 8_192,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 30.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 60.00,
    },
    # ------------------------------------------------------------------
    # Legacy GPT-3.5 (kept for back-compat)
    # ------------------------------------------------------------------
    "gpt-3.5-turbo": {
        "family": "chat",
        "context": 16_385,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 0.50,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 1.50,
    },
    "gpt-3.5-turbo-0125": {
        "family": "chat",
        "context": 16_385,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 0.50,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 1.50,
    },
    "gpt-3.5-turbo-16k": {
        "family": "chat",
        "context": 16_385,
        "max_output": 4_096,
        "supports_reasoning": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 3.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 4.00,
    },
    # ------------------------------------------------------------------
    # o1 series (reasoning)
    # ------------------------------------------------------------------
    "o1": {
        "family": "reasoning",
        "context": 200_000,
        "max_output": 100_000,
        "supports_reasoning": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 15.00,
        "cached_input_price_per_1m": 7.50,
        "output_price_per_1m": 60.00,
    },
    "o1-pro": {
        "family": "reasoning",
        "context": 200_000,
        "max_output": 100_000,
        "supports_reasoning": True,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 150.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 600.00,
    },
    # ------------------------------------------------------------------
    # o3 series (reasoning)
    # ------------------------------------------------------------------
    "o3": {
        "family": "reasoning",
        "context": 200_000,
        "max_output": 100_000,
        "supports_reasoning": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 2.00,
        "cached_input_price_per_1m": 0.50,
        "output_price_per_1m": 8.00,
    },
    "o3-mini": {
        "family": "reasoning",
        "context": 200_000,
        "max_output": 100_000,
        "supports_reasoning": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 1.10,
        "cached_input_price_per_1m": 0.55,
        "output_price_per_1m": 4.40,
    },
    "o3-pro": {
        "family": "reasoning",
        "context": 200_000,
        "max_output": 100_000,
        "supports_reasoning": True,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "input_price_per_1m": 20.00,
        "cached_input_price_per_1m": None,
        "output_price_per_1m": 80.00,
    },
    # ------------------------------------------------------------------
    # o4-mini (reasoning)
    # ------------------------------------------------------------------
    "o4-mini": {
        "family": "reasoning",
        "context": 200_000,
        "max_output": 100_000,
        "supports_reasoning": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "input_price_per_1m": 1.10,
        "cached_input_price_per_1m": 0.275,
        "output_price_per_1m": 4.40,
    },
}

# Default model for examples and tests (cheapest chat model)
OPENAI_DEFAULT_MODEL = "gpt-5.4-nano"

# Valid reasoning effort values (per OpenAI SDK ReasoningEffort type, 2026-04-24)
# Note: 'minimal' is only supported on o1-series; o3/o4 require at least 'low'
VALID_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    """Return all registered OpenAI model IDs."""
    return list(OPENAI_MODELS.keys())


def chat_models() -> list[str]:
    """Return models in the 'chat' family."""
    return [mid for mid, info in OPENAI_MODELS.items() if info["family"] == "chat"]


def reasoning_models() -> list[str]:
    """Return models in the 'reasoning' family."""
    return [mid for mid, info in OPENAI_MODELS.items() if info["family"] == "reasoning"]


def model_info(model_id: str) -> dict:
    """Return metadata dict for *model_id*.

    Raises:
        KeyError: If model_id is not in the registry.
    """
    if model_id not in OPENAI_MODELS:
        raise KeyError(
            f"Unknown OpenAI model '{model_id}'. "
            f"Available: {available_models()}"
        )
    return dict(OPENAI_MODELS[model_id])


def supports_reasoning(model_id: str) -> bool:
    """Return True if *model_id* is a reasoning model."""
    return OPENAI_MODELS.get(model_id, {}).get("supports_reasoning", False)


def get_context_length(model_id: str) -> int:
    """Return context window for *model_id*, defaulting to 128k for unknown models."""
    return OPENAI_MODELS.get(model_id, {}).get("context", 128_000)


def get_max_output(model_id: str) -> int:
    """Return max output tokens for *model_id*."""
    info = OPENAI_MODELS.get(model_id)
    if info:
        return info["max_output"]
    # Fallback guess based on name
    if any(tag in model_id for tag in ("o1", "o3", "o4")):
        return 65_536
    return 16_384


_VISION_MODEL_PREFIXES = (
    "gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4",
)
_NON_VISION_SUFFIXES = ("-audio",)


def supports_vision(model_id: str) -> bool:
    """Return True if *model_id* is known to accept image inputs.

    GPT-4o and GPT-4.1 families accept images.  Reasoning models (o1/o3/o4)
    accept images on supported variants.  GPT-3.5 and text-only variants do not.
    """
    if any(model_id.endswith(s) for s in _NON_VISION_SUFFIXES):
        return False
    return any(model_id.startswith(p) or p in model_id for p in _VISION_MODEL_PREFIXES)


def get_pricing(model_id: str) -> tuple[float | None, float | None, float | None]:
    """Return (input_price_per_1m, cached_input_price_per_1m, output_price_per_1m)."""
    info = OPENAI_MODELS.get(model_id, {})
    return (
        info.get("input_price_per_1m"),
        info.get("cached_input_price_per_1m"),
        info.get("output_price_per_1m"),
    )
