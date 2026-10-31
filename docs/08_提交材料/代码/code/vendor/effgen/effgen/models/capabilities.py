"""
Model capability registry for effGen framework.

Maps model names to capability scores across dimensions like math, code,
reasoning, tool calling, and multilingual support. Used by the ModelRouter
to select the optimal model for a given task.

Pre-populated for common SLMs (Qwen, Llama, Phi, Mistral, Gemma).
Users can register custom capability profiles at runtime.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Capability(Enum):
    """Provider-level feature flags used by the policy-based ModelRouter.

    These describe what a cloud provider/model supports — distinct from the
    per-model quality scores (math, code, reasoning) in ModelCapability below.
    """
    chat = "chat"
    tools = "tools"
    streaming = "streaming"
    vision = "vision"
    audio_input = "audio_input"
    video_input = "video_input"
    grounding = "grounding"
    thinking = "thinking"
    json_schema = "json_schema"


# Standard capability dimensions
CAPABILITY_DIMENSIONS = [
    "math",
    "code",
    "reasoning",
    "tool_calling",
    "multilingual",
    "creative",
    "instruction_following",
]


@dataclass
class ModelCapability:
    """Capability profile for a single model.

    Attributes:
        model_name: Full model identifier (e.g. "Qwen/Qwen2.5-3B-Instruct").
        scores: Dict mapping capability dimension to score (0.0-1.0).
        size_billions: Approximate parameter count in billions (e.g. 3.0).
        context_length: Maximum context length the model supports.
        gpu_memory_gb: Approximate GPU memory required in GB (FP16).
        metadata: Arbitrary extra info (quantization support, etc.).
    """

    model_name: str
    scores: dict[str, float] = field(default_factory=dict)
    size_billions: float = 0.0
    context_length: int = 4096
    gpu_memory_gb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_score(self, dimension: str) -> float:
        """Get capability score for a dimension, defaulting to 0.5."""
        return self.scores.get(dimension, 0.5)

    def overall_score(self) -> float:
        """Compute average capability score across all dimensions."""
        if not self.scores:
            return 0.5
        return sum(self.scores.values()) / len(self.scores)


# ---------------------------------------------------------------------------
# Pre-populated capability profiles for common SLMs
# ---------------------------------------------------------------------------
# Scores are heuristic estimates (0.0 = poor, 1.0 = excellent) calibrated
# relative to other models in the same weight class. They are intentionally
# conservative so that the router does not over-promise.

MODEL_CAPABILITIES: dict[str, ModelCapability] = {}


def _register_defaults() -> None:
    """Populate the registry with known SLM profiles."""

    defaults: list[dict[str, Any]] = [
        # ---- Qwen 2.5 family ----
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "scores": {"math": 0.5, "code": 0.35, "reasoning": 0.4,
                       "tool_calling": 0.6, "multilingual": 0.5,
                       "creative": 0.3, "instruction_following": 0.55},
            "size_billions": 0.5, "context_length": 32768, "gpu_memory_gb": 1.2,
        },
        {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "scores": {"math": 0.65, "code": 0.55, "reasoning": 0.55,
                       "tool_calling": 0.75, "multilingual": 0.6,
                       "creative": 0.45, "instruction_following": 0.65},
            "size_billions": 1.5, "context_length": 32768, "gpu_memory_gb": 3.5,
        },
        {
            "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "scores": {"math": 0.8, "code": 0.7, "reasoning": 0.7,
                       "tool_calling": 0.9, "multilingual": 0.75,
                       "creative": 0.6, "instruction_following": 0.8},
            "size_billions": 3.0, "context_length": 32768, "gpu_memory_gb": 7.0,
        },
        {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "scores": {"math": 0.9, "code": 0.85, "reasoning": 0.85,
                       "tool_calling": 0.95, "multilingual": 0.85,
                       "creative": 0.75, "instruction_following": 0.9},
            "size_billions": 7.0, "context_length": 131072, "gpu_memory_gb": 15.0,
        },
        # ---- Llama 3.2 family ----
        {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "scores": {"math": 0.5, "code": 0.45, "reasoning": 0.45,
                       "tool_calling": 0.6, "multilingual": 0.4,
                       "creative": 0.4, "instruction_following": 0.55},
            "size_billions": 1.0, "context_length": 131072, "gpu_memory_gb": 2.5,
        },
        {
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "scores": {"math": 0.65, "code": 0.6, "reasoning": 0.6,
                       "tool_calling": 0.75, "multilingual": 0.55,
                       "creative": 0.55, "instruction_following": 0.7},
            "size_billions": 3.0, "context_length": 131072, "gpu_memory_gb": 7.0,
        },
        # ---- Phi family ----
        {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "scores": {"math": 0.8, "code": 0.75, "reasoning": 0.75,
                       "tool_calling": 0.7, "multilingual": 0.5,
                       "creative": 0.55, "instruction_following": 0.75},
            "size_billions": 3.8, "context_length": 4096, "gpu_memory_gb": 8.0,
        },
        {
            "model_name": "microsoft/Phi-3.5-mini-instruct",
            "scores": {"math": 0.82, "code": 0.78, "reasoning": 0.78,
                       "tool_calling": 0.75, "multilingual": 0.6,
                       "creative": 0.6, "instruction_following": 0.8},
            "size_billions": 3.8, "context_length": 128000, "gpu_memory_gb": 8.0,
        },
        {
            "model_name": "microsoft/phi-4",
            "scores": {"math": 0.9, "code": 0.85, "reasoning": 0.88,
                       "tool_calling": 0.8, "multilingual": 0.65,
                       "creative": 0.7, "instruction_following": 0.88},
            "size_billions": 14.0, "context_length": 16384, "gpu_memory_gb": 28.0,
        },
        # ---- Mistral family ----
        {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "scores": {"math": 0.75, "code": 0.7, "reasoning": 0.75,
                       "tool_calling": 0.8, "multilingual": 0.7,
                       "creative": 0.7, "instruction_following": 0.8},
            "size_billions": 7.0, "context_length": 32768, "gpu_memory_gb": 15.0,
        },
        # ---- Gemma family ----
        {
            "model_name": "google/gemma-2-2b-it",
            "scores": {"math": 0.6, "code": 0.55, "reasoning": 0.55,
                       "tool_calling": 0.6, "multilingual": 0.5,
                       "creative": 0.5, "instruction_following": 0.6},
            "size_billions": 2.0, "context_length": 8192, "gpu_memory_gb": 5.0,
        },
        {
            "model_name": "google/gemma-2-9b-it",
            "scores": {"math": 0.8, "code": 0.75, "reasoning": 0.8,
                       "tool_calling": 0.75, "multilingual": 0.7,
                       "creative": 0.7, "instruction_following": 0.8},
            "size_billions": 9.0, "context_length": 8192, "gpu_memory_gb": 19.0,
        },
    ]

    for d in defaults:
        cap = ModelCapability(**d)
        MODEL_CAPABILITIES[cap.model_name] = cap


# Populate on module load
_register_defaults()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model_capability(
    model_name: str,
    scores: dict[str, float],
    size_billions: float = 0.0,
    context_length: int = 4096,
    gpu_memory_gb: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> ModelCapability:
    """Register (or update) a capability profile for a model.

    Args:
        model_name: Full model identifier.
        scores: Dict of dimension -> score (0.0-1.0).
        size_billions: Parameter count in billions.
        context_length: Max context tokens.
        gpu_memory_gb: Approximate GPU memory in GB.
        metadata: Extra info.

    Returns:
        The registered ModelCapability.
    """
    cap = ModelCapability(
        model_name=model_name,
        scores=scores,
        size_billions=size_billions,
        context_length=context_length,
        gpu_memory_gb=gpu_memory_gb,
        metadata=metadata or {},
    )
    MODEL_CAPABILITIES[model_name] = cap
    logger.info("Registered capability profile for '%s'", model_name)
    return cap


def get_model_capability(model_name: str) -> ModelCapability | None:
    """Look up a capability profile by exact name, then by fuzzy match.

    Fuzzy matching normalises casing and strips organisation prefixes so
    that e.g. ``get_model_capability("qwen2.5-3b-instruct")`` finds
    ``Qwen/Qwen2.5-3B-Instruct``.
    """
    # Exact match first
    if model_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_name]

    # Fuzzy: normalise both sides
    norm = _normalise_name(model_name)
    for registered, cap in MODEL_CAPABILITIES.items():
        if _normalise_name(registered) == norm:
            return cap

    return None


def estimate_capability(model_name: str) -> ModelCapability:
    """Return known capability or synthesise a rough estimate from model name.

    The estimate uses the parameter count (extracted from the name) to
    interpolate plausible scores so that the router always has *something*
    to work with even for unknown models.
    """
    known = get_model_capability(model_name)
    if known is not None:
        return known

    # Estimate from name
    size = _extract_size_billions(model_name)
    base = min(0.4 + size * 0.07, 0.95)  # rough linear mapping

    scores = {dim: round(base, 2) for dim in CAPABILITY_DIMENSIONS}
    # Instruct-tuned models are better at instruction following / tool calling
    lower = model_name.lower()
    if "instruct" in lower or "chat" in lower or "it" in lower:
        scores["instruction_following"] = round(min(base + 0.1, 1.0), 2)
        scores["tool_calling"] = round(min(base + 0.05, 1.0), 2)

    cap = ModelCapability(
        model_name=model_name,
        scores=scores,
        size_billions=size,
        context_length=4096,
        gpu_memory_gb=round(size * 2.2, 1),  # rough FP16 estimate
    )
    logger.debug("Estimated capability for unknown model '%s': size=%.1fB", model_name, size)
    return cap


def list_registered_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(MODEL_CAPABILITIES.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    """Normalise a model name for fuzzy matching."""
    # Remove org prefix (e.g. "Qwen/")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name.lower().replace("-", "").replace("_", "")


def _extract_size_billions(model_name: str) -> float:
    """Extract parameter count in billions from a model name string."""
    lower = model_name.lower()
    # Match patterns like "7b", "1.5b", "0.5b", "14b"
    match = re.search(r"(\d+\.?\d*)\s*b", lower)
    if match:
        return float(match.group(1))
    return 1.0  # conservative default
