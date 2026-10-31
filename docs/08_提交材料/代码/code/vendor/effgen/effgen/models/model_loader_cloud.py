"""Cloud-adapter construction for the model loader.

Holds the deferred provider-adapter imports and the loader methods that build
the OpenAI, Anthropic and Gemini adapters. Mixed into
:class:`~effgen.models.model_loader.ModelLoader`; not usable on its own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Cloud adapters and the heavy local-inference deps (torch, the
    # Transformers/vLLM engines) are imported lazily inside the ``_load_*``
    # methods that actually construct them. This matches the six other adapters
    # (HF/Replicate/Cerebras/Groq/Together/Fireworks) and — importantly — breaks
    # the import cycle that arises when ``effgen.models`` is imported before
    # ``effgen.core.agent`` (anthropic_adapter -> core.messages -> core.agent ->
    # model_loader). It also keeps `from effgen import Agent` / the CLI from
    # pulling torch or transformers for a pure cloud-API workflow.
    from effgen.models.anthropic_adapter import AnthropicAdapter
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.models.openai_adapter import OpenAIAdapter
    from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter

logger = logging.getLogger("effgen.models.model_loader")

# Loader kwargs that configure a local engine and mean nothing to a remote API.
# Dropped before an adapter is constructed so a caller can pass one loader
# call's worth of options without the cloud path rejecting them.
_LOCAL_ONLY_KWARGS = (
    "apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization",
    "trust_remote_code", "quantization", "device", "torch_dtype",
)


# Cerebras import is deferred to avoid hard dependency when cerebras extra is absent.
_CerebrasAdapter = None
# Groq import is deferred to avoid hard dependency when groq extra is absent.
_GroqAdapter = None
# Together import is deferred to avoid hard dependency when together extra is absent.
_TogetherAdapter = None
# Fireworks import is deferred to avoid hard dependency when fireworks extra is absent.
_FireworksAdapter = None
# Replicate import is deferred to avoid hard dependency when replicate extra is absent.
_ReplicateAdapter = None
# HF Inference import is deferred to avoid hard dependency when hf extra is absent.
_HFInferenceAdapter = None


def _get_hf_inference_adapter():
    global _HFInferenceAdapter
    if _HFInferenceAdapter is None:
        from effgen.models.hf_inference_adapter import HFInferenceAdapter
        _HFInferenceAdapter = HFInferenceAdapter
    return _HFInferenceAdapter


def _get_replicate_adapter():
    global _ReplicateAdapter
    if _ReplicateAdapter is None:
        from effgen.models.replicate_adapter import ReplicateAdapter
        _ReplicateAdapter = ReplicateAdapter
    return _ReplicateAdapter


def _get_cerebras_adapter():
    global _CerebrasAdapter
    if _CerebrasAdapter is None:
        from effgen.models.cerebras_adapter import CerebrasAdapter
        _CerebrasAdapter = CerebrasAdapter
    return _CerebrasAdapter


def _get_groq_adapter():
    global _GroqAdapter
    if _GroqAdapter is None:
        from effgen.models.groq_adapter import GroqAdapter
        _GroqAdapter = GroqAdapter
    return _GroqAdapter


def _get_together_adapter():
    global _TogetherAdapter
    if _TogetherAdapter is None:
        from effgen.models.together_adapter import TogetherAdapter
        _TogetherAdapter = TogetherAdapter
    return _TogetherAdapter


def _get_fireworks_adapter():
    global _FireworksAdapter
    if _FireworksAdapter is None:
        from effgen.models.fireworks_adapter import FireworksAdapter
        _FireworksAdapter = FireworksAdapter
    return _FireworksAdapter


class ModelLoaderCloudMixin:
    """Construction of the cloud provider adapters."""

    def _load_openai_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> OpenAIAdapter:
        """
        Load OpenAI model.

        Args:
            model_name: OpenAI model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            OpenAIAdapter instance
        """
        from effgen.models.openai_adapter import OpenAIAdapter

        logger.info(f"Loading OpenAI model: {model_name}")

        params = config or {}
        params.update(kwargs)
        # Drop kwargs only meaningful for local/HF engines
        for k in _LOCAL_ONLY_KWARGS:
            params.pop(k, None)

        return OpenAIAdapter(model_name=model_name, **params)

    def _load_openai_compatible_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> "OpenAICompatibleAdapter":
        """
        Load a model served over the OpenAI protocol at a caller-supplied URL.

        Args:
            model_name: The id the server serves the model under
            config: Optional configuration
            **kwargs: Additional parameters, including ``base_url`` and
                ``api_key``

        Returns:
            OpenAICompatibleAdapter instance
        """
        from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter

        params = config or {}
        params.update(kwargs)
        # Drop kwargs only meaningful for local/HF engines
        for k in _LOCAL_ONLY_KWARGS:
            params.pop(k, None)

        logger.info(
            "Loading model %r over the OpenAI protocol from %s",
            model_name, params.get("base_url") or "the configured endpoint",
        )
        return OpenAICompatibleAdapter(model_name=model_name, **params)

    def _load_anthropic_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> AnthropicAdapter:
        """
        Load Anthropic model.

        Args:
            model_name: Anthropic model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            AnthropicAdapter instance
        """
        from effgen.models.anthropic_adapter import AnthropicAdapter

        logger.info(f"Loading Anthropic model: {model_name}")

        params = config or {}
        params.update(kwargs)
        for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
            params.pop(k, None)

        return AnthropicAdapter(model_name=model_name, **params)

    def _load_gemini_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> GeminiAdapter:
        """
        Load Gemini model.

        Args:
            model_name: Gemini model identifier
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            GeminiAdapter instance
        """
        from effgen.models.gemini_adapter import GeminiAdapter

        logger.info(f"Loading Gemini model: {model_name}")

        params = config or {}
        params.update(kwargs)
        for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
            params.pop(k, None)

        return GeminiAdapter(model_name=model_name, **params)
