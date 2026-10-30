"""
Smart model loader with automatic detection and fallback.

This module provides intelligent model loading with:
- Automatic model type detection (HuggingFace vs API)
- Transformers-first with vLLM as optional production backend
- GPU allocation and VRAM management
- Quantization decision based on available memory
- Model validation and health checks
"""

from __future__ import annotations

# A few names below are no longer read by this module itself. They stay bound
# here so an existing ``from effgen.models.model_loader import ...`` keeps
# resolving them.
import logging
import os
from typing import TYPE_CHECKING, Any  # noqa: F401

from effgen.models.base import BaseModel, ModelType  # noqa: F401
from effgen.models.model_loader_capacity import ModelLoaderCapacityMixin
from effgen.models.model_loader_cloud import (  # noqa: F401  re-exported
    ModelLoaderCloudMixin,
    _get_cerebras_adapter,
    _get_fireworks_adapter,
    _get_groq_adapter,
    _get_hf_inference_adapter,
    _get_replicate_adapter,
    _get_together_adapter,
)
from effgen.models.model_loader_local import (  # noqa: F401  re-exported
    ModelLoaderLocalMixin,
    _missing_module,
    _require_torch,
)
from effgen.models.model_loader_routing import ModelLoaderRoutingMixin

logger = logging.getLogger(__name__)


class ModelLoader(ModelLoaderRoutingMixin, ModelLoaderCloudMixin, ModelLoaderLocalMixin, ModelLoaderCapacityMixin):
    """
    Smart model loader with automatic detection and configuration.

    This class handles:
    1. Model type detection (local, HuggingFace, or API)
    2. Engine selection (Transformers default, vLLM optional, or API adapter)
    3. GPU allocation and memory management
    4. Automatic quantization decisions
    5. Fallback strategies
    6. Model validation

    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load_model("Qwen/Qwen2.5-1.5B-Instruct")
        >>> # Uses Transformers by default, can specify vLLM with engine='vllm'

        >>> model = loader.load_model("gpt-4")
        >>> # Automatically uses OpenAI adapter
    """

    # API model prefixes for automatic detection
    OPENAI_MODELS = [
        "gpt-3.5", "gpt-4", "gpt-5", "text-davinci", "text-curie",
        "text-babbage", "text-ada",
        "o1", "o1-mini", "o1-preview",
        "o3", "o3-mini",
        "o4", "o4-mini",
    ]

    ANTHROPIC_MODELS = [
        "claude-3", "claude-2", "claude-instant", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku"
    ]

    GEMINI_MODELS = [
        "gemini-pro", "gemini-ultra", "gemini-flash", "gemini-1.5",
        "gemini-2", "gemini-3",
    ]

    # Local-engine prefixes recognized in an "engine:model_id" string, mirroring
    # the cloud "provider:model_id" syntax (e.g. "transformers:Qwen/Qwen2.5-7B-Instruct").
    _LOCAL_ENGINE_PREFIXES = frozenset({"transformers", "vllm", "gguf", "mlx"})

    def __init__(
        self,
        cache_dir: str | None = None,
        default_device: str = "auto",
        force_engine: str | None = None,
    ) -> None:
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
            default_device: Default device allocation ('auto', 'cuda', 'cpu')
            force_engine: Force specific engine ('vllm', 'transformers', 'auto-fast',
                or None for auto). 'auto-fast' prefers vLLM when it is importable
                and a GPU is usable, else Transformers; None defaults to Transformers.
        """
        # Expand ~ to full path and use environment variable if set
        default_cache = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
        self.cache_dir = os.path.expanduser(cache_dir) if cache_dir else default_cache
        self.default_device = default_device
        self.force_engine = force_engine

        self.loaded_models: dict[str, BaseModel] = {}

    def _validate_model(self, model: BaseModel) -> None:
        """
        Validate that the model is loaded and can generate.

        Args:
            model: Model instance to validate

        Raises:
            RuntimeError: If validation fails
        """
        logger.info("Validating model...")

        if not model.is_loaded():
            raise RuntimeError("Model validation failed: model not loaded")

        # Test token counting
        try:
            test_text = "Hello, world!"
            token_count = model.count_tokens(test_text)
            logger.info(f"Token counting works: '{test_text}' = {token_count.count} tokens")
        except Exception as e:
            logger.warning(f"Token counting validation failed: {e}")

        # Test context length
        try:
            context_length = model.get_context_length()
            logger.info(f"Context length: {context_length} tokens")
        except Exception as e:
            logger.warning(f"Context length validation failed: {e}")

        logger.info("Model validation passed")

    def unload_model(self, model_name: str) -> None:
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of model to unload
        """
        if model_name in self.loaded_models:
            logger.info(f"Unloading model: {model_name}")
            model = self.loaded_models[model_name]
            model.unload()
            del self.loaded_models[model_name]
            logger.info(f"Model '{model_name}' unloaded")
        else:
            logger.warning(f"Model '{model_name}' not found in loaded models")

    def unload_all(self) -> None:
        """
        Unload all loaded models.
        """
        logger.info("Unloading all models...")
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        logger.info("All models unloaded")

    def get_loaded_models(self) -> dict[str, BaseModel]:
        """
        Get dictionary of all loaded models.

        Returns:
            Dict mapping model names to model instances
        """
        return self.loaded_models.copy()

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """
        Get information about a loaded model.

        Args:
            model_name: Name of the model

        Returns:
            Model metadata dict or None if not loaded
        """
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            return model.get_metadata()
        return None


# Convenience function for quick model loading
def load_model(
    model_name: str,
    engine: str | None = None,
    engine_config: dict[str, Any] | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    apply_chat_template: bool = True,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any
) -> BaseModel:
    """Convenience function to quickly load a model.

    Provider prefixes route to a remote API (``"openai:..."``, ``"gemini:..."``,
    ``"hf:..."`` etc.). In particular ``"hf:<repo>"`` is the **remote**
    HuggingFace Inference API — to run the same repo **locally** on your GPU,
    pass ``engine="transformers"`` (or ``"vllm"``) with a bare model id instead
    of the ``hf:`` prefix.

    ``base_url`` points effGen at any server speaking the OpenAI protocol —
    vLLM, SGLang, TGI, llama.cpp, Ollama, LM Studio, a gateway or a corporate
    proxy — instead of loading a second copy of the weights in this process.

    Args:
        model_name: Model identifier
        engine: Engine to use ('vllm', 'transformers', 'auto-fast', or None for auto).
                'auto-fast' prefers vLLM when available and the GPU is usable, else
                Transformers. None defaults to Transformers.
        engine_config: Optional engine configuration
        tensor_parallel_size: Number of GPUs for tensor parallelism (vLLM only).
                             If not specified, auto-detected based on model size.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0, vLLM only).
                               Default is 0.90. Lower this if you get CUDA OOM errors.
        apply_chat_template: Whether to automatically apply chat templates for
                            instruction-tuned models (default: True, vLLM only).
                            This ensures proper formatting for models like Qwen-Instruct.
        provider: Route to this remote provider instead of a local engine; the
            same choice a ``"provider:model"`` prefix makes. Use
            ``"openai_compatible"`` for a server of your own speaking the
            OpenAI protocol.
        base_url: Endpoint for the OpenAI protocol, e.g.
            ``"http://127.0.0.1:8000/v1"``. Giving one routes the call to the
            OpenAI-compatible adapter whatever ``provider`` says, since the
            model ids, context window and pricing are then the server's.
            Falls back to ``EFFGEN_BASE_URL``, ``OPENAI_BASE_URL`` or
            ``OPENAI_API_BASE`` when ``provider="openai_compatible"`` is asked
            for without one.
        api_key: Credential for the endpoint. A local server that checks
            nothing needs none; effGen sends a placeholder.
        **kwargs: Additional parameters (e.g., quantization="4bit", trust_remote_code=True)

    Returns:
        Loaded model instance

    Example:
        >>> from effgen.models import load_model
        >>> # Default uses Transformers engine
        >>> model = load_model("Qwen/Qwen2.5-1.5B-Instruct")
        >>> result = model.generate("Hello, how are you?")

        >>> # Explicitly use vLLM for production (5-10x faster)
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm")

        >>> # With tensor parallelism for large models
        >>> model = load_model("meta-llama/Llama-3.3-70B-Instruct", engine="vllm", tensor_parallel_size=4)

        >>> # Lower GPU memory usage if getting OOM errors
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm", gpu_memory_utilization=0.7)

        >>> # Disable chat template for raw text generation
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="vllm", apply_chat_template=False)

        >>> # Fail instead of falling back to CPU when the GPU can't hold the model
        >>> model = load_model("Qwen/Qwen2.5-7B-Instruct", engine="transformers", require_gpu=True)

        >>> # Talk to a model you already serve, instead of loading it again
        >>> model = load_model(
        ...     "Qwen/Qwen2.5-7B-Instruct",
        ...     provider="openai_compatible",
        ...     base_url="http://127.0.0.1:8000/v1",
        ... )
    """
    # Pass tensor_parallel_size to kwargs if specified
    if tensor_parallel_size is not None:
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    # Pass gpu_memory_utilization to kwargs if specified
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    # Pass apply_chat_template for vLLM
    kwargs["apply_chat_template"] = apply_chat_template

    if provider is not None:
        kwargs["provider"] = provider

    # A base_url names an OpenAI-protocol server, which is a routing decision
    # as much as an adapter argument: with no provider named, it is the whole
    # instruction, so it selects the compatible adapter rather than falling
    # through to a local download of the model id.
    if base_url is not None:
        kwargs["base_url"] = base_url
        kwargs.setdefault("provider", "openai_compatible")
    if api_key is not None:
        kwargs["api_key"] = api_key

    loader = ModelLoader(force_engine=engine)
    return loader.load_model(model_name, engine_config, **kwargs)
