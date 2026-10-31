"""Weight loading, quantization and unloading for the Transformers engine.

Assembles the ``from_pretrained`` kwargs, loads and drops model weights, builds
the quantization config and reads the model's maximum length. Mixed into
:class:`~effgen.models.transformers_engine.TransformersEngine`; not usable on
its own.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - only on an install without torch
    from effgen.models._adapter_utils import missing_torch_error

    raise missing_torch_error("transformers") from exc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from effgen.models._adapter_utils import (
    attach_error_context,
    provider_runtime_error,
)
from effgen.models.errors import ModelNotFoundError
from effgen.models.transformers_engine_support import (
    GPUPlacementError,
    ModelNotCachedError,
    _is_cache_miss_error,
    _list_cached_model_repos,
    _offline_mode_active,
    _quiet_model_load,
)

logger = logging.getLogger("effgen.models.transformers_engine")


class TransformersLoadingMixin:
    """Weight loading, quantization and unloading."""

    def _assemble_model_kwargs(
        self, quantization_config: BitsAndBytesConfig | None
    ) -> dict[str, Any]:
        """Build kwargs dict for AutoModelForCausalLM.from_pretrained."""
        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            import transformers
            if hasattr(transformers, 'VERSION') or int(transformers.__version__.split('.')[0]) >= 5:
                model_kwargs["dtype"] = self.torch_dtype
            else:
                model_kwargs["torch_dtype"] = self.torch_dtype

        if self.device == "cuda":
            model_kwargs["device_map"] = self._effective_device_map()
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            if self.offload_folder:
                model_kwargs["offload_folder"] = self.offload_folder

        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model_kwargs.update(self.additional_kwargs)
        return model_kwargs

    def _from_pretrained_with_flash_fallback(self, model_kwargs: dict[str, Any]) -> None:
        """Load self.model, falling back to standard attention if Flash Attention fails.

        Loads quietly: weight-loading progress bars / INFO reports are suppressed
        (diagnostics, not user output) via _quiet_model_load().
        """
        try:
            with warnings.catch_warnings(), _quiet_model_load():
                warnings.filterwarnings('ignore', message='.*FlashAttention.*')
                warnings.filterwarnings('ignore', message='.*flash_attn.*')
                self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        except Exception as e:
            if self.use_flash_attention and "flash" in str(e).lower():
                logger.debug("Flash Attention not available, using standard attention")
                model_kwargs.pop("attn_implementation", None)
                with warnings.catch_warnings(), _quiet_model_load():
                    warnings.filterwarnings('ignore', message='.*FlashAttention.*')
                    warnings.filterwarnings('ignore', message='.*flash_attn.*')
                    self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            else:
                raise

    def _load_model_weights(
        self, quantization_config: BitsAndBytesConfig | None = None
    ) -> None:
        """Load or reload causal LM weights (tokenizer must already be loaded)."""
        if quantization_config is None and self.quantization_bits is not None:
            quantization_config = self._create_quantization_config()

        model_kwargs = self._assemble_model_kwargs(quantization_config)
        self._from_pretrained_with_flash_fallback(model_kwargs)

        if "device_map" not in model_kwargs and self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

    def _drop_model_weights(self) -> None:
        """Free model weights and CUDA cache without unloading the tokenizer."""
        # The next weights loaded have not been probed yet.
        self._device_map_probe_passed = False
        if self.model is not None:
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.model, recurse=True)
            except Exception:
                pass
            del self.model
            self.model = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def load(self) -> None:
        """
        Load the model using HuggingFace Transformers.

        Raises:
            RuntimeError: If model loading fails
            ValueError: If configuration is invalid
        """
        try:
            logger.debug(f"Loading model '{self.model_name}' with Transformers...")

            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.debug(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
            else:
                self.device = "cpu"
                # If the host actually has NVIDIA GPUs but torch can't use them
                # (almost always a torch-CUDA vs driver mismatch), emit one clear,
                # actionable warning instead of a bland "CUDA not available".
                from effgen.gpu.cuda_compat import warn_cuda_mismatch_once
                if not warn_cuda_mismatch_once():
                    logger.warning("CUDA not available, using CPU (this will be slow)")

            # Setup quantization config if specified
            quantization_config = None
            if self.quantization_bits is not None:
                quantization_config = self._create_quantization_config()

            # Determine torch dtype
            if self.torch_dtype is None:
                if self.device == "cuda":
                    # Use bfloat16 if available, else float16
                    if torch.cuda.is_bf16_supported():
                        self.torch_dtype = torch.bfloat16
                    else:
                        self.torch_dtype = torch.float16
                else:
                    self.torch_dtype = torch.float32

            logger.debug(
                f"Configuration: quantization={self.quantization_bits}-bit, "
                f"dtype={self.torch_dtype}, flash_attention={self.use_flash_attention}"
            )

            # Load tokenizer (quiet: suppress Transformers progress bars / INFO
            # reports during load — these are diagnostics, not user output)
            with _quiet_model_load():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )

            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._load_model_weights(quantization_config)

            # Reconcile the requested device with where the parameters actually
            # landed. With device_map="auto", accelerate offloads layers to CPU
            # (or disk) when the GPU can't hold the model, so an intended "cuda"
            # load can end up partly or wholly on CPU. Report the real placement
            # so callers aren't told "cuda" while inference runs on CPU.
            intended_device = self.device
            self.device = self._resolve_placement()
            if intended_device == "cuda" and self.device != "cuda":
                free_gb = self._free_vram_gb()
                where = "CPU" if self.device == "cpu" else "CPU/disk (mixed placement)"
                if self.require_gpu:
                    raise GPUPlacementError(
                        f"Model '{self.model_name}' could not be placed entirely on the GPU "
                        f"(only {free_gb:.2f} GB free) and require_gpu is set. Free GPU memory, "
                        f"choose a smaller model, or enable quantization (e.g. quantization='4bit')."
                    )
                logger.warning(
                    "Model '%s' did not fit in available GPU memory (%.2f GB free); "
                    "running on %s. Inference will be slower. Free GPU memory or pass "
                    "quantization='4bit' to keep it on the GPU.",
                    self.model_name, free_gb, where,
                )

            # Store metadata
            self._context_length = self._get_max_length()
            self._metadata = {
                "model_name": self.model_name,
                "quantization": f"{self.quantization_bits}-bit" if self.quantization_bits else None,
                "dtype": str(self.torch_dtype),
                "device": str(self.device),
                "flash_attention": self.use_flash_attention,
                "max_length": self._context_length,
                "num_parameters": self.model.num_parameters(),
            }

            self._is_loaded = True
            logger.debug(f"Model '{self.model_name}' loaded successfully with Transformers")
            self._ensure_device_map_viable_before_sampling()

        except GPUPlacementError:
            # An explicit require_gpu policy failure — surface it unchanged
            # rather than wrapping it as a generic load failure.
            raise
        except Exception as e:
            # In offline mode a missing/misspelled repo surfaces as a
            # connectivity error ("couldn't connect to huggingface.co"). Report
            # it as a local cache miss and name what is cached instead.
            offline = _offline_mode_active()
            if offline or _is_cache_miss_error(e):
                cached = _list_cached_model_repos()
                listed = ", ".join(cached) if cached else "none"
                reason = (
                    "offline mode is set"
                    if offline
                    else "it could not be downloaded (no network)"
                )
                raise attach_error_context(
                    ModelNotCachedError(
                        f"Model '{self.model_name}' is not in the local "
                        f"HuggingFace cache and {reason}. Cached models: "
                        f"{listed}."
                    ),
                    "transformers", self.model_name, "load",
                    source=ModelNotFoundError("transformers", self.model_name),
                ) from e
            logger.error(f"Failed to load model with Transformers: {e}")
            raise provider_runtime_error(
                "transformers", self.model_name, "load", e,
                message="Transformers model loading failed",
            ) from e

    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create quantization configuration.

        Returns:
            BitsAndBytesConfig for bitsandbytes quantization

        Raises:
            ValueError: If quantization_bits is invalid
        """
        if self.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(
                f"Invalid quantization_bits: {self.quantization_bits}. "
                "Must be 4 or 8."
            )

    def _get_max_length(self) -> int:
        """
        Get maximum context length from model config.

        Returns:
            int: Maximum sequence length
        """
        config = self.model.config

        # Try different config attributes
        for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
            if hasattr(config, attr):
                return getattr(config, attr)

        # Some models (e.g. gemma-3) nest config inside text_config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
                if hasattr(text_config, attr):
                    return getattr(text_config, attr)

        logger.warning("Could not determine max length from config, using 2048")
        return 2048

    def unload(self) -> None:
        """
        Unload the model and free memory.

        Removes any accelerate device-dispatch hooks before deleting the
        model — leftover hooks can corrupt the CUDA forward state of
        subsequently-loaded models in the same process (observed as
        intermittent C-level aborts inside Qwen2 RMSNorm under pytest).

        The device memory the weights occupied is returned to the GPU, so the
        next model to load sees it as free.
        """
        if self.model is not None:
            logger.debug(f"Unloading model '{self.model_name}'...")
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(self.model, recurse=True)
            except Exception:
                logger.debug("Failed to remove accelerate hooks during unload", exc_info=True)
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Drop the cached capability probe with the tokenizer it measured, so
        # unload releases it and a later load re-measures.
        self._tool_template_probe = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug("torch.cuda.synchronize() failed during unload", exc_info=True)
            from effgen.gpu.utils import release_cached_memory

            release_cached_memory()

        self._is_loaded = False
        self._pin_device_map_for_cuda = False
        self._retrying_after_cuda_assert = False
        logger.debug(f"Model '{self.model_name}' unloaded successfully")
