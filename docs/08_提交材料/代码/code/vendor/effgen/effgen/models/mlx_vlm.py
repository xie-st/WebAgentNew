"""
MLX-VLM adapter for effGen — thin effGen-interface wrapper around MLXVLMEngine.

This module provides an ``MLXVLMAdapter`` that:
- Implements the ``BaseModel`` interface expected by the effGen router and presets.
- Accepts ``List[Message]`` inputs (the standard effGen wire format) including
  ``ImagePart`` content for vision-language queries.
- Delegates heavy lifting to ``MLXVLMEngine``, which itself wraps the ``mlx-vlm``
  library.
- Raises ``CapabilityNotSupportedError(Capability.audio_input)`` and
  ``CapabilityNotSupportedError(Capability.video_input)`` because MLX-VLM
  models only accept static image input.
- Raises ``RuntimeError`` with a clear Apple Silicon requirement message on
  non-macOS platforms (Linux CI is expected to skip live tests for this adapter).

Supported architectures (via mlx-vlm):
    Qwen2-VL, LLaVA, Phi-3-Vision, Pixtral, PaliGemma, and 30+ others.
    See https://github.com/Blaizzy/mlx-vlm for the full list.

Usage (macOS + Apple Silicon only)::

    from effgen.models.mlx_vlm import MLXVLMAdapter

    adapter = MLXVLMAdapter("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    adapter.load()

    from effgen.core.messages import Message, Role
    from effgen.core.multimodal import image_from

    img = image_from("https://example.com/photo.jpg")
    msg = Message(role=Role.USER, content=[img, TextPart(text="What is in this image?")])
    result = adapter.generate([msg])
    print(result.text)
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from typing import Any

from effgen.core.messages import AudioPart, ImagePart, Message, TextPart, VideoPart
from effgen.errors import CapabilityNotSupportedError
from effgen.models._adapter_utils import not_loaded_error
from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)
from effgen.models.capabilities import Capability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public default model suggestions
# ---------------------------------------------------------------------------

#: Recommended 4-bit quantised VLMs on the mlx-community HuggingFace org.
#: These are verified to work with mlx-vlm on M-series Apple Silicon.
RECOMMENDED_MODELS: dict[str, str] = {
    "qwen2-vl-2b": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "qwen2-vl-7b": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
    "llava-phi-3": "mlx-community/llava-phi-3-mini-4bit",
    "phi-3-vision": "mlx-community/Phi-3.5-vision-instruct-4bit",
    "pixtral-12b": "mlx-community/pixtral-12b-4bit",
    "paligemma-3b": "mlx-community/paligemma-3b-mix-448-8bit",
    "idefics3": "mlx-community/Idefics3-8B-Llama3-8bit",
}

DEFAULT_MODEL = RECOMMENDED_MODELS["qwen2-vl-2b"]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MLXVLMAdapter(BaseModel):
    """
    effGen ``BaseModel`` adapter for local MLX-based vision-language models.

    This adapter wraps ``MLXVLMEngine`` and translates between the effGen
    ``List[Message]`` wire format and the raw prompt / image-list form
    expected by ``mlx-vlm``.

    Only static image input is supported; audio and video parts raise
    ``CapabilityNotSupportedError`` immediately.

    The adapter is designed so that unit tests can run on any platform by
    injecting a fake ``_engine`` attribute — no Apple Silicon required for
    testing.

    Args:
        model_name: HuggingFace model ID or local path to MLX-VLM weights.
            Defaults to ``"mlx-community/Qwen2-VL-2B-Instruct-4bit"``.
        max_tokens: Context / output token limit.  Passed to ``MLXVLMEngine``.
        system_prompt: Optional default system prompt.
        temperature: Default generation temperature (overridable per call).
        **kwargs: Extra kwargs forwarded to ``MLXVLMEngine``.

    Raises:
        RuntimeError: On non-Apple-Silicon platforms (raised at :meth:`load` time).
        CapabilityNotSupportedError: If audio or video parts are present in
            the input messages.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=ModelType.MLX_VLM,
            context_length=None,
        )
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._engine_kwargs = kwargs
        self._engine: Any = None  # Populated by load(); injectable for tests.

    # ------------------------------------------------------------------
    # Capability declarations
    # ------------------------------------------------------------------

    @property
    def supported_capabilities(self) -> frozenset[Capability]:
        """MLX-VLM supports chat + vision; audio and video are not supported."""
        return frozenset({Capability.chat, Capability.vision, Capability.streaming})

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the VLM weights via MLXVLMEngine.

        Raises:
            RuntimeError: If not on Apple Silicon or mlx-vlm is not installed.
        """
        if self._engine is not None:
            logger.debug("MLXVLMAdapter already loaded, skipping reload.")
            return

        from effgen.models.mlx_vlm_engine import MLXVLMEngine

        self._engine = MLXVLMEngine(
            model_name=self.model_name,
            max_tokens=self._max_tokens,
            system_prompt=self._system_prompt,
            **self._engine_kwargs,
        )
        self._engine.load()
        self._is_loaded = True
        logger.info(f"MLXVLMAdapter loaded '{self.model_name}'")

    def unload(self) -> None:
        """Unload the VLM weights and free device memory."""
        if self._engine is not None:
            self._engine.unload()
            self._engine = None
        self._is_loaded = False
        logger.info(f"MLXVLMAdapter unloaded '{self.model_name}'")

    @property
    def loaded(self) -> bool:
        """True while the MLX-VLM engine is loaded."""
        return self._is_loaded and self._engine is not None

    # ------------------------------------------------------------------
    # Message → prompt + images translation
    # ------------------------------------------------------------------

    def _messages_to_prompt_and_images(
        self,
        messages: list[Message],
    ) -> tuple[str, list[Any]]:
        """Flatten ``List[Message]`` into a (prompt_str, image_list) pair.

        Text parts are concatenated with newlines.
        ImagePart bytes are converted to ``PIL.Image`` objects for mlx-vlm.
        Audio/video parts trigger ``CapabilityNotSupportedError``.

        Returns:
            prompt: Combined text prompt string.
            images: List of ``PIL.Image`` objects (may be empty for text-only).
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            PILImage = None  # type: ignore[assignment]

        text_parts: list[str] = []
        images: list[Any] = []

        for msg in messages:
            if isinstance(msg.content, str):
                # Back-compat: plain string content
                text_parts.append(msg.content)
                continue

            for part in msg.content:
                if isinstance(part, TextPart):
                    text_parts.append(part.text)
                elif isinstance(part, ImagePart):
                    if PILImage is None:
                        raise RuntimeError(
                            "Pillow is required to process ImagePart with MLXVLMAdapter. "
                            "Install it: pip install Pillow"
                        )
                    pil_img = PILImage.open(io.BytesIO(part.image))
                    images.append(pil_img)
                elif isinstance(part, AudioPart):
                    raise CapabilityNotSupportedError(
                        Capability.audio_input,
                        provider="mlx_vlm",
                        hint=(
                            "MLX-VLM models do not support audio input. "
                            "Use GeminiAdapter or OpenAIAdapter for audio."
                        ),
                    )
                elif isinstance(part, VideoPart):
                    raise CapabilityNotSupportedError(
                        Capability.video_input,
                        provider="mlx_vlm",
                        hint=(
                            "MLX-VLM models do not support video input directly. "
                            "Pre-sample video frames into ImageParts before calling this adapter, "
                            "or use GeminiAdapter for native video support."
                        ),
                    )
                # ToolCallPart / ToolResultPart: skip silently (no function calling)

        prompt = "\n".join(text_parts)
        return prompt, images

    def _build_config(self, config: GenerationConfig | None) -> GenerationConfig:
        if config is None:
            config = GenerationConfig()
        if config.temperature is None or config.temperature == 0.7:
            config = GenerationConfig(
                temperature=self._temperature,
                max_tokens=config.max_tokens or self._max_tokens or 512,
                stop_sequences=config.stop_sequences,
            )
        return config

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response for the given messages.

        Args:
            messages: List of ``Message`` objects (may include ``ImagePart``s).
            config: Optional generation configuration.
            **kwargs: Extra kwargs (ignored for compatibility).

        Returns:
            ``GenerationResult`` with the generated text.

        Raises:
            RuntimeError: If the model is not loaded.
            CapabilityNotSupportedError: If audio or video parts are present.
        """
        if not self.loaded:
            raise not_loaded_error("mlx_vlm", self.model_name, "generate")

        prompt, images = self._messages_to_prompt_and_images(messages)
        cfg = self._build_config(config)

        result = self._engine.generate(
            prompt=prompt,
            config=cfg,
            system_prompt=self._system_prompt,
            images=images if images else None,
        )
        return result

    def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated text for the given messages.

        Note: MLX-VLM streams text-only; multimodal (image) generation
        generates the full response and yields it in one chunk.

        Args:
            messages: List of ``Message`` objects.
            config: Optional generation configuration.
            **kwargs: Extra parameters forwarded to the MLX-VLM backend.

        Yields:
            str chunks of generated text.

        Raises:
            RuntimeError: If the model is not loaded.
            CapabilityNotSupportedError: If audio or video parts are present.
        """
        if not self.loaded:
            raise not_loaded_error("mlx_vlm", self.model_name, "generate_stream")

        prompt, images = self._messages_to_prompt_and_images(messages)
        cfg = self._build_config(config)

        yield from self._engine.generate_stream(
            prompt=prompt,
            config=cfg,
            system_prompt=self._system_prompt,
            images=images if images else None,
        )

    # ------------------------------------------------------------------
    # Token counting + context length
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> TokenCount:
        """Estimate token count using the loaded tokenizer.

        Falls back to word-count heuristic if the engine is not yet loaded.
        """
        if self.loaded and self._engine is not None and self._engine.tokenizer is not None:
            try:
                n = len(self._engine.tokenizer.encode(text))
                return TokenCount(count=n, model_name=self.model_name)
            except Exception:
                logger.debug("Tokenizer-based token counting failed", exc_info=True)
        # Heuristic: ~1.3 tokens per whitespace token for English text
        return TokenCount(
            count=max(1, int(len(text.split()) * 1.3)),
            model_name=self.model_name,
        )

    def get_context_length(self) -> int:
        """Return the model's maximum context length."""
        if self.loaded and self._engine is not None:
            return self._engine._context_length or 4096
        return 4096  # Conservative fallback

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict[str, Any]:
        """Return adapter + engine metadata."""
        base = {
            "adapter": "mlx_vlm",
            "model_name": self.model_name,
            "platform": "apple_silicon",
            "supported_capabilities": [c.value for c in self.supported_capabilities],
        }
        if self._engine is not None:
            base.update(self._engine._metadata or {})
        return base

    def __repr__(self) -> str:
        loaded = "loaded" if self.loaded else "not loaded"
        return f"MLXVLMAdapter(model_name={self.model_name!r}, {loaded})"
