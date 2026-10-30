"""
Unit tests for effgen.models.mlx_vlm.MLXVLMAdapter.

These tests use a fake MLXVLMEngine injected via adapter._engine so no Apple
Silicon / mlx-vlm library is required. All tests run on any platform.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from effgen.core.messages import AudioPart, ImagePart, Message, Role, TextPart, VideoPart
from effgen.errors import CapabilityNotSupportedError
from effgen.models.base import GenerationConfig, GenerationResult, ModelType
from effgen.models.capabilities import Capability
from effgen.models.mlx_vlm import DEFAULT_MODEL, RECOMMENDED_MODELS, MLXVLMAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_bytes() -> bytes:
    """Return a minimal valid PNG header (8 bytes) + padding."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _fake_engine(text: str = "fake response") -> MagicMock:
    """Return a MagicMock that mimics MLXVLMEngine."""
    engine = MagicMock()
    engine._is_loaded = True
    engine._context_length = 4096
    engine._metadata = {"engine": "mlx_vlm_fake"}
    engine.tokenizer = MagicMock()
    engine.tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

    result = GenerationResult(
        text=text,
        tokens_used=5,
        finish_reason="stop",
        model_name="fake-model",
        metadata={"num_images": 0, "engine": "mlx_vlm_fake"},
    )
    engine.generate = MagicMock(return_value=result)
    engine.generate_stream = MagicMock(return_value=iter([text]))
    return engine


def _make_adapter(engine: MagicMock | None = None) -> MLXVLMAdapter:
    """Create an adapter with an injected fake engine (no Apple Silicon needed)."""
    adapter = MLXVLMAdapter(model_name="fake-vlm")
    if engine is None:
        engine = _fake_engine()
    adapter._engine = engine
    adapter._is_loaded = True
    return adapter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_adapter_default_model_name():
    adapter = MLXVLMAdapter()
    assert adapter.model_name == DEFAULT_MODEL


def test_adapter_custom_model_name():
    adapter = MLXVLMAdapter(model_name="mlx-community/test-model")
    assert adapter.model_name == "mlx-community/test-model"


def test_adapter_model_type():
    adapter = MLXVLMAdapter()
    assert adapter.model_type == ModelType.MLX_VLM


def test_recommended_models_non_empty():
    assert len(RECOMMENDED_MODELS) >= 1
    for key, value in RECOMMENDED_MODELS.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert "/" in value  # HuggingFace org/model format


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def test_adapter_supports_vision():
    adapter = MLXVLMAdapter()
    assert Capability.vision in adapter.supported_capabilities


def test_adapter_supports_chat():
    adapter = MLXVLMAdapter()
    assert Capability.chat in adapter.supported_capabilities


def test_adapter_supports_streaming():
    adapter = MLXVLMAdapter()
    assert Capability.streaming in adapter.supported_capabilities


# ---------------------------------------------------------------------------
# Message translation — text only
# ---------------------------------------------------------------------------

def test_messages_to_prompt_text_only():
    adapter = _make_adapter()
    msg = Message(role=Role.USER, content=[TextPart(text="Hello world")])
    prompt, images = adapter._messages_to_prompt_and_images([msg])
    assert prompt == "Hello world"
    assert images == []


def test_messages_to_prompt_multiple_text_messages():
    adapter = _make_adapter()
    msgs = [
        Message(role=Role.SYSTEM, content=[TextPart(text="System prompt.")]),
        Message(role=Role.USER, content=[TextPart(text="User question.")]),
    ]
    prompt, images = adapter._messages_to_prompt_and_images(msgs)
    assert "System prompt." in prompt
    assert "User question." in prompt
    assert images == []


def test_messages_to_prompt_back_compat_string_content():
    """Message with a plain string content (back-compat shim)."""
    adapter = _make_adapter()
    msg = Message(role=Role.USER, content="plain string")
    prompt, images = adapter._messages_to_prompt_and_images([msg])
    assert "plain string" in prompt
    assert images == []


# ---------------------------------------------------------------------------
# Message translation — image input (fake PIL)
# ---------------------------------------------------------------------------

def test_messages_to_prompt_image_part_converted(monkeypatch):
    """ImagePart bytes should become a PIL.Image in the images list."""
    # Fake PIL.Image module (Pillow is optional; skip if it can't be imported).
    pytest.importorskip("PIL", reason="Pillow not installed (pip install effgen[tools])")
    fake_pil_image = MagicMock()
    fake_image_open = MagicMock(return_value=fake_pil_image)

    import PIL.Image as PILImage
    monkeypatch.setattr(PILImage, "open", fake_image_open)

    adapter = _make_adapter()
    img_bytes = _make_image_bytes()
    img_part = ImagePart(image=img_bytes, mime="image/png")
    msg = Message(role=Role.USER, content=[
        TextPart(text="Describe this image"),
        img_part,
    ])

    prompt, images = adapter._messages_to_prompt_and_images([msg])

    assert "Describe this image" in prompt
    assert len(images) == 1
    fake_image_open.assert_called_once()


def test_messages_to_prompt_multiple_image_parts(monkeypatch):
    """Multiple ImagePart objects should all appear in the images list."""
    pytest.importorskip("PIL", reason="Pillow not installed (pip install effgen[tools])")
    fake_pil_image = MagicMock()
    import PIL.Image as PILImage
    monkeypatch.setattr(PILImage, "open", MagicMock(return_value=fake_pil_image))

    adapter = _make_adapter()
    img_bytes = _make_image_bytes()
    msg = Message(role=Role.USER, content=[
        ImagePart(image=img_bytes, mime="image/png"),
        TextPart(text="Compare these images"),
        ImagePart(image=img_bytes, mime="image/jpeg"),
    ])

    prompt, images = adapter._messages_to_prompt_and_images([msg])
    assert len(images) == 2
    assert "Compare these images" in prompt


# ---------------------------------------------------------------------------
# Capability gating — audio and video raise errors
# ---------------------------------------------------------------------------

def test_audio_part_raises_capability_error():
    adapter = _make_adapter()
    audio_bytes = b"\xff\xfb" + b"\x00" * 32  # MP3 magic
    audio_part = AudioPart(audio=audio_bytes, mime="audio/mpeg")
    msg = Message(role=Role.USER, content=[audio_part])

    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        adapter._messages_to_prompt_and_images([msg])

    err = exc_info.value
    # capability is stored as the string value (not the Enum)
    assert err.capability == Capability.audio_input.value
    assert "mlx_vlm" in err.provider.lower()


def test_video_part_raises_capability_error():
    adapter = _make_adapter()
    img_bytes = _make_image_bytes()
    video_part = VideoPart(frames=[img_bytes], fps=1.0, mime="image/jpeg")
    msg = Message(role=Role.USER, content=[video_part])

    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        adapter._messages_to_prompt_and_images([msg])

    err = exc_info.value
    # capability is stored as the string value (not the Enum)
    assert err.capability == Capability.video_input.value
    assert "mlx_vlm" in err.provider.lower()


# ---------------------------------------------------------------------------
# generate() integration
# ---------------------------------------------------------------------------

def test_generate_calls_engine_generate():
    engine = _fake_engine("A cat is visible.")
    adapter = _make_adapter(engine)

    msg = Message(role=Role.USER, content=[TextPart(text="What is in this image?")])
    result = adapter.generate([msg])

    engine.generate.assert_called_once()
    assert result.text == "A cat is visible."


def test_generate_raises_if_not_loaded():
    adapter = MLXVLMAdapter(model_name="fake-vlm")
    # Do NOT set adapter._engine or adapter._is_loaded
    msg = Message(role=Role.USER, content=[TextPart(text="hello")])
    with pytest.raises(RuntimeError, match="not loaded") as exc_info:
        adapter.generate([msg])
    # A missing local model cannot be fixed by retrying the call.
    assert exc_info.value.error_context["category"] == "not_loaded"
    assert exc_info.value.error_context["retry_status"] == "non_retryable"


def test_generate_forwards_config():
    engine = _fake_engine()
    adapter = _make_adapter(engine)

    msg = Message(role=Role.USER, content=[TextPart(text="test")])
    cfg = GenerationConfig(temperature=0.1, max_tokens=64)
    adapter.generate([msg], config=cfg)

    call_kwargs = engine.generate.call_args
    assert call_kwargs is not None


# ---------------------------------------------------------------------------
# generate_stream() integration
# ---------------------------------------------------------------------------

def test_generate_stream_yields_chunks():
    engine = _fake_engine("streamed response")
    adapter = _make_adapter(engine)

    msg = Message(role=Role.USER, content=[TextPart(text="Tell me a story")])
    chunks = list(adapter.generate_stream([msg]))

    assert len(chunks) >= 1
    assert "streamed response" in "".join(chunks)


def test_generate_stream_raises_if_not_loaded():
    adapter = MLXVLMAdapter(model_name="fake-vlm")
    msg = Message(role=Role.USER, content=[TextPart(text="hello")])
    with pytest.raises(RuntimeError, match="not loaded") as exc_info:
        list(adapter.generate_stream([msg]))
    assert exc_info.value.error_context["category"] == "not_loaded"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def test_count_tokens_uses_engine_tokenizer():
    engine = _fake_engine()
    engine.tokenizer.encode = MagicMock(return_value=list(range(10)))
    adapter = _make_adapter(engine)

    tc = adapter.count_tokens("hello world test")
    assert tc.count == 10
    assert tc.model_name == "fake-vlm"


def test_count_tokens_fallback_when_no_tokenizer():
    engine = _fake_engine()
    engine.tokenizer = None
    adapter = _make_adapter(engine)

    tc = adapter.count_tokens("hello world")
    # Heuristic: ≥1
    assert tc.count >= 1


def test_count_tokens_fallback_when_not_loaded():
    adapter = MLXVLMAdapter(model_name="fake-vlm")
    tc = adapter.count_tokens("hello world test")
    assert tc.count >= 1


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------

def test_get_context_length_from_engine():
    engine = _fake_engine()
    engine._context_length = 8192
    adapter = _make_adapter(engine)
    assert adapter.get_context_length() == 8192


def test_get_context_length_fallback_when_not_loaded():
    adapter = MLXVLMAdapter()
    assert adapter.get_context_length() == 4096


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_get_metadata_contains_adapter_info():
    engine = _fake_engine()
    engine._metadata = {"engine": "mlx_vlm_fake", "unified_memory_gb": 16.0}
    adapter = _make_adapter(engine)

    meta = adapter.get_metadata()
    assert meta["adapter"] == "mlx_vlm"
    assert meta["platform"] == "apple_silicon"
    assert "vision" in meta["supported_capabilities"]


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------

def test_unload_clears_engine():
    engine = _fake_engine()
    adapter = _make_adapter(engine)
    assert adapter.loaded is True

    adapter.unload()

    assert adapter._engine is None
    assert adapter._is_loaded is False
    assert adapter.loaded is False
    engine.unload.assert_called_once()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_loaded():
    adapter = _make_adapter()
    r = repr(adapter)
    assert "MLXVLMAdapter" in r
    assert "loaded" in r


def test_repr_not_loaded():
    adapter = MLXVLMAdapter(model_name="test-model")
    r = repr(adapter)
    assert "not loaded" in r
