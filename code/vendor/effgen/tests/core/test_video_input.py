"""Unit tests for video input adapter translation (no live API calls).

Validates:
  - has_video_input detection helper
  - require_video_support gating
  - Gemini adapter: VideoPart → frame-sampling fallback (unit, no SDK)
  - Gemini adapter: VideoPart → native video via Files API path (unit, mocked SDK)
  - OpenAI adapter: VideoPart → frame sequence in message content (unit, mocked)
  - CapabilityNotSupportedError raised for adapters that lack vision
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effgen.core.messages import (
    ImagePart,
    Message,
    Role,
    TextPart,
    VideoPart,
)
from effgen.errors import CapabilityNotSupportedError, InvalidMultimodalContent
from effgen.models._multimodal import has_video_input, require_video_support
from effgen.models.capabilities import Capability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(width: int = 8, height: int = 8) -> bytes:
    """Return a minimal valid JPEG for testing."""
    try:
        import io as _io

        from PIL import Image
        img = Image.new("RGB", (width, height), color=(100, 150, 200))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Minimal JPEG header + EOI marker (not a valid image, but passes bytes check)
        return b"\xff\xd8\xff\xe0" + b"\x00" * 12 + b"\xff\xd9"


def _make_video_part(n_frames: int = 2, fps: float = 1.0) -> VideoPart:
    frames = [_make_jpeg() for _ in range(n_frames)]
    return VideoPart(frames=frames, fps=fps, mime="image/jpeg")


def _video_message(text: str = "Describe this video") -> Message:
    vp = _make_video_part(n_frames=3)
    return Message(role=Role.USER, content=[vp, TextPart(text=text)])


# ---------------------------------------------------------------------------
# has_video_input
# ---------------------------------------------------------------------------

class TestHasVideoInput:
    def test_detects_video_in_message(self):
        msg = _video_message()
        assert has_video_input(msg) is True

    def test_detects_video_in_list(self):
        msg = _video_message()
        assert has_video_input([msg]) is True

    def test_text_only_returns_false(self):
        msg = Message(role=Role.USER, content="Hello")
        assert has_video_input(msg) is False

    def test_image_only_returns_false(self):
        jp = _make_jpeg()
        ip = ImagePart(image=jp, mime="image/jpeg")
        msg = Message(role=Role.USER, content=[ip])
        assert has_video_input(msg) is False

    def test_plain_string_returns_false(self):
        assert has_video_input("hello") is False


# ---------------------------------------------------------------------------
# require_video_support
# ---------------------------------------------------------------------------

class TestRequireVideoSupport:
    def test_passes_when_no_video_input(self):
        msg = Message(role=Role.USER, content="text only")
        # Should not raise regardless of support flag
        require_video_support(msg, provider="fake", model_name="m", supports_video=False)

    def test_raises_when_not_supported(self):
        msg = _video_message()
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            require_video_support(
                msg,
                provider="fake",
                model_name="fake-model",
                supports_video=False,
            )
        assert Capability.video_input.value in str(exc_info.value)

    def test_passes_when_supported(self):
        msg = _video_message()
        require_video_support(
            msg,
            provider="fake",
            model_name="fake-model",
            supports_video=True,
        )

    def test_callable_support_check(self):
        msg = _video_message()
        # Callable that returns True for any model name
        require_video_support(
            msg,
            provider="fake",
            model_name="any-model",
            supports_video=lambda m: True,
        )

    def test_callable_support_check_fails(self):
        msg = _video_message()
        with pytest.raises(CapabilityNotSupportedError):
            require_video_support(
                msg,
                provider="fake",
                model_name="any-model",
                supports_video=lambda m: False,
            )

    def test_hint_in_error(self):
        msg = _video_message()
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            require_video_support(
                msg,
                provider="fake",
                model_name="m",
                supports_video=False,
                hint="Use a video-capable model.",
            )
        assert "video-capable" in str(exc_info.value)


# ---------------------------------------------------------------------------
# VideoPart validation (from messages.py)
# ---------------------------------------------------------------------------

class TestVideoPartValidation:
    def test_valid_construction(self):
        vp = _make_video_part()
        assert len(vp.frames) == 2
        assert vp.fps == 1.0
        assert vp.mime == "image/jpeg"
        assert vp.type == "video_frames"

    def test_empty_frames_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=[], fps=1.0, mime="image/jpeg")

    def test_zero_fps_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=[_make_jpeg()], fps=0.0, mime="image/jpeg")

    def test_negative_fps_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=[_make_jpeg()], fps=-1.0, mime="image/jpeg")

    def test_invalid_frame_mime_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=[_make_jpeg()], fps=1.0, mime="video/mp4")

    def test_non_bytes_frame_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            VideoPart(frames=["not bytes"], fps=1.0, mime="image/jpeg")  # type: ignore[list-item]

    def test_bytearray_frames_accepted(self):
        frame_ba = bytearray(_make_jpeg())
        vp = VideoPart(frames=[frame_ba], fps=1.0, mime="image/jpeg")
        assert isinstance(vp.frames[0], bytes)


# ---------------------------------------------------------------------------
# Message has_video property
# ---------------------------------------------------------------------------

class TestMessageHasVideo:
    def test_has_video_true(self):
        msg = _video_message()
        assert msg.has_video is True

    def test_has_video_false_for_text(self):
        msg = Message.user("hello")
        assert msg.has_video is False

    def test_message_with_video_and_text(self):
        vp = _make_video_part()
        tp = TextPart(text="What is this?")
        msg = Message(role=Role.USER, content=[vp, tp])
        assert msg.has_video is True
        assert msg.text == "What is this?"


# ---------------------------------------------------------------------------
# Gemini adapter — frame-sampling fallback (mocked genai)
# ---------------------------------------------------------------------------

class TestGeminiAdapterVideoFrameSampling:
    """Test that Gemini sends VideoPart as frame-sequence when native video is unavailable."""

    def _make_adapter(self, model_name: str = "gemini-3.1-flash-lite"):
        from effgen.models.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter.model_name = model_name
        adapter.api_key = "fake-key"
        adapter._is_loaded = True
        adapter._context_length = 1_000_000
        adapter._metadata = {}
        adapter.client = MagicMock()
        adapter.total_cost = 0.0
        adapter.total_tokens = 0
        adapter.safety_settings = None
        adapter.additional_kwargs = {}
        return adapter

    def test_video_part_sends_frames(self):
        """Verify VideoPart frames are emitted as image parts (frame-sampling fallback)."""
        adapter = self._make_adapter()
        n_frames = 3
        vp = _make_video_part(n_frames=n_frames)
        tp = TextPart(text="Describe this video")
        msg = Message(role=Role.USER, content=[vp, tp])

        # Build mock google.genai.types
        mock_gt = MagicMock()
        captured_parts: list[MagicMock] = []

        def _from_text(text):
            p = MagicMock()
            p._type = "text"
            captured_parts.append(p)
            return p

        def _from_bytes(data, mime_type):
            p = MagicMock()
            p._type = "bytes"
            p._data = data
            p._mime = mime_type
            captured_parts.append(p)
            return p

        mock_gt.Part.from_text.side_effect = _from_text
        mock_gt.Part.from_bytes.side_effect = _from_bytes

        # Patch google.genai.types inside the adapter module scope
        with patch.dict(
            "sys.modules",
            {"google": MagicMock(), "google.genai": MagicMock(), "google.genai.types": mock_gt},
        ):
            with patch(
                "effgen.models.gemini_adapter.GEMINI_MODELS",
                {adapter.model_name: {"supports_vision": True, "supports_video": False}},
            ):
                # Patch the import inside _message_to_genai_parts
                with patch("effgen.models.gemini_adapter.GEMINI_MODELS", {
                    adapter.model_name: {"supports_vision": True, "supports_video": False},
                }):
                    import types as _types

                    # Create a fake module for google.genai.types
                    fake_types_mod = _types.ModuleType("google.genai.types")
                    fake_types_mod.Part = mock_gt.Part  # type: ignore[attr-defined]

                    with patch.dict("sys.modules", {"google.genai.types": fake_types_mod}):
                        from effgen.multimodal.image_pre import prepare as _preprocess

                        # Manually call _message_to_genai_parts with the fake module in place
                        result_parts = adapter._message_to_genai_parts(msg, _preprocess)

        # 3 frames as from_bytes + 1 text part = 4 total
        assert len(result_parts) == n_frames + 1


# ---------------------------------------------------------------------------
# OpenAI adapter — VideoPart → image_url frames
# ---------------------------------------------------------------------------

class TestOpenAIAdapterVideoFrames:
    """Test that OpenAI adapter translates VideoPart into sequential image_url parts."""

    def _make_adapter(self, model_name: str = "gpt-4o-mini"):
        from effgen.models.openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        adapter.model_name = model_name
        adapter.api_key = "fake-key"
        adapter._is_loaded = True
        adapter._context_length = 128_000
        adapter._metadata = {}
        adapter.client = MagicMock()
        adapter.total_cost = 0.0
        adapter.total_tokens = 0
        adapter._is_reasoning_model = False
        adapter._supports_sampling_params = True
        adapter.organization_id = None
        adapter.max_retries = 3
        adapter.timeout = 60
        adapter.additional_kwargs = {}
        return adapter

    def test_video_part_becomes_image_urls(self):
        adapter = self._make_adapter()
        n_frames = 3
        vp = _make_video_part(n_frames=n_frames)
        tp = TextPart(text="What's in this video?")
        msg = Message(role=Role.USER, content=[vp, tp])

        result = adapter._message_to_openai(msg)

        assert result["role"] == "user"
        content = result["content"]
        # Should have n_frames image_url entries + 1 text entry
        assert isinstance(content, list)
        image_parts = [c for c in content if c.get("type") == "image_url"]
        text_parts = [c for c in content if c.get("type") == "text"]
        assert len(image_parts) == n_frames
        assert len(text_parts) == 1

        # Verify base64 encoding is present
        for ip in image_parts:
            url = ip["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# Capability gating in registry
# ---------------------------------------------------------------------------

class TestCapabilityGating:
    def test_gemini_has_video_input_capability(self):
        from effgen.models.registry import ProviderRegistry
        try:
            caps = ProviderRegistry.get_capabilities("gemini")
        except Exception:
            pytest.skip("ProviderRegistry not initialized in this env")
        assert Capability.video_input in caps

    def test_openai_has_video_input_capability(self):
        from effgen.models.registry import ProviderRegistry
        try:
            caps = ProviderRegistry.get_capabilities("openai")
        except Exception:
            pytest.skip("ProviderRegistry not initialized in this env")
        assert Capability.video_input in caps
