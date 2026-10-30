"""
Tests for ImageCaptionTool.

Unit: provider selection logic, error handling, parameter validation.
Integration: live caption via available vision provider (skipped if no key).
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Pillow is an optional dependency (image/vision extras); these tests build real
# PIL images, so skip cleanly when it is absent (e.g. the lean ``[dev]`` install
# used by the offline CI lane).
pytest.importorskip("PIL", reason="Pillow not installed (pip install effgen[tools])")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_png_bytes() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_png_path(tmp_path: Path, sample_png_bytes: bytes) -> str:
    p = tmp_path / "test_caption.png"
    p.write_bytes(sample_png_bytes)
    return str(p)


# ---------------------------------------------------------------------------
# Unit — instantiation
# ---------------------------------------------------------------------------

def test_image_caption_tool_instantiation():
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    assert tool.metadata.name == "image_caption"
    assert "vision" in tool.metadata.tags
    assert tool._forced_provider is None
    assert tool._forced_model is None


def test_image_caption_tool_forced_provider():
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool(provider="openai", model_id="gpt-4o-mini")
    assert tool._forced_provider == "openai"
    assert tool._forced_model == "gpt-4o-mini"


def test_image_caption_tool_parameters():
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    param_names = {p.name for p in tool.metadata.parameters}
    assert {"operation", "image_path", "image_bytes", "prompt", "detail"} <= param_names


# ---------------------------------------------------------------------------
# Unit — NoVisionProviderAvailable is raised when no key is available
# ---------------------------------------------------------------------------

def test_no_vision_provider_raises():
    from effgen.errors import NoVisionProviderAvailable
    from effgen.tools.builtin.image_caption import _pick_vision_provider

    # Patch ProviderRegistry so that no provider has a key
    with patch("effgen.tools.builtin.image_caption.os.environ.get", return_value=None):
        # is_available checks env keys
        from effgen.models.registry import ProviderRegistry
        orig = ProviderRegistry.is_available
        ProviderRegistry.is_available = classmethod(lambda cls, p: False)
        try:
            with pytest.raises(NoVisionProviderAvailable):
                _pick_vision_provider()
        finally:
            ProviderRegistry.is_available = orig


def test_no_vision_provider_error_message():
    from effgen.errors import NoVisionProviderAvailable
    err = NoVisionProviderAvailable(tried_providers=["openai(no key)", "gemini(no key)"])
    assert "openai" in str(err)
    assert "OPENAI_API_KEY" in str(err)


def test_pick_provider_override_uses_default_model():
    from effgen.tools.builtin.image_caption import _pick_vision_provider

    adapter_cls = MagicMock
    calls = []

    def fake_check(provider_name: str, model_id: str):
        calls.append((provider_name, model_id))
        return adapter_cls, None

    with patch("effgen.tools.builtin.image_caption._ensure_provider_registry"), \
         patch("effgen.tools.builtin.image_caption._check_vision_candidate", side_effect=fake_check):
        provider, model, chosen_adapter = _pick_vision_provider(provider_name="gemini")

    assert provider == "gemini"
    assert model == "gemini-3.1-flash-lite"
    assert chosen_adapter is adapter_cls
    assert calls[0] == ("gemini", "gemini-3.1-flash-lite")


def test_forced_provider_missing_key_raises_clean_error():
    from effgen.errors import NoVisionProviderAvailable
    from effgen.tools.builtin.image_caption import _pick_vision_provider

    with patch("effgen.tools.builtin.image_caption._ensure_provider_registry"), \
         patch(
             "effgen.tools.builtin.image_caption._check_vision_candidate",
             return_value=(None, "gemini(missing GOOGLE_API_KEY)"),
         ):
        with pytest.raises(NoVisionProviderAvailable, match="GOOGLE_API_KEY"):
            _pick_vision_provider(
                provider_name="gemini",
                model_id="gemini-2.5-flash-lite",
            )


# ---------------------------------------------------------------------------
# Unit — _image_to_base64_url
# ---------------------------------------------------------------------------

def test_image_to_base64_url_from_bytes(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_caption import _image_to_base64_url
    url = _image_to_base64_url(sample_png_bytes)
    assert url.startswith("data:image/png;base64,")
    # Decode and verify it's valid base64
    b64_part = url.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == sample_png_bytes


def test_image_to_base64_url_from_path(sample_png_path: str, sample_png_bytes: bytes):
    from effgen.tools.builtin.image_caption import _image_to_base64_url
    url = _image_to_base64_url(sample_png_path)
    assert url.startswith("data:image/png;base64,")
    b64_part = url.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == sample_png_bytes


# ---------------------------------------------------------------------------
# Unit — operation prompts
# ---------------------------------------------------------------------------

def test_caption_operation_prompt_override():
    """caption operation with default prompt should use concise one-sentence prompt."""
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    # We can't easily test the actual prompt without calling the API,
    # but we can verify the tool accepts the operation parameter.
    assert "caption" in [e for p in tool.metadata.parameters if p.name == "operation" for e in (p.enum or [])]


def test_describe_operation_in_enum():
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    ops = next(p.enum for p in tool.metadata.parameters if p.name == "operation")
    assert "describe" in ops


# ---------------------------------------------------------------------------
# Unit — resolve_source
# ---------------------------------------------------------------------------

def test_resolve_source_path(sample_png_path: str):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    src = tool._resolve_source(sample_png_path, None)
    assert isinstance(src, str)
    assert src == sample_png_path


def test_resolve_source_bytes(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    src = tool._resolve_source(None, sample_png_bytes)
    assert isinstance(src, bytes)
    assert src == sample_png_bytes


def test_resolve_source_base64(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    b64 = base64.b64encode(sample_png_bytes).decode()
    src = tool._resolve_source(None, b64)
    assert isinstance(src, bytes)
    assert src == sample_png_bytes


def test_resolve_source_both_raises(sample_png_path: str, sample_png_bytes: bytes):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    with pytest.raises(ValueError, match="not both"):
        tool._resolve_source(sample_png_path, sample_png_bytes)


def test_resolve_source_neither_raises():
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    with pytest.raises(ValueError, match="required"):
        tool._resolve_source(None, None)


def test_resolve_source_missing_file_raises():
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    with pytest.raises(FileNotFoundError):
        tool._resolve_source("/no/such/file.png", None)


# ---------------------------------------------------------------------------
# Integration — live caption (skipped if no vision provider key present)
# ---------------------------------------------------------------------------

def _has_vision_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _caption(tool, **kwargs) -> dict:
    """Run one captioning call, reporting a spent free-tier quota as a skip.

    Every vision provider here throttles on a daily budget. Once that budget is
    gone there is no caption to assert on, and the throttle says nothing about
    the captioning path.
    """
    from effgen.models.errors import classify_provider_error

    try:
        return run(tool._execute(**kwargs))
    except Exception as exc:  # noqa: BLE001 - a throttle is not a caption defect
        if classify_provider_error(exc).category == "rate_limited":
            pytest.skip(f"vision provider rate limited this call: {str(exc)[:200]}")
        raise


@pytest.mark.skipif(not _has_vision_key(), reason="No vision provider API key configured")
def test_live_caption(sample_png_path: str):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    result = _caption(tool, operation="caption", image_path=sample_png_path)
    assert result["success"] is True
    assert isinstance(result["caption"], str)
    assert len(result["caption"]) > 5
    assert result["provider"] in ("openai", "gemini", "replicate")
    print(f"\nLive caption ({result['provider']}/{result['model']}): {result['caption']}")


@pytest.mark.skipif(not _has_vision_key(), reason="No vision provider API key configured")
def test_live_describe(sample_png_path: str):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    result = _caption(tool, operation="describe", image_path=sample_png_path, detail="low")
    assert result["success"] is True
    assert len(result["caption"]) > 10


@pytest.mark.skipif(not _has_vision_key(), reason="No vision provider API key configured")
def test_live_caption_from_bytes(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_caption import ImageCaptionTool
    tool = ImageCaptionTool()
    result = _caption(tool, operation="caption", image_bytes=sample_png_bytes)
    assert result["success"] is True
    assert isinstance(result["caption"], str)
