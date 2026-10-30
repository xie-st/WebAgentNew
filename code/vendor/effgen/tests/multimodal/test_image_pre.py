"""
Tests for effgen.multimodal.image_pre.

All tests use fake in-memory image data — no network calls, no API keys.
"""

from __future__ import annotations

import io

import pytest

from effgen.core.messages import ImagePart
from effgen.multimodal.image_pre import (
    _PROVIDER_CONSTRAINTS,
    ImagePreprocessor,
    prepare,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_png(width: int = 100, height: int = 100) -> bytes:
    """Return minimal PNG bytes for a solid-colour image."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Fallback: use a 1x1 PNG from known magic bytes
        return (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )


def _make_jpeg(width: int = 100, height: int = 100) -> bytes:
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(50, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 500


def _make_part(data: bytes, mime: str = "image/jpeg") -> ImagePart:
    return ImagePart(image=data, mime=mime)


# ---------------------------------------------------------------------------
# Tests: prepare() module-level function
# ---------------------------------------------------------------------------

class TestPrepareFunction:
    def test_prepare_noop_on_valid_jpeg(self):
        part = _make_part(_make_jpeg())
        processed = prepare(part, "openai", "gpt-4o-mini")
        assert processed.mime == "image/jpeg"
        assert processed.meta.get("preprocessing") == []

    def test_prepare_adds_preprocessing_meta(self):
        part = _make_part(_make_jpeg())
        processed = prepare(part, "groq", "qwen/qwen3.6-27b")
        assert "preprocessing" in processed.meta

    def test_prepare_unknown_provider_returns_original(self):
        data = _make_jpeg()
        part = _make_part(data)
        processed = prepare(part, "unknown_provider", "some-model")
        assert processed.image == data

    def test_prepare_does_not_mutate_original(self):
        data = _make_jpeg()
        part = _make_part(data, mime="image/jpeg")
        orig_meta = dict(part.meta)
        _ = prepare(part, "openai")
        assert part.meta == orig_meta  # original unchanged


# ---------------------------------------------------------------------------
# Tests: ImagePreprocessor
# ---------------------------------------------------------------------------

class TestImagePreprocessor:
    def test_all_providers_in_constraints_table(self):
        known = {"gemini", "openai", "groq", "together", "hf_inference", "anthropic", "fireworks"}
        assert known.issubset(_PROVIDER_CONSTRAINTS.keys())

    def test_unknown_provider_returns_unchanged(self):
        prep = ImagePreprocessor("totally_unknown")
        data = _make_jpeg()
        part = _make_part(data)
        result = prep.prepare(part)
        assert result.image == data

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("PIL"),
        reason="Pillow not installed",
    )
    def test_resize_large_image(self):
        data = _make_jpeg(4000, 3000)
        part = _make_part(data)
        prep = ImagePreprocessor("groq")
        result = prep.prepare(part)
        # Groq max is 1568px; image should be smaller after resize
        assert len(result.image) <= len(data)
        assert "resized" in " ".join(result.meta.get("preprocessing", []))

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("PIL"),
        reason="Pillow not installed",
    )
    def test_mime_conversion_gif_to_jpeg(self):
        from PIL import Image
        img = Image.new("RGB", (50, 50), color=(100, 200, 50))
        buf = io.BytesIO()
        img.save(buf, format="GIF")
        gif_data = buf.getvalue()
        # Groq doesn't allow GIF — should be converted
        part = ImagePart(image=gif_data, mime="image/gif")
        prep = ImagePreprocessor("groq")
        result = prep.prepare(part)
        assert result.mime in ("image/jpeg", "image/png", "image/webp")
        assert any("mime_converted" in s for s in result.meta.get("preprocessing", []))

    def test_provider_constraints_byte_limits(self):
        for provider, constraints in _PROVIDER_CONSTRAINTS.items():
            assert constraints["max_bytes"] > 0, f"{provider} max_bytes must be positive"
            assert constraints["max_px"] > 0, f"{provider} max_px must be positive"
            assert len(constraints["allowed_mimes"]) > 0, f"{provider} must allow at least one MIME"

    def test_preprocessing_recorded_in_meta(self):
        pytest.importorskip("PIL")
        # Create oversized image for Groq (max 1568px)
        data = _make_jpeg(2000, 2000)
        part = _make_part(data)
        result = ImagePreprocessor("groq").prepare(part)
        assert "preprocessing" in result.meta

    def test_gemini_allows_gif(self):
        # Gemini supports GIF — no conversion expected
        constraints = _PROVIDER_CONSTRAINTS["gemini"]
        assert "image/gif" in constraints["allowed_mimes"]

    def test_groq_does_not_allow_gif(self):
        constraints = _PROVIDER_CONSTRAINTS["groq"]
        assert "image/gif" not in constraints["allowed_mimes"]
