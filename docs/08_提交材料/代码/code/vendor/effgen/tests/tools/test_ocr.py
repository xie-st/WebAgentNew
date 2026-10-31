"""Tests for OCRTool - unit (backend selection) + integration (Tesseract round-trip)."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Pillow is an optional dependency (the ``tools`` extra); these tests build real
# PIL images, so skip cleanly when it is absent (e.g. the lean ``[dev]`` install
# used by the offline CI lane). Tesseract-backed cases are additionally gated by
# the pytesseract/binary skipif checks below.
pytest.importorskip("PIL", reason="Pillow not installed (pip install effgen[tools])")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VERIFY_TEXT = "EFFGEN_OCR_VERIFY_2026"
VERIFY_TESSERACT_CONFIG = (
    "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"
)


def _normalize_ocr_text(text: str) -> str:
    return "".join(text.upper().split())


def _make_test_image(text: str = VERIFY_TEXT) -> bytes:
    """Render known text onto a white PNG using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1200, 180), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 28), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tesseract_langs() -> set[str]:
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available languages")
    }


# ---------------------------------------------------------------------------
# Unit tests - error classes
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_system_dependency_import(self):
        from effgen.errors import MissingSystemDependency
        err = MissingSystemDependency("tesseract", "brew install tesseract")
        assert "tesseract" in str(err)
        assert "brew install tesseract" in str(err)

    def test_ocr_backend_unavailable_import(self):
        from effgen.errors import OCRBackendUnavailable
        err = OCRBackendUnavailable(tried_backends=["tesseract", "ocr.space"])
        assert "tesseract" in str(err)
        assert "ocr.space" in str(err)

    def test_ocr_backend_unavailable_default(self):
        from effgen.errors import OCRBackendUnavailable
        err = OCRBackendUnavailable()
        assert "No OCR backend" in str(err)


# ---------------------------------------------------------------------------
# Unit tests - OCRTool instantiation + backend detection
# ---------------------------------------------------------------------------

class TestOCRToolUnit:
    def test_import(self):
        from effgen.tools.builtin.ocr import OCRTool  # noqa: F401

    def test_metadata(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        assert tool.metadata.name == "ocr"
        param_names = [p.name for p in tool.metadata.parameters]
        assert "operation" in param_names
        assert "image_path" in param_names
        assert "lang" in param_names
        assert "tesseract_config" in param_names

    def test_resolve_source_path(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_test_image())
            tmp = f.name
        try:
            src = tool._resolve_source(tmp, None)
            assert src == tmp
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_resolve_source_bytes(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image()
        b64 = base64.b64encode(raw).decode()
        src = tool._resolve_source(None, b64)
        assert isinstance(src, bytes)
        assert src == raw

    def test_resolve_source_raw_bytes(self):
        from effgen.tools.builtin.ocr import OCRTool

        tool = OCRTool()
        raw = _make_test_image()
        src = tool._resolve_source(None, raw)
        assert isinstance(src, bytes)
        assert src == raw

    def test_resolve_source_data_url(self):
        from effgen.tools.builtin.ocr import OCRTool

        tool = OCRTool()
        raw = _make_test_image()
        b64 = "data:image/png;base64," + base64.b64encode(raw).decode()
        src = tool._resolve_source(None, b64)
        assert src == raw

    def test_resolve_source_both_raises(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        with pytest.raises(ValueError, match="not both"):
            tool._resolve_source("some/path", base64.b64encode(b"x").decode())

    def test_resolve_source_none_raises(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        with pytest.raises(ValueError, match="required"):
            tool._resolve_source(None, None)

    def test_no_backend_raises_ocr_backend_unavailable(self):
        from effgen.errors import OCRBackendUnavailable
        from effgen.tools.builtin.ocr import OCRTool

        tool = OCRTool()
        raw = _make_test_image()
        b64 = base64.b64encode(raw).decode()

        with patch("effgen.tools.builtin.ocr._check_tesseract", return_value=False), \
             patch("effgen.tools.builtin.ocr._ocr_space_key", return_value=None):
            with pytest.raises(OCRBackendUnavailable):
                run(tool._execute(operation="extract", image_bytes=b64))

    def test_no_backend_raises_with_install_instructions(self, monkeypatch):
        from effgen.errors import OCRBackendUnavailable
        from effgen.tools.builtin import ocr
        from effgen.tools.builtin.ocr import OCRTool

        tool = OCRTool()
        raw = _make_test_image()

        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("OCR_SPACE_API_KEY", raising=False)
        monkeypatch.setattr(ocr, "_tesseract_available", None)

        with pytest.raises(OCRBackendUnavailable) as exc_info:
            run(tool._execute(operation="extract", image_bytes=raw))
        message = str(exc_info.value)
        assert "sudo apt install tesseract-ocr" in message
        assert "brew install tesseract" in message
        assert "choco install tesseract" in message

    def test_unknown_operation_raises(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image()
        b64 = base64.b64encode(raw).decode()
        with pytest.raises(ValueError, match="Unknown operation"):
            run(tool._execute(operation="flip", image_bytes=b64))

    def test_extract_regions_no_regions_raises(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image()
        b64 = base64.b64encode(raw).decode()
        with patch("effgen.tools.builtin.ocr._check_tesseract", return_value=True):
            with pytest.raises(ValueError, match="regions"):
                run(tool._execute(operation="extract_regions", image_bytes=b64))

    def test_file_not_found(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        with pytest.raises(FileNotFoundError):
            tool._resolve_source("/nonexistent/path/to/image.png", None)


# ---------------------------------------------------------------------------
# Integration tests - Tesseract path (skipped if not installed)
# ---------------------------------------------------------------------------

_TESSERACT_AVAILABLE = False
try:
    import subprocess
    version_result = subprocess.run(
        ["tesseract", "--version"], capture_output=True, timeout=5
    )
    if version_result.returncode == 0:
        langs_result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        langs_text = (langs_result.stdout or "") + (langs_result.stderr or "")
        _TESSERACT_AVAILABLE = (
            langs_result.returncode == 0 and "eng" in langs_text.split()
        )
except Exception:
    pass

_PYTESSERACT_AVAILABLE = importlib.util.find_spec("pytesseract") is not None


@pytest.mark.skipif(
    not (_TESSERACT_AVAILABLE and _PYTESSERACT_AVAILABLE),
    reason="Tesseract or pytesseract not installed",
)
class TestOCRToolTesseract:
    def test_extract_from_bytes(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image("HELLO EFFGEN")
        b64 = base64.b64encode(raw).decode()
        result = run(
            tool._execute(
                operation="extract",
                image_bytes=b64,
                tesseract_config=VERIFY_TESSERACT_CONFIG,
            )
        )
        assert result["success"] is True
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0
        assert result["backend"] == "tesseract"
        assert isinstance(result["words"], list)

    def test_extract_from_file(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image("TESSERACT TEST")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            result = run(tool._execute(operation="extract", image_path=tmp))
            assert result["success"] is True
            assert result["backend"] == "tesseract"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_extract_text_content(self):
        """The OCR'd text should match the rendered verification string."""
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image(VERIFY_TEXT)
        b64 = base64.b64encode(raw).decode()
        result = run(tool._execute(operation="extract", image_bytes=b64))
        assert result["success"] is True
        assert _normalize_ocr_text(result["text"]) == VERIFY_TEXT.upper()

    def test_execute_accepts_raw_bytes(self):
        from effgen.tools.builtin.ocr import OCRTool

        tool = OCRTool()
        raw = _make_test_image("RAW_BYTES_OK")
        result = run(tool.execute(operation="extract", image_bytes=raw))
        assert result.success is True
        assert result.output["success"] is True
        assert "RAW" in result.output["text"].upper()

    def test_extract_regions(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image("REGION TEST")
        b64 = base64.b64encode(raw).decode()
        regions = [{"left": 0, "top": 0, "width": 400, "height": 100}]
        result = run(tool._execute(
            operation="extract_regions", image_bytes=b64, regions=regions
        ))
        assert result["success"] is True
        assert len(result["regions"]) == 1
        assert "combined_text" in result

    def test_confidence_range(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image("CONFIDENCE")
        b64 = base64.b64encode(raw).decode()
        result = run(tool._execute(operation="extract", image_bytes=b64))
        assert result["success"] is True
        conf = result["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_words_structure(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image("WORDS")
        b64 = base64.b64encode(raw).decode()
        result = run(tool._execute(operation="extract", image_bytes=b64))
        assert result["success"] is True
        for w in result["words"]:
            assert "word" in w
            assert "confidence" in w

    @pytest.mark.skipif("fra" not in _tesseract_langs(), reason="Tesseract French language pack not installed")
    def test_extract_french_language_pack(self):
        from effgen.tools.builtin.ocr import OCRTool

        tool = OCRTool()
        raw = _make_test_image("BONJOUR EFFGEN")
        result = run(tool._execute(operation="extract", image_bytes=raw, lang="fra"))
        assert result["success"] is True
        assert "BONJOUR" in result["text"].upper()


# ---------------------------------------------------------------------------
# Integration tests - OCR.space path (skipped if no key)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("OCR_SPACE_API_KEY"),
    reason="OCR_SPACE_API_KEY not set",
)
class TestOCRToolOCRSpace:
    def test_extract_via_ocr_space(self):
        from effgen.tools.builtin.ocr import OCRTool
        tool = OCRTool()
        raw = _make_test_image("OCR SPACE TEST")
        b64 = base64.b64encode(raw).decode()

        with patch("effgen.tools.builtin.ocr._check_tesseract", return_value=False):
            result = run(tool._execute(operation="extract", image_bytes=b64))
        assert result["success"] is True
        assert result["backend"] == "ocr.space"
        assert isinstance(result["text"], str)
