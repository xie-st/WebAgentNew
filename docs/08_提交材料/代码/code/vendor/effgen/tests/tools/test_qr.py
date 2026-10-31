"""Tests for QRGenerateTool and QRReadTool (round-trip + error handling)."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path

import pytest

# QR tooling relies on optional dependencies (the ``qr`` extra: qrcode[pil] +
# pyzbar, which also needs the system zbar library). Skip cleanly when any are
# absent (e.g. the lean ``[dev]`` install used by the offline CI lane).
pytest.importorskip("qrcode", reason="qrcode not installed (pip install effgen[qr])")
pytest.importorskip("PIL", reason="Pillow not installed (pip install effgen[qr])")
pytest.importorskip("pyzbar", reason="pyzbar/zbar not installed (pip install effgen[qr])")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# QRGenerateTool tests
# ---------------------------------------------------------------------------

class TestQRGenerateTool:
    """Tests for QRGenerateTool."""

    def test_import(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool  # noqa: F401

    def test_basic_generate_returns_base64(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        result = run(tool._execute(operation="generate", data="https://effgen.org"))
        assert result["success"] is True
        b64 = result["qr_base64"]
        assert isinstance(b64, str) and len(b64) > 100
        # must be valid base64 PNG
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_custom_size(self):
        import io

        from PIL import Image

        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        result = run(tool._execute(operation="generate", data="test", size=300))
        assert result["success"] is True
        raw = base64.b64decode(result["qr_base64"])
        img = Image.open(io.BytesIO(raw))
        assert img.size == (300, 300)
        assert result["requested_size_px"] == 300
        assert result["size_adjusted"] is False

    def test_large_payload_round_trip_default_size(self):
        """A 1KB payload should remain readable even with the default requested size."""
        from effgen.tools.builtin.qr_generate import QRGenerateTool
        from effgen.tools.builtin.qr_read import QRReadTool

        data = "A" * 1024
        gen_tool = QRGenerateTool()
        gen_result = run(gen_tool._execute(operation="generate", data=data))
        assert gen_result["success"] is True
        assert gen_result["requested_size_px"] == 200
        assert gen_result["size_adjusted"] is True
        assert gen_result["module_count"] > 21

        read_tool = QRReadTool()
        read_result = run(read_tool._execute(operation="read", image_base64=gen_result["qr_base64"]))
        assert read_result["success"] is True
        assert data in [c["data"] for c in read_result["codes"]]

    def test_generate_data_url(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        result = run(tool._execute(operation="generate", data="hello", data_url_return=True))
        assert result["success"] is True
        assert "data_url" in result
        assert result["data_url"].startswith("data:image/png;base64,")

    def test_generate_saves_file(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out = f.name
        try:
            result = run(tool._execute(operation="generate", data="file_test", output_path=out))
            assert result["success"] is True
            assert result["saved_path"] is not None
            assert Path(out).exists()
            assert Path(out).stat().st_size > 0
        finally:
            os.unlink(out)

    def test_generate_all_error_corrections_round_trip_and_density(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool
        from effgen.tools.builtin.qr_read import QRReadTool

        data = "EFFGEN_DENSITY_" + ("0123456789" * 80)
        tool = QRGenerateTool()
        read_tool = QRReadTool()
        module_counts = []
        for ec in ("L", "M", "Q", "H"):
            result = run(
                tool._execute(operation="generate", data=data, size=700, error_correction=ec)
            )
            assert result["success"] is True, f"Failed for error_correction={ec}"
            module_counts.append(result["module_count"])
            read_result = run(read_tool._execute(operation="read", image_base64=result["qr_base64"]))
            assert read_result["success"] is True, f"Failed to decode error_correction={ec}"
            assert data in [c["data"] for c in read_result["codes"]]
        assert len(set(module_counts)) > 1

    def test_generate_invalid_error_correction(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        with pytest.raises(ValueError, match="Invalid error_correction"):
            run(tool._execute(operation="generate", data="test", error_correction="Z"))

    def test_generate_missing_data_raises(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        with pytest.raises(ValueError, match="data"):
            run(tool._execute(operation="generate", data=""))

    def test_unknown_operation_raises(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        with pytest.raises(ValueError, match="Unknown operation"):
            run(tool._execute(operation="delete", data="x"))

    def test_metadata(self):
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        assert tool.metadata.name == "qr_generate"
        ops = [p for p in tool.metadata.parameters if p.name == "operation"]
        assert ops and "generate" in (ops[0].enum or [])


# ---------------------------------------------------------------------------
# QRReadTool tests
# ---------------------------------------------------------------------------

class TestQRReadTool:
    """Tests for QRReadTool."""

    def test_import(self):
        from effgen.tools.builtin.qr_read import QRReadTool  # noqa: F401

    def _make_qr_file(self, data: str) -> str:
        """Generate a QR PNG and return the temp file path."""
        from effgen.tools.builtin.qr_generate import QRGenerateTool

        tool = QRGenerateTool()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out = f.name
        run(tool._execute(operation="generate", data=data, output_path=out))
        return out

    def test_round_trip_file(self):
        """Generate a QR and read it back via file path."""
        from effgen.tools.builtin.qr_read import QRReadTool

        data = "HELLO_EFFGEN_V0.2.5"
        path = self._make_qr_file(data)
        try:
            tool = QRReadTool()
            result = run(tool._execute(operation="read", image_path=path))
            assert result["success"] is True
            assert result["count"] >= 1
            decoded = [c["data"] for c in result["codes"]]
            assert data in decoded
        finally:
            os.unlink(path)

    def test_round_trip_base64(self):
        """Generate a QR, encode to base64, and read it back via image_base64."""
        from effgen.tools.builtin.qr_generate import QRGenerateTool
        from effgen.tools.builtin.qr_read import QRReadTool

        data = "BASE64_ROUND_TRIP"
        gen_tool = QRGenerateTool()
        gen_result = run(gen_tool._execute(operation="generate", data=data))
        b64 = gen_result["qr_base64"]

        read_tool = QRReadTool()
        result = run(read_tool._execute(operation="read", image_base64=b64))
        assert result["success"] is True
        assert result["count"] >= 1
        decoded = [c["data"] for c in result["codes"]]
        assert data in decoded

    def test_read_nonexistent_file(self):
        from effgen.tools.builtin.qr_read import QRReadTool

        tool = QRReadTool()
        result = run(tool._execute(operation="read", image_path="/nonexistent/path/qr.png"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_read_invalid_base64(self):
        from effgen.tools.builtin.qr_read import QRReadTool

        tool = QRReadTool()
        result = run(tool._execute(operation="read", image_base64="not_valid_base64!!!"))
        assert result["success"] is False

    def test_read_blank_image_returns_empty(self):
        """A plain white image should decode to zero codes (not crash)."""
        import io

        from PIL import Image

        from effgen.tools.builtin.qr_read import QRReadTool

        img = Image.new("RGB", (200, 200), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        tool = QRReadTool()
        result = run(tool._execute(operation="read", image_base64=b64))
        assert result["success"] is True
        assert result["count"] == 0

    def test_read_random_image_returns_empty(self):
        """A deterministic random-noise image should decode to zero codes (not crash)."""
        import io
        import random

        from PIL import Image

        from effgen.tools.builtin.qr_read import QRReadTool

        rng = random.Random(42)
        img = Image.new("RGB", (128, 128))
        img.putdata(
            [
                (rng.randrange(256), rng.randrange(256), rng.randrange(256))
                for _ in range(128 * 128)
            ]
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        tool = QRReadTool()
        result = run(tool._execute(operation="read", image_base64=b64))
        assert result["success"] is True
        assert result["count"] == 0

    def test_read_corrupt_file_returns_error(self):
        from effgen.tools.builtin.qr_read import QRReadTool

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not an image")
            path = f.name
        try:
            tool = QRReadTool()
            result = run(tool._execute(operation="read", image_path=path))
            assert result["success"] is False
            assert "failed to load image" in result["error"].lower()
        finally:
            os.unlink(path)

    def test_read_missing_args_raises(self):
        from effgen.tools.builtin.qr_read import QRReadTool

        tool = QRReadTool()
        with pytest.raises(ValueError, match="image_path.*image_base64"):
            run(tool._execute(operation="read"))

    def test_unknown_operation_raises(self):
        from effgen.tools.builtin.qr_read import QRReadTool

        tool = QRReadTool()
        with pytest.raises(ValueError, match="Unknown operation"):
            run(tool._execute(operation="scan"))

    def test_rect_position_present(self):
        """Decoded result includes rect dict with position fields."""
        from effgen.tools.builtin.qr_read import QRReadTool

        path = self._make_qr_file("RECT_TEST")
        try:
            tool = QRReadTool()
            result = run(tool._execute(operation="read", image_path=path))
            assert result["success"] is True and result["count"] >= 1
            rect = result["codes"][0]["rect"]
            for key in ("left", "top", "width", "height"):
                assert key in rect
        finally:
            os.unlink(path)

    def test_metadata(self):
        from effgen.tools.builtin.qr_read import QRReadTool

        tool = QRReadTool()
        assert tool.metadata.name == "qr_read"
        ops = [p for p in tool.metadata.parameters if p.name == "operation"]
        assert ops and "read" in (ops[0].enum or [])
