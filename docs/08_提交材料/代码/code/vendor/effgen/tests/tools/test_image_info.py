"""
Tests for ImageInfoTool.

Unit: PIL-based metadata extraction, resize, thumbnail — zero network.
Integration: uses a real PIL-generated test image.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import pytest

# Pillow is an optional dependency (the ``tools`` extra); these tests build real
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
    """Generate a 200×100 solid-colour PNG in memory."""
    from PIL import Image
    img = Image.new("RGB", (200, 100), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_png_path(tmp_path: Path, sample_png_bytes: bytes) -> str:
    p = tmp_path / "test_image.png"
    p.write_bytes(sample_png_bytes)
    return str(p)


# ---------------------------------------------------------------------------
# Unit tests — ImageInfoTool instantiation
# ---------------------------------------------------------------------------

def test_image_info_tool_instantiation():
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    assert tool.metadata.name == "image_info"
    assert tool.metadata.timeout_seconds == 30
    assert "image" in tool.metadata.tags


def test_image_info_tool_parameters():
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    param_names = {p.name for p in tool.metadata.parameters}
    assert {"operation", "image_path", "image_bytes", "width", "height", "max_dim"} <= param_names


# ---------------------------------------------------------------------------
# Integration tests — info operation
# ---------------------------------------------------------------------------

def test_info_from_path(sample_png_path: str):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    result = run(tool._execute(operation="info", image_path=sample_png_path))
    assert result["success"] is True
    data = result["data"]
    assert data["width"] == 200
    assert data["height"] == 100
    assert data["mode"] == "RGB"
    # format may be None when opened from bytes but path should give PNG
    assert "color_stats" in data


def test_info_from_bytes(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    result = run(tool._execute(operation="info", image_bytes=sample_png_bytes))
    assert result["success"] is True
    assert result["data"]["width"] == 200
    assert result["data"]["height"] == 100


def test_info_from_base64(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    b64 = base64.b64encode(sample_png_bytes).decode()
    result = run(tool._execute(operation="info", image_bytes=b64))
    assert result["success"] is True
    assert result["data"]["width"] == 200


def test_info_color_stats(sample_png_bytes: bytes):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    result = run(tool._execute(operation="info", image_bytes=sample_png_bytes))
    stats = result["data"]["color_stats"]
    # Solid colour image — stddev should be ~0
    assert stats["stddev_r"] < 1.0
    assert stats["stddev_g"] < 1.0
    assert stats["stddev_b"] < 1.0
    # Mean should match our colour (120, 80, 40) within rounding
    assert abs(stats["mean_r"] - 120) < 5
    assert abs(stats["mean_g"] - 80) < 5
    assert abs(stats["mean_b"] - 40) < 5


# ---------------------------------------------------------------------------
# Integration tests — resize operation
# ---------------------------------------------------------------------------

def test_resize_to_bytes(sample_png_path: str):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    result = run(tool._execute(operation="resize", image_path=sample_png_path, width=50, height=30))
    assert result["success"] is True
    data = result["data"]
    assert data["width"] == 50
    assert data["height"] == 30
    # bytes_b64 returned when no dest
    assert data["bytes_b64"] is not None
    # Decode and verify
    img_bytes = base64.b64decode(data["bytes_b64"])
    from PIL import Image
    with Image.open(io.BytesIO(img_bytes)) as img:
        assert img.width == 50
        assert img.height == 30


def test_resize_to_file(sample_png_path: str, tmp_path: Path):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    dest = str(tmp_path / "resized.png")
    result = run(tool._execute(operation="resize", image_path=sample_png_path, width=40, height=40, dest=dest))
    assert result["success"] is True
    assert Path(dest).exists()
    from PIL import Image
    with Image.open(dest) as img:
        assert img.width == 40


def test_resize_rgba_to_jpeg(tmp_path: Path):
    from PIL import Image

    from effgen.tools.builtin.image_info import ImageInfoTool

    src = tmp_path / "transparent.png"
    img = Image.new("RGBA", (20, 10), color=(255, 0, 0, 128))
    img.save(src, format="PNG")

    tool = ImageInfoTool()
    result = run(
        tool._execute(
            operation="resize",
            image_path=str(src),
            width=10,
            height=5,
            format="JPEG",
        )
    )
    assert result["success"] is True
    decoded = base64.b64decode(result["data"]["bytes_b64"])
    with Image.open(io.BytesIO(decoded)) as out:
        assert out.format == "JPEG"
        assert out.mode == "RGB"
        assert out.size == (10, 5)


def test_resize_missing_dimensions_error(sample_png_path: str):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    with pytest.raises(ValueError, match="width and height"):
        run(tool._execute(operation="resize", image_path=sample_png_path))


# ---------------------------------------------------------------------------
# Integration tests — thumbnail operation
# ---------------------------------------------------------------------------

def test_thumbnail_preserves_aspect(sample_png_path: str):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    result = run(tool._execute(operation="thumbnail", image_path=sample_png_path, max_dim=80))
    assert result["success"] is True
    data = result["data"]
    # 200×100 → max_dim=80 → 80×40
    assert data["width"] <= 80
    assert data["height"] <= 80
    # Aspect ratio preserved (2:1)
    assert data["width"] == data["height"] * 2


def test_thumbnail_missing_max_dim_error(sample_png_path: str):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    with pytest.raises(ValueError, match="max_dim"):
        run(tool._execute(operation="thumbnail", image_path=sample_png_path))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_invalid_path_raises():
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    with pytest.raises(FileNotFoundError):
        run(tool._execute(operation="info", image_path="/nonexistent/path/image.png"))


def test_both_source_raises(sample_png_path: str, sample_png_bytes: bytes):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    with pytest.raises(ValueError, match="not both"):
        run(tool._execute(operation="info", image_path=sample_png_path, image_bytes=sample_png_bytes))


def test_no_source_raises():
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    with pytest.raises(ValueError, match="required"):
        run(tool._execute(operation="info"))


def test_unknown_operation_raises(sample_png_path: str):
    from effgen.tools.builtin.image_info import ImageInfoTool
    tool = ImageInfoTool()
    with pytest.raises(ValueError, match="Unknown operation"):
        run(tool._execute(operation="blur", image_path=sample_png_path))
