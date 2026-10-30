"""
ImageInfoTool — PIL-based image metadata extraction and basic manipulation.

Zero network; Pillow only. Accepts file paths OR raw bytes.

Operations
----------
- info        : extract size/format/mode/EXIF/color histogram summary
- resize      : resize an image and return bytes (or save to path)
- thumbnail   : fit within max_dim × max_dim, return bytes (or save to path)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._fs import confine_path, normalize_allowed_dirs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(source: str | bytes | Path) -> "tuple[Image.Image, str]":  # noqa: F821
    """Return (image_copy, format_str) — format is captured before copy loses it."""
    from PIL import Image
    if isinstance(source, str | Path):
        with Image.open(str(source)) as img:
            fmt = img.format or Path(str(source)).suffix.lstrip(".").upper() or "unknown"
            return img.copy(), fmt
    if isinstance(source, bytes | bytearray):
        with Image.open(io.BytesIO(bytes(source))) as img:
            return img.copy(), img.format or "unknown"
    raise TypeError(f"Expected str, Path, or bytes; got {type(source)}")


def _exif_dict(img: "Image.Image") -> dict[str, Any]:  # noqa: F821
    """Extract EXIF data as a plain dict (str keys → str/int values)."""
    exif: dict[str, Any] = {}
    try:
        from PIL.ExifTags import TAGS
        raw = img.getexif()
        if raw:
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, str(tag_id))
                # Keep only scalar/string values — skip raw bytes blobs
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = repr(value)
                exif[tag] = value
    except Exception as exc:
        logger.debug("EXIF extraction failed: %s", exc)
    return exif


def _histogram_summary(img: "Image.Image") -> dict[str, Any]:  # noqa: F821
    """Return per-channel mean brightness (0-255)."""
    try:
        from PIL import ImageStat
        stat = ImageStat.Stat(img.convert("RGB"))
        return {
            "mean_r": round(stat.mean[0], 2),
            "mean_g": round(stat.mean[1], 2),
            "mean_b": round(stat.mean[2], 2),
            "stddev_r": round(stat.stddev[0], 2),
            "stddev_g": round(stat.stddev[1], 2),
            "stddev_b": round(stat.stddev[2], 2),
        }
    except Exception as exc:
        logger.debug("Histogram failed: %s", exc)
        return {}


def _prepare_for_save(img: "Image.Image", fmt: str) -> "Image.Image":  # noqa: F821
    """Convert modes that Pillow cannot save in common output formats."""
    from PIL import Image

    save_fmt = fmt.upper()
    if save_fmt in {"JPEG", "JPG"} and img.mode in {"RGBA", "LA", "P"}:
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        alpha = img.getchannel("A") if img.mode in {"RGBA", "LA"} else None
        background.paste(img.convert("RGB"), mask=alpha)
        return background
    if save_fmt == "WEBP" and img.mode == "P":
        return img.convert("RGBA")
    return img


def _info(source: str | bytes | Path) -> dict[str, Any]:
    img, fmt = _load_image(source)
    exif = _exif_dict(img)
    hist = _histogram_summary(img)
    data = {
        "width": img.width,
        "height": img.height,
        "format": fmt,
        "mode": img.mode,
        "n_frames": getattr(img, "n_frames", 1),
        "exif": exif,
        "color_stats": hist,
    }
    return {"success": True, "data": data, "error": None}


def _resize(
    source: str | bytes | Path,
    width: int,
    height: int,
    dest: str | None = None,
    fmt: str = "PNG",
) -> dict[str, Any]:
    from PIL import Image
    img, _ = _load_image(source)
    resized = img.resize((width, height), Image.LANCZOS)
    resized = _prepare_for_save(resized, fmt)
    buf = io.BytesIO()
    save_fmt = fmt.upper()
    resized.save(buf, format=save_fmt)
    img_bytes = buf.getvalue()
    if dest:
        Path(dest).write_bytes(img_bytes)
    return {
        "success": True,
        "data": {
            "width": width,
            "height": height,
            "format": save_fmt,
            "saved_to": dest,
            "bytes_b64": base64.b64encode(img_bytes).decode() if not dest else None,
        },
        "error": None,
    }


def _thumbnail(
    source: str | bytes | Path,
    max_dim: int,
    dest: str | None = None,
    fmt: str = "PNG",
) -> dict[str, Any]:
    from PIL import Image
    img, _ = _load_image(source)
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    img = _prepare_for_save(img, fmt)
    buf = io.BytesIO()
    save_fmt = fmt.upper()
    img.save(buf, format=save_fmt)
    img_bytes = buf.getvalue()
    if dest:
        Path(dest).write_bytes(img_bytes)
    return {
        "success": True,
        "data": {
            "width": img.width,
            "height": img.height,
            "format": save_fmt,
            "saved_to": dest,
            "bytes_b64": base64.b64encode(img_bytes).decode() if not dest else None,
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class ImageInfoTool(BaseTool):
    """
    Extract metadata and perform basic transformations on images.

    Zero-network; Pillow only. Accepts file paths OR raw bytes.

    Operations:
      - info      : size, format, mode, EXIF, color stats
      - resize    : resize to exact dimensions
      - thumbnail : fit within max_dim × max_dim (preserves aspect ratio)
    """

    def __init__(self, allowed_directories: list[str] | None = None) -> None:
        """Args:
            allowed_directories: Roots a file path may be read from. By default
                any path is allowed except protected system and credential
                locations (/etc, /proc, ~/.ssh, cloud creds, …), which are
                always refused. Pass a list to confine reads to those roots only.
        """
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)
        super().__init__(
            metadata=ToolMetadata(
                name="image_info",
                description=(
                    "Analyze images locally with Pillow — no network required. "
                    "Operations: info (metadata + EXIF + color stats), "
                    "resize (exact dimensions), thumbnail (aspect-preserving fit). "
                    "Accepts file paths or raw image bytes."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'info', 'resize', or 'thumbnail'.",
                        required=True,
                        enum=["info", "resize", "thumbnail"],
                    ),
                    ParameterSpec(
                        name="image_path",
                        type=ParameterType.STRING,
                        description="Path to the image file. Mutually exclusive with image_bytes.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="image_bytes",
                        type=ParameterType.STRING,
                        description=(
                            "Base64-encoded image string. "
                            "Mutually exclusive with image_path."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="width",
                        type=ParameterType.INTEGER,
                        description="Target width in pixels (for resize).",
                        required=False,
                    ),
                    ParameterSpec(
                        name="height",
                        type=ParameterType.INTEGER,
                        description="Target height in pixels (for resize).",
                        required=False,
                    ),
                    ParameterSpec(
                        name="max_dim",
                        type=ParameterType.INTEGER,
                        description="Max side length in pixels (for thumbnail).",
                        required=False,
                    ),
                    ParameterSpec(
                        name="dest",
                        type=ParameterType.STRING,
                        description="Optional output file path. If omitted, bytes are returned base64-encoded.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="format",
                        type=ParameterType.STRING,
                        description="Output format for resize/thumbnail ('PNG', 'JPEG', 'WEBP'). Default: PNG.",
                        required=False,
                        default="PNG",
                    ),
                ],
                timeout_seconds=30,
                tags=["image", "metadata", "exif", "resize", "pillow", "vision"],
                examples=[
                    {"operation": "info", "image_path": "/path/to/photo.jpg"},
                    {"operation": "resize", "image_path": "/path/to/img.png", "width": 800, "height": 600},
                    {"operation": "thumbnail", "image_path": "/path/to/img.png", "max_dim": 256},
                ],
            )
        )

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        """Validate parameters, letting raw ``image_bytes`` pass the string check."""
        if isinstance(kwargs.get("image_bytes"), bytes | bytearray):
            kwargs = dict(kwargs)
            kwargs["image_bytes"] = "__raw_bytes__"
        return super().validate_parameters(**kwargs)

    def _resolve_source(
        self,
        image_path: str | None,
        image_bytes: str | bytes | bytearray | None,
    ) -> str | bytes:
        if image_path and image_bytes:
            raise ValueError("Provide image_path OR image_bytes, not both.")
        if image_path:
            return str(confine_path(image_path, self._allowed_dirs))
        if image_bytes is not None:
            if isinstance(image_bytes, bytes | bytearray):
                return bytes(image_bytes)
            encoded = image_bytes.strip()
            if encoded.startswith("data:image/") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            import binascii
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image_bytes must be raw bytes or a base64-encoded string.") from exc
        raise ValueError("Either image_path or image_bytes is required.")

    async def _execute(
        self,
        operation: str,
        image_path: str | None = None,
        image_bytes: str | bytes | bytearray | None = None,
        width: int | None = None,
        height: int | None = None,
        max_dim: int | None = None,
        dest: str | None = None,
        format: str = "PNG",  # noqa: A002
        **kwargs: Any,
    ) -> dict[str, Any]:
        source = self._resolve_source(image_path, image_bytes)
        op = operation.lower()

        if op == "info":
            return await asyncio.to_thread(_info, source)

        elif op == "resize":
            if width is None or height is None:
                raise ValueError("'resize' requires both width and height.")
            return await asyncio.to_thread(_resize, source, int(width), int(height), dest, format)

        elif op == "thumbnail":
            if max_dim is None:
                raise ValueError("'thumbnail' requires max_dim.")
            return await asyncio.to_thread(_thumbnail, source, int(max_dim), dest, format)

        else:
            raise ValueError(f"Unknown operation: {operation!r}. Use 'info', 'resize', or 'thumbnail'.")
