"""
QR code generation tool for the effGen framework.

Backend: qrcode + Pillow (local, zero network).

Operations:
- generate: Generate a QR code from data, returning a base64-encoded PNG or
  saving to a file path.
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

logger = logging.getLogger(__name__)

_VALID_ERROR_CORRECTIONS = ("L", "M", "Q", "H")
_MIN_SIZE_PX = 64
_MAX_SIZE_PX = 4096
_MIN_PIXELS_PER_MODULE = 10
_BORDER_MODULES = 4


def _get_error_correction(level: str):
    """Return the qrcode error-correction constant for a level string."""
    import qrcode

    mapping = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    return mapping[level.upper()]


def _generate_qr(
    data: str,
    size: int = 200,
    error_correction: str = "M",
    output_path: str | None = None,
    data_url_return: bool = False,
    include_base64: bool | None = None,
) -> dict[str, Any]:
    """Generate a QR code image and return base64/data URL or save to file."""
    import qrcode
    from PIL import Image, PngImagePlugin

    ec_level = error_correction.upper()
    if ec_level not in _VALID_ERROR_CORRECTIONS:
        raise ValueError(
            f"Invalid error_correction '{error_correction}'. "
            f"Use one of: {', '.join(_VALID_ERROR_CORRECTIONS)}"
        )
    if size < _MIN_SIZE_PX or size > _MAX_SIZE_PX:
        raise ValueError(f"'size' must be between {_MIN_SIZE_PX} and {_MAX_SIZE_PX} pixels")

    qr = qrcode.QRCode(
        error_correction=_get_error_correction(ec_level),
        box_size=10,
        border=_BORDER_MODULES,
    )
    qr.add_data(data)
    qr.make(fit=True)

    module_count = qr.modules_count
    total_modules = module_count + (2 * _BORDER_MODULES)
    min_readable_size = total_modules * _MIN_PIXELS_PER_MODULE
    final_size = max(size, min_readable_size)

    img: Image.Image = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    # Keep enough pixels per QR module for dense payloads to stay readable
    # across both zbar and OpenCV scanners.
    resampling = getattr(Image, "Resampling", Image).NEAREST
    img = img.resize((final_size, final_size), resampling)

    # Encode to PNG bytes
    buf = io.BytesIO()
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("effgen_qr_data", data)
    img.save(buf, format="PNG", pnginfo=png_info)
    png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode("ascii")

    saved_path: str | None = None
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png_bytes)
        saved_path = str(dest.resolve())
        logger.info("QR code saved to %s", saved_path)

    # Default: omit the (often huge) base64 from the agent-facing payload
    # when the QR is being saved to a file and a data URL wasn't requested.
    # Callers can force inclusion with include_base64=True. This keeps small-
    # context models (e.g. llama3.1-8b with 8k ctx) from blowing up when the
    # qr_generate result is fed back to them.
    if include_base64 is None:
        include_base64 = not (saved_path and not data_url_return)

    b64_value: str | None = b64 if include_base64 else None
    b64_size = len(b64)

    result: dict[str, Any] = {
        "success": True,
        "data": {
            "qr_base64": b64_value,
            "qr_base64_bytes": b64_size,
            "size_px": final_size,
            "requested_size_px": size,
            "module_count": module_count,
            "qr_version": qr.version,
            "pixels_per_module": final_size / total_modules,
            "size_adjusted": final_size != size,
            "error_correction": ec_level,
            "data_encoded": data,
            "saved_path": saved_path,
        },
        "qr_base64": b64_value,
        "qr_base64_bytes": b64_size,
        "size_px": final_size,
        "requested_size_px": size,
        "module_count": module_count,
        "qr_version": qr.version,
        "pixels_per_module": final_size / total_modules,
        "size_adjusted": final_size != size,
        "error_correction": ec_level,
        "saved_path": saved_path,
        "error": None,
    }

    if data_url_return:
        result["data_url"] = f"data:image/png;base64,{b64}"
        result["data"]["data_url"] = result["data_url"]

    return result


class QRGenerateTool(BaseTool):
    """Generate QR codes locally (no network required)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="qr_generate",
                description=(
                    "Generate a QR code from text or URL data. "
                    "Returns a base64-encoded PNG (or saves to a file). "
                    "Fully local — no network required. "
                    "Operations: generate (create QR code from data)."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform. Currently: 'generate'.",
                        required=True,
                        enum=["generate"],
                    ),
                    ParameterSpec(
                        name="data",
                        type=ParameterType.STRING,
                        description="Text or URL to encode in the QR code.",
                        required=True,
                        max_length=4296,
                    ),
                    ParameterSpec(
                        name="size",
                        type=ParameterType.INTEGER,
                        description="Output image size in pixels (width = height). Default 200.",
                        required=False,
                        default=200,
                    ),
                    ParameterSpec(
                        name="error_correction",
                        type=ParameterType.STRING,
                        description=(
                            "QR error-correction level: L (7%), M (15%), Q (25%), H (30%). "
                            "Higher levels survive more damage but produce denser codes. Default M."
                        ),
                        required=False,
                        default="M",
                        enum=["L", "M", "Q", "H"],
                    ),
                    ParameterSpec(
                        name="output_path",
                        type=ParameterType.STRING,
                        description="Optional file path to save the PNG. If omitted, only base64 is returned.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="data_url_return",
                        type=ParameterType.BOOLEAN,
                        description=(
                            "If True, also include a data:image/png;base64,... URL "
                            "suitable for inline HTML embedding."
                        ),
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="include_base64",
                        type=ParameterType.BOOLEAN,
                        description=(
                            "Include the raw base64 PNG in the response. Default behaviour: "
                            "include base64 when no output_path is given (so the caller can still "
                            "use the image); omit when output_path is given to keep tool output "
                            "small enough for short-context LLMs. Set True to force inclusion."
                        ),
                        required=False,
                    ),
                ],
                timeout_seconds=30,
                tags=["qr", "qrcode", "barcode", "image", "local", "offline"],
                examples=[
                    {"operation": "generate", "data": "https://effgen.org"},
                    {
                        "operation": "generate",
                        "data": "Hello effGen!",
                        "size": 300,
                        "error_correction": "H",
                        "output_path": "/tmp/hello_qr.png",
                    },
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        data: str | None = None,
        size: int = 200,
        error_correction: str = "M",
        output_path: str | None = None,
        data_url_return: bool = False,
        include_base64: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        op = operation.lower()

        if op == "generate":
            if not data or not data.strip():
                raise ValueError("'data' is required and must be non-empty for the generate operation")
            return await asyncio.to_thread(
                _generate_qr,
                data,
                size,
                error_correction,
                output_path,
                data_url_return,
                include_base64,
            )

        raise ValueError(f"Unknown operation: {operation!r}. Use 'generate'.")
