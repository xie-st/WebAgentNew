"""
OCR tool for the effGen framework.

Primary backend : pytesseract (wraps local Tesseract binary).
Fallback backend: OCR.space cloud API, activated when OCR_SPACE_API_KEY is set
                  and Tesseract is not found.

Operations
----------
- extract           : OCR an image file or raw bytes -> text + confidence.
- extract_regions   : OCR specific rectangular sub-regions of an image.

Input: file path (str/Path) OR raw image bytes for both operations.
Output: {"text": str, "confidence": float, "words": list[dict]}
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from effgen.errors import OCRBackendUnavailable

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._fs import confine_path, normalize_allowed_dirs
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

# Fixed OCR API host (pinned; the shared SSRF guard also blocks internal targets).
_OCR_SPACE_HOSTS = frozenset({"api.ocr.space"})

# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

_tesseract_available: bool | None = None  # cached after first probe


def _check_tesseract() -> bool:
    """Return True if the `tesseract` binary is reachable on PATH."""
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    found = shutil.which("tesseract") is not None
    if not found:
        # also try subprocess in case PATH differs
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                timeout=5,
            )
            found = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            found = False
    _tesseract_available = found
    logger.debug("Tesseract available: %s", found)
    return found


def _ocr_space_key() -> str | None:
    """Return OCR.space API key from env, or None."""
    return os.environ.get("OCR_SPACE_API_KEY") or None


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------

def _load_image(source: str | bytes | Path) -> "Image.Image":  # noqa: F821
    """Load a PIL Image from a file path or raw bytes."""
    from PIL import Image

    if isinstance(source, str | Path):
        with Image.open(str(source)) as img:
            return img.convert("RGB")
    if isinstance(source, bytes | bytearray):
        with Image.open(io.BytesIO(source)) as img:
            return img.convert("RGB")
    raise TypeError(f"Expected str, Path, or bytes; got {type(source)}")


def _tesseract_extract(
    source: str | bytes | Path,
    lang: str = "eng",
    tesseract_config: str | None = None,
) -> dict[str, Any]:
    """Extract text using local Tesseract via pytesseract."""
    import pytesseract

    img = _load_image(source)

    # --oem 3 = LSTM + legacy, --psm 6 = assume uniform block of text
    config = tesseract_config or "--oem 3 --psm 6"
    raw_text: str = pytesseract.image_to_string(img, lang=lang, config=config).strip()

    # Detailed word-level data
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    words: list[dict] = []
    confidences: list[float] = []
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        confidences.append(conf)
        words.append(
            {
                "word": word,
                "confidence": conf / 100.0,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "block_num": data["block_num"][i],
                "line_num": data["line_num"][i],
            }
        )

    avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    return {
        "success": True,
        "data": {
            "text": raw_text,
            "confidence": round(avg_conf, 4),
            "words": words,
            "backend": "tesseract",
            "lang": lang,
        },
        "text": raw_text,
        "confidence": round(avg_conf, 4),
        "words": words,
        "backend": "tesseract",
        "error": None,
    }


# ---------------------------------------------------------------------------
# OCR.space fallback backend
# ---------------------------------------------------------------------------

_OCR_SPACE_URL = "https://api.ocr.space/parse/image"
_OCR_SPACE_TIMEOUT = 30


def _ocrspace_extract(
    source: str | bytes | Path,
    lang: str = "eng",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Extract text using the OCR.space free cloud API."""
    key = api_key or _ocr_space_key()
    if not key:
        raise OCRBackendUnavailable(tried_backends=["tesseract", "ocr.space"])

    # Map ISO 639-1 / tesseract lang codes to OCR.space language codes
    _lang_map = {
        "eng": "eng",
        "fra": "fre",
        "deu": "ger",
        "spa": "spa",
        "por": "por",
        "ita": "ita",
        "zho": "chs",
        "jpn": "jpn",
        "kor": "kor",
        "ara": "ara",
        "rus": "rus",
    }
    oc_lang = _lang_map.get(lang, "eng")

    # Load image bytes
    if isinstance(source, str | Path):
        img_bytes = Path(source).read_bytes()
        filename = Path(source).name
    else:
        img_bytes = bytes(source)
        filename = "image.png"

    boundary = b"----effgen_ocr_boundary"
    body_parts: list[bytes] = []

    def _field(name: str, value: str) -> bytes:
        return (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n\r\n'
            + value.encode() + b"\r\n"
        )

    body_parts.append(_field("apikey", key))
    body_parts.append(_field("language", oc_lang))
    body_parts.append(_field("isOverlayRequired", "true"))
    body_parts.append(_field("detectOrientation", "true"))
    body_parts.append(_field("scale", "true"))
    body_parts.append(_field("OCREngine", "2"))

    # File field
    file_part = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="'
        + filename.encode() + b'"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n"
        + img_bytes + b"\r\n"
    )
    body_parts.append(file_part)
    body_parts.append(b"--" + boundary + b"--\r\n")

    body = b"".join(body_parts)
    content_type = b"multipart/form-data; boundary=" + boundary

    try:
        with safe_urlopen(
            _OCR_SPACE_URL,
            data=body,
            headers={"Content-Type": content_type.decode()},
            method="POST",
            timeout=_OCR_SPACE_TIMEOUT,
            allowed_hosts=_OCR_SPACE_HOSTS,
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"OCR.space API request failed: {exc}") from exc

    if payload.get("IsErroredOnProcessing"):
        err = payload.get("ErrorMessage", ["Unknown OCR.space error"])
        raise RuntimeError(f"OCR.space error: {err}")

    parsed = payload.get("ParsedResults") or []
    full_text = " ".join(r.get("ParsedText", "") for r in parsed).strip()

    # Build word list from overlay
    words: list[dict] = []
    for r in parsed:
        overlay = r.get("TextOverlay") or {}
        for line in overlay.get("Lines") or []:
            for word_obj in line.get("Words") or []:
                words.append(
                    {
                        "word": word_obj.get("WordText", ""),
                        "confidence": None,  # OCR.space doesn't return per-word conf
                        "left": word_obj.get("Left", 0),
                        "top": word_obj.get("Top", 0),
                        "width": word_obj.get("Width", 0),
                        "height": word_obj.get("Height", 0),
                    }
                )

    return {
        "success": True,
        "data": {
            "text": full_text,
            "confidence": None,
            "words": words,
            "backend": "ocr.space",
            "lang": lang,
        },
        "text": full_text,
        "confidence": None,
        "words": words,
        "backend": "ocr.space",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _extract_regions(
    source: str | bytes | Path,
    regions: list[dict],
    lang: str = "eng",
    backend: str = "auto",
    tesseract_config: str | None = None,
) -> dict[str, Any]:
    """Crop image to each region and OCR individually."""
    img = _load_image(source)
    results: list[dict] = []

    for i, region in enumerate(regions):
        left = region.get("left", 0)
        top = region.get("top", 0)
        width = region.get("width", img.width)
        height = region.get("height", img.height)
        cropped = img.crop((left, top, left + width, top + height))

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        crop_bytes = buf.getvalue()

        try:
            if backend == "tesseract":
                res = _tesseract_extract(
                    crop_bytes,
                    lang=lang,
                    tesseract_config=tesseract_config,
                )
            elif backend == "ocr.space":
                res = _ocrspace_extract(crop_bytes, lang=lang)
            else:
                raise OCRBackendUnavailable(tried_backends=["tesseract", "ocr.space"])
            results.append(
                {
                    "region_index": i,
                    "region": region,
                    "text": res["text"],
                    "confidence": res["confidence"],
                    "words": res["words"],
                    "backend": res["backend"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "region_index": i,
                    "region": region,
                    "text": "",
                    "error": str(exc),
                }
            )

    combined_text = "\n".join(r.get("text", "") for r in results)
    return {
        "success": True,
        "data": {"regions": results, "combined_text": combined_text},
        "regions": results,
        "combined_text": combined_text,
        "error": None,
    }


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class OCRTool(BaseTool):
    """
    Extract text from images using OCR.

    Primary backend: Tesseract (local, no auth required).
    Fallback backend: OCR.space cloud API (set OCR_SPACE_API_KEY).

    Accepts image file paths OR raw bytes for all operations.
    """

    def __init__(self, allowed_directories: list[str] | None = None) -> None:
        """Args:
            allowed_directories: Roots an image path may be read from. By
                default any path is allowed except protected system and
                credential locations (/etc, /proc, ~/.ssh, cloud creds, …),
                which are always refused. Pass a list to confine reads to
                those roots only. This also bounds what OCR may upload to the
                OCR.space backend.
        """
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)
        super().__init__(
            metadata=ToolMetadata(
                name="ocr",
                description=(
                    "Extract text from images (PNG, JPEG, TIFF, etc.) using OCR. "
                    "Primary backend: local Tesseract. "
                    "Fallback: OCR.space cloud API (set OCR_SPACE_API_KEY). "
                    "Accepts file paths or raw image bytes. "
                    "Operations: extract (full image), extract_regions (sub-regions)."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'extract' or 'extract_regions'.",
                        required=True,
                        enum=["extract", "extract_regions"],
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
                            "Raw image bytes or a base64-encoded image string. "
                            "Mutually exclusive with image_path."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="lang",
                        type=ParameterType.STRING,
                        description=(
                            "Language code for OCR (e.g. 'eng', 'fra', 'deu'). "
                            "Defaults to 'eng'."
                        ),
                        required=False,
                        default="eng",
                    ),
                    ParameterSpec(
                        name="regions",
                        type=ParameterType.ARRAY,
                        description=(
                            "For extract_regions: list of dicts with keys "
                            "'left', 'top', 'width', 'height' (pixels)."
                        ),
                        required=False,
                        items_type=ParameterType.OBJECT,
                    ),
                    ParameterSpec(
                        name="backend",
                        type=ParameterType.STRING,
                        description=(
                            "Force a specific backend: 'tesseract' or 'ocr.space'. "
                            "Default: auto (tesseract first, then ocr.space)."
                        ),
                        required=False,
                        enum=["tesseract", "ocr.space", "auto"],
                        default="auto",
                    ),
                    ParameterSpec(
                        name="tesseract_config",
                        type=ParameterType.STRING,
                        description=(
                            "Optional raw Tesseract config string used only by the local "
                            "Tesseract backend, e.g. '--oem 3 --psm 7'."
                        ),
                        required=False,
                    ),
                ],
                timeout_seconds=60,
                tags=["ocr", "image", "text-extraction", "tesseract", "vision"],
                examples=[
                    {"operation": "extract", "image_path": "/path/to/scan.png"},
                    {
                        "operation": "extract_regions",
                        "image_path": "/path/to/doc.png",
                        "regions": [
                            {"left": 0, "top": 0, "width": 400, "height": 100},
                            {"left": 0, "top": 200, "width": 400, "height": 100},
                        ],
                    },
                ],
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        """Allow programmatic callers to pass raw bytes while keeping JSON schema valid."""
        image_bytes = kwargs.get("image_bytes")
        if isinstance(image_bytes, bytes | bytearray):
            kwargs = dict(kwargs)
            kwargs["image_bytes"] = "__raw_image_bytes__"
        return super().validate_parameters(**kwargs)

    def _resolve_source(
        self,
        image_path: str | None,
        image_bytes: str | bytes | bytearray | None,
    ) -> str | bytes:
        """Return a file path string or raw bytes from the caller's inputs."""
        if image_path and image_bytes:
            raise ValueError("Provide image_path OR image_bytes, not both.")
        if image_path:
            return str(confine_path(image_path, self._allowed_dirs))
        if image_bytes:
            if isinstance(image_bytes, bytes | bytearray):
                return bytes(image_bytes)
            encoded = image_bytes.strip()
            if encoded.startswith("data:image/") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(
                    "image_bytes must be raw image bytes or a valid base64-encoded image string."
                ) from exc
        raise ValueError("Either image_path or image_bytes is required.")

    def _pick_backend(
        self,
        preferred: str,
        tried: list[str],
    ) -> str:
        """Determine which backend to use, raising if none available."""
        if preferred not in {"auto", "tesseract", "ocr.space"}:
            raise ValueError("backend must be one of: auto, tesseract, ocr.space")
        if preferred in ("tesseract", "auto"):
            if _check_tesseract():
                return "tesseract"
            tried.append("tesseract")
        if preferred in ("ocr.space", "auto"):
            if _ocr_space_key():
                return "ocr.space"
            tried.append("ocr.space")
        raise OCRBackendUnavailable(tried_backends=tried)

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(
        self,
        operation: str,
        image_path: str | None = None,
        image_bytes: str | bytes | bytearray | None = None,
        lang: str = "eng",
        regions: list[dict] | None = None,
        backend: str = "auto",
        tesseract_config: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        source = self._resolve_source(image_path, image_bytes)
        op = operation.lower()
        tried: list[str] = []

        if op == "extract":
            chosen = self._pick_backend(backend, tried)
            if chosen == "tesseract":
                return await asyncio.to_thread(
                    _tesseract_extract,
                    source,
                    lang,
                    tesseract_config,
                )
            else:
                return await asyncio.to_thread(_ocrspace_extract, source, lang)

        elif op == "extract_regions":
            if not regions:
                raise ValueError("'regions' list is required for extract_regions.")
            chosen = self._pick_backend(backend, tried)
            return await asyncio.to_thread(
                _extract_regions,
                source,
                regions,
                lang,
                chosen,
                tesseract_config,
            )

        else:
            raise ValueError(f"Unknown operation: {operation!r}. Use 'extract' or 'extract_regions'.")
