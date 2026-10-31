"""
Image preprocessor for effGen multimodal inputs.

``prepare(part, provider, model)`` applies per-provider constraints
(max bytes, max dimensions, supported MIMEs) and records what was done in
``part.meta["preprocessing"]``.  The returned ``ImagePart`` is safe to send
to that provider.

Constraints table (verified against docs 2026-05):
  - Gemini: max 20 MB inline, max 3072 px/side (recommendation), JPEG/PNG/WEBP/GIF
  - OpenAI : max 20 MB inline, max 2000 px/side, JPEG/PNG/WEBP/GIF
  - Groq   : max 4 MB inline, max 1568 px/side (Llama-4 Scout), JPEG/PNG/WEBP
  - Together: max 8 MB inline, max 2048 px/side, JPEG/PNG/WEBP
  - HF     : max 10 MB inline, max 2048 px/side, JPEG/PNG/WEBP
  - Anthropic: max 5 MB inline, max 1568 px on short side, JPEG/PNG/WEBP/GIF
"""

from __future__ import annotations

import io
import logging
from typing import Any

from effgen.core.messages import ImagePart

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-provider constraint tables
# ---------------------------------------------------------------------------

_PROVIDER_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "gemini": {
        "max_bytes": 20 * 1024 * 1024,
        "max_px": 3072,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp", "image/gif"},
        "prefer_mime": "image/jpeg",
    },
    "openai": {
        "max_bytes": 20 * 1024 * 1024,
        "max_px": 2000,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp", "image/gif"},
        "prefer_mime": "image/jpeg",
    },
    "groq": {
        "max_bytes": 4 * 1024 * 1024,
        "max_px": 1568,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp"},
        "prefer_mime": "image/jpeg",
    },
    "together": {
        "max_bytes": 8 * 1024 * 1024,
        "max_px": 2048,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp"},
        "prefer_mime": "image/jpeg",
    },
    "hf_inference": {
        "max_bytes": 10 * 1024 * 1024,
        "max_px": 2048,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp"},
        "prefer_mime": "image/jpeg",
    },
    "anthropic": {
        "max_bytes": 5 * 1024 * 1024,
        "max_px": 1568,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp", "image/gif"},
        "prefer_mime": "image/jpeg",
    },
    "fireworks": {
        "max_bytes": 10 * 1024 * 1024,
        "max_px": 2048,
        "allowed_mimes": {"image/jpeg", "image/png", "image/webp"},
        "prefer_mime": "image/jpeg",
    },
}

_MIME_TO_PIL_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}


class ImagePreprocessor:
    """Applies per-provider preprocessing to an ``ImagePart``.

    The returned part has ``meta["preprocessing"]`` populated with a list of
    actions taken (e.g. ``["resized 4000x3000→3072x2304", "jpeg_converted"]``).
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider.lower()
        self._constraints = _PROVIDER_CONSTRAINTS.get(self.provider)

    def prepare(self, part: ImagePart) -> ImagePart:
        """Pre-process *part* for ``self.provider``.

        Returns:
            A new ``ImagePart`` (original is not mutated).

        Raises:
            CapabilityNotSupportedError: If the provider is not in the constraints
                table (caller should raise before reaching here).
        """
        if self._constraints is None:
            return part

        constraints = self._constraints
        preprocessing: list[str] = []
        data = part.image
        mime = part.mime

        # 1. MIME conversion (if needed)
        allowed = constraints["allowed_mimes"]
        if mime not in allowed:
            prefer = constraints["prefer_mime"]
            data, mime = _convert_mime(data, mime, prefer)
            preprocessing.append(f"mime_converted:{part.mime}->{mime}")

        # 2. Resize if image exceeds max pixel dimension
        max_px = constraints["max_px"]
        w, h = _get_dimensions(data)
        if w is not None and h is not None and max(w, h) > max_px:
            data, w2, h2 = _resize_image(data, mime, max_px)
            preprocessing.append(f"resized {w}x{h}->{w2}x{h2}")

        # 3. Byte limit — last resort: re-compress as JPEG at lower quality
        max_bytes = constraints["max_bytes"]
        if len(data) > max_bytes:
            data, mime = _reduce_size(data, mime, max_bytes)
            preprocessing.append(f"size_reduced:{len(part.image)}->{len(data)}B")

        if len(data) > max_bytes:
            logger.warning(
                "Image still exceeds %d B after reduction (%d B) for provider '%s'. "
                "The API may reject it.",
                max_bytes, len(data), self.provider,
            )

        meta = dict(part.meta)
        meta["preprocessing"] = preprocessing

        return ImagePart(image=data, mime=mime, meta=meta)


def prepare(part: ImagePart, provider: str, model: str = "") -> ImagePart:
    """Module-level convenience wrapper for ``ImagePreprocessor.prepare``.

    Args:
        part: The ``ImagePart`` to pre-process.
        provider: Provider name (e.g. ``"gemini"``, ``"openai"``).
        model: Optional model name (reserved for future per-model overrides).

    Returns:
        A pre-processed ``ImagePart`` with ``meta["preprocessing"]`` populated.
    """
    return ImagePreprocessor(provider).prepare(part)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Return (width, height) from image bytes using PIL, or (None, None)."""
    try:
        from PIL import Image as _PIL  # type: ignore[import]
        img = _PIL.open(io.BytesIO(data))
        return img.size  # (width, height)
    except Exception:
        return None, None


def _convert_mime(data: bytes, src_mime: str, dst_mime: str) -> tuple[bytes, str]:
    """Convert image bytes from *src_mime* to *dst_mime* using PIL."""
    try:
        from PIL import Image as _PIL  # type: ignore[import]
    except ImportError:
        logger.warning("Pillow not available — skipping MIME conversion %s→%s", src_mime, dst_mime)
        return data, src_mime

    fmt = _MIME_TO_PIL_FORMAT.get(dst_mime, "JPEG")
    try:
        img = _PIL.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA") and fmt == "JPEG":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=90)
        return buf.getvalue(), dst_mime
    except Exception as exc:
        logger.warning("MIME conversion %s→%s failed: %s — keeping original", src_mime, dst_mime, exc)
        return data, src_mime


def _resize_image(data: bytes, mime: str, max_px: int) -> tuple[bytes, int, int]:
    """Downscale *data* so that max(width, height) ≤ *max_px* using Lanczos."""
    try:
        from PIL import Image as _PIL  # type: ignore[import]
    except ImportError:
        logger.warning("Pillow not available — skipping resize")
        w, h = _get_dimensions(data)
        return data, w or 0, h or 0

    fmt = _MIME_TO_PIL_FORMAT.get(mime, "JPEG")
    try:
        img = _PIL.open(io.BytesIO(data))
        w, h = img.size
        scale = max_px / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), _PIL.LANCZOS)
        if img.mode not in ("RGB", "RGBA") and fmt == "JPEG":
            img = img.convert("RGB")
        buf = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": fmt}
        if fmt == "JPEG":
            save_kwargs["quality"] = 85
        img.save(buf, **save_kwargs)
        return buf.getvalue(), new_w, new_h
    except Exception as exc:
        logger.warning("Resize failed: %s — keeping original", exc)
        w, h = _get_dimensions(data)
        return data, w or 0, h or 0


def _reduce_size(data: bytes, mime: str, max_bytes: int) -> tuple[bytes, str]:
    """Re-compress image at progressively lower quality until it fits."""
    try:
        from PIL import Image as _PIL  # type: ignore[import]
    except ImportError:
        return data, mime

    try:
        img = _PIL.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        for quality in (80, 65, 50, 35, 20):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            compressed = buf.getvalue()
            if len(compressed) <= max_bytes:
                return compressed, "image/jpeg"
    except Exception as exc:
        logger.warning("Size reduction failed: %s", exc)

    return data, mime
