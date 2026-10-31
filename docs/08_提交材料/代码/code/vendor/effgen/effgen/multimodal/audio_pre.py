"""
Audio preprocessor for effGen multimodal inputs.

``prepare(part, provider, model)`` applies per-provider constraints
(sample rate, channel count, max duration, max bytes).

``chunk(part, provider, model)`` splits audio that exceeds a provider's
maximum duration or byte limit into sequential chunks.  The caller is
responsible for concatenating the resulting transcriptions.

Provider constraints (verified 2026-05):
  - Gemini : WAV/MP3/MP4/OGG/FLAC/M4A/WebM audio; max 20 MB inline
             (longer files should be uploaded via the Files API).  Audio is
             natively understood — no forced resample.
  - OpenAI : Whisper /audio/transcriptions; max 25 MB per file; supports
             mp3, mp4, mpeg, mpga, m4a, wav, webm. Chunks >25 MB or >30 min split.
             gpt-4o-audio-preview: max 20 MB inline in chat completions.
  - HF     : automatic_speech_recognition endpoint; uses Whisper-large-v3
             by default; no strict byte cap enforced client-side (server
             returns 413 for oversized inputs — warn, don't hard-fail).
  - Anthropic: not supported — raise CapabilityNotSupportedError.

Downsampling:
  Uses ``pydub``. Compressed formats require ffmpeg/avconv on PATH.
"""

from __future__ import annotations

import io
import logging
import shutil
from typing import Any

from effgen.core.messages import AudioPart
from effgen.errors import InvalidMultimodalContent, MissingSystemDependency

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-provider constraints
# ---------------------------------------------------------------------------

_PROVIDER_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "gemini": {
        "max_bytes": 20 * 1024 * 1024,
        "max_duration_s": 60 * 60,  # Files API for longer, but inline cap = 1 hr
        "resample_hz": None,        # Gemini handles resampling natively
        "allowed_mimes": {
            "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav",
            "audio/flac", "audio/ogg", "audio/m4a", "audio/x-m4a",
            "audio/mp4", "audio/mpga", "audio/webm",
        },
    },
    "openai": {
        "max_bytes": 25 * 1024 * 1024,
        "max_duration_s": 30 * 60,  # Whisper transcription max ~30 min chunks
        "resample_hz": None,        # Whisper handles resampling
        "allowed_mimes": {
            "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav",
            "audio/m4a", "audio/x-m4a", "audio/ogg", "audio/webm",
            "audio/flac", "audio/mp4", "audio/mpga",
        },
    },
    "hf_inference": {
        "max_bytes": 100 * 1024 * 1024,  # liberal; server enforces its own limit
        "max_duration_s": 60 * 30,
        "resample_hz": 16_000,  # Whisper-class models prefer 16 kHz mono
        "allowed_mimes": {
            "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav",
            "audio/flac", "audio/ogg", "audio/m4a", "audio/x-m4a",
            "audio/mp4", "audio/mpga", "audio/webm",
        },
    },
}

# MIME → pydub/ffmpeg format hint
_MIME_TO_FORMAT: dict[str, str] = {
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
    "audio/mp4": "mp4",
    "audio/mpga": "mpga",
}

_FFMPEG_FORMATS = {"flac", "m4a", "mp3", "mp4", "mpga", "ogg", "webm"}
_FFMPEG_HINT = (
    "Install ffmpeg to preprocess compressed audio:\n"
    "  Ubuntu/Debian : sudo apt install ffmpeg\n"
    "  macOS         : brew install ffmpeg\n"
    "  conda         : conda install -c conda-forge ffmpeg\n"
    "  Windows       : choco install ffmpeg"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare(part: AudioPart, provider: str, model: str = "") -> AudioPart:
    """Pre-process *part* for *provider*.

    Returns a new ``AudioPart`` (original is not mutated).  If the provider
    requires 16 kHz mono and ``pydub`` is available, the audio is resampled.

    Args:
        part: The audio to pre-process.
        provider: The provider the audio is being prepared for.
        model: The model id, when a provider's limits vary by model.
    """
    return AudioPreprocessor(provider).prepare(part)


def chunk(part: AudioPart, provider: str, model: str = "") -> list[AudioPart]:
    """Split *part* into provider-safe chunks.

    If the audio fits within the provider's limits, returns a one-element list
    with the (possibly pre-processed) part.  Otherwise splits on duration
    boundaries using ``pydub``.

    Args:
        part: The audio to split.
        provider: The provider the chunks must fit.
        model: The model id, when a provider's limits vary by model.

    Returns:
        List of ``AudioPart`` objects each within the provider's max_bytes and
        max_duration_s limits.  Chunks are in sequential order.
    """
    return AudioPreprocessor(provider).chunk(part)


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """Applies per-provider pre-processing to an ``AudioPart``."""

    def __init__(self, provider: str) -> None:
        self.provider = provider.lower()
        self._constraints = _PROVIDER_CONSTRAINTS.get(self.provider, {})

    def prepare(self, part: AudioPart) -> AudioPart:
        """Validate *part* against the provider's audio constraints and return it."""
        constraints = self._constraints
        if not constraints:
            return part

        data = part.audio
        mime = part.mime
        allowed_mimes = constraints.get("allowed_mimes")
        if allowed_mimes and mime not in allowed_mimes:
            raise InvalidMultimodalContent(
                "audio",
                f"MIME type '{mime}' is not supported by provider '{self.provider}'. "
                f"Allowed: {sorted(allowed_mimes)}",
            )

        # Resample to target Hz if provider prefers.
        target_hz = constraints.get("resample_hz")
        if target_hz:
            data, mime = _resample(data, mime, target_hz)

        # Warn if size exceeds provider max
        max_bytes = constraints.get("max_bytes", 0)
        if max_bytes and len(data) > max_bytes:
            logger.warning(
                "Audio (%d B) exceeds provider '%s' max (%d B). "
                "Consider using chunk() to split the audio.",
                len(data), self.provider, max_bytes,
            )

        return AudioPart(audio=data, mime=mime, duration_s=part.duration_s)

    def chunk(self, part: AudioPart) -> list[AudioPart]:
        """Split audio into chunks that fit provider limits."""
        constraints = self._constraints
        max_bytes = constraints.get("max_bytes", 0)
        max_dur = constraints.get("max_duration_s", 0)

        # If no splitting needed, just prepare
        if not max_bytes and not max_dur:
            return [self.prepare(part)]

        fmt = _MIME_TO_FORMAT.get(part.mime, "mp3")
        seg = _load_audio_segment(part.audio, fmt)

        duration_ms = len(seg)
        duration_s = duration_ms / 1000.0

        # If everything fits, return a single prepared part
        if (not max_dur or duration_s <= max_dur) and (not max_bytes or len(part.audio) <= max_bytes):
            return [self.prepare(part)]

        duration_limit_ms = int(max_dur * 1000) if max_dur else duration_ms
        byte_limit_ms = duration_ms
        if max_bytes and len(part.audio) > max_bytes:
            byte_limit_ms = max(1, int(duration_ms * (max_bytes / len(part.audio)) * 0.95))
        chunk_ms = max(1, min(duration_limit_ms, byte_limit_ms, duration_ms))

        # Split into chunks
        chunks: list[AudioPart] = []
        offset = 0
        target_hz = constraints.get("resample_hz")

        while offset < duration_ms:
            current_ms = min(chunk_ms, duration_ms - offset)
            seg_chunk = seg[offset: offset + current_ms]

            if target_hz and seg_chunk.frame_rate != target_hz:
                seg_chunk = seg_chunk.set_frame_rate(target_hz).set_channels(1)

            chunk_bytes = _export_segment(seg_chunk, fmt)

            while max_bytes and len(chunk_bytes) > max_bytes and current_ms > 1000:
                current_ms = max(1000, int(current_ms * (max_bytes / len(chunk_bytes)) * 0.95))
                seg_chunk = seg[offset: offset + current_ms]
                if target_hz and seg_chunk.frame_rate != target_hz:
                    seg_chunk = seg_chunk.set_frame_rate(target_hz).set_channels(1)
                chunk_bytes = _export_segment(seg_chunk, fmt)

            offset += current_ms

            # If a single chunk is still too large, warn but keep it
            if max_bytes and len(chunk_bytes) > max_bytes:
                logger.warning(
                    "Chunk (%d B) still exceeds max_bytes=%d for provider '%s'. "
                    "The API may reject this chunk.",
                    len(chunk_bytes), max_bytes, self.provider,
                )

            chunks.append(AudioPart(
                audio=chunk_bytes,
                mime=part.mime,
                duration_s=len(seg_chunk) / 1000.0,
            ))
            logger.debug(
                "audio_pre: chunk %d/%d — %.1fs, %d B",
                len(chunks),
                -(-duration_ms // chunk_ms),
                len(seg_chunk) / 1000.0,
                len(chunk_bytes),
            )

        logger.info(
            "audio_pre: split %s into %d chunk(s) for provider '%s'",
            part.mime, len(chunks), self.provider,
        )
        return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(data: bytes, mime: str, target_hz: int) -> tuple[bytes, str]:
    """Resample audio to *target_hz* mono using pydub."""
    fmt = _MIME_TO_FORMAT.get(mime, "mp3")
    seg = _load_audio_segment(data, fmt)
    if seg.frame_rate != target_hz or seg.channels != 1:
        seg = seg.set_frame_rate(target_hz).set_channels(1)
        return _export_segment(seg, fmt), mime
    return data, mime


def _load_audio_segment(data: bytes, fmt: str) -> Any:
    """Decode audio bytes with pydub, raising clean dependency/content errors."""
    _ensure_audio_decoder(fmt)
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            from pydub import AudioSegment  # type: ignore[import]
    except ImportError:
        raise MissingSystemDependency(
            "pydub",
            "Install pydub to preprocess audio:\n  pip install pydub",
        )

    try:
        return AudioSegment.from_file(io.BytesIO(data), format=fmt)
    except Exception as exc:
        raise InvalidMultimodalContent("audio", f"Could not decode {fmt} audio: {exc}") from exc


def _export_segment(seg: Any, fmt: str) -> bytes:
    """Encode a pydub segment to bytes."""
    _ensure_audio_decoder(fmt)
    buf = io.BytesIO()
    seg.export(buf, format=fmt)
    return buf.getvalue()


def _ensure_audio_decoder(fmt: str) -> None:
    """Raise a clean error when compressed audio needs ffmpeg/avconv."""
    if fmt in _FFMPEG_FORMATS and shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
        raise MissingSystemDependency("ffmpeg", _FFMPEG_HINT)


def get_audio_duration(data: bytes, mime: str) -> float | None:
    """Return duration in seconds for *data*, or None if pydub is unavailable."""
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            from pydub import AudioSegment  # type: ignore[import]
        fmt = _MIME_TO_FORMAT.get(mime, "mp3")
        seg = AudioSegment.from_file(io.BytesIO(data), format=fmt)
        return len(seg) / 1000.0
    except Exception:
        return None
