"""
Audio transcription tool for the effGen framework.

Primary backend : faster-whisper (local CPU/GPU inference).
Fallback backend: HuggingFace Inference API (openai/whisper-large-v3),
                  activated when HF_TOKEN env var is set and faster-whisper
                  is unavailable or explicitly overridden.

Operations
----------
- transcribe : Transcribe an audio file or raw bytes → text + segments + language.

Input: file path (str/Path) OR raw audio bytes.
Output: {"text": str, "segments": list[dict], "language": str, "backend": str}

System dependency: ffmpeg is required by faster-whisper for non-WAV formats.
  Ubuntu/Debian : sudo apt install ffmpeg
  macOS         : brew install ffmpeg
  Windows       : choco install ffmpeg
  conda         : conda install -c conda-forge ffmpeg
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

from effgen.errors import AudioBackendUnavailable, MissingSystemDependency

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
# Backend detection helpers
# ---------------------------------------------------------------------------

_FASTER_WHISPER_AVAILABLE: bool | None = None
_FFMPEG_AVAILABLE: bool | None = None
_FFMPEG_REQUIRED_SUFFIXES = {
    ".aac",
    ".m4a",
    ".mp4",
    ".webm",
    ".wma",
}
_FFMPEG_INSTALL_HINT = (
    "Install ffmpeg for compressed audio formats:\n"
    "  Ubuntu/Debian : sudo apt install ffmpeg\n"
    "  macOS         : brew install ffmpeg\n"
    "  Windows       : choco install ffmpeg\n"
    "  conda         : conda install -c conda-forge ffmpeg"
)


def _check_faster_whisper() -> bool:
    global _FASTER_WHISPER_AVAILABLE
    if _FASTER_WHISPER_AVAILABLE is not None:
        return _FASTER_WHISPER_AVAILABLE
    try:
        import importlib.util
        _FASTER_WHISPER_AVAILABLE = importlib.util.find_spec("faster_whisper") is not None
    except Exception:
        _FASTER_WHISPER_AVAILABLE = False
    return _FASTER_WHISPER_AVAILABLE


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or None
    )


def _check_ffmpeg() -> bool:
    """Return True if the ffmpeg binary is reachable on PATH."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is not None:
        return _FFMPEG_AVAILABLE
    found = shutil.which("ffmpeg") is not None
    if not found:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            found = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            found = False
    _FFMPEG_AVAILABLE = found
    logger.debug("ffmpeg available: %s", found)
    return found


def _source_suffix(source: str | bytes | bytearray | Path) -> str:
    if isinstance(source, str | Path):
        return Path(source).suffix.lower()
    return ".mp3"


def _requires_ffmpeg(source: str | bytes | bytearray | Path) -> bool:
    return _source_suffix(source) in _FFMPEG_REQUIRED_SUFFIXES


def _ensure_ffmpeg_for_source(source: str | bytes | bytearray | Path) -> None:
    if _requires_ffmpeg(source) and not _check_ffmpeg():
        raise MissingSystemDependency("ffmpeg", _FFMPEG_INSTALL_HINT)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper.

    Uses nvidia-smi to detect CUDA; falls back to CPU.
    compute_type: 'float16' for CUDA, 'int8' for CPU.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.debug("GPU detected: %s", result.stdout.strip().splitlines()[0])
            return "cuda", "float16"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):  # GPU probe failed; fall back to CPU
        pass
    return "cpu", "int8"


# ---------------------------------------------------------------------------
# Audio source loader
# ---------------------------------------------------------------------------

def _resolve_audio_source(source: str | bytes | Path) -> str:
    """Return a file path suitable for passing to faster-whisper.

    If source is bytes, writes to a temp file and returns that path.
    The caller is responsible for cleaning up temp files.
    """
    if isinstance(source, str | Path):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")
        return str(p)
    if isinstance(source, bytes | bytearray):
        suffix = ".mp3"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(bytes(source))
        tmp.flush()
        tmp.close()
        return tmp.name
    raise TypeError(f"Expected str, Path, or bytes; got {type(source)}")


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

def _whisper_transcribe(
    source: str | bytes | Path,
    model_size: str = "base",
    language: str | None = None,
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """Transcribe audio using faster-whisper (local)."""
    from faster_whisper import WhisperModel

    _ensure_ffmpeg_for_source(source)
    device, compute_type = _detect_device()

    if device == "cpu" and model_size not in ("tiny", "tiny.en", "base", "base.en"):
        warnings.warn(
            f"Running faster-whisper model '{model_size}' on CPU — this may be slow. "
            "Consider using 'base' or 'tiny' for faster inference on CPU, "
            "or set up a CUDA-capable GPU.",
            RuntimeWarning,
            stacklevel=3,
        )

    audio_path = _resolve_audio_source(source)
    _tmp_created = audio_path if isinstance(source, bytes | bytearray) else None

    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            beam_size=5,
        )

        segments: list[dict] = []
        full_text_parts: list[str] = []

        for seg in segments_iter:
            seg_dict: dict[str, Any] = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            }
            if word_timestamps and seg.words:
                seg_dict["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                    for w in seg.words
                ]
            segments.append(seg_dict)
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts).strip()
        detected_language = info.language if info.language else (language or "unknown")

        return {
            "success": True,
            "data": {
                "text": full_text,
                "segments": segments,
                "language": detected_language,
                "backend": "faster-whisper",
                "model_size": model_size,
                "duration": round(info.duration, 2) if info.duration else None,
            },
            "text": full_text,
            "segments": segments,
            "language": detected_language,
            "backend": "faster-whisper",
            "error": None,
        }

    finally:
        if _tmp_created and Path(_tmp_created).exists():
            try:
                Path(_tmp_created).unlink()
            except OSError:  # best-effort temp-file cleanup
                pass


# ---------------------------------------------------------------------------
# HuggingFace Inference fallback
# ---------------------------------------------------------------------------

_HF_ASR_MODEL = "openai/whisper-large-v3"


def _hf_transcribe(
    source: str | bytes | Path,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe using HuggingFace Inference API (whisper-large-v3)."""
    token = _hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it to use the HuggingFace Inference API fallback."
        )

    if isinstance(source, str | Path):
        audio_data: bytes = Path(source).read_bytes()
    else:
        audio_data = bytes(source)

    try:
        from effgen.models.hf_inference_adapter import HFInferenceAdapter
    except Exception as exc:
        raise AudioBackendUnavailable(
            tried_backends=["hf-inference"],
            extra=(
                "HuggingFace Inference fallback requires the effGen HF adapter "
                "and huggingface_hub. Install with: pip install 'effgen[audio]'"
            ),
        ) from exc

    adapter = HFInferenceAdapter(
        _HF_ASR_MODEL,
        api_token=token,
        timeout=120,
        enable_rate_limiting=False,
        enable_cost_tracking=False,
        warn_unknown_model=False,
    )

    try:
        adapter.load()
        client = adapter._client
        if client is None:
            raise RuntimeError("HFInferenceAdapter did not create an inference client.")
        result = client.automatic_speech_recognition(
            audio=audio_data,
            model=_HF_ASR_MODEL,
        )
    except Exception as exc:
        raise RuntimeError(f"HuggingFace Inference API error: {exc}") from exc
    finally:
        adapter.unload()

    text = result.text if hasattr(result, "text") else str(result)

    chunks: list[dict] = []
    if hasattr(result, "chunks") and result.chunks:
        for chunk in result.chunks:
            chunks.append(
                {
                    "text": chunk.text if hasattr(chunk, "text") else str(chunk),
                    "timestamp": list(chunk.timestamp) if hasattr(chunk, "timestamp") else None,
                }
            )

    return {
        "success": True,
        "data": {
            "text": text,
            "segments": chunks,
            "language": language or "unknown",
            "backend": "hf-inference",
            "model": _HF_ASR_MODEL,
        },
        "text": text,
        "segments": chunks,
        "language": language or "unknown",
        "backend": "hf-inference",
        "error": None,
    }


# ---------------------------------------------------------------------------
# AudioTranscribeTool — BaseTool subclass
# ---------------------------------------------------------------------------

class AudioTranscribeTool(BaseTool):
    """
    Transcribe audio files to text.

    Primary backend: faster-whisper (local CPU/GPU, no auth required).
    Fallback backend: HuggingFace Inference API (set HF_TOKEN).

    Accepts audio file paths OR raw audio bytes.

    System dependency: ffmpeg is required for non-WAV formats (MP3, OGG, etc.).
      Ubuntu/Debian : sudo apt install ffmpeg
      macOS         : brew install ffmpeg
      Windows       : choco install ffmpeg
      conda         : conda install -c conda-forge ffmpeg
    """

    def __init__(self, allowed_directories: list[str] | None = None) -> None:
        """Args:
            allowed_directories: Roots an audio path may be read from. By
                default any path is allowed except protected system and
                credential locations (/etc, /proc, ~/.ssh, cloud creds, …),
                which are always refused. Pass a list to confine reads to
                those roots only.
        """
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)
        super().__init__(
            metadata=ToolMetadata(
                name="audio_transcribe",
                description=(
                    "Transcribe audio files (MP3, WAV, OGG, FLAC, etc.) to text using "
                    "faster-whisper (local) or HuggingFace Inference API (fallback). "
                    "Accepts file paths or raw audio bytes. "
                    "Operation: transcribe. "
                    "Set HF_TOKEN for the cloud fallback."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform. Currently only 'transcribe'.",
                        required=True,
                        enum=["transcribe"],
                    ),
                    ParameterSpec(
                        name="audio_path",
                        type=ParameterType.STRING,
                        description="Path to the audio file. Mutually exclusive with audio_bytes.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="audio_bytes",
                        type=ParameterType.ANY,
                        description=(
                            "Raw audio bytes (passed programmatically). "
                            "Mutually exclusive with audio_path."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="model_size",
                        type=ParameterType.STRING,
                        description=(
                            "faster-whisper model size: tiny, base, small, medium, large-v2, "
                            "large-v3. Defaults to 'base'. Larger models are more accurate "
                            "but require more memory and time, especially on CPU."
                        ),
                        required=False,
                        default="base",
                        enum=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                              "medium", "medium.en", "large-v1", "large-v2", "large-v3",
                              "distil-small.en", "distil-medium.en", "distil-large-v2",
                              "distil-large-v3"],
                    ),
                    ParameterSpec(
                        name="language",
                        type=ParameterType.STRING,
                        description=(
                            "ISO 639-1 language code (e.g. 'en', 'fr', 'de'). "
                            "If None, language is auto-detected."
                        ),
                        required=False,
                        default=None,
                    ),
                    ParameterSpec(
                        name="word_timestamps",
                        type=ParameterType.BOOLEAN,
                        description=(
                            "If True, include per-word start/end timestamps in segments. "
                            "Only supported by the faster-whisper backend."
                        ),
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="backend",
                        type=ParameterType.STRING,
                        description=(
                            "Force a specific backend: 'faster-whisper', 'hf-inference', "
                            "or 'auto' (try faster-whisper first, then hf-inference). "
                            "Defaults to 'auto'."
                        ),
                        required=False,
                        default="auto",
                        enum=["faster-whisper", "hf-inference", "auto"],
                    ),
                ],
                timeout_seconds=300,
                tags=["audio", "transcription", "speech-to-text", "whisper", "asr"],
                examples=[
                    {"operation": "transcribe", "audio_path": "/path/to/speech.mp3"},
                    {
                        "operation": "transcribe",
                        "audio_path": "/path/to/interview.wav",
                        "model_size": "small",
                        "language": "en",
                        "word_timestamps": True,
                    },
                ],
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_source(
        self,
        audio_path: str | None,
        audio_bytes: str | bytes | bytearray | None,
    ) -> str | bytes:
        if audio_path and audio_bytes is not None:
            raise ValueError("Provide audio_path OR audio_bytes, not both.")
        if audio_path:
            return str(confine_path(audio_path, self._allowed_dirs))
        if audio_bytes is not None:
            if isinstance(audio_bytes, bytes | bytearray):
                return bytes(audio_bytes)
            raise TypeError("audio_bytes must be raw bytes when passed programmatically.")
        raise ValueError("Either audio_path or audio_bytes is required.")

    def _pick_backend(self, preferred: str) -> str:
        if preferred == "faster-whisper":
            if not _check_faster_whisper():
                raise AudioBackendUnavailable(
                    tried_backends=["faster-whisper"],
                    extra=(
                        "Install faster-whisper with: pip install 'effgen[audio]'\n\n"
                        f"{_FFMPEG_INSTALL_HINT}"
                    ),
                )
            return "faster-whisper"
        if preferred == "hf-inference":
            if not _hf_token():
                raise RuntimeError(
                    "HF_TOKEN is not set. Set it to use the HuggingFace Inference API."
                )
            return "hf-inference"
        # auto
        if _check_faster_whisper():
            return "faster-whisper"
        if _hf_token():
            logger.warning(
                "faster-whisper not installed; falling back to HuggingFace Inference API."
            )
            return "hf-inference"
        raise AudioBackendUnavailable(tried_backends=["faster-whisper", "hf-inference"])

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(
        self,
        operation: str,
        audio_path: str | None = None,
        audio_bytes: str | bytes | bytearray | None = None,
        model_size: str = "base",
        language: str | None = None,
        word_timestamps: bool = False,
        backend: str = "auto",
        **kwargs: Any,
    ) -> dict[str, Any]:
        op = operation.lower()
        if op != "transcribe":
            raise ValueError(
                f"Unknown operation: {operation!r}. Only 'transcribe' is supported."
            )

        source = self._resolve_source(audio_path, audio_bytes)
        chosen = self._pick_backend(backend)

        if chosen == "faster-whisper":
            return await asyncio.to_thread(
                _whisper_transcribe,
                source,
                model_size,
                language,
                word_timestamps,
            )
        else:
            return await asyncio.to_thread(
                _hf_transcribe,
                source,
                language,
            )
