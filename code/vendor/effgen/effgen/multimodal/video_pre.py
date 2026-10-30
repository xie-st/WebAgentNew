"""
Video preprocessor for effGen multimodal inputs.

``VideoSource`` wraps a local file, bytes payload, or URL and provides:
  - ``sample_frames(fps, max_frames)`` — extract JPEG keyframes via ffmpeg
  - ``extract_audio()``               — pull the audio track as an AudioPart

ffmpeg is required.  A missing binary raises
``MissingSystemDependency("ffmpeg", ...)``.

Provider strategy:
  - Gemini 2.x/3.x: supports native video via the Files API (video/mp4 etc.).
    Use ``VideoSource.to_gemini_file_ref()`` to upload and get a FileRef.
  - All others (OpenAI, Groq, Together, HF): no native video — use
    ``sample_frames()`` to produce a sequence of ImageParts that the adapter
    sends as a multi-image turn.

Accepted sources:
  - ``bytes``     — raw MP4/WebM/MOV/AVI/MKV bytes
  - ``str / Path``— local file path
  - ``str``       — http(s) URL; downloaded once and cached in memory
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from effgen.core.messages import AudioPart, ImagePart
from effgen.errors import InvalidMultimodalContent, MissingSystemDependency

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Install hints
# ---------------------------------------------------------------------------

_FFMPEG_HINT = (
    "Install ffmpeg to process video:\n"
    "  Ubuntu/Debian : sudo apt install ffmpeg\n"
    "  macOS         : brew install ffmpeg\n"
    "  conda         : conda install -c conda-forge ffmpeg\n"
    "  Windows       : choco install ffmpeg"
)

_FFPROBE_HINT = (
    "Install ffprobe (part of ffmpeg) to inspect video:\n"
    "  Ubuntu/Debian : sudo apt install ffmpeg\n"
    "  macOS         : brew install ffmpeg\n"
    "  conda         : conda install -c conda-forge ffmpeg"
)

# Supported video MIME types (accepted as inline source bytes)
SUPPORTED_VIDEO_MIMES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",  # AVI
    "video/x-matroska",  # MKV
    "video/mpeg",
    "video/ogg",
    "video/3gpp",
}

# Extension → MIME
_EXT_TO_MIME: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".ogv": "video/ogg",
    ".3gp": "video/3gpp",
}


def _require_ffmpeg() -> str:
    """Return ffmpeg path or raise MissingSystemDependency."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise MissingSystemDependency("ffmpeg", _FFMPEG_HINT)
    return path


def _require_ffprobe() -> str:
    """Return ffprobe path or raise MissingSystemDependency."""
    path = shutil.which("ffprobe")
    if path is None:
        raise MissingSystemDependency("ffprobe", _FFPROBE_HINT)
    return path


# ---------------------------------------------------------------------------
# VideoSource
# ---------------------------------------------------------------------------

class VideoSource:
    """Wraps a video source (file path, URL, or raw bytes) for preprocessing.

    Args:
        source: Local file path (str/Path), HTTP(S) URL, or raw video bytes.
        mime_type: Override the inferred MIME type (e.g. ``"video/mp4"``).

    Raises:
        InvalidMultimodalContent: For empty bytes or an unsupported source type.
        MissingSystemDependency:  If ffmpeg/ffprobe is absent when needed.
    """

    def __init__(
        self,
        source: str | Path | bytes,
        mime_type: str | None = None,
    ) -> None:
        if mime_type is not None and mime_type not in SUPPORTED_VIDEO_MIMES:
            raise InvalidMultimodalContent(
                "video",
                f"MIME type '{mime_type}' is not supported. "
                f"Allowed: {sorted(SUPPORTED_VIDEO_MIMES)}",
            )
        self._source = source
        self._mime_type = mime_type
        # Resolved path — populated lazily
        self._tmp_path: str | None = None
        self._own_tmp = False  # True when we created the temp file

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mime_type(self) -> str:
        """Best-guess MIME type for the video source."""
        if self._mime_type:
            return self._mime_type
        src = self._source
        if isinstance(src, str | Path):
            ext = Path(str(src)).suffix.lower()
            return _EXT_TO_MIME.get(ext, "video/mp4")
        return "video/mp4"  # bytes default

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def sample_frames(
        self,
        fps: float = 1.0,
        max_frames: int = 16,
    ) -> list[ImagePart]:
        """Extract keyframes from the video at *fps* frames-per-second.

        Args:
            fps:        Frames to extract per second of video (default 1).
            max_frames: Hard cap on total frames returned (default 16).

        Returns:
            List of :class:`~effgen.core.messages.ImagePart` objects (JPEG).

        Raises:
            MissingSystemDependency: If ffmpeg is not on PATH.
            InvalidMultimodalContent: If the video cannot be decoded.
        """
        _require_ffmpeg()
        if fps <= 0:
            raise InvalidMultimodalContent("video", "fps must be positive")
        if max_frames < 1:
            raise InvalidMultimodalContent("video", "max_frames must be at least 1")

        video_path = self._resolve_path()
        logger.info(
            "video_pre: sampling frames at %.2f fps, max=%d from '%s'",
            fps, max_frames, video_path,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
            ffmpeg_cmd = [
                _require_ffmpeg(),
                "-loglevel", "error",
                "-i", video_path,
                "-vf", f"fps={fps}",
                "-q:v", "2",  # JPEG quality (2 = high)
                "-frames:v", str(max_frames),
                out_pattern,
            ]
            try:
                subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                raise InvalidMultimodalContent(
                    "video",
                    f"ffmpeg frame extraction failed: {stderr[:500]}",
                ) from exc

            frame_files = sorted(
                f for f in os.listdir(tmpdir) if f.endswith(".jpg")
            )
            if not frame_files:
                raise InvalidMultimodalContent(
                    "video",
                    "ffmpeg produced no frames. The video may be corrupt or have no video track.",
                )

            frames: list[ImagePart] = []
            for fname in frame_files[:max_frames]:
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "rb") as fh:
                    data = fh.read()
                frames.append(ImagePart(image=data, mime="image/jpeg"))

        logger.info("video_pre: extracted %d frames", len(frames))
        return frames

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def extract_audio(
        self,
        target_mime: str = "audio/mp3",
    ) -> AudioPart | None:
        """Extract the audio track from the video.

        Args:
            target_mime: Desired output MIME (default ``"audio/mp3"``).

        Returns:
            :class:`~effgen.core.messages.AudioPart` or ``None`` if the video
            has no audio track.

        Raises:
            MissingSystemDependency: If ffmpeg is not on PATH.
        """
        _require_ffmpeg()
        video_path = self._resolve_path()

        # Check whether the video has an audio stream
        if not self._has_audio_stream(video_path):
            logger.info("video_pre: no audio track found in '%s'", video_path)
            return None

        ext_map = {
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/flac": "flac",
            "audio/m4a": "m4a",
            "audio/x-m4a": "m4a",
            "audio/ogg": "ogg",
        }
        fmt = ext_map.get(target_mime, "mp3")
        resolved_mime = target_mime if target_mime in ext_map else "audio/mp3"

        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tf:
            out_path = tf.name

        try:
            ffmpeg_cmd = [
                _require_ffmpeg(),
                "-loglevel", "error",
                "-i", video_path,
                "-vn",               # drop video stream
                "-acodec", "libmp3lame" if fmt == "mp3" else "copy",
                "-q:a", "2",
                "-y",
                out_path,
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                logger.warning("video_pre: audio extraction failed: %s", stderr[:300])
                return None

            with open(out_path, "rb") as fh:
                audio_data = fh.read()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

        if not audio_data:
            return None

        return AudioPart(audio=audio_data, mime=resolved_mime)

    # ------------------------------------------------------------------
    # Video duration
    # ------------------------------------------------------------------

    def duration_seconds(self) -> float | None:
        """Return video duration in seconds using ffprobe, or None on failure."""
        try:
            _require_ffprobe()
        except MissingSystemDependency:
            return None

        video_path = self._resolve_path()
        cmd = [
            _require_ffprobe(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as exc:
            logger.debug("video_pre: could not determine duration: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Gemini native video upload
    # ------------------------------------------------------------------

    def to_gemini_file_ref(self, api_key: str | None = None) -> Any:
        """Upload the video to the Gemini Files API and return a FileRef.

        Suitable for Gemini 2.x/3.x models that accept native video input.
        The uploaded file is valid for 48 hours.

        Args:
            api_key: Google API key (falls back to ``GOOGLE_API_KEY`` env var).

        Returns:
            :class:`~effgen.models.gemini_files.FileRef`

        Raises:
            RuntimeError: If the upload fails or google-genai is not installed.
        """
        from effgen.models.gemini_files import upload_file
        video_path = self._resolve_path()
        return upload_file(video_path, api_key=api_key, mime_type=self.mime_type)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_path(self) -> str:
        """Return a local filesystem path to the video, downloading if needed."""
        if self._tmp_path is not None:
            return self._tmp_path

        src = self._source

        if isinstance(src, bytes):
            if not src:
                raise InvalidMultimodalContent("video", "video bytes are empty")
            suffix = _mime_to_ext(self.mime_type)
            tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tf.write(src)
            tf.flush()
            tf.close()
            self._tmp_path = tf.name
            self._own_tmp = True
            return self._tmp_path

        if isinstance(src, Path) or (
            isinstance(src, str)
            and not src.startswith(("http://", "https://"))
        ):
            path = str(src)
            if not os.path.exists(path):
                raise InvalidMultimodalContent("video", f"File not found: {path}")
            self._tmp_path = path
            return self._tmp_path

        if isinstance(src, str) and src.startswith(("http://", "https://")):
            logger.info("video_pre: downloading video from URL '%s'", src)
            suffix = _mime_to_ext(self.mime_type)
            tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                with urllib.request.urlopen(src, timeout=60) as response:
                    tf.write(response.read())
            except Exception as exc:
                tf.close()
                os.unlink(tf.name)
                raise InvalidMultimodalContent(
                    "video", f"Could not download video from URL: {exc}"
                ) from exc
            tf.flush()
            tf.close()
            self._tmp_path = tf.name
            self._own_tmp = True
            return self._tmp_path

        raise InvalidMultimodalContent(
            "video",
            f"Unsupported source type: {type(src)}. "
            "Pass bytes, a local path (str/Path), or an HTTP(S) URL.",
        )

    def _has_audio_stream(self, video_path: str) -> bool:
        """Return True if the video has at least one audio stream."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            # If ffprobe is absent, attempt extraction anyway (ffmpeg will report the failure)
            return True
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except Exception:
            return False

    def cleanup(self) -> None:
        """Remove any temporary files created by this instance."""
        if self._own_tmp and self._tmp_path and os.path.exists(self._tmp_path):
            try:
                os.unlink(self._tmp_path)
                logger.debug("video_pre: removed temp file '%s'", self._tmp_path)
            except OSError as exc:
                logger.debug("video_pre: cleanup failed: %s", exc)
            self._tmp_path = None

    def __del__(self) -> None:
        self.cleanup()


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

def sample_frames(
    source: str | Path | bytes,
    fps: float = 1.0,
    max_frames: int = 16,
    mime_type: str | None = None,
) -> list[ImagePart]:
    """Convenience wrapper — create a VideoSource and sample frames.

    Args:
        source:     Local path, URL, or raw bytes.
        fps:        Frames per second to extract (default 1).
        max_frames: Maximum frames to return (default 16).
        mime_type:  Override inferred MIME type.

    Returns:
        List of :class:`~effgen.core.messages.ImagePart` (JPEG).
    """
    vs = VideoSource(source, mime_type=mime_type)
    return vs.sample_frames(fps=fps, max_frames=max_frames)


def extract_audio(
    source: str | Path | bytes,
    target_mime: str = "audio/mp3",
    mime_type: str | None = None,
) -> AudioPart | None:
    """Convenience wrapper — extract the audio track from a video.

    Args:
        source:      Local path, URL, or raw bytes.
        target_mime: Desired audio MIME (default ``"audio/mp3"``).
        mime_type:   Override inferred video MIME type.

    Returns:
        :class:`~effgen.core.messages.AudioPart` or ``None`` if no audio track.
    """
    vs = VideoSource(source, mime_type=mime_type)
    return vs.extract_audio(target_mime=target_mime)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mime_to_ext(mime: str) -> str:
    """Return a file extension for a video MIME type."""
    mapping = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/x-matroska": ".mkv",
        "video/mpeg": ".mpg",
        "video/ogg": ".ogv",
        "video/3gpp": ".3gp",
    }
    return mapping.get(mime, ".mp4")
