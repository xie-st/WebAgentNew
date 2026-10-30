"""
Multimodal input helpers for effGen.

image_from / audio_from / video_from each accept the widest possible set of
input types (bytes, path, URL, PIL.Image, np.ndarray) and return the
corresponding ContentPart.
"""

from __future__ import annotations

import io
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image as PILImage

from effgen.core.messages import AudioPart, ImagePart, VideoPart
from effgen.errors import InvalidMultimodalContent

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    ImageSource: TypeAlias = bytes | str | Path | PILImage.Image | np.ndarray  # noqa: UP040
else:
    ImageSource: TypeAlias = object  # noqa: UP040
AudioSource: TypeAlias = bytes | str | Path  # noqa: UP040
VideoSource: TypeAlias = bytes | str | Path  # noqa: UP040


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_image_mime(data: bytes) -> str:
    """Detect image MIME from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # safe fallback for PIL-produced data


_MIME_ALIASES: dict[str, str] = {
    "audio/x-wav": "audio/wav",
    "audio/x-m4a": "audio/m4a",
    "audio/x-flac": "audio/flac",
    "audio/x-mpeg": "audio/mpeg",
}


def _normalise_mime(mime: str) -> str:
    """Normalise non-canonical MIME aliases to the canonical form."""
    return _MIME_ALIASES.get(mime, mime)


def _detect_audio_mime(data: bytes) -> str:
    """Detect audio MIME from magic bytes / extension hint."""
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb" or data[:2] == b"\xff\xf3":
        return "audio/mpeg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/wav"


# A descriptive User-Agent so hosts that reject the default ``Python-urllib``
# client (e.g. Wikimedia) still serve the media.
_FETCH_USER_AGENT = "effGen/multimodal (+https://github.com/effgen)"


def _fetch_url(url: str, timeout: float = 30.0) -> bytes:
    """Fetch raw bytes from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": _FETCH_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — URL comes from user
        return resp.read()


#: How much of a rejected source is echoed back. A refused value can be a whole
#: base64 payload, and printing megabytes of it into a terminal or a log buries
#: the reason it was refused.
_ECHO_LIMIT = 120


def _abbreviate(value: str) -> str:
    """Shorten *value* for an error message, marking that it was cut."""
    text = str(value)
    if len(text) <= _ECHO_LIMIT:
        return text
    return f"{text[:_ECHO_LIMIT]}… ({len(text)} characters)"


def _decode_data_uri(source: str) -> tuple[bytes, str | None] | None:
    """Decode an inline ``data:`` URI into its bytes and declared MIME type.

    Returns ``None`` when *source* is not a ``data:`` URI, so a caller can fall
    through to its path/URL handling. Accepts the base64 form the
    OpenAI-compatible endpoint accepts (``data:image/png;base64,<payload>``)
    and the percent-encoded text form.

    Raises:
        InvalidMultimodalContent: if the URI is a ``data:`` URI but malformed.
    """
    if not source.startswith("data:"):
        return None

    header, _, payload = source[len("data:"):].partition(",")
    if not _:
        raise InvalidMultimodalContent(
            "source",
            f"Malformed data URI (no comma separating the header from the "
            f"payload): {_abbreviate(source)}",
        )

    parameters = header.split(";")
    is_base64 = parameters and parameters[-1].strip().lower() == "base64"
    mime = parameters[0].strip() or None

    try:
        if is_base64:
            import base64

            data = base64.b64decode(payload, validate=True)
        else:
            import urllib.parse

            data = urllib.parse.unquote_to_bytes(payload)
    except Exception as exc:
        raise InvalidMultimodalContent(
            "source", f"Could not decode the data URI payload: {exc}"
        ) from exc

    if not data:
        raise InvalidMultimodalContent("source", "The data URI carries no data.")
    return data, mime


def _path_to_bytes(path: str | Path) -> bytes:
    """Read *path* as bytes, reporting anything unreadable as a typed error.

    A value that is not a usable path at all — too long for the filesystem, an
    embedded NUL — makes the operating system raise before the file is even
    looked for; that is reported the same way a missing file is, so a caller
    sees one error type for "this source could not be read".
    """
    p = Path(path)
    accepted = (
        "Accepted sources are a file path, an http(s):// URL, an inline data: "
        "URI, or raw bytes."
    )
    try:
        exists = p.exists()
    except (OSError, ValueError) as exc:
        raise InvalidMultimodalContent(
            "source",
            f"Not a usable file path ({exc.strerror or exc}): "
            f"{_abbreviate(str(p))}. {accepted}",
        ) from exc
    if not exists:
        raise InvalidMultimodalContent(
            "source", f"File not found: {_abbreviate(str(p))}. {accepted}"
        )
    return p.read_bytes()


# ---------------------------------------------------------------------------
# image_from
# ---------------------------------------------------------------------------

def image_from(source: ImageSource, mime: str | None = None) -> ImagePart:
    """Create an ImagePart from bytes, path, URL, PIL.Image, or np.ndarray.

    Args:
        source: The image data in any supported form.
        mime: Override the detected MIME type (e.g. "image/webp").

    Returns:
        ImagePart ready to embed in a Message.
    """
    data: bytes

    if isinstance(source, bytes | bytearray):
        data = bytes(source)

    elif isinstance(source, str | Path):
        src = str(source)
        inline = _decode_data_uri(src)
        if inline is not None:
            data, declared = inline
            if mime is None and declared and declared.startswith("image/"):
                mime = declared
        elif src.startswith("http://") or src.startswith("https://"):
            data = _fetch_url(src)
            if mime is None:
                # Try to infer from URL extension
                ext = os.path.splitext(src.split("?")[0])[-1].lower()
                mime_guess, _ = mimetypes.guess_type(f"x{ext}")
                if mime_guess and mime_guess.startswith("image/"):
                    mime = mime_guess
        else:
            data = _path_to_bytes(source)
            if mime is None:
                mime_guess, _ = mimetypes.guess_type(str(source))
                if mime_guess and mime_guess.startswith("image/"):
                    mime = mime_guess

    else:
        # Try PIL.Image
        try:
            from PIL import Image as _PIL  # type: ignore[import]
            if isinstance(source, _PIL.Image):
                buf = io.BytesIO()
                fmt = source.format or "PNG"
                source.save(buf, format=fmt)
                data = buf.getvalue()
                if mime is None:
                    mime = f"image/{fmt.lower()}"
            else:
                raise TypeError
        except (ImportError, TypeError):
            pass
        else:
            detected = mime or _detect_image_mime(data)
            return ImagePart(image=data, mime=detected)

        # Try np.ndarray
        try:
            import numpy as np  # type: ignore[import]
            if isinstance(source, np.ndarray):
                try:
                    from PIL import Image as _PIL  # type: ignore[import]
                    img = _PIL.fromarray(source)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    data = buf.getvalue()
                    if mime is None:
                        mime = "image/png"
                except ImportError:
                    raise InvalidMultimodalContent(
                        "image",
                        "Pillow is required to convert np.ndarray to image bytes. "
                        "Install with: pip install Pillow"
                    )
            else:
                raise TypeError
        except (ImportError, TypeError):
            raise InvalidMultimodalContent(
                "image",
                f"Cannot convert source of type {type(source).__name__} to ImagePart. "
                "Accepted types: bytes, str (path or URL), Path, PIL.Image, np.ndarray."
            )
        else:
            detected = mime or _detect_image_mime(data)
            return ImagePart(image=data, mime=detected)

    detected = mime or _detect_image_mime(data)
    return ImagePart(image=data, mime=detected)


# ---------------------------------------------------------------------------
# audio_from
# ---------------------------------------------------------------------------

def audio_from(source: AudioSource, mime: str | None = None) -> AudioPart:
    """Create an AudioPart from bytes, path, or URL.

    Args:
        source: The audio data in any supported form.
        mime: Override the detected MIME type (e.g. "audio/wav").

    Returns:
        AudioPart ready to embed in a Message.
    """
    data: bytes

    if isinstance(source, bytes | bytearray):
        data = bytes(source)

    elif isinstance(source, str | Path):
        src = str(source)
        inline = _decode_data_uri(src)
        if inline is not None:
            data, declared = inline
            if mime is None and declared and declared.startswith("audio/"):
                mime = _normalise_mime(declared)
        elif src.startswith("http://") or src.startswith("https://"):
            data = _fetch_url(src)
            if mime is None:
                ext = os.path.splitext(src.split("?")[0])[-1].lower()
                mime_guess, _ = mimetypes.guess_type(f"x{ext}")
                if mime_guess and mime_guess.startswith("audio/"):
                    mime = _normalise_mime(mime_guess)
        else:
            if mime is None:
                mime_guess, _ = mimetypes.guess_type(str(source))
                if mime_guess and mime_guess.startswith("audio/"):
                    mime = _normalise_mime(mime_guess)
            data = _path_to_bytes(source)

    else:
        raise InvalidMultimodalContent(
            "audio",
            f"Cannot convert source of type {type(source).__name__} to AudioPart. "
            "Accepted types: bytes, str (path or URL), Path."
        )

    detected = mime or _detect_audio_mime(data)
    return AudioPart(audio=data, mime=detected)


# ---------------------------------------------------------------------------
# video_from
# ---------------------------------------------------------------------------

def video_from(
    source: VideoSource,
    fps: float = 1.0,
    mime: str = "image/jpeg",
    max_frames: int = 16,
) -> VideoPart:
    """Create a VideoPart by sampling keyframes from a video file.

    Requires ffmpeg to be installed on the system PATH.

    Args:
        source: Path, URL, or raw bytes of the video.
        fps: Frames per second to sample (default 1).
        mime: MIME type for each extracted frame (default image/jpeg).
        max_frames: Hard cap on extracted frames (default 16).

    Returns:
        VideoPart containing sampled frame bytes.
    """
    if fps <= 0:
        raise InvalidMultimodalContent("video_frames", "fps must be positive")
    if max_frames < 1:
        raise InvalidMultimodalContent("video_frames", "max_frames must be at least 1")

    from effgen.multimodal.video_pre import VideoSource as _VideoSource

    if isinstance(source, str):
        inline = _decode_data_uri(source)
        if inline is not None:
            source = inline[0]
        elif source.startswith(("http://", "https://")):
            source = _fetch_url(source)

    vs = _VideoSource(source)
    try:
        frames = vs.sample_frames(fps=fps, max_frames=max_frames)
        return VideoPart(frames=[frame.image for frame in frames], fps=fps, mime=mime)
    finally:
        vs.cleanup()


# ---------------------------------------------------------------------------
# Bare-path / URL coercion
# ---------------------------------------------------------------------------

# Extension → media kind, for the common cases where mimetypes can't help
# (or guesses differently across platforms).
_AUDIO_EXTS = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".webm", ".weba",
})
_VIDEO_EXTS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv",
})
_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif",
})


def _media_kind_from_name(name: str) -> str | None:
    """Best-effort image/audio/video classification from a path or URL string.

    Returns ``"image"``, ``"audio"``, ``"video"`` or ``None`` (unknown).
    """
    # An inline data: URI declares its own type in the header; nothing after the
    # comma is a filename, so the extension maps below would misread the payload.
    if name.startswith("data:"):
        declared = name[len("data:"):].split(";", 1)[0].split(",", 1)[0].strip()
        top = declared.split("/", 1)[0]
        return top if top in ("image", "audio", "video") else None

    # Strip any URL query/fragment before looking at the extension.
    base = name.split("?", 1)[0].split("#", 1)[0]
    ext = os.path.splitext(base)[1].lower()
    # ``.webm`` is overloaded (audio or video); the extension maps are checked
    # video-first so a bare ``clip.webm`` becomes a VideoPart.
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _IMAGE_EXTS:
        return "image"
    guessed, _ = mimetypes.guess_type(base)
    if guessed:
        top = guessed.split("/", 1)[0]
        if top in ("image", "audio", "video"):
            return top
    return None


def _document_extension(name: str) -> str | None:
    """Return the extension if ``name`` looks like a document effGen can read
    via ingestion (PDF/DOCX/XLSX/text/...) rather than as image/audio/video.

    Returns ``None`` for anything not recognized as a document format, so
    callers can fall through to their own "unknown media type" handling.
    """
    base = name.split("?", 1)[0].split("#", 1)[0]
    ext = os.path.splitext(base)[1].lower()
    if not ext:
        return None
    try:
        from effgen.rag.ingest import LOADERS
    except ImportError:
        return None
    return ext if ext in LOADERS else None


def part_from(source: str | Path) -> ImagePart | AudioPart | VideoPart:
    """Wrap a bare file path or URL as the matching multimodal ContentPart.

    Detects image/audio/video from the file extension (falling back to the
    MIME type) and delegates to :func:`image_from`, :func:`audio_from`, or
    :func:`video_from`. Use this when you have a path/URL and don't want to pick
    the specific helper yourself; :meth:`Agent.run` applies it automatically to
    bare paths passed in ``inputs=``.

    Raises:
        InvalidMultimodalContent: if the media kind can't be determined from the
            name — pass ``image_from(...)`` / ``audio_from(...)`` /
            ``video_from(...)`` explicitly in that case. A recognized document
            extension (PDF/DOCX/XLSX/text/...) raises a message that points at
            RAG ingestion or the ``pdf``/``excel`` tools instead, since none of
            the three media wrappers fit a document.
    """
    name = str(source)
    kind = _media_kind_from_name(name)
    if kind == "image":
        return image_from(source)
    if kind == "audio":
        return audio_from(source)
    if kind == "video":
        return video_from(source)
    doc_ext = _document_extension(name)
    if doc_ext:
        raise InvalidMultimodalContent(
            "source",
            f"{name!r} is a document ('{doc_ext}'), not an image, audio, or "
            "video file. Read it with create_agent(\"rag\", "
            f"knowledge_base=[{name!r}]) to answer questions grounded in its "
            "content, or extract its text directly with the 'pdf'/'excel' tool.",
        )
    raise InvalidMultimodalContent(
        "source",
        f"Could not infer the media type of {name!r} from its extension. "
        "Wrap it explicitly with image_from(...), audio_from(...), or "
        "video_from(...) (importable from effgen).",
    )
