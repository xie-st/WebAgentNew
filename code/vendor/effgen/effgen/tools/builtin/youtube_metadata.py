"""
YouTube metadata tool for the effGen framework.

Backend: yt-dlp in metadata-only mode (--dump-json --skip-download).
No authentication required for public content.

Operations:
- metadata: get metadata for a video
- channel: get metadata for a channel's videos

Note: yt-dlp may require occasional updates as YouTube changes its internal API.
Install: pip install yt-dlp
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
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

_YTDLP_FIELDS = [
    "id", "title", "description", "channel", "channel_id", "channel_url",
    "uploader", "uploader_id", "duration", "view_count", "like_count",
    "comment_count", "upload_date", "timestamp", "categories", "tags",
    "thumbnail", "webpage_url", "age_limit", "live_status", "availability",
    "language", "subtitles", "automatic_captions",
]


def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from URL or return the ID directly."""
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/v/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url_or_id)
        if m:
            return m.group(1)

    raise ValueError(
        f"Could not extract a valid YouTube video ID from: {url_or_id!r}. "
        "Expected watch?v=, youtu.be/, shorts/, or an 11-character video ID."
    )


def _find_ytdlp() -> str:
    """Find yt-dlp executable, preferring the current Python env's bin directory."""
    # Prefer yt-dlp co-located with the running Python interpreter
    python_bin = str(Path(sys.executable).parent)
    path = shutil.which("yt-dlp", path=python_bin + ":" + (shutil.os.environ.get("PATH", "")))
    if path:
        return path
    # Fallback to PATH search
    path = shutil.which("yt-dlp")
    if path:
        return path
    raise ImportError(
        "yt-dlp is required: pip install yt-dlp. "
        "Make sure it is available on your PATH."
    )


def _ytdlp_upgrade_message(stderr: str) -> str:
    """Return a concise, actionable yt-dlp failure message."""
    detail = (stderr or "no stderr output").strip()
    hint = (
        "Install or upgrade yt-dlp with `python -m pip install -U yt-dlp`. "
        "YouTube changes frequently, so an outdated yt-dlp is a common cause "
        "of extraction failures."
    )
    if any(
        marker in detail.lower()
        for marker in (
            "update",
            "unable to extract",
            "signature",
            "nsig",
            "player",
            "http error 403",
            "http error 429",
        )
    ):
        return f"{detail[:300]} {hint}"
    return f"{detail[:300]} {hint}"


def _filter_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract the most useful metadata fields from the raw yt-dlp output."""
    out: dict[str, Any] = {}
    for field in _YTDLP_FIELDS:
        if field in raw:
            out[field] = raw[field]

    # Summarize captions availability without dumping all URLs
    if "subtitles" in raw:
        out["subtitle_languages"] = list(raw["subtitles"].keys())
    if "automatic_captions" in raw:
        out["auto_caption_languages"] = list(raw["automatic_captions"].keys())

    # Remove raw subtitle/caption dicts to keep output compact
    out.pop("subtitles", None)
    out.pop("automatic_captions", None)

    return out


# Phrases yt-dlp uses for an age gate. A bare "age" test would also match
# "Unable to download API page", so a network failure would be reported as an
# age restriction.
_AGE_RESTRICTION_PHRASES = ("age-restricted", "age restricted", "confirm your age")


def _mentions_age_restriction(stderr: str) -> bool:
    """Whether *stderr* reports an age gate rather than some other failure."""
    lowered = stderr.lower()
    return any(phrase in lowered for phrase in _AGE_RESTRICTION_PHRASES)


def _metadata_sync(url: str, timeout: int = 45) -> dict[str, Any]:
    """Synchronous yt-dlp metadata fetch."""
    ytdlp = _find_ytdlp()

    cmd = [
        ytdlp,
        "--dump-json",
        "--skip-download",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"yt-dlp timed out after {timeout}s fetching metadata for {url!r}")
    except FileNotFoundError:
        raise ImportError("yt-dlp not found on PATH: pip install yt-dlp")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "Video unavailable" in stderr or "Private video" in stderr:
            raise ValueError(f"Video is unavailable (private or deleted): {url!r}")
        if "Sign in" in stderr or _mentions_age_restriction(stderr):
            raise ValueError(
                f"Video requires sign-in or is age-restricted: {url!r}. {stderr[:200]}"
            )
        raise RuntimeError(
            f"yt-dlp failed (code {proc.returncode}): {_ytdlp_upgrade_message(stderr)}"
        )

    if not proc.stdout.strip():
        raise RuntimeError(f"yt-dlp returned no output for {url!r}")

    raw = json.loads(proc.stdout)
    return _filter_metadata(raw)


def _channel_sync(channel_url: str, max_videos: int = 10, timeout: int = 60) -> list[dict[str, Any]]:
    """Synchronous yt-dlp channel metadata fetch (first N videos)."""
    ytdlp = _find_ytdlp()

    cmd = [
        ytdlp,
        "--dump-json",
        "--skip-download",
        "--quiet",
        "--no-warnings",
        "--playlist-end", str(max_videos),
        channel_url,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"yt-dlp timed out after {timeout}s fetching channel {channel_url!r}")
    except FileNotFoundError:
        raise ImportError("yt-dlp not found on PATH: pip install yt-dlp")

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(
            f"yt-dlp failed fetching channel (code {proc.returncode}): "
            f"{_ytdlp_upgrade_message(stderr)}"
        )

    videos = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            if raw.get("_type") in ("playlist", None) and "entries" in raw:
                continue
            videos.append(_filter_metadata(raw))
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON line from yt-dlp: %s", line[:80])

    return videos


class YouTubeMetadataTool(BaseTool):
    """Fetch YouTube video or channel metadata using yt-dlp (no auth required)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="youtube_metadata",
                description=(
                    "Fetch YouTube video or channel metadata without a Google API key. "
                    "Uses yt-dlp in metadata-only mode (no download). "
                    "Operations: metadata (get video info: title, description, views, duration, tags, etc.), "
                    "channel (list recent videos from a channel URL). "
                    "Works on any public video or channel. "
                    "Caveat: yt-dlp may need occasional updates as YouTube evolves."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="metadata",
                        enum=["metadata", "channel"],
                    ),
                    ParameterSpec(
                        name="video_id",
                        type=ParameterType.STRING,
                        description=(
                            "YouTube video ID or URL for 'metadata' operation. "
                            "Accepts watch?v=, youtu.be/, shorts/, or plain video ID."
                        ),
                        required=False,
                        min_length=3,
                        max_length=2048,
                    ),
                    ParameterSpec(
                        name="channel_url",
                        type=ParameterType.STRING,
                        description=(
                            "YouTube channel URL for 'channel' operation "
                            "(e.g. https://www.youtube.com/@RickAstleyYT)."
                        ),
                        required=False,
                        min_length=5,
                        max_length=2048,
                    ),
                    ParameterSpec(
                        name="max_videos",
                        type=ParameterType.INTEGER,
                        description="Max number of videos to return for 'channel' operation.",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=50,
                    ),
                ],
                timeout_seconds=60,
                tags=["youtube", "metadata", "video", "channel", "yt-dlp", "free"],
                examples=[
                    {"operation": "metadata", "video_id": "dQw4w9WgXcQ"},
                    {"operation": "metadata", "video_id": "https://youtu.be/dQw4w9WgXcQ"},
                    {"operation": "channel", "channel_url": "https://www.youtube.com/@RickAstleyYT", "max_videos": 5},
                ],
            )
        )

    async def _execute(
        self,
        operation: str = "metadata",
        video_id: str | None = None,
        channel_url: str | None = None,
        max_videos: int = 10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        op = operation.lower()
        if op not in ("metadata", "channel"):
            raise ValueError(f"Unknown operation: {operation!r}")

        try:
            if op == "metadata":
                if not video_id:
                    raise ValueError(
                        "'video_id' is required for operation='metadata'. "
                        "Pass video_id, or the full watch URL, which the tool "
                        "reads the id from."
                    )
                try:
                    vid = _extract_video_id(video_id)
                except ValueError as exc:
                    return {"success": False, "data": None, "error": str(exc)}

                url = f"https://www.youtube.com/watch?v={vid}"
                meta = await asyncio.to_thread(_metadata_sync, url)
                return {
                    "success": True,
                    "operation": op,
                    "data": meta,
                    "error": None,
                }

            else:  # channel
                if not channel_url:
                    raise ValueError(
                        "'channel_url' is required for operation='channel'. "
                        "Pass channel_url as the channel's page address."
                    )
                videos = await asyncio.to_thread(_channel_sync, channel_url, max_videos)
                return {
                    "success": True,
                    "operation": op,
                    "data": {
                        "channel_url": channel_url,
                        "videos": videos,
                        "count": len(videos),
                    },
                    "error": None,
                }

        except (ValueError, ImportError) as exc:
            return {"success": False, "data": None, "error": str(exc)}
        except (TimeoutError, RuntimeError, ConnectionError) as exc:
            return {"success": False, "data": None, "error": str(exc)}
        except Exception as exc:
            logger.exception("Unexpected error in YouTubeMetadataTool")
            return {"success": False, "data": None, "error": f"Unexpected error: {exc}"}
