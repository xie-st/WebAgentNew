"""
YouTube transcript tool for the effGen framework.

Backend: youtube-transcript-api (no auth required, works on public videos).
No authentication required. Works with public YouTube videos that have captions.

Operations:
- get_transcript: retrieve transcript/captions for a video
- list_available_languages: list languages for which transcripts are available
- translated: get transcript translated to a target language
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

# Raised when no transcript is available (re-exported for callers)
class NoTranscriptAvailableError(Exception):
    """Raised when no transcript can be found for a video."""


def _extract_video_id(url_or_id: str) -> str:
    """Extract YouTube video ID from a URL or return the ID directly.

    Handles:
    - watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - /shorts/VIDEO_ID
    - Plain 11-character video IDs
    """
    url_or_id = url_or_id.strip()

    # Plain video ID (11 alphanumeric chars + _ -)
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
        "Expected a URL containing watch?v=, youtu.be/, or shorts/, "
        "or an 11-character video ID."
    )


def _find_ytdlp() -> str:
    """Find yt-dlp executable for transcript fallback."""
    python_bin = str(Path(sys.executable).parent)
    path = shutil.which("yt-dlp", path=python_bin + ":" + (shutil.os.environ.get("PATH", "")))
    if path:
        return path
    path = shutil.which("yt-dlp")
    if path:
        return path
    raise ImportError(
        "yt-dlp is required for transcript fallback: pip install yt-dlp. "
        "Install or upgrade with `python -m pip install -U yt-dlp`."
    )


def _ytdlp_hint(stderr: str) -> str:
    detail = (stderr or "no stderr output").strip()
    return (
        f"{detail[:300]} Install or upgrade yt-dlp with "
        "`python -m pip install -U yt-dlp`. YouTube changes frequently, "
        "so an outdated yt-dlp can break subtitle extraction."
    )


# Phrases yt-dlp uses for an age gate. A bare "age" test would also match
# "Unable to download API page", so a network failure would be reported as an
# age restriction.
_AGE_RESTRICTION_PHRASES = ("age-restricted", "age restricted", "confirm your age")


def _mentions_age_restriction(stderr: str) -> bool:
    """Whether *stderr* reports an age gate rather than some other failure."""
    lowered = stderr.lower()
    return any(phrase in lowered for phrase in _AGE_RESTRICTION_PHRASES)


def _metadata_with_ytdlp(video_id: str, timeout: int = 45) -> dict[str, Any]:
    ytdlp = _find_ytdlp()
    proc = subprocess.run(
        [
            ytdlp,
            "--dump-json",
            "--skip-download",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "Video unavailable" in stderr or "Private video" in stderr:
            raise ValueError(
                f"Video '{video_id}' is unavailable (private or deleted): "
                f"{stderr[:200]}. Check the id, or request a video that is "
                "publicly listed."
            )
        if "Sign in" in stderr or _mentions_age_restriction(stderr):
            raise ValueError(
                f"Video '{video_id}' requires sign-in or is age-restricted: "
                f"{stderr[:200]}. This tool fetches only publicly viewable "
                "captions — request a video that needs no sign-in."
            )
        raise RuntimeError(f"yt-dlp metadata fallback failed: {_ytdlp_hint(stderr)}")
    if not proc.stdout.strip():
        raise RuntimeError("yt-dlp metadata fallback returned no output")
    return json.loads(proc.stdout)


def _caption_languages(raw: dict[str, Any]) -> list[dict[str, Any]]:
    languages: dict[str, dict[str, Any]] = {}
    for source_key, generated in (("subtitles", False), ("automatic_captions", True)):
        for code, entries in (raw.get(source_key) or {}).items():
            if not entries:
                continue
            languages.setdefault(
                code,
                {
                    "language": code,
                    "language_code": code,
                    "is_generated": generated,
                    "is_translatable": source_key == "subtitles",
                    "source": "yt-dlp",
                },
            )
    return list(languages.values())


def _select_caption_entries(raw: dict[str, Any], lang: str) -> tuple[str, bool, list[dict[str, Any]]]:
    for source_key, generated in (("subtitles", False), ("automatic_captions", True)):
        captions = raw.get(source_key) or {}
        if lang in captions and captions[lang]:
            return lang, generated, captions[lang]

    # Accept regional variants for short language requests, e.g. en -> en-US.
    if "-" not in lang:
        for source_key, generated in (("subtitles", False), ("automatic_captions", True)):
            captions = raw.get(source_key) or {}
            for code, entries in captions.items():
                if code.split("-", 1)[0] == lang and entries:
                    return code, generated, entries

    available = sorted(set((raw.get("subtitles") or {}) | (raw.get("automatic_captions") or {})))
    raise NoTranscriptAvailableError(
        f"No transcript available for video '{raw.get('id', 'unknown')}' in language '{lang}'. "
        f"Available languages: {available[:20]}"
    )


def _download_caption_payload(entries: list[dict[str, Any]]) -> tuple[str, str]:
    preferred = ("json3", "vtt", "srt", "srv3", "srv2", "srv1", "ttml")
    by_ext = {entry.get("ext"): entry for entry in entries if entry.get("url")}
    errors: list[str] = []
    for ext in preferred:
        entry = by_ext.get(ext)
        if not entry:
            continue
        try:
            with safe_urlopen(
                entry["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=20
            ) as resp:
                return ext, resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{ext}: {exc}")
    raise ConnectionError(
        "yt-dlp found captions but could not download them. "
        f"Errors: {'; '.join(errors) or 'no downloadable caption URLs'}"
    )


def _clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_json3(payload: str) -> list[dict[str, Any]]:
    data = json.loads(payload)
    snippets = []
    for event in data.get("events", []):
        text = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
        text = _clean_caption_text(text)
        if not text:
            continue
        start_ms = event.get("tStartMs", 0) or 0
        duration_ms = event.get("dDurationMs", 0) or 0
        snippets.append(
            {
                "text": text,
                "start": start_ms / 1000,
                "duration": duration_ms / 1000,
            }
        )
    return snippets


def _timestamp_to_seconds(value: str) -> float:
    hours = 0
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        return 0.0
    return hours * 3600 + minutes * 60 + seconds


def _parse_text_captions(payload: str) -> list[dict[str, Any]]:
    snippets = []
    blocks = re.split(r"\n\s*\n", payload.replace("\r\n", "\n"))
    timing_re = re.compile(
        r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{3})\s+-->\s+"
        r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{3})"
    )
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or lines[0].upper().startswith("WEBVTT"):
            continue
        timing_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_idx is None:
            continue
        match = timing_re.search(lines[timing_idx])
        if not match:
            continue
        text = _clean_caption_text(" ".join(lines[timing_idx + 1:]))
        if not text:
            continue
        start = _timestamp_to_seconds(match.group("start"))
        end = _timestamp_to_seconds(match.group("end"))
        snippets.append({"text": text, "start": start, "duration": max(0.0, end - start)})
    return snippets


def _parse_caption_payload(ext: str, payload: str) -> list[dict[str, Any]]:
    if ext == "json3":
        snippets = _parse_json3(payload)
    else:
        snippets = _parse_text_captions(payload)
    if not snippets:
        raise NoTranscriptAvailableError(
            "Caption payload was empty or could not be parsed. Try another "
            "caption language for this video, or a video with captions."
        )
    return snippets


def _get_transcript_ytdlp_sync(video_id: str, lang: str) -> dict[str, Any]:
    raw = _metadata_with_ytdlp(video_id)
    selected_lang, is_generated, entries = _select_caption_entries(raw, lang)
    ext, payload = _download_caption_payload(entries)
    snippets = _parse_caption_payload(ext, payload)
    full_text = " ".join(s["text"] for s in snippets)
    return {
        "success": True,
        "data": {
            "video_id": video_id,
            "language": selected_lang,
            "language_code": selected_lang,
            "is_generated": is_generated,
            "snippet_count": len(snippets),
            "full_text": full_text,
            "snippets": snippets,
            "source": "yt-dlp",
        },
        "error": None,
    }


def _list_languages_ytdlp_sync(video_id: str) -> dict[str, Any]:
    raw = _metadata_with_ytdlp(video_id)
    langs = _caption_languages(raw)
    if not langs:
        raise NoTranscriptAvailableError(
            f"No transcript languages available for video '{video_id}'. "
            "Captions are absent or disabled for it — try another video."
        )
    return {
        "success": True,
        "data": {
            "video_id": video_id,
            "languages": langs,
            "count": len(langs),
            "source": "yt-dlp",
        },
        "error": None,
    }


def _get_transcript_sync(video_id: str, lang: str) -> dict[str, Any]:
    """Synchronous transcript fetch — runs in thread via asyncio.to_thread."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            IpBlocked,
            NoTranscriptFound,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError as e:
        raise ImportError(
            "youtube-transcript-api is required: pip install youtube-transcript-api"
        ) from e

    api = YouTubeTranscriptApi()
    try:
        ft = api.fetch(video_id, languages=[lang])
        snippets = ft.to_raw_data()
        full_text = " ".join(s["text"] for s in snippets)
        return {
            "success": True,
            "data": {
                "video_id": video_id,
                "language": ft.language,
                "language_code": ft.language_code,
                "is_generated": ft.is_generated,
                "snippet_count": len(snippets),
                "full_text": full_text,
                "snippets": snippets,
            },
            "error": None,
        }
    except (NoTranscriptFound, TranscriptsDisabled) as exc:
        raise NoTranscriptAvailableError(
            f"No transcript available for video '{video_id}' in language "
            f"'{lang}': {exc}. List the languages this video does carry, then "
            "request one of those."
        ) from exc
    except VideoUnavailable as exc:
        raise ValueError(
            f"Video '{video_id}' is unavailable (private or deleted): {exc}. "
            "Check the id, or request a video that is publicly listed."
        ) from exc
    except (IpBlocked, RequestBlocked) as exc:
        try:
            return _get_transcript_ytdlp_sync(video_id, lang)
        except Exception as fallback_exc:
            raise ConnectionError(
                "YouTube is blocking transcript requests from this IP, and the "
                f"yt-dlp subtitle fallback also failed: {fallback_exc}. Original error: {exc}"
            ) from fallback_exc


def _list_languages_sync(video_id: str) -> dict[str, Any]:
    """Synchronous language list — runs in thread via asyncio.to_thread."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            IpBlocked,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except ImportError as e:
        raise ImportError(
            "youtube-transcript-api is required: pip install youtube-transcript-api"
        ) from e

    api = YouTubeTranscriptApi()
    try:
        tl = api.list(video_id)
        langs = []
        for t in tl:
            langs.append(
                {
                    "language": t.language,
                    "language_code": t.language_code,
                    "is_generated": t.is_generated,
                    "is_translatable": t.is_translatable,
                }
            )
        return {
            "success": True,
            "data": {
                "video_id": video_id,
                "languages": langs,
                "count": len(langs),
            },
            "error": None,
        }
    except TranscriptsDisabled as exc:
        raise NoTranscriptAvailableError(
            f"Transcripts are disabled for video '{video_id}': {exc}. The "
            "uploader turned captions off — try another video."
        ) from exc
    except VideoUnavailable as exc:
        raise ValueError(
            f"Video '{video_id}' is unavailable: {exc}. Check the id, or "
            "request a video that is publicly listed."
        ) from exc
    except (IpBlocked, RequestBlocked) as exc:
        try:
            return _list_languages_ytdlp_sync(video_id)
        except Exception as fallback_exc:
            raise ConnectionError(
                "YouTube is blocking transcript-language requests from this IP, and the "
                f"yt-dlp fallback also failed: {fallback_exc}. Original error: {exc}"
            ) from fallback_exc


def _translated_sync(video_id: str, target_lang: str) -> dict[str, Any]:
    """Synchronous translated transcript fetch."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            IpBlocked,
            NoTranscriptFound,
            NotTranslatable,
            RequestBlocked,
            TranscriptsDisabled,
            TranslationLanguageNotAvailable,
            VideoUnavailable,
        )
    except ImportError as e:
        raise ImportError(
            "youtube-transcript-api is required: pip install youtube-transcript-api"
        ) from e

    api = YouTubeTranscriptApi()
    try:
        tl = api.list(video_id)
        # Find a translatable transcript
        transcript = None
        for t in tl:
            if t.is_translatable:
                transcript = t
                break
        if transcript is None:
            raise NoTranscriptAvailableError(
                f"No translatable transcript found for video '{video_id}'. "
                "Fetch the transcript in a language the video carries and "
                "translate it yourself."
            )
        translated = transcript.translate(target_lang)
        ft = translated.fetch()
        snippets = ft.to_raw_data()
        full_text = " ".join(s["text"] for s in snippets)
        return {
            "success": True,
            "data": {
                "video_id": video_id,
                "source_language": transcript.language_code,
                "target_language": target_lang,
                "language": ft.language,
                "language_code": ft.language_code,
                "snippet_count": len(snippets),
                "full_text": full_text,
                "snippets": snippets,
            },
            "error": None,
        }
    except (NoTranscriptFound, TranscriptsDisabled) as exc:
        raise NoTranscriptAvailableError(
            f"No transcript available for video '{video_id}': {exc}. "
            "Captions are absent or disabled for it — try another video."
        ) from exc
    except (NotTranslatable, TranslationLanguageNotAvailable) as exc:
        raise ValueError(
            f"Translation to '{target_lang}' not available for video '{video_id}': {exc}"
        ) from exc
    except VideoUnavailable as exc:
        raise ValueError(
            f"Video '{video_id}' is unavailable: {exc}. Check the id, or "
            "request a video that is publicly listed."
        ) from exc
    except (IpBlocked, RequestBlocked) as exc:
        try:
            fallback = _get_transcript_ytdlp_sync(video_id, target_lang)
            fallback["data"]["source_language"] = fallback["data"].get("language_code")
            fallback["data"]["target_language"] = target_lang
            return fallback
        except Exception as fallback_exc:
            raise ConnectionError(
                "YouTube is blocking translated transcript requests from this IP, and the "
                f"yt-dlp caption fallback also failed: {fallback_exc}. Original error: {exc}"
            ) from fallback_exc


class YouTubeTranscriptTool(BaseTool):
    """Fetch YouTube video transcripts/captions without requiring a Google API key."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="youtube_transcript",
                description=(
                    "Fetch YouTube video transcripts/captions without a Google API key. "
                    "Handles watch?v=, youtu.be/, and shorts/ URL formats. "
                    "Operations: get_transcript (fetch captions in a language), "
                    "list_available_languages (show which languages have captions), "
                    "translated (fetch captions translated to a target language). "
                    "Raises NoTranscriptAvailableError when no captions exist."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="get_transcript",
                        enum=["get_transcript", "list_available_languages", "translated"],
                    ),
                    ParameterSpec(
                        name="video_id",
                        type=ParameterType.STRING,
                        description=(
                            "YouTube video ID or URL. Accepts watch?v=, youtu.be/, "
                            "shorts/, or plain 11-char video ID."
                        ),
                        required=True,
                        min_length=3,
                        max_length=2048,
                    ),
                    ParameterSpec(
                        name="lang",
                        type=ParameterType.STRING,
                        description="Language code for transcript (e.g. 'en', 'fr', 'ja').",
                        required=False,
                        default="en",
                        min_length=2,
                        max_length=10,
                    ),
                    ParameterSpec(
                        name="target_lang",
                        type=ParameterType.STRING,
                        description="Target language code for 'translated' operation.",
                        required=False,
                        min_length=2,
                        max_length=10,
                    ),
                ],
                timeout_seconds=30,
                tags=["youtube", "transcript", "captions", "video", "content", "free"],
                examples=[
                    {"operation": "get_transcript", "video_id": "dQw4w9WgXcQ"},
                    {"operation": "list_available_languages", "video_id": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                    {"operation": "translated", "video_id": "dQw4w9WgXcQ", "target_lang": "fr"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str = "get_transcript",
        video_id: str = "",
        lang: str = "en",
        target_lang: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not video_id:
            raise ValueError("'video_id' is required")

        op = operation.lower()
        if op not in ("get_transcript", "list_available_languages", "translated"):
            raise ValueError(f"Unknown operation: {operation!r}")

        try:
            vid = _extract_video_id(video_id)
        except ValueError as exc:
            return {"success": False, "data": None, "error": str(exc)}

        try:
            if op == "get_transcript":
                result = await asyncio.to_thread(_get_transcript_sync, vid, lang)
            elif op == "list_available_languages":
                result = await asyncio.to_thread(_list_languages_sync, vid)
            else:  # translated
                if not target_lang:
                    raise ValueError("operation='translated' requires 'target_lang'")
                result = await asyncio.to_thread(_translated_sync, vid, target_lang)
        except NoTranscriptAvailableError as exc:
            return {"success": False, "data": None, "error": f"NoTranscriptAvailableError: {exc}"}
        except (ValueError, ConnectionError) as exc:
            return {"success": False, "data": None, "error": str(exc)}
        except Exception as exc:
            logger.exception("Unexpected error in YouTubeTranscriptTool")
            return {"success": False, "data": None, "error": f"Unexpected error: {exc}"}

        result["operation"] = op
        result["video_id"] = vid
        return result
