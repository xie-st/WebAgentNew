"""Tests for YouTubeTranscriptTool.

Live tests hit the real YouTube transcript API.
Network/IP-block failures are treated as skips.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import YouTubeTranscriptTool
from effgen.tools.builtin.youtube_transcript import (
    NoTranscriptAvailableError,
    _extract_video_id,
    _parse_json3,
    _select_caption_entries,
)

# A stable public video with English captions (Rick Astley — Never Gonna Give You Up)
STABLE_VIDEO_ID = "dQw4w9WgXcQ"
STABLE_VIDEO_URL_WATCH = f"https://www.youtube.com/watch?v={STABLE_VIDEO_ID}"
STABLE_VIDEO_URL_SHORT = f"https://youtu.be/{STABLE_VIDEO_ID}"
STABLE_VIDEO_URL_SHORTS = f"https://www.youtube.com/shorts/{STABLE_VIDEO_ID}"


def _has_network(host: str = "www.youtube.com", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")

_SKIP_MARKERS = (
    "IpBlocked",
    "RequestBlocked",
    "YouTube is blocking",
    "timed out",
    "Timeout",
    "Connection",
    "NoTranscriptAvailableError",
)


def _run(coro):
    return asyncio.run(coro)


def _ok(result: ToolResult) -> dict:
    assert isinstance(result, ToolResult)
    if not result.success and result.error and any(m in result.error for m in _SKIP_MARKERS):
        pytest.skip(f"transient error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    out = result.output
    if isinstance(out, dict) and not out.get("success"):
        err = out.get("error", "")
        if err and any(m in err for m in _SKIP_MARKERS):
            pytest.skip(f"transient inner error: {err}")
    return out


# ---------------------------------------------------------------------------
# URL parsing tests (no network required)
# ---------------------------------------------------------------------------

def test_extract_plain_id():
    assert _extract_video_id(STABLE_VIDEO_ID) == STABLE_VIDEO_ID


def test_extract_watch_url():
    assert _extract_video_id(STABLE_VIDEO_URL_WATCH) == STABLE_VIDEO_ID


def test_extract_short_url():
    assert _extract_video_id(STABLE_VIDEO_URL_SHORT) == STABLE_VIDEO_ID


def test_extract_shorts_url():
    assert _extract_video_id(STABLE_VIDEO_URL_SHORTS) == STABLE_VIDEO_ID


def test_extract_embed_url():
    url = f"https://www.youtube.com/embed/{STABLE_VIDEO_ID}"
    assert _extract_video_id(url) == STABLE_VIDEO_ID


def test_extract_invalid_raises():
    with pytest.raises(ValueError, match="Could not extract"):
        _extract_video_id("not-a-valid-url-or-id")


def test_no_caption_selection_raises_helpful_error():
    with pytest.raises(NoTranscriptAvailableError, match="No transcript available"):
        _select_caption_entries(
            {"id": "NO_CAPTION1", "subtitles": {}, "automatic_captions": {}},
            "en",
        )


def test_parse_json3_caption_payload():
    payload = """
    {
      "events": [
        {"tStartMs": 1000, "dDurationMs": 2500, "segs": [{"utf8": "hello "}, {"utf8": "world"}]},
        {"tStartMs": 4000, "dDurationMs": 1000, "segs": [{"utf8": "\\n"}]}
      ]
    }
    """
    snippets = _parse_json3(payload)
    assert snippets == [{"text": "hello world", "start": 1.0, "duration": 2.5}]


# ---------------------------------------------------------------------------
# Metadata / unit tests
# ---------------------------------------------------------------------------

def test_transcript_metadata():
    tool = YouTubeTranscriptTool()
    assert tool.metadata.name == "youtube_transcript"
    assert "youtube" in tool.metadata.tags
    assert "transcript" in tool.metadata.tags


def _is_failure(r: ToolResult) -> bool:
    """Return True if the tool result indicates failure (at ToolResult or inner dict level)."""
    if not r.success:
        return True
    out = r.output
    if isinstance(out, dict) and out.get("success") is False:
        return True
    return False


def _get_error(r: ToolResult) -> str:
    """Return error message from ToolResult or inner dict."""
    if r.error:
        return r.error
    if isinstance(r.output, dict):
        return r.output.get("error") or ""
    return ""


def test_transcript_missing_video_id_fails():
    r = _run(YouTubeTranscriptTool().execute(operation="get_transcript"))
    assert _is_failure(r)


def test_transcript_unknown_operation_fails():
    r = _run(YouTubeTranscriptTool().execute(operation="nope", video_id=STABLE_VIDEO_ID))
    assert _is_failure(r)


def test_transcript_translated_requires_target_lang():
    r = _run(YouTubeTranscriptTool().execute(operation="translated", video_id=STABLE_VIDEO_ID))
    assert _is_failure(r)
    assert "target_lang" in _get_error(r)


def test_translated_fallback_preserves_target_language(monkeypatch):
    """IP-block fallback must still return translated-operation metadata."""
    import youtube_transcript_api
    from youtube_transcript_api._errors import IpBlocked

    from effgen.tools.builtin import youtube_transcript as yt

    class FakeApi:
        def list(self, video_id):
            raise IpBlocked(video_id)

    def fake_fallback(video_id, target_lang):
        return {
            "success": True,
            "data": {
                "video_id": video_id,
                "language": target_lang,
                "language_code": target_lang,
                "is_generated": False,
                "snippet_count": 1,
                "full_text": "bonjour",
                "snippets": [{"text": "bonjour", "start": 0.0, "duration": 1.0}],
                "source": "yt-dlp",
            },
            "error": None,
        }

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", lambda: FakeApi())
    monkeypatch.setattr(yt, "_get_transcript_ytdlp_sync", fake_fallback)

    result = yt._translated_sync(STABLE_VIDEO_ID, "fr")
    assert result["success"] is True
    assert result["data"]["target_language"] == "fr"
    assert result["data"]["source_language"] == "fr"


def test_transcript_invalid_url_fails():
    r = _run(YouTubeTranscriptTool().execute(operation="get_transcript", video_id="not-a-youtube-url-at-all"))
    assert _is_failure(r)
    assert _get_error(r)


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
def test_transcript_get_transcript():
    out = _ok(_run(YouTubeTranscriptTool().execute(
        operation="get_transcript",
        video_id=STABLE_VIDEO_ID,
        lang="en",
    )))
    assert out["success"] is True
    data = out["data"]
    assert data["video_id"] == STABLE_VIDEO_ID
    assert data["language_code"] == "en"
    assert data["snippet_count"] >= 50
    assert isinstance(data["snippets"], list)
    assert len(data["full_text"]) > 10
    # Each snippet must have text, start, duration
    s = data["snippets"][0]
    assert "text" in s
    assert "start" in s
    assert "duration" in s


@needs_net
def test_transcript_via_watch_url():
    out = _ok(_run(YouTubeTranscriptTool().execute(
        operation="get_transcript",
        video_id=STABLE_VIDEO_URL_WATCH,
    )))
    assert out["success"] is True
    assert out["data"]["video_id"] == STABLE_VIDEO_ID


@needs_net
def test_transcript_via_short_url():
    out = _ok(_run(YouTubeTranscriptTool().execute(
        operation="get_transcript",
        video_id=STABLE_VIDEO_URL_SHORT,
    )))
    assert out["success"] is True


@needs_net
def test_transcript_list_available_languages():
    out = _ok(_run(YouTubeTranscriptTool().execute(
        operation="list_available_languages",
        video_id=STABLE_VIDEO_ID,
    )))
    assert out["success"] is True
    data = out["data"]
    assert data["video_id"] == STABLE_VIDEO_ID
    assert data["count"] >= 2
    langs = data["languages"]
    assert isinstance(langs, list)
    lang_codes = [lang["language_code"] for lang in langs]
    assert "en" in lang_codes
    # Each language entry has required fields
    for lang in langs:
        assert "language" in lang
        assert "language_code" in lang
        assert "is_generated" in lang
        assert "is_translatable" in lang


@needs_net
def test_transcript_no_captions_returns_error():
    """Videos without captions should return a clean error."""
    # Use an invalid video ID that doesn't exist → VideoUnavailable or NoTranscript
    r = _run(YouTubeTranscriptTool().execute(
        operation="get_transcript",
        video_id="XXXXXXXXXXX",
        lang="en",
    ))
    assert _is_failure(r)
    assert _get_error(r)  # must have a meaningful error message


@needs_net
def test_transcript_translated_operation():
    """Translated transcript — may be blocked by IP on cloud hosts, skip if so."""
    r = _run(YouTubeTranscriptTool().execute(
        operation="translated",
        video_id=STABLE_VIDEO_ID,
        target_lang="fr",
    ))
    err = _get_error(r)
    if _is_failure(r):
        if any(m in err for m in _SKIP_MARKERS):
            pytest.skip(f"IP blocked or transcript unavailable: {err}")
        pytest.fail(f"translated failed: {err}")
    out = r.output
    assert out["success"] is True
    data = out["data"]
    assert data["target_language"] == "fr"
    assert data["snippet_count"] > 0


# ---------------------------------------------------------------------------
# yt-dlp failure classification
# ---------------------------------------------------------------------------

# What yt-dlp writes when the host cannot be resolved. It contains the word
# "page", so a bare "age" substring test reads it as an age restriction.
_DNS_FAILURE_STDERR = (
    "ERROR: [youtube] dQw4w9WgXcQ: Unable to download API page: "
    "HTTPSConnection(host='www.youtube.com', port=443): Failed to resolve "
    "'www.youtube.com' ([Errno -3] Temporary failure in name resolution)"
)


def test_a_network_failure_is_not_read_as_an_age_restriction():
    from effgen.tools.builtin.youtube_transcript import _mentions_age_restriction

    assert _mentions_age_restriction(_DNS_FAILURE_STDERR) is False


def test_an_age_gate_is_still_recognized():
    from effgen.tools.builtin.youtube_transcript import _mentions_age_restriction

    for stderr in (
        "ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate.",
        "ERROR: [youtube] abc: This video is age-restricted.",
    ):
        assert _mentions_age_restriction(stderr) is True, stderr
