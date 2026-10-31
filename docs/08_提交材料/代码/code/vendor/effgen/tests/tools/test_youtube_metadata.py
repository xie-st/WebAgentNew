"""Tests for YouTubeMetadataTool.

Live tests hit the real YouTube via yt-dlp.
Transient failures (timeouts, availability issues) are treated as skips.
"""

from __future__ import annotations

import asyncio
import shutil
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import YouTubeMetadataTool
from effgen.tools.builtin.youtube_metadata import _extract_video_id, _ytdlp_upgrade_message

STABLE_VIDEO_ID = "dQw4w9WgXcQ"
STABLE_VIDEO_URL = f"https://www.youtube.com/watch?v={STABLE_VIDEO_ID}"

YTDLP_AVAILABLE = shutil.which("yt-dlp") is not None


def _has_network(host: str = "www.youtube.com", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")
needs_ytdlp = pytest.mark.skipif(not YTDLP_AVAILABLE, reason="yt-dlp not installed")

_SKIP_MARKERS = (
    "timed out",
    "Timeout",
    "Connection",
    "unavailable",
    "Video unavailable",
    "sign-in",
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
    assert _extract_video_id(STABLE_VIDEO_URL) == STABLE_VIDEO_ID


def test_extract_short_url():
    assert _extract_video_id(f"https://youtu.be/{STABLE_VIDEO_ID}") == STABLE_VIDEO_ID


def test_extract_shorts_url():
    assert _extract_video_id(f"https://www.youtube.com/shorts/{STABLE_VIDEO_ID}") == STABLE_VIDEO_ID


def test_extract_invalid_raises():
    with pytest.raises(ValueError, match="Could not extract"):
        _extract_video_id("not-a-valid-url-or-id")


def test_ytdlp_upgrade_message_is_actionable():
    msg = _ytdlp_upgrade_message("ERROR: unable to extract player response")
    assert "python -m pip install -U yt-dlp" in msg
    assert "outdated yt-dlp" in msg


# ---------------------------------------------------------------------------
# Metadata / unit tests
# ---------------------------------------------------------------------------

def test_metadata_tool_metadata():
    tool = YouTubeMetadataTool()
    assert tool.metadata.name == "youtube_metadata"
    assert "youtube" in tool.metadata.tags
    assert "yt-dlp" in tool.metadata.tags


def _is_failure(r: ToolResult) -> bool:
    """Return True if tool result indicates failure at any level."""
    if not r.success:
        return True
    out = r.output
    if isinstance(out, dict) and out.get("success") is False:
        return True
    return False


def _get_error(r: ToolResult) -> str:
    if r.error:
        return r.error
    if isinstance(r.output, dict):
        return r.output.get("error") or ""
    return ""


def test_metadata_unknown_operation_fails():
    r = _run(YouTubeMetadataTool().execute(operation="nope", video_id=STABLE_VIDEO_ID))
    assert _is_failure(r)


def test_metadata_missing_video_id_fails():
    r = _run(YouTubeMetadataTool().execute(operation="metadata"))
    assert _is_failure(r)
    assert "video_id" in _get_error(r)


def test_metadata_channel_missing_url_fails():
    r = _run(YouTubeMetadataTool().execute(operation="channel"))
    assert _is_failure(r)
    assert "channel_url" in _get_error(r)


def test_metadata_invalid_url_fails():
    r = _run(YouTubeMetadataTool().execute(operation="metadata", video_id="not-a-youtube-url"))
    assert _is_failure(r)


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
@needs_ytdlp
def test_metadata_video():
    out = _ok(_run(YouTubeMetadataTool().execute(
        operation="metadata",
        video_id=STABLE_VIDEO_ID,
    )))
    assert out["success"] is True
    data = out["data"]
    assert data["id"] == STABLE_VIDEO_ID
    assert "title" in data
    assert len(data["title"]) > 0
    assert "duration" in data
    assert data["duration"] > 0
    assert "view_count" in data
    assert data["view_count"] > 0
    assert "channel" in data
    assert "upload_date" in data


@needs_net
@needs_ytdlp
def test_metadata_video_via_url():
    out = _ok(_run(YouTubeMetadataTool().execute(
        operation="metadata",
        video_id=STABLE_VIDEO_URL,
    )))
    assert out["success"] is True
    assert out["data"]["id"] == STABLE_VIDEO_ID


@needs_net
@needs_ytdlp
def test_metadata_video_via_short_url():
    out = _ok(_run(YouTubeMetadataTool().execute(
        operation="metadata",
        video_id=f"https://youtu.be/{STABLE_VIDEO_ID}",
    )))
    assert out["success"] is True
    assert out["data"]["id"] == STABLE_VIDEO_ID


@needs_net
@needs_ytdlp
def test_metadata_fields_present():
    """Key metadata fields must be present in the output."""
    out = _ok(_run(YouTubeMetadataTool().execute(
        operation="metadata",
        video_id=STABLE_VIDEO_ID,
    )))
    data = out["data"]
    for field in ["id", "title", "channel", "duration", "view_count", "webpage_url"]:
        assert field in data, f"Missing field: {field}"


@needs_net
@needs_ytdlp
def test_metadata_channel():
    """Channel operation should return a list of videos or skip if auth required."""
    r = _run(YouTubeMetadataTool().execute(
        operation="channel",
        channel_url="https://www.youtube.com/@RickAstleyYT",
        max_videos=3,
    ))
    err = _get_error(r)
    if _is_failure(r):
        # YouTube may require sign-in for channel pages on cloud IPs
        if any(m in err for m in ("Sign in", "sign-in", "bot", "403", "401", "timed out", "Timeout")):
            pytest.skip(f"Channel access blocked or requires auth: {err[:200]}")
        pytest.fail(f"Channel operation failed: {err}")
    out = r.output
    assert out["success"] is True
    data = out["data"]
    assert "videos" in data
    assert isinstance(data["videos"], list)
    assert data["count"] >= 1
    for video in data["videos"]:
        assert "title" in video


@needs_net
@needs_ytdlp
def test_metadata_private_video_returns_error():
    """Non-existent or invalid video IDs should return a clean error."""
    r = _run(YouTubeMetadataTool().execute(
        operation="metadata",
        video_id="XXXXXXXXXXX",
    ))
    assert _is_failure(r)
    assert _get_error(r)


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
    from effgen.tools.builtin.youtube_metadata import _mentions_age_restriction

    assert _mentions_age_restriction(_DNS_FAILURE_STDERR) is False


def test_an_age_gate_is_still_recognized():
    from effgen.tools.builtin.youtube_metadata import _mentions_age_restriction

    for stderr in (
        "ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate.",
        "ERROR: [youtube] abc: This video is age-restricted.",
        "ERROR: [youtube] abc: Content is AGE RESTRICTED in your region.",
    ):
        assert _mentions_age_restriction(stderr) is True, stderr
