"""Tests for RSSFeedTool.

Live tests hit real RSS feeds. Network failures are treated as skips.
"""

from __future__ import annotations

import asyncio
import logging
import socket

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import RSSFeedTool

HN_RSS = "https://hnrss.org/frontpage"
BBC_RSS = "http://feeds.bbci.co.uk/news/world/rss.xml"


def _has_network(host: str = "hnrss.org", port: int = 443, timeout: float = 4.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


_TRANSIENT_MARKERS = (
    "HTTP 429",
    "HTTP 5",
    "Too Many Requests",
    "timed out",
    "Timeout",
    "Network error",
    "Connection",
)


def _ok(result: ToolResult) -> dict:
    assert isinstance(result, ToolResult)
    if not result.success and result.error and any(m in result.error for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient network error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    out = result.output
    inner_err = out.get("error") if isinstance(out, dict) else None
    if isinstance(out, dict) and out.get("success") is False and inner_err and any(m in str(inner_err) for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient feed error: {inner_err}")
    return out


# ---------------------------------------------------------------------------
# Metadata / unit tests
# ---------------------------------------------------------------------------

def test_rss_metadata():
    tool = RSSFeedTool()
    assert tool.metadata.name == "rss_feed"
    assert "rss" in tool.metadata.tags


def test_rss_missing_url_fails():
    r = _run(RSSFeedTool().execute(operation="fetch"))
    assert not r.success


def test_rss_unknown_operation_fails():
    r = _run(RSSFeedTool().execute(operation="nope", url=HN_RSS))
    assert not r.success


def test_rss_search_requires_query():
    r = _run(RSSFeedTool().execute(operation="search_in_feed", url=HN_RSS))
    assert not r.success


@needs_net
def test_rss_malformed_feed_returns_a_typed_error(caplog):
    """Malformed/non-feed URL should not crash and should return empty entries."""
    caplog.set_level(logging.WARNING)
    r = _ok(_run(RSSFeedTool().execute(operation="fetch", url="https://example.com")))
    assert r["success"] is True
    assert r["entries"] == []
    assert r["entry_count"] == 0
    assert r["data"]["entries"] == []
    assert "valid RSS/Atom feed" in caplog.text


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
def test_rss_fetch_hn():
    out = _ok(_run(RSSFeedTool().execute(operation="fetch", url=HN_RSS)))
    assert out["success"] is True
    assert "data" in out
    assert out["entry_count"] >= 1
    assert isinstance(out["entries"], list)
    first = out["entries"][0]
    assert "title" in first
    assert "link" in first


@needs_net
def test_rss_latest_n():
    out = _ok(_run(RSSFeedTool().execute(operation="latest", url=HN_RSS, n=5)))
    assert out["entry_count"] <= 5
    assert len(out["entries"]) <= 5


@needs_net
def test_rss_search_in_feed():
    out = _ok(_run(RSSFeedTool().execute(operation="search_in_feed", url=HN_RSS, query="python")))
    assert out["success"] is True
    # May be empty if no current HN front-page posts mention Python, that's fine
    assert isinstance(out["entries"], list)


@needs_net
def test_rss_bbc_feed():
    out = _ok(_run(RSSFeedTool().execute(operation="fetch", url=BBC_RSS)))
    assert out["entry_count"] >= 10
    assert out["entries"][0]["title"]
