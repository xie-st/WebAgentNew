"""Tests for HackerNewsTool.

Unit tests use mocked HTTP. Live integration tests hit the HN Firebase API.
Network failures are treated as skips.
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import patch

import pytest

from effgen.tools.builtin import HackerNewsTool


def _has_network(host: str = "hacker-news.firebaseio.com", port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


_TRANSIENT_MARKERS = ("timed out", "Timeout", "Network error", "Connection")


def _ok(result) -> dict:
    assert hasattr(result, "success")
    if not result.success and result.error and any(m.lower() in result.error.lower() for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    return result.output


# ---------------------------------------------------------------------------
# Metadata / static tests
# ---------------------------------------------------------------------------

def test_hn_metadata():
    tool = HackerNewsTool()
    assert tool.metadata.name == "hackernews"
    assert "hackernews" in tool.metadata.tags
    assert "hn" in tool.metadata.tags


def test_hn_user_agent_has_version():
    from effgen.tools.builtin.hackernews import _user_agent
    ua = _user_agent()
    assert ua.startswith("effGen/")


def test_hn_story_requires_item_id():
    r = _run(HackerNewsTool().execute(operation="story"))
    assert not r.success


def test_hn_user_requires_username():
    r = _run(HackerNewsTool().execute(operation="user"))
    assert not r.success


def test_hn_unknown_operation_fails():
    r = _run(HackerNewsTool().execute(operation="invalid_op"))
    assert not r.success


# ---------------------------------------------------------------------------
# Unit tests with mocked HTTP
# ---------------------------------------------------------------------------

_SAMPLE_STORY = {
    "id": 39000001,
    "type": "story",
    "title": "Test Hacker News Story",
    "url": "https://example.com/story",
    "score": 200,
    "by": "testuser",
    "time": 1700000000,
    "descendants": 42,
    "text": "",
}

_SAMPLE_USER = {
    "id": "testuser",
    "karma": 1500,
    "created": 1300000000,
    "about": "Builder of things",
    "submitted": list(range(39000001, 39000021)),
}


def test_hn_top_stories_mock():
    ids = [39000001, 39000002, 39000003]
    stories = [dict(_SAMPLE_STORY, id=i, title=f"Story {i}") for i in ids]

    call_count = [0]
    def fake_get_json(url, timeout=15):
        call_count[0] += 1
        if "topstories" in url:
            return ids
        for story in stories:
            if str(story["id"]) in url:
                return story
        return _SAMPLE_STORY

    with patch("effgen.tools.builtin.hackernews._get_json", side_effect=fake_get_json):
        r = _run(HackerNewsTool().execute(operation="top_stories", n=3))

    out = r.output
    assert out["success"] is True
    assert out["list"] == "topstories"
    assert isinstance(out["stories"], list)
    assert len(out["stories"]) == 3
    assert out["stories"][0]["title"].startswith("Story")
    assert out["stories"][0]["url"]
    assert out["stories"][0]["hn_url"].startswith("https://news.ycombinator.com/item?id=")


def test_hn_new_stories_mock():
    ids = [39000010, 39000011]

    def fake_get_json(url, timeout=15):
        if "newstories" in url:
            return ids
        return dict(_SAMPLE_STORY, id=int(url.split("/item/")[1].split(".")[0]))

    with patch("effgen.tools.builtin.hackernews._get_json", side_effect=fake_get_json):
        r = _run(HackerNewsTool().execute(operation="new_stories", n=2))

    out = r.output
    assert out["success"] is True
    assert out["list"] == "newstories"


def test_hn_story_mock():
    with patch("effgen.tools.builtin.hackernews._fetch_item", return_value=_SAMPLE_STORY):
        r = _run(HackerNewsTool().execute(operation="story", item_id=39000001))

    out = r.output
    assert out["success"] is True
    assert out["item"]["title"] == "Test Hacker News Story"
    assert out["item"]["score"] == 200


def test_hn_story_not_found_mock():
    with patch("effgen.tools.builtin.hackernews._fetch_item", return_value=None):
        r = _run(HackerNewsTool().execute(operation="story", item_id=99999999))

    out = r.output
    assert out["success"] is False
    assert "not found" in out["error"].lower()


def test_hn_user_mock():
    with patch("effgen.tools.builtin.hackernews._get_json", return_value=_SAMPLE_USER):
        r = _run(HackerNewsTool().execute(operation="user", username="testuser"))

    out = r.output
    assert out["success"] is True
    assert out["user"]["id"] == "testuser"
    assert out["user"]["karma"] == 1500
    assert out["user"]["submitted_count"] == 20


def test_hn_user_not_found_mock():
    with patch("effgen.tools.builtin.hackernews._get_json", return_value=None):
        r = _run(HackerNewsTool().execute(operation="user", username="nonexistent_xyz"))

    out = r.output
    assert out["success"] is False
    assert "not found" in out["error"].lower()


@pytest.mark.parametrize("op,endpoint", [
    ("top_stories", "topstories"),
    ("new_stories", "newstories"),
    ("best_stories", "beststories"),
    ("ask_stories", "askstories"),
    ("show_stories", "showstories"),
    ("job_stories", "jobstories"),
])
def test_hn_list_ops_supported(op, endpoint):
    """Verify all list operations are handled (each maps to correct Firebase endpoint)."""
    seen_urls = []

    def fake_get_json(url, timeout=15):
        seen_urls.append(url)
        if endpoint in url:
            return []
        return _SAMPLE_STORY

    with patch("effgen.tools.builtin.hackernews._get_json", side_effect=fake_get_json):
        r = _run(HackerNewsTool().execute(operation=op, n=1))

    assert r.output["success"] is True, f"operation {op} failed"
    assert any(endpoint in url for url in seen_urls), f"expected {endpoint} in URL for op {op}"


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
def test_hn_live_top_stories():
    out = _ok(_run(HackerNewsTool().execute(operation="top_stories", n=10)))
    assert out["success"] is True
    assert isinstance(out["stories"], list)
    assert len(out["stories"]) >= 1
    first = out["stories"][0]
    assert "title" in first
    assert "id" in first
    assert first["url"]
    assert "score" in first
    assert "hn_url" in first
    assert first["hn_url"].startswith("https://news.ycombinator.com/item?id=")


@needs_net
def test_hn_live_story():
    # Item 1 is the original "test" item on HN — always exists
    out = _ok(_run(HackerNewsTool().execute(operation="story", item_id=1)))
    assert out["success"] is True
    assert out["item"] is not None
    assert "by" in out["item"]


@needs_net
def test_hn_live_user():
    out = _ok(_run(HackerNewsTool().execute(operation="user", username="pg")))
    assert out["success"] is True
    assert out["user"]["id"] == "pg"
    assert out["user"]["karma"] > 0


@needs_net
def test_hn_live_new_stories():
    out = _ok(_run(HackerNewsTool().execute(operation="new_stories", n=5)))
    assert out["success"] is True
    assert len(out["stories"]) >= 1
