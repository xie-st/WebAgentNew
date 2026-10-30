"""Tests for RedditTool.

Unit tests use mocked HTTP. Live integration tests hit Reddit's public JSON API.
Network failures and rate limits are treated as skips.
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import MagicMock, patch

import pytest

from effgen.tools.builtin import RedditTool


def _has_network(host: str = "old.reddit.com", port: int = 443, timeout: float = 5.0) -> bool:
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
    "429",
    "rate limit",
    "timed out",
    "Timeout",
    "Network error",
    "Connection",
    "403",
    "Blocked",
    "500",
    "502",
    "503",
    "504",
    "Internal Server Error",
    "Bad Gateway",
    "Service Unavailable",
    "Gateway Timeout",
    "non-json",
    "non-JSON response",
)


def _ok(result) -> dict:
    assert hasattr(result, "success")
    if not result.success and result.error and any(m.lower() in result.error.lower() for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    out = result.output
    inner_err = out.get("error") if isinstance(out, dict) else None
    if isinstance(out, dict) and out.get("success") is False and inner_err and any(m.lower() in str(inner_err).lower() for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient inner error: {inner_err}")
    return out


# ---------------------------------------------------------------------------
# Metadata / static tests
# ---------------------------------------------------------------------------

def test_reddit_metadata():
    tool = RedditTool()
    assert tool.metadata.name == "reddit"
    assert "reddit" in tool.metadata.tags


def test_reddit_user_agent_has_version():
    from effgen.tools.builtin.reddit import _user_agent
    ua = _user_agent()
    assert ua.startswith("effGen/")


def test_reddit_missing_subreddit_fails():
    r = _run(RedditTool().execute(operation="subreddit_top"))
    assert not r.success


def test_reddit_missing_username_fails():
    r = _run(RedditTool().execute(operation="user_submissions"))
    assert not r.success


def test_reddit_missing_thread_id_fails():
    r = _run(RedditTool().execute(operation="thread_comments"))
    assert not r.success


def test_reddit_unknown_operation_fails():
    r = _run(RedditTool().execute(operation="invalid_op", subreddit="python"))
    assert not r.success


# ---------------------------------------------------------------------------
# Unit tests with mocked HTTP
# ---------------------------------------------------------------------------

def _make_listing(posts: list[dict]) -> dict:
    return {
        "kind": "Listing",
        "data": {
            "children": [{"kind": "t3", "data": p} for p in posts]
        }
    }


def _make_post(**kwargs) -> dict:
    defaults = {
        "id": "abc123",
        "title": "Test Post",
        "author": "testuser",
        "subreddit": "python",
        "url": "https://example.com",
        "permalink": "/r/python/comments/abc123/test_post/",
        "score": 100,
        "upvote_ratio": 0.95,
        "num_comments": 10,
        "created_utc": 1700000000,
        "selftext": "",
        "is_self": False,
        "link_flair_text": None,
    }
    defaults.update(kwargs)
    return defaults


def test_reddit_subreddit_top_mock():
    mock_data = _make_listing([_make_post(title="Post 1"), _make_post(title="Post 2", id="def456")])

    with patch("effgen.tools.builtin.reddit._get_json", return_value=mock_data) as m_get:
        r = _run(RedditTool().execute(operation="subreddit_top", subreddit="python", time_filter="day", n=5))

    out = r.output
    assert out["success"] is True
    assert "old.reddit.com/r/python/top.json" in m_get.call_args.args[0]
    assert "t=day" in m_get.call_args.args[0]
    assert len(out["posts"]) == 2
    assert out["posts"][0]["title"] == "Post 1"
    assert out["subreddit"] == "python"
    assert out["sort"] == "top"
    assert out["time_filter"] == "day"
    assert out["requested_count"] == 5
    assert out["source_exhausted"] is True


def test_reddit_subreddit_hot_mock():
    mock_data = _make_listing([_make_post(title="Hot Post", score=5000)])

    with patch("effgen.tools.builtin.reddit._get_json", return_value=mock_data):
        r = _run(RedditTool().execute(operation="subreddit_hot", subreddit="MachineLearning", n=5))

    out = r.output
    assert out["success"] is True
    assert out["sort"] == "hot"
    assert out["posts"][0]["score"] == 5000


def test_reddit_user_submissions_mock():
    mock_data = _make_listing([_make_post(author="gaurav", title="My submission")])

    with patch("effgen.tools.builtin.reddit._get_json", return_value=mock_data):
        r = _run(RedditTool().execute(operation="user_submissions", username="gaurav", n=5))

    out = r.output
    assert out["success"] is True
    assert out["username"] == "gaurav"
    assert out["post_count"] == 1


def test_reddit_thread_comments_mock():
    post_listing = {
        "kind": "Listing",
        "data": {"children": [{"kind": "t3", "data": _make_post()}]}
    }
    comment_listing = {
        "kind": "Listing",
        "data": {
            "children": [
                {
                    "kind": "t1",
                    "data": {
                        "id": "c1",
                        "author": "user1",
                        "body": "Great post!",
                        "score": 50,
                        "created_utc": 1700000100,
                        "depth": 0,
                        "permalink": "/r/python/comments/abc123/_/c1/",
                        "replies": "",
                    }
                }
            ]
        }
    }
    with patch("effgen.tools.builtin.reddit._get_json", return_value=[post_listing, comment_listing]):
        r = _run(RedditTool().execute(operation="thread_comments", thread_id="abc123", subreddit="python", n=50))

    out = r.output
    assert out["success"] is True
    assert out["comment_count"] == 1
    assert out["comments"][0]["body"] == "Great post!"


def test_reddit_429_backoff_mock():
    """429 should drive _get_json through its retry/backoff loop.

    Mocks requests.get to return 429 every time, then asserts:
    - requests.get was called exactly `retries` times (3),
    - time.sleep was called between retryable attempts with growing delays (2s, 4s).
    """
    from effgen.tools.builtin import reddit as reddit_mod

    sleep_calls: list[float] = []

    def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    fake_resp = MagicMock()
    fake_resp.status_code = 429
    fake_resp.reason = "Too Many Requests"

    with patch.object(reddit_mod, "safe_requests_get", return_value=fake_resp) as m_get, \
         patch.object(reddit_mod.time, "sleep", side_effect=fake_sleep):
        r = _run(RedditTool().execute(operation="subreddit_top", subreddit="python"))

    assert m_get.call_count == 3, f"expected 3 retries, got {m_get.call_count}"
    assert sleep_calls == [2.0, 4.0], f"unexpected backoff sequence: {sleep_calls}"

    out = r.output
    assert out["success"] is False
    assert "error" in out and out["error"]


def test_reddit_403_falls_back_to_www_mock():
    from effgen.tools.builtin import reddit as reddit_mod

    blocked = MagicMock()
    blocked.status_code = 403
    blocked.reason = "Forbidden"
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = _make_listing([_make_post(title="Fallback Post")])

    with patch.object(reddit_mod, "safe_requests_get", side_effect=[blocked, ok]) as m_get:
        r = _run(RedditTool().execute(operation="subreddit_top", subreddit="python", n=5))

    # safe_requests_get(requests_module, url, ...): the URL is the 2nd positional.
    called_urls = [call.args[1] for call in m_get.call_args_list]
    assert called_urls[0].startswith("https://old.reddit.com/")
    assert called_urls[1].startswith("https://www.reddit.com/")
    assert r.output["success"] is True
    assert r.output["posts"][0]["title"] == "Fallback Post"


def test_reddit_login_redirect_falls_back_to_www_mock():
    """A bounce to Reddit's login page is a block, so the fallback host is tried."""
    from effgen.tools.builtin import reddit as reddit_mod

    bounced = MagicMock()
    bounced.status_code = 404
    bounced.reason = "Not Found"
    bounced.url = "https://old.reddit.com/login/?reason=lor2&dest=%2Fr%2Fpython"
    bounced.history = ()
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = _make_listing([_make_post(title="Fallback Post")])

    with patch.object(reddit_mod, "safe_requests_get", side_effect=[bounced, ok]) as m_get:
        r = _run(RedditTool().execute(operation="subreddit_hot", subreddit="python", n=5))

    called_urls = [call.args[1] for call in m_get.call_args_list]
    assert called_urls[0].startswith("https://old.reddit.com/")
    assert called_urls[1].startswith("https://www.reddit.com/")
    assert r.output["success"] is True
    assert r.output["posts"][0]["title"] == "Fallback Post"


def test_reddit_login_redirect_reports_a_block_not_a_missing_page():
    """Both hosts bouncing to login reports the block, not "404 Not Found"."""
    from effgen.tools.builtin import reddit as reddit_mod

    def _bounced(url: str) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 404
        resp.reason = "Not Found"
        resp.url = url.split("/r/")[0] + "/login/?reason=lor2"
        resp.history = ()
        return resp

    with patch.object(
        reddit_mod,
        "safe_requests_get",
        side_effect=lambda _mod, url, **kw: _bounced(url),
    ):
        r = _run(RedditTool().execute(operation="subreddit_hot", subreddit="python", n=5))

    error = r.output["error"]
    assert r.output["success"] is False
    assert "blocked" in error.lower()
    assert "login" in error.lower()
    assert "Not Found" not in error


def test_reddit_genuine_404_is_still_reported_as_not_found():
    """A 404 that did not come from a login bounce is reported as-is."""
    from effgen.tools.builtin import reddit as reddit_mod

    missing = MagicMock()
    missing.status_code = 404
    missing.reason = "Not Found"
    missing.url = "https://old.reddit.com/r/thissubdoesnotexist/hot.json?limit=5"
    missing.history = ()

    with patch.object(reddit_mod, "safe_requests_get", return_value=missing) as m_get:
        r = _run(
            RedditTool().execute(
                operation="subreddit_hot", subreddit="thissubdoesnotexist", n=5
            )
        )

    assert m_get.call_count == 1, "a genuine 404 must not retry the fallback host"
    assert r.output["success"] is False
    assert "404" in r.output["error"]
    assert "Not Found" in r.output["error"]


@pytest.mark.parametrize(
    "url",
    [
        "https://old.reddit.com/r/login/hot.json?limit=5",
        "https://old.reddit.com/user/login_name/submitted.json?limit=5",
        "https://old.reddit.com/r/f/comments/abc/logins_are_hard/.json",
    ],
)
def test_reddit_404_on_a_login_named_target_is_not_read_as_a_block(url):
    """A subreddit/user/thread whose name contains "login" is not a login page."""
    from effgen.tools.builtin import reddit as reddit_mod

    missing = MagicMock()
    missing.status_code = 404
    missing.reason = "Not Found"
    missing.url = url
    missing.history = ()

    with patch.object(reddit_mod, "safe_requests_get", return_value=missing) as m_get:
        with pytest.raises(ConnectionError) as excinfo:
            reddit_mod._get_json(url)

    assert m_get.call_count == 1, "a genuine 404 must not retry the fallback host"
    assert "404" in str(excinfo.value)
    assert "blocked" not in str(excinfo.value).lower()


def test_reddit_login_redirect_detected_from_the_redirect_chain():
    """A client that followed the bounce itself is recognised from its history."""
    from effgen.tools.builtin import reddit as reddit_mod

    hop = MagicMock()
    hop.headers = {"Location": "/login/?dest=%2Fr%2Fpython"}
    bounced = MagicMock()
    bounced.status_code = 404
    bounced.reason = "Not Found"
    bounced.url = "https://www.reddit.com/r/python/hot.json?limit=5"
    bounced.history = [hop]

    with patch.object(reddit_mod, "safe_requests_get", return_value=bounced):
        with pytest.raises(ConnectionError) as excinfo:
            reddit_mod._get_json("https://www.reddit.com/r/python/hot.json?limit=5")

    assert "blocked" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
def test_reddit_live_subreddit_hot():
    out = _ok(_run(RedditTool().execute(operation="subreddit_hot", subreddit="python", n=5)))
    assert out["success"] is True
    assert isinstance(out["posts"], list)
    assert len(out["posts"]) >= 1
    first = out["posts"][0]
    assert "title" in first
    assert "id" in first
    assert "score" in first
    assert "permalink" in first
    # Validate User-Agent enforcement — if Reddit rejects UA, we'd get an error
    assert out.get("error") is None


@needs_net
def test_reddit_live_subreddit_top_day():
    out = _ok(_run(RedditTool().execute(operation="subreddit_top", subreddit="python", time_filter="day", n=25)))
    assert out["success"] is True
    assert out["requested_count"] == 25
    assert len(out["posts"]) >= 1
    assert len(out["posts"]) <= 25


@needs_net
def test_reddit_live_user_submissions():
    out = _ok(_run(RedditTool().execute(operation="user_submissions", username="spez", n=5)))
    assert out["success"] is True
    assert isinstance(out["posts"], list)
