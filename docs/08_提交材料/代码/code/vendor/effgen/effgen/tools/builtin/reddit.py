"""
Reddit tool for the effGen framework.

Backend: Reddit public JSON endpoints (no OAuth required for read-only access).
User-Agent is set to effGen/<version> per Reddit ToS.

Operations:
- subreddit_top: top posts from a subreddit (time: hour/day/week/month/year/all)
- subreddit_hot: hot posts from a subreddit
- user_submissions: recent submissions by a user
- thread_comments: comments from a thread (given permalink or thread ID)

Rate limiting: Backs off with exponential delay on HTTP 429.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_requests_get

logger = logging.getLogger(__name__)

_BASE_URL = "https://old.reddit.com"
_FALLBACK_BASE_URL = "https://www.reddit.com"
_DEFAULT_TIMEOUT = 20
# Reddit read-only JSON hosts (pinned; the shared SSRF guard also blocks
# internal targets and re-validates every redirect hop).
_ALLOWED_HOSTS = frozenset({"*.reddit.com"})


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__} (https://github.com/effgen-ai/effgen)"


def _fallback_url(url: str) -> str | None:
    """Return the www.reddit.com equivalent for an old.reddit.com URL."""
    parsed = urlsplit(url)
    if parsed.netloc != "old.reddit.com":
        return None
    fallback_host = urlsplit(_FALLBACK_BASE_URL).netloc
    return urlunsplit((parsed.scheme, fallback_host, parsed.path, parsed.query, parsed.fragment))


def _is_login_url(url: Any) -> bool:
    """Whether ``url`` addresses Reddit's login page itself.

    Matched on the first path segment (``/login`` or ``/login/...``) so a
    subreddit, user, or thread slug that merely contains the word — ``r/login``,
    ``u/login_name``, a title like "logins_are_hard" — is not mistaken for it.
    """
    if not isinstance(url, str):
        return False
    segments = [segment for segment in urlsplit(url).path.split("/") if segment]
    return bool(segments) and segments[0] == "login"


def _redirected_to_login(resp: Any) -> bool:
    """Whether Reddit bounced this request to its login page.

    A blocked client is not always answered with 403: Reddit also redirects it
    to ``/login/?reason=...``, and that page answers 404. Taken at face value
    the result reads as "no such subreddit", so the redirect is recognised and
    reported as the block it is.

    Checks the URL the response actually came from — redirects are followed
    manually by the SSRF-safe fetch helper — and the redirect chain, for a
    client that followed them itself.
    """
    if _is_login_url(getattr(resp, "url", "")):
        return True
    history = getattr(resp, "history", ()) or ()
    if not isinstance(history, list | tuple):
        return False
    for hop in history:
        headers = getattr(hop, "headers", None) or {}
        if _is_login_url(headers.get("location") or headers.get("Location")):
            return True
    return False


def _get_json(url: str, timeout: int = _DEFAULT_TIMEOUT, retries: int = 3) -> Any:
    """Fetch JSON from Reddit with 429 backoff.

    Reddit's bot detection is sensitive to header fingerprints: clients that
    mix browser-style hints (e.g. Accept-Language) with a non-browser
    User-Agent get blocked with 403. We send only the headers Reddit's API
    expects for a script client: a unique User-Agent + Accept.
    """
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json",
    }
    delay = 2.0
    current_url = url
    tried_fallback = False
    for attempt in range(retries):
        try:
            resp = safe_requests_get(
                requests,
                current_url,
                headers=headers,
                timeout=timeout,
                allowed_hosts=_ALLOWED_HOSTS,
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"Network error: {exc}") from exc

        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise ConnectionError(f"Reddit returned non-JSON response: {exc}") from exc
        blocked = status == 403 or _redirected_to_login(resp)
        if blocked and not tried_fallback:
            fallback = _fallback_url(current_url)
            if fallback:
                logger.warning(
                    "Reddit blocked old.reddit.com (HTTP %d); retrying via www.reddit.com",
                    status,
                )
                current_url = fallback
                tried_fallback = True
                continue
        if blocked:
            raise ConnectionError(
                f"Reddit blocked this request (HTTP {status}). Reddit rejects "
                "unauthenticated API traffic from some networks and redirects "
                "it to a login page; retry from another network or use an "
                "authenticated Reddit client."
            )
        if status == 429:
            if attempt == retries - 1:
                break
            logger.warning(
                "Reddit 429 rate limit — backing off %.1fs (attempt %d/%d)",
                delay, attempt + 1, retries,
            )
            time.sleep(delay)
            delay *= 2
            continue
        raise ConnectionError(f"Reddit HTTP {status}: {resp.reason}")
    raise ConnectionError("Reddit rate limit exceeded after retries")


def _parse_post(data: dict) -> dict[str, Any]:
    """Extract key fields from a Reddit post data dict."""
    return {
        "id": data.get("id", ""),
        "title": data.get("title", ""),
        "author": data.get("author", ""),
        "subreddit": data.get("subreddit", ""),
        "url": data.get("url", ""),
        "permalink": f"https://reddit.com{data.get('permalink', '')}",
        "score": data.get("score", 0),
        "upvote_ratio": data.get("upvote_ratio", 0.0),
        "num_comments": data.get("num_comments", 0),
        "created_utc": data.get("created_utc", 0),
        "selftext": (data.get("selftext") or "")[:500],
        "is_self": data.get("is_self", False),
        "flair": data.get("link_flair_text") or "",
    }


def _parse_comment(data: dict, depth: int = 0) -> dict[str, Any] | None:
    """Extract key fields from a Reddit comment data dict."""
    if data.get("kind") == "more":
        return None
    body = (data.get("body") or "")
    if not body or body in ("[deleted]", "[removed]"):
        return None
    return {
        "id": data.get("id", ""),
        "author": data.get("author", ""),
        "body": body[:1000],
        "score": data.get("score", 0),
        "created_utc": data.get("created_utc", 0),
        "depth": depth,
        "permalink": f"https://reddit.com{data.get('permalink', '')}",
    }


def _flatten_comments(listing: list, max_depth: int = 3, depth: int = 0) -> list[dict]:
    """Recursively flatten a Reddit comment tree."""
    results = []
    if depth > max_depth:
        return results
    for item in listing:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "")
        d = item.get("data", {})
        if kind == "t1":
            comment = _parse_comment(d, depth=depth)
            if comment:
                results.append(comment)
                replies = d.get("replies")
                if isinstance(replies, dict):
                    children = replies.get("data", {}).get("children", [])
                    results.extend(_flatten_comments(children, max_depth=max_depth, depth=depth + 1))
    return results


def _fetch_subreddit(sub: str, sort: str, time_filter: str | None, n: int) -> dict[str, Any]:
    limit = min(n, 100)
    params = f"?limit={limit}&raw_json=1"
    if sort == "top" and time_filter:
        params += f"&t={time_filter}"
    url = f"{_BASE_URL}/r/{sub}/{sort}.json{params}"
    try:
        data = _get_json(url)
    except ConnectionError as exc:
        return {"success": False, "data": {"posts": []}, "error": str(exc)}

    posts = []
    for child in data.get("data", {}).get("children", []):
        if child.get("kind") == "t3":
            posts.append(_parse_post(child.get("data", {})))

    return {
        "success": True,
        "data": {
            "subreddit": sub,
            "sort": sort,
            "time_filter": time_filter,
            "requested_count": limit,
            "posts": posts,
        },
        "subreddit": sub,
        "sort": sort,
        "time_filter": time_filter,
        "requested_count": limit,
        "posts": posts,
        "post_count": len(posts),
        "source_exhausted": len(posts) < limit,
        "error": None,
    }


def _fetch_user_submissions(username: str, n: int) -> dict[str, Any]:
    limit = min(n, 100)
    url = f"{_BASE_URL}/user/{username}/submitted.json?limit={limit}&raw_json=1"
    try:
        data = _get_json(url)
    except ConnectionError as exc:
        return {"success": False, "data": {"posts": []}, "error": str(exc)}

    posts = []
    for child in data.get("data", {}).get("children", []):
        if child.get("kind") == "t3":
            posts.append(_parse_post(child.get("data", {})))

    return {
        "success": True,
        "data": {"username": username, "requested_count": limit, "posts": posts},
        "username": username,
        "requested_count": limit,
        "posts": posts,
        "post_count": len(posts),
        "source_exhausted": len(posts) < limit,
        "error": None,
    }


def _fetch_thread_comments(thread_id: str, subreddit: str | None, n: int) -> dict[str, Any]:
    limit = min(n, 200)
    # thread_id may be a full permalink or just an ID
    if thread_id.startswith("http"):
        # strip base and ensure .json
        path = thread_id.split("reddit.com")[-1].rstrip("/")
        url = f"{_BASE_URL}{path}.json?limit={limit}&raw_json=1"
    elif subreddit:
        url = f"{_BASE_URL}/r/{subreddit}/comments/{thread_id}.json?limit={limit}&raw_json=1"
    else:
        url = f"{_BASE_URL}/comments/{thread_id}.json?limit={limit}&raw_json=1"

    try:
        data = _get_json(url)
    except ConnectionError as exc:
        return {"success": False, "data": {"comments": []}, "error": str(exc)}

    if not isinstance(data, list) or len(data) < 2:
        return {"success": False, "data": {"comments": []}, "error": "Unexpected Reddit response format"}

    # First element is the post, second is the comment tree
    post_data = {}
    post_children = data[0].get("data", {}).get("children", [])
    if post_children:
        post_data = _parse_post(post_children[0].get("data", {}))

    comment_children = data[1].get("data", {}).get("children", [])
    comments = _flatten_comments(comment_children, max_depth=3)

    return {
        "success": True,
        "data": {
            "thread_id": thread_id,
            "requested_count": limit,
            "post": post_data,
            "comments": comments,
        },
        "thread_id": thread_id,
        "requested_count": limit,
        "post": post_data,
        "comments": comments,
        "comment_count": len(comments),
        "error": None,
    }


class RedditTool(BaseTool):
    """Fetch posts and comments from Reddit using public JSON endpoints (no auth required)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="reddit",
                description=(
                    "Fetch posts and comments from Reddit using public JSON endpoints. "
                    "No authentication required for read-only access. "
                    "Operations: subreddit_top (top posts), subreddit_hot (hot posts), "
                    "user_submissions (user's posts), thread_comments (comments on a thread). "
                    "Automatically backs off on rate limits (HTTP 429)."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: subreddit_top, subreddit_hot, user_submissions, thread_comments.",
                        required=True,
                        enum=["subreddit_top", "subreddit_hot", "user_submissions", "thread_comments"],
                    ),
                    ParameterSpec(
                        name="subreddit",
                        type=ParameterType.STRING,
                        description="Subreddit name without 'r/' prefix (used by subreddit_top/hot).",
                        required=False,
                        min_length=1,
                        max_length=100,
                    ),
                    ParameterSpec(
                        name="username",
                        type=ParameterType.STRING,
                        description="Reddit username (used by user_submissions).",
                        required=False,
                        min_length=1,
                        max_length=100,
                    ),
                    ParameterSpec(
                        name="thread_id",
                        type=ParameterType.STRING,
                        description="Thread ID or full permalink URL (used by thread_comments).",
                        required=False,
                        min_length=1,
                        max_length=512,
                    ),
                    ParameterSpec(
                        name="time_filter",
                        type=ParameterType.STRING,
                        description="Time filter for subreddit_top: hour, day, week, month, year, all.",
                        required=False,
                        default="day",
                        enum=["hour", "day", "week", "month", "year", "all"],
                    ),
                    ParameterSpec(
                        name="n",
                        type=ParameterType.INTEGER,
                        description="Max number of posts/comments to return.",
                        required=False,
                        default=25,
                        min_value=1,
                        max_value=100,
                    ),
                ],
                timeout_seconds=30,
                tags=["reddit", "social", "posts", "comments", "community", "free"],
                examples=[
                    {"operation": "subreddit_top", "subreddit": "python", "time_filter": "day", "n": 10},
                    {"operation": "subreddit_hot", "subreddit": "MachineLearning", "n": 5},
                    {"operation": "thread_comments", "thread_id": "abc123", "subreddit": "python", "n": 50},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        subreddit: str | None = None,
        username: str | None = None,
        thread_id: str | None = None,
        time_filter: str = "day",
        n: int = 25,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()

        if op == "subreddit_top":
            if not subreddit:
                raise ValueError("'subreddit' is required for subreddit_top")
            return await asyncio.to_thread(_fetch_subreddit, subreddit.strip(), "top", time_filter, n)

        elif op == "subreddit_hot":
            if not subreddit:
                raise ValueError("'subreddit' is required for subreddit_hot")
            return await asyncio.to_thread(_fetch_subreddit, subreddit.strip(), "hot", None, n)

        elif op == "user_submissions":
            if not username:
                raise ValueError("'username' is required for user_submissions")
            return await asyncio.to_thread(_fetch_user_submissions, username.strip(), n)

        elif op == "thread_comments":
            if not thread_id:
                raise ValueError("'thread_id' is required for thread_comments")
            return await asyncio.to_thread(_fetch_thread_comments, thread_id.strip(), subreddit, n)

        else:
            raise ValueError(f"Unknown operation: {operation!r}")
