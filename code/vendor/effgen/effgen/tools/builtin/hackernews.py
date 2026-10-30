"""
Hacker News tool for the effGen framework.

Backend: Hacker News Firebase API (https://hacker-news.firebaseio.com/v0/).
No authentication required. No documented rate limit — be polite.

Operations:
- top_stories: Fetch the current top N story IDs and their details.
- new_stories: Fetch the newest N stories.
- best_stories: Fetch the best-ranked N stories.
- story: Fetch a single item (story, comment, poll, etc.) by ID.
- user: Fetch a HN user profile.
- ask_stories: Fetch current Ask HN stories.
- show_stories: Fetch current Show HN stories.
- job_stories: Fetch current job postings.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
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

_BASE_URL = "https://hacker-news.firebaseio.com/v0"
_DEFAULT_TIMEOUT = 15
# Fixed API host (pinned; the shared SSRF guard also blocks internal targets).
_ALLOWED_HOSTS = frozenset({"hacker-news.firebaseio.com"})


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


def _get_json(url: str, timeout: int = _DEFAULT_TIMEOUT) -> Any:
    headers = {"User-Agent": _user_agent(), "Accept": "application/json"}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=timeout, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise ConnectionError(f"HN API HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise ConnectionError(f"Network error: {exc.reason}") from exc


def _fetch_item(item_id: int) -> dict[str, Any] | None:
    """Fetch a single HN item by ID."""
    try:
        data = _get_json(f"{_BASE_URL}/item/{item_id}.json")
        return data
    except Exception as exc:
        logger.debug("Failed to fetch HN item %d: %s", item_id, exc)
        return None


def _parse_story(item: dict) -> dict[str, Any]:
    """Normalize a HN story item into a flat dict."""
    item_id = item.get("id")
    hn_url = f"https://news.ycombinator.com/item?id={item_id}"
    url = item.get("url") or hn_url
    return {
        "id": item_id,
        "type": item.get("type", ""),
        "title": item.get("title", ""),
        "url": url,
        "hn_url": hn_url,
        "external_url": item.get("url") or "",
        "score": item.get("score", 0),
        "author": item.get("by", ""),
        "time": item.get("time", 0),
        "descendants": item.get("descendants", 0),
        "text": (item.get("text") or "")[:500],
    }


def _parse_user(user: dict) -> dict[str, Any]:
    return {
        "id": user.get("id", ""),
        "karma": user.get("karma", 0),
        "created": user.get("created", 0),
        "about": (user.get("about") or "")[:500],
        "submitted_count": len(user.get("submitted") or []),
        "submitted_ids": (user.get("submitted") or [])[:20],
    }


def _fetch_story_list(list_name: str, n: int) -> dict[str, Any]:
    """Fetch a list of story IDs then hydrate up to n stories in parallel."""
    try:
        ids = _get_json(f"{_BASE_URL}/{list_name}.json")
    except ConnectionError as exc:
        return {"success": False, "data": {"stories": []}, "error": str(exc)}

    if not isinstance(ids, list):
        return {"success": False, "data": {"stories": []}, "error": "Unexpected response format"}

    ids = ids[:n]

    if not ids:
        return {
            "success": True,
            "data": {"list": list_name, "stories": []},
            "list": list_name,
            "stories": [],
            "story_count": 0,
            "error": None,
        }

    with ThreadPoolExecutor(max_workers=min(10, len(ids))) as pool:
        items = list(pool.map(_fetch_item, ids))

    stories = []
    for item in items:
        if item and item.get("type") in ("story", "job", "poll", "ask"):
            stories.append(_parse_story(item))

    return {
        "success": True,
        "data": {"list": list_name, "stories": stories},
        "list": list_name,
        "stories": stories,
        "story_count": len(stories),
        "error": None,
    }


class HackerNewsTool(BaseTool):
    """Fetch Hacker News stories, items, and user profiles via the Firebase API."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="hackernews",
                description=(
                    "Fetch Hacker News stories, items, and user profiles using the "
                    "official Firebase API. No authentication required. "
                    "Operations: top_stories, new_stories, best_stories, "
                    "ask_stories, show_stories, job_stories, story (single item by ID), "
                    "user (user profile by username)."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description=(
                            "Operation: top_stories, new_stories, best_stories, "
                            "ask_stories, show_stories, job_stories, story, user."
                        ),
                        required=True,
                        enum=[
                            "top_stories",
                            "new_stories",
                            "best_stories",
                            "ask_stories",
                            "show_stories",
                            "job_stories",
                            "story",
                            "user",
                        ],
                    ),
                    ParameterSpec(
                        name="n",
                        type=ParameterType.INTEGER,
                        description="Max number of stories to fetch (used by list operations).",
                        required=False,
                        default=30,
                        min_value=1,
                        max_value=500,
                    ),
                    ParameterSpec(
                        name="item_id",
                        type=ParameterType.INTEGER,
                        description="HN item ID (used by 'story' operation).",
                        required=False,
                    ),
                    ParameterSpec(
                        name="username",
                        type=ParameterType.STRING,
                        description="HN username (used by 'user' operation).",
                        required=False,
                        min_length=1,
                        max_length=100,
                    ),
                ],
                timeout_seconds=60,
                tags=["hackernews", "hn", "tech", "news", "community", "free"],
                examples=[
                    {"operation": "top_stories", "n": 10},
                    {"operation": "story", "item_id": 39000000},
                    {"operation": "user", "username": "pg"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        n: int = 30,
        item_id: int | None = None,
        username: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()

        _list_ops = {
            "top_stories": "topstories",
            "new_stories": "newstories",
            "best_stories": "beststories",
            "ask_stories": "askstories",
            "show_stories": "showstories",
            "job_stories": "jobstories",
        }

        if op in _list_ops:
            return await asyncio.to_thread(_fetch_story_list, _list_ops[op], n)

        elif op == "story":
            if item_id is None:
                raise ValueError("'item_id' is required for the 'story' operation")
            item = await asyncio.to_thread(_fetch_item, int(item_id))
            if item is None:
                return {
                    "success": False,
                    "data": {"item": None},
                    "error": f"Item {item_id} not found or fetch failed",
                }
            return {
                "success": True,
                "data": {"item": item},
                "item": item,
                "error": None,
            }

        elif op == "user":
            if not username:
                raise ValueError("'username' is required for the 'user' operation")
            try:
                user_data = await asyncio.to_thread(
                    _get_json, f"{_BASE_URL}/user/{username}.json"
                )
            except ConnectionError as exc:
                return {"success": False, "data": {"user": None}, "error": str(exc)}
            if user_data is None:
                return {
                    "success": False,
                    "data": {"user": None},
                    "error": f"User '{username}' not found",
                }
            parsed = _parse_user(user_data)
            return {
                "success": True,
                "data": {"user": parsed},
                "user": parsed,
                "error": None,
            }

        else:
            raise ValueError(f"Unknown operation: {operation!r}")
