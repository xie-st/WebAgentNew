"""
RSS feed tool for the effGen framework.

Backend: feedparser library (pip install feedparser).
No authentication required. Works with any publicly accessible RSS/Atom feed.

Operations:
- fetch: retrieve all entries from a feed URL
- latest: retrieve the N most recent entries
- search_in_feed: search entries by keyword within a feed
"""

from __future__ import annotations

import asyncio
import logging
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


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


def _parse_entry(entry: Any) -> dict[str, Any]:
    """Convert a feedparser entry to a plain dict."""
    published = ""
    for attr in ("published", "updated", "created"):
        val = getattr(entry, attr, None)
        if val:
            published = val
            break

    link = getattr(entry, "link", None) or ""
    summary = getattr(entry, "summary", None) or ""

    return {
        "title": getattr(entry, "title", "") or "",
        "link": link,
        "published": published,
        "summary": summary,
        "author": getattr(entry, "author", "") or "",
        "id": getattr(entry, "id", link) or link,
    }


def _download_feed(url: str, timeout: int = 15, allow_private: bool = False) -> bytes:
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }
    try:
        with safe_urlopen(
            url, headers=headers, timeout=timeout, allow_private=allow_private
        ) as resp:
            return resp.read()
    except HTTPError as exc:
        raise ConnectionError(f"HTTP {exc.code} fetching feed: {exc.reason}") from exc
    except URLError as exc:
        raise ConnectionError(f"Network error fetching feed: {exc.reason}") from exc


def _fetch_feed(url: str, allow_private: bool = False) -> dict[str, Any]:
    """Synchronous feed fetch — runs in thread via asyncio.to_thread."""
    try:
        import feedparser
    except ImportError as e:
        raise ImportError(
            "feedparser is required: pip install feedparser"
        ) from e

    try:
        raw = _download_feed(url, allow_private=allow_private)
        feed = feedparser.parse(raw)
    except Exception as exc:
        return {
            "success": False,
            "data": {"feed": None, "entries": []},
            "error": f"Failed to fetch or parse feed: {exc}",
            "feed": None,
            "entries": [],
            "entry_count": 0,
        }

    bozo_exc = getattr(feed, "bozo_exception", None)
    if feed.bozo and bozo_exc:
        # Some malformed feeds (e.g. encoding issues) still yield usable entries.
        logger.warning("Feed '%s' is malformed: %s", url, bozo_exc)

    feed_meta = {
        "title": getattr(feed.feed, "title", url) or url,
        "link": getattr(feed.feed, "link", url) or url,
        "description": getattr(feed.feed, "description", "") or getattr(feed.feed, "subtitle", "") or "",
    }

    entries = []
    for entry in (feed.entries or []):
        try:
            entries.append(_parse_entry(entry))
        except Exception as exc:
            logger.debug("Skipping unparseable entry: %s", exc)

    version = getattr(feed, "version", "") or ""
    malformed = bool(feed.bozo)
    if not entries and not version:
        malformed = True
        logger.warning("URL '%s' did not contain a valid RSS/Atom feed", url)

    data = {"feed": feed_meta, "entries": entries}
    return {
        "success": True,
        "data": data,
        "feed": feed_meta,
        "entries": entries,
        "entry_count": len(entries),
        "malformed": malformed,
        "error": None,
    }


class RSSFeedTool(BaseTool):
    """Fetch and search RSS/Atom feeds from any URL."""

    def __init__(self, allow_private: bool = False) -> None:
        """Args:
            allow_private: Allow fetching private/loopback/link-local/metadata
                addresses. Default ``False`` blocks them (SSRF protection).
        """
        self.allow_private = allow_private
        super().__init__(
            metadata=ToolMetadata(
                name="rss_feed",
                description=(
                    "Fetch, browse, and search RSS/Atom feeds from any URL. "
                    "Operations: fetch (all entries), latest (N most recent entries), "
                    "search_in_feed (keyword search within a feed). "
                    "Handles malformed feeds without raising. No authentication required."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="fetch",
                        enum=["fetch", "latest", "search_in_feed"],
                    ),
                    ParameterSpec(
                        name="url",
                        type=ParameterType.STRING,
                        description="RSS/Atom feed URL.",
                        required=True,
                        min_length=5,
                        max_length=2048,
                    ),
                    ParameterSpec(
                        name="n",
                        type=ParameterType.INTEGER,
                        description="Max entries to return (used by 'latest').",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=200,
                    ),
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search keyword (used by 'search_in_feed').",
                        required=False,
                        min_length=1,
                        max_length=500,
                    ),
                ],
                timeout_seconds=20,
                tags=["rss", "news", "feed", "atom", "content", "free"],
                examples=[
                    {"operation": "latest", "url": "https://feeds.bbci.co.uk/news/rss.xml", "n": 5},
                    {"operation": "search_in_feed", "url": "https://hnrss.org/frontpage", "query": "python"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str = "fetch",
        url: str = "",
        n: int = 10,
        query: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if not url:
            raise ValueError("'url' is required")

        op = operation.lower()
        if op not in ("fetch", "latest", "search_in_feed"):
            raise ValueError(f"Unknown operation: {operation!r}")

        if op == "search_in_feed" and not query:
            raise ValueError(
                "operation='search_in_feed' requires 'query'. "
                "Pass query with the terms to match, or use operation='fetch' "
                "to read the feed unfiltered."
            )

        result = await asyncio.to_thread(_fetch_feed, url, self.allow_private)
        if not result["success"]:
            return result

        entries = result["entries"]

        if op == "latest":
            result["entries"] = entries[:n]
            result["entry_count"] = len(result["entries"])
            result["data"]["entries"] = result["entries"]

        elif op == "search_in_feed":
            q = query.lower()
            matched = [
                e for e in entries
                if q in e.get("title", "").lower()
                or q in e.get("summary", "").lower()
                or q in e.get("author", "").lower()
            ]
            result["entries"] = matched
            result["entry_count"] = len(matched)
            result["data"]["entries"] = matched
            result["query"] = query

        result["operation"] = op
        result["url"] = url
        return result
