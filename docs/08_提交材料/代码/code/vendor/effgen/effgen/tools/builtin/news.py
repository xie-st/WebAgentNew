"""
News aggregation tool for the effGen framework.

Backend: curated list of reputable RSS feeds + optional NewsAPI.org.
Set NEWS_API_KEY env var for higher quality results via NewsAPI.
Without key, falls back to aggregating public RSS feeds.

Operations:
- top_headlines: fetch top headlines (optionally filter by category or region)
- search: search news by query across configured sources
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

# NewsAPI is the only fixed third-party host this module pins; RSS feeds come
# from the curated source list below and additionally pass the SSRF guard.
_NEWSAPI_HOSTS = frozenset({"newsapi.org"})

# ---------------------------------------------------------------------------
# Curated RSS sources by category
# ---------------------------------------------------------------------------

_RSS_SOURCES: dict[str, list[dict[str, str]]] = {
    "general": [
        {"name": "BBC World News", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "region": "global"},
        {"name": "BBC News", "url": "https://feeds.bbci.co.uk/news/rss.xml", "region": "global"},
        {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "region": "us"},
        {"name": "The Guardian", "url": "https://www.theguardian.com/world/rss", "region": "global"},
        {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "region": "global"},
    ],
    "technology": [
        {"name": "Hacker News (front page)", "url": "https://hnrss.org/frontpage", "region": "global"},
        {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "region": "global"},
        {"name": "Wired", "url": "https://www.wired.com/feed/rss", "region": "global"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index", "region": "global"},
        {"name": "MIT Technology Review", "url": "https://www.technologyreview.com/feed/", "region": "global"},
    ],
    "science": [
        {"name": "Nature News", "url": "https://www.nature.com/nature.rss", "region": "global"},
        {"name": "Science Daily", "url": "https://www.sciencedaily.com/rss/all.xml", "region": "global"},
        {"name": "New Scientist", "url": "https://www.newscientist.com/feed/home/?cmpid=RSS|NSNS-2012-GLOBAL|newscientist.com-RSS", "region": "global"},
    ],
    "business": [
        {"name": "Financial Times", "url": "https://www.ft.com/?format=rss", "region": "global"},
        {"name": "Bloomberg Markets", "url": "https://feeds.bloomberg.com/markets/news.rss", "region": "global"},
        {"name": "BBC Business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml", "region": "global"},
    ],
    "health": [
        {"name": "BBC Health", "url": "https://feeds.bbci.co.uk/news/health/rss.xml", "region": "global"},
        {"name": "NPR Health", "url": "https://feeds.npr.org/1128/rss.xml", "region": "us"},
    ],
    "sports": [
        {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport/rss.xml", "region": "global"},
        {"name": "ESPN", "url": "https://www.espn.com/espn/rss/news", "region": "global"},
    ],
}

# All sources for cross-category search
_ALL_SOURCES = [src for sources in _RSS_SOURCES.values() for src in sources]


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


def _download_feed(url: str, timeout: int = 15) -> bytes:
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    }
    try:
        with safe_urlopen(url, headers=headers, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as exc:
        raise ConnectionError(f"HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise ConnectionError(f"Network error: {exc.reason}") from exc


def _fetch_rss_source(url: str, name: str = "") -> tuple[list[dict[str, Any]], str | None]:
    """Fetch and parse one RSS feed.

    Returns the feed's entries and, when the feed could not be fetched or
    parsed, the reason. An empty list with no reason means the feed answered
    and carried no entries — the callers keep the two apart so a run in which
    every source was unreachable is reported as a failure rather than as no
    news.
    """
    try:
        import feedparser
    except ImportError as e:
        raise ImportError("feedparser is required: pip install feedparser") from e

    try:
        raw = _download_feed(url)
        feed = feedparser.parse(raw)
    except Exception as exc:
        logger.warning("Failed to fetch feed '%s' (%s): %s", name or url, url, exc)
        return [], f"{name or url}: {exc}"

    bozo_exc = getattr(feed, "bozo_exception", None)
    if feed.bozo and bozo_exc:
        logger.warning("Malformed feed '%s': %s", name or url, bozo_exc)

    entries = []
    feed_name = name or getattr(feed.feed, "title", url) or url
    for entry in (feed.entries or []):
        try:
            published = ""
            for attr in ("published", "updated", "created"):
                val = getattr(entry, attr, None)
                if val:
                    published = val
                    break

            link = getattr(entry, "link", None) or ""
            entries.append({
                "title": getattr(entry, "title", "") or "",
                "link": link,
                "published": published,
                "summary": getattr(entry, "summary", "") or "",
                "source": feed_name,
                "source_url": url,
            })
        except Exception as exc:
            logger.debug("Skipping entry in '%s': %s", name, exc)

    version = getattr(feed, "version", "") or ""
    if not entries and not version:
        logger.warning("URL for feed '%s' did not contain a valid RSS/Atom feed", name or url)
        return [], f"{name or url}: not a valid RSS/Atom feed"

    return entries, None


async def _gather_sources(
    feed_list: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Fetch every feed in *feed_list* concurrently.

    Returns the combined entries, how many sources answered, and the reason
    each of the others gave.
    """
    tasks = [asyncio.to_thread(_fetch_rss_source, src["url"], src["name"]) for src in feed_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    entries: list[dict[str, Any]] = []
    reached = 0
    reasons: list[str] = []
    for src, result in zip(feed_list, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("RSS fetch error: %s", result)
            reasons.append(f"{src['name']}: {result}")
            continue
        source_entries, reason = result
        if reason:
            reasons.append(reason)
            continue
        reached += 1
        entries.extend(source_entries)
    return entries, reached, reasons


def _unreachable_error(feed_list: list[dict[str, str]], reasons: list[str]) -> str:
    """Message for a run in which not one news source answered."""
    first = reasons[0] if reasons else "no reason reported"
    if len(first) > 200:
        first = first[:197] + "..."
    return (
        f"No news source could be reached ({len(feed_list)} feed"
        f"{'s' if len(feed_list) != 1 else ''} tried; first failure: {first}). "
        "Check this machine's network access to the feed hosts, or set NEWS_API_KEY "
        "to fetch through NewsAPI.org instead."
    )


def _newsapi_top_headlines(
    api_key: str,
    category: str | None = None,
    country: str = "us",
    page_size: int = 20,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"apiKey": api_key, "pageSize": page_size}
    if category:
        params["category"] = category.lower()
    params["country"] = country.lower()
    url = f"https://newsapi.org/v2/top-headlines?{urlencode(params)}"
    return _newsapi_request(url)


def _newsapi_search(
    api_key: str,
    query: str,
    sources: str | None = None,
    page_size: int = 20,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"apiKey": api_key, "q": query, "pageSize": page_size}
    if sources:
        params["sources"] = sources
    url = f"https://newsapi.org/v2/everything?{urlencode(params)}"
    return _newsapi_request(url)


def _newsapi_request(url: str) -> list[dict[str, Any]]:
    headers = {"User-Agent": _user_agent()}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=15, allowed_hosts=_NEWSAPI_HOSTS
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise ConnectionError(f"NewsAPI HTTP {e.code}: {e.reason}")
    except URLError as e:
        raise ConnectionError(f"NewsAPI network error: {e.reason}")

    if data.get("status") != "ok":
        raise ConnectionError(f"NewsAPI error: {data.get('message', 'unknown')}")

    articles = data.get("articles", []) or []
    return [
        {
            "title": a.get("title") or "",
            "link": a.get("url") or "",
            "published": a.get("publishedAt") or "",
            "summary": a.get("description") or "",
            "source": (a.get("source") or {}).get("name") or "",
            "source_url": a.get("url") or "",
        }
        for a in articles
    ]


class NewsTool(BaseTool):
    """Aggregate news headlines from reputable sources via RSS and optionally NewsAPI."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="news",
                description=(
                    "Aggregate news headlines from reputable sources. "
                    "Operations: top_headlines (latest news, optionally by category/region), "
                    "search (query-based news search). "
                    "Uses curated RSS feeds by default. Set NEWS_API_KEY for enhanced results via NewsAPI.org."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="top_headlines",
                        enum=["top_headlines", "search"],
                    ),
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query (for operation='search').",
                        required=False,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="category",
                        type=ParameterType.STRING,
                        description="News category filter: general, technology, science, business, health, sports.",
                        required=False,
                        enum=["general", "technology", "science", "business", "health", "sports"],
                    ),
                    ParameterSpec(
                        name="region",
                        type=ParameterType.STRING,
                        description="Region filter (e.g. 'us', 'global'). Only applies to RSS backend.",
                        required=False,
                        min_length=2,
                        max_length=10,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Maximum number of articles to return.",
                        required=False,
                        default=20,
                        min_value=1,
                        max_value=100,
                    ),
                    ParameterSpec(
                        name="sources",
                        type=ParameterType.STRING,
                        description="Comma-separated NewsAPI source IDs (NewsAPI backend only, for operation='search').",
                        required=False,
                    ),
                ],
                timeout_seconds=30,
                tags=["news", "headlines", "rss", "media", "current-events", "free"],
                examples=[
                    {"operation": "top_headlines", "category": "technology", "max_results": 10},
                    {"operation": "search", "query": "artificial intelligence", "max_results": 10},
                ],
            )
        )

    @staticmethod
    def _has_api_key() -> str | None:
        return os.environ.get("NEWS_API_KEY") or None

    async def _execute(
        self,
        operation: str = "top_headlines",
        query: str | None = None,
        category: str | None = None,
        region: str | None = None,
        max_results: int = 20,
        sources: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()
        if op not in ("top_headlines", "search"):
            raise ValueError(f"Unknown operation: {operation!r}")

        api_key = self._has_api_key()

        if op == "top_headlines":
            return await self._top_headlines(api_key, category, region, max_results)
        else:
            if not query:
                raise ValueError(
                    "operation='search' requires 'query'. "
                    "Pass query with the search terms, or use "
                    "operation='headlines' for the unfiltered feed."
                )
            return await self._search(api_key, query, sources, category, max_results)

    async def _top_headlines(
        self,
        api_key: str | None,
        category: str | None,
        region: str | None,
        max_results: int,
    ) -> dict[str, Any]:
        if api_key:
            try:
                country = (region or "us").lower()[:2]
                articles = await asyncio.to_thread(
                    _newsapi_top_headlines, api_key, category, country, min(max_results, 100)
                )
                return {
                    "success": True,
                    "operation": "top_headlines",
                    "backend": "newsapi",
                    "category": category,
                    "region": region,
                    "count": len(articles),
                    "articles": articles[:max_results],
                    "data": {"articles": articles[:max_results]},
                    "error": None,
                }
            except Exception as exc:
                logger.warning("NewsAPI top_headlines failed, falling back to RSS: %s", exc)

        # RSS fallback
        return await self._rss_top_headlines(category, region, max_results)

    async def _rss_top_headlines(
        self,
        category: str | None,
        region: str | None,
        max_results: int,
    ) -> dict[str, Any]:
        if category and category.lower() in _RSS_SOURCES:
            feed_list = _RSS_SOURCES[category.lower()]
        else:
            feed_list = _RSS_SOURCES["general"]

        if region:
            region_lower = region.lower()
            filtered = [s for s in feed_list if s.get("region") == region_lower or s.get("region") == "global"]
            if filtered:
                feed_list = filtered

        articles, reached, reasons = await _gather_sources(feed_list)
        if not reached:
            return {
                "success": False,
                "operation": "top_headlines",
                "backend": "rss",
                "category": category,
                "region": region,
                "count": 0,
                "articles": [],
                "data": {"articles": []},
                "error": _unreachable_error(feed_list, reasons),
            }

        # De-duplicate by link
        seen: set[str] = set()
        deduped = []
        for a in articles:
            key = a.get("link") or a.get("title", "")
            if key and key not in seen:
                seen.add(key)
                deduped.append(a)

        selected = deduped[:max_results]
        return {
            "success": True,
            "operation": "top_headlines",
            "backend": "rss",
            "category": category,
            "region": region,
            "count": len(selected),
            "articles": selected,
            "data": {"articles": selected},
            "sources_reached": reached,
            "sources_tried": len(feed_list),
            "error": None,
        }

    async def _search(
        self,
        api_key: str | None,
        query: str,
        sources: str | None,
        category: str | None,
        max_results: int,
    ) -> dict[str, Any]:
        if api_key:
            try:
                articles = await asyncio.to_thread(
                    _newsapi_search, api_key, query, sources, min(max_results, 100)
                )
                selected = articles[:max_results]
                return {
                    "success": True,
                    "operation": "search",
                    "backend": "newsapi",
                    "query": query,
                    "count": len(selected),
                    "articles": selected,
                    "data": {"articles": selected},
                    "error": None,
                }
            except Exception as exc:
                logger.warning("NewsAPI search failed, falling back to RSS: %s", exc)

        # RSS fallback — search across all or category-specific sources
        if category and category.lower() in _RSS_SOURCES:
            feed_list = _RSS_SOURCES[category.lower()]
        else:
            feed_list = _ALL_SOURCES

        entries, reached, reasons = await _gather_sources(feed_list)
        if not reached:
            return {
                "success": False,
                "operation": "search",
                "backend": "rss",
                "query": query,
                "count": 0,
                "articles": [],
                "data": {"articles": []},
                "error": _unreachable_error(feed_list, reasons),
            }

        q_lower = query.lower()
        matched: list[dict[str, Any]] = []
        for entry in entries:
            if (
                q_lower in entry.get("title", "").lower()
                or q_lower in entry.get("summary", "").lower()
            ):
                matched.append(entry)

        # De-duplicate
        seen: set[str] = set()
        deduped = []
        for a in matched:
            key = a.get("link") or a.get("title", "")
            if key and key not in seen:
                seen.add(key)
                deduped.append(a)

        selected = deduped[:max_results]
        return {
            "success": True,
            "operation": "search",
            "backend": "rss",
            "query": query,
            "count": len(selected),
            "articles": selected,
            "data": {"articles": selected},
            "sources_reached": reached,
            "sources_tried": len(feed_list),
            "error": None,
        }
