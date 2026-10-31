"""
Semantic Scholar tool for the effGen framework.

Backend: Semantic Scholar Graph API (https://api.semanticscholar.org/graph/v1/).
No authentication required for read-only queries; an optional
SEMANTIC_SCHOLAR_API_KEY (header ``x-api-key``) lifts the rate limit
from 100 req / 5 min to a much higher partner-program tier.

Operations:
- search:     full-text paper search
- paper:      single paper by id (s2 id, DOI, ARXIV, MAG, ACL, PMID:..., URL)
- citations:  inbound citations of a paper
- references: outbound references of a paper
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
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

S2_BASE = "https://api.semanticscholar.org/graph/v1"
# Fixed API host (pinned; the shared SSRF guard also blocks internal targets).
_ALLOWED_HOSTS = frozenset({"api.semanticscholar.org"})

PAPER_FIELDS = (
    "paperId,externalIds,url,title,abstract,venue,year,referenceCount,"
    "citationCount,influentialCitationCount,isOpenAccess,fieldsOfStudy,authors"
)


def _is_transient_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "429" in lowered
        or "too many requests" in lowered
        or "rate limit" in lowered
        or "http 500" in lowered
        or "http 502" in lowered
        or "http 503" in lowered
        or "http 504" in lowered
        or "internal server error" in lowered
    )


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


class _SlidingWindow:
    """Sliding-window rate limiter: at most N events per window seconds."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = float(window)
        self._events: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.window
            self._events = [t for t in self._events if t >= cutoff]
            if len(self._events) >= self.limit:
                wait = self._events[0] + self.window - now
                if wait > 0:
                    await asyncio.sleep(wait)
                self._events = [t for t in self._events if t >= time.monotonic() - self.window]
            self._events.append(time.monotonic())


_LIMITER: _SlidingWindow | None = None


def _limiter() -> _SlidingWindow:
    global _LIMITER
    if _LIMITER is None:
        # Unauthenticated: 100 / 5 min. Authenticated keys get higher quota
        # upstream, but we apply the conservative unauth cap locally regardless,
        # which is correct (we always stay below quota) — the keyed quota is
        # enforced by the server side.
        _LIMITER = _SlidingWindow(limit=100, window=300.0)
    return _LIMITER


def _http_get(url: str, headers: dict[str, str], timeout: int = 25) -> bytes:
    try:
        with safe_urlopen(
            url, headers=headers, timeout=timeout, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return resp.read()
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:  # error body unreadable; use the status code only
            pass
        # Mark rate-limit errors distinctly so the async wrapper can back off.
        if e.code == 429:
            raise _RateLimited(f"HTTP 429 from {url}: {body}")
        raise ConnectionError(f"HTTP {e.code} from {url}: {e.reason} {body}")
    except URLError as e:
        raise ConnectionError(f"Network error fetching {url}: {e.reason}")


class _RateLimited(Exception):
    """Raised on 429 from Semantic Scholar — triggers backoff in the caller."""


class SemanticScholarTool(BaseTool):
    """Query Semantic Scholar's Graph API for papers, citations and refs."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="semantic_scholar",
                description=(
                    "Search and look up academic papers via the Semantic Scholar "
                    "Graph API. Operations: search (by query), paper (by id/DOI/"
                    "ARXIV id), citations, references. Free; SEMANTIC_SCHOLAR_API_KEY "
                    "honored if present."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="search",
                        enum=["search", "paper", "citations", "references"],
                    ),
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query (search only).",
                        required=False,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="paper_id",
                        type=ParameterType.STRING,
                        description=(
                            "Paper identifier: S2 id, DOI:..., ARXIV:..., MAG:..., "
                            "ACL:..., PMID:..., URL:... (paper/citations/references)."
                        ),
                        required=False,
                        min_length=1,
                        max_length=200,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Max items returned.",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=100,
                    ),
                ],
                timeout_seconds=30,
                tags=["knowledge", "semantic_scholar", "papers", "academic", "free"],
                examples=[
                    {"operation": "search", "query": "transformer attention", "max_results": 5},
                    {"operation": "paper", "paper_id": "ARXIV:1706.03762"},
                ],
            )
        )

    @staticmethod
    def _headers() -> dict[str, str]:
        headers = {
            "User-Agent": _user_agent(),
            "Accept": "application/json",
        }
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if key:
            headers["x-api-key"] = key
        return headers

    async def _get(self, url: str, *, retries: int = 4) -> Any:
        backoff = 2.0
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            await _limiter().acquire()
            try:
                raw = await asyncio.to_thread(_http_get, url, self._headers())
                return json.loads(raw.decode("utf-8"))
            except _RateLimited as e:
                last_err = e
                if attempt == retries:
                    raise ConnectionError(str(e))
                logger.warning(
                    "Semantic Scholar 429 — backing off %.1fs (attempt %d/%d)",
                    backoff, attempt + 1, retries,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        # unreachable
        raise ConnectionError(str(last_err))

    async def _execute(
        self,
        operation: str = "search",
        query: str | None = None,
        paper_id: str | None = None,
        max_results: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()
        if op == "search":
            if not query:
                raise ValueError(
                    "operation='search' requires 'query'. "
                    "Pass query with the search terms, or use operation='paper' "
                    "with a paper id."
                )
            return await self._search(query, max_results)
        if op == "paper":
            if not paper_id:
                raise ValueError("operation='paper' requires 'paper_id'")
            return await self._paper(paper_id)
        if op == "citations":
            if not paper_id:
                raise ValueError("operation='citations' requires 'paper_id'")
            return await self._citations(paper_id, max_results)
        if op == "references":
            if not paper_id:
                raise ValueError("operation='references' requires 'paper_id'")
            return await self._references(paper_id, max_results)
        raise ValueError(f"Unknown operation: {operation}")

    async def _search(self, query: str, max_results: int) -> dict[str, Any]:
        params = urlencode({"query": query, "limit": max_results, "fields": PAPER_FIELDS})
        url = f"{S2_BASE}/paper/search?{params}"
        try:
            data = await self._get(url, retries=0)
            endpoint = "paper/search"
        except ConnectionError as exc:
            if not _is_transient_error(str(exc)):
                raise
            logger.warning(
                "Semantic Scholar paper/search transiently failed; falling back to paper/search/match"
            )
            match_url = f"{S2_BASE}/paper/search/match?{params}"
            data = await self._get(match_url, retries=2)
            endpoint = "paper/search/match"
        items = data.get("data", []) or []
        return {
            "query": query,
            "count": len(items),
            "total_count": int(data.get("total", len(items)) or 0),
            "results": [_format_paper(p) for p in items],
            "source": "semantic_scholar",
            "endpoint": endpoint,
        }

    async def _paper(self, paper_id: str) -> dict[str, Any]:
        params = urlencode({"fields": PAPER_FIELDS})
        url = f"{S2_BASE}/paper/{paper_id}?{params}"
        try:
            data = await self._get(url)
            endpoint = "paper"
        except ConnectionError as exc:
            if "429" not in str(exc):
                raise
            data = await self._paper_fallback_from_arxiv(paper_id)
            endpoint = "arxiv_fallback"
        return {
            "paper_id": paper_id,
            "result": _format_paper(data),
            "source": "semantic_scholar",
            "endpoint": endpoint,
        }

    async def _paper_fallback_from_arxiv(self, paper_id: str) -> dict[str, Any]:
        prefix, sep, raw_id = paper_id.partition(":")
        if sep and prefix.lower() != "arxiv":
            raise ConnectionError(f"Semantic Scholar paper lookup rate-limited for {paper_id}")
        arxiv_id = raw_id if sep else paper_id

        from .arxiv import ArXivTool

        arxiv_result = await ArXivTool().execute(operation="fetch", arxiv_id=arxiv_id)
        if not arxiv_result.success or not arxiv_result.output:
            raise ConnectionError(f"Semantic Scholar paper lookup rate-limited for {paper_id}")
        paper = arxiv_result.output["result"]
        title = paper.get("title")
        if not title:
            raise ConnectionError(f"Semantic Scholar paper lookup rate-limited for {paper_id}")

        return {
            "paperId": None,
            "externalIds": {"ArXiv": paper.get("arxiv_id") or arxiv_id},
            "url": paper.get("url"),
            "title": title,
            "abstract": paper.get("summary"),
            "venue": "arXiv",
            "year": (paper.get("published") or "")[:4] or None,
            "referenceCount": None,
            "citationCount": None,
            "influentialCitationCount": None,
            "isOpenAccess": True,
            "fieldsOfStudy": paper.get("categories") or [],
            "authors": [{"name": author} for author in paper.get("authors", [])],
        }

    async def _citations(self, paper_id: str, max_results: int) -> dict[str, Any]:
        params = urlencode({"limit": max_results, "fields": PAPER_FIELDS})
        url = f"{S2_BASE}/paper/{paper_id}/citations?{params}"
        data = await self._get(url)
        items = data.get("data", []) or []
        out = []
        for entry in items:
            paper = entry.get("citingPaper") or {}
            out.append({
                "contexts": entry.get("contexts", []),
                "intents": entry.get("intents", []),
                "isInfluential": entry.get("isInfluential"),
                "paper": _format_paper(paper),
            })
        return {
            "paper_id": paper_id,
            "count": len(out),
            "results": out,
            "source": "semantic_scholar",
        }

    async def _references(self, paper_id: str, max_results: int) -> dict[str, Any]:
        params = urlencode({"limit": max_results, "fields": PAPER_FIELDS})
        url = f"{S2_BASE}/paper/{paper_id}/references?{params}"
        data = await self._get(url)
        items = data.get("data", []) or []
        out = []
        for entry in items:
            paper = entry.get("citedPaper") or {}
            out.append({
                "contexts": entry.get("contexts", []),
                "intents": entry.get("intents", []),
                "isInfluential": entry.get("isInfluential"),
                "paper": _format_paper(paper),
            })
        return {
            "paper_id": paper_id,
            "count": len(out),
            "results": out,
            "source": "semantic_scholar",
        }


def _format_paper(p: dict[str, Any] | None) -> dict[str, Any]:
    if not p:
        return {}
    authors = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")]
    return {
        "paperId": p.get("paperId"),
        "externalIds": p.get("externalIds"),
        "title": p.get("title"),
        "abstract": p.get("abstract"),
        "venue": p.get("venue"),
        "year": p.get("year"),
        "authors": authors,
        "citationCount": p.get("citationCount"),
        "referenceCount": p.get("referenceCount"),
        "influentialCitationCount": p.get("influentialCitationCount"),
        "isOpenAccess": p.get("isOpenAccess"),
        "fieldsOfStudy": p.get("fieldsOfStudy"),
        "url": p.get("url"),
    }
