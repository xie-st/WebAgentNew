"""
ArXiv tool for the effGen framework.

Backend: arXiv Atom-feed API (http://export.arxiv.org/api/query). No auth.

Operations:
- search: full-text query (Atom feed parsed to structured papers)
- fetch: lookup a single paper by arXiv ID (e.g. 2103.00020 or arxiv.org/abs/...)
- download_pdf: stream the PDF to disk and return the path + size
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

ARXIV_BASE = "http://export.arxiv.org/api/query"
# Fixed API hosts (pinned; the shared SSRF guard also blocks internal targets).
# The Atom API lives on export.arxiv.org; PDFs are served from arxiv.org.
_ALLOWED_HOSTS = frozenset({"export.arxiv.org", "arxiv.org"})
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
ARXIV_MIN_REQUEST_INTERVAL = 3.0

_ARXIV_REQUEST_LOCK = asyncio.Lock()
_ARXIV_LAST_REQUEST = 0.0


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


def _http_get(url: str, accept: str = "application/atom+xml", timeout: int = 12) -> bytes:
    headers = {"User-Agent": _user_agent(), "Accept": accept}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=timeout, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return resp.read()
    except HTTPError as e:
        raise ConnectionError(f"HTTP {e.code} from {url}: {e.reason}")
    except URLError as e:
        raise ConnectionError(f"Network error fetching {url}: {e.reason}")
    except TimeoutError as e:
        raise ConnectionError(f"Timeout fetching {url}") from e


async def _http_get_polite(
    url: str,
    accept: str = "application/atom+xml",
    *,
    retries: int = 2,
) -> bytes:
    """Fetch arXiv while respecting its single-connection, 3-second API limit."""
    global _ARXIV_LAST_REQUEST
    backoff = 1.0
    for attempt in range(retries + 1):
        async with _ARXIV_REQUEST_LOCK:
            elapsed = time.monotonic() - _ARXIV_LAST_REQUEST
            if elapsed < ARXIV_MIN_REQUEST_INTERVAL:
                await asyncio.sleep(ARXIV_MIN_REQUEST_INTERVAL - elapsed)
            _ARXIV_LAST_REQUEST = time.monotonic()
            try:
                return await asyncio.to_thread(_http_get, url, accept)
            except ConnectionError as exc:
                error_text = str(exc)
                retryable = (
                    "HTTP 429" in error_text
                    or "HTTP 5" in error_text
                    or "timeout" in error_text.lower()
                )
                if attempt == retries or not retryable:
                    raise
                logger.warning(
                    "arXiv request failed (%s); retrying in %.1fs",
                    exc,
                    backoff,
                )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 16.0)
    raise ConnectionError(f"arXiv request failed after {retries} retries: {url}")


_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


def _normalize_arxiv_id(raw: str) -> str:
    raw = raw.strip()
    m = _ID_RE.search(raw)
    if m:
        return m.group(1)
    # Old-style ids (e.g. cs.CL/0301001) — accept as-is.
    return raw


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    def t(el: ET.Element | None) -> str:
        return (el.text or "").strip() if el is not None and el.text else ""

    id_url = t(entry.find(f"{ATOM_NS}id"))
    arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url
    title = t(entry.find(f"{ATOM_NS}title")).replace("\n", " ").strip()
    summary = t(entry.find(f"{ATOM_NS}summary"))
    published = t(entry.find(f"{ATOM_NS}published"))
    updated = t(entry.find(f"{ATOM_NS}updated"))
    authors = [
        t(a.find(f"{ATOM_NS}name"))
        for a in entry.findall(f"{ATOM_NS}author")
        if a.find(f"{ATOM_NS}name") is not None
    ]
    categories = [c.attrib.get("term", "") for c in entry.findall(f"{ATOM_NS}category")]
    primary_cat_el = entry.find(f"{ARXIV_NS}primary_category")
    primary_category = primary_cat_el.attrib.get("term", "") if primary_cat_el is not None else ""
    pdf_url = ""
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href", "")
            break
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    comment_el = entry.find(f"{ARXIV_NS}comment")
    doi_el = entry.find(f"{ARXIV_NS}doi")
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "summary": summary,
        "published": published,
        "updated": updated,
        "primary_category": primary_category,
        "categories": [c for c in categories if c],
        "comment": (comment_el.text or "").strip() if comment_el is not None and comment_el.text else "",
        "doi": (doi_el.text or "").strip() if doi_el is not None and doi_el.text else None,
        "url": id_url,
        "pdf_url": pdf_url,
    }


class ArXivTool(BaseTool):
    """Search and fetch papers from arXiv via the free Atom API."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="arxiv",
                description=(
                    "Search arXiv.org for academic papers, fetch a paper by ID, "
                    "or download its PDF. Free, no authentication required. "
                    "Operations: search, fetch, download_pdf."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="search",
                        enum=["search", "fetch", "download_pdf"],
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
                        name="arxiv_id",
                        type=ParameterType.STRING,
                        description="arXiv paper ID (fetch / download_pdf).",
                        required=False,
                        min_length=1,
                        max_length=64,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Max search results.",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="dest",
                        type=ParameterType.STRING,
                        description="Destination path for download_pdf.",
                        required=False,
                    ),
                ],
                timeout_seconds=60,
                tags=["knowledge", "arxiv", "papers", "academic", "free"],
                examples=[
                    {"operation": "search", "query": "transformer attention", "max_results": 5},
                    {"operation": "fetch", "arxiv_id": "1706.03762"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str = "search",
        query: str | None = None,
        arxiv_id: str | None = None,
        max_results: int = 10,
        dest: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()
        if op == "search":
            if not query:
                raise ValueError(
                    "operation='search' requires 'query'. "
                    "Pass query with the search terms, or use operation='get' "
                    "with an arXiv id."
                )
            return await self._search(query, max_results)
        if op == "fetch":
            if not arxiv_id:
                raise ValueError("operation='fetch' requires 'arxiv_id'")
            return await self._fetch(arxiv_id)
        if op == "download_pdf":
            if not arxiv_id:
                raise ValueError("operation='download_pdf' requires 'arxiv_id'")
            return await self._download_pdf(arxiv_id, dest)
        raise ValueError(f"Unknown operation: {operation}")

    async def _search(self, query: str, max_results: int) -> dict[str, Any]:
        params = urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        })
        url = f"{ARXIV_BASE}?{params}"
        raw = await _http_get_polite(url, "application/atom+xml")
        root = ET.fromstring(raw)
        papers = [_parse_entry(e) for e in root.findall(f"{ATOM_NS}entry")]
        total_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        total = int(total_el.text) if total_el is not None and total_el.text else len(papers)
        return {
            "query": query,
            "count": len(papers),
            "total_count": total,
            "results": papers,
            "source": "arxiv",
        }

    async def _fetch(self, arxiv_id: str) -> dict[str, Any]:
        aid = _normalize_arxiv_id(arxiv_id)
        params = urlencode({"id_list": aid})
        url = f"{ARXIV_BASE}?{params}"
        raw = await _http_get_polite(url, "application/atom+xml")
        root = ET.fromstring(raw)
        entries = root.findall(f"{ATOM_NS}entry")
        papers = [_parse_entry(e) for e in entries]
        # arXiv returns a stub entry with no title when the id is unknown.
        papers = [p for p in papers if p.get("title")]
        if not papers:
            raise ValueError(f"arXiv paper not found: {aid}")
        return {
            "arxiv_id": aid,
            "result": papers[0],
            "source": "arxiv",
        }

    async def _download_pdf(self, arxiv_id: str, dest: str | None) -> dict[str, Any]:
        aid = _normalize_arxiv_id(arxiv_id)
        url = f"https://arxiv.org/pdf/{aid}.pdf"
        raw = await _http_get_polite(url, "application/pdf")
        out_path = dest or os.path.join(os.getcwd(), f"{aid.replace('/', '_')}.pdf")
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(raw)
        return {
            "arxiv_id": aid,
            "path": os.path.abspath(out_path),
            "bytes": len(raw),
            "url": url,
            "source": "arxiv",
        }
