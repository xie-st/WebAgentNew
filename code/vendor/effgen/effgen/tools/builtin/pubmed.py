"""
PubMed tool for the effGen framework.

Backend: NCBI E-utilities (https://eutils.ncbi.nlm.nih.gov/entrez/eutils/).
No authentication required. With NCBI_API_KEY set, rate-limit budget rises
from 3 req/s to 10 req/s; this is enforced via a built-in token bucket.

Operations:
- search: full-text search returning PMIDs
- fetch: retrieve full metadata for a PMID (or list)
- abstract: shortcut returning only the abstract text
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from http.client import HTTPException
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

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Fixed API host (pinned; the shared SSRF guard also blocks internal targets).
_ALLOWED_HOSTS = frozenset({"eutils.ncbi.nlm.nih.gov"})

# The two elements an efetch response wraps a record in. A journal article is a
# ``PubmedArticle``; a chapter indexed from the NCBI Bookshelf (StatPearls,
# GeneReviews) is a ``PubmedBookArticle`` whose ``BookDocument`` carries the same
# fields this reads.
_ARTICLE_TAGS = frozenset({"PubmedArticle", "PubmedBookArticle"})


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


class _TokenBucket:
    """Simple token bucket for rate limiting (process-local).

    Starts empty (no initial burst) so the very first ``acquire`` waits for
    one token to accumulate. This matches NCBI's per-second cap, which
    treats sudden bursts (even within the 3 rps budget) as 429-worthy.
    """

    def __init__(self, rate_per_sec: float, burst: int | None = None) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_sec)))
        self._tokens = 0.0
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._last = time.monotonic()
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# Module-level buckets, sized at first use based on whether NCBI_API_KEY is set.
_BUCKETS: dict[str, _TokenBucket] = {}


def _bucket() -> _TokenBucket:
    has_key = bool(os.environ.get("NCBI_API_KEY"))
    key = "auth" if has_key else "anon"
    if key not in _BUCKETS:
        rate = 10.0 if has_key else 3.0
        # capacity=1 — no initial burst; one request at a time, paced by `rate`.
        _BUCKETS[key] = _TokenBucket(rate, burst=1)
    return _BUCKETS[key]


class _Transient(Exception):
    """An upstream condition worth retrying with backoff.

    Subclasses carry the short label the backoff log names.
    """

    label = "transient error"


class _RateLimited(_Transient):
    """Raised on 429 from NCBI E-utilities to trigger caller backoff."""

    label = "429"


class _Truncated(_Transient):
    """Raised when the connection closes before the response body is complete.

    NCBI sheds load this way as well as with a 429 — a burst that draws 429s
    often ends with a body cut short mid-read — so it gets the same backoff
    rather than surfacing as a failed tool call.
    """

    label = "truncated response"


def _http_get(url: str, accept: str = "application/json", timeout: int = 20) -> bytes:
    headers = {"User-Agent": _user_agent(), "Accept": accept}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=timeout, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code == 429:
            raise _RateLimited(f"HTTP 429 from {url}") from e
        if e.code in (502, 503, 504):
            raise _Truncated(f"HTTP {e.code} from {url}: {e.reason}") from e
        raise ConnectionError(f"HTTP {e.code} from {url}: {e.reason}") from e
    except URLError as e:
        if isinstance(e.reason, TimeoutError | ConnectionResetError):
            raise _Truncated(f"Connection to {url} dropped: {e.reason}") from e
        raise ConnectionError(f"Network error fetching {url}: {e.reason}") from e
    except (HTTPException, ConnectionResetError, TimeoutError) as e:
        # A body cut short (``IncompleteRead``), a reset, or a read timeout
        # arrives here rather than as a URLError, because the connection was
        # already open when it broke.
        raise _Truncated(f"Truncated response from {url}: {e!r}") from e


class PubMedTool(BaseTool):
    """Search PubMed via the NCBI E-utilities API (free, no key required)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="pubmed",
                description=(
                    "Search the PubMed biomedical literature database via NCBI "
                    "E-utilities. Operations: search (returns PMIDs and brief "
                    "metadata), fetch (full metadata for a PMID), abstract "
                    "(abstract text for a PMID). Free; no authentication needed. "
                    "Honors NCBI_API_KEY when set for higher rate limits."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform.",
                        required=False,
                        default="search",
                        enum=["search", "fetch", "abstract"],
                    ),
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query (only for operation=search).",
                        required=False,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="pmid",
                        type=ParameterType.STRING,
                        description="PubMed ID (only for fetch/abstract). Multiple comma-separated IDs accepted.",
                        required=False,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Max search results.",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=100,
                    ),
                ],
                timeout_seconds=25,
                tags=["knowledge", "pubmed", "research", "academic", "free"],
                examples=[
                    {"operation": "search", "query": "transformer attention", "max_results": 5},
                    {"operation": "abstract", "pmid": "37541975"},
                ],
            )
        )

    @staticmethod
    def _api_key_query() -> str:
        key = os.environ.get("NCBI_API_KEY")
        return f"&api_key={key}" if key else ""

    async def _http_get_throttled(
        self, url: str, accept: str = "application/json", *, retries: int = 4
    ) -> bytes:
        backoff = 1.0
        last: _Transient | None = None
        for attempt in range(retries + 1):
            await _bucket().acquire()
            try:
                return await asyncio.to_thread(_http_get, url, accept)
            except _Transient as exc:
                last = exc
                if attempt == retries:
                    break
                logger.warning(
                    "PubMed %s — backing off %.1fs (attempt %d/%d)",
                    exc.label, backoff, attempt + 1, retries,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16.0)
        raise ConnectionError(f"{last} after {retries} retries")

    async def _execute(
        self,
        operation: str = "search",
        query: str | None = None,
        pmid: str | None = None,
        max_results: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        op = operation.lower()
        if op == "search":
            if not query:
                raise ValueError(
                    "operation='search' requires 'query'. "
                    "Pass query with the search terms, or use operation='fetch' "
                    "with a PubMed id."
                )
            return await self._search(query, max_results)
        if op == "fetch":
            if not pmid:
                raise ValueError("operation='fetch' requires 'pmid'")
            return await self._fetch(pmid)
        if op == "abstract":
            if not pmid:
                raise ValueError("operation='abstract' requires 'pmid'")
            return await self._abstract(pmid)
        raise ValueError(f"Unknown operation: {operation}")

    async def _search(self, query: str, max_results: int) -> dict[str, Any]:
        params = urlencode({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        })
        url = f"{NCBI_BASE}/esearch.fcgi?{params}{self._api_key_query()}"
        raw = await self._http_get_throttled(url)
        data = json.loads(raw.decode("utf-8"))
        esearch = data.get("esearchresult", {}) or {}
        # A search that matches nothing is an answer; an ERROR field is not one.
        # NCBI reports a rejected or unserviceable search that way, with HTTP 200
        # and an empty id list. (``errorlist``/``warninglist`` are advisory —
        # a phrase not indexed, a term corrected — and stay a successful search.)
        upstream_error = esearch.get("ERROR") or data.get("ERROR")
        if upstream_error:
            raise ConnectionError(
                f"PubMed returned an error for the search {query!r}: {upstream_error}"
            )
        ids: list[str] = esearch.get("idlist", []) or []
        summaries: list[dict[str, Any]] = []
        if ids:
            sum_params = urlencode({
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            })
            sum_url = f"{NCBI_BASE}/esummary.fcgi?{sum_params}{self._api_key_query()}"
            sum_raw = await self._http_get_throttled(sum_url)
            sum_data = json.loads(sum_raw.decode("utf-8"))
            result = sum_data.get("result", {}) or {}
            for pid in ids:
                item = result.get(pid)
                if not item:
                    continue
                authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
                summaries.append({
                    "pmid": pid,
                    "title": item.get("title", "").strip(),
                    "authors": authors,
                    "journal": item.get("fulljournalname") or item.get("source", ""),
                    "pubdate": item.get("pubdate", ""),
                    "doi": next(
                        (i.get("value") for i in item.get("articleids", [])
                         if i.get("idtype") == "doi"),
                        None,
                    ),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                })
        return {
            "query": query,
            "count": len(summaries),
            "total_count": int(esearch.get("count", len(summaries)) or 0),
            "results": summaries,
            "source": "pubmed",
        }

    async def _fetch(self, pmid: str) -> dict[str, Any]:
        pmid_clean = ",".join(p.strip() for p in pmid.split(",") if p.strip())
        params = urlencode({
            "db": "pubmed",
            "id": pmid_clean,
            "retmode": "xml",
            "rettype": "abstract",
        })
        url = f"{NCBI_BASE}/efetch.fcgi?{params}{self._api_key_query()}"
        raw = await self._http_get_throttled(url, accept="application/xml")
        root = ET.fromstring(raw)
        # Journal articles and Bookshelf chapters are both records; document order
        # is preserved so a comma-separated fetch answers in the order it read.
        if root.tag in _ARTICLE_TAGS:
            found = [root]
        else:
            found = [el for el in root.iter() if el.tag in _ARTICLE_TAGS]
        articles: list[dict[str, Any]] = [_parse_pubmed_article(art) for art in found]
        if not articles:
            # A fetch names the records it wants, so a response carrying none is
            # a failed call — not a result with an empty list in it.
            raise _no_record_error(pmid_clean, raw, root)
        return {
            "pmid": pmid_clean,
            "count": len(articles),
            "results": articles,
            "source": "pubmed",
        }

    async def _abstract(self, pmid: str) -> dict[str, Any]:
        full = await self._fetch(pmid)
        abstracts = [
            {"pmid": a.get("pmid"), "title": a.get("title"), "abstract": a.get("abstract")}
            for a in full.get("results", [])
        ]
        return {
            "pmid": full.get("pmid"),
            "count": len(abstracts),
            "results": abstracts,
            "source": "pubmed",
        }


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _first(art: ET.Element, *paths: str) -> ET.Element | None:
    """Return the first element matching any of *paths*.

    An element with no children is falsy, so ``find(a) or find(b)`` silently
    discards a leaf match like ``<Year>2016</Year>`` and falls through to the
    alternative. These are all leaves, so the identity check is the only correct
    test.
    """
    for path in paths:
        el = art.find(path)
        if el is not None:
            return el
    return None


def _no_record_error(pmid: str, raw: bytes, root: ET.Element) -> Exception:
    """Return the error for an efetch response that carried no article record.

    The three shapes are reported apart, because they call for different things
    from the caller: NCBI's own error document names the reason, a response with
    no article markup at all is an absent or withheld record (or a service under
    load, which is worth retrying), and a response that *does* carry article
    markup this parser could not reach is a defect here.
    """
    # ``<PubmedArticleSet>`` shares the prefix, so require the tag to end here —
    # by whitespace, by ``>``, or by the ``/`` of a self-closing element.
    if re.search(rb"<Pubmed(Book)?Article[\s>/]", raw):
        return ValueError(
            f"PubMed returned a record for pmid {pmid} in a structure this tool "
            "could not read; please report it with that id."
        )
    upstream = " ".join(
        text
        for text in (_text(el) for el in root.iter() if el.tag.upper() == "ERROR")
        if text
    )
    detail = f" NCBI reported: {upstream}." if upstream else ""
    first = pmid.split(",")[0]
    return ConnectionError(
        f"PubMed returned no record for pmid {pmid}.{detail} Check the id at "
        f"https://pubmed.ncbi.nlm.nih.gov/{first}/ — if it resolves there, the "
        "service returned an empty response and the call is worth retrying."
    )


def _parse_pubmed_article(art: ET.Element) -> dict[str, Any]:
    pmid_el = art.find(".//PMID")
    # A record for a whole book carries no chapter title, so it is named by its
    # book; a chapter has no journal, and its book is the containing work.
    title_el = _first(art, ".//ArticleTitle", ".//Book/BookTitle")
    journal_el = _first(art, ".//Journal/Title", ".//Book/BookTitle")
    year_el = _first(art, ".//PubDate/Year", ".//PubDate/MedlineDate")
    authors: list[str] = []
    for au in art.findall(".//AuthorList/Author"):
        last = au.findtext("LastName") or ""
        first = au.findtext("ForeName") or au.findtext("Initials") or ""
        name = f"{first} {last}".strip()
        if name:
            authors.append(name)
    abstract_parts: list[str] = []
    for ab in art.findall(".//Abstract/AbstractText"):
        label = ab.attrib.get("Label")
        text = _text(ab)
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    doi = None
    for aid in art.findall(".//ArticleIdList/ArticleId"):
        if aid.attrib.get("IdType", "").lower() == "doi":
            doi = (aid.text or "").strip()
            break
    pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
    return {
        "pmid": pmid,
        "title": _text(title_el),
        "authors": authors,
        "journal": _text(journal_el),
        "year": _text(year_el),
        "abstract": "\n".join(p for p in abstract_parts if p),
        "doi": doi,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    }
