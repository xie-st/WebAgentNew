"""Tests for PubMedTool.

The operation tests hit the real NCBI E-utilities endpoint; a network failure or
a transient upstream condition (a 429, a body cut short) is a skip rather than a
failure, because it says nothing about this code. The retry contract those
conditions trigger cannot be requested from NCBI on demand, so it is pinned at
the end of the file against a stand-in transport.
"""

from __future__ import annotations

import asyncio
import socket
import time
from http.client import IncompleteRead
from urllib.error import HTTPError

import pytest

from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin import PubMedTool
from effgen.tools.builtin import pubmed as pubmed_module


def _has_network(host: str = "eutils.ncbi.nlm.nih.gov", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


#: Upstream conditions the tool reports after exhausting its own backoff. NCBI
#: sheds load with 429s and with bodies cut short mid-read, and neither says
#: anything about this code, so they skip instead of failing the run.
_UPSTREAM_SIGNS = (
    "429",
    "truncated response",
    "network error",
    "dropped",
    "timed out",
    "after 4 retries",
    # An empty efetch body: NCBI sheds load this way too, and a fetch that got
    # no record now reports it instead of returning an empty result list. A
    # response whose article markup could not be read says so in different
    # words, so it still fails the run.
    "returned no record",
    # A search NCBI answered with an ERROR field rather than matches.
    "returned an error for the search",
)


def _ok(result: ToolResult):
    assert isinstance(result, ToolResult)
    if not result.success:
        error = result.error or ""
        if any(sign in error.lower() for sign in _UPSTREAM_SIGNS):
            pytest.skip(f"PubMed upstream unavailable: {error}")
        pytest.fail(f"tool failed: {error}")
    return result.output


def test_pubmed_metadata():
    tool = PubMedTool()
    assert tool.metadata.name == "pubmed"
    assert tool.metadata.category.value == "information_retrieval"


def test_pubmed_unknown_operation_fails():
    r = _run(PubMedTool().execute(operation="nope", query="x"))
    assert not r.success
    assert "operation" in (r.error or "").lower() or "unknown" in (r.error or "").lower()


def test_pubmed_search_requires_query():
    r = _run(PubMedTool().execute(operation="search"))
    assert not r.success


def test_pubmed_fetch_requires_pmid():
    r = _run(PubMedTool().execute(operation="fetch"))
    assert not r.success


@needs_net
def test_pubmed_search_returns_results():
    out = _ok(_run(PubMedTool().execute(query="crispr gene editing", max_results=3)))
    assert out["source"] == "pubmed"
    assert out["count"] >= 1
    first = out["results"][0]
    assert first["pmid"]
    assert first["pmid"].isdigit()
    assert first["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    assert isinstance(first["authors"], list)


@needs_net
def test_pubmed_abstract_for_known_pmid():
    # PMID 28980624 — "Attention Is All You Need" companion-style biomed paper
    # is not on PubMed; pick a stable biomed PMID instead.
    # 26952870 — well-known CRISPR review; stable abstract.
    out = _ok(_run(PubMedTool().execute(operation="abstract", pmid="26952870")))
    assert out["source"] == "pubmed"
    assert out["count"] >= 1
    item = out["results"][0]
    assert item["pmid"] == "26952870"
    assert isinstance(item.get("abstract"), str)


@needs_net
def test_pubmed_fetch_returns_metadata():
    out = _ok(_run(PubMedTool().execute(operation="fetch", pmid="26952870")))
    assert out["count"] >= 1
    paper = out["results"][0]
    assert paper["title"]
    assert paper["pmid"] == "26952870"
    assert isinstance(paper["authors"], list)


@needs_net
def test_pubmed_rate_limiter_handles_concurrent_burst():
    async def burst():
        tool = PubMedTool()
        started = time.monotonic()
        results = await asyncio.gather(*[
            tool.execute(operation="search", query="crispr gene editing", max_results=1)
            for _ in range(5)
        ])
        return time.monotonic() - started, results

    elapsed, results = _run(burst())
    outputs = [_ok(result) for result in results]
    assert len(outputs) == 5
    assert all(output["count"] >= 1 for output in outputs)
    # Each search performs esearch + esummary. Five concurrent searches should
    # be paced by the local limiter rather than leaving as an immediate burst.
    assert elapsed >= 2.0


# -- transient upstream conditions retry rather than failing the call ---------
#
# NCBI cannot be asked for a truncated body on demand, so the retry contract is
# pinned against a stand-in transport; the operations above prove the live path.

class _FakeResponse:
    """A response whose ``read`` either returns a body or breaks mid-body."""

    def __init__(self, body: bytes | None) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def read(self) -> bytes:
        if self._body is None:
            raise IncompleteRead(b"")
        return self._body


_SEARCH_BODY = b'{"esearchresult": {"idlist": [], "count": "0"}}'


def _transport(bodies: list[bytes | None], seen: list[str]):
    def opener(url, headers=None, timeout=20, allowed_hosts=None):
        seen.append(url)
        return _FakeResponse(bodies[min(len(seen) - 1, len(bodies) - 1)])

    return opener


def test_truncated_body_is_retried(monkeypatch):
    """A body cut short mid-read is transient: back off and ask again."""
    seen: list[str] = []
    monkeypatch.setattr(
        pubmed_module, "safe_urlopen", _transport([None, None, _SEARCH_BODY], seen),
    )
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success, result.error
    assert len(seen) == 3, "two truncated reads, then the answer"


def test_truncated_body_that_never_recovers_names_the_cause(monkeypatch):
    """Once the retries are spent the tool reports a network failure, not a repr."""
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([None], seen))
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success is False
    assert "truncated response" in (result.error or "").lower()
    assert "retries" in (result.error or "")
    assert result.metadata["error_type"] == "ConnectionError"
    assert len(seen) == 5, "the initial call plus four retries"


def test_a_gateway_error_is_retried_too(monkeypatch):
    """A 503 from the gateway is the same shedding behavior as a 429."""
    seen: list[str] = []

    def opener(url, headers=None, timeout=20, allowed_hosts=None):
        seen.append(url)
        if len(seen) == 1:
            raise HTTPError(url, 503, "Service Unavailable", {}, None)
        return _FakeResponse(_SEARCH_BODY)

    monkeypatch.setattr(pubmed_module, "safe_urlopen", opener)
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success, result.error
    assert len(seen) == 2


def test_an_esearch_error_field_is_reported_as_a_failure(monkeypatch):
    """NCBI answering a search with an ERROR field is a failure, not zero matches."""
    seen: list[str] = []
    body = b'{"esearchresult": {"ERROR": "Search Backend failed", "idlist": []}}'
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success is False
    assert "Search Backend failed" in (result.error or "")
    assert result.metadata["error_type"] == "ConnectionError"


def test_a_search_that_matches_nothing_still_succeeds(monkeypatch):
    """No match is an answer: zero results, reported as a successful search."""
    seen: list[str] = []
    body = (
        b'{"esearchresult": {"count": "0", "idlist": [],'
        b' "warninglist": {"phrasesnotfound": ["zzzznotaterm"]}}}'
    )
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="search", query="zzzznotaterm"))
    assert result.success, result.error
    assert result.output["count"] == 0
    assert result.output["results"] == []


def test_an_efetch_error_document_is_reported_as_a_failure(monkeypatch):
    """NCBI answering 200 with an error document is a failed fetch, not a result."""
    seen: list[str] = []
    body = b"<eFetchResult><ERROR>Empty id list - nothing to do</ERROR></eFetchResult>"
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="26952870"))
    assert result.success is False
    assert "returned no record for pmid 26952870" in (result.error or "")
    assert "Empty id list" in (result.error or "")
    assert result.metadata["error_type"] == "ConnectionError"


def test_an_empty_article_set_is_reported_as_a_failure(monkeypatch):
    """An empty set names the id, where to check it, and that a retry may help."""
    seen: list[str] = []
    monkeypatch.setattr(
        pubmed_module, "safe_urlopen", _transport([b"<PubmedArticleSet/>"], seen)
    )
    result = _run(PubMedTool().execute(operation="fetch", pmid="26952870"))
    assert result.success is False
    error = result.error or ""
    assert "returned no record for pmid 26952870" in error
    assert "https://pubmed.ncbi.nlm.nih.gov/26952870/" in error
    assert "retrying" in error


def test_an_empty_article_set_fails_the_abstract_operation_too(monkeypatch):
    """``abstract`` reads the same response, so it reports the same failure."""
    seen: list[str] = []
    monkeypatch.setattr(
        pubmed_module, "safe_urlopen", _transport([b"<PubmedArticleSet/>"], seen)
    )
    result = _run(PubMedTool().execute(operation="abstract", pmid="26952870"))
    assert result.success is False
    assert "returned no record for pmid 26952870" in (result.error or "")


def test_a_single_article_without_the_set_wrapper_is_read(monkeypatch):
    """efetch returning the article as the root element still yields the record."""
    seen: list[str] = []
    body = (
        b"<PubmedArticle><MedlineCitation><PMID>26952870</PMID>"
        b"<Article><ArticleTitle>A title</ArticleTitle></Article>"
        b"</MedlineCitation></PubmedArticle>"
    )
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="26952870"))
    assert result.success, result.error
    assert result.output["count"] == 1
    assert result.output["results"][0]["pmid"] == "26952870"
    assert result.output["results"][0]["title"] == "A title"


def test_article_markup_that_cannot_be_read_is_reported_as_a_defect(monkeypatch):
    """Article markup the parser cannot reach is this tool's problem, said plainly.

    Worded apart from an empty response so a live run fails on it rather than
    treating it as the upstream having nothing to give.
    """
    seen: list[str] = []
    body = b"<PubmedArticleSet><!-- <PubmedArticle>dropped</PubmedArticle> --></PubmedArticleSet>"
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="26952870"))
    assert result.success is False
    error = result.error or ""
    assert "could not read" in error
    assert "returned no record" not in error
    assert result.metadata["error_type"] == "ValueError"


_BOOK_BODY = (
    b'<?xml version="1.0" ?><PubmedArticleSet><PubmedBookArticle><BookDocument>'
    b"<PMID>29262147</PMID>"
    b"<Book><BookTitle>StatPearls</BookTitle><PubDate><Year>2026</Year></PubDate></Book>"
    b"<ArticleTitle>Corns</ArticleTitle>"
    b"<AuthorList><Author><LastName>Al Aboud</LastName><ForeName>Ahmad</ForeName>"
    b"</Author></AuthorList>"
    b"<Abstract><AbstractText>A corn is a focal hyperkeratosis.</AbstractText></Abstract>"
    b"</BookDocument></PubmedBookArticle></PubmedArticleSet>"
)


def test_a_bookshelf_chapter_is_a_record(monkeypatch):
    """A chapter comes back as ``PubmedBookArticle`` and is still a record.

    StatPearls and GeneReviews are indexed this way. Reading only
    ``PubmedArticle`` would report an id that resolves on PubMed as one the
    service has nothing for, and tell the caller to retry a call that can only
    return the same thing.
    """
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([_BOOK_BODY], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="29262147"))
    assert result.success, result.error
    assert result.output["count"] == 1
    record = result.output["results"][0]
    assert record["pmid"] == "29262147"
    assert record["title"] == "Corns"
    # A chapter has no journal, so it is placed by the book that contains it.
    assert record["journal"] == "StatPearls"
    assert record["abstract"].startswith("A corn")
    assert record["authors"] == ["Ahmad Al Aboud"]


def test_a_bookshelf_chapter_answers_the_abstract_operation(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([_BOOK_BODY], seen))
    result = _run(PubMedTool().execute(operation="abstract", pmid="29262147"))
    assert result.success, result.error
    assert result.output["results"][0]["abstract"].startswith("A corn")


def test_a_record_for_a_whole_book_is_named_by_its_book(monkeypatch):
    """A book's own record carries no chapter title, so the book names it."""
    body = (
        b"<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>20301295</PMID>"
        b"<Book><BookTitle>GeneReviews</BookTitle></Book>"
        b"</BookDocument></PubmedBookArticle></PubmedArticleSet>"
    )
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="20301295"))
    assert result.success, result.error
    assert result.output["results"][0]["title"] == "GeneReviews"


def test_the_publication_year_is_reported(monkeypatch):
    """``year`` carries the publication year rather than an empty string.

    ``<Year>`` is a leaf, and an element with no children is falsy, so selecting
    it with ``find(a) or find(b)`` discarded every match it found.
    """
    body = (
        b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>26952870</PMID>"
        b"<Article><ArticleTitle>A title</ArticleTitle>"
        b"<Journal><Title>Cell stem cell</Title>"
        b"<JournalIssue><PubDate><Year>2016</Year></PubDate></JournalIssue></Journal>"
        b"</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="26952870"))
    assert result.success, result.error
    record = result.output["results"][0]
    assert record["year"] == "2016"
    assert record["journal"] == "Cell stem cell"


def test_a_medline_date_is_used_when_there_is_no_year(monkeypatch):
    """The documented fallback for a record dated as a range."""
    body = (
        b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>"
        b"<Article><ArticleTitle>T</ArticleTitle>"
        b"<Journal><JournalIssue><PubDate>"
        b"<MedlineDate>1998 Dec-1999 Jan</MedlineDate>"
        b"</PubDate></JournalIssue></Journal></Article>"
        b"</MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="1"))
    assert result.success, result.error
    assert result.output["results"][0]["year"] == "1998 Dec-1999 Jan"


def test_unreadable_chapter_markup_is_reported_as_a_defect_too(monkeypatch):
    """A chapter element the parser cannot reach is this tool's problem, not NCBI's."""
    body = b"<PubmedArticleSet><!-- <PubmedBookArticle/> --></PubmedArticleSet>"
    seen: list[str] = []
    monkeypatch.setattr(pubmed_module, "safe_urlopen", _transport([body], seen))
    result = _run(PubMedTool().execute(operation="fetch", pmid="29262147"))
    assert result.success is False
    assert "could not read" in (result.error or "")
    assert result.metadata["error_type"] == "ValueError"


def test_a_client_error_is_not_retried(monkeypatch):
    """A 400 is the request's fault, so it is reported at once."""
    seen: list[str] = []

    def opener(url, headers=None, timeout=20, allowed_hosts=None):
        seen.append(url)
        raise HTTPError(url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(pubmed_module, "safe_urlopen", opener)
    result = _run(PubMedTool().execute(operation="search", query="crispr", max_results=1))
    assert result.success is False
    assert "400" in (result.error or "")
    assert len(seen) == 1, "a client error is not worth a retry"
