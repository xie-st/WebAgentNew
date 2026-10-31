"""Property tests for HTML-report escaping.

A shareable report renders a result document whose text came from a model, a tool,
a provider error and a user's own task. Any of it can carry markup. The report is
opened in a browser, so text that reaches the page as live markup is an injection
into whatever the reader opens.

Asserts, for every report kind and every string position in its document:
  1. No injected payload reaches the page as markup — the rendered document has
     the same element and attribute set it has with inert text in the same place.
  2. No ``href`` is emitted for a scheme other than ``http``/``https``, so a
     ``javascript:`` or ``data:`` source from a model is inert text.
  3. The text is not silently dropped either: an escaped form of it is present.
  4. A document that is not a result document raises ``ReportError`` — the only
     error the renderer contracts for.

Exit criterion: every payload × every string position of every report kind, plus
≥200 Hypothesis examples over generated text.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.ui.report_html import (
    REPORT_KINDS,
    ReportError,
    build_html_report,
    write_html_report,
)
from tests.fuzz.documents import INJECTION_PAYLOADS

pytestmark = pytest.mark.fuzz

_INERT = "PLAINTEXTMARKER"

#: A minimal but realistic document for every report kind, with a string leaf in
#: each position the renderer reads.
DOCUMENTS: dict[str, dict[str, Any]] = {
    "run": {
        "output": _INERT,
        "success": True,
        "task": _INERT,
        "model": _INERT,
        "provider": _INERT,
        "run_id": _INERT,
        "sources": [_INERT, f"https://example.com/{_INERT}"],
        "citations": [{"index": 1, "quote": _INERT, "url": _INERT, "title": _INERT}],
        "metadata": {"cost_usd": 0.5, "total_tokens": 10, "finish_reason": _INERT},
        "execution_tree": {
            "name": _INERT,
            "children": [
                {
                    "name": _INERT,
                    "type": "tool",
                    "metadata": {"tool_input": _INERT, "output": _INERT},
                }
            ],
        },
        "error": {"message": _INERT, "type": _INERT, "category": _INERT,
                  "provider": _INERT, "model": _INERT},
    },
    "comparison": {
        "scores": [
            {
                "model": _INERT,
                "suite": _INERT,
                "accuracy": 0.5,
                "avg_latency": 1.0,
                "avg_cost_usd": 0.1,
                "total_tokens": 10,
                "responses": [
                    {"query": _INERT, "output": _INERT, "passed": True}
                ],
            }
        ],
        "recommendations": {_INERT: _INERT},
        "recommendation_rationale": {_INERT: _INERT},
        "optimize": _INERT,
        "scoring": _INERT,
        "num_cases": 1,
    },
    "eval": {
        "suite": _INERT,
        "results": [
            {"query": _INERT, "expected": _INERT, "actual": _INERT, "passed": True,
             "model": _INERT}
        ],
        "summary": {"accuracy": 0.5, "model": _INERT},
        "model": _INERT,
    },
    "cost": {
        "period": _INERT,
        "rows": [{"model": _INERT, "provider": _INERT, "cost_usd": 0.5, "runs": 2}],
        "total_cost_usd": 0.5,
        "budget": {"limit_usd": 1.0, "name": _INERT},
    },
    "loadtest": {
        "latency": {"p50_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.5},
        "throughput_rps": 5.0,
        "target": _INERT,
        "model": _INERT,
        "errors": {_INERT: 2},
        "summary": {"total": 10, "failed": 2, "note": _INERT},
    },
    "battle": {
        "prompt": _INERT,
        "contenders": [
            {"model": _INERT, "output": _INERT, "provider": _INERT,
             "cost_usd": 0.1, "latency_ms": 2.0}
        ],
        "winner": _INERT,
        "verdict": _INERT,
    },
}


class _Page(HTMLParser):
    """Collect the element and attribute names a rendered page actually contains."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attrs: set[str] = set()
        self.hrefs: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            self.attrs.add(name)
            if name == "href" and value is not None:
                self.hrefs.append(value)

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


def _inject(document: Any, payload: str) -> Any:
    """Return *document* with every ``_INERT`` leaf replaced by *payload*."""
    if isinstance(document, str):
        return document.replace(_INERT, payload)
    if isinstance(document, list):
        return [_inject(item, payload) for item in document]
    if isinstance(document, dict):
        return {_inject(k, payload): _inject(v, payload) for k, v in document.items()}
    return document


# ---------------------------------------------------------------------------
# The page shape does not change with the payload
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(DOCUMENTS))
@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_a_payload_never_changes_the_page_structure(kind, payload) -> None:
    """Injected markup renders as text: same elements, same attributes."""
    baseline = _parse(build_html_report(DOCUMENTS[kind], kind=kind))
    injected = _parse(
        build_html_report(_inject(DOCUMENTS[kind], payload), kind=kind)
    )

    assert not set(injected.tags) - set(baseline.tags), (
        f"{kind}: the payload opened elements the page does not otherwise have: "
        f"{sorted(set(injected.tags) - set(baseline.tags))}"
    )
    assert not injected.attrs - baseline.attrs, (
        f"{kind}: the payload added attributes: "
        f"{sorted(injected.attrs - baseline.attrs)}"
    )
    assert injected.tags.count("script") == baseline.tags.count("script"), (
        f"{kind}: the payload changed how many <script> elements the page has"
    )


@pytest.mark.parametrize("kind", sorted(DOCUMENTS))
@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_a_payload_is_rendered_but_never_as_markup(kind, payload) -> None:
    html = build_html_report(_inject(DOCUMENTS[kind], payload), kind=kind)
    if any(ch in payload for ch in '<>&"'):
        # A payload carrying markup characters must never survive verbatim:
        # every occurrence has to have gone through escaping.
        assert payload not in html, f"{kind}: raw payload survived into the page"
    # ... and it must not have been silently dropped either: the parsed page
    # reads it back as text.
    page = _parse(html)
    assert any(payload in chunk for chunk in page.text), (
        f"{kind}: the payload text is neither escaped in the page nor present"
    )


@pytest.mark.parametrize("kind", sorted(DOCUMENTS))
@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_no_link_is_emitted_for_a_scheme_a_browser_would_execute(kind, payload) -> None:
    page = _parse(build_html_report(_inject(DOCUMENTS[kind], payload), kind=kind))
    for href in page.hrefs:
        scheme = href.split(":", 1)[0].lower() if ":" in href else ""
        assert scheme in ("", "http", "https", "#"), (
            f"{kind}: rendered a clickable {scheme}: link — {href!r}"
        )


@pytest.mark.parametrize("kind", sorted(DOCUMENTS))
def test_every_kind_renders_a_complete_page(kind) -> None:
    html = build_html_report(DOCUMENTS[kind], kind=kind)
    assert html.startswith("<!DOCTYPE html>") or html.lstrip().startswith("<!DOCTYPE")
    assert html.rstrip().endswith("</html>")
    # Self-contained: no request leaves the page.
    assert not re.search(r'(src|href)\s*=\s*"https?://(?!example\.com)', html)


# ---------------------------------------------------------------------------
# Generated text
# ---------------------------------------------------------------------------

_HOSTILE_TEXT = st.one_of(
    st.text(alphabet='<>&"\'/\\ abcTAG=script', max_size=60),
    st.text(max_size=40),
    st.sampled_from(list(INJECTION_PAYLOADS)),
)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_HOSTILE_TEXT, kind=st.sampled_from(sorted(DOCUMENTS)))
def test_generated_text_never_opens_an_element(text, kind) -> None:
    """No generated string reaches the page as an element or an attribute."""
    baseline = _parse(build_html_report(DOCUMENTS[kind], kind=kind))
    page = _parse(build_html_report(_inject(DOCUMENTS[kind], text), kind=kind))
    assert not set(page.tags) - set(baseline.tags), (
        f"{kind}: {text!r} opened {sorted(set(page.tags) - set(baseline.tags))}"
    )
    assert not page.attrs - baseline.attrs, (
        f"{kind}: {text!r} added {sorted(page.attrs - baseline.attrs)}"
    )
    assert page.tags.count("script") == baseline.tags.count("script")


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(url=st.text(max_size=60) | st.sampled_from([
    "javascript:alert(1)", "JavaScript:alert(1)", " javascript:alert(1) ",
    "data:text/html,<script>x</script>", "file:///etc/passwd", "vbscript:x",
    "http://example.com/ok", "https://example.com/ok", "//example.com/protocol",
]))
def test_a_source_url_is_clickable_only_when_it_is_http(url) -> None:
    document = dict(DOCUMENTS["run"], sources=[url], citations=[])
    page = _parse(build_html_report(document, kind="run"))
    for href in page.hrefs:
        assert href.split(":", 1)[0].lower() in ("", "http", "https") or ":" not in href


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("document", [None, [], [1, 2], "text", 5, True, {}])
def test_a_document_that_is_not_a_result_raises_report_error(document) -> None:
    with pytest.raises(ReportError):
        build_html_report(document)


def test_an_unknown_kind_names_the_kinds_there_are() -> None:
    with pytest.raises(ReportError) as excinfo:
        build_html_report(DOCUMENTS["run"], kind="nonexistent")
    for kind in REPORT_KINDS:
        assert kind in str(excinfo.value)


@pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
def test_a_written_report_carries_no_raw_payload(tmp_path, payload) -> None:
    target = write_html_report(
        tmp_path / "sub" / "report.html", _inject(DOCUMENTS["run"], payload),
        kind="run",
    )
    written = target.read_text(encoding="utf-8")
    if any(ch in payload for ch in '<>&"'):
        assert payload not in written
