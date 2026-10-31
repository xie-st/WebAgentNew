"""Self-contained HTML reports for ``run`` / ``compare`` / ``eval`` / ``cost`` / ``loadtest``.

Renders a result object that a command already produced into a single ``.html``
file that can be emailed, attached, or opened from disk. The output has no
external references of any kind: every style, script, and chart is inline, so
the file renders identically with the network disabled.

Colors come from :mod:`effgen.ui.palette`, the same source the dashboard uses,
and are emitted as CSS custom properties for a light and a dark palette. The
page follows the reader's ``prefers-color-scheme`` and carries a toggle that
overrides it per view.

Six report shapes share one visual system, one per module:

``run`` (:mod:`effgen.ui.report_html_run`)
    One agent run as a shareable card: the task, the model it ran on, the
    answer (or the typed error), the step-by-step tool trace with per-step
    durations, sources and citations, and the run's tokens, cost and latency.
``comparison`` (:mod:`effgen.ui.report_html_comparison`)
    Per-model accuracy / latency / cost for a bake-off, with the recommended
    model and the reason it was picked.
``eval`` (:mod:`effgen.ui.report_html_eval`)
    Suite pass rate, the exit gate, a by-difficulty breakdown, and every case.
``cost`` (:mod:`effgen.ui.report_html_cost`)
    Spend for a period against the configured daily budget, by provider/model.
``loadtest`` (:mod:`effgen.ui.report_html_loadtest`)
    Latency percentiles, throughput, error rate, and the error breakdown.
``battle`` (:mod:`effgen.ui.report_html_battle`)
    Several models on one prompt: the tally, the verdict, and every answer.

All six draw on the same three shared layers: :mod:`effgen.ui.report_html_format`
(escaping, value formatting, document normalizing),
:mod:`effgen.ui.report_html_components` (cards, tables, badges, inline charts),
and :mod:`effgen.ui.report_html_page` (the inline stylesheet, script and page
shell). Every name they define is re-exported here, so this module stays the
one import path for the report renderer.

Typical use::

    from effgen.ui.report_html import write_html_report

    write_html_report("bakeoff.html", matrix.to_dict(), kind="comparison")

:func:`detect_report_kind` identifies a saved ``--json`` document, so a result
captured earlier can be rendered later without re-running any model.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .report_html_battle import _battle_body
from .report_html_comparison import _comparison_body
from .report_html_components import (  # noqa: F401  re-exported for import/patch parity
    _SERIES_ROLES,
    _badge,
    _bar_chart,
    _card,
    _cards,
    _donut,
    _link,
    _meter,
    _table,
)
from .report_html_cost import _cost_body
from .report_html_eval import _eval_body
from .report_html_format import (  # noqa: F401  re-exported for import/patch parity
    _DASH,
    ReportError,
    _esc,
    _fraction,
    _int,
    _mapping,
    _ms,
    _number,
    _pct,
    _rps,
    _secs,
    _sequence,
    _truncate,
    _usd,
)
from .report_html_loadtest import _loadtest_body
from .report_html_page import (  # noqa: F401  re-exported for import/patch parity
    _THEME_SCRIPT,
    _css,
    _page,
    _provenance_items,
)
from .report_html_run import (  # noqa: F401  re-exported for import/patch parity
    _LOCAL_ENGINES,
    _children,
    _run_body,
    _run_command,
    _sort_key,
    _tool_steps,
    _unpriced_label,
)

__all__ = [
    "REPORT_KINDS",
    "ReportError",
    "build_html_report",
    "detect_report_kind",
    "load_result_document",
    "write_html_report",
]

REPORT_KINDS: tuple[str, ...] = ("run", "comparison", "eval", "cost", "loadtest", "battle")

#: The keys each kind renders from. A document carrying none of a kind's keys
#: cannot produce a report with data in it, so rendering is refused rather than
#: writing a file of em dashes.
_KIND_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("output", "task", "execution_trace", "execution_tree"),
    "comparison": ("scores",),
    "eval": ("results", "suite", "accuracy"),
    "cost": ("rows", "total_cost_usd", "period"),
    "loadtest": ("latency", "throughput_rps", "total_requests"),
    "battle": ("contenders", "prompt", "verdict"),
}


def detect_report_kind(data: dict[str, Any]) -> str | None:
    """Return the report kind a result document describes, or ``None``.

    Recognizes the ``--json`` documents emitted by ``run`` (``output`` +
    ``success``), ``compare`` (``scores``), ``eval`` (``suite`` + ``results``),
    ``cost`` (``rows`` + ``period``), ``loadtest`` (``latency`` +
    ``throughput_rps``), and ``battle`` (``contenders`` + ``prompt``), plus a
    stored run-history record (``run_id`` + ``status``).
    """
    if not isinstance(data, dict):
        return None
    if "contenders" in data and "prompt" in data:
        return "battle"
    if "scores" in data and "recommendations" in data:
        return "comparison"
    if "results" in data and "suite" in data:
        return "eval"
    if "rows" in data and "period" in data:
        return "cost"
    if "latency" in data and "throughput_rps" in data:
        return "loadtest"
    if "output" in data and ("success" in data or "execution_tree" in data):
        return "run"
    if "run_id" in data and "status" in data and "ts" in data:
        return "run"
    return None


#: The builder for each report kind. One module per kind, so the shape a kind
#: renders is read in one place.
_BODY_RENDERERS = {
    "run": _run_body,
    "comparison": _comparison_body,
    "eval": _eval_body,
    "cost": _cost_body,
    "loadtest": _loadtest_body,
    "battle": _battle_body,
}


def _require_kind_data(data: dict[str, Any], kind: str) -> None:
    """Refuse a document that carries none of the keys *kind* renders.

    Rendering it would produce a page of em dashes, so the mismatch is reported
    instead — naming the keys the kind needs, the keys the document has, and
    the kind the document does look like when that can be told.

    Raises:
        ReportError: If the document has none of the kind's keys.
    """
    required = _KIND_KEYS.get(kind, ())
    if not required or any(key in data for key in required):
        return
    present = ", ".join(sorted(data)[:8]) or "no keys"
    actual = detect_report_kind(data)
    hint = (
        f" This document looks like a {actual} result — render it with --kind {actual}."
        if actual and actual != kind
        else ""
    )
    raise ReportError(
        f"This document has none of the fields the '{kind}' report renders "
        f"({', '.join(required)}). It contains: {present}.{hint} "
        "Pass the JSON that command emitted, or name the shape with --kind."
    )


def build_html_report(
    data: dict[str, Any],
    *,
    kind: str | None = None,
    command: str | None = None,
    generated_at: str | None = None,
) -> str:
    """Return a self-contained HTML document for a result *data* mapping.

    Args:
        data: The result document — the same mapping a command's ``--json``
            emits (``AgentResponse.to_dict()``, ``ComparisonMatrix.to_dict()``,
            ``SuiteResults.summary()``, the ``cost`` spend document, or
            ``LoadReport.to_dict()``).
        kind: One of :data:`REPORT_KINDS`. Inferred from the document shape
            when omitted.
        command: The invocation that produced the result, stamped into the
            report header so a shared file can be traced back and re-run.
        generated_at: Timestamp for the header. Defaults to the current UTC
            time in ISO-8601 form.

    Returns:
        A complete HTML document with every style, script, and chart inline.

    Raises:
        ReportError: If *kind* is unknown, cannot be inferred, or the document
            is missing the data the report needs.
    """
    if not isinstance(data, dict):
        raise ReportError(
            "A report is rendered from a JSON object, not a list or scalar. "
            "Pass the JSON an effGen command emitted, not one element of it."
        )
    resolved = kind or detect_report_kind(data)
    if resolved is None:
        raise ReportError(
            "Could not tell which report this result is. Expected the JSON emitted by "
            "`effgen run --json`, `effgen compare --json`, `effgen eval --json`, "
            "`effgen cost --json`, or `effgen loadtest`. Pass --kind to say explicitly "
            f"({', '.join(REPORT_KINDS)})."
        )
    if resolved not in _BODY_RENDERERS:
        raise ReportError(
            f"Unknown report kind '{resolved}'. Choose one of: {', '.join(REPORT_KINDS)}."
        )
    _require_kind_data(data, resolved)
    stamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    title, subtitle, body = _BODY_RENDERERS[resolved](data)
    return _page(
        title=title, subtitle=subtitle, command=command, generated_at=stamp, body=body,
    )


def write_html_report(
    path: str | Path,
    data: dict[str, Any],
    *,
    kind: str | None = None,
    command: str | None = None,
) -> Path:
    """Render *data* and write it to *path*, creating parent directories.

    Returns the path written.

    Args:
        path: Where to write the report.
        data: The report contents to render.
        kind: Which report template to use, inferred from *data* when absent.
        command: The command line recorded in the report header.
    """
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_html_report(data, kind=kind, command=command), encoding="utf-8"
    )
    return target


def load_result_document(path: str | Path) -> dict[str, Any]:
    """Read a saved ``--json`` result document from *path*.

    Raises:
        ReportError: If the file is missing or is not a JSON object.
    """
    src = Path(path)
    if not src.exists():
        raise ReportError(
            f"No such result file: {src}. Check the path, or produce one with "
            "`effgen eval --suite math --json > results.json`."
        )
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportError(
            f"{src} is not valid JSON ({exc}). Save one first with, for example, "
            "`effgen eval --suite math --json > results.json`."
        ) from exc
    if not isinstance(data, dict):
        raise ReportError(
            f"{src} holds a {type(data).__name__}, not a result object. Expected the "
            "JSON emitted by run/compare/eval/cost/loadtest."
        )
    return data
