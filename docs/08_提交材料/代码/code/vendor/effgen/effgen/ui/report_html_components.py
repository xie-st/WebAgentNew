"""Shared page components for HTML reports: cards, tables, badges and charts.

Every report kind is assembled from these, so the five shapes share one visual
system. The charts are inline SVG and CSS-sized bars rather than a plotting
library, which is what keeps a rendered report self-contained with no external
reference of any kind.

Two conventions run through the components: a state is always spelled out in
text beside its color, so color is never the only cue; and a value the document
does not carry renders as a hatched gap rather than a zero-length bar.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlparse

from .report_html_format import _esc, _fraction, _usd

# Series colors for multi-category charts, drawn from the shared palette roles
# and ordered so adjacent series stay distinguishable.
_SERIES_ROLES: tuple[str, ...] = ("accent", "accent2", "ok", "warn", "err")


def _card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="card-note">{_esc(note)}</div>' if note else ""
    return (
        '<div class="card">'
        f'<div class="card-label">{_esc(label)}</div>'
        f'<div class="card-value">{_esc(value)}</div>'
        f"{note_html}</div>"
    )


def _cards(items: list[tuple[str, str, str]]) -> str:
    return '<div class="cards">' + "".join(_card(*i) for i in items) + "</div>"


def _table(columns: list[str], rows: list[list[str]], *, caption: str = "") -> str:
    """Render a table. Cells are pre-escaped markup so a cell may carry a badge."""
    if not rows:
        return '<p class="empty">No rows to show.</p>'
    head = "".join(f"<th scope=\"col\">{_esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (
        '<div class="table-wrap"><table>'
        f"{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody>"
        "</table></div>"
    )


def _badge(text: str, role: str) -> str:
    """A status pill. The text carries the state, so color is never the only cue."""
    return f'<span class="badge badge-{_esc(role)}">{_esc(text)}</span>'


def _link(url: Any, text: Any = None) -> str:
    """Render *url* as a link when its scheme is ``http`` or ``https``.

    Sources and citations come from model and tool output, so any other scheme
    (``javascript:``, ``data:``, ``file:``, a bare fragment) is rendered as
    inert escaped text instead of a clickable target.
    """
    raw = "" if url is None else str(url)
    label = _esc(raw if text is None else text)
    try:
        scheme = urlparse(raw.strip()).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in ("http", "https"):
        return f'<span class="wrap">{label}</span>'
    return (
        f'<a class="wrap" href="{_esc(raw.strip())}" rel="noreferrer noopener" '
        f'target="_blank">{label}</a>'
    )


def _bar_chart(
    title: str,
    rows: list[tuple[str, float | None, str]],
    *,
    role: str = "accent",
    maximum: float | None = None,
) -> str:
    """Horizontal bars: ``(label, value, formatted_value)`` per row.

    A row whose value is ``None`` renders as a labeled gap rather than a
    zero-length bar, so an unpriced or unmeasured entry is visibly distinct
    from a genuine zero.
    """
    numeric = [v for _, v, _ in rows if isinstance(v, int | float) and not isinstance(v, bool)]
    top = maximum if maximum is not None else (max(numeric) if numeric else 0.0)
    items = []
    for label, value, formatted in rows:
        if value is None:
            fill = '<div class="bar-track"><div class="bar-none"></div></div>'
        else:
            width = _fraction(value, top) * 100
            fill = (
                '<div class="bar-track">'
                f'<div class="bar-fill bar-{_esc(role)}" style="width:{width:.2f}%"></div>'
                "</div>"
            )
        items.append(
            '<div class="bar-row">'
            f'<div class="bar-label" title="{_esc(label)}">{_esc(label)}</div>'
            f"{fill}"
            f'<div class="bar-value">{_esc(formatted)}</div>'
            "</div>"
        )
    return (
        f'<section class="chart"><h3>{_esc(title)}</h3>'
        + "".join(items)
        + "</section>"
    )


def _donut(title: str, slices: list[tuple[str, float]]) -> str:
    """Inline-SVG donut of cost share. Each slice is also listed in the legend."""
    total = sum(v for _, v in slices if v > 0)
    if total <= 0:
        return ""
    radius, stroke = 60.0, 26.0
    circumference = 2 * math.pi * radius
    offset = 0.0
    arcs, legend = [], []
    for idx, (label, value) in enumerate(slices):
        if value <= 0:
            continue
        role = _SERIES_ROLES[idx % len(_SERIES_ROLES)]
        share = value / total
        length = share * circumference
        arcs.append(
            f'<circle class="slice slice-{role}" cx="80" cy="80" r="{radius}" '
            f'fill="none" stroke-width="{stroke}" '
            f'stroke-dasharray="{length:.3f} {circumference - length:.3f}" '
            f'stroke-dashoffset="{-offset:.3f}" transform="rotate(-90 80 80)">'
            f"<title>{_esc(label)}: {share * 100:.1f}%</title></circle>"
        )
        legend.append(
            '<li><span class="swatch swatch-' + role + '" aria-hidden="true"></span>'
            f"{_esc(label)} <span class=\"legend-value\">{share * 100:.1f}%</span></li>"
        )
        offset += length
    return (
        f'<section class="chart"><h3>{_esc(title)}</h3>'
        '<div class="donut-wrap">'
        '<svg viewBox="0 0 160 160" role="img" width="160" height="160" '
        f'aria-label="{_esc(title)}">' + "".join(arcs) + "</svg>"
        '<ul class="legend">' + "".join(legend) + "</ul>"
        "</div></section>"
    )


def _meter(label: str, used: float, limit: float | None, *, note: str = "") -> str:
    """Budget meter: spend against a configured limit, with the state spelled out."""
    if not isinstance(limit, int | float) or isinstance(limit, bool) or limit <= 0:
        return (
            '<section class="chart"><h3>' + _esc(label) + "</h3>"
            '<p class="empty">No daily budget configured. '
            "Set one with <code>effgen cost set-budget 1.00</code>.</p></section>"
        )
    ratio = used / limit
    role = "err" if ratio >= 1.0 else "warn" if ratio >= 0.8 else "ok"
    state = "over budget" if ratio >= 1.0 else "near limit" if ratio >= 0.8 else "within budget"
    return (
        f'<section class="chart"><h3>{_esc(label)}</h3>'
        '<div class="bar-row">'
        '<div class="bar-track" role="img" '
        f'aria-label="{ratio * 100:.0f} percent of the daily budget used">'
        f'<div class="bar-fill bar-{role}" style="width:{min(ratio, 1.0) * 100:.2f}%"></div>'
        "</div>"
        f'<div class="bar-value">{_usd(used, 4)} / {_usd(limit, 2)}</div>'
        "</div>"
        f'<p class="meter-note">{ratio * 100:.0f}% used {_badge(state, role)}'
        + (f" {_esc(note)}" if note else "")
        + "</p></section>"
    )
