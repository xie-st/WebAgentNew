"""The model-comparison report: per-model accuracy, latency and cost.

Renders a bake-off document: the recommended model per suite and the reason it
was picked, the per-model table, the charts, and what each model actually
answered.

A result document may have been written by a different build or edited by hand,
so every collection is normalized before it is read.
"""

from __future__ import annotations

from typing import Any

from .report_html_components import _badge, _bar_chart, _cards, _table
from .report_html_format import (
    _DASH,
    ReportError,
    _esc,
    _int,
    _mapping,
    _pct,
    _secs,
    _sequence,
    _truncate,
    _usd,
)


def _comparison_body(data: dict[str, Any]) -> tuple[str, str, str]:
    # A result document may have been written by a different build or edited by
    # hand, so every collection is normalized before it is read: a row that is
    # not a mapping renders as an empty row rather than crashing the report.
    scores = [_mapping(s) for s in _sequence(data.get("scores"))]
    if not scores:
        raise ReportError(
            "The comparison result carries no scores to render. Pass the JSON "
            "`effgen compare --json` emitted, with its scores intact."
        )
    suites = sorted({str(s.get("suite", "")) for s in scores})
    models = sorted({str(s.get("model", "")) for s in scores})
    optimize = str(data.get("optimize", "accuracy"))
    recommendations = _mapping(data.get("recommendations"))
    rationales = _mapping(data.get("recommendation_rationale"))

    parts: list[str] = []

    # Hero verdict — the recommended model per suite plus why it won.
    for suite in suites:
        model = recommendations.get(suite)
        if not model:
            continue
        note = rationales.get(suite) or f"Optimized for {optimize}."
        parts.append(
            '<div class="verdict">'
            f'<div class="verdict-label">Recommended for {_esc(suite)} '
            f"(optimized for {_esc(optimize)})</div>"
            f'<div class="verdict-value">{_esc(model)}</div>'
            f'<p class="verdict-note">{_esc(note)}</p></div>'
        )

    num_cases = data.get("num_cases")
    parts.append(_cards([
        ("Models", str(len(models)), ""),
        ("Suites", ", ".join(suites) if suites else _DASH, ""),
        ("Cases per model", _int(num_cases) if num_cases is not None else _DASH, ""),
        ("Scoring", str(data.get("scoring") or _DASH), ""),
    ]))

    # Under llm_judge scoring, say what graded the answers so a reader knows
    # whether the scores came from a model with a stake in them.
    self_judged = data.get("self_judged")
    if self_judged is not None:
        note = (
            "Each model graded its own answers."
            if self_judged
            else f"Graded by {data.get('judge_model') or 'a named judge'}, "
                 "which is not one of the compared models."
        )
        parts.append(f'<p class="verdict-note">{_esc(note)}</p>')

    parts.append("<h2>Results</h2>")
    rows = []
    for s in scores:
        error = s.get("error")
        if error:
            status = _badge("ERROR", "err")
            cells = [_DASH, _DASH, _DASH, _DASH]
        else:
            status = _badge("ok", "ok")
            cost = s.get("avg_cost_usd")
            cells = [
                _pct(s.get("accuracy")),
                _secs(s.get("avg_latency")),
                _usd(cost) if cost is not None else "unpriced",
                _int(s.get("total_tokens")),
            ]
        rows.append([
            f'<span class="wrap">{_esc(s.get("model"))}</span>',
            _esc(s.get("suite")),
            status,
            *[f'<span class="num">{c}</span>' for c in cells],
        ])
    parts.append(_table(
        ["Model", "Suite", "Status", "Accuracy", "Avg latency", "Avg cost/run", "Tokens"],
        rows,
    ))

    ok = [s for s in scores if not s.get("error")]
    if ok:
        parts.append("<h2>Charts</h2>")
        parts.append(_bar_chart(
            "Accuracy",
            [(str(s.get("model")), s.get("accuracy"), _pct(s.get("accuracy"))) for s in ok],
            role="ok",
            maximum=1.0,
        ))
        parts.append(_bar_chart(
            "Average cost per run (unpriced models shown as a hatched bar)",
            [
                (
                    str(s.get("model")),
                    s.get("avg_cost_usd"),
                    _usd(s.get("avg_cost_usd")) if s.get("avg_cost_usd") is not None else "unpriced",
                )
                for s in ok
            ],
            role="accent",
        ))
        parts.append(_bar_chart(
            "Average latency",
            [(str(s.get("model")), s.get("avg_latency"), _secs(s.get("avg_latency"))) for s in ok],
            role="accent2",
        ))

    # What each model actually answered, so a shared bake-off shows its work
    # rather than only the percentages above.
    answered = [s for s in scores if _sequence(s.get("responses"))]
    if answered:
        parts.append("<h2>Answers</h2>")
        for s in answered:
            parts.append(f'<h3 class="wrap">{_esc(s.get("model"))}</h3>')
            rows = []
            for raw in _sequence(s.get("responses")):
                r = _mapping(raw)
                if r.get("error"):
                    outcome = _badge("ERROR", "err")
                elif r.get("passed"):
                    outcome = _badge("pass", "ok")
                else:
                    outcome = _badge("fail", "warn")
                rows.append([
                    f'<span class="wrap">{_esc(_truncate(str(r.get("query", "")), 160))}</span>',
                    f'<span class="wrap">{_esc(_truncate(str(r.get("output", "")), 400))}</span>',
                    outcome,
                ])
            parts.append(_table(["Case", "Answer", "Result"], rows))

    subtitle = (
        f"{len(models)} models on {', '.join(suites)}"
        + (f" · {num_cases} cases" if num_cases is not None else "")
        + f" · scoring {data.get('scoring') or 'contains'}"
    )
    return "Model Comparison", subtitle, "".join(parts)
