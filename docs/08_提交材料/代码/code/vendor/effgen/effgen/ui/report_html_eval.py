"""The evaluation report: suite pass rate, the exit gate, and every case.

Renders a suite result: the ``--fail-under`` gate and whether it passed, the
headline metrics, the by-difficulty breakdown, and the per-case table with what
each case expected and what the agent answered.
"""

from __future__ import annotations

from typing import Any

from .report_html_components import _badge, _bar_chart, _cards, _table
from .report_html_format import (
    _DASH,
    _esc,
    _int,
    _mapping,
    _pct,
    _secs,
    _sequence,
    _truncate,
    _usd,
)


def _eval_body(data: dict[str, Any]) -> tuple[str, str, str]:
    suite = str(data.get("suite") or "suite")
    total = data.get("total")
    passed = data.get("passed")
    accuracy = data.get("accuracy")
    metadata = data.get("metadata") or {}
    fail_under = data.get("fail_under", metadata.get("fail_under"))

    parts: list[str] = []
    if isinstance(fail_under, int | float) and isinstance(accuracy, int | float):
        gate_passed = accuracy >= fail_under
        parts.append(
            '<div class="verdict">'
            '<div class="verdict-label">Exit gate</div>'
            '<div class="verdict-value">'
            f'{_badge("PASS" if gate_passed else "FAIL", "ok" if gate_passed else "err")}</div>'
            f'<p class="verdict-note">Accuracy {_pct(accuracy)} '
            f'{">=" if gate_passed else "<"} the required {_pct(fail_under, 0)}.</p></div>'
        )

    parts.append(_cards([
        ("Pass rate", _pct(accuracy), f"{_int(passed)} of {_int(total)} cases"),
        ("Avg latency", _secs(data.get("avg_latency"), 4), ""),
        ("Total tokens", _int(data.get("total_tokens")), ""),
        (
            "Total cost",
            _usd(data.get("total_cost_usd")) if data.get("total_cost_usd") is not None else "unpriced",
            str(metadata.get("model") or ""),
        ),
    ]))

    by_difficulty = data.get("by_difficulty") or {}
    if by_difficulty:
        parts.append("<h2>By difficulty</h2>")
        parts.append(_bar_chart(
            "Accuracy by difficulty",
            [
                (
                    f"{name} ({info.get('passed')}/{info.get('total')})",
                    info.get("accuracy"),
                    _pct(info.get("accuracy")),
                )
                for name, info in sorted(by_difficulty.items())
            ],
            role="ok",
            maximum=1.0,
        ))

    results = [_mapping(r) for r in _sequence(data.get("results"))]
    if results:
        parts.append("<h2>Cases</h2>")
        rows = []
        for r in results:
            passed_case = bool(r.get("passed"))
            rows.append([
                f'<span class="wrap">{_esc(_truncate(r.get("query", "")))}</span>',
                f'<span class="wrap">{_esc(_truncate(r.get("expected_output", ""), 120))}</span>',
                f'<span class="wrap">{_esc(_truncate(r.get("agent_output", ""), 200))}</span>',
                _badge("PASS" if passed_case else "FAIL", "ok" if passed_case else "err"),
                f'<span class="num">{_secs(r.get("latency"))}</span>',
                (f'<span class="num">'
                f'{_usd(r.get("cost_usd")) if r.get("cost_usd") is not None else "unpriced"}'
                "</span>"),
                _esc(r.get("difficulty") or _DASH),
            ])
        parts.append(_table(
            ["Query", "Expected", "Got", "Result", "Latency", "Cost", "Difficulty"],
            rows,
        ))

    subtitle = (
        f"{suite} · {_int(passed)}/{_int(total)} passed"
        + (f" · scoring {metadata['scoring']}" if metadata.get("scoring") else "")
    )
    return f"Evaluation Report — {suite}", subtitle, "".join(parts)
