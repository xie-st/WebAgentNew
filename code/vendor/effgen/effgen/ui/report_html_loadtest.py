"""The load-test report: latency percentiles, throughput and the error breakdown.

Shows the outcome and its error rate, the headline percentiles, the settings the
run used, and what the failed requests were classified as.
"""

from __future__ import annotations

from typing import Any

from .report_html_components import _badge, _bar_chart, _cards, _table
from .report_html_format import _DASH, _esc, _int, _mapping, _ms, _pct, _rps, _secs


def _loadtest_body(data: dict[str, Any]) -> tuple[str, str, str]:
    lat = _mapping(data.get("latency"))
    error_rate = data.get("error_rate")
    scenario = str(data.get("scenario") or "load test")

    role = "ok"
    state = "no failed requests"
    if isinstance(error_rate, int | float):
        if error_rate >= 0.05:
            role, state = "err", "elevated error rate"
        elif error_rate > 0:
            role, state = "warn", "some requests failed"

    parts: list[str] = [
        ('<div class="verdict">'
        '<div class="verdict-label">Outcome</div>'
        f'<div class="verdict-value">{_pct(error_rate, 2)} errors '
        f"{_badge(state, role)}</div>"
        f'<p class="verdict-note">{_int(data.get("successful_requests"))} of '
        f'{_int(data.get("total_requests"))} requests succeeded at '
        f'{data.get("concurrency")} concurrent users.</p></div>'),
        _cards([
            ("Throughput", _rps(data.get("throughput_rps")), ""),
            ("p95 latency", _ms(lat.get("p95")), f"p50 {_ms(lat.get('p50'))}"),
            ("p99 latency", _ms(lat.get("p99")), f"max {_ms(lat.get('max'))}"),
            (
                "Duration",
                _secs(data.get("duration_s"), 1),
                f"requested {data.get('requested_duration_s')}s",
            ),
        ]),
        "<h2>Latency distribution</h2>",
        _bar_chart(
            "Percentiles",
            [
                (name.upper(), lat.get(key), _ms(lat.get(key)))
                for name, key in (
                    ("min", "min"), ("p50", "p50"), ("mean", "mean"),
                    ("p95", "p95"), ("p99", "p99"), ("max", "max"),
                )
            ],
            role="accent2",
        ),
    ]

    parts.append("<h2>Run</h2>")
    target = data.get("model") or data.get("provider")
    parts.append(_table(
        ["Setting", "Value"],
        [
            ["Scenario", _esc(scenario)],
            ["Concurrency", f'<span class="num">{_int(data.get("concurrency"))}</span>'],
            ["Target", _esc(target) if target else "local mock"],
            ["Provider", _esc(data.get("provider")) if data.get("provider") else _DASH],
            [
                "Drain after window",
                f'<span class="num">{_secs(data.get("drain_s"), 1)}</span>',
            ],
        ],
    ))

    breakdown = data.get("error_breakdown") or {}
    parts.append("<h2>Errors</h2>")
    if breakdown:
        parts.append(_table(
            ["Error category", "Count"],
            [
                [_esc(k), f'<span class="num">{_int(v)}</span>']
                for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])
            ],
        ))
    else:
        parts.append('<p class="empty">No failed requests to break down.</p>')

    return (
        f"Load Test Report — {scenario}",
        (f"{_int(data.get('total_requests'))} requests · "
        f"{data.get('concurrency')} concurrent · {_pct(error_rate, 2)} errors"),
        "".join(parts),
    )
