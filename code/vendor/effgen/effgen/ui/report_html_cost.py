"""The spend report: cost for a period against the configured daily budget.

Shows the total for the window, the spend by provider and model, and the share
each provider took. The budget is configured per day, so it is only compared
like for like once scaled to the window the spend covers; a lifetime total
spans no fixed window and is reported without a budget verdict.
"""

from __future__ import annotations

from typing import Any

from .report_html_components import _bar_chart, _cards, _donut, _meter, _table
from .report_html_format import _DASH, _esc, _int, _mapping, _number, _sequence, _usd


def _cost_body(data: dict[str, Any]) -> tuple[str, str, str]:
    period = str(data.get("period") or "Spend")
    rows_in = [_mapping(r) for r in _sequence(data.get("rows"))]
    total = data.get("total_cost_usd")
    budget = data.get("daily_budget_usd")

    # The budget is configured per day, so it is only a like-for-like comparison
    # once scaled to the window this spend covers. A lifetime total spans no
    # fixed window, so it is reported without a budget verdict.
    period_days = data.get("period_days")
    scaled_budget: float | None = None
    if isinstance(budget, int | float) and not isinstance(budget, bool):
        if isinstance(period_days, int) and not isinstance(period_days, bool) and period_days > 0:
            scaled_budget = float(budget) * period_days

    parts: list[str] = [_cards([
        ("Total spend", _usd(total, 4) if total is not None else _DASH, period),
        ("Requests", _int(data.get("total_requests")), ""),
        ("Providers", str(len({r.get("provider") for r in rows_in})) if rows_in else "0", ""),
        (
            "Daily budget",
            _usd(budget, 2) if budget is not None else "not set",
            "" if budget is not None else "effgen cost set-budget",
        ),
    ])]

    if scaled_budget is not None:
        window = "day" if period_days == 1 else f"{period_days} days"
        note = "" if period_days == 1 else f"({_usd(budget, 2)}/day over {window})"
        parts.append(_meter(
            f"Budget for this period — {window}",
            _number(total),
            scaled_budget,
            note=note,
        ))
    elif budget is not None:
        parts.append(
            '<section class="chart"><h3>Budget</h3>'
            f'<p class="empty">The configured budget is {_usd(budget, 2)} per day. '
            f"This total covers {_esc(period.lower())}, which spans no fixed budget "
            "window, so it is not measured against that limit.</p></section>"
        )
    else:
        parts.append(_meter("Daily budget", _number(total), budget))

    if not rows_in:
        parts.append('<p class="empty">No spend recorded for this period.</p>')
        return f"Spend Report — {period}", period, "".join(parts)

    parts.append("<h2>Spend by provider and model</h2>")
    table_rows = []
    for r in rows_in:
        table_rows.append([
            _esc(r.get("provider")),
            f'<span class="wrap">{_esc(r.get("model"))}</span>',
            f'<span class="num">{_int(r.get("requests"))}</span>',
            f'<span class="num">{_int(r.get("prompt_tokens"))}</span>',
            f'<span class="num">{_int(r.get("completion_tokens"))}</span>',
            f'<span class="num">{_esc(r.get("cost_label") or _usd(r.get("cost_usd")))}</span>',
        ])
    parts.append(_table(
        ["Provider", "Model", "Requests", "Prompt tokens", "Completion tokens", "Cost"],
        table_rows,
    ))

    by_provider: dict[str, float] = {}
    for r in rows_in:
        by_provider[str(r.get("provider"))] = (
            by_provider.get(str(r.get("provider")), 0.0) + _number(r.get("cost_usd"))
        )
    ranked = sorted(by_provider.items(), key=lambda kv: kv[1], reverse=True)
    parts.append("<h2>Charts</h2>")
    parts.append(_bar_chart(
        "Cost by provider",
        [(name, value, _usd(value)) for name, value in ranked],
        role="accent",
    ))
    donut = _donut("Share of spend", ranked)
    if donut:
        parts.append(donut)

    return (
        f"Spend Report — {period}",
        (f"{_usd(total, 4) if total is not None else _DASH} across "
        f"{_int(data.get('total_requests'))} requests"),
        "".join(parts),
    )
