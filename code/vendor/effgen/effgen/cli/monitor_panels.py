"""The static, labelled tables a pipe or ``--once`` gets.

:func:`render_snapshot` prints one collected snapshot as a sequence of titled
tables through the shared table renderer, so an interactive terminal gets Rich
tables and a pipe gets aligned plain text with no color or box drawing. Each
table names its panel and the scope that panel measures.

This is the path taken when there is no terminal, when ``--once`` or
``--no-animation`` is passed, or when color is off. The full-screen layout that
refreshes in place is :mod:`effgen.cli.monitor_live`; both render the same five
panels from the same snapshot, and neither reads a source itself.
``effgen.cli.monitor`` stays the import path for every name here.
"""

from __future__ import annotations

from typing import Any

from effgen.cli.monitor_format import (
    SCOPE_ACTIVITY,
    SCOPE_GPU,
    SCOPE_SPEND,
    SCOPE_TRAFFIC,
    _fmt_clock,
    _fmt_pct,
    _fmt_seconds,
    _fmt_usd,
)
from effgen.ui.tables import console_is_interactive, render_table


def render_snapshot(snapshot: dict[str, Any], *, console: Any = None) -> None:
    """Print one snapshot as a sequence of labelled tables.

    Uses the shared table renderer, so an interactive terminal gets Rich tables
    and a pipe gets aligned plain text with no color or box drawing.
    """
    header = snapshot.get("header") or {}
    reach = (
        f"server {header.get('server_url')} reachable"
        if header.get("server_reachable")
        else f"server {header.get('server_url')} unavailable"
    )
    lines = [
        f"effGen {header.get('version')} — {header.get('hostname')} — {snapshot.get('ts')}",
        reach,
    ]
    for line in lines:
        _emit(console, line)
    _emit(console, "")

    _render_activity_table(snapshot.get("activity") or {}, console)
    _render_traffic_table(snapshot.get("traffic") or {}, console)
    _render_by_model_table(snapshot.get("by_model") or {}, console)
    _render_spend_table(snapshot.get("spend") or {}, console)
    _render_gpu_table(snapshot.get("gpu") or {}, console)


def _emit(console: Any, text: str) -> None:
    if console_is_interactive(console):
        console.print(text, highlight=False)
    else:
        print(text)


def _unavailable(console: Any, title: str, panel: dict[str, Any]) -> None:
    reason = panel.get("unavailable_reason") or "no data source"
    _emit(console, f"{title} — unavailable: {reason}")
    _emit(console, "")


def _render_activity(panel: dict[str, Any]) -> tuple[list[str], list[list[str]], str]:
    columns = ["Time", "Status", "Model", "Duration", "Cost", "Task"]
    rows = [
        [
            _fmt_clock(run.get("ts")),
            str(run.get("status") or "—"),
            str(run.get("model") or "—"),
            _fmt_seconds(run.get("duration_s")),
            _fmt_usd(run.get("cost_usd")),
            run.get("task") or "—",
        ]
        for run in panel.get("runs") or []
    ]
    return columns, rows, f"Activity — {panel.get('scope', SCOPE_ACTIVITY)}"


def _render_activity_table(panel: dict[str, Any], console: Any) -> None:
    if not panel.get("available"):
        _unavailable(console, "Activity", panel)
        return
    columns, rows, title = _render_activity(panel)
    if not rows:
        _emit(console, f"{title}\n  no runs recorded yet")
        _emit(console, "")
        return
    render_table(columns=columns, rows=rows, console=console, title=title)
    _emit(console, "")


def _render_traffic_table(panel: dict[str, Any], console: Any) -> None:
    title = f"Traffic — {panel.get('scope', SCOPE_TRAFFIC)}"
    if not panel.get("available"):
        _unavailable(console, "Traffic", panel)
        return
    status_counts = panel.get("by_status") or {}
    status_text = ", ".join(f"{code}×{count}" for code, count in sorted(status_counts.items()))
    rows = [
        ["Generation requests", str(panel.get("requests", 0))],
        ["Errors", f"{panel.get('errors', 0)} ({_fmt_pct(100 * (panel.get('error_rate') or 0))})"],
        ["Latency p50 / p95 / p99", " / ".join(
            _fmt_seconds(panel.get(key))
            for key in ("p50_latency_s", "p95_latency_s", "p99_latency_s")
        )],
        ["Tokens", str(panel.get("tokens", 0))],
        # Counted over every route, so this total is expected to exceed the
        # generation-request count above rather than match it.
        ["HTTP responses (all routes)", status_text or "—"],
    ]
    render_table(columns=["Metric", "Value"], rows=rows, console=console, title=title)
    _emit(console, "")


def _render_by_model_table(panel: dict[str, Any], console: Any) -> None:
    if not panel.get("available"):
        _unavailable(console, "Per-model", panel)
        return
    rows = [
        [
            str(row.get("model") or "—"),
            str(row.get("provider") or "—"),
            str(row.get("calls") or 0),
            str(row.get("errors") or 0),
            _fmt_pct(100 * float(row.get("error_rate") or 0.0)),
            _fmt_seconds(row.get("p95_latency_s")),
            _fmt_usd(row.get("cost_usd")),
        ]
        for row in panel.get("rows") or []
    ]
    title = f"Per-model — {panel.get('scope', SCOPE_TRAFFIC)}"
    if not rows:
        _emit(console, f"{title}\n  no model traffic yet")
        _emit(console, "")
        return
    render_table(
        columns=["Model", "Provider", "Calls", "Errors", "Error rate", "p95", "Cost"],
        rows=rows,
        console=console,
        title=title,
    )
    _emit(console, "")


def _render_spend_table(panel: dict[str, Any], console: Any) -> None:
    title = f"Spend — {panel.get('scope', SCOPE_SPEND)}"
    if not panel.get("available"):
        _unavailable(console, "Spend", panel)
        return
    budget = panel.get("daily_budget_usd")
    rows = [
        ["Total", _fmt_usd(panel.get("total_cost_usd"))],
        ["Requests", str(panel.get("requests", 0))],
        [
            "Daily budget",
            f"{_fmt_usd(budget, places=2)} ({_fmt_pct(panel.get('budget_used_pct'))} used)"
            if budget
            else "not configured (effgen cost set-budget AMOUNT)",
        ],
        ["Burn rate", f"{_fmt_usd(panel.get('burn_rate_usd_per_hour'))}/h (last hour)"],
    ]
    unpriced = panel.get("unpriced_models") or []
    if unpriced:
        rows.append(["Unpriced", f"{len(unpriced)} model(s) excluded from the total"])
    render_table(columns=["Metric", "Value"], rows=rows, console=console, title=title)
    _emit(console, "")


def _render_gpu_table(panel: dict[str, Any], console: Any) -> None:
    title = f"GPU — {panel.get('scope', SCOPE_GPU)}"
    if not panel.get("available"):
        _unavailable(console, "GPU", panel)
        return
    rows = [
        [
            str(device.get("index")),
            str(device.get("name") or "—"),
            f"{device.get('used_gb')} / {device.get('total_gb')} GB",
            _fmt_pct(device.get("memory_used_pct"), places=0),
            _fmt_pct(device.get("utilization_pct"), places=0),
        ]
        for device in panel.get("devices") or []
    ]
    render_table(
        columns=["GPU", "Name", "Memory", "Mem %", "Util"],
        rows=rows,
        console=console,
        title=title,
    )
    _emit(console, "")
