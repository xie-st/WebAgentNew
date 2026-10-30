"""Scope labels and the value formatters the monitor panels share.

Both rendering paths — the static tables :mod:`effgen.cli.monitor_panels`
prints and the full-screen layout :mod:`effgen.cli.monitor_live` refreshes —
format their numbers through this module, so a duration, a dollar amount or a
percentage reads the same however it was rendered. ``effgen.cli.monitor`` stays
the import path for every name here.

The ``SCOPE_*`` strings are the other half of that agreement. Each panel carries
one so a reader can tell which window and which process a number covers, and
they are separate constants because the populations behind them differ: the
generation counters and the all-routes HTTP histogram are not expected to
reconcile.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Scope labels. Each panel carries one so a reader can tell which window and
# which process a number covers.
SCOPE_ACTIVITY = "completed runs, local run history (all processes)"
SCOPE_TRAFFIC = "this server process since start"
# The request/error counters cover generation calls; the status histogram comes
# from the HTTP request counter, which also records health checks, metrics
# scrapes and dashboard reads. The two populations differ, so they are labelled
# separately rather than left to look like they should reconcile.
SCOPE_HTTP_STATUS = "all HTTP routes, this server process since start"
SCOPE_SPEND = "local cost ledger, last 24 hours"
SCOPE_GPU = "physical devices, all processes"


def _preview(text: Any, width: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= width else value[: width - 1] + "…"


def _fmt_usd(value: Any, *, places: int = 6) -> str:
    """Format a dollar amount; ``None`` renders as an em dash, never ``$0``."""
    if value is None:
        return "—"
    try:
        return f"${float(value):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_seconds(value: Any) -> str:
    if value is None:
        return "—"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_pct(value: Any, *, places: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_clock(ts: Any) -> str:
    """Render a run timestamp as a local wall-clock time."""
    if not ts:
        return "—"
    try:
        return datetime.fromisoformat(str(ts)).strftime("%H:%M:%S")
    except ValueError:
        return str(ts)[:8]


_STATUS_STYLES = {
    "ok": "effgen.success",
    "error": "effgen.error",
    "failed": "effgen.error",
}


def _status_style(status: Any) -> str:
    return _STATUS_STYLES.get(str(status or "").lower(), "effgen.muted")


def _health_style(*, error_rate: float | None, reachable: bool) -> str:
    if not reachable:
        return "effgen.muted"
    if error_rate is None:
        return "effgen.success"
    if error_rate >= 0.1:
        return "effgen.error"
    if error_rate > 0:
        return "effgen.warning"
    return "effgen.success"
