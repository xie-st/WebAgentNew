"""The full-screen view and the loop that refreshes it in place.

:func:`_build_live_view` assembles one snapshot into a Rich layout — a header
row plus the five panels, each sized so a narrow terminal ellipsizes rather than
wraps — and :func:`run_live` redraws it on the configured interval until ``q``,
Escape, Ctrl-C, or a requested refresh count. The static tables a pipe gets are
:mod:`effgen.cli.monitor_panels`; both render the same five panels from the same
snapshot. ``effgen.cli.monitor`` stays the import path for every name here.

:class:`_QuitKeyReader` puts the terminal in cbreak mode so a single keypress
ends the loop, and restores the terminal settings on the way out. Where raw
input is unavailable it falls back to sleeping, and Ctrl-C remains the way to
quit.
"""

from __future__ import annotations

import select
import sys
import time
from typing import Any

from effgen.cli.monitor_collect import _collect_gpu, collect_snapshot
from effgen.cli.monitor_format import (
    SCOPE_ACTIVITY,
    SCOPE_GPU,
    SCOPE_SPEND,
    SCOPE_TRAFFIC,
    _fmt_clock,
    _fmt_pct,
    _fmt_seconds,
    _fmt_usd,
    _health_style,
    _status_style,
)

# The GPU sample is the slowest source, so it runs on a slower cadence than the
# rest of the snapshot and the previous reading is reused in between.
GPU_SAMPLE_INTERVAL_S = 5.0


def _model_label(row: dict[str, Any], repeated: set[str]) -> str:
    """The model name, qualified by provider only when the name repeats."""
    name = str(row.get("model") or "—")
    provider = str(row.get("provider") or "").strip()
    return f"{name} ({provider})" if provider and name in repeated else name


def _build_live_view(snapshot: dict[str, Any]) -> Any:
    """Assemble the full-screen layout for one snapshot."""
    from rich.console import Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def _table(
        columns: list[str],
        rows: list[list[Any]],
        *,
        justify: list[str] | None = None,
        widths: list[int | None] | None = None,
        ratios: list[int | None] | None = None,
        expand: bool = True,
    ) -> Any:
        """A borderless table sized for a panel.

        ``widths`` pins the narrow, fixed-shape columns (a clock, a status, a
        cost) so they survive a narrow terminal; the remaining columns share the
        leftover width by ``ratios`` and ellipsize rather than wrap.
        """
        table = Table(box=None, expand=expand, pad_edge=False, show_edge=False, padding=(0, 1, 0, 0))
        for index, column in enumerate(columns):
            table.add_column(
                column,
                justify=(justify[index] if justify else "left"),
                style="effgen.value",
                header_style="effgen.muted",
                no_wrap=True,
                overflow="ellipsis",
                width=(widths[index] if widths else None),
                ratio=(ratios[index] if ratios else None),
            )
        for row in rows:
            table.add_row(*row)
        return table

    header = snapshot.get("header") or {}
    traffic = snapshot.get("traffic") or {}
    reachable = bool(header.get("server_reachable"))
    health = _health_style(
        error_rate=traffic.get("error_rate") if reachable else None, reachable=reachable
    )

    identity = Text()
    identity.append(f"effGen {header.get('version')} ", style="effgen.title")
    identity.append(f"· {header.get('hostname')} · ", style="effgen.muted")
    identity.append(
        f"server {header.get('server_url')} " + ("up" if reachable else "unreachable"),
        style=health,
    )
    interval = header.get("interval_s")
    clock = Text(
        f"{snapshot.get('ts')}"
        + (f" · every {interval:g}s" if interval else "")
        + " · q quits",
        style="effgen.muted",
    )
    title = Table.grid(expand=True)
    title.add_column(justify="left", overflow="ellipsis", no_wrap=True)
    title.add_column(justify="right", overflow="ellipsis", no_wrap=True)
    title.add_row(identity, clock)

    # Activity
    activity = snapshot.get("activity") or {}
    if activity.get("available"):
        runs = activity.get("runs") or []
        activity_body: Any = (
            _table(
                ["Time", "Status", "Model", "Dur", "Cost", "Task"],
                [
                    [
                        _fmt_clock(run.get("ts")),
                        Text(str(run.get("status") or "—"), style=_status_style(run.get("status"))),
                        str(run.get("model") or "—"),
                        _fmt_seconds(run.get("duration_s")),
                        _fmt_usd(run.get("cost_usd")),
                        run.get("task") or "—",
                    ]
                    for run in runs
                ],
                justify=["left", "left", "left", "right", "right", "left"],
                widths=[8, 6, None, 7, 10, None],
                ratios=[None, None, 2, None, None, 3],
            )
            if runs
            else Text("no runs recorded yet", style="effgen.muted")
        )
    else:
        activity_body = Text(str(activity.get("unavailable_reason") or ""), style="effgen.warning")

    # Traffic
    if traffic.get("available"):
        status_counts = traffic.get("by_status") or {}
        traffic_body: Any = Group(
            _table(
                ["", ""],
                [
                    ["Generations", str(traffic.get("requests", 0))],
                    [
                        "Errors",
                        Text(
                            f"{traffic.get('errors', 0)} "
                            f"({_fmt_pct(100 * (traffic.get('error_rate') or 0))})",
                            style=health,
                        ),
                    ],
                    [
                        "p50 / p95 / p99",
                        " / ".join(
                            _fmt_seconds(traffic.get(key))
                            for key in ("p50_latency_s", "p95_latency_s", "p99_latency_s")
                        ),
                    ],
                    ["Tokens", str(traffic.get("tokens", 0))],
                    # Every route, not just generations — so this exceeds the
                    # generation count above rather than matching it.
                    [
                        "HTTP all routes",
                        ", ".join(f"{c}×{n}" for c, n in sorted(status_counts.items())) or "—",
                    ],
                ],
                widths=[16, None],
            )
        )
    else:
        traffic_body = Text(str(traffic.get("unavailable_reason") or ""), style="effgen.warning")

    # Per-model
    by_model = snapshot.get("by_model") or {}
    model_rows = by_model.get("rows") or []
    if not by_model.get("available"):
        model_body: Any = Text("server unavailable — see Traffic", style="effgen.muted")
    elif model_rows:
        # The live layout has no room for a Provider column, so a model name
        # served by two providers would show as two rows with the same label and
        # different numbers. Qualify only the names that repeat.
        repeated = {
            name
            for name in {str(row.get("model") or "—") for row in model_rows}
            if sum(1 for row in model_rows if str(row.get("model") or "—") == name) > 1
        }
        model_body = _table(
            ["Model", "Calls", "Err", "p95", "Cost"],
            [
                [
                    _model_label(row, repeated),
                    str(row.get("calls") or 0),
                    str(row.get("errors") or 0),
                    _fmt_seconds(row.get("p95_latency_s")),
                    _fmt_usd(row.get("cost_usd")),
                ]
                for row in model_rows
            ],
            justify=["left", "right", "right", "right", "right"],
            widths=[None, 5, 4, 7, 10],
        )
    else:
        model_body = Text("no model traffic yet", style="effgen.muted")

    # Spend
    spend = snapshot.get("spend") or {}
    if spend.get("available"):
        budget = spend.get("daily_budget_usd")
        used_pct = spend.get("budget_used_pct")
        spend_rows = [
            ["Total (24h)", _fmt_usd(spend.get("total_cost_usd"))],
            ["Burn", f"{_fmt_usd(spend.get('burn_rate_usd_per_hour'))}/h"],
        ]
        if budget:
            bar_width = 16
            filled = max(0, min(bar_width, int(bar_width * (used_pct or 0) / 100.0)))
            bar_style = (
                "effgen.error"
                if (used_pct or 0) >= 100
                else "effgen.warning"
                if (used_pct or 0) >= 80
                else "effgen.success"
            )
            bar = Text("█" * filled, style=bar_style)
            bar.append("░" * (bar_width - filled), style="effgen.muted")
            bar.append(f" {_fmt_pct(used_pct)} of {_fmt_usd(budget, places=2)}")
            spend_rows.append(["Budget", bar])
        else:
            spend_rows.append(["Budget", Text("not configured", style="effgen.muted")])
        unpriced = spend.get("unpriced_models") or []
        if unpriced:
            spend_rows.append(
                ["Unpriced", Text(f"{len(unpriced)} model(s) not in total", style="effgen.muted")]
            )
        spend_body: Any = _table(["", ""], spend_rows, widths=[11, None])
    else:
        spend_body = Text(str(spend.get("unavailable_reason") or ""), style="effgen.warning")

    # GPU
    gpu = snapshot.get("gpu") or {}
    if gpu.get("available"):
        gpu_body: Any = _table(
            ["GPU", "Memory", "Mem%", "Util"],
            [
                [
                    str(device.get("index")),
                    f"{device.get('used_gb'):.1f}/{device.get('total_gb'):.0f}G",
                    _fmt_pct(device.get("memory_used_pct"), places=0),
                    _fmt_pct(device.get("utilization_pct"), places=0),
                ]
                for device in gpu.get("devices") or []
            ],
            justify=["right", "right", "right", "right"],
            widths=[4, 11, 6, 6],
            expand=False,
        )
    else:
        gpu_body = Text(str(gpu.get("unavailable_reason") or ""), style="effgen.muted")

    def _panel(body: Any, heading: str, scope: str) -> Any:
        return Panel(
            body,
            title=heading,
            subtitle=scope,
            subtitle_align="right",
            border_style="effgen.muted",
            padding=(0, 1),
        )

    layout = Layout()
    layout.split_column(
        Layout(Panel(title, border_style="effgen.muted", padding=(0, 1)), name="head", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
    layout["left"].split_column(
        Layout(
            _panel(activity_body, "Activity", activity.get("scope", SCOPE_ACTIVITY)),
            name="activity",
            ratio=3,
        ),
        Layout(
            _panel(model_body, "Per-model", by_model.get("scope", SCOPE_TRAFFIC)),
            name="models",
            ratio=2,
        ),
    )
    # The GPU panel is sized for the device count so a multi-GPU host is not
    # clipped to its first few cards.
    gpu_rows = len(gpu.get("devices") or [])
    layout["right"].split_column(
        Layout(
            _panel(traffic_body, "Traffic", traffic.get("scope", SCOPE_TRAFFIC)),
            name="traffic",
            size=9,
        ),
        Layout(
            _panel(gpu_body, "GPU", gpu.get("scope", SCOPE_GPU)),
            name="gpu",
            size=min(gpu_rows + 4, 14) if gpu_rows else 4,
        ),
        Layout(_panel(spend_body, "Spend", spend.get("scope", SCOPE_SPEND)), name="spend"),
    )
    return layout


class _QuitKeyReader:
    """Read single keypresses without blocking, restoring the terminal on exit.

    Falls back to a no-op (Ctrl-C remains the way to quit) on a platform or a
    stream where raw terminal input is unavailable.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: Any = None

    def __enter__(self) -> _QuitKeyReader:
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                return self
            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:  # noqa: BLE001 - no raw input available; Ctrl-C still quits
            self._fd = None
            self._saved = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None and self._saved is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001 - the terminal is being torn down anyway
                pass

    def quit_requested(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for a quit key (``q``, ``Q``, or Escape)."""
        if self._fd is None:
            time.sleep(max(0.0, timeout))
            return False
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                ready, _, _ = select.select([sys.stdin], [], [], remaining)
            except (OSError, ValueError):
                time.sleep(max(0.0, remaining))
                return False
            if not ready:
                return False
            char = sys.stdin.read(1)
            if char in ("q", "Q", "\x1b", "\x03"):
                return True


def run_live(
    *,
    console: Any,
    url: str | None,
    port: int | None,
    api_key: str | None,
    limit: int,
    interval: float,
    count: int | None = None,
) -> int:
    """Refresh the full-screen view in place until a quit key, Ctrl-C, or *count* refreshes.

    Args:
        console: The console the view is drawn on.
        url: Base URL of the server to read, when one is running.
        port: Port on localhost to read, used when *url* is absent.
        api_key: Credential for the server's authenticated routes.
        limit: Most activity rows to show.
        interval: Seconds between refreshes.
        count: Stop after this many refreshes instead of running until quit.

    Returns:
        The process exit code.
    """
    from rich.live import Live

    gpu_panel: dict[str, Any] | None = None
    gpu_sampled_at = 0.0
    refreshes = 0
    try:
        with _QuitKeyReader() as keys, Live(
            console=console, screen=True, auto_refresh=False, transient=False
        ) as live:
            while True:
                now = time.monotonic()
                if gpu_panel is None or (now - gpu_sampled_at) >= GPU_SAMPLE_INTERVAL_S:
                    gpu_panel = _collect_gpu()
                    gpu_sampled_at = now
                snapshot = collect_snapshot(
                    url=url,
                    port=port,
                    api_key=api_key,
                    limit=limit,
                    interval=interval,
                    gpu=gpu_panel,
                )
                live.update(_build_live_view(snapshot), refresh=True)
                refreshes += 1
                if count is not None and refreshes >= count:
                    return 0
                if keys.quit_requested(interval):
                    return 0
    except KeyboardInterrupt:
        return 0
