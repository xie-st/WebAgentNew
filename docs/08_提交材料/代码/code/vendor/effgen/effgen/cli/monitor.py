"""Terminal mission-control view over effGen's existing telemetry.

``effgen top`` (alias ``effgen monitor``) assembles one screen from the data
effGen already records:

- **Activity** — completed runs from the durable run history
  (:func:`effgen.observability.run_log.read_runs`), which is written by every
  process on the host.
- **Traffic** and **Per-model** — the running server's own counters, scraped
  from its ``/dashboard/data.json`` endpoint. These cover a single server
  process since it started.
- **Spend** — the local SQLite cost ledger: a 24-hour total against the
  configured daily budget, plus a trailing-hour burn rate.
- **GPU** — physical device memory and utilisation across all processes, the
  same view ``effgen models status`` reports.

Those sources measure different things over different windows, so every panel
states its own scope and the numbers are never blended into a single figure.

Only the traffic and per-model panels need a server. When none is reachable
they are shown as unavailable, naming the URL that was tried, so an empty
screen is never mistaken for zero traffic.

On an interactive terminal the view refreshes in place until ``q`` or Ctrl-C.
When stdout is not a terminal, or ``--json``/``--once`` is passed, or animation
is disabled (``--no-animation``, ``NO_COLOR``, ``EFFGEN_NO_ANIM=1``), it prints
one snapshot and exits.

This module is the import path; the work is in four siblings.
:mod:`effgen.cli.monitor_collect` reads the sources into one snapshot document,
:mod:`effgen.cli.monitor_format` holds the scope labels and the value
formatters both rendering paths share, :mod:`effgen.cli.monitor_panels` prints
the static labelled tables, and :mod:`effgen.cli.monitor_live` builds the
full-screen layout and drives the refresh loop.

Imports marked ``noqa: F401`` below are no longer read here: they are kept so
every name this module has ever exposed still resolves against it. Do not run
``ruff --fix`` over that block; it would read them as unused and drop them.
"""

from __future__ import annotations

import http.client  # noqa: F401 - re-exported
import json
import os  # noqa: F401 - re-exported
import select  # noqa: F401 - re-exported
import socket  # noqa: F401 - re-exported
import sys
import time  # noqa: F401 - re-exported
import urllib.error  # noqa: F401 - re-exported
import urllib.request  # noqa: F401 - re-exported
from datetime import datetime, timezone  # noqa: F401 - re-exported
from typing import Any

from effgen.cli.monitor_collect import (  # noqa: F401 - re-exported
    _HTTP_TIMEOUT_S,
    BURN_WINDOW_S,
    DEFAULT_ACTIVITY_LIMIT,
    DEFAULT_SERVER_URL,
    _collect_activity,
    _collect_gpu,
    _collect_server,
    _collect_spend,
    _effgen_version,
    _fetch_json,
    collect_snapshot,
    resolve_api_key,
    resolve_server_url,
)
from effgen.cli.monitor_format import (  # noqa: F401 - re-exported
    _STATUS_STYLES,
    SCOPE_ACTIVITY,
    SCOPE_GPU,
    SCOPE_HTTP_STATUS,
    SCOPE_SPEND,
    SCOPE_TRAFFIC,
    _fmt_clock,
    _fmt_pct,
    _fmt_seconds,
    _fmt_usd,
    _health_style,
    _preview,
    _status_style,
)
from effgen.cli.monitor_live import (  # noqa: F401 - re-exported
    GPU_SAMPLE_INTERVAL_S,
    _build_live_view,
    _QuitKeyReader,
    run_live,
)
from effgen.cli.monitor_panels import (  # noqa: F401 - re-exported
    _emit,
    _render_activity,
    _render_activity_table,
    _render_by_model_table,
    _render_gpu_table,
    _render_spend_table,
    _render_traffic_table,
    _unavailable,
    render_snapshot,
)
from effgen.ui.render import json_ensure_ascii
from effgen.ui.tables import console_is_interactive, render_table  # noqa: F401 - re-exported
from effgen.ui.theme import color_enabled, get_console

DEFAULT_INTERVAL = 2.0
MIN_INTERVAL = 0.5
MAX_INTERVAL = 300.0


# ---------------------------------------------------------------------------
# Command entry point
# ---------------------------------------------------------------------------


def run_monitor_command(args: Any) -> int:
    """Entry point for ``effgen top`` / ``effgen monitor``."""
    from effgen.cli import progress as _progress

    interval = float(getattr(args, "interval", DEFAULT_INTERVAL) or DEFAULT_INTERVAL)
    if not MIN_INTERVAL <= interval <= MAX_INTERVAL:
        print(
            f"--interval must be between {MIN_INTERVAL:g} and {MAX_INTERVAL:g} seconds "
            f"(got {interval:g}).",
            file=sys.stderr,
        )
        return 1
    limit = max(1, int(getattr(args, "limit", DEFAULT_ACTIVITY_LIMIT) or DEFAULT_ACTIVITY_LIMIT))
    url = getattr(args, "url", None)
    port = getattr(args, "port", None)
    api_key = getattr(args, "api_key", None)
    json_mode = bool(getattr(args, "output_json", False))
    once = bool(getattr(args, "once", False))

    if json_mode:
        snapshot = collect_snapshot(
            url=url, port=port, api_key=api_key, limit=limit, interval=None
        )
        print(json.dumps(snapshot, indent=2, ensure_ascii=json_ensure_ascii()))
        return 0

    console = get_console(theme_name=getattr(args, "theme", None))
    live_capable = _progress.animation_enabled(
        quiet=getattr(args, "quiet", False),
        no_animation=getattr(args, "no_animation", False),
    )
    if once or not live_capable or not color_enabled() or not console_is_interactive(console):
        snapshot = collect_snapshot(
            url=url, port=port, api_key=api_key, limit=limit, interval=None
        )
        render_snapshot(snapshot, console=console)
        return 0

    return run_live(
        console=console,
        url=url,
        port=port,
        api_key=api_key,
        limit=limit,
        interval=interval,
        count=getattr(args, "count", None),
    )
