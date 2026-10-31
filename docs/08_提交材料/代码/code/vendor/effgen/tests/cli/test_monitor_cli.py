"""Tests for the `effgen top` / `effgen monitor` live terminal view.

The snapshot itself is assembled from local sources (run history, cost ledger,
GPU layer) plus an optional HTTP read of a server's dashboard aggregation. The
server-backed panels are exercised against a real ``create_app`` served over a
loopback socket, not a stub.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import threading
from contextlib import closing

import pytest

from effgen.cli import monitor
from effgen.cli._main import create_parser
from effgen.observability import run_log


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Keep run history and any server URL/key out of the developer's real state."""
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "state" / "runs"))
    monkeypatch.delenv("EFFGEN_SERVER_URL", raising=False)
    monkeypatch.delenv("EFFGEN_API_KEY", raising=False)
    run_log.clear()
    yield
    run_log.clear()


def _args(**kwargs):
    defaults = {
        "output_json": False,
        "once": False,
        "interval": monitor.DEFAULT_INTERVAL,
        "count": None,
        "limit": monitor.DEFAULT_ACTIVITY_LIMIT,
        "url": None,
        "port": None,
        "api_key": None,
        "quiet": False,
        "no_animation": False,
        "theme": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["top", "monitor"])
def test_both_command_names_parse(command):
    """`monitor` is an alias for `top` and carries the same flags."""
    args = create_parser().parse_args([command, "--json", "--interval", "1.5", "--port", "9123"])
    assert args.command == command
    assert args.output_json is True
    assert args.interval == 1.5
    assert args.port == 9123


def test_rejects_an_out_of_range_interval(capsys):
    assert monitor.run_monitor_command(_args(interval=0.01)) == 1
    assert "--interval must be between" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Server discovery
# ---------------------------------------------------------------------------


def test_server_url_precedence(monkeypatch):
    assert monitor.resolve_server_url() == "http://127.0.0.1:8000"
    monkeypatch.setenv("EFFGEN_SERVER_URL", "http://ops.example:9000/")
    assert monitor.resolve_server_url() == "http://ops.example:9000"
    assert monitor.resolve_server_url(port=8123) == "http://127.0.0.1:8123"
    assert monitor.resolve_server_url("http://other:1/") == "http://other:1"
    # A bare host:port is a natural thing to type and gains a scheme.
    assert monitor.resolve_server_url("box:8000") == "http://box:8000"


# ---------------------------------------------------------------------------
# Snapshot shape and scope labelling
# ---------------------------------------------------------------------------


def test_snapshot_panels_each_declare_a_scope():
    """Every panel names the window and process it measures."""
    snapshot = monitor.collect_snapshot(url="http://127.0.0.1:1")
    assert set(snapshot) == {"ts", "header", "activity", "traffic", "by_model", "spend", "gpu"}
    for name in ("activity", "traffic", "by_model", "spend", "gpu"):
        panel = snapshot[name]
        assert panel["scope"], f"{name} panel has no scope label"
        assert "available" in panel
    # The server-backed panels and the local ledger measure different things and
    # must not claim the same window.
    assert snapshot["traffic"]["scope"] != snapshot["spend"]["scope"]
    assert snapshot["activity"]["scope"] != snapshot["traffic"]["scope"]


def test_unreachable_server_reports_a_reason_not_zeros():
    snapshot = monitor.collect_snapshot(url=f"http://127.0.0.1:{_free_port()}")
    traffic = snapshot["traffic"]
    assert traffic["available"] is False
    assert "requests" not in traffic, "an unreachable server must not report a request count"
    assert traffic["server_url"] in traffic["unavailable_reason"]
    assert snapshot["header"]["server_reachable"] is False
    # The local panels still work with no server.
    assert snapshot["activity"]["available"] is True


@pytest.mark.parametrize(
    "bad_url",
    ["://x", "file:///etc/passwd", "not a url"],
)
def test_a_malformed_url_is_reported_in_the_panel_not_raised(bad_url):
    """A mistyped --url must not cost the panels that need no server."""
    document, error = monitor._fetch_json(bad_url, api_key=None)
    assert document is None
    assert error, "a URL problem must come back as a reason"

    snapshot = monitor.collect_snapshot(url=bad_url)
    assert snapshot["traffic"]["available"] is False
    assert snapshot["traffic"]["unavailable_reason"]
    # The local sources are unaffected by a bad server URL.
    assert snapshot["activity"]["available"] is True
    assert snapshot["spend"]["available"] is True


def test_generation_count_and_http_histogram_are_labelled_separately():
    """The two live in one panel but count different things; neither may imply the other."""
    assert monitor.SCOPE_HTTP_STATUS != monitor.SCOPE_TRAFFIC
    panel = {
        "available": True,
        "scope": monitor.SCOPE_TRAFFIC,
        "requests": 10,
        "errors": 1,
        "error_rate": 0.1,
        "tokens": 808,
        "by_status": {"200": 13, "400": 1},
        "by_status_scope": monitor.SCOPE_HTTP_STATUS,
    }
    buffer = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(buffer):
        monitor._render_traffic_table(panel, None)
    text = buffer.getvalue()
    # A bare "Requests: 10" next to "200x13" reads as a contradiction.
    assert "Generation requests" in text
    assert "all routes" in text


def test_activity_reads_the_durable_history_and_keeps_unpriced_cost_null():
    run_log.record_run(
        run_id="aaaa1111",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        task="local run",
        output="done",
        duration_s=1.25,
        cost_usd=None,
    )
    run_log.record_run(
        run_id="bbbb2222",
        model="gpt-5-nano",
        task="cloud run",
        output="",
        duration_s=0.5,
        cost_usd=0.0001,
        error="boom",
    )
    panel = monitor.collect_snapshot(url="http://127.0.0.1:1")["activity"]
    by_id = {run["run_id"]: run for run in panel["runs"]}
    assert by_id["aaaa1111"]["cost_usd"] is None, "an unpriced run must not become $0"
    assert by_id["bbbb2222"]["status"] == "error"
    # An absent price renders as an em dash, never as a zero amount.
    assert monitor._fmt_usd(None) == "—"
    assert monitor._fmt_usd(0.0001) == "$0.000100"


def test_spend_panel_reports_burn_rate_and_budget_keys():
    spend = monitor.collect_snapshot(url="http://127.0.0.1:1")["spend"]
    assert spend["available"] is True
    for key in ("total_cost_usd", "burn_rate_usd_per_hour", "daily_budget_usd", "unpriced_models"):
        assert key in spend


# ---------------------------------------------------------------------------
# Non-TTY / --json behavior
# ---------------------------------------------------------------------------


def test_json_mode_prints_one_document_and_exits(capsys):
    assert monitor.run_monitor_command(_args(output_json=True, url="http://127.0.0.1:1")) == 0
    out = capsys.readouterr().out
    document = json.loads(out)  # a single well-formed document, nothing else on stdout
    assert set(document) >= {"header", "activity", "traffic", "spend", "gpu"}
    assert "\x1b[" not in out, "no ANSI control sequences on a non-TTY stream"


def test_piped_output_is_a_static_snapshot_with_no_cursor_control(capsys):
    """Not a terminal: one snapshot, plain text, no full-screen takeover."""
    assert monitor.run_monitor_command(_args(url="http://127.0.0.1:1")) == 0
    out = capsys.readouterr().out
    assert "Activity" in out and "Spend" in out and "GPU" in out
    assert "\x1b[?1049h" not in out, "must not switch to the alternate screen buffer"
    assert "\x1b[" not in out
    # The unavailable server is named, not silently rendered as zero traffic.
    assert "Traffic — unavailable" in out


def test_every_documented_panel_appears_when_the_server_is_unreachable(capsys):
    """A panel that has no data still says so; it never simply disappears.

    A reader scanning ``--once`` output cannot distinguish an omitted panel from
    one that does not exist, so each of the five keeps its title and states the
    reason it is empty.
    """
    assert monitor.run_monitor_command(_args(url="http://127.0.0.1:1")) == 0
    out = capsys.readouterr().out
    for title in ("Activity", "Traffic", "Per-model", "Spend", "GPU"):
        assert title in out, f"{title} panel is missing from the static snapshot"
    assert "Per-model — unavailable" in out


def test_no_color_forces_the_static_path(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    assert monitor.run_monitor_command(_args(url="http://127.0.0.1:1")) == 0
    assert "\x1b[" not in capsys.readouterr().out


def test_no_animation_forces_the_static_path(capsys):
    assert monitor.run_monitor_command(_args(no_animation=True, url="http://127.0.0.1:1")) == 0
    assert "\x1b[?1049h" not in capsys.readouterr().out


def test_render_snapshot_writes_plain_text_without_a_console():
    snapshot = monitor.collect_snapshot(url="http://127.0.0.1:1")
    buffer = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(buffer):
        monitor.render_snapshot(snapshot, console=None)
    text = buffer.getvalue()
    assert monitor.SCOPE_ACTIVITY in text
    assert monitor.SCOPE_SPEND in text


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------


def test_live_loop_is_bounded_by_count():
    """The refresh loop terminates on its own rather than running forever."""
    from effgen.ui.theme import get_console

    console = get_console(file=io.StringIO(), force_terminal=True, width=100, height=30)
    assert (
        monitor.run_live(
            console=console,
            url="http://127.0.0.1:1",
            port=None,
            api_key=None,
            limit=3,
            interval=0.5,
            count=2,
        )
        == 0
    )


def test_live_per_model_rows_name_their_provider_when_the_model_repeats():
    """The narrow live layout has no Provider column, so a repeated name is qualified.

    One model name served by two providers carries different numbers per row;
    two rows labelled identically would leave an operator unable to tell which
    is which.
    """
    from effgen.ui.theme import get_console

    snapshot = monitor.collect_snapshot(url="http://127.0.0.1:1", interval=2.0)
    snapshot["by_model"] = {
        "scope": "this server process since start",
        "available": True,
        "unavailable_reason": None,
        "rows": [
            {"model": "gpt-oss-120b", "provider": "cerebras", "calls": 4,
             "errors": 1, "error_rate": 0.25, "p95_latency_s": 8.0, "cost_usd": 0.75},
            {"model": "gpt-oss-120b", "provider": "groq", "calls": 2,
             "errors": 0, "error_rate": 0.0, "p95_latency_s": 0.3, "cost_usd": 0.25},
            {"model": "gpt-5-nano", "provider": "openai", "calls": 1,
             "errors": 0, "error_rate": 0.0, "p95_latency_s": 0.4, "cost_usd": 0.01},
        ],
    }
    console = get_console(file=io.StringIO(), force_terminal=True, width=160, height=40)
    console.print(monitor._build_live_view(snapshot))
    rendered = console.file.getvalue()

    assert "gpt-oss-120b (cerebras)" in rendered
    assert "gpt-oss-120b (groq)" in rendered
    # A name that appears once is left alone.
    assert "gpt-5-nano (openai)" not in rendered
    assert "gpt-5-nano" in rendered


def test_live_view_builds_for_a_snapshot_with_no_server():
    from effgen.ui.theme import get_console

    snapshot = monitor.collect_snapshot(url="http://127.0.0.1:1", interval=2.0)
    console = get_console(file=io.StringIO(), force_terminal=True, width=120, height=40)
    console.print(monitor._build_live_view(snapshot))
    rendered = console.file.getvalue()
    for heading in ("Activity", "Traffic", "Spend", "GPU"):
        assert heading in rendered


# ---------------------------------------------------------------------------
# Against a real server
# ---------------------------------------------------------------------------


@pytest.fixture
def live_server(monkeypatch):
    """Run a real effGen app on a loopback port for the duration of one test."""
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")

    port = _free_port()
    monkeypatch.setenv("EFFGEN_API_KEY", "monitor-test-key")
    monkeypatch.setenv("EFFGEN_PUBLIC_DASHBOARD", "0")

    from effgen.server.app import create_app

    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="error", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = 30
        while not server.started and deadline > 0:
            threading.Event().wait(0.1)
            deadline -= 1
        assert server.started, "server did not start"
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=15)


def test_server_panels_populate_from_a_running_server(live_server):
    snapshot = monitor.collect_snapshot(
        port=live_server, api_key=os.environ["EFFGEN_API_KEY"], limit=3
    )
    assert snapshot["header"]["server_reachable"] is True
    traffic = snapshot["traffic"]
    assert traffic["available"] is True
    assert traffic["scope"] == monitor.SCOPE_TRAFFIC
    for key in ("requests", "errors", "error_rate", "by_status"):
        assert key in traffic
    assert snapshot["by_model"]["available"] is True


def test_a_rejected_key_is_reported_rather_than_shown_as_no_traffic(live_server):
    snapshot = monitor.collect_snapshot(port=live_server, api_key="not-the-key")
    traffic = snapshot["traffic"]
    assert traffic["available"] is False
    reason = traffic["unavailable_reason"]
    assert "401" in reason
    assert "--api-key" in reason, "the message must say how to fix it"
