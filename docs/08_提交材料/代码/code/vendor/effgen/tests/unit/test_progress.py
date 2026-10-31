"""Unit tests for the CLI live-progress / status presentation layer.

These exercise the *decision* logic (when to animate), the pure formatting
(summary line, model labels, sticky tool labels), the execution-tracker
listener wiring, and the streaming soft-cursor — all without any live model
calls. The animated rendering itself is verified end-to-end with real models.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

from effgen.cli import progress as P

# ---------------------------------------------------------------------------
# animation_enabled / color / CI gating
# ---------------------------------------------------------------------------


class _FakeTTY(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:  # noqa: D401
        return self._tty


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start each test from a known, non-CI, color-on environment."""
    for var in ("NO_COLOR", "EFFGEN_NO_ANIM", *P._CI_ENV_VARS):
        monkeypatch.delenv(var, raising=False)
    yield


def test_animation_enabled_on_interactive_tty():
    # The live region is Rich's; without it there is no animation to enable.
    pytest.importorskip("rich")
    assert P.animation_enabled(stream=_FakeTTY(True)) is True


def test_animation_disabled_when_not_a_tty():
    assert P.animation_enabled(stream=_FakeTTY(False)) is False


@pytest.mark.parametrize("kwargs", [{"quiet": True}, {"no_animation": True}])
def test_animation_disabled_by_flags(kwargs):
    assert P.animation_enabled(stream=_FakeTTY(True), **kwargs) is False


def test_animation_disabled_by_no_anim_env(monkeypatch):
    monkeypatch.setenv("EFFGEN_NO_ANIM", "1")
    assert P.animation_enabled(stream=_FakeTTY(True)) is False


def test_animation_disabled_by_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert P.animation_enabled(stream=_FakeTTY(True)) is False
    assert P.color_enabled() is False


def test_animation_disabled_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert P.is_ci() is True
    assert P.animation_enabled(stream=_FakeTTY(True)) is False


# ---------------------------------------------------------------------------
# pure formatting helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
        ("openai:gpt-5-nano", "gpt-5-nano"),
        ("gpt-5-nano", "gpt-5-nano"),
        (None, "model"),
        ("", "model"),
    ],
)
def test_short_model_label(raw, expected):
    assert P.short_model_label(raw) == expected


def test_short_model_label_truncates():
    out = P.short_model_label("a" * 60, limit=10)
    assert len(out) == 10
    assert out.endswith("…")


@dataclass
class _FakeModel:
    model_name: str = ""
    _is_reasoning_model: bool = False


@dataclass
class _FakeAgent:
    model: Any = None


def test_is_reasoning_agent_by_flag():
    agent = _FakeAgent(model=_FakeModel("o3-mini", _is_reasoning_model=True))
    assert P.is_reasoning_agent(agent) is True


def test_is_reasoning_agent_by_name_heuristic():
    assert P.is_reasoning_agent(_FakeAgent(model=_FakeModel("o4-mini"))) is True
    assert P.is_reasoning_agent(_FakeAgent(model=_FakeModel("deepseek-r1"))) is True


def test_is_reasoning_agent_false_for_chat_models():
    assert P.is_reasoning_agent(_FakeAgent(model=_FakeModel("gpt-5-nano"))) is False
    assert P.is_reasoning_agent(_FakeAgent(model=_FakeModel("Qwen/Qwen2.5-1.5B-Instruct"))) is False
    assert P.is_reasoning_agent(_FakeAgent(model=None)) is False


@dataclass
class _FakeResponse:
    success: bool = True
    execution_time: float = 3.2
    tool_calls: int = 2
    tokens_used: int = 1204
    metadata: dict = field(default_factory=dict)


def test_summary_line_success():
    plain, markup = P.summary_line(_FakeResponse(metadata={"cost_usd": 0.0006}))
    assert plain == "✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006"
    assert "[green]" in markup


def test_summary_line_singular_tool_and_no_cost():
    plain, _ = P.summary_line(_FakeResponse(tool_calls=1, tokens_used=10, metadata={}))
    assert "1 tool ·" in plain
    assert "$" not in plain  # no cost when none recorded (never a fake $0)


def test_summary_line_failure_shows_reason():
    resp = _FakeResponse(success=False, tool_calls=0, tokens_used=0,
                         metadata={"reason": "model_not_found"})
    plain, markup = P.summary_line(resp)
    assert plain.startswith("✗ Failed (model_not_found)")
    assert "[red]" in markup


# ---------------------------------------------------------------------------
# execution_trace_lines — the renderer the quickstart shares
# ---------------------------------------------------------------------------

def test_execution_trace_lines_reports_a_tool_call():
    # The agent's trace is a list of typed events (tool_call_start/complete), not
    # a ReAct action/observation shape. The shared renderer must surface the tool
    # so the quickstart no longer says "answered directly" when a tool ran.
    trace = [
        {"type": "task_start", "message": "Starting task"},
        {"type": "reasoning_step", "message": "Iteration 1"},
        {"type": "tool_call_start",
         "data": {"tool_name": "calculator", "tool_input": "25 * 17"}},
        {"type": "tool_call_complete", "data": {"result": "425"}},
    ]
    lines = P.execution_trace_lines(trace)
    text = " ".join(t for _style, t in lines)
    assert "calculator" in text


def test_execution_trace_lines_empty_for_direct_answer():
    # A pure direct answer (no tool/reasoning events) yields no lines, so the
    # quickstart correctly falls back to "answered directly, no tools needed".
    assert P.execution_trace_lines([{"type": "task_start", "message": "x"}]) == []
    assert P.execution_trace_lines([]) == []
    assert P.execution_trace_lines(None) == []


def test_execution_trace_lines_shows_per_step_duration():
    # The gap before a tool_call_start is the model's think time for that step;
    # the renderer surfaces it so a slow step no longer looks instant.
    trace = [
        {"type": "reasoning_step", "message": "Iteration 1", "timestamp": 100.0},
        {"type": "tool_call_start", "timestamp": 114.4,
         "data": {"tool_name": "calculator", "tool_input": "1+1"}},
        {"type": "tool_call_complete", "timestamp": 114.5, "data": {"result": "2"}},
    ]
    text = " ".join(t for _s, t in P.execution_trace_lines(trace))
    assert "14.4s" in text  # think time attributed to the tool step


def test_execution_timeline_lines_bars_and_total():
    trace = [
        {"type": "reasoning_step", "message": "i1", "timestamp": 0.0},
        {"type": "tool_call_start", "timestamp": 6.0,
         "data": {"tool_name": "calculator", "tool_input": "a"}},
        {"type": "tool_call_complete", "timestamp": 6.1, "data": {"result": "x"}},
        {"type": "reasoning_step", "message": "i2", "timestamp": 6.1},
        {"type": "tool_call_start", "timestamp": 9.1,
         "data": {"tool_name": "calculator", "tool_input": "b"}},
        {"type": "tool_call_complete", "timestamp": 9.2, "data": {"result": "y"}},
    ]
    lines = P.execution_timeline_lines(trace)
    text = "\n".join(t for _s, t in lines)
    assert "█" in text          # a proportional bar was drawn
    assert "6.0s" in text       # the first step's duration
    assert lines[-1][1].startswith("total")
    assert "9.0s" in lines[-1][1]  # 6.0 + 3.0 across the two steps


def test_execution_timeline_lines_empty_without_steps():
    assert P.execution_timeline_lines([]) == []
    assert P.execution_timeline_lines(None) == []
    # reasoning-only (no tool/delegation) has nothing to time.
    assert P.execution_timeline_lines([{"type": "reasoning_step", "timestamp": 1.0}]) == []


# ---------------------------------------------------------------------------
# sticky status state
# ---------------------------------------------------------------------------


def test_status_state_sticky_tool_label():
    s = P._StatusState("gpt-5-nano", reasoning=False)
    assert s.effective_label() == "Calling gpt-5-nano…"
    s.set_step("Running calculator…")
    # Immediately after a fast tool completes, the base reverts but the sticky
    # step label must remain visible for its minimum window.
    s.set_base(s._idle_label())
    assert s.effective_label() == "Running calculator…"
    # Force the sticky window to expire.
    s.sticky_until = 0.0
    assert s.effective_label() == "Calling gpt-5-nano…"


def test_status_state_reasoning_label():
    s = P._StatusState("o3-mini", reasoning=True)
    assert "reasoning" in s.effective_label()
    assert "🧠" in s.effective_label()


# ---------------------------------------------------------------------------
# LiveStatus event mapping (no rendering)
# ---------------------------------------------------------------------------


class _EvType(Enum):
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TASK_DECOMPOSITION = "task_decomposition"


@dataclass
class _Ev:
    type: _EvType
    data: dict = field(default_factory=dict)
    agent_id: str | None = None


def _live_status():
    # console=None makes LiveStatus a non-rendering no-op, but on_event still
    # mutates the shared state, which is what we assert on.
    ls = P.LiveStatus(None, model_label="gpt-5-nano")
    return ls


def test_live_status_tool_label_transition():
    ls = _live_status()
    ls.on_event(_Ev(_EvType.TOOL_CALL_START, data={"tool_name": "calculator"}))
    assert ls.state.effective_label() == "Running calculator…"
    ls.on_event(_Ev(_EvType.TOOL_CALL_COMPLETE, data={"tool_name": "calculator"}))
    # sticky still showing immediately after completion
    assert ls.state.effective_label() == "Running calculator…"
    ls.state.sticky_until = 0.0
    assert ls.state.effective_label() == "Calling gpt-5-nano…"


def test_live_status_planning_label():
    ls = _live_status()
    ls.on_event(_Ev(_EvType.TASK_DECOMPOSITION))
    assert ls.state.effective_label() == "Planning…"


# ---------------------------------------------------------------------------
# ExecutionTracker listener wiring
# ---------------------------------------------------------------------------


def test_execution_tracker_listener_receives_events():
    from effgen.core.execution_tracker import EventType, ExecutionEvent, ExecutionTracker

    tracker = ExecutionTracker()
    seen = []
    cb = seen.append
    tracker.add_listener(cb)
    tracker.track_event(ExecutionEvent(type=EventType.TASK_START, message="x"))
    assert len(seen) == 1
    tracker.remove_listener(cb)
    tracker.track_event(ExecutionEvent(type=EventType.TASK_COMPLETE, message="y"))
    assert len(seen) == 1  # no more after removal


def test_execution_tracker_listener_exception_is_swallowed():
    from effgen.core.execution_tracker import EventType, ExecutionEvent, ExecutionTracker

    tracker = ExecutionTracker()

    def boom(_event):
        raise RuntimeError("listener blew up")

    tracker.add_listener(boom)
    # A misbehaving listener must never break tracking.
    tracker.track_event(ExecutionEvent(type=EventType.TASK_START, message="x"))
    assert len(tracker.events) == 1


# ---------------------------------------------------------------------------
# StepProgress no-op path
# ---------------------------------------------------------------------------


def test_step_progress_noop_without_animation():
    with P.StepProgress(None, total=3, description="Batch", animate=False) as bar:
        bar.update(1)
        bar.advance()
        bar.update(3, total=3)
    # no exception, no console required


# ---------------------------------------------------------------------------
# Suspending the live region so a question can be read
# ---------------------------------------------------------------------------


class _FakeLive:
    """Records the start/stop calls ``LiveStatus`` makes on a live region."""

    def __init__(self, events):
        self._events = events

    def __enter__(self):
        self._events.append("enter")
        return self

    def __exit__(self, *_exc):
        self._events.append("exit")
        return False

    def stop(self):
        self._events.append("stop")

    def start(self, refresh=False):
        self._events.append("start")


def _patch_live(monkeypatch, events):
    monkeypatch.setattr(P, "_RICH_AVAILABLE", True)
    monkeypatch.setattr(P, "Live", lambda *a, **k: _FakeLive(events), raising=False)


def test_paused_stops_and_restarts_the_live_region(monkeypatch):
    events: list[str] = []
    _patch_live(monkeypatch, events)
    status = P.LiveStatus(object(), model_label="m")
    with status:
        assert P._ACTIVE_STATUS is status
        with status.paused():
            events.append("asked")
    assert events == ["enter", "stop", "asked", "start", "exit"]
    assert P._ACTIVE_STATUS is None


def test_suspend_live_status_is_a_noop_without_one():
    with P.suspend_live_status():
        pass  # nothing animating: the block still runs


def test_confirmation_prompt_takes_the_status_line_down(monkeypatch, capsys):
    """A question asked mid-run must not be repainted over by the status line."""
    from effgen.cli.code import permissions

    events: list[str] = []
    _patch_live(monkeypatch, events)
    monkeypatch.setattr("builtins.input", lambda: "y")
    with P.LiveStatus(object(), model_label="m"):
        answer = permissions._prompt_confirm("Write a.py (+1/-0)")
    assert answer == "y"
    assert events == ["enter", "stop", "start", "exit"]
    assert "Write a.py (+1/-0) [y/N/a]" in capsys.readouterr().err
