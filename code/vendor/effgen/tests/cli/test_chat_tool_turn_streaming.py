"""A chat turn with tools attached streams its answer where it can.

``chat`` split its turns: no tools streamed through ``Agent.stream()``, any tool
attached went through the blocking ``Agent.run()`` under the status line. The
reason was that the streaming path's tool loop was the prompt-based ReAct
scaffold, and trading the loop's reliability for token deltas is the wrong way
round. ``Agent.stream()`` now dispatches native tool calls on any model whose
adapter records the calls it streams, so that reason no longer holds for those
models — and the coding session already had the two-region handoff to reuse.

The split is now one branch guarded by the same capability probe, and everything
that does not qualify keeps the blocking turn exactly as before.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class _Turn:
    """The mixin under test, with the surrounding session stubbed out."""

    def __init__(self, *, can_stream, events=(), response=None, console=object()):
        from effgen.cli.chat_turn import ChatTurnMixin

        self.__class__ = type("_TurnImpl", (ChatTurnMixin, _Turn), {})
        self.animate = True
        self.console = console
        self.interactive = True
        self.quiet = False
        self.model_id = "fake:model"
        self.last_trace = None
        self.shown: list[str] = []
        self.agent = SimpleNamespace(
            execution_tracker=None,
            last_stream_response=response,
            _can_stream_native_tools=lambda: can_stream,
            stream=lambda *a, **k: iter(events),
            run=self._blocking_run,
        )
        self.ran_blocking = False

    def _blocking_run(self, user_input, **kwargs):
        self.ran_blocking = True
        return SimpleNamespace(
            output="blocking answer", success=True, metadata={},
            tokens_used=3, execution_trace=None,
        )

    def _show_answer(self, answer):
        self.shown.append(answer)

    def _show_progress(self, text):
        self.shown.append(text)


class _FakeRegion:
    """Stands in for the two live terminal regions."""

    def __init__(self, *a, **k):
        self.is_open = False
        self.text = ""

    def open(self):
        self.is_open = True

    def push(self, chunk):
        self.text += chunk
        _PUSHED.append(chunk)

    def close(self):
        self.is_open = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_PUSHED: list[str] = []


def _event(kind, text=""):
    return SimpleNamespace(kind=kind, text=text)


def _response(answer):
    return SimpleNamespace(
        output=answer, success=True, metadata={}, tokens_used=7,
        execution_trace=None,
    )


def test_a_model_that_cannot_stream_tool_calls_keeps_the_blocking_turn(monkeypatch):
    from effgen.cli import progress as _progress

    monkeypatch.setattr(_progress, "LiveStatus", _FakeRegion)
    turn = _Turn(can_stream=False)
    result = turn._run_with_tools("hi")
    assert turn.ran_blocking is True
    assert result == "blocking answer"


def test_a_qualifying_model_streams_and_does_not_reprint_the_answer(monkeypatch):
    from effgen.cli import progress as _progress

    _PUSHED.clear()

    monkeypatch.setattr(_progress, "LiveAnswer", _FakeRegion)
    monkeypatch.setattr(_progress, "LiveStatus", _FakeRegion)

    events = [
        _event("tool_call"),
        _event("observation"),
        _event("answer", "The answer "),
        _event("answer", "is 42."),
    ]
    turn = _Turn(can_stream=True, events=events, response=_response("The answer is 42."))
    result = turn._run_with_tools("hi")

    assert turn.ran_blocking is False, "the turn ran twice"
    assert result == "The answer is 42."
    assert _PUSHED == ["The answer ", "is 42."]
    # Already on screen: printing it again would show it twice.
    assert turn.shown == []


def test_a_stream_that_produced_no_response_falls_back(monkeypatch):
    from effgen.cli import progress as _progress

    monkeypatch.setattr(_progress, "LiveAnswer", _FakeRegion)
    monkeypatch.setattr(_progress, "LiveStatus", _FakeRegion)

    turn = _Turn(can_stream=True, events=[_event("answer", "partial")], response=None)
    result = turn._run_with_tools("hi")

    assert turn.ran_blocking is True, "the turn must be run the blocking way"
    assert result == "blocking answer"


@pytest.mark.parametrize("raises", [True, False])
def test_the_capability_probe_never_breaks_a_turn(raises):
    turn = _Turn(can_stream=False)
    if raises:
        def _boom():
            raise RuntimeError("probe exploded")

        turn.agent._can_stream_native_tools = _boom
    assert turn._can_stream_tool_turn() is False
