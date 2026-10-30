"""How a coding turn renders an answer that arrives as it is written.

Two claims are proven here. The first is that answer text reaches a real
terminal *while the turn is still running* — driven through a pty, because a
captured string buffer cannot tell a live region from a final print. The second
is that every non-interactive surface keeps the blocking turn: piped output,
``--json``, ``--quiet`` and ``NO_COLOR`` must never take the streaming branch,
which is what keeps their output what it was.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from effgen.cli import _main
from effgen.cli.code.engine import CodeEngine
from effgen.cli.code.permissions import PermissionMode
from effgen.cli.code.repl import CodeREPL
from effgen.cli.progress import _CI_ENV_VARS
from effgen.core.agent_response import AgentResponse, StreamEvent
from effgen.memory.short_term import ShortTermMemory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")

ANSWER = "Created hello.py in the workspace. Run it with python hello.py."


class _StreamingAgent:
    """An agent that streams an answer in deltas, then reports the turn."""

    def __init__(self, *, delta_pause: float = 0.0, capable: bool = True) -> None:
        self.tools = {}
        self.short_term_memory = ShortTermMemory()
        self.execution_tracker = None
        self.session = None
        self.model = SimpleNamespace(total_tokens=0, total_cost=0.0)
        self.calls: list[str] = []
        self._pause = delta_pause
        self._capable = capable
        self.last_stream_response: AgentResponse | None = None

    def _can_stream_native_tools(self) -> bool:
        return self._capable

    def _response(self) -> AgentResponse:
        return AgentResponse(
            output=ANSWER, success=True, iterations=2, tool_calls=1,
            tokens_used=99, execution_time=0.2,
            metadata={"reason": "final_answer", "tool_calling_strategy": "native"},
        )

    def run(self, task, mode=None):
        self.calls.append("run")
        return self._response()

    def stream(self, task, include_events=False, **kwargs):
        self.calls.append("stream")
        yield StreamEvent(kind="tool_call", tool="file_operations", tool_input="{}")
        yield StreamEvent(kind="observation", tool="file_operations", text="written")
        for word in ANSWER.split(" "):
            if self._pause:
                time.sleep(self._pause)
            yield StreamEvent(kind="answer", text=word + " ")
        self.last_stream_response = self._response()
        yield StreamEvent(kind="usage", usage={"total_tokens": 99})

    def reset_memory(self) -> None:
        self.short_term_memory = ShortTermMemory()

    def close(self) -> None:
        pass


def make_repl(tmp_path, *, console=None, quiet=False, capable=True, pause=0.0):
    """Build a session whose agent streams, matching the shipped wiring."""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    args = SimpleNamespace(
        workspace=str(ws), model="fake:model", provider=None,
        plan_only=False, auto_edit=False, assume_yes=False,
        temperature=None, max_tokens=None, max_iterations=None,
        quiet=quiet, verbose=False, no_animation=False,
    )
    cli = _main.CLIInterface()
    cli.console = console
    repl = CodeREPL(cli, args)
    repl.model_id = "fake:model"
    repl.mode = PermissionMode.AUTO_EDIT
    repl.mode_explicit = True
    repl.quiet = quiet
    repl.interactive = True
    engine = CodeEngine(
        model="fake:model", workspace=ws, mode=PermissionMode.AUTO_EDIT,
        interactive=True, on_event=repl._on_event, on_diff=repl._on_diff,
    )
    engine._agent = _StreamingAgent(delta_pause=pause, capable=capable)
    repl.engine = engine
    repl.project = engine.load_project(refresh=True)
    return repl, ws


# --------------------------------------------------------------------------- #
# Non-interactive surfaces keep the blocking turn
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("env", "console", "quiet"),
    [
        ({}, None, False),                       # no console: piped / --json
        ({"NO_COLOR": "1"}, None, False),        # colour disabled
        ({}, None, True),                        # --quiet
        ({"EFFGEN_NO_ANIM": "1"}, None, False),  # animation opted out
    ],
    ids=["piped", "no-color", "quiet", "no-anim"],
)
def test_a_non_animating_surface_never_streams_the_turn(
    tmp_path, monkeypatch, env, console, quiet
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    repl, _ = make_repl(tmp_path, console=console, quiet=quiet)
    repl._do_turn("write hello.py")
    assert repl.agent.calls == ["run"], (
        "a non-animating surface must keep the blocking turn"
    )


def test_a_model_that_cannot_stream_its_tool_calls_keeps_the_blocking_turn(tmp_path):
    repl, _ = make_repl(tmp_path, capable=False)
    repl._do_turn("write hello.py")
    assert repl.agent.calls == ["run"]


# --------------------------------------------------------------------------- #
# The answer is not printed twice
# --------------------------------------------------------------------------- #
def test_an_answer_that_already_streamed_is_not_printed_again(tmp_path):
    """The record still renders; only the answer body is left off."""
    printed: list[str] = []
    repl, _ = make_repl(tmp_path)

    import effgen.ui.render as render

    original = render.answer_surface

    def _spy(text, **kwargs):
        printed.append(text)
        return original(text, **kwargs)

    render.answer_surface = _spy
    try:
        result = SimpleNamespace(
            answer=ANSWER, success=True, partial=False, partial_output="",
        )
        repl.cli.console = SimpleNamespace()  # any console at all
        repl._render_answer(result, already_streamed=True)
        assert printed == []
        repl._render_answer(result, already_streamed=False)
        assert printed == [ANSWER]
    finally:
        render.answer_surface = original


# --------------------------------------------------------------------------- #
# The interactive claim, on a real pty
# --------------------------------------------------------------------------- #
CHILD = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {repo!r})
    from tests.cli.test_code_stream_render import make_repl
    from pathlib import Path
    from effgen.ui.theme import get_console

    ws = Path({ws!r})
    repl, _ = make_repl(ws, console=get_console(), pause=0.15)
    repl._do_turn("write hello.py")
    print("CHILD-DONE", flush=True)
    """
)


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="no pty on this platform")
def test_answer_deltas_reach_a_real_terminal_before_the_turn_ends(tmp_path):
    """Deltas must be on screen while the turn runs, not printed at the end.

    The child writes to a pty, so ``rich`` renders live exactly as it does for a
    user. The parent timestamps what arrives: the first word of the answer has
    to land well before the turn's closing line, which it cannot if the answer
    is printed in one block when the turn finishes.
    """
    child = CHILD.format(repo=REPO_ROOT, ws=str(tmp_path))
    primary, secondary = os.openpty()
    env = dict(os.environ)
    env.update({
        "TERM": "xterm-256color", "COLUMNS": "100", "LINES": "40",
        "PYTHONPATH": REPO_ROOT, "FORCE_COLOR": "1",
    })
    env.pop("NO_COLOR", None)
    env.pop("EFFGEN_NO_ANIM", None)
    # The child must see an animating surface, and animation is switched off by any of
    # the CI markers -- not by ``CI`` alone. Clearing them from the product's own list
    # keeps this in step: a runner that sets only ``GITHUB_ACTIONS`` would otherwise
    # take the blocking branch and the timing assertion below would report streaming
    # that never had a chance to happen.
    for marker in _CI_ENV_VARS:
        env.pop(marker, None)
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdin=secondary, stdout=secondary, stderr=secondary,
        env=env, cwd=REPO_ROOT, close_fds=True,
    )
    os.close(secondary)

    chunks: list[tuple[float, str]] = []
    started = time.monotonic()
    selector = selectors.DefaultSelector()
    selector.register(primary, selectors.EVENT_READ)
    try:
        while time.monotonic() - started < 90:
            if not selector.select(timeout=1.0):
                if proc.poll() is not None:
                    break
                continue
            try:
                data = os.read(primary, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append((time.monotonic() - started, data.decode("utf-8", "replace")))
            if "CHILD-DONE" in chunks[-1][1]:
                break
    finally:
        selector.close()
        os.close(primary)
        proc.wait(timeout=30)

    transcript = "".join(text for _at, text in chunks)
    plain = ANSI.sub("", transcript)
    assert "Traceback" not in transcript, transcript[-3000:]
    assert "hello.py" in plain, plain[-3000:]

    first_word_at = next(
        (at for at, text in chunks if "Created" in ANSI.sub("", text)), None
    )
    last_word_at = next(
        (at for at, text in chunks if "python" in ANSI.sub("", text)), None
    )
    assert first_word_at is not None, plain[-3000:]
    assert last_word_at is not None, plain[-3000:]
    assert last_word_at - first_word_at > 0.3, (
        f"the answer arrived all at once ({first_word_at:.2f}s → {last_word_at:.2f}s); "
        "it was not streamed to the terminal"
    )
