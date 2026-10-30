"""The ``effgen code`` interactive REPL, driven offline with a fake agent.

The plan/run/fix loop and streaming are proven live on real models; here the REPL
plumbing — slash dispatch, files-in-context management, the plan → stage → apply
flow, session save/load round-trip, and the choice between the interactive loop
and the single-shot path — is pinned deterministically. The fake agent drives the
real gated tools, so confinement and the permission gate under test are the
shipped behavior, not a reimplementation.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from effgen.cli import _main
from effgen.cli.code.engine import CodeEngine
from effgen.cli.code.permissions import PermissionMode
from effgen.cli.code.repl import _SLASH_COMMANDS, CodeREPL, _estimate_tokens
from effgen.cli.code.tools import build_code_tools
from effgen.memory.short_term import MessageRole, ShortTermMemory


class _FakeResponse:
    def __init__(self, output, success=True, reason="final_answer", strategy=""):
        self.output = output
        self.success = success
        self.metadata = {"reason": reason}
        if strategy:
            self.metadata["tool_calling_strategy"] = strategy
        self.iterations = 1
        self.tool_calls = 1
        self.tokens_used = 42
        self.execution_time = 0.1
        self.provider = "fake"
        self.execution_trace = None


class _FakeAgent:
    """An agent whose ``run`` drives the gated tools, then returns a response."""

    def __init__(self, tools, *, actions, answer="done", strategy=""):
        self.tools = {t.name: t for t in tools}
        self._strategy = strategy
        self._ce, self._repl, self._fops, self._bash = tools
        self._actions = actions
        self._answer = answer
        self.short_term_memory = ShortTermMemory()
        self.execution_tracker = None
        self.session = None
        self.model = SimpleNamespace(total_tokens=0, total_cost=0.0)
        self.closed = False

    def run(self, task, mode=None):
        for action in self._actions:
            asyncio.run(self._apply(action))
        return _FakeResponse(self._answer, strategy=self._strategy)

    async def _apply(self, action):
        kind, arg = action
        if kind == "write":
            path, content = arg
            await self._fops.execute(operation="write", path=path, content=content)
        elif kind == "run":
            await self._ce.execute(code=arg, language="python")
        elif kind == "shell":
            await self._bash.execute(command=arg)

    def reset_memory(self):
        self.short_term_memory = ShortTermMemory()

    def close(self):
        self.closed = True


def _make_repl(tmp_path, *, mode=PermissionMode.ASK, actions=None, answer="done",
               console=None, strategy=""):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    args = SimpleNamespace(
        workspace=str(ws), model="fake:model", provider=None,
        plan_only=False, auto_edit=False, assume_yes=False,
        temperature=None, max_tokens=None, max_iterations=None,
        quiet=False, verbose=False,
    )
    cli = _main.CLIInterface()
    cli.console = console  # None: the plain-text path
    repl = CodeREPL(cli, args)
    repl.model_id = "fake:model"
    repl.mode = mode
    repl.mode_explicit = True

    engine = CodeEngine(
        model="fake:model", workspace=ws, mode=mode, interactive=True,
        on_event=repl._on_event, on_diff=repl._on_diff,
    )
    engine._agent = _FakeAgent(
        build_code_tools(engine.gate), actions=actions or [], answer=answer,
        strategy=strategy,
    )
    repl.engine = engine
    return repl, ws


def _capture(repl, fn, *a, **k):
    """Run *fn* capturing stdout/stderr, returning ``(stdout, stderr)``."""
    import sys

    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        fn(*a, **k)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return out.getvalue(), err.getvalue()


# -- slash dispatch ---------------------------------------------------------

def test_every_slash_command_has_help_and_completes(tmp_path):
    repl, _ = _make_repl(tmp_path)
    # Every command carries a one-line description for /help and tab-completion.
    assert all(desc for desc in _SLASH_COMMANDS.values())
    # The dispatcher has a handler for each documented command.
    handled = []
    for cmd in _SLASH_COMMANDS:
        if cmd in ("/exit",):
            assert repl._dispatch(cmd) == "exit"
        else:
            # Commands that need an argument just print usage; none raise.
            _capture(repl, lambda c=cmd: handled.append(repl._dispatch(c)))
    assert all(h == "handled" for h in handled)


def test_unknown_command_is_reported_with_hint(tmp_path):
    repl, _ = _make_repl(tmp_path)
    _out, err = _capture(repl, repl._dispatch, "/contxt")
    assert "Unknown command" in err
    assert "/context" in err  # close-match hint


def test_bare_message_is_not_a_command(tmp_path):
    repl, _ = _make_repl(tmp_path)
    assert repl._dispatch("write a fibonacci function") is None


# -- files in context -------------------------------------------------------

def test_add_drop_and_context_estimate(tmp_path):
    repl, ws = _make_repl(tmp_path)
    (ws / "app.py").write_text("x = 1\n" * 10, encoding="utf-8")

    _capture(repl, repl._dispatch, "/add app.py")
    assert repl.context_files == ["app.py"]

    out, err = _capture(repl, repl._dispatch, "/context")
    text = out + err
    assert "app.py" in text
    assert "tok" in text  # a token estimate is shown

    _capture(repl, repl._dispatch, "/drop app.py")
    assert repl.context_files == []


def test_add_refuses_outside_workspace(tmp_path):
    repl, ws = _make_repl(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    _out, err = _capture(repl, repl._dispatch, "/add ../secret.txt")
    assert "outside the workspace" in err
    assert repl.context_files == []


def test_context_is_injected_into_the_task(tmp_path):
    repl, ws = _make_repl(tmp_path)
    (ws / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    repl.context_files = ["app.py"]
    composed = repl._compose_task("bump VALUE")
    assert "VALUE = 7" in composed
    assert "app.py" in composed
    assert composed.strip().endswith("bump VALUE")


# -- plan -> diff -> apply / reject / undo ----------------------------------

def test_plan_stages_then_apply_writes(tmp_path):
    repl, ws = _make_repl(
        tmp_path, mode=PermissionMode.ASK,
        actions=[("write", ("fib.py", "def fib(n):\n    return n\n"))],
    )
    # /plan runs in plan posture: nothing is written, the edit is staged.
    out, err = _capture(repl, repl._dispatch, "/plan write fib.py")
    assert set(repl.pending_edits) == {"fib.py"}
    assert not (ws / "fib.py").exists()
    # The note names the mode the turn ran under, not the session's mode.
    assert "Plan mode:" in out + err
    assert "in ask mode" not in out + err
    # The session mode is restored after the plan turn.
    assert repl.engine.gate.mode is PermissionMode.ASK

    # /apply writes the staged edit and clears it; /undo can reverse it.
    _capture(repl, repl._dispatch, "/apply")
    assert (ws / "fib.py").read_text() == "def fib(n):\n    return n\n"
    assert repl.pending_edits == {}

    _capture(repl, repl._dispatch, "/undo")
    assert not (ws / "fib.py").exists()


def test_reject_discards_without_writing(tmp_path):
    repl, ws = _make_repl(
        tmp_path,
        actions=[("write", ("a.py", "print('a')\n"))],
    )
    _capture(repl, repl._dispatch, "/plan add a.py")
    assert "a.py" in repl.pending_edits
    _capture(repl, repl._dispatch, "/reject")
    assert repl.pending_edits == {}
    assert not (ws / "a.py").exists()


def test_apply_by_number_applies_one(tmp_path):
    repl, ws = _make_repl(
        tmp_path,
        actions=[
            ("write", ("a.py", "a\n")),
            ("write", ("b.py", "b\n")),
        ],
    )
    _capture(repl, repl._dispatch, "/plan two files")
    assert set(repl.pending_edits) == {"a.py", "b.py"}
    _capture(repl, repl._dispatch, "/apply 1")
    # Only the first staged edit was written; the other stays pending.
    assert (ws / "a.py").exists()
    assert not (ws / "b.py").exists()
    assert set(repl.pending_edits) == {"b.py"}


def test_apply_with_nothing_staged_is_reported(tmp_path):
    repl, _ = _make_repl(tmp_path)
    out, err = _capture(repl, repl._dispatch, "/apply")
    assert "Nothing is staged" in (out + err)


# -- mode -------------------------------------------------------------------

def test_mode_change_updates_gate(tmp_path):
    repl, _ = _make_repl(tmp_path, mode=PermissionMode.ASK)
    _capture(repl, repl._dispatch, "/mode auto-edit")
    assert repl.mode is PermissionMode.AUTO_EDIT
    assert repl.engine.gate.mode is PermissionMode.AUTO_EDIT

    _out, err = _capture(repl, repl._dispatch, "/mode bogus")
    assert "Unknown mode" in err


# -- a normal turn writes inline in auto-edit -------------------------------

def test_turn_writes_inline_in_auto_edit(tmp_path):
    repl, ws = _make_repl(
        tmp_path, mode=PermissionMode.AUTO_EDIT,
        actions=[("write", ("main.py", "print('hi')\n"))],
        answer="Wrote main.py.",
    )
    _capture(repl, repl._do_turn, "make main.py")
    assert (ws / "main.py").read_text() == "print('hi')\n"
    assert repl.turns == 1
    assert repl.last_result is not None
    assert repl.last_result.files_written == ["main.py"]


def test_session_names_the_tool_calling_path_once(tmp_path):
    """The path a session's tool calls travel on is stated once, not per turn."""
    repl, _ws = _make_repl(
        tmp_path, mode=PermissionMode.AUTO_EDIT, answer="done", strategy="hybrid",
    )
    _out, err = _capture(repl, repl._do_turn, "first")
    assert "Tool calling: hybrid" in err
    _out2, err2 = _capture(repl, repl._do_turn, "second")
    assert "Tool calling" not in err2


# -- session save / load round-trip -----------------------------------------

def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    repl, ws = _make_repl(tmp_path)
    (ws / "app.py").write_text("x = 1\n", encoding="utf-8")
    repl.context_files = ["app.py"]
    repl.agent.short_term_memory.add_user_message("hello")
    repl.agent.short_term_memory.add_assistant_message("hi there")

    _capture(repl, repl._dispatch, "/save mysession")

    # A fresh REPL loads the snapshot: conversation + files-in-context restored.
    repl2, _ = _make_repl(tmp_path)
    _capture(repl2, repl2._dispatch, "/load mysession")
    assert repl2.context_files == ["app.py"]
    roles = [
        (m.role, m.content) for m in repl2.agent.short_term_memory.get_messages()
    ]
    assert (MessageRole.USER, "hello") in roles
    assert (MessageRole.ASSISTANT, "hi there") in roles


def test_clear_resets_live_context(tmp_path):
    repl, ws = _make_repl(tmp_path)
    (ws / "app.py").write_text("x\n", encoding="utf-8")
    repl.context_files = ["app.py"]
    repl.pending_edits = {"a.py": object()}
    repl.agent.short_term_memory.add_user_message("something")
    _capture(repl, repl._dispatch, "/clear")
    assert repl.context_files == []
    assert repl.pending_edits == {}
    assert repl.agent.short_term_memory.get_messages() == []


# -- /run feeds shell output back -------------------------------------------

def test_run_shows_stdout_and_feeds_it_back(tmp_path):
    repl, ws = _make_repl(tmp_path)
    (ws / "hello.txt").write_text("hi\n", encoding="utf-8")
    out, err = _capture(repl, repl._dispatch, "/run ls -1")
    text = out + err
    assert "hello.txt" in text
    # The raw result dict is not shown; only the command's output.
    assert "return_code" not in text
    # The run is added to the conversation so the next turn can react to it.
    msgs = [m.content for m in repl.agent.short_term_memory.get_messages()]
    assert any("ls -1" in m and "hello.txt" in m for m in msgs)


def test_blocked_command_reports_the_reason(tmp_path):
    """A command the shell tool blocks says why, and is logged as a failed action."""
    repl, _ws = _make_repl(tmp_path)

    out, err = _capture(repl, repl._dispatch, "/run rm -rf /")

    text = out + err
    assert "blocked" in text.lower()
    assert "(no output)" not in text
    last = repl.engine.gate.actions[-1]
    assert (last.kind, last.decision, last.outcome) == ("shell", "allowed", "error")


def test_shell_output_text_reduces_dict_and_refusal():
    assert CodeREPL._shell_output_text(
        SimpleNamespace(output={"stdout": "a\n", "stderr": "", "return_code": 0})
    ) == "a"
    assert (
        CodeREPL._shell_output_text(
            SimpleNamespace(output={"success": False, "error": "blocked: rm -rf"})
        )
        == "blocked: rm -rf"
    )


# -- output is never read as console markup ---------------------------------

def _rich_repl(tmp_path):
    """A REPL rendering through a real console, captured in a buffer."""
    from rich.console import Console

    buf = io.StringIO()
    console = Console(file=buf, width=100, force_terminal=False, highlight=False)
    repl, ws = _make_repl(tmp_path, console=console)
    repl.interactive = True
    repl.cli._human_to_stderr = False  # human output goes to the captured console
    return repl, ws, buf


def test_command_output_with_brackets_is_printed_verbatim(tmp_path):
    """A command whose output looks like markup prints as-is and does not raise."""
    repl, ws, buf = _rich_repl(tmp_path)
    (ws / "weird.txt").write_text(
        "path is [/usr/bin] and list [1, 2]\nroot:x:0:0:root:/root\n", encoding="utf-8"
    )

    repl._dispatch("/run cat weird.txt")

    out = buf.getvalue()
    assert "path is [/usr/bin] and list [1, 2]" in out
    assert "root:x:0:0:root:/root" in out  # not rewritten into an emoji


def test_help_keeps_usage_brackets(tmp_path):
    """The help table shows '/apply [n]' rather than dropping the argument."""
    repl, _ws, buf = _rich_repl(tmp_path)
    repl._color = True

    repl._cmd_help()

    text = buf.getvalue()
    assert "/apply [n]" in text
    assert "[name|number]" in text


def test_help_keeps_usage_brackets_without_a_console(tmp_path):
    repl, _ = _make_repl(tmp_path)
    out, err = _capture(repl, repl._cmd_help)
    assert "/apply [n]" in (out + err)


# -- token estimate ---------------------------------------------------------

def test_estimate_tokens():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") >= 1


# -- the launch decision (REPL vs single-shot) ------------------------------

def test_piped_stdin_does_not_launch_the_repl(monkeypatch, tmp_path):
    """A piped invocation runs the single-shot path, never the interactive REPL."""
    from effgen.cli.commands import code as code_cmd

    launched = {"repl": False}

    class _Boom(CodeREPL):
        def run(self):
            launched["repl"] = True
            return 0

    monkeypatch.setattr("effgen.cli.code.repl.CodeREPL", _Boom)

    ws = tmp_path / "ws"
    ws.mkdir()
    args = SimpleNamespace(
        task=None, print_task=None, model="fake:model", provider=None,
        workspace=str(ws), plan_only=False, auto_edit=False, assume_yes=False,
        undo=False, max_iterations=None, temperature=None, max_tokens=None,
        output_json=False, quiet=True,
    )
    cli = _main.CLIInterface()
    cli.console = None


    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    # Piped stdin: not a TTY -> single-shot, and here empty -> the "needs a task"
    # error path, but crucially the REPL is never launched.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code_cmd.run_code_command(cli, args)
    assert launched["repl"] is False


# ---------------------------------------------------------------------------
# /git is framed like every other read-only view
# ---------------------------------------------------------------------------
#
# Every other read-only view opens with a header — "Files in context",
# "Session cost", "Last turn — reasoning trace", "Coding tools". /git printed
# the bare porcelain line inside a repository, and git's own lower-case stderr
# outside one, with no hint that 'staged' and 'commit' exist.


class TestGitViewIsFramed:
    def test_a_repository_view_has_a_header_and_names_the_other_operations(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        repl, ws = _make_repl(tmp_path)
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        out, err = _capture(repl, repl._cmd_git, "status")
        text = " ".join((out + err).split())
        assert "Git — status" in text
        assert "staged" in text and "commit" in text

    def test_outside_a_repository_the_session_speaks_not_git(self, tmp_path):
        repl, ws = _make_repl(tmp_path)
        (ws / "app.py").write_text("x = 1\n")
        out, err = _capture(repl, repl._cmd_git, "status")
        # Wrapped by the console, so compare on normalized whitespace.
        text = " ".join((out + err).split())
        # git's own sentence is replaced by one naming the workspace and a way out.
        assert "fatal: not a git repository" not in text
        assert "is not a git repository" in text
        assert "git init" in text

    def test_a_genuine_git_failure_keeps_gits_own_text(self):
        """The friendlier wording is for the common case only.

        When git itself is the problem its message is what the reader needs, so
        the match is on the sentence rather than on an empty stderr — which is
        what made the fallback unreachable.
        """
        from effgen.cli.code.repl_session import _is_not_a_repository

        assert _is_not_a_repository(
            "fatal: not a git repository (or any of the parent directories): .git"
        )
        assert not _is_not_a_repository("fatal: bad revision 'HEAD'")
        assert not _is_not_a_repository("")


# ---------------------------------------------------------------------------
# /compact summarizes; it does not act
# ---------------------------------------------------------------------------
#
# /compact is described as "Summarize the conversation so far". It handed its
# prompt to the same agent in AgentMode.AUTO, which drives the whole tool loop,
# so the model was free to call file_operations, code_executor, python_repl and
# bash while it was supposed to be writing a summary — and on a real pty session
# it did exactly that, writing app.py and leaving it empty. The permission gate
# was not re-entered for the new turn either, so the writes travelled on
# whatever mode the session was already in.


class TestCompactCannotModifyTheWorkspace:
    def test_the_summarization_turn_runs_with_no_tools(self, tmp_path):
        repl, ws = _make_repl(tmp_path, answer="a summary")
        agent = repl.engine._agent
        assert agent.tools, "the session's agent normally holds the coding tools"

        seen: list = []
        original = agent.run

        def record(prompt, **kwargs):
            seen.append(dict(agent.tools))
            return original(prompt, **kwargs)

        agent.run = record
        repl.agent.short_term_memory.add_user_message("write app.py")
        repl.agent.short_term_memory.add_assistant_message("done")

        _capture(repl, repl._cmd_compact, "")

        assert seen, "the summarization turn never ran"
        assert seen[0] == {}, "the summarization turn still had tools attached"

    def test_the_tools_come_back_after_the_summary(self, tmp_path):
        repl, ws = _make_repl(tmp_path, answer="a summary")
        agent = repl.engine._agent
        before = dict(agent.tools)
        repl.agent.short_term_memory.add_user_message("write app.py")
        repl.agent.short_term_memory.add_assistant_message("done")

        _capture(repl, repl._cmd_compact, "")

        assert dict(agent.tools) == before

    def test_the_tools_come_back_even_when_the_turn_raises(self, tmp_path):
        repl, ws = _make_repl(tmp_path)
        agent = repl.engine._agent
        before = dict(agent.tools)

        def boom(prompt, **kwargs):
            raise RuntimeError("provider is down")

        agent.run = boom
        repl.agent.short_term_memory.add_user_message("write app.py")
        repl.agent.short_term_memory.add_assistant_message("done")

        _capture(repl, repl._cmd_compact, "")

        assert dict(agent.tools) == before
