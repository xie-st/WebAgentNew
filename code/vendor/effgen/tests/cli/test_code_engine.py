"""The ``effgen code`` engine and command, driven by a fake agent.

The plan/run/fix loop itself is proven with real models; here the engine wiring
(workspace resolution, the run-result record, the permission mode threaded into
the tools) and the command's output contract (byte-clean stdout when piped, a
JSON document under ``--json``, the withheld-changes exit code) are pinned with a
fake agent so the checks are deterministic and offline.

The fake stands in for the LLM: when it "runs" it drives the gated tools exactly
as the real agent would, so the confinement and permission behavior under test is
the shipped behavior, not a reimplementation.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from types import SimpleNamespace

from effgen.cli.code.engine import CodeEngine, resolve_workspace
from effgen.cli.code.permissions import PermissionMode
from effgen.cli.code.tools import build_code_tools
from effgen.tools.builtin._fs import WORKSPACE_ENV_VAR


class _FakeResponse:
    """Duck-typed like ``AgentResponse`` for the fields the engine reads."""

    def __init__(self, output, success=True, reason="final_answer", **meta):
        self.output = output
        self.success = success
        self.metadata = {"reason": reason, **meta}
        self.iterations = 1
        self.tool_calls = 1
        self.tokens_used = 100
        self.execution_time = 0.5
        self.provider = "openai"


class _FakeAgent:
    """An agent whose ``run`` drives the gated tools, then returns a response."""

    def __init__(self, tools, *, actions, answer="done", success=True,
                 reason="final_answer", strategy=""):
        # Tools are the engine's gated instances, in preset order.
        self._ce, self._repl, self._fops, self._bash = tools
        self._actions = actions
        self._answer = answer
        self._success = success
        self._reason = reason
        self._strategy = strategy
        self.closed = False

    def run(self, task):
        for action in self._actions:
            asyncio.run(self._apply(action))
        meta = {"tool_calling_strategy": self._strategy} if self._strategy else {}
        return _FakeResponse(
            self._answer, success=self._success, reason=self._reason, **meta,
        )

    async def _apply(self, action):
        kind, arg = action
        if kind == "write":
            path, content = arg
            await self._fops.execute(operation="write", path=path, content=content)
        elif kind == "run":
            await self._ce.execute(code=arg, language="python")
        elif kind == "shell":
            await self._bash.execute(command=arg)

    def close(self):
        self.closed = True


def _engine(tmp_path, mode, actions, *, mode_explicit=False, interactive=False, **kw):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    engine = CodeEngine(
        model="fake:model", workspace=ws, mode=mode,
        mode_explicit=mode_explicit, interactive=interactive,
    )
    engine._agent = _FakeAgent(
        build_code_tools(engine.gate), actions=actions, **kw,
    )
    return engine, ws


# -- workspace resolution ---------------------------------------------------

def test_resolve_workspace_prefers_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    assert resolve_workspace(str(explicit)) == explicit.resolve()
    assert explicit.exists()


def test_resolve_workspace_falls_back_to_env(tmp_path, monkeypatch):
    env_ws = tmp_path / "env"
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(env_ws))
    assert resolve_workspace(None) == env_ws.resolve()


# -- the run record ---------------------------------------------------------

def test_auto_edit_writes_and_records(tmp_path):
    engine, ws = _engine(
        tmp_path, PermissionMode.AUTO_EDIT,
        actions=[("write", ("main.py", "print('hi')\n")), ("run", "print(1)")],
    )
    result = engine.run("build it")
    assert result.success is True
    assert result.files_written == ["main.py"]
    assert (ws / "main.py").read_text() == "print('hi')\n"
    assert result.permission_mode == "auto-edit"
    assert not result.withheld

    payload = result.to_dict()
    assert payload["files_written"] == ["main.py"]
    assert payload["workspace"] == str(ws)
    assert {a["kind"] for a in payload["actions"]} == {"write", "run"}


def test_run_record_names_the_tool_calling_path(tmp_path):
    """The record says which path the turn's tool calls travelled on."""
    engine, ws = _engine(
        tmp_path, PermissionMode.AUTO_EDIT, actions=[], strategy="hybrid",
    )
    result = engine.run("build it")
    assert result.tool_calling == "hybrid"
    assert result.to_dict()["tool_calling"] == "hybrid"


def test_run_record_tool_calling_is_empty_without_one(tmp_path):
    engine, _ws = _engine(tmp_path, PermissionMode.AUTO_EDIT, actions=[])
    assert engine.run("build it").tool_calling == ""


def test_workspace_env_is_set_during_run_and_restored(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    seen = {}

    engine, ws = _engine(tmp_path, PermissionMode.PLAN, actions=[])

    original_run = engine._agent.run

    def _capture(task):
        seen["workspace"] = os.environ.get(WORKSPACE_ENV_VAR)
        return original_run(task)

    engine._agent.run = _capture
    engine.run("noop")
    assert seen["workspace"] == str(ws)
    # Restored afterward (was unset before the run).
    assert WORKSPACE_ENV_VAR not in os.environ


def test_plan_mode_withholds_every_change(tmp_path):
    engine, ws = _engine(
        tmp_path, PermissionMode.PLAN, mode_explicit=True,
        actions=[("write", ("a.py", "x")), ("shell", "ls")],
    )
    result = engine.run("plan only")
    assert result.files_written == []
    assert not (ws / "a.py").exists()
    assert {a.kind for a in result.withheld} == {"write", "shell"}


def test_escape_attempt_is_refused(tmp_path):
    engine, ws = _engine(
        tmp_path, PermissionMode.YES,
        actions=[("write", ("../escape.py", "x")), ("write", ("ok.py", "y"))],
    )
    result = engine.run("try to escape")
    assert result.files_written == ["ok.py"]
    assert not (ws.parent / "escape.py").exists()
    assert len(result.refused) == 1


# -- the command output contract --------------------------------------------

def _run_command(monkeypatch, tmp_path, args_extra, actions, **agent_kw):
    """Invoke ``run_code_command`` with a fake engine and captured streams."""
    from effgen.cli import _main
    from effgen.cli.commands import code as code_cmd

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)

    class _PatchedEngine(CodeEngine):
        def build_agent(self):
            self._agent = _FakeAgent(
                build_code_tools(self.gate), actions=actions, **agent_kw,
            )
            return self._agent

    monkeypatch.setattr(code_cmd, "CodeEngine", _PatchedEngine)
    # Skip the sandbox probe (offline).
    monkeypatch.setattr(code_cmd, "workspace_execution_note", lambda _ws: None)

    args = SimpleNamespace(
        task=None, print_task="do it", model="fake:model", provider=None,
        workspace=str(ws), plan_only=False, auto_edit=False, assume_yes=False,
        max_iterations=None, temperature=None, max_tokens=None,
        output_json=False, quiet=False,
    )
    for key, value in args_extra.items():
        setattr(args, key, value)

    cli = _main.CLIInterface()
    cli.console = None  # exercise the plain-text path deterministically

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    # Force the non-interactive branch regardless of the test runner's TTY.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code = code_cmd.run_code_command(cli, args)
    return code, out.getvalue(), err.getvalue()


def test_piped_stdout_is_answer_only(monkeypatch, tmp_path):
    exit_code, stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True},
        actions=[("write", ("main.py", "print('hi')\n"))],
        answer="The file was written and printed hi.",
    )
    assert exit_code == 0
    # stdout carries the answer and nothing else — no summary, no file list.
    assert stdout.strip() == "The file was written and printed hi."
    assert "Files written" in stderr  # the chatter went to stderr


def test_json_mode_emits_one_document(monkeypatch, tmp_path):
    exit_code, stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True, "output_json": True},
        actions=[("write", ("main.py", "x = 1\n"))],
        answer="ok",
    )
    assert exit_code == 0
    payload = json.loads(stdout)
    assert payload["answer"] == "ok"
    assert payload["files_written"] == ["main.py"]
    assert payload["permission_mode"] == "auto-edit"
    assert payload["success"] is True


def test_piped_without_optin_withholds_and_exits_two(monkeypatch, tmp_path):
    # No permission flag + no terminal -> plan default -> writes withheld.
    exit_code, stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {},
        actions=[("write", ("main.py", "x = 1\n"))],
        answer="I would write main.py.",
    )
    assert exit_code == 2  # completed, but changes were withheld
    assert stdout.strip() == "I would write main.py."
    assert not (tmp_path / "ws" / "main.py").exists()
    assert "No changes were made" in stderr


def test_partial_withholding_says_what_was_carried_out(monkeypatch, tmp_path):
    # --auto-edit applies the write but still gates the shell command: the note
    # must not claim nothing happened, and must name the flag that would allow
    # what was withheld.
    exit_code, _stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True},
        actions=[("write", ("main.py", "x = 1\n")), ("shell", "ls")],
        answer="wrote the file; the command needs confirmation.",
    )
    assert exit_code == 0
    assert (tmp_path / "ws" / "main.py").exists()
    assert "Some actions were not carried out" in stderr
    assert "--yes" in stderr
    assert "No changes were made" not in stderr


def test_conflicting_permission_flags_error(monkeypatch, tmp_path):
    exit_code, stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True, "assume_yes": True},
        actions=[],
    )
    assert exit_code == 1
    assert "cannot be combined" in stderr


def test_report_names_the_tool_calling_path(monkeypatch, tmp_path):
    """The run report says how the tool calls were meant to travel."""
    exit_code, stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True},
        actions=[],
        answer="nothing to do", strategy="react",
    )
    assert exit_code == 0
    assert stdout.strip() == "nothing to do"
    assert "Tool calling: react" in stderr
    assert "read out of the model's text" in stderr


def test_json_document_carries_the_tool_calling_path(monkeypatch, tmp_path):
    exit_code, stdout, _stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True, "output_json": True},
        actions=[],
        answer="ok", strategy="hybrid",
    )
    assert exit_code == 0
    assert json.loads(stdout)["tool_calling"] == "hybrid"


def test_failed_run_exits_one(monkeypatch, tmp_path):
    exit_code, stdout, stderr = _run_command(
        monkeypatch, tmp_path,
        {"auto_edit": True},
        actions=[],
        answer="model call failed", success=False, reason="generation_failed",
    )
    assert exit_code == 1
