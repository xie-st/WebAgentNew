"""The permission gate and gated tools behind ``effgen code``.

These pin the policy table (which action kinds each mode allows), workspace
confinement (an out-of-workspace or credentials-named target is refused before
the tool runs), and the fact that a gated tool returns a normal tool failure the
agent can read rather than raising. No live model calls: the tools are driven
directly with a fixed workspace so the checks are deterministic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from effgen.cli.code.permissions import (
    ACTION_KINDS,
    PermissionGate,
    PermissionMode,
    default_mode,
)
from effgen.cli.code.tools import build_code_tools
from effgen.tools.builtin._fs import PathNotAllowedError


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _run(coro):
    return asyncio.run(coro)


# -- policy table -----------------------------------------------------------

@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            PermissionMode.PLAN,
            {"write": False, "run": False, "shell": False, "git": False},
        ),
        (
            PermissionMode.ASK,
            {"write": False, "run": False, "shell": False, "git": False},
        ),
        (
            PermissionMode.AUTO_EDIT,
            {"write": True, "run": True, "shell": False, "git": False},
        ),
        (
            PermissionMode.YES,
            {"write": True, "run": True, "shell": True, "git": True},
        ),
    ],
)
def test_policy_without_a_terminal(mode, expected, workspace):
    # No terminal to confirm on: a `confirm` policy becomes a withheld action.
    gate = PermissionGate(mode, workspace, interactive=False)
    for kind in ACTION_KINDS:
        decision = gate.request(kind, f"do {kind}")
        assert decision.allowed is expected[kind], (mode, kind)


def test_ask_confirms_when_interactive(workspace):
    answers = iter(["y", "n", "a"])
    gate = PermissionGate(
        PermissionMode.ASK, workspace, interactive=True,
        confirm=lambda _q: next(answers),
    )
    assert gate.request("write", "w1").allowed is True   # y
    assert gate.request("write", "w2").allowed is False  # n
    assert gate.request("write", "w3").allowed is True   # a -> allow
    # "always" sticks for the rest of the run without another prompt.
    assert gate.request("write", "w4").allowed is True


def test_default_mode_follows_terminal():
    assert default_mode(interactive=True) is PermissionMode.ASK
    assert default_mode(interactive=False) is PermissionMode.PLAN


# -- confinement ------------------------------------------------------------

def test_resolve_path_confines_to_workspace(workspace):
    gate = PermissionGate(PermissionMode.YES, workspace)
    inside = gate.resolve_path("sub/a.py")
    assert str(inside).startswith(str(workspace.resolve()))

    with pytest.raises(PathNotAllowedError):
        gate.resolve_path("/etc/passwd")
    with pytest.raises(PathNotAllowedError):
        gate.resolve_path("../escape.py")
    with pytest.raises(PathNotAllowedError):
        gate.resolve_path(".env")


def test_gated_write_refuses_escape_and_records_it(workspace):
    gate = PermissionGate(PermissionMode.YES, workspace, interactive=False)
    _, _, fops, _ = build_code_tools(gate)

    ok = _run(fops.execute(operation="write", path="in.py", content="x = 1\n"))
    assert ok.success is True
    assert (workspace / "in.py").read_text() == "x = 1\n"

    escaped = _run(fops.execute(operation="write", path="../out.py", content="x"))
    assert escaped.success is False
    assert "outside the workspace" in (escaped.error or "")
    assert not (workspace.parent / "out.py").exists()

    assert gate.files_written == ["in.py"]
    assert len(gate.refused) == 1


def test_gated_write_refuses_credentials_name(workspace):
    gate = PermissionGate(PermissionMode.YES, workspace, interactive=False)
    _, _, fops, _ = build_code_tools(gate)
    result = _run(fops.execute(operation="write", path=".env", content="K=V"))
    assert result.success is False
    assert "credentials" in (result.error or "")


# -- gated refusals are readable, not raised --------------------------------

def test_plan_mode_withholds_writes_as_tool_failures(workspace):
    gate = PermissionGate(PermissionMode.PLAN, workspace, interactive=False)
    ce, repl, fops, bash = build_code_tools(gate)

    write = _run(fops.execute(operation="write", path="a.py", content="x"))
    run = _run(ce.execute(code="print(1)", language="python"))
    shell = _run(bash.execute(command="echo hi"))

    for result in (write, run, shell):
        assert result.success is False
        assert "plan mode" in (result.error or "").lower()
    assert not (workspace / "a.py").exists()
    # Every attempt is recorded so the caller can report what was withheld.
    assert {a.kind for a in gate.withheld} == {"write", "run", "shell"}


def test_reads_are_never_gated(workspace):
    # A read must work even in plan mode so the agent can see what it edits.
    (workspace / "seen.py").write_text("print('hi')\n")
    gate = PermissionGate(PermissionMode.PLAN, workspace, interactive=False)
    _, _, fops, _ = build_code_tools(gate)
    result = _run(fops.execute(operation="read", path="seen.py"))
    assert result.success is True
    assert not gate.actions  # a read is not a decided action


def test_bash_language_on_code_executor_is_shelled(workspace):
    # code_executor(language="bash") is a shell command, gated as shell:
    # allowed under --yes, withheld under --auto-edit.
    for mode, allowed in ((PermissionMode.YES, True), (PermissionMode.AUTO_EDIT, False)):
        gate = PermissionGate(mode, workspace, interactive=False)
        ce, *_ = build_code_tools(gate)
        result = _run(ce.execute(code="echo hi", language="bash"))
        assert result.success is allowed, mode
        assert gate.actions[-1].kind == "shell"


def test_a_permitted_action_that_failed_does_not_tick_as_a_success(workspace, capsys):
    # The glyph is the only signal when there is no color: an action that ran
    # and failed reads as a failure, not as a completed step.
    from effgen.cli import _main
    from effgen.cli.code.permissions import ActionRecord
    from effgen.cli.code.render import print_action

    cli = _main.CLIInterface()
    cli.console = None

    done = ActionRecord(kind="run", summary="Run python: print(1)", decision="allowed",
                        reason="", target="python_repl")
    done.outcome = "ok"
    failed = ActionRecord(kind="run", summary="Run python: import runpy", decision="allowed",
                          reason="", target="python_repl")
    failed.outcome, failed.detail = "error", "Import of 'runpy' is not allowed"

    print_action(cli, done)
    print_action(cli, failed)
    ok_line, failed_line = capsys.readouterr().out.strip().splitlines()

    assert "failed" not in ok_line
    assert "failed: Import of 'runpy' is not allowed" in failed_line
    # The two lines do not open with the same mark.
    assert ok_line.strip()[0] != failed_line.strip()[0]
