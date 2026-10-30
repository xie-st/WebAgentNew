"""The edit preview, apply/reject and undo flow of ``effgen code``.

These drive the gated file tool and the command directly (no model calls): a
write is previewed as a diff before it lands, ``ask`` mode can accept one file
and reject another, ``plan`` mode writes nothing, ``--json`` carries the diffs,
and ``--undo`` reverses the last applied edit.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from effgen.cli.code.edits import EditJournal, ProposedEdit
from effgen.cli.code.engine import CodeEngine, undo_workspace
from effgen.cli.code.permissions import PermissionGate, PermissionMode
from effgen.cli.code.tools import build_code_tools


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _run(coro):
    return asyncio.run(coro)


# -- preview + apply/reject -------------------------------------------------

def test_write_is_announced_as_a_diff_before_it_lands(workspace):
    seen: list[ProposedEdit] = []
    gate = PermissionGate(
        PermissionMode.AUTO_EDIT, workspace, interactive=False,
        on_diff=seen.append, journal=EditJournal(workspace, root=workspace.parent / "h"),
    )
    _, _, fops, _ = build_code_tools(gate)
    result = _run(fops.execute(operation="write", path="a.py", content="x = 1\n"))
    assert result.success is True
    # The diff was surfaced before the write, and it names the new file.
    assert len(seen) == 1
    assert seen[0].rel_path == "a.py"
    assert seen[0].is_new is True
    assert "+x = 1" in seen[0].diff_text
    # And the write actually happened.
    assert (workspace / "a.py").read_text() == "x = 1\n"
    assert gate.edits[0]["path"] == "a.py"
    assert gate.edits[0]["is_new"] is True


def test_ask_mode_accepts_one_file_and_rejects_another(workspace):
    # Confirm says yes to the first write, no to the second.
    answers = iter(["y", "n"])
    gate = PermissionGate(
        PermissionMode.ASK, workspace, interactive=True,
        confirm=lambda _q: next(answers),
        journal=EditJournal(workspace, root=workspace.parent / "h"),
    )
    _, _, fops, _ = build_code_tools(gate)

    a = _run(fops.execute(operation="write", path="a.py", content="A\n"))
    b = _run(fops.execute(operation="write", path="b.py", content="B\n"))

    assert a.success is True
    assert b.success is False
    # Only the accepted file is on disk; the rejected one never was.
    assert (workspace / "a.py").read_text() == "A\n"
    assert not (workspace / "b.py").exists()
    assert gate.files_written == ["a.py"]
    # The rejection is recorded so the caller can report it.
    declined = [x for x in gate.actions if x.decision == "declined"]
    assert [d.target for d in declined] == ["b.py"]
    # Both proposals are reported; only the accepted one is marked applied.
    assert [(e["path"], e["applied"]) for e in gate.edits] == [
        ("a.py", True), ("b.py", False),
    ]


def test_plan_mode_shows_the_diff_but_writes_nothing(workspace):
    seen: list[ProposedEdit] = []
    gate = PermissionGate(
        PermissionMode.PLAN, workspace, interactive=False, on_diff=seen.append,
        journal=EditJournal(workspace, root=workspace.parent / "h"),
    )
    _, _, fops, _ = build_code_tools(gate)
    result = _run(fops.execute(operation="write", path="a.py", content="x\n"))
    assert result.success is False
    assert "plan mode" in (result.error or "").lower()
    # The proposal was still shown as a diff.
    assert len(seen) == 1
    # ...and reported as a proposal that did not reach disk, so --plan --json
    # (which prints no live diff) still carries the change it would make.
    assert [(e["path"], e["applied"]) for e in gate.edits] == [("a.py", False)]
    assert "+x" in gate.edits[0]["diff"]
    # Nothing was written and nothing recorded on the undo stack.
    assert not (workspace / "a.py").exists()
    assert len(EditJournal(workspace, root=workspace.parent / "h")) == 0


def test_edit_of_existing_file_snapshots_for_undo(workspace):
    (workspace / "a.py").write_text("v1\n")
    root = workspace.parent / "h"
    gate = PermissionGate(
        PermissionMode.YES, workspace, interactive=False,
        journal=EditJournal(workspace, root=root),
    )
    _, _, fops, _ = build_code_tools(gate)
    _run(fops.execute(operation="write", path="a.py", content="v2\n"))
    assert (workspace / "a.py").read_text() == "v2\n"

    outcome = EditJournal(workspace, root=root).undo()
    assert outcome.action == "restored"
    assert (workspace / "a.py").read_text() == "v1\n"


# -- the run result carries the diffs ---------------------------------------

class _FakeResponse:
    def __init__(self, output="done"):
        self.output = output
        self.success = True
        self.metadata = {"reason": "final_answer"}
        self.iterations = 1
        self.tool_calls = 1
        self.tokens_used = 10
        self.execution_time = 0.1
        self.provider = "openai"


class _FakeAgent:
    def __init__(self, tools, writes):
        self._fops = tools[2]
        self._writes = writes

    def run(self, task):
        for path, content in self._writes:
            asyncio.run(self._fops.execute(operation="write", path=path, content=content))
        return _FakeResponse()

    def close(self):
        pass


def test_result_diffs_and_json(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    engine = CodeEngine(
        model="fake:model", workspace=ws, mode=PermissionMode.AUTO_EDIT,
    )
    engine._agent = _FakeAgent(
        build_code_tools(engine.gate),
        writes=[("a.py", "print(1)\n"), ("b.py", "print(2)\n")],
    )
    result = engine.run("build")
    assert result.files_written == ["a.py", "b.py"]
    assert [d["path"] for d in result.diffs] == ["a.py", "b.py"]
    assert all(d["is_new"] for d in result.diffs)
    payload = result.to_dict()
    assert [d["path"] for d in payload["diffs"]] == ["a.py", "b.py"]
    assert all(d["applied"] for d in payload["diffs"])
    assert "+print(1)" in payload["diffs"][0]["diff"]


def test_plan_mode_json_carries_the_diff_it_did_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    engine = CodeEngine(model="fake:model", workspace=ws, mode=PermissionMode.PLAN)
    engine._agent = _FakeAgent(
        build_code_tools(engine.gate), writes=[("a.py", "print(1)\n")],
    )
    result = engine.run("build")

    assert result.files_written == []
    assert not (ws / "a.py").exists()
    payload = result.to_dict()
    assert [(d["path"], d["applied"]) for d in payload["diffs"]] == [("a.py", False)]
    assert "+print(1)" in payload["diffs"][0]["diff"]


# -- the command --undo path ------------------------------------------------

def test_undo_workspace_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    journal = EditJournal(ws)  # default root, honors EFFGEN_HOME
    (ws / "a.py").write_text("v2\n")
    journal.record_write("a.py", before="v1\n", after="v2\n")

    outcomes, remaining = undo_workspace(ws, 1)
    assert len(outcomes) == 1 and outcomes[0].action == "restored"
    assert remaining == 0
    assert (ws / "a.py").read_text() == "v1\n"


def test_command_undo_reverts_and_reports(tmp_path, monkeypatch):
    from effgen.cli import _main
    from effgen.cli.commands import code as code_cmd

    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("edited\n")
    EditJournal(ws).record_write("a.py", before="original\n", after="edited\n")

    args = SimpleNamespace(
        undo=True, undo_count=1, workspace=str(ws), output_json=False,
    )
    cli = _main.CLIInterface()
    cli.console = None

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    code = code_cmd.run_code_command(cli, args)
    assert code == 0
    assert (ws / "a.py").read_text() == "original\n"
    assert "Restored a.py" in err.getvalue()


def test_command_undo_json_and_nothing_to_undo(tmp_path, monkeypatch):
    from effgen.cli import _main
    from effgen.cli.commands import code as code_cmd

    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()

    args = SimpleNamespace(
        undo=True, undo_count=1, workspace=str(ws), output_json=True,
    )
    cli = _main.CLIInterface()
    cli.console = None
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    code = code_cmd.run_code_command(cli, args)
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["undone"] == []
    assert payload["remaining"] == 0


def test_workspace_write_extensions_cover_more_languages(workspace):
    # A coding agent can now create the source types it can already read.
    gate = PermissionGate(PermissionMode.YES, workspace, interactive=False)
    _, _, fops, _ = build_code_tools(gate)
    for name in ("main.go", "lib.rs", "App.tsx", "widget.c"):
        result = _run(fops.execute(operation="write", path=name, content="// x\n"))
        assert result.success is True, name
        assert (workspace / name).exists()
