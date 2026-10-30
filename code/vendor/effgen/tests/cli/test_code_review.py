"""``effgen code --review``: a run that reads and cannot write.

Three separable things are pinned here:

* the **subject** — what a review was asked to look at, assembled from git and
  from named files, with a typed error for every way of naming nothing;
* the **tool set** — a review holds no tool that writes a file, runs code or
  runs a command, and a write that arrives anyway is refused and recorded;
* the **command** — flag resolution, the exit codes, and the additive keys in
  the JSON document.

No model is called: everything here is decided before the first request.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from effgen.cli import _main
from effgen.cli.code.engine import CodeEngine, CodeRunResult
from effgen.cli.code.permissions import PermissionGate, PermissionMode
from effgen.cli.code.project import (
    ReviewSubject,
    review_subject,
    verify_rev,
    working_diff,
)
from effgen.cli.code.tools import (
    ReadOnlyFileOperations,
    WorkspaceGitTool,
    build_code_tools,
    build_review_tools,
)
from effgen.cli.commands import code as code_cmd

BEFORE = "def add(a, b):\n    return a + b\n"
AFTER = "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n"


def _git(repo, *args):
    return subprocess.run(  # noqa: S603 - fixed argv in a temp repo
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path):
    """A repository with one commit, one staged change and one unstaged change."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", ".")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@example.com")
    (root / "calc.py").write_text(BEFORE, encoding="utf-8")
    _git(root, "add", "calc.py")
    _git(root, "commit", "-q", "-m", "initial")
    (root / "calc.py").write_text(AFTER, encoding="utf-8")
    _git(root, "add", "calc.py")
    (root / "notes.txt").write_text("a note\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# The subject
# --------------------------------------------------------------------------

def test_subject_staged_reads_the_index(repo):
    subject = review_subject(repo, "staged")
    assert subject.error == ""
    assert subject.kind == "diff"
    assert subject.ref == "staged"
    assert "def divide" in subject.text
    assert subject.line_count > 0


def test_subject_uncommitted_covers_staged_and_unstaged(repo):
    (repo / "calc.py").write_text(AFTER + "\n# trailing\n", encoding="utf-8")
    subject = review_subject(repo, "uncommitted")
    assert subject.error == ""
    assert "def divide" in subject.text
    assert "# trailing" in subject.text


def test_subject_accepts_a_revision(repo):
    subject = review_subject(repo, "HEAD")
    assert subject.error == ""
    assert subject.ref == "HEAD"
    assert "def divide" in subject.text


def test_subject_accepts_a_range(repo):
    _git(repo, "branch", "other")          # branches at the first commit
    _git(repo, "commit", "-q", "-m", "second")
    subject = review_subject(repo, "other...HEAD")
    assert subject.error == "", subject.error
    assert "def divide" in subject.text


def test_subject_reports_a_revision_git_rejects(repo):
    subject = review_subject(repo, "no-such-rev")
    assert "no-such-rev" in subject.error
    assert "main...HEAD" in subject.error
    assert subject.text == ""


def test_verify_rev_accepts_what_working_diff_reads(repo):
    assert verify_rev(repo, "HEAD") is None
    assert verify_rev(repo, "no-such-rev") is not None
    assert "def divide" in working_diff(repo, ref="HEAD")


def test_subject_outside_a_repository_says_how_to_name_one(tmp_path):
    subject = review_subject(tmp_path, "uncommitted")
    assert "not inside a git repository" in subject.error
    assert "-f/--file" in subject.error


def test_subject_reports_an_empty_diff_rather_than_reviewing_nothing(repo):
    _git(repo, "commit", "-q", "-m", "second")
    (repo / "notes.txt").unlink()
    subject = review_subject(repo, "staged")
    assert "nothing is staged" in subject.error
    assert "-f/--file" in subject.error


def test_subject_takes_a_file_set_without_a_diff(repo):
    subject = review_subject(repo, None, ["calc.py"])
    assert subject.error == ""
    assert subject.kind == "files"
    assert subject.files == ["calc.py"]
    assert "def divide" in subject.text


def test_subject_takes_a_file_set_alongside_a_diff(repo):
    subject = review_subject(repo, "staged", ["notes.txt"])
    assert subject.error == ""
    assert subject.kind == "diff+files"
    assert "a note" in subject.text
    assert "def divide" in subject.text


def test_subject_refuses_a_file_outside_the_workspace(repo):
    subject = review_subject(repo, None, ["../escape.py"])
    assert "outside the workspace" in subject.error


def test_subject_names_a_file_it_cannot_read(repo):
    subject = review_subject(repo, None, ["nope.py"])
    assert "nope.py" in subject.error


def test_subject_refuses_a_credentials_filename(repo):
    (repo / ".env").write_text("KEY=value\n", encoding="utf-8")
    subject = review_subject(repo, None, [".env"])
    assert "credentials" in subject.error


def test_subject_marks_and_counts_the_cut(repo):
    subject = review_subject(repo, "staged", max_chars=80)
    assert subject.truncated_at == 80
    assert "truncated at 80 characters" in subject.text
    assert "not shown" in subject.text
    assert "truncated" in subject.describe()


def test_subject_to_dict_omits_the_body(repo):
    payload = review_subject(repo, "staged").to_dict()
    assert set(payload) == {"kind", "ref", "files", "line_count", "truncated_at"}


# --------------------------------------------------------------------------
# The tool set
# --------------------------------------------------------------------------

def _gate(path):
    return PermissionGate(PermissionMode.PLAN, path, interactive=False)


def test_the_review_set_holds_nothing_that_writes_or_runs(tmp_path):
    tools = build_review_tools(_gate(tmp_path))
    assert [t.metadata.name for t in tools] == ["file_operations", "git"]
    coding = {type(t).__name__ for t in build_code_tools(_gate(tmp_path))}
    assert coding & {type(t).__name__ for t in tools} == set()


def test_the_review_file_tool_offers_no_writing_operation(tmp_path):
    tool = ReadOnlyFileOperations(_gate(tmp_path))
    operation = next(p for p in tool.metadata.parameters if p.name == "operation")
    assert operation.enum == ["read", "list", "search", "metadata"]
    assert {p.name for p in tool.metadata.parameters} & {"content", "target_format"} == set()
    assert "cannot write" in tool.metadata.description


def test_the_shipped_file_tool_metadata_is_untouched(tmp_path):
    ReadOnlyFileOperations(_gate(tmp_path))
    from effgen.tools.builtin.file_ops import FileOperations

    operation = next(
        p for p in FileOperations(allowed_directories=[str(tmp_path)]).metadata.parameters
        if p.name == "operation"
    )
    assert operation.enum == ["read", "write", "list", "search", "metadata", "convert"]


@pytest.mark.parametrize("operation", ["write", "save", "create", "convert"])
def test_a_forced_write_is_refused_and_recorded(tmp_path, monkeypatch, operation):
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path))
    gate = _gate(tmp_path)
    tool = ReadOnlyFileOperations(gate)
    result = asyncio.run(tool.execute(operation=operation, path="x.py", content="hi"))
    assert result.success is False
    assert "read-only review" in (result.error or "")
    assert [a.decision for a in gate.actions] == ["refused"]
    assert not (tmp_path / "x.py").exists()


def test_a_read_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path))
    (tmp_path / "calc.py").write_text(BEFORE, encoding="utf-8")
    tool = ReadOnlyFileOperations(_gate(tmp_path))
    result = asyncio.run(tool.execute(operation="read", path="calc.py"))
    assert result.success is True
    assert "def add" in str(result.output)


def test_the_review_file_tool_declares_itself_retrieved_context(tmp_path):
    assert ReadOnlyFileOperations(_gate(tmp_path)).is_context_retrieval is True
    from effgen.cli.code.tools import GatedFileOperations

    assert GatedFileOperations(_gate(tmp_path)).is_context_retrieval is False


def test_every_review_tool_declares_itself_retrieved_context(tmp_path):
    """Both halves of the set read; asking either one twice must not end the turn.

    A review's second move is as often "what changed here?" as "show me that
    file again", so a repeated ``git`` call has to earn the same tool-free turn
    a repeated read does — otherwise the repository report is handed back as the
    review.
    """
    tools = build_review_tools(_gate(tmp_path))
    assert [t.metadata.name for t in tools if t.is_context_retrieval] == [
        "file_operations", "git",
    ]
    from effgen.tools.builtin.devops import GitTool

    assert GitTool().is_context_retrieval is False


def test_git_is_pinned_to_the_workspace(repo, tmp_path):
    gate = _gate(repo)
    tool = WorkspaceGitTool(gate)
    inside = asyncio.run(tool.execute(operation="status"))
    assert inside.success is True
    outside = asyncio.run(tool.execute(operation="status", cwd=str(tmp_path)))
    assert outside.success is False
    assert "outside it" in (outside.error or "")


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

def _engine(repo, subject=None):
    return CodeEngine(
        model="fake:model", workspace=repo, mode=PermissionMode.PLAN, review=subject
    )


def test_a_review_engine_is_read_only_and_swaps_its_prompt(repo):
    plain = _engine(repo)
    reviewing = _engine(repo, review_subject(repo, "staged"))
    assert plain.read_only is False
    assert reviewing.read_only is True
    assert "use the file tool's write operation" in plain.system_prompt()
    assert "read-only" in reviewing.system_prompt()
    assert "write operation" not in reviewing.system_prompt()


def test_the_subject_travels_with_the_task(repo):
    engine = _engine(repo, review_subject(repo, "staged"))
    composed = engine.compose_review_task("what breaks here?")
    assert composed.startswith("what breaks here?")
    assert "def divide" in composed
    # A run that is not reviewing composes nothing.
    assert _engine(repo).compose_review_task("x") == "x"


def test_the_review_turn_restores_the_tools_even_when_it_raises(repo):
    engine = _engine(repo)
    engine._agent = SimpleNamespace(
        tools={"bash": object(), "file_operations": object()},
        config=SimpleNamespace(system_prompt="original"),
    )
    before = dict(engine._agent.tools)
    with pytest.raises(RuntimeError):
        with engine.review_turn(review_subject(repo, "staged")):
            assert set(engine._agent.tools) == {"file_operations", "git"}
            assert engine.read_only is True
            raise RuntimeError("the turn failed")
    assert engine._agent.tools == before
    assert engine._agent.config.system_prompt == "original"
    assert engine.read_only is False
    assert engine.gate.mode is PermissionMode.PLAN


def test_the_run_record_carries_the_review_keys(repo):
    engine = _engine(repo, review_subject(repo, "staged"))
    response = SimpleNamespace(
        output="a review", success=True, metadata={"reason": "final_answer"},
        iterations=1, tool_calls=1, tokens_used=5, execution_time=0.1,
    )
    payload = engine.result_from_response("review it", response).to_dict()
    assert payload["read_only"] is True
    assert payload["review"]["ref"] == "staged"
    assert payload["answer_source"] == ""
    assert payload["files_written"] == []
    assert payload["permission_mode"] == "plan"


@pytest.mark.parametrize(
    "source,recovered",
    [("loop_detected", True), ("repeated_tool_result", True),
     ("null_final_from_model", True), ("direct_calculator_result", False), ("", False)],
)
def test_a_recovered_answer_is_labelled(source, recovered):
    result = CodeRunResult(
        task="t", answer="a", success=True, reason="final_answer", model="m",
        provider=None, workspace=".", permission_mode="plan", answer_source=source,
    )
    assert result.recovered_answer is recovered
    from effgen.cli.code.render import recovered_answer_note

    note = recovered_answer_note(result)
    assert bool(note) is recovered
    if recovered:
        assert "did not write an answer" in note


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------

def _args(**over):
    base = {
        "task": None, "print_task": "review it", "model": "fake:model",
        "provider": None, "workspace": None, "plan_only": False,
        "auto_edit": False, "assume_yes": False, "undo": False, "undo_count": 1,
        "max_iterations": None, "temperature": None, "max_tokens": None,
        "output_json": False, "quiet": True, "commit": False,
        "commit_message": None, "review": None, "review_files": None,
        "session_id": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "flag,attr",
    [("--plan", "plan_only"), ("--auto-edit", "auto_edit"), ("--yes", "assume_yes"),
     ("--commit", "commit"), ("--undo", "undo")],
)
def test_review_cannot_be_combined_with(flag, attr):
    target, _files, error = code_cmd._resolve_review(_args(review="staged", **{attr: True}))
    assert target is None
    assert flag in error
    assert "cannot be combined" in error


def test_review_alone_resolves():
    target, files, error = code_cmd._resolve_review(_args(review="staged"))
    assert (target, files, error) == ("staged", [], None)


def test_no_review_flag_resolves_to_nothing():
    assert code_cmd._resolve_review(_args())[0] is None


def _run(args):
    cli = _main.CLIInterface()
    cli.console = None
    return code_cmd.run_code_command(cli, args)


def _flat(text: str) -> str:
    """Collapse the console's line wrapping so a phrase can be matched."""
    return " ".join(text.split())


def test_a_review_outside_a_repository_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = _run(_args(review="uncommitted", workspace=str(tmp_path), quiet=False))
    assert rc == 1
    assert "not inside a git repository" in _flat(capsys.readouterr().err)


def test_a_bad_revision_exits_one_before_any_model_call(repo, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = _run(_args(review="no-such-rev", workspace=str(repo), quiet=False))
    assert rc == 1
    assert "no-such-rev" in _flat(capsys.readouterr().err)


def test_a_conflicting_flag_exits_one(repo, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = _run(_args(review="staged", assume_yes=True, workspace=str(repo)))
    assert rc == 1
    captured = capsys.readouterr()
    # A piped run keeps stdout for the result; an argument error is not one.
    assert captured.out == ""
    assert "cannot be combined" in _flat(captured.err)


def test_the_json_document_carries_the_review_block(repo, monkeypatch, capsys):
    """The command's own JSON shape, with the engine's run stubbed out."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    def _fake_build(self):
        self._agent = SimpleNamespace(model=None, close=lambda: None, session=None)
        return self._agent

    def _fake_run(self, task):
        return self.result_from_response(task, SimpleNamespace(
            output="a review", success=True, metadata={"reason": "final_answer"},
            iterations=1, tool_calls=1, tokens_used=5, execution_time=0.1,
        ))

    monkeypatch.setattr(CodeEngine, "build_agent", _fake_build)
    monkeypatch.setattr(CodeEngine, "run", _fake_run)
    rc = _run(_args(review="staged", workspace=str(repo), output_json=True))
    assert rc == 0
    document = json.loads(capsys.readouterr().out)
    assert document["read_only"] is True
    assert document["review"]["ref"] == "staged"
    assert document["files_written"] == []
    assert document["coding_suitability"]["verdict"]
    assert isinstance(ReviewSubject().to_dict(), dict)


# --------------------------------------------------------------------------
# ``/review`` in the session
# --------------------------------------------------------------------------

def test_review_is_documented_routed_and_completed(repo):
    from effgen.cli.code.repl import _SLASH_COMMANDS
    from effgen.cli.code.repl_commands import CodeCommandsMixin

    assert "/review" in _SLASH_COMMANDS
    assert _SLASH_COMMANDS["/review"].strip()
    assert "/review" in inspect.getsource(CodeCommandsMixin._dispatch)


def test_a_review_turn_swaps_the_tools_and_puts_them_back(repo, monkeypatch):
    """The live proof is a pty session; this pins the swap and the restore."""
    from effgen.cli.code.repl import CodeREPL

    engine = CodeEngine(model="fake:model", workspace=repo, mode=PermissionMode.ASK)
    engine._agent = SimpleNamespace(
        tools={"bash": object(), "file_operations": object()},
        config=SimpleNamespace(system_prompt="original"),
    )
    args = SimpleNamespace(
        workspace=str(repo), model="fake:model", provider=None, plan_only=False,
        auto_edit=False, assume_yes=False, temperature=None, max_tokens=None,
        max_iterations=None, quiet=True, verbose=False, session_id=None,
    )
    cli = _main.CLIInterface()
    cli.console = None
    repl = CodeREPL(cli, args)
    repl.engine = engine
    repl.model_id = "fake:model"

    seen = {}

    def _turn(message):
        seen["tools"] = set(engine._agent.tools)
        seen["mode"] = engine.gate.mode
        seen["task"] = repl._compose_task(message)

    monkeypatch.setattr(type(repl), "_do_turn", staticmethod(_turn))
    repl._cmd_review("staged")

    assert seen["tools"] == {"file_operations", "git"}
    assert seen["mode"] is PermissionMode.PLAN
    assert "def divide" in seen["task"]
    assert set(engine._agent.tools) == {"bash", "file_operations"}
    assert engine.gate.mode is PermissionMode.ASK
    assert engine.read_only is False


def test_a_review_turn_with_nothing_to_review_says_so(tmp_path, monkeypatch):
    from effgen.cli.code.repl import CodeREPL

    args = SimpleNamespace(
        workspace=str(tmp_path), model="fake:model", provider=None, plan_only=False,
        auto_edit=False, assume_yes=False, temperature=None, max_tokens=None,
        max_iterations=None, quiet=True, verbose=False, session_id=None,
    )
    cli = _main.CLIInterface()
    cli.console = None
    repl = CodeREPL(cli, args)
    repl.engine = CodeEngine(
        model="fake:model", workspace=tmp_path, mode=PermissionMode.ASK
    )
    ran = []
    monkeypatch.setattr(type(repl), "_do_turn", staticmethod(lambda m: ran.append(m)))
    repl._cmd_review("")
    assert ran == []


# --------------------------------------------------------------------------
# A review that produced no review
# --------------------------------------------------------------------------

FILE_READ_BACK = "def add(a, b):\n    return a + b\n"


def _recovered_response(source="loop_detected"):
    """The response the loop produces when it stops holding a tool result.

    The run has what a tool returned and no answer, so it reports the stop and
    carries the text as progress — which is what these assertions read.
    """
    from effgen.core.agent import AgentResponse

    return AgentResponse(
        output=f"'m' did not write an answer: the run stopped ({source}).",
        success=False,
        metadata={
            "reason": source, "answer_source": source, "partial": True,
            "partial_output": FILE_READ_BACK,
            "error": {"type": "UnsynthesizedToolResult", "category": source,
                      "message": "stopped", "retryable": False},
        },
        iterations=3, tool_calls=1, tokens_used=99, execution_time=0.2,
    )


@pytest.mark.parametrize("source", ["loop_detected", "repeated_tool_result"])
def test_a_review_never_answers_with_the_file_it_read(repo, source):
    """The loop can only hand back what it read; that is not a review."""
    engine = _engine(repo, review_subject(repo, "staged"))
    result = engine.result_from_response("review it", _recovered_response(source))

    assert result.success is False
    assert result.outcome == "stopped"
    assert result.partial is True
    assert result.answer != FILE_READ_BACK
    assert result.partial_output.startswith("def add")
    assert result.reason == source
    assert result.stop_reason == source


def test_a_run_that_is_not_a_review_reports_the_stop_the_same_way(repo):
    """A writing run reads the same: the file it read is progress, not an answer."""
    result = _engine(repo).result_from_response("do it", _recovered_response())

    assert result.success is False
    assert result.outcome == "stopped"
    assert result.answer != FILE_READ_BACK
    assert result.partial_output.startswith("def add")
    assert result.answer_source == "loop_detected"
    assert result.recovered_answer is True


def test_a_review_that_the_model_answered_is_untouched(repo):
    engine = _engine(repo, review_subject(repo, "staged"))
    result = engine.result_from_response("review it", SimpleNamespace(
        output="calc.py:6 swallows every exception", success=True,
        metadata={"reason": "final_answer"}, iterations=2, tool_calls=1,
        tokens_used=10, execution_time=0.1,
    ))
    assert result.success is True
    assert result.answer_source == ""
    assert result.partial is False
