"""Repository awareness and the confirmed commit behind ``effgen code``.

These run against a real git repository created in a temp directory — the point
is what git actually does, so nothing here is simulated: the inventory really
honors ``.gitignore``, the commit really uses the repository's configured
identity, and the pathspec-limited commit really leaves the user's own staged
work alone. What is *not* reachable is checked just as concretely: a push, a
reset, a force or an amend raises before any subprocess starts.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from effgen.cli import _main
from effgen.cli.code import git_actions as ga
from effgen.cli.code.engine import CodeEngine, CodeRunResult
from effgen.cli.code.permissions import PermissionMode
from effgen.cli.code.project import (
    DETAIL_THRESHOLD,
    MAX_BRIEF_CHARS,
    build_project_context,
    detect_repo,
    staged_diff,
)
from effgen.cli.commands.code import _offer_commit
from effgen.tools.builtin.devops import GitTool

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return proc.stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A small repository with one commit, an ignored file and a subdirectory."""
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Repo Owner")
    _git(root, "config", "user.email", "owner@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("build/\n*.log\n", encoding="utf-8")
    (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "lib.py").write_text("x = 1\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "out.o").write_text("binary\n", encoding="utf-8")
    (root / "debug.log").write_text("noise\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")
    return root


# -- repo detection & inventory ---------------------------------------------

def test_detects_the_repository_and_its_branch(repo: Path):
    info = detect_repo(repo)
    assert info is not None
    assert info.root == repo.resolve()
    assert info.branch  # a name, or "detached at <sha>"
    assert info.dirty is False


def test_detects_changes_by_kind(repo: Path):
    (repo / "app.py").write_text("print('bye')\n", encoding="utf-8")
    (repo / "fresh.py").write_text("y = 2\n", encoding="utf-8")
    _git(repo, "add", "fresh.py")
    info = detect_repo(repo)
    assert info is not None and info.dirty
    assert info.staged == 1 and info.unstaged == 1 and info.untracked == 0
    assert any("app.py" in line for line in info.status_lines)


def test_outside_a_repository_degrades_without_error(tmp_path: Path):
    plain = tmp_path / "plain"
    (plain / "pkg").mkdir(parents=True)
    (plain / "a.py").write_text("a = 1\n", encoding="utf-8")
    (plain / "pkg" / "b.py").write_text("b = 2\n", encoding="utf-8")
    (plain / "__pycache__").mkdir()
    (plain / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")

    context = build_project_context(plain)
    assert context.repo is None
    assert "No git repository" in context.summary_line()
    assert sorted(context.files) == ["a.py", "pkg/b.py"]  # cache dir skipped
    assert "Not a git repository" in context.as_prompt()


def test_inventory_respects_gitignore(repo: Path):
    context = build_project_context(repo)
    assert context.ignored_respected is True
    assert sorted(context.files) == [".gitignore", "app.py", "src/lib.py"]
    prompt = context.as_prompt()
    assert "build/out.o" not in prompt and "debug.log" not in prompt


def test_untracked_but_unignored_files_are_inventoried(repo: Path):
    (repo / "draft.py").write_text("d = 1\n", encoding="utf-8")
    context = build_project_context(repo)
    assert "draft.py" in context.files


@pytest.mark.parametrize("name", [".env", "id_rsa", "credentials"])
def test_credential_files_are_left_out_of_the_inventory(repo: Path, tmp_path: Path, name: str):
    # The write and read paths refuse these names, so neither the git listing
    # nor the plain walk offers one to the model.
    (repo / name).write_text("SECRET_TOKEN=xyz\n", encoding="utf-8")
    listed = build_project_context(repo)
    assert name not in listed.files
    assert not any(name in line for line in listed.layout_lines())

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / name).write_text("SECRET_TOKEN=xyz\n", encoding="utf-8")
    (plain / "keep.py").write_text("k = 1\n", encoding="utf-8")
    walked = build_project_context(plain)
    assert walked.files == ["keep.py"]


def test_a_repository_without_commits_reads_as_a_phrase(tmp_path: Path):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _git(fresh, "init", "-q")
    (fresh / "seed.py").write_text("s = 1\n", encoding="utf-8")
    info = detect_repo(fresh)
    assert info is not None and info.branch == "no commits yet"
    # Not "branch no commits yet" — the phrase stands on its own.
    assert info.branch_label == "no commits yet"
    summary = build_project_context(fresh).summary_line()
    assert "no commits yet" in summary and "branch no commits yet" not in summary


def test_a_named_branch_reads_as_a_branch(repo: Path):
    info = detect_repo(repo)
    assert info is not None
    assert info.branch_label == f"branch {info.branch}"
    assert f"branch {info.branch}" in build_project_context(repo).as_prompt()


def test_large_tree_is_summarized_by_directory(repo: Path):
    for i in range(DETAIL_THRESHOLD + 5):
        (repo / "src" / f"mod_{i}.py").write_text(f"n = {i}\n", encoding="utf-8")
    context = build_project_context(repo)
    lines = context.layout_lines()
    assert any(line.strip().startswith("src/ —") for line in lines)
    # Every file is accounted for by the per-directory counts.
    counted = sum(
        int(line.split("—")[1].split()[0]) for line in lines if "—" in line
    ) + len([line for line in lines if "—" not in line])
    assert counted == context.total_files
    assert len("\n".join(lines)) < 8000


def test_small_tree_lists_every_file_with_a_size(repo: Path):
    lines = build_project_context(repo).layout_lines()
    assert any("app.py (" in line and " B)" in line for line in lines)
    assert len(lines) == 3


# -- project brief ----------------------------------------------------------

def test_project_brief_is_read_and_bounded(repo: Path):
    (repo / "AGENTS.md").write_text("Use tabs. " * 2000, encoding="utf-8")
    context = build_project_context(repo)
    assert context.brief_path and context.brief_path.endswith("AGENTS.md")
    assert context.brief is not None
    assert len(context.brief) <= MAX_BRIEF_CHARS + 20
    assert "Use tabs." in context.as_prompt()


def test_no_brief_is_not_an_error(repo: Path):
    context = build_project_context(repo)
    assert context.brief_path is None and context.brief is None


# -- the context reaches the model ------------------------------------------

def test_project_context_is_part_of_the_system_prompt(repo: Path):
    engine = CodeEngine(
        model="fake:model", workspace=repo, mode=PermissionMode.PLAN,
        project=build_project_context(repo),
    )
    prompt = engine.system_prompt()
    assert "Git repository:" in prompt
    assert "app.py" in prompt
    assert str(repo) in prompt


def test_load_project_builds_and_refreshes(repo: Path):
    engine = CodeEngine(model="fake:model", workspace=repo, mode=PermissionMode.PLAN)
    first = engine.load_project()
    assert first.repo is not None and "app.py" in first.files
    (repo / "later.py").write_text("z = 0\n", encoding="utf-8")
    assert "later.py" not in engine.load_project().files  # cached
    assert "later.py" in engine.load_project(refresh=True).files


# -- the read-only surface ---------------------------------------------------

def test_staged_diff_reads_only_what_is_staged(repo: Path):
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    assert staged_diff(repo) == ""  # not staged yet
    _git(repo, "add", "app.py")
    patch = staged_diff(repo)
    assert "app.py" in patch and "+print('changed')" in patch


def test_git_tool_stays_read_only():
    # The coding surface reuses the read-only tool; it is not widened here. Its
    # operations report state (``git branch -a``, ``git remote -v``); none of
    # them writes to the repository.
    assert GitTool.ALLOWED == {"status", "log", "diff", "branch", "show", "remote"}
    writes = {"add", "commit", "push", "reset", "checkout", "clean", "tag", "stash"}
    assert not (GitTool.ALLOWED & writes)


# -- what is not reachable ---------------------------------------------------

@pytest.mark.parametrize(
    "args",
    [
        ["push"],
        ["push", "origin", "main"],
        ["reset", "--hard", "HEAD~1"],
        ["checkout", "."],
        ["restore", "app.py"],
        ["clean", "-fd"],
        ["tag", "v1"],
        ["stash"],
        ["rebase", "main"],
        ["branch", "-D", "main"],
        ["remote", "add", "origin", "http://example.invalid"],
        ["commit", "--amend", "-m", "rewrite"],
        ["commit", "-m", "skip hooks", "--no-verify"],
        ["add", "--force", "ignored.log"],
    ],
)
def test_destructive_and_publishing_commands_are_refused(args):
    with pytest.raises(ga.UnsafeGitAction):
        ga.ensure_safe(args)


def test_a_refused_command_never_runs(repo: Path, monkeypatch):
    ran: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: ran.append(cmd) or SimpleNamespace()
    )
    with pytest.raises(ga.UnsafeGitAction):
        ga.run_git(repo, ["push", "origin", "main"])
    assert ran == []


@pytest.mark.parametrize(
    "command",
    [
        "git push",
        "git push origin main --force",
        "git -C /elsewhere push origin main",
        "cd /tmp && git push",
        "make build; git push origin HEAD",
        "git reset --hard HEAD~1",
        "git checkout -- .",
        "git clean -fdx",
        "git stash",
        "git tag v1.0.0",
        "git rebase -i main",
        "git branch -D main",
        "git commit --amend -m 'rewrite'",
        "git commit -m ok --no-verify",
        "/usr/bin/git push",
        "git push  # unbalanced quote '",
        # A command that follows a readable one in the same line is still read.
        "git status; git push origin main",
        "git add -A; git commit -m 'wip'; git push",
        "git status --porcelain\ngit push origin main",
        "git status | tee out.txt; git push",
        "(cd sub; git push)",
        "xargs git push",
        # Another interpreter is not a way around the limit.
        "bash -c 'git push origin main'",
        "bash -lc 'git push'",
        "sh -c \"git reset --hard HEAD~1\"",
        "eval 'git push'",
        "python -c \"import subprocess; subprocess.run(['git', 'push'])\"",
        "python3 -c 'import os; os.system(\"git reset --hard\")'",
        "node -e \"require('child_process').execSync('git push')\"",
    ],
)
def test_the_shell_cannot_publish_or_rewrite_history(command):
    assert ga.unsafe_shell_git(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "git status --short",
        "git log --oneline -n 5",
        "git diff --cached",
        "git show HEAD",
        "git ls-files",
        "pytest -q",
        "echo 'git push origin main'",
        "grep -rn 'git reset' docs/",
        "git log --grep=push",
        "git -C sub status",
        "git status; git diff",
        "git status --porcelain\ngit log --oneline",
        "git log --oneline | head -5",
        "bash -c 'git status'",
        "python -c \"print('git is a program')\"",
        "echo \"run git push yourself\" > NOTES.md",
    ],
)
def test_reading_a_repository_from_the_shell_still_works(command):
    assert ga.unsafe_shell_git(command) is None


def test_the_gated_shell_refuses_a_publishing_command_in_every_mode(repo: Path):
    # --yes turns off the prompts; it does not turn on publishing. The refusal
    # is recorded, so the run reports it and the model reads it as an
    # observation rather than a silent no-op.
    import asyncio

    from effgen.cli.code.permissions import PermissionGate
    from effgen.cli.code.tools import build_code_tools

    for mode in PermissionMode:
        gate = PermissionGate(mode, repo, interactive=False)
        bash = next(t for t in build_code_tools(gate) if t.name == "bash")
        result = asyncio.run(bash._execute(command="git push origin master"))
        assert result["success"] is False
        assert "not available to a coding session" in result["error"]
        assert "/git commit" in result["error"]
        assert [a.decision for a in gate.actions] == ["refused"]

    # Reading the repository is untouched by the guard; the mode still decides.
    gate = PermissionGate(PermissionMode.YES, repo, interactive=False)
    bash = next(t for t in build_code_tools(gate) if t.name == "bash")
    allowed = asyncio.run(bash._execute(command="git status --short"))
    assert allowed.get("success") is not False
    assert [a.decision for a in gate.actions] == ["allowed"]


def test_a_destructive_command_really_does_not_run(repo: Path):
    import asyncio

    from effgen.cli.code.permissions import PermissionGate
    from effgen.cli.code.tools import build_code_tools

    head = _git(repo, "rev-parse", "HEAD").strip()
    gate = PermissionGate(PermissionMode.YES, repo, interactive=False)
    bash = next(t for t in build_code_tools(gate) if t.name == "bash")
    asyncio.run(bash._execute(command="git reset --hard HEAD~1"))
    assert _git(repo, "rev-parse", "HEAD").strip() == head


@pytest.mark.parametrize(
    "command",
    [
        "git status --porcelain; git push origin main",
        "git status --porcelain\ngit push origin main",
        "bash -c 'git push origin main'",
        "sh -c 'git reset --hard HEAD~1'",
        "eval 'git push origin main'",
    ],
)
def test_a_chained_or_nested_command_reaches_neither_the_remote_nor_history(
    repo: Path, tmp_path: Path, command: str
):
    # The refusal has to hold for the line the model actually writes, not only
    # for a bare `git push`: a second command on the same line and a command
    # handed to another interpreter are both read.
    import asyncio

    from effgen.cli.code.permissions import PermissionGate
    from effgen.cli.code.tools import build_code_tools

    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / "second.py").write_text("y = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    head = _git(repo, "rev-parse", "HEAD").strip()

    gate = PermissionGate(PermissionMode.YES, repo, interactive=False)
    bash = next(t for t in build_code_tools(gate) if t.name == "bash")
    result = asyncio.run(bash._execute(command=command))

    assert result["success"] is False
    assert "not available to a coding session" in result["error"]
    assert [a.decision for a in gate.actions] == ["refused"]
    assert _git(repo, "rev-parse", "HEAD").strip() == head
    assert _git(remote, "rev-list", "--all", "--count").strip() == "0"


def test_the_allowed_and_forbidden_sets_do_not_overlap():
    assert not (ga.ALLOWED_SUBCOMMANDS & ga.FORBIDDEN_SUBCOMMANDS)


def test_a_commit_message_may_contain_a_flag_shaped_word(repo: Path):
    # The message is data, not an option: only options are checked.
    ga.ensure_safe(["commit", "-m", "--force is documented", "--", "app.py"])


# -- the confirmed commit ----------------------------------------------------

def _engine(repo: Path, mode: PermissionMode, answers=None, interactive=False) -> CodeEngine:
    replies = iter(answers or [])
    return CodeEngine(
        model="fake:model",
        workspace=repo,
        mode=mode,
        interactive=interactive,
        confirm=(lambda _q: next(replies)) if answers else None,
        project=build_project_context(repo),
    )


def test_commit_needs_a_yes(repo: Path):
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.ASK, answers=["n"], interactive=True)
    plan = engine.plan_commit(["written.py"], task="add a module")
    outcome = engine.perform_commit(plan)
    assert outcome.success is False
    assert "declined" in outcome.detail
    assert _git(repo, "log", "--format=%s").splitlines() == ["initial"]
    assert [a.decision for a in engine.gate.actions] == ["declined"]


def test_commit_on_yes_uses_the_repository_identity(repo: Path):
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.ASK, answers=["y"], interactive=True)
    plan = engine.plan_commit(["written.py"], task="add a module")
    assert plan.subject == "Add written.py"
    outcome = engine.perform_commit(plan)
    assert outcome.success and outcome.commit
    assert _git(repo, "log", "-1", "--format=%an <%ae>").strip() == (
        "Repo Owner <owner@example.invalid>"
    )
    assert _git(repo, "log", "-1", "--format=%s").strip() == "Add written.py"
    assert _git(repo, "log", "-1", "--format=%b").strip() == "add a module"


def test_commit_covers_only_the_named_paths(repo: Path):
    # The user staged something of their own; the coding commit must not take it.
    (repo / "mine.py").write_text("mine = 1\n", encoding="utf-8")
    _git(repo, "add", "mine.py")
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")

    engine = _engine(repo, PermissionMode.YES)
    plan = engine.plan_commit(["written.py"], task="add a module")
    assert plan.other_staged == ["mine.py"]
    assert engine.perform_commit(plan).success

    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["written.py"]
    assert _git(repo, "diff", "--cached", "--name-only").split() == ["mine.py"]


def test_commit_message_carries_no_attribution(repo: Path):
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.YES)
    plan = engine.plan_commit(["written.py"], task="add a module")
    assert engine.perform_commit(plan).success
    body = _git(repo, "log", "-1", "--format=%B%n%an%n%ae").lower()
    for token in ("co-authored-by", "generated with", "assistant", "agent", "effgen"):
        assert token not in body


def test_commit_without_an_identity_reports_it(repo: Path, monkeypatch, tmp_path: Path):
    # No identity anywhere: the repository's own settings are removed and the
    # global/system files are pointed at an empty file for this process.
    _git(repo, "config", "--unset", "user.email")
    _git(repo, "config", "--unset", "user.name")
    empty = tmp_path / "empty.gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    for var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(var, raising=False)
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.YES)
    outcome = engine.perform_commit(engine.plan_commit(["written.py"]))
    assert outcome.success is False
    assert "git config user.email" in outcome.detail
    assert _git(repo, "log", "--format=%s").splitlines() == ["initial"]


def test_outside_a_repository_there_is_nothing_to_commit(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.py").write_text("a = 1\n", encoding="utf-8")
    engine = CodeEngine(
        model="fake:model", workspace=plain, mode=PermissionMode.YES,
        project=build_project_context(plain),
    )
    plan = engine.plan_commit(["a.py"])
    assert "not inside a git repository" in plan.error
    assert engine.perform_commit(plan).success is False


def test_nothing_written_means_nothing_to_commit(repo: Path):
    engine = _engine(repo, PermissionMode.YES)
    plan = engine.plan_commit([])
    assert "nothing to commit" in plan.error
    assert engine.perform_commit(plan).success is False


# -- the non-interactive contract -------------------------------------------

def _run_result(repo: Path, files: list[str]) -> CodeRunResult:
    return CodeRunResult(
        task="add a module", answer="done", success=True, reason="final_answer",
        model="fake:model", provider=None, workspace=str(repo),
        permission_mode="plan", files_written=files,
    )


def _capture(fn, *a, **k):
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        fn(*a, **k)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return out.getvalue(), err.getvalue()


def _cli():
    cli = _main.CLIInterface()
    cli.console = None
    cli._human_to_stderr = True
    return cli


def test_piped_run_withholds_the_commit_without_an_opt_in(repo: Path):
    # No terminal and no --yes: the default mode is plan, so the commit is
    # proposed and withheld rather than made on an implied yes.
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.PLAN)
    result = _run_result(repo, ["written.py"])
    args = SimpleNamespace(commit=True, commit_message=None)
    _, err = _capture(_offer_commit, _cli(), args, engine, result, quiet=False)

    assert result.commit is not None and result.commit["success"] is False
    assert "Nothing was committed" in err
    assert _git(repo, "log", "--format=%s").splitlines() == ["initial"]


def test_the_commit_record_has_one_shape(repo: Path, tmp_path: Path):
    # Whether there was something to commit or not, ``--json`` carries the same
    # keys, so a script can read .commit.success without probing for the rest.
    keys = {"success", "commit", "message", "detail", "paths"}

    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.YES)
    done = _run_result(repo, ["written.py"])
    _capture(_offer_commit, _cli(), SimpleNamespace(commit=True, commit_message=None),
             engine, done, quiet=False)
    assert done.commit is not None and set(done.commit) == keys
    assert done.commit["success"] is True

    plain = tmp_path / "plain"
    plain.mkdir()
    outside = CodeEngine(
        model="fake:model", workspace=plain, mode=PermissionMode.YES,
        project=build_project_context(plain),
    )
    nothing = _run_result(plain, [])
    _capture(_offer_commit, _cli(), SimpleNamespace(commit=True, commit_message=None),
             outside, nothing, quiet=False)
    assert nothing.commit is not None and set(nothing.commit) == keys
    assert nothing.commit["success"] is False
    assert "not inside a git repository" in nothing.commit["detail"]


def test_piped_run_commits_with_yes(repo: Path):
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")
    engine = _engine(repo, PermissionMode.YES)
    result = _run_result(repo, ["written.py"])
    args = SimpleNamespace(commit=True, commit_message="Add the module")
    _capture(_offer_commit, _cli(), args, engine, result, quiet=False)

    assert result.commit is not None and result.commit["success"] is True
    assert _git(repo, "log", "-1", "--format=%s").strip() == "Add the module"
    # The decision is on the run's action log, so --json reports it.
    assert any(a.kind == "git" for a in result.actions)
    assert result.to_dict()["commit"]["paths"] == ["written.py"]


def _run_code_command(monkeypatch, repo: Path, args_extra: dict):
    """Drive ``run_code_command`` over *repo* with an agent that writes one file."""
    import asyncio

    from effgen.cli.code.tools import build_code_tools
    from effgen.cli.commands import code as code_cmd

    class _Agent:
        def __init__(self, tools):
            self._fops = tools[2]

        def run(self, task):
            asyncio.run(self._fops.execute(
                operation="write", path="written.py", content="w = 1\n"
            ))
            return SimpleNamespace(
                output="wrote it", success=True, metadata={"reason": "final_answer"},
                iterations=1, tool_calls=1, tokens_used=10, execution_time=0.1,
                provider="fake",
            )

        def close(self):
            pass

    class _PatchedEngine(CodeEngine):
        def build_agent(self):
            self._agent = _Agent(build_code_tools(self.gate))
            return self._agent

    monkeypatch.setattr(code_cmd, "CodeEngine", _PatchedEngine)
    monkeypatch.setattr(code_cmd, "workspace_execution_note", lambda _ws: None)

    args = SimpleNamespace(
        task=None, print_task="write it", model="fake:model", provider=None,
        workspace=str(repo), plan_only=False, auto_edit=False, assume_yes=False,
        max_iterations=None, temperature=None, max_tokens=None,
        output_json=False, quiet=False, commit=False, commit_message=None,
    )
    for key, value in args_extra.items():
        setattr(args, key, value)

    cli = _cli()
    cli.console = None
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    exit_code = code_cmd.run_code_command(cli, args)
    return exit_code, out.getvalue(), err.getvalue()


def test_a_withheld_commit_is_reported_in_the_exit_code(repo: Path, monkeypatch):
    # --auto-edit writes without asking but does not confirm a commit, and there
    # is no terminal to ask on: the run did less than it was told to, so it exits
    # 2 rather than reporting a clean success.
    exit_code, _out, err = _run_code_command(
        monkeypatch, repo, {"auto_edit": True, "commit": True}
    )
    assert exit_code == 2
    assert (repo / "written.py").exists()
    assert "Nothing was committed" in err
    assert _git(repo, "log", "--format=%s").splitlines() == ["initial"]


def test_a_commit_that_was_made_exits_zero(repo: Path, monkeypatch):
    exit_code, _out, _err = _run_code_command(
        monkeypatch, repo, {"assume_yes": True, "commit": True}
    )
    assert exit_code == 0
    assert _git(repo, "log", "-1", "--format=%s").strip() == "Add written.py"


# -- the interactive surface -------------------------------------------------

class _StubAgent:
    """Enough agent for the slash commands that do not call a model."""

    def __init__(self) -> None:
        from effgen.memory.short_term import ShortTermMemory

        self.tools: dict = {}
        self.short_term_memory = ShortTermMemory()
        self.config = SimpleNamespace(system_prompt="")
        self.session = None

    def close(self) -> None:
        pass


def _make_repl(repo: Path, mode=PermissionMode.ASK, answers=None):
    from effgen.cli.code.repl import CodeREPL

    args = SimpleNamespace(
        workspace=str(repo), model="fake:model", provider=None,
        plan_only=False, auto_edit=False, assume_yes=False,
        temperature=None, max_tokens=None, max_iterations=None,
        quiet=False, verbose=False,
    )
    cli = _cli()
    repl = CodeREPL(cli, args)
    repl.model_id = "fake:model"
    repl.mode, repl.mode_explicit = mode, True
    engine = _engine(repo, mode, answers=answers, interactive=bool(answers))
    engine._agent = _StubAgent()
    repl.engine = engine
    return repl


def test_repl_banner_names_the_repository(repo: Path):
    repl = _make_repl(repo)
    _, err = _capture(repl._banner)
    assert "Repo: proj" in err and "branch" in err


def test_repl_git_shows_the_read_only_surface(repo: Path):
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    repl = _make_repl(repo)
    _, status = _capture(repl._cmd_git, "status")
    assert "app.py" in status
    _, log = _capture(repl._cmd_git, "log")
    assert "initial" in log
    _, unknown = _capture(repl._cmd_git, "push")
    assert "not available" in unknown and "commit" in unknown


def test_repl_git_staged_pulls_the_patch_into_context(repo: Path):
    repl = _make_repl(repo)
    _, nothing = _capture(repl._cmd_git, "staged")
    assert "Nothing is staged" in nothing

    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _, shown = _capture(repl._cmd_git, "staged")
    assert "+print('changed')" in shown
    history = repl.agent.short_term_memory.get_messages()
    assert any("git diff --cached" in m.content for m in history)


def test_repl_commit_asks_before_it_commits(repo: Path):
    (repo / "written.py").write_text("w = 1\n", encoding="utf-8")

    declined = _make_repl(repo, answers=["n"])
    declined._note_written(["written.py"])
    _capture(declined._cmd_git, "commit")
    assert _git(repo, "log", "--format=%s").splitlines() == ["initial"]
    assert declined.session_files == ["written.py"]

    accepted = _make_repl(repo, answers=["y"])
    accepted._note_written(["written.py"])
    _capture(accepted._cmd_git, "commit my message")
    assert _git(repo, "log", "-1", "--format=%s").strip() == "my message"
    assert accepted.session_files == []


def test_repl_commit_needs_something_written(repo: Path):
    repl = _make_repl(repo)
    _, err = _capture(repl._cmd_git, "commit")
    assert "nothing to commit" in err
    assert _git(repo, "log", "--format=%s").splitlines() == ["initial"]


def test_a_written_file_reaches_the_next_turn(repo: Path):
    # A file the last turn created must appear in the layout the next turn is
    # given, or the prompt's "do not assume a path that is not listed exists"
    # points at a stale listing.
    repl = _make_repl(repo)
    assert "made.py" not in repl.engine.system_prompt()
    (repo / "made.py").write_text("m = 1\n", encoding="utf-8")
    repl._note_written(["made.py"])
    assert "made.py" in repl.project.files
    assert "made.py" in repl.engine.system_prompt()
    assert "made.py" in repl.agent.config.system_prompt


def test_repl_context_refresh_rereads_the_project(repo: Path):
    repl = _make_repl(repo)
    _, before = _capture(repl._cmd_context, "")
    assert "Repo: proj" in before
    (repo / "later.py").write_text("z = 0\n", encoding="utf-8")
    _capture(repl._cmd_context, "refresh")
    assert "later.py" in repl.project.files
    assert "later.py" in repl.engine.system_prompt()
    # The live agent's prompt is updated too, so the next turn sees the new file.
    assert "later.py" in repl.agent.config.system_prompt


def test_run_result_reports_the_repository(repo: Path):
    engine = _engine(repo, PermissionMode.PLAN)
    response = SimpleNamespace(
        output="ok", success=True, metadata={"reason": "final_answer"},
        iterations=1, tool_calls=0, tokens_used=10, execution_time=0.1, provider="fake",
    )
    result = engine.result_from_response("task", response)
    assert result.repo is not None
    assert result.repo["root"] == str(repo.resolve())
    assert result.to_dict()["repo"]["branch"] == detect_repo(repo).branch
