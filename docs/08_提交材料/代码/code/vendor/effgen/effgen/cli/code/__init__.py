"""The coding-agent surface behind ``effgen code``.

``effgen code`` runs a plan → act → observe loop over the ``coding`` preset: the
model proposes an approach, writes files, executes code in the configured
sandbox, reads the real output, and iterates until the task is done or the
iteration cap is reached. Everything it can touch is confined to one workspace
root, and every write or command it wants to run passes a permission gate first.

The pieces:

- :mod:`~effgen.cli.code.permissions` — the permission modes (``plan``, ``ask``,
  ``auto-edit``, ``yes``), the gate that applies them, and the action log.
- :mod:`~effgen.cli.code.diffs` — unified-diff generation, colorized rendering,
  and hunk-level application for the edit preview.
- :mod:`~effgen.cli.code.edits` — a pending edit (:class:`ProposedEdit`) and the
  per-workspace undo journal (:class:`EditJournal`) behind ``--undo``.
- :mod:`~effgen.cli.code.tools` — the ``coding`` preset's tools with the gate
  wired in front of their write/execute paths. The tool names and schemas the
  model sees are unchanged, and a read-only review set holds nothing that
  writes, runs or executes.
- :mod:`~effgen.cli.code.project` — repository detection, the bounded workspace
  inventory (``.gitignore`` respected) the session starts from, and the diff or
  file set a review reads.
- :mod:`~effgen.cli.code.readiness` — the workspace/sandbox/git checks
  ``effgen doctor`` and the session's ``/doctor`` report.
- :mod:`~effgen.cli.code.git_actions` — the one repository-changing action a
  session offers: a confirmed commit of the files it wrote.
- :mod:`~effgen.cli.code.engine` — workspace resolution, agent construction, and
  the run-result record the CLI prints or serializes.
- :mod:`~effgen.cli.code.render` — the terminal rendering both surfaces share:
  a diff, a decided action, a run footer, and the verbatim line printer.
- :mod:`~effgen.cli.code.repl` — the interactive session (:class:`CodeREPL`) a
  terminal with no task opens, with its slash commands. The session is composed
  from four sibling modules, and stays the import path for every name they
  define:

  - :mod:`~effgen.cli.code.repl_commands` — the slash-command table, the
    dispatcher, and the commands that change the working set.
  - :mod:`~effgen.cli.code.repl_session` — the commands that retune, inspect or
    persist the session.
  - :mod:`~effgen.cli.code.repl_turn` — one turn, from the composed task to the
    notes printed after it.
  - :mod:`~effgen.cli.code.repl_view` — what the session prints around a turn.

The command handler lives in :mod:`effgen.cli.commands.code`.
"""

from __future__ import annotations

from .diffs import (
    ApplyResult,
    DiffStat,
    Hunk,
    apply_hunks,
    diff_stat,
    render_diff,
    split_hunks,
    unified_diff_text,
)
from .edits import AppliedEdit, EditJournal, ProposedEdit, UndoOutcome
from .engine import (
    CodeEngine,
    CodeRunResult,
    CommitPlan,
    resolve_workspace,
    undo_workspace,
    workspace_env,
    workspace_execution_note,
)
from .git_actions import (
    CommitOutcome,
    UnsafeGitAction,
    commit_paths,
    ensure_safe,
    suggest_message,
    unsafe_shell_git,
)
from .permissions import (
    ACTION_KINDS,
    ActionRecord,
    Decision,
    PermissionGate,
    PermissionMode,
    default_mode,
)
from .project import (
    ProjectContext,
    RepoInfo,
    ReviewSubject,
    build_project_context,
    detect_repo,
    review_subject,
    staged_diff,
    working_diff,
)
from .readiness import ReadinessCheck, ReadinessReport, code_readiness
from .repl import CodeREPL
from .tools import build_code_tools, build_review_tools

__all__ = [
    "ACTION_KINDS",
    "ActionRecord",
    "AppliedEdit",
    "ApplyResult",
    "CodeEngine",
    "CodeREPL",
    "CodeRunResult",
    "CommitOutcome",
    "CommitPlan",
    "Decision",
    "DiffStat",
    "EditJournal",
    "Hunk",
    "PermissionGate",
    "PermissionMode",
    "ProjectContext",
    "ProposedEdit",
    "ReadinessCheck",
    "ReadinessReport",
    "RepoInfo",
    "ReviewSubject",
    "UndoOutcome",
    "UnsafeGitAction",
    "apply_hunks",
    "build_code_tools",
    "build_project_context",
    "build_review_tools",
    "code_readiness",
    "commit_paths",
    "default_mode",
    "detect_repo",
    "diff_stat",
    "ensure_safe",
    "render_diff",
    "resolve_workspace",
    "review_subject",
    "split_hunks",
    "staged_diff",
    "suggest_message",
    "undo_workspace",
    "unified_diff_text",
    "unsafe_shell_git",
    "working_diff",
    "workspace_env",
    "workspace_execution_note",
]
