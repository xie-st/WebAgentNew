"""The ``effgen code`` command — a coding agent that writes, runs and fixes code.

Resolves the workspace, the permission mode and the task (from the argument, from
``-p``, or from piped stdin), runs the loop in
:class:`~effgen.cli.code.engine.CodeEngine`, and prints either a human-readable
report or a single JSON document.

Output contract:

- With a terminal, the answer is framed and followed by a one-line summary and,
  when anything was written or withheld, a short action list.
- Piped or with ``--json``, every human-facing line goes to stderr and stdout
  carries only the answer text (or the JSON document), so the command composes
  with ``jq`` and with shell pipelines.
- The exit code is ``0`` for a completed run, ``1`` for a failed one, and ``2``
  when the run completed but changes were withheld because there was no terminal
  to confirm on and no ``--auto-edit``/``--yes`` was given, or because a
  ``--commit`` that was asked for could not be confirmed.

Stdin contract: when stdin is not a terminal it is read to EOF before the run
starts — with a task in hand the piped text becomes context in front of it,
without one it *is* the task. Reading to EOF is what lets a slow producer be
folded in whole, so a pipe that stays open holds the run; a stream that has not
closed within a couple of seconds prints a note on stderr naming the wait and
how to skip it (``< /dev/null``), rather than waiting silently.

When the workspace is in a git repository, the branch, the short status and a
bounded file layout are read before the first model call and become part of the
agent's context. ``--commit`` offers to commit the files the run wrote — only
after a y/N confirmation, or, without a terminal, only when ``--yes`` was also
given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any

from effgen.cli.code.engine import (
    REVIEW_TASK,
    CodeEngine,
    CodeRunResult,
    resolve_workspace,
    undo_workspace,
    workspace_execution_note,
)
from effgen.cli.code.git_actions import CommitOutcome
from effgen.cli.code.permissions import (
    MODE_DESCRIPTIONS,
    PermissionMode,
    default_mode,
)
from effgen.cli.code.project import build_project_context, review_subject
from effgen.cli.code.render import (
    print_action,
    print_diff,
    print_plain,
    print_status,
    print_summary,
)
from effgen.cli.commands._shared import (
    _preflight_model_hint,
    _quickstart_suggest_model,
    resolve_provider_name,
)
from effgen.models._coding import coding_suitability
from effgen.ui.render import json_ensure_ascii

if TYPE_CHECKING:
    from collections.abc import Callable

    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)

#: Exit code for a run that completed but withheld every change.
EXIT_WITHHELD = 2


def _resolve_mode(args: argparse.Namespace, interactive: bool) -> tuple[PermissionMode, bool, str | None]:
    """Return ``(mode, explicit, error)`` from the permission flags.

    At most one of ``--plan``/``--auto-edit``/``--yes`` may be given; naming two
    is an error rather than a silent precedence rule.
    """
    chosen = [
        (PermissionMode.PLAN, "--plan", bool(getattr(args, "plan_only", False))),
        (PermissionMode.AUTO_EDIT, "--auto-edit", bool(getattr(args, "auto_edit", False))),
        (PermissionMode.YES, "--yes", bool(getattr(args, "assume_yes", False))),
    ]
    named = [(mode, flag) for mode, flag, given in chosen if given]
    if len(named) > 1:
        flags = ", ".join(flag for _, flag in named)
        return default_mode(interactive), False, (
            f"{flags} cannot be combined — they select different permission "
            "modes. Pass one."
        )
    if named:
        return named[0][0], True, None
    return default_mode(interactive), False, None


def _resolve_review(args: argparse.Namespace) -> tuple[str | None, list[str], str | None]:
    """Return ``(target, files, error)`` for a read-only review run.

    ``target`` is ``None`` when ``--review`` was not given; ``files`` is what
    ``-f/--file`` named. A review cannot be combined with a permission flag,
    ``--commit`` or ``--undo``: each of those asks for something a read-only run
    will not do, so naming both is an error rather than a silent precedence rule.
    """
    target = getattr(args, "review", None)
    files = list(getattr(args, "review_files", None) or [])
    if target is None:
        return None, files, None

    conflicts = [
        flag for flag, given in (
            ("--plan", bool(getattr(args, "plan_only", False))),
            ("--auto-edit", bool(getattr(args, "auto_edit", False))),
            ("--yes", bool(getattr(args, "assume_yes", False))),
            ("--commit", bool(getattr(args, "commit", False))),
            ("--undo", bool(getattr(args, "undo", False))),
        ) if given
    ]
    if conflicts:
        return None, files, (
            f"--review cannot be combined with {', '.join(conflicts)} — a review "
            "writes nothing, runs nothing and commits nothing. Pass one."
        )
    return target, files, None


#: Seconds a piped-stdin read may go without closing before the wait is announced.
STDIN_WAIT_NOTE_SECONDS = 2.0


def _stdin_wait_note(has_task: bool) -> str:
    """Return the one-line stderr note for a piped stdin that has not closed."""
    if has_task:
        return (
            "Reading piped stdin as context before starting; the stream has not "
            "closed. Close it (Ctrl-D) or re-run with < /dev/null to skip it."
        )
    return (
        "Reading the task from piped stdin; the stream has not closed. "
        "Close it (Ctrl-D), or pass the task with -p to skip the read."
    )


class _WaitNotice:
    """Call *notify* once if it is not cancelled within *delay* seconds.

    Used to announce a stdin read that has not finished, from a timer thread,
    while the reading thread is still blocked. Firing and leaving the block race
    for the same flag, so *notify* runs at most once and never after the read has
    returned; leaving also waits for a note that won the race to finish printing,
    and leaves no live timer thread behind.
    """

    def __init__(self, notify: "Callable[[], None]", delay: float) -> None:
        self._notify = notify
        self._lock = threading.Lock()
        self._settled = False
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True

    def _claim(self) -> bool:
        with self._lock:
            if self._settled:
                return False
            self._settled = True
            return True

    def _fire(self) -> None:
        if self._claim():
            self._notify()

    def __enter__(self) -> "_WaitNotice":
        self._timer.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._claim()
        self._timer.cancel()
        # Join so a note that won the race has finished printing before the run
        # continues; a cancelled timer returns from its wait immediately.
        self._timer.join(timeout=5.0)


def _read_piped_stdin(on_wait: "Callable[[], None] | None" = None) -> str:
    """Return piped stdin, or an empty string when stdin is a terminal.

    The read runs to EOF, so a producer that writes slowly (a build log, ``tail
    -f``) is folded in whole. A stream that has not closed within
    ``STDIN_WAIT_NOTE_SECONDS`` calls *on_wait* once, which lets the caller name
    what the command is waiting for instead of leaving it silent.
    """
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return ""
        if on_wait is None:
            return sys.stdin.read()
        with _WaitNotice(on_wait, STDIN_WAIT_NOTE_SECONDS):
            return sys.stdin.read()
    except (ValueError, OSError):  # pragma: no cover - closed stdin
        return ""


def _resolve_task(args: argparse.Namespace, interactive: bool) -> tuple[str | None, str | None]:
    """Return ``(task, error)`` from the positional argument, ``-p`` and stdin.

    Piped stdin with a task becomes context in front of it (``cat err.log |
    effgen code -p "explain this"``); piped stdin with no task *is* the task.
    The read waits for the stream to close either way; when it has not closed
    within a couple of seconds a note naming the wait goes to stderr, so an open
    pipe inherited from a supervisor or a harness is visible rather than silent.
    """
    explicit = getattr(args, "print_task", None)
    task = explicit if explicit else getattr(args, "task", None)
    has_task = bool(task and task.strip())

    def _announce() -> None:
        print(_stdin_wait_note(has_task), file=sys.stderr, flush=True)

    piped = _read_piped_stdin(on_wait=_announce)

    if piped.strip():
        if task and task.strip():
            task = f"--- stdin ---\n{piped.strip()}\n\n---\n\n{task}"
        else:
            task = piped.strip()

    if task and task.strip():
        return task, None

    if interactive:
        try:
            typed = input("Task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None, None
        if typed:
            return typed, None
    return None, (
        "effgen code needs a task. Pass one as an argument "
        '(effgen code "add a retry to fetch.py"), with -p, or on stdin '
        '(echo "..." | effgen code).'
    )


def _report(cli: "CLIInterface", result: CodeRunResult, *, quiet: bool) -> None:
    """Print the framed answer, the summary line and the file/refusal list.

    Used for an interactive run; the answer is rendered to the human console.
    A run stopped at its iteration cap has no answer: the panel carries what
    stopped it, and the tool output and reasoning it had reached follows in its
    own panel, labelled as progress.
    """
    from effgen.cli.commands.run import PARTIAL_PROGRESS_TITLE
    from effgen.ui.render import answer_surface

    console = cli._human()
    stopped = not result.success and result.partial
    if result.read_only:
        title = "Review"
    else:
        title = "Coding Agent"
    if stopped:
        title = "Stopped"
    if not result.success and not result.partial:
        cli.print_error_panel(result.answer or "The run produced no answer.", title="Error")
    elif console:
        answer_surface(
            result.answer,
            success=result.success,
            partial=result.partial,
            framed=True,
            title="Stopped" if stopped else title,
            console=console,
        )
        if stopped and result.partial_output:
            answer_surface(
                result.partial_output,
                success=False,
                partial=True,
                framed=True,
                title=PARTIAL_PROGRESS_TITLE,
                console=console,
            )
    else:
        print(result.answer, file=cli._human_stream())
        if stopped and result.partial_output:
            print(
                f"\n{PARTIAL_PROGRESS_TITLE}:\n{result.partial_output}",
                file=cli._human_stream(),
            )

    if quiet:
        return

    print_summary(cli, result)

    if result.files_written:
        names = ", ".join(result.files_written)
        print_plain(cli, f"Files written in {result.workspace}: {names}")
    _report_partial_hunks(cli, result)
    for record in result.refused:
        print_status(cli, "warning", record.reason)


def _withheld_note(result: CodeRunResult, explicit_mode: bool) -> str:
    """Return the one-line explanation for the actions a run did not carry out.

    The wording follows what actually happened: a run that wrote nothing says so,
    while a run that applied its edits but could not confirm a shell command or a
    commit says only those were withheld. The suggested flag is the one that
    covers the kinds that were withheld.
    """
    kinds = sorted({a.kind for a in result.withheld})
    what = ", ".join(kinds)
    opt_in = (
        "--auto-edit (or --yes)"
        if set(kinds) <= {"write", "run"}
        else "--yes"
    )
    if explicit_mode and result.permission_mode == PermissionMode.PLAN.value:
        return (
            f"Plan mode: {len(result.withheld)} action(s) ({what}) were proposed "
            f"and not carried out. Re-run with {opt_in} to apply them."
        )
    lead = (
        "Some actions were not carried out"
        if result.files_written
        else "No changes were made"
    )
    return (
        f"{lead}: {len(result.withheld)} action(s) ({what}) needed confirmation "
        f"and this session has no terminal to confirm on. Re-run with {opt_in} "
        "to allow them without asking."
    )


def _run_undo(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Reverse the last applied edit(s) in the workspace and report what changed."""
    json_mode = bool(getattr(args, "output_json", False))
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, OSError):  # pragma: no cover - closed std streams
        interactive = False
    if json_mode or not interactive:
        cli._human_to_stderr = True

    try:
        workspace = resolve_workspace(getattr(args, "workspace", None))
    except OSError as exc:
        cli.print_error(f"--workspace: {exc}")
        return 1

    count = int(getattr(args, "undo_count", None) or 1)
    outcomes, remaining = undo_workspace(workspace, count)

    if json_mode:
        print(json.dumps(
            {
                "undone": [
                    {"path": o.rel_path, "action": o.action, "detail": o.detail}
                    for o in outcomes
                ],
                "remaining": remaining,
            },
            indent=2, ensure_ascii=json_ensure_ascii(),
        ))
        return 0

    if not outcomes:
        print_plain(cli, f"Nothing to undo in {workspace}.")
        return 0
    for outcome in outcomes:
        if outcome.action == "restored":
            print_plain(cli, f"Restored {outcome.rel_path} to its previous content.")
        elif outcome.action == "removed":
            print_plain(cli, f"Removed {outcome.rel_path} (created by a coding run).")
        elif outcome.action == "skipped":
            print_status(cli, "warning", f"Skipped {outcome.rel_path}: {outcome.detail}")
        else:
            print_status(cli, "error", f"Could not undo {outcome.rel_path}: {outcome.detail}")
    cli.print(f"{remaining} earlier change(s) can still be undone.")
    return 0


def _offer_commit(
    cli: "CLIInterface",
    args: argparse.Namespace,
    engine: CodeEngine,
    result: CodeRunResult,
    *,
    quiet: bool,
) -> None:
    """Offer to commit the files this run wrote, and record the outcome.

    The plan is printed first, so the user sees the repository, the exact paths
    and the message before answering. The permission gate then decides: a
    terminal is asked y/N, ``--yes`` commits without asking, and any other
    non-interactive combination withholds the commit. Nothing outside the listed
    paths is ever staged or committed.
    """
    plan = engine.plan_commit(
        result.files_written,
        task=result.task,
        message=getattr(args, "commit_message", None),
    )
    if plan.error:
        print_status(cli, "warning", plan.error)
        # The same shape as a completed attempt, so ``--json`` consumers read one
        # commit record whether or not there was anything to commit.
        result.commit = CommitOutcome(False, plan.message, plan.error).to_dict()
        return

    if not quiet:
        print_plain(cli, plan.describe())
        print_plain(cli, f'Message: "{plan.subject}"')
        if plan.other_staged:
            print_status(
                cli, "warning",
                f"{len(plan.other_staged)} other path(s) are staged in this "
                "repository; they stay staged and are not part of this commit: "
                + ", ".join(plan.other_staged[:5])
                + ("…" if len(plan.other_staged) > 5 else ""),
            )

    outcome = engine.perform_commit(plan)
    result.actions = list(engine.gate.actions)
    result.commit = outcome.to_dict()
    if outcome.success:
        print_status(
            cli, "success",
            f"Committed {len(outcome.paths)} file(s) as {outcome.commit or 'HEAD'}: "
            f"{plan.subject}",
        )
    else:
        print_status(cli, "warning", f"Nothing was committed: {outcome.detail}")


def _report_partial_hunks(cli: "CLIInterface", result: CodeRunResult) -> None:
    """Warn about any edit whose hunks did not all apply."""
    for diff in result.diffs:
        failed = int(diff.get("hunks_failed", 0) or 0)
        if failed:
            print_status(
                cli,
                "warning",
                f"{diff.get('path')}: {failed} hunk(s) did not apply because the "
                "file changed since it was read; the rest were applied.",
            )


def run_code_command(cli: "CLIInterface", args: argparse.Namespace) -> int:
    """Run one ``effgen code`` task and return the process exit code."""
    review_target, review_files, review_error = _resolve_review(args)
    reviewing = review_target is not None or bool(review_files)

    json_mode = bool(getattr(args, "output_json", False))
    quiet = bool(getattr(args, "quiet", False))

    try:
        stdin_tty = sys.stdin.isatty()
        interactive = stdin_tty and sys.stdout.isatty()
    except (ValueError, OSError):  # pragma: no cover - closed std streams
        stdin_tty = interactive = False

    # Piped or JSON output keeps stdout for the result alone; everything a
    # human reads — including an argument error — goes to stderr.
    if json_mode or not interactive:
        cli._human_to_stderr = True

    if review_error:
        cli.print_error(review_error)
        return 1

    if bool(getattr(args, "undo", False)):
        return _run_undo(cli, args)

    # A terminal with no task and no piped stdin opens the interactive REPL; a
    # given task, ``-p``, piped stdin, ``--json`` or a review runs the
    # single-shot path so scripted and non-TTY invocations stay byte-clean.
    if (
        interactive
        and stdin_tty
        and not json_mode
        and not reviewing
        and getattr(args, "print_task", None) is None
        and not getattr(args, "task", None)
    ):
        # The provider and the workspace are validated here, before the session
        # starts, so a typo or an unusable directory is a one-line error rather
        # than a failure on the first turn.
        provider, provider_error = resolve_provider_name(getattr(args, "provider", None))
        if provider_error:
            cli.print_error(provider_error)
            return 1
        args.provider = provider
        try:
            resolve_workspace(getattr(args, "workspace", None))
        except OSError as exc:
            cli.print_error(f"--workspace: {exc}")
            return 1

        from effgen.cli.code.repl import CodeREPL

        return CodeREPL(cli, args).run()

    mode, mode_explicit, mode_error = _resolve_mode(args, interactive)
    if mode_error:
        cli.print_error(mode_error)
        return 1
    if reviewing:
        # A review holds no tool that writes or runs, and the gate denies all
        # four kinds behind them. ``permission_mode`` keeps reporting one of the
        # four documented values; ``read_only`` in the record says what this is.
        mode, mode_explicit = PermissionMode.PLAN, True

    task, task_error = _resolve_task(args, interactive and not reviewing)
    if task is None:
        if reviewing:
            task = REVIEW_TASK
        else:
            if task_error:
                cli.print_error(task_error)
            return 1

    provider, provider_error = resolve_provider_name(getattr(args, "provider", None))
    if provider_error:
        cli.print_error(provider_error)
        return 1

    try:
        workspace = resolve_workspace(getattr(args, "workspace", None))
    except OSError as exc:
        cli.print_error(f"--workspace: {exc}")
        return 1

    model = getattr(args, "model", None)
    if model:
        _preflight_model_hint(cli, model, provider)
    else:
        model, suggested_provider, reason = _quickstart_suggest_model()
        if provider is None and suggested_provider:
            provider = suggested_provider
        if not quiet:
            print_plain(cli, f"Using model {model} ({reason}); override with -m/--model.")

    # The repository state and file layout are read once, before the first model
    # call, and travel with the run as context.
    project = build_project_context(workspace)

    # A review's subject is assembled here, before any model call, so a bad
    # revision or a missing file is a one-line error rather than a wasted run.
    subject = None
    if reviewing:
        subject = review_subject(workspace, review_target, review_files)
        if subject.error:
            cli.print_error(subject.error)
            return 1

    if not quiet:
        print_plain(cli, f"Workspace: {workspace}")
        print_plain(cli, project.summary_line())
        if project.brief_path:
            print_plain(cli, f"Project instructions: {project.brief_path}")
        if subject is not None:
            print_plain(cli, subject.describe())
            cli.print("Read-only review: no file is written and no command runs.")
        else:
            cli.print(f"Permissions: {mode.value} — {MODE_DESCRIPTIONS[mode]}")
        if mode is not PermissionMode.PLAN:
            note = workspace_execution_note(workspace)
            if note:
                print_status(cli, "warning", note)

    # Live per-action ticks are only shown on an interactive run; a piped or
    # JSON run reports the actions once, at the end, so stdout stays clean.
    show_ticks = interactive and not quiet and not json_mode
    on_event = (lambda record: print_action(cli, record)) if show_ticks else None

    # Each pending edit's diff is shown (on the human stream) before the write is
    # decided, so a proposal is visible even in plan mode and even when piped.
    # Under --json the diffs are in the document, so the live render is skipped.
    show_diffs = not quiet and not json_mode
    on_diff = (lambda edit: print_diff(cli, edit)) if show_diffs else None

    engine = CodeEngine(
        model=model,
        provider=provider,
        workspace=workspace,
        mode=mode,
        mode_explicit=mode_explicit,
        interactive=interactive,
        max_iterations=getattr(args, "max_iterations", None),
        temperature=getattr(args, "temperature", None),
        max_tokens=getattr(args, "max_tokens", None),
        on_event=on_event,
        on_diff=on_diff,
        project=project,
        review=subject,
        session_id=getattr(args, "session_id", None),
    )

    agent: Any = None
    suitability = None
    try:
        agent = engine.build_agent()
        # The model is loaded here, so what it says about its own tool calling
        # is answerable; the note goes out before the first billed call.
        suitability = coding_suitability(model, provider, getattr(agent, "model", None))
        if not quiet and not suitability.is_suitable:
            print_status(cli, "warning", suitability.note())
        result = engine.run(task)
        # Recorded before the agent is closed, so a session continued on the
        # command line knows where it was working and what it wrote.
        engine.save_coding_state(files_written=result.files_written)
    except KeyboardInterrupt:
        cli.print_warning("Interrupted; no further actions were taken.")
        return 130
    except Exception as exc:  # noqa: BLE001 - reported to the user, not swallowed
        cli.print_error_panel(str(exc), title="Error")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        if json_mode:
            print(json.dumps(
                {"success": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                indent=2, ensure_ascii=json_ensure_ascii(),
            ))
        return 1
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception as exc:  # noqa: BLE001 - close is best effort
                logger.debug("Agent close failed: %s", exc)

    if getattr(args, "commit", False):
        _offer_commit(cli, args, engine, result, quiet=quiet)

    if json_mode:
        document = result.to_dict()
        if suitability is not None:
            document["coding_suitability"] = suitability.to_dict()
        print(json.dumps(document, indent=2, ensure_ascii=json_ensure_ascii()))
    elif interactive:
        _report(cli, result, quiet=quiet)
    else:
        # Piped: stdout is the answer text and nothing else.
        print(result.answer)
        if not quiet:
            _report_to_stderr(cli, result)

    if result.hit_iteration_cap and not quiet:
        # The answer already names the cap and what it cost; this adds the flag
        # that raises it from the command line.
        cli.print_warning("Raise the cap for this task with --max-iterations N.")

    if result.withheld and not quiet:
        cli.print(_withheld_note(result, mode_explicit))

    if not result.success:
        return 1
    if result.withheld and (not mode_explicit or _asked_for_withheld_commit(result)):
        return EXIT_WITHHELD
    return 0


def _asked_for_withheld_commit(result: CodeRunResult) -> bool:
    """True when ``--commit`` was given and the commit was withheld.

    A permission flag speaks for the model's actions: choosing ``--auto-edit``
    means accepting that a shell command still needs confirmation, so a run that
    withholds one still exits 0. ``--commit`` is different — it asks for one
    specific action by name. When that action is withheld the run did less than
    it was told to, and the exit code says so.
    """
    return any(action.kind == "git" for action in result.withheld)


def _report_to_stderr(cli: "CLIInterface", result: CodeRunResult) -> None:
    """Print the summary and file list for a piped run (stdout stays the answer)."""
    print_summary(cli, result)
    if result.files_written:
        print_plain(cli, f"Files written in {result.workspace}: {', '.join(result.files_written)}")
    _report_partial_hunks(cli, result)
    for record in result.refused:
        print_status(cli, "warning", record.reason)
