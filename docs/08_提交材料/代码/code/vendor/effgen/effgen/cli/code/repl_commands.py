"""The slash-command table and the commands that act on the working set.

:mod:`effgen.cli.code.repl` owns the session itself and mixes these methods into
:class:`~effgen.cli.code.repl.CodeREPL`, so ``effgen.cli.code.repl`` stays the
import path callers use — it re-imports the table, the tool list and the token
estimate below. Holds the one command table that ``/help``, tab-completion and
the dispatcher all read, and the commands that change what the session is
working on: the plan → review → apply flow, the confined shell runs, the
files-in-context set, the tool list and the help itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from effgen.cli.code.engine import undo_workspace, workspace_env
from effgen.cli.code.permissions import PermissionMode
from effgen.cli.code.project import build_project_context
from effgen.cli.code.render import print_diff

# Slash commands surfaced in /help and tab-completion. One source of truth so the
# help text, the completer and the dispatcher never drift apart.
_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help (or type just / for the menu)",
    "/plan": "Propose a change without writing it:  /plan add retries to fetch.py",
    "/review": "Review, read-only, without writing or running anything:  "
               "/review [uncommitted|staged|<rev>]",
    "/run": "Run a shell command in the workspace:  /run ls -la",
    "/test": "Run tests in the workspace:  /test [pytest args]",
    "/diff": "Show the edits staged by the last /plan (or the last turn's edits)",
    "/apply": "Apply the staged edits:  /apply [n]  (n = one edit by number)",
    "/reject": "Discard the staged edits:  /reject [n]",
    "/undo": "Reverse the last applied edit(s):  /undo [n]",
    "/context": "Show the files in context and their token/size estimate "
                "(/context refresh re-reads the project layout)",
    "/add": "Add a workspace file to context:  /add src/app.py",
    "/drop": "Remove a file from context:  /drop src/app.py",
    "/mode": "Show or set the permission mode:  /mode ask|auto-edit|yes|plan",
    "/model": "Hot-swap the active model:  /model gpt-5-nano",
    "/tools": "List the coding tools the agent can call",
    "/cost": "Session token + cost total",
    "/trace": "Show the last turn's reasoning/tool steps",
    "/git": "Git for the workspace:  /git [status|diff|log|branch|show|remote] · "
            "/git staged (pull the staged diff into context) · /git commit [message]",
    "/reset": "Clear the conversation memory (keep files-in-context)",
    "/clear": "Reset live context (memory + files) but keep the session on disk",
    "/compact": "Summarize the conversation so far to extend a long session",
    "/save": "Save this coding session to a file:  /save [name]",
    "/session": "Show, name or resume the session id:  /session [id]  (the same store as `effgen code --session-id`)",
    "/load": "Load a saved coding session:  /load [name|number]",
    "/doctor": "Run a quick environment check",
    "/exit": "Leave the coding agent (also: /quit, exit, quit)",
}

# The four tools the coding preset gives the agent, by tool name, for /tools.
_CODING_TOOLS: tuple[tuple[str, str], ...] = (
    ("file_operations", "read, write, list and search files in the workspace"),
    ("code_executor", "run code in the configured sandbox"),
    ("python_repl", "evaluate short Python expressions in a restricted namespace"),
    ("bash", "run a shell command in the workspace"),
)


def _estimate_tokens(text: str) -> int:
    """A rough token estimate (~4 characters per token) for context budgeting."""
    return max(1, len(text) // 4) if text else 0


class CodeCommandsMixin:
    """The dispatcher and working-set commands of :class:`~effgen.cli.code.repl.CodeREPL`.

    Every method reads state ``CodeREPL.__init__`` sets — ``self.engine``,
    ``self.pending_edits``, ``self.context_files`` and ``self.workspace`` — and
    prints through the view methods.
    """

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, text: str) -> str | None:
        """Route a line. Returns 'exit', 'handled', or None (treat as a task)."""
        low = text.lower()
        if low in ("exit", "quit"):
            return "exit"
        if low == "help":
            self._cmd_help()
            return "handled"

        if not text.startswith("/"):
            return None
        if text.strip() == "/":
            self._cmd_help()
            return "handled"

        parts = text[1:].split(maxsplit=1)
        cmd = ("/" + parts[0]).lower() if parts else "/"
        arg = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/help": lambda: self._cmd_help(),
            "/exit": lambda: "exit",
            "/quit": lambda: "exit",
            "/plan": lambda: self._cmd_plan(arg),
            "/review": lambda: self._cmd_review(arg),
            "/run": lambda: self._cmd_run(arg),
            "/test": lambda: self._cmd_test(arg),
            "/diff": lambda: self._cmd_diff(),
            "/apply": lambda: self._cmd_apply(arg),
            "/reject": lambda: self._cmd_reject(arg),
            "/undo": lambda: self._cmd_undo(arg),
            "/context": lambda: self._cmd_context(arg),
            "/add": lambda: self._cmd_add(arg),
            "/drop": lambda: self._cmd_drop(arg),
            "/mode": lambda: self._cmd_mode(arg),
            "/model": lambda: self._cmd_model(arg),
            "/tools": lambda: self._cmd_tools(),
            "/cost": lambda: self._cmd_cost(),
            "/trace": lambda: self._cmd_trace(),
            "/git": lambda: self._cmd_git(arg),
            "/reset": lambda: self._cmd_reset(),
            "/clear": lambda: self._cmd_clear(),
            "/compact": lambda: self._cmd_compact(arg),
            "/save": lambda: self._cmd_save(arg),
            "/session": lambda: self._cmd_session(arg),
            "/load": lambda: self._cmd_load(arg),
            "/doctor": lambda: self._cmd_doctor(),
        }
        handler = handlers.get(cmd)
        if handler is None:
            import difflib

            typed = text.split()[0]
            close = difflib.get_close_matches(cmd, list(_SLASH_COMMANDS), n=3)
            hint = (
                f" Did you mean: {', '.join(close)}?" if close else " Type /help for the list."
            )
            self._status("warning", f"Unknown command '{typed}'.{hint}")
            return "handled"
        result = handler()
        return result if result == "exit" else "handled"

    # ------------------------------------------------------------------
    # /plan, /diff, /apply, /reject, /undo
    # ------------------------------------------------------------------
    def _cmd_plan(self, arg: str) -> None:
        """Propose a change for *arg* without writing anything; stage the edits."""
        if not arg:
            self._say(
                "Usage: /plan <task>. Proposes the edits for a task without writing "
                "them, then /diff to review and /apply to carry them out."
            )
            return
        assert self.engine is not None
        prev_mode = self.engine.gate.mode
        self.pending_edits.clear()
        self.engine.gate.mode = PermissionMode.PLAN
        self._staging = True
        try:
            self._do_turn(arg)
        finally:
            self._staging = False
            self.engine.gate.mode = prev_mode
        if self.pending_edits:
            self._say(
                f"{len(self.pending_edits)} edit(s) staged: "
                f"{', '.join(self.pending_edits)}. /diff to review, /apply to write, "
                "/reject to discard."
            )
        else:
            self._say("No edits were proposed.")

    def _cmd_review(self, arg: str) -> None:
        """Run one read-only turn over a diff, or over the files in context.

        For that turn the agent holds only reading tools and the gate is in
        ``plan`` mode; both are restored afterwards, including when the turn
        raises, so the next turn can write again.
        """
        assert self.engine is not None
        from effgen.cli.code.engine import REVIEW_TASK
        from effgen.cli.code.project import review_subject

        target = arg.split()[0] if arg.strip() else "uncommitted"
        # The files pinned with /add are already inlined by the composed task,
        # so the subject carries the diff alone and nothing is sent twice.
        subject = review_subject(self.workspace, target)
        if subject.error:
            self._status("warning", subject.error)
            return
        self._say(subject.describe())
        with self.engine.review_turn(subject):
            self._do_turn(REVIEW_TASK)

    def _cmd_diff(self) -> None:
        """Show the staged edits, or the edits from the last turn."""
        if self.pending_edits:
            self._say(f"Staged edits ({len(self.pending_edits)}):")
            for edit in self.pending_edits.values():
                print_diff(self.cli, edit)
            return
        result = self.last_result
        if result is None or not result.diffs:
            self._say("No staged edits, and the last turn changed no files.")
            return
        from effgen.cli.code.diffs import render_diff

        stream = self.cli._human_stream()
        console = self.cli._human()

        def show(diffs: list[Any], heading: str) -> None:
            if not diffs:
                return
            self._say(f"{heading} ({len(diffs)}):")
            for diff in diffs:
                plain, markup = render_diff(str(diff.get("diff", "")), stream)
                if console:
                    console.print(markup, highlight=False)
                else:
                    self._say(plain)

        show([d for d in result.diffs if d.get("applied", True)], "Last turn's edits")
        show(
            [d for d in result.diffs if not d.get("applied", True)],
            "Proposed last turn, not written",
        )

    def _staged_selection(self, arg: str) -> list[Any] | None:
        """Resolve ``/apply``/``/reject`` arguments to a list of staged edits.

        A bare command selects every staged edit; a 1-based number selects one.
        Returns ``None`` (after reporting) when the selection is invalid.
        """
        edits = list(self.pending_edits.values())
        if not edits:
            self._say("Nothing is staged. Use /plan <task> first.")
            return None
        if not arg:
            return edits
        token = arg.split()[0]
        if not token.isdigit():
            self._status("warning", f"Expected an edit number, got '{token}'.")
            return None
        idx = int(token) - 1
        if not 0 <= idx < len(edits):
            self._status("warning", f"No staged edit #{token} (have {len(edits)}).")
            return None
        return [edits[idx]]

    def _cmd_apply(self, arg: str) -> None:
        """Write the staged edits to disk through the gated, journaled write path."""
        selected = self._staged_selection(arg)
        if selected is None:
            return
        assert self.engine is not None
        fops = self._tool("file_operations")
        if fops is None:
            self._status("error", "The file tool is unavailable.")
            return
        prev_mode = self.engine.gate.mode
        self.engine.gate.mode = PermissionMode.AUTO_EDIT
        self.engine.gate.begin_turn()
        written: list[str] = []
        try:
            for edit in selected:
                result = asyncio.run(
                    fops.execute(operation="write", path=edit.rel_path, content=edit.new_content)
                )
                ok = getattr(result, "success", None)
                if ok is None and isinstance(result, dict):
                    ok = result.get("success")
                if ok is not False:
                    written.append(edit.rel_path)
                    self.pending_edits.pop(edit.rel_path, None)
        finally:
            self.engine.gate.mode = prev_mode
        if written:
            self._note_written(written)
            self._status("success", f"Applied: {', '.join(written)} (/undo to reverse).")
        else:
            self._status("warning", "No edits were applied.")

    def _cmd_reject(self, arg: str) -> None:
        """Discard the staged edits without writing them."""
        selected = self._staged_selection(arg)
        if selected is None:
            return
        for edit in selected:
            self.pending_edits.pop(edit.rel_path, None)
        names = ", ".join(e.rel_path for e in selected)
        self._say(f"Discarded staged edit(s): {names}.")

    def _cmd_undo(self, arg: str) -> None:
        """Reverse the last applied edit(s) via the on-disk journal."""
        count = 1
        if arg:
            token = arg.split()[0]
            if token.isdigit():
                count = max(1, int(token))
        outcomes, remaining = undo_workspace(self.workspace, count)
        if not outcomes:
            self._say(f"Nothing to undo in {self.workspace}.")
            return
        for outcome in outcomes:
            if outcome.action == "restored":
                self._say(f"Restored {outcome.rel_path} to its previous content.")
            elif outcome.action == "removed":
                self._say(f"Removed {outcome.rel_path} (created by a coding run).")
            elif outcome.action == "skipped":
                self._status("warning", f"Skipped {outcome.rel_path}: {outcome.detail}")
            else:
                self._status("error", f"Could not undo {outcome.rel_path}: {outcome.detail}")
        self._say(f"{remaining} earlier change(s) can still be undone.")

    # ------------------------------------------------------------------
    # /run, /test
    # ------------------------------------------------------------------
    def _cmd_run(self, arg: str) -> None:
        if not arg:
            self._say("Usage: /run <command>   (e.g. /run pytest -q)")
            return
        self._execute_shell(arg)

    def _cmd_test(self, arg: str) -> None:
        command = f"pytest {arg}".strip() if arg else "pytest"
        self._say(f"Running: {command}")
        self._execute_shell(command)

    def _execute_shell(self, command: str) -> None:
        """Run *command* in the workspace and feed the output back to the agent."""
        assert self.engine is not None
        bash = self._tool("bash")
        if bash is None:
            self._status("error", "The shell tool is unavailable.")
            return
        prev_mode = self.engine.gate.mode
        # The user typed the command explicitly, so it is allowed without a second
        # prompt; the shell's own blocked-command and credential-file guards still
        # apply underneath, and it stays confined to the workspace.
        self.engine.gate.mode = PermissionMode.YES
        self.engine.gate.begin_turn()
        try:
            with workspace_env(self.workspace):
                result = asyncio.run(bash.execute(command=command))
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Command failed to start: {e}")
            return
        finally:
            self.engine.gate.mode = prev_mode

        text = self._shell_output_text(result)
        self._say(text or "(no output)")
        # Feed the run back so the next turn can react to it.
        self._feed_back(f"I ran `{command}` in the workspace. Output:\n{text or '(no output)'}")

    @staticmethod
    def _shell_output_text(result: Any) -> str:
        """Return the combined stdout/stderr text from a shell tool result.

        ``bash``/``code_executor`` return their run detail as a dict
        (``stdout``/``stderr``/``return_code``); a gated refusal returns
        ``{"success": False, "error": ...}``. Both are reduced to the lines a
        user wants to read, so ``/run`` shows the command's output, not a dict.
        A command the shell tool blocked (or one that failed before producing
        any output) carries its reason on the result rather than in the output;
        that reason is returned, so a refusal is never reported as no output.
        """
        output = getattr(result, "output", None)
        if output is None and isinstance(result, dict):
            output = result
        if isinstance(output, dict):
            if output.get("success") is False and output.get("error"):
                return str(output["error"]).rstrip()
            parts = [str(output.get("stdout") or ""), str(output.get("stderr") or "")]
            text = "\n".join(p for p in parts if p.strip()).rstrip()
            if text:
                return text
        elif output:
            return str(output).rstrip()
        return str(getattr(result, "error", "") or "").rstrip()

    def _feed_back(self, note: str) -> None:
        """Add an observation to the conversation so the next turn sees it."""
        agent = self.agent
        if agent is None:
            return
        try:
            agent.short_term_memory.add_user_message(note)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Files in context
    # ------------------------------------------------------------------
    def _cmd_context(self, arg: str = "") -> None:
        """Show what the agent is working from; ``refresh`` re-reads the project."""
        if arg.split()[:1] == ["refresh"]:
            self.project = (
                self.engine.load_project(refresh=True)
                if self.engine is not None
                else build_project_context(self.workspace)
            )
            self._status("success", f"Project context refreshed — {self.project.summary_line()}")
            return

        self.cli.print_header("Files in context")
        self._say(f"Workspace: {self.workspace}")
        self._say(self.project.summary_line())
        if self.project.brief_path:
            self._say(f"Project instructions: {self.project.brief_path}")
        self._say(
            f"Project layout in the prompt: {self.project.file_count_text()}, "
            f"{len(self.project.layout_lines())} line(s) — /context refresh to re-read"
        )
        if not self.context_files:
            self._say("Files pinned with /add: (none) — add one with /add <file>")
            return
        self._say("Files pinned with /add (their content is sent every turn):")
        total = 0
        for rel in self.context_files:
            content = self._read_context_file(rel)
            if content is None:
                self._status("warning", f"{rel} — unreadable (removed or not text)")
                continue
            tokens = _estimate_tokens(content)
            total += tokens
            self._say(f"  {rel}  ({len(content):,} chars, ~{tokens:,} tok)")
        self._say(f"Total: {len(self.context_files)} file(s), ~{total:,} tokens (estimate)")

    def _cmd_add(self, arg: str) -> None:
        if not arg:
            self._say("Usage: /add <file>   (a path inside the workspace)")
            return
        rel = arg.split()[0]
        try:
            resolved = (self.workspace / rel).resolve()
            resolved.relative_to(self.workspace.resolve())
        except (ValueError, OSError):
            self._status(
                "warning",
                f"'{rel}' is outside the workspace {self.workspace}; not added."
            )
            return
        if not resolved.is_file():
            self._status("warning", f"No file at '{rel}' in the workspace.")
            return
        norm = str(resolved.relative_to(self.workspace.resolve()))
        if norm in self.context_files:
            self._say(f"'{norm}' is already in context.")
            return
        if self._read_context_file(norm) is None:
            self._status("warning", f"'{norm}' is not readable as text; not added.")
            return
        self.context_files.append(norm)
        self._save_coding_state()
        self._status("success", f"Added '{norm}' to context ({len(self.context_files)} file(s)).")

    def _cmd_drop(self, arg: str) -> None:
        if not arg:
            self._say("Usage: /drop <file>")
            return
        rel = arg.split()[0]
        # Accept either the stored relative path or the raw argument.
        match = None
        for stored in self.context_files:
            if stored == rel or Path(stored).name == rel:
                match = stored
                break
        if match is None:
            self._status("warning", f"'{rel}' is not in context.")
            return
        self.context_files.remove(match)
        self._save_coding_state()
        self._status("success", f"Dropped '{match}' ({len(self.context_files)} file(s) left).")

    # ------------------------------------------------------------------
    # /tools, /help
    # ------------------------------------------------------------------
    def _cmd_tools(self) -> None:
        self.cli.print_header("Coding tools")
        self._say(
            "The agent calls these directly; every write and command passes the "
            f"permission gate ({self.mode.value} mode)."
        )
        for name, desc in _CODING_TOOLS:
            self._say(f"  {name:16s} {desc}")

    def _cmd_help(self) -> None:
        if self.console and self.interactive and self._color:
            from rich.table import Table
            from rich.text import Text

            table = Table(title="Coding commands", show_header=True, header_style="cyan")
            table.add_column("Command", style="green", no_wrap=True)
            table.add_column("Does")
            for cmd, desc in _SLASH_COMMANDS.items():
                # Plain text, so a usage example such as "/apply [n]" keeps its
                # brackets instead of being read as console markup.
                table.add_row(cmd, Text(desc))
            self.console.print(table)
        else:
            self._say("Coding commands:")
            for cmd, desc in _SLASH_COMMANDS.items():
                self._say(f"  {cmd:9s} {desc}")
