"""Interactive REPL for ``effgen code``.

``effgen code`` with a terminal and no task opens this loop: the coding agent
answers turn by turn, writing files, running code in the configured sandbox, and
reading the real output — with every change previewed as a unified diff and gated
by the session's permission mode. It parallels :class:`~effgen.cli.chat.ChatREPL`
and reuses its plumbing (TTY detection, ``NO_COLOR`` handling, stderr routing for
a piped session, readline history + tab-completion, the ``/model`` hot-swap, and
the session store), adding the parts a coding session needs:

- a **files-in-context** set the user manages with ``/add`` / ``/drop`` /
  ``/context``, injected into each turn so the agent sees the files being worked
  on, with a token/size estimate so context stays bounded;
- a **plan → review → apply** flow: ``/plan <task>`` proposes edits without
  writing them, ``/diff`` shows the staged edits, and ``/apply`` / ``/reject``
  carry them out or discard them (reusing the diff/apply/undo layer);
- ``/run`` / ``/test`` to execute a command through the confined sandbox and feed
  the output back into the conversation;
- ``/undo`` over the on-disk edit journal, ``/compact`` to summarize a long
  conversation, and ``/save`` / ``/session`` / ``/load`` to round-trip a session;
- **project awareness**: the repository's branch, short status and file layout
  are read at startup and travel with each turn, ``/git`` shows the read-only git
  surface and pulls the staged diff into context, and ``/git commit`` records the
  session's edits after an explicit confirmation.

The interactive chrome is a terminal affordance only. When there is no terminal
the command never reaches this class — :func:`effgen.cli.commands.code.run_code_command`
runs the single-shot path instead — so the piped/``--json`` output stays clean.
"""

# The session is composed from four sibling modules, and this module stays the
# import path for everything they define. Imports marked ``noqa: F401`` below are
# no longer read here: they are kept so every name this module has ever exposed
# still resolves against it. Do not run ``ruff --fix`` over this block; it would
# read them as unused and drop them.

from __future__ import annotations

import asyncio  # noqa: F401 - re-exported: moved to repl_commands, imported from here
import json  # noqa: F401 - re-exported: moved to repl_session, imported from here
import logging
import sys
import time  # noqa: F401 - re-exported: moved to repl_turn, imported from here
from datetime import datetime  # noqa: F401 - re-exported: moved to repl_session, imported from here
from pathlib import Path
from typing import Any

from effgen.cli.chat import _history_dir
from effgen.cli.code.engine import (
    CodeEngine,
    resolve_workspace,
    undo_workspace,  # noqa: F401 - re-exported: moved to repl_commands, imported from here
    workspace_env,  # noqa: F401 - re-exported: moved to repl_commands, imported from here
)
from effgen.cli.code.permissions import (
    MODE_DESCRIPTIONS,  # noqa: F401 - re-exported: moved to repl_session, imported from here
    ActionRecord,
    PermissionMode,
    default_mode,
)
from effgen.cli.code.project import (
    build_project_context,
    staged_diff,  # noqa: F401 - re-exported: moved to repl_session, imported from here
)
from effgen.cli.code.render import (
    print_action,
    print_diff,
    print_plain,  # noqa: F401 - re-exported: moved to repl_view, imported from here
    print_status,  # noqa: F401 - re-exported: moved to repl_view, imported from here
)
from effgen.cli.code.repl_commands import (
    _CODING_TOOLS,  # noqa: F401 - re-exported: moved to repl_commands, imported from here
    _SLASH_COMMANDS,
    CodeCommandsMixin,
    _estimate_tokens,  # noqa: F401 - re-exported: moved to repl_commands, imported from here
)
from effgen.cli.code.repl_session import CodeSessionMixin
from effgen.cli.code.repl_turn import CodeTurnMixin
from effgen.cli.code.repl_view import CodeViewMixin
from effgen.ui.theme import color_enabled

logger: logging.Logger = logging.getLogger(__name__)


class CodeREPL(CodeCommandsMixin, CodeSessionMixin, CodeTurnMixin, CodeViewMixin):
    """An interactive coding session over a single :class:`CodeEngine`."""

    def __init__(self, cli: Any, args: Any) -> None:
        self.cli = cli
        self.args = args
        self.console = getattr(cli, "console", None)

        self.quiet = bool(getattr(args, "quiet", False))
        self.verbose = bool(getattr(args, "verbose", False))
        self._color = color_enabled()
        try:
            self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        except (ValueError, OSError):  # pragma: no cover - closed std streams
            self.interactive = False
        if not self.interactive:
            try:
                self.cli._human_to_stderr = True
            except Exception:  # noqa: BLE001
                pass

        self.workspace: Path = resolve_workspace(getattr(args, "workspace", None))
        self.provider = getattr(args, "provider", None)
        self._model_defaulted = getattr(args, "model", None) is None
        self.model_id = getattr(args, "model", None)
        self.temperature = getattr(args, "temperature", None)
        self.max_tokens = getattr(args, "max_tokens", None)
        self.max_iterations = getattr(args, "max_iterations", None)

        # The permission mode for the session. A named flag pins it; otherwise a
        # terminal session confirms each action (ask).
        self.mode, self.mode_explicit = self._initial_mode(args)

        # Files the user has pulled into context; their live content is injected
        # into each turn so the agent works from what is on disk right now.
        self.context_files: list[str] = []
        # Edits proposed by the last /plan turn, keyed by path (latest wins),
        # awaiting /apply or /reject.
        self.pending_edits: dict[str, Any] = {}
        # True only while a /plan turn runs, so on_diff stages instead of applying.
        self._staging = False

        # The repository state, file layout and project brief, read once at
        # startup and refreshed on request with /context refresh.
        self.project = build_project_context(self.workspace)
        # Workspace-relative paths this session has written, in order, so
        # /git commit records exactly the session's own edits.
        self.session_files: list[str] = []

        self.engine: CodeEngine | None = None
        # A session id given on the command line resumes that session: its turns
        # are recalled and every new one is written back.
        self.session_id: str | None = getattr(args, "session_id", None)
        self.session_tokens = 0
        self.session_cost = 0.0
        self.turns = 0
        self.last_trace: list[dict[str, Any]] | None = None
        self.last_result: Any = None
        self._named_tool_calling = False
        self._last_task = ""
        # Messages already on the attached session when the current turn began.
        self._stored_before = 0
        self._history_file = _history_dir() / "code_input_history"

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    def _initial_mode(self, args: Any) -> tuple[PermissionMode, bool]:
        """Return ``(mode, explicit)`` from the permission flags, defaulting to ask."""
        if getattr(args, "plan_only", False):
            return PermissionMode.PLAN, True
        if getattr(args, "auto_edit", False):
            return PermissionMode.AUTO_EDIT, True
        if getattr(args, "assume_yes", False):
            return PermissionMode.YES, True
        return default_mode(self.interactive), False

    # ------------------------------------------------------------------
    # Engine / agent construction
    # ------------------------------------------------------------------
    def _resolve_model(self) -> None:
        """Pick a model when the user named none, mirroring the single-shot path."""
        if self.model_id:
            return
        from effgen.cli.commands._shared import _quickstart_suggest_model

        model, suggested_provider, reason = _quickstart_suggest_model()
        self.model_id = model
        if self.provider is None and suggested_provider:
            self.provider = suggested_provider
        if not self.quiet:
            self._say(f"Using model {model} ({reason}); swap with /model.")

    def _build_engine(self, carry_from: CodeEngine | None = None) -> CodeEngine:
        """Build the engine and its agent, carrying conversation memory across a swap.

        The persistent session is attached on the first build only: a ``/model``
        swap carries the live conversation and the same session object across
        instead, so the stored turns are never loaded — and written — twice.
        """
        engine = CodeEngine(
            model=self.model_id or "",
            provider=self.provider,
            workspace=self.workspace,
            mode=self.mode,
            mode_explicit=self.mode_explicit,
            interactive=self.interactive,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            on_event=self._on_event,
            on_diff=self._on_diff,
            project=self.project,
            session_id=self.session_id if carry_from is None else None,
        )
        agent = engine.build_agent()
        if carry_from is not None and carry_from._agent is not None:
            self._carry_memory(carry_from._agent, agent)
        return engine

    def _carry_memory(self, old_agent: Any, new_agent: Any) -> None:
        """Copy the running conversation from *old_agent* into *new_agent*."""
        try:
            from effgen.memory.short_term import MessageRole

            for msg in old_agent.short_term_memory.get_messages():
                if msg.role == MessageRole.USER:
                    new_agent.short_term_memory.add_user_message(msg.content)
                elif msg.role == MessageRole.ASSISTANT:
                    new_agent.short_term_memory.add_assistant_message(msg.content)
        except Exception:  # noqa: BLE001 - memory carry is best-effort
            pass
        if getattr(old_agent, "session", None) is not None:
            new_agent.session = old_agent.session
            new_agent._session_id = getattr(old_agent, "_session_id", self.session_id)

    @property
    def agent(self) -> Any:
        """The live agent, or ``None`` before startup."""
        return self.engine._agent if self.engine is not None else None

    def _tool(self, name: str) -> Any:
        """Return the named gated tool from the live agent, or ``None``."""
        agent = self.agent
        if agent is None:
            return None
        return (getattr(agent, "tools", {}) or {}).get(name)

    # ------------------------------------------------------------------
    # Gate callbacks (live per-action ticks and diffs)
    # ------------------------------------------------------------------
    def _on_event(self, record: ActionRecord) -> None:
        """Print a decided action as a live tick (interactive, non-quiet only)."""
        if self.interactive and not self.quiet:
            print_action(self.cli, record)

    def _on_diff(self, edit: Any) -> None:
        """Show a pending edit's diff, and stage it while a /plan turn runs."""
        if not self.quiet:
            print_diff(self.cli, edit)
        if self._staging and not edit.unchanged:
            self.pending_edits[edit.rel_path] = edit

    # ------------------------------------------------------------------
    # Readline / input
    # ------------------------------------------------------------------
    def _setup_readline(self) -> None:
        try:
            import readline
        except Exception:  # noqa: BLE001 - readline absent (e.g. Windows)
            return
        try:
            readline.read_history_file(str(self._history_file))
        except (FileNotFoundError, OSError):
            pass
        try:
            readline.set_history_length(1000)
        except Exception:  # noqa: BLE001
            pass

        commands = sorted(_SLASH_COMMANDS)

        def _completer(text: str, state: int) -> str | None:
            if not text.startswith("/"):
                return None
            matches = [c + " " for c in commands if c.startswith(text)]
            return matches[state] if state < len(matches) else None

        try:
            readline.set_completer(_completer)
            readline.parse_and_bind("tab: complete")
        except Exception:  # noqa: BLE001
            pass

    def _save_readline(self) -> None:
        try:
            import readline

            readline.write_history_file(str(self._history_file))
        except Exception:  # noqa: BLE001
            pass

    def _read_input(self) -> str | None:
        """Read one (possibly multi-line) entry. ``None`` on EOF."""
        prompt = self._prompt_str() if self.interactive else ""
        cont = "… " if self.interactive else ""
        lines: list[str] = []
        while True:
            try:
                line = input(prompt if not lines else cont)
            except EOFError:
                return None
            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            lines.append(line)
            break
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Run the coding REPL until exit; returns the process exit code."""
        if self.session_id:
            # Read before the model is picked and the engine is built, so a
            # stored model is the one this session starts on.
            self._adopt_session_state(self.session_id)
        self._resolve_model()
        if self.interactive:
            self._banner()
        self._setup_readline()
        try:
            self.engine = self._build_engine()
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Could not start the coding agent: {e}")
            return 1
        self._report_coding_suitability()

        try:
            while True:
                try:
                    user_input = self._read_input()
                except KeyboardInterrupt:
                    if self.interactive:
                        self._say("")
                    continue
                if user_input is None:  # EOF / Ctrl-D
                    if self.interactive:
                        self._say("\nGoodbye!")
                    break
                if not user_input.strip():
                    continue

                action = self._dispatch(user_input.strip())
                if action == "exit":
                    if self.interactive:
                        self._say("Goodbye!")
                    break
                if action == "handled":
                    continue

                self._do_turn(user_input)
        finally:
            self._save_readline()
            if self.agent is not None:
                try:
                    self.agent.close()
                except Exception:  # noqa: BLE001
                    pass
        return 0

