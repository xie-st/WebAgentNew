"""Interactive chat REPL for ``effgen chat``.

This is the most "alive" surface in the CLI, so it aims to feel like a real
chat: streaming tokens with a thinking spinner, a model/tool-aware prompt,
slash commands (``/model``, ``/tools``, ``/status``, ``/cost``, ``/trace``, …),
persistent ↑/↓ history, multiline input, and a per-turn Ctrl-C that cancels the
current answer without dropping you out of the session.

Everything here is presentation/coordination over the existing :class:`Agent`
machinery — it adds no model, provider, or tool behavior. Plain chat (no tools)
streams the model's answer directly for the cleanest live feel; once tools are
attached (via ``/tools``) a turn runs under the shared live-status spinner so
per-tool ticks and an accurate token/cost footer can be shown.

A non-interactive invocation (``echo "hi" | effgen chat``) keeps a plain
fallback: the interactive chrome (banner, prompt echo, ``assistant`` label,
footer, "Goodbye!") is kept off stdout so the answer is the only thing there,
with ``-q`` yielding one answer per input line that a caller can parse directly.
"""

# The session is composed from four sibling modules, and this module stays the
# import path for everything they define. Imports marked ``noqa: F401`` below are
# no longer read here: they are kept so every name this module has ever exposed
# still resolves against it. Do not run ``ruff --fix`` over this block; it would
# read them as unused and drop them.

from __future__ import annotations

import json  # noqa: F401 - re-exported: moved to chat_session, imported from here
import logging  # noqa: F401 - re-exported: moved to chat_session, imported from here
import os  # noqa: F401 - re-exported: moved to chat_commands, imported from here
import sys
import time  # noqa: F401 - re-exported: moved to chat_turn, imported from here
from datetime import datetime  # noqa: F401 - re-exported: moved to chat_session, imported from here
from pathlib import Path  # noqa: F401 - re-exported: moved to chat_session, imported from here
from typing import Any

from effgen.cli import progress as _progress  # noqa: F401 - re-exported: read by the mixins
from effgen.cli.chat_commands import (
    _SLASH_COMMANDS,
    ChatCommandsMixin,
)
from effgen.cli.chat_session import (
    ChatSessionMixin,
    _history_dir,
)
from effgen.cli.chat_turn import ChatTurnMixin
from effgen.cli.chat_view import ChatViewMixin
from effgen.ui.theme import color_enabled


class ChatREPL(
    ChatCommandsMixin,
    ChatSessionMixin,
    ChatTurnMixin,
    ChatViewMixin,
):
    """An interactive chat session over a single :class:`Agent`.

    This class owns what the session *is*: how it was configured, the agent
    it runs, how it reads a line, and the loop that ties them together. The
    command table, one turn, the render layer and the commands that persist a
    session are contributed by the mixins above, each in its own module.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, cli: Any, args: Any) -> None:
        self.cli = cli
        self.args = args
        self.console = getattr(cli, "console", None)

        self.quiet = getattr(args, "quiet", False)
        self.verbose = getattr(args, "verbose", False)
        self.animate = cli._animate(args)
        # Whether styled output (bold/dim/color SGR) is allowed. ``NO_COLOR``
        # turns this off, and the REPL then emits its own surfaces (banner,
        # answer label, footer, /help) as plain text with no escape codes.
        self._color = color_enabled()
        # A piped session (``echo hi | effgen chat``) has no terminal, so the
        # interactive chrome is kept off stdout and the answer is emitted plainly.
        try:
            self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        except (ValueError, OSError):  # pragma: no cover - closed std streams
            self.interactive = False
        # Route all human-facing chatter (banner, warnings, footer, "Goodbye!")
        # to stderr when there is no terminal, so stdout carries only answers and
        # `echo … | effgen chat -q` is directly parseable.
        if not self.interactive:
            try:
                self.cli._human_to_stderr = True
            except Exception:  # noqa: BLE001 - CLI without the stderr routing hook
                pass

        # Whether the model was chosen by the user or fell back to the local
        # default (drives a one-line first-run download note in the banner).
        self._model_defaulted = getattr(args, "model", None) is None
        self.model_id = getattr(args, "model", None) or self.DEFAULT_MODEL
        self.preset = getattr(args, "preset", None)
        # Optional custom persona: steers every reply (e.g. a Socratic tutor),
        # unlike --preset which only labels the session. Carried across /model
        # and /tools rebuilds so the persona persists for the whole session.
        self.system_prompt = getattr(args, "system_prompt", None)
        # Optional persistent session: when set, the chat agent loads prior turns
        # for this id and saves new ones (same store as `effgen run --session-id`
        # and `effgen sessions`), so a customer's conversation can be continued.
        self.session_id = getattr(args, "session_id", None)
        self.temperature = getattr(args, "temperature", None)
        self.max_tokens = getattr(args, "max_tokens", None)
        # Optional guardrail preset name (e.g. "phi", "strict") applied to
        # every turn; None matches the previous no-guardrails default.
        self.guardrails = getattr(args, "guardrails", None)
        # Provider was validated by the caller; keep the canonical name.
        self.provider = getattr(args, "_provider", None) or getattr(args, "provider", None)

        # A tool-bearing preset attaches its tools by default, mirroring
        # `effgen run --preset` (a preset chosen for its tools must actually
        # route to them, not just label the session). Plain chat with no
        # --preset still starts tool-free for the cleanest streaming; users opt
        # into tools with ``/tools <name>``. Incompatible-tool filtering below
        # (`_load_tools` -> `filter_incompatible_tools`) still applies, so this
        # never trips the native-tool incompatibility for a non-Claude model.
        self.tool_names: list[str] = []
        self._preset_system_prompt: str | None = None
        self._preset_temperature: float | None = None
        if self.preset:
            try:
                from effgen.presets import get_preset

                _preset_cfg = get_preset(self.preset)
            except Exception:  # noqa: BLE001 - fall back to label-only on lookup failure
                pass
            else:
                self.tool_names = list(_preset_cfg.tool_names)
                self._preset_system_prompt = _preset_cfg.system_prompt
                self._preset_temperature = _preset_cfg.temperature

        # Tools named with --tools/-t start attached (in addition to any the
        # preset brought), matching `effgen run --tools`. An unknown name is
        # reported with a close-match hint rather than silently ignored.
        requested_tools = getattr(args, "tools", None) or []
        if requested_tools:
            try:
                registry = self.cli.tool_registry
                registry.discover_builtin_tools()
                available = set(registry.list_tools())
            except Exception:  # noqa: BLE001 - never let tool discovery sink startup
                available = set()
            for name in requested_tools:
                if available and name not in available:
                    import difflib

                    hint = difflib.get_close_matches(name, sorted(available), n=1)
                    suffix = f" Did you mean: {hint[0]}?" if hint else ""
                    self.cli.print_warning(
                        f"No tool named '{name}'.{suffix} See: effgen tools list"
                    )
                    continue
                if name not in self.tool_names:
                    self.tool_names.append(name)

        self.agent: Any = None
        self.session_tokens = 0
        self.session_cost = 0.0
        self.turns = 0
        self.last_trace: list[dict[str, Any]] | None = None
        self.last_answer: str = ""
        self._history_file = _history_dir() / "chat_input_history"

    # ------------------------------------------------------------------
    # Agent construction & tools
    # ------------------------------------------------------------------
    def _load_tools(self, names: list[str]) -> list:
        """Load the named builtin tools and drop any incompatible with the model."""
        from effgen.cli._main import filter_incompatible_tools

        registry = self.cli.tool_registry
        registry.discover_builtin_tools()
        available = set(registry.list_tools())
        tools = []
        for name in names:
            if name not in available:
                continue
            try:
                tools.append(registry.get_tool_sync(name))
            except Exception:  # noqa: BLE001 - a single bad tool shouldn't sink chat
                pass
        tools, _skipped = filter_incompatible_tools(
            tools, self.model_id, warn=self.cli.print_warning
        )
        return tools

    def _build_agent(self, carry_from: Any = None) -> Any:
        """Build the chat Agent, optionally carrying memory from a prior agent."""
        from effgen import Agent, AgentConfig

        tools = self._load_tools(self.tool_names)
        config_kwargs: dict[str, Any] = {
            "name": "chat-agent",
            "model": self.model_id,
            "provider": self.provider,
            "tools": tools,
            "temperature": (
                self.temperature
                if self.temperature is not None
                else (
                    self._preset_temperature if self._preset_temperature is not None else 0.7
                )
            ),
            "max_tokens": self.max_tokens,
            "enable_sub_agents": not getattr(self.args, "no_sub_agents", False),
            "enable_streaming": True,
            "guardrails": self.guardrails,
        }
        # An explicit --system-prompt/--persona always wins; otherwise a
        # tool-bearing preset's own system prompt applies (it tells the model
        # when to use the preset's tools, the same as `effgen run --preset`).
        if self.system_prompt:
            config_kwargs["system_prompt"] = self.system_prompt
        elif self._preset_system_prompt:
            config_kwargs["system_prompt"] = self._preset_system_prompt
        # The chat loop prints a failed turn and stays open, so it reads the
        # response rather than letting run() raise out of the REPL.
        config_kwargs.setdefault("raise_on_error", False)
        config = AgentConfig(**config_kwargs)
        # Attach the persistent session only on the FIRST build, so its saved
        # turns are loaded into memory exactly once. On a /model or /tools rebuild
        # we carry memory from the old agent instead (below) and reuse the same
        # Session object — re-loading from disk would double the history.
        if self.session_id and carry_from is None:
            agent = Agent(config, session_id=self.session_id)
        else:
            agent = Agent(config)

        # Carry the running conversation across a /model or /tools rebuild so the
        # new model still "remembers" what was said, and keep persisting to the
        # same session file.
        if carry_from is not None:
            try:
                from effgen.memory.short_term import MessageRole

                for msg in carry_from.short_term_memory.get_messages():
                    if msg.role == MessageRole.USER:
                        agent.short_term_memory.add_user_message(msg.content)
                    elif msg.role == MessageRole.ASSISTANT:
                        agent.short_term_memory.add_assistant_message(msg.content)
            except Exception:  # noqa: BLE001 - memory carry is best-effort
                pass
            # Reuse the live Session object so new turns keep saving to the same id.
            if getattr(carry_from, "session", None) is not None:
                agent.session = carry_from.session
                agent._session_id = getattr(carry_from, "_session_id", self.session_id)
        return agent

    def _rebuild(self) -> None:
        """Rebuild the agent in place, preserving conversation memory."""
        old = self.agent
        self.agent = self._build_agent(carry_from=old)
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass

    @property
    def tool_count(self) -> int:
        """Number of tools attached to the running agent (0 before startup)."""
        agent = self.agent
        if agent is None:
            return 0
        return len(getattr(agent, "tools", {}) or {})

    # ------------------------------------------------------------------
    # readline history + tab completion
    # ------------------------------------------------------------------
    def _setup_readline(self) -> None:
        try:
            import readline
        except Exception:  # noqa: BLE001 - readline absent (e.g. Windows); skip
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
        """Read one (possibly multi-line) user entry. ``None`` on EOF."""
        from effgen.ui.render import ascii_fold

        # A piped session shows no prompt so it can't leak onto answer-only stdout.
        prompt = self._prompt_str() if self.interactive else ""
        cont = ascii_fold("… ", self._human_stream()) if self.interactive else ""
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
        """Run the chat loop until exit; returns the process exit code."""
        # The banner is interactive chrome; a piped run stays answer-only.
        if self.interactive:
            self._banner()
        self._setup_readline()
        try:
            self.agent = self._build_agent()
        except Exception as e:  # noqa: BLE001
            hint = self._unknown_model_message(self.model_id, e)
            if hint:
                self.cli.print_error(f"Could not start chat: {hint}")
            else:
                self.cli.print_error(f"Could not start chat: {e}")
                self._teach_model_error(e)
            return 1

        try:
            while True:
                try:
                    user_input = self._read_input()
                except KeyboardInterrupt:
                    # Ctrl-C at an empty prompt: clear the line, keep the session.
                    if self.interactive:
                        self.cli.print("")
                    continue
                if user_input is None:  # EOF / Ctrl-D
                    if self.interactive:
                        self.cli.print("\nGoodbye!")
                    break
                if not user_input.strip():
                    continue

                action = self._dispatch(user_input.strip())
                if action == "exit":
                    if self.interactive:
                        self.cli.print("Goodbye!")
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


# The mixins are how this module is organised internally, not part of what
# it offers: they are composed into ``ChatREPL`` above and stay importable
# from their own modules. Unbinding them here keeps the set of names this
# module exposes the same as when it was a single file.
del ChatCommandsMixin, ChatSessionMixin, ChatTurnMixin, ChatViewMixin
