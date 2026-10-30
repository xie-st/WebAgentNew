"""Model swapping, saved conversations and the resumable session id.

:mod:`effgen.cli.chat` owns the session itself and mixes these methods into
:class:`~effgen.cli.chat.ChatREPL`, so ``effgen.cli.chat`` stays the import path
callers use. Holds what a chat keeps beyond one turn: the ``/model`` hot-swap and
the messages that explain a failed one, the ``/save`` and ``/load`` snapshots
under :func:`_history_dir`, and ``/session``, which hands the conversation to the
core session store ``effgen sessions`` lists.

:func:`_history_dir` lives here because this module owns what that directory
holds. :mod:`effgen.cli.code.repl` and :mod:`effgen.cli.code.repl_session` bind
it at import through the ``effgen.cli.chat`` facade, so an ``effgen code``
session reads and writes the same directory an ``effgen chat`` session does.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from effgen.cli import progress as _progress


def _history_dir() -> Path:
    """Return ``~/.effgen/history`` (configurable via ``EFFGEN_HOME``)."""
    home = os.environ.get("EFFGEN_HOME")
    base = Path(home) if home else Path.home() / ".effgen"
    d = base / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ChatSessionMixin:
    """The persistence half of :class:`~effgen.cli.chat.ChatREPL`.

    Every method reads state ``ChatREPL.__init__`` sets — ``self.cli``,
    ``self.model_id``, ``self.provider``, ``self.tool_names``, ``self.agent`` and
    ``self.session_id`` — and reaches a new agent through ``self._rebuild()``,
    which the facade owns.
    """

    # ------------------------------------------------------------------
    # Model swapping
    # ------------------------------------------------------------------
    def _resolve_swap_target(self, new_id: str) -> tuple[str | None, str]:
        """Resolve a ``/model`` argument the same way a fresh ``chat -m <id>``
        (no ``--provider``) would.

        A ``provider:model`` prefix pins that provider explicitly, mirroring
        startup resolution. A bare id clears any provider left over from a
        prior explicit ``--provider`` flag, so the loader's own auto-detection
        (catalog lookup / known-prefix matching) picks the right adapter
        instead of forcing the new id through the *previous* session's
        provider — which silently keeps the old adapter and fails every turn.
        """
        if ":" in new_id:
            from effgen.models.registry import ProviderRegistry

            prefix, rest = new_id.split(":", 1)
            try:
                known = ProviderRegistry.list_providers()
            except Exception:  # noqa: BLE001
                known = []
            if prefix in known and rest:
                return prefix, rest
        return None, new_id

    def _unknown_model_message(self, attempted: str, exc: Exception) -> str | None:
        """Return a teach message when *attempted* resolves to no known model.

        A bare id (no ``provider:`` prefix, not a ``org/repo`` path) that fails
        to load is almost always a mistyped/nonexistent model rather than an
        auth or download problem, so it earns a "check the id / name a provider"
        hint instead of the loader's Hugging Face token message.
        """
        if ":" in attempted or "/" in attempted:
            return None
        m = str(exc).lower()
        markers = (
            "not a valid model identifier",
            "is not a local folder",
            "not found",
            "404",
            "model_not_found",
            "no model",
        )
        if any(s in m for s in markers):
            return (
                f"No model '{attempted}'. Check the id with `effgen models list`, "
                "or name a provider with `/model provider:id` "
                "(e.g. /model openai:gpt-5-nano)."
            )
        return None

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            self.cli.print(f"Active model: {self.model_id}")
            self.cli.print("Swap with:  /model <model-id>   (e.g. /model gpt-5-nano)")
            return
        new_id = arg.split()[0]
        old_id = self.model_id
        old_provider = self.provider
        new_provider, resolved_id = self._resolve_swap_target(new_id)
        self.model_id = resolved_id
        self.provider = new_provider
        # A failed swap is fully handled below (a styled ✗ message + the
        # restore rebuild) — suppress the library's own ERROR-level log line
        # for it so the transcript shows one failure presentation, not two.
        # --verbose still surfaces it.
        _effgen_logger = logging.getLogger("effgen")
        _prev_level = _effgen_logger.level
        if not self.verbose:
            _effgen_logger.setLevel(logging.CRITICAL)
        try:
            self._rebuild()
        except Exception as e:  # noqa: BLE001
            self.model_id = old_id
            self.provider = old_provider
            # An unresolvable id gets a "check the id / name a provider" hint
            # rather than the loader's raw download/auth message.
            hint = self._unknown_model_message(new_id, e)
            if hint:
                self.cli.print_error(hint)
            else:
                self.cli.print_error(f"Could not switch to '{new_id}': {e}")
                self._teach_model_error(e)
            # Restore the working agent.
            try:
                self._rebuild()
            except Exception:  # noqa: BLE001
                pass
            return
        finally:
            _effgen_logger.setLevel(_prev_level)
        self.cli.print_success(
            f"Switched model: {_progress.short_model_label(new_id)} "
            f"(conversation kept)"
        )

    # ------------------------------------------------------------------
    # The resumable session id
    # ------------------------------------------------------------------
    def _cmd_session(self, arg: str) -> None:
        """Show, or begin persisting under, the resumable session id.

        This id maps to the core Session store that ``effgen sessions`` lists and
        ``effgen chat --session-id`` / ``effgen run --session-id`` resume — a
        different store from the file-based ``/save`` + ``/load`` snapshots.
        """
        if not arg:
            if self.session_id:
                self.cli.print(f"Session id: {self.session_id}")
                self.cli.print(
                    "Resume later with:  effgen chat --session-id "
                    f"{self.session_id}   (also shown by `effgen sessions`)."
                )
            else:
                self.cli.print("This chat is not being saved to the session store.")
                self.cli.print(
                    "Start persisting:  /session <id>   "
                    "(then `effgen sessions` and --session-id can resume it)."
                )
            return
        name = arg.split()[0]
        if self.session_id == name:
            self.cli.print(f"Already on session '{name}'.")
            return
        if self.session_id:
            self.cli.print_warning(
                f"Already saving to '{self.session_id}'. Restart with "
                f"`effgen chat --session-id {name}` to switch id."
            )
            return
        try:
            from effgen.core.session import Session

            session = Session.load_or_create(name)
            for msg in self._dump_history():
                if msg["role"] == "user":
                    session.add_user_message(msg["content"])
                elif msg["role"] == "assistant":
                    session.add_assistant_message(msg["content"])
            session.save()
            self.session_id = name
            if self.agent is not None:
                self.agent.session = session
                self.agent._session_id = name
            self.cli.print_success(
                f"Saving this conversation as session '{name}'. "
                f"Resume with `effgen chat --session-id {name}`."
            )
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"Could not start session '{name}': {e}")

    # ------------------------------------------------------------------
    # Saved conversations
    # ------------------------------------------------------------------
    def _cmd_save(self, arg: str) -> None:
        history = self._dump_history()
        if not history:
            self.cli.print("Nothing to save yet.")
            return
        name = arg.split()[0] if arg else datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.endswith(".json"):
            name += ".json"
        path = _history_dir() / f"chat_{name}"
        with open(path, "w") as f:
            json.dump(
                {"model": self.model_id, "tools": self.tool_names, "messages": history},
                f,
                indent=2,
            )
        self.cli.print_success(f"Saved to {path}")

    def _cmd_load(self, arg: str) -> None:
        files = sorted(_history_dir().glob("chat_*.json"), reverse=True)
        if not files:
            self.cli.print("No saved conversations found.")
            return
        target: Path | None = None
        if not arg:
            self._list_saved()
            self.cli.print("Load with:  /load <name|number>")
            return
        token = arg.split()[0]
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(files):
                target = files[idx]
        else:
            cand = token if token.endswith(".json") else token + ".json"
            for f in files:
                if f.name == cand or f.name == f"chat_{cand}":
                    target = f
                    break
        if target is None:
            self.cli.print_warning(f"No saved conversation matching '{token}'.")
            return
        try:
            with open(target) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.cli.print_error(f"Could not read {target.name}: {e}")
            return
        messages = data.get("messages", data if isinstance(data, list) else [])
        saved_model = data.get("model")
        if saved_model and saved_model != self.model_id:
            self.model_id = saved_model
        self.tool_names = list(data.get("tools", []) or [])
        self._rebuild()
        # Replay the saved turns into memory.
        try:
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    self.agent.short_term_memory.add_user_message(content)
                elif role in ("assistant", "agent"):
                    self.agent.short_term_memory.add_assistant_message(content)
        except Exception:  # noqa: BLE001
            pass
        self.cli.print_success(
            f"Loaded {target.name} ({len(messages)} messages, model {self.model_id})."
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _dump_history(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        try:
            from effgen.memory.short_term import MessageRole

            for msg in self.agent.short_term_memory.get_messages():
                role = "user" if msg.role == MessageRole.USER else (
                    "assistant" if msg.role == MessageRole.ASSISTANT else "system"
                )
                out.append({"role": role, "content": msg.content})
        except Exception:  # noqa: BLE001
            pass
        return out

    def _list_saved(self) -> None:
        files = sorted(_history_dir().glob("chat_*.json"), reverse=True)
        if not files:
            self.cli.print("No saved conversations found.")
            return
        self.cli.print("Saved conversations:")
        for i, f in enumerate(files[:20], 1):
            self.cli.print(f"  {i}. {f.name}  ({f.stat().st_size} bytes)")
