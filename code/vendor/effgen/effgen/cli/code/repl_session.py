"""The slash commands that retune, inspect or persist a coding session.

:mod:`effgen.cli.code.repl` owns the session itself and mixes these methods into
:class:`~effgen.cli.code.repl.CodeREPL`, so ``effgen.cli.code.repl`` stays the
import path callers use. Holds the commands that act on the session rather than
on the working set: the permission mode and the model hot-swap, the cost and
reasoning-trace views, the git surface and the confirmed commit, the memory
reset/clear/compact, the save/session/load round trip, and the environment
check.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from effgen.cli.chat import _history_dir
from effgen.cli.code.engine import workspace_env
from effgen.cli.code.permissions import MODE_DESCRIPTIONS, PermissionMode
from effgen.cli.code.project import staged_diff


class CodeSessionMixin:
    """The session-control commands of :class:`~effgen.cli.code.repl.CodeREPL`.

    Every method reads state ``CodeREPL.__init__`` sets — ``self.engine``,
    ``self.mode``, ``self.model_id``, ``self.session_files`` and the session
    counters — and prints through the view methods.
    """

    def _cmd_mode(self, arg: str) -> None:
        if not arg:
            self._say(f"Permission mode: {self.mode.value} — {MODE_DESCRIPTIONS[self.mode]}")
            self._say("Change with:  /mode ask | auto-edit | yes | plan")
            return
        name = arg.split()[0].lower()
        try:
            new_mode = PermissionMode(name)
        except ValueError:
            self._status(
                "warning",
                f"Unknown mode '{name}'. Choose: ask, auto-edit, yes, plan."
            )
            return
        self.mode = new_mode
        self.mode_explicit = True
        if self.engine is not None:
            self.engine.gate.mode = new_mode
        self._save_coding_state()
        self._status("success", f"Permission mode: {new_mode.value} — {MODE_DESCRIPTIONS[new_mode]}")

    def _resolve_swap_target(self, new_id: str) -> tuple[str | None, str]:
        """Resolve a ``/model`` argument like a fresh ``code -m <id>`` would."""
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

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            self._say(f"Active model: {self.model_id}")
            self._say("Swap with:  /model <model-id>   (e.g. /model gpt-5-nano)")
            return
        new_id = arg.split()[0]
        old_engine = self.engine
        old_id, old_provider = self.model_id, self.provider
        new_provider, resolved_id = self._resolve_swap_target(new_id)
        self.model_id, self.provider = resolved_id, new_provider

        _effgen_logger = logging.getLogger("effgen")
        _prev_level = _effgen_logger.level
        if not self.verbose:
            _effgen_logger.setLevel(logging.CRITICAL)
        try:
            self.engine = self._build_engine(carry_from=old_engine)
        except Exception as e:  # noqa: BLE001
            self.model_id, self.provider = old_id, old_provider
            self.engine = old_engine
            self._status("error", f"Could not switch to '{new_id}': {e}")
            return
        finally:
            _effgen_logger.setLevel(_prev_level)
        if old_engine is not None and old_engine._agent is not None:
            try:
                old_engine._agent.close()
            except Exception:  # noqa: BLE001
                pass
        from effgen.cli import progress as _progress

        self._status(
            "success",
            f"Switched model: {_progress.short_model_label(new_id)} (conversation kept)"
        )

    def _cmd_cost(self) -> None:
        from effgen.ui.render import format_cost

        self.cli.print_header("Session cost")
        self._say(f"Turns: {self.turns}")
        self._say(f"Tokens: {self.session_tokens:,}")
        self._say(
            f"Cost: {format_cost(self.session_cost) if self.session_cost else '$0.00'}"
        )

    def _cmd_trace(self) -> None:
        if not self.last_trace:
            self._say("No turn to trace yet.")
            return
        from effgen.cli import progress as _progress
        from effgen.ui.render import ascii_fold

        self.cli.print_header("Last turn — reasoning trace")
        lines = _progress.execution_trace_lines(self.last_trace, stream=self.cli._human_stream())
        if not lines:
            self._say("(no detailed steps recorded for this turn)")
            return
        for style, text in lines:
            self._say(ascii_fold(text, self.cli._human_stream()), style=style)

    def _cmd_git(self, arg: str) -> None:
        """Git for the workspace: the read-only surface, plus a confirmed commit.

        ``status``/``diff``/``log``/``branch``/``show``/``remote`` go through the
        read-only ``git`` tool. ``staged`` shows the staged patch and adds it to
        the conversation so the next turn can work from it. ``commit`` is the one
        action that changes the repository, and it asks first.
        """
        from effgen.tools.builtin.devops import GitTool

        parts = arg.split(maxsplit=1) if arg else []
        op = parts[0].lower() if parts else "status"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if op == "staged":
            self._git_staged()
            return
        if op == "commit":
            self._git_commit(rest)
            return

        if op not in GitTool.ALLOWED:
            self._status(
                "warning",
                f"'{op}' is not available. Choose a read-only operation "
                f"({', '.join(sorted(GitTool.ALLOWED))}), 'staged' to pull the "
                "staged diff into context, or 'commit' to record this session's "
                "edits after confirming."
            )
            return
        self.cli.print_header(f"Git — {op}")
        try:
            result = asyncio.run(GitTool().execute(operation=op, cwd=str(self.workspace)))
        except Exception as e:  # noqa: BLE001
            self._status("error", f"git {op} failed: {e}")
            return
        output = getattr(result, "output", None)
        if isinstance(output, dict):
            body = str(output.get("stdout") or "")
            if int(output.get("returncode", 0) or 0) != 0 and not body.strip():
                stderr = (output.get("stderr") or "").strip()
                # Match the message, not an empty stderr: real git always writes
                # a sentence here, so the friendlier wording below was
                # unreachable. The common case gets the session's own voice; a
                # genuine git failure keeps git's text, which is what a reader
                # needs when git itself is the problem.
                if _is_not_a_repository(stderr):
                    self._status(
                        "warning",
                        f"{self.workspace} is not a git repository, so there is "
                        "nothing to show. Run 'git init' there, or open the "
                        "session in a repository."
                    )
                else:
                    self._status(
                        "warning", stderr or "git is unavailable."
                    )
                return
        else:
            body = str(output or "")
        self._say(body.rstrip() or "(no output)")
        self._say(
            "  'staged' pulls the staged diff into the conversation; "
            "'commit' records this session's edits after confirming.",
            style="effgen.muted",
        )

    def _git_staged(self) -> None:
        """Show the staged patch and add it to the conversation as context."""
        repo = self.project.repo
        if repo is None:
            self._status("warning", f"{self.workspace} is not inside a git repository.")
            return
        patch = staged_diff(repo.root)
        if not patch.strip():
            self._say("Nothing is staged in this repository.")
            return
        self._say(patch.rstrip())
        self._feed_back(
            "The staged changes in this repository (git diff --cached):\n" + patch
        )
        self._status("success", "The staged diff is now part of this conversation.")

    def _git_commit(self, message: str) -> None:
        """Commit the files this session wrote, after an explicit confirmation.

        Only the session's own paths are committed; anything else the user has
        staged stays staged. The permission gate asks before the commit runs, so
        a commit never happens on an implied yes.
        """
        assert self.engine is not None
        if not self.session_files:
            self._say(
                "This session has written no files yet, so there is nothing to commit."
            )
            return
        plan = self.engine.plan_commit(
            self.session_files, task=self._last_task, message=message or None
        )
        if plan.error:
            self._status("warning", plan.error)
            return
        self._say(plan.describe())
        self._say(f'Message: "{plan.subject}"')
        if plan.other_staged:
            self._status(
                "warning",
                f"{len(plan.other_staged)} other path(s) are staged; they stay "
                "staged and are not part of this commit: "
                + ", ".join(plan.other_staged[:5])
                + ("…" if len(plan.other_staged) > 5 else ""),
            )
        outcome = self.engine.perform_commit(plan)
        if outcome.success:
            self._status(
                "success",
                f"Committed {len(outcome.paths)} file(s) as "
                f"{outcome.commit or 'HEAD'}: {plan.subject}",
            )
            self.session_files.clear()
            self.project = self.engine.load_project(refresh=True)
        else:
            self._status("warning", f"Nothing was committed: {outcome.detail}")

    def _cmd_reset(self) -> None:
        try:
            self.agent.reset_memory()
        except Exception:  # noqa: BLE001
            pass
        self.last_trace = None
        self.last_result = None
        self._status("success", "Conversation memory cleared (files-in-context kept).")

    def _cmd_clear(self) -> None:
        try:
            self.agent.reset_memory()
        except Exception:  # noqa: BLE001
            pass
        self.context_files.clear()
        self.pending_edits.clear()
        self.last_trace = None
        self.last_result = None
        note = "Live context reset (memory, files, staged edits cleared)."
        if self.session_id:
            note += f" Session '{self.session_id}' stays on disk."
        self._status("success", note)

    def _cmd_compact(self, arg: str) -> None:
        """Summarize the conversation so far and replace memory with the summary.

        The summarization turn runs with the tools detached. ``/compact`` is
        described to the user as "summarize", and with the tool loop live it
        could — and on a real session did — execute code and write a file while
        it was supposed to be writing a summary, on whatever permission mode the
        session happened to be in. Nothing in the summary needs a tool: the whole
        transcript is in the prompt.
        """
        history = self._dump_history()
        if len(history) < 2:
            self._say("Not enough conversation to compact yet.")
            return
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        extra = f" Focus on: {arg}." if arg else ""
        prompt = (
            "Summarize the coding conversation below into a compact brief that lets "
            "the work continue: the goal, decisions made, files touched, and what is "
            f"left to do.{extra}\n\n{transcript}"
        )
        try:
            with workspace_env(self.workspace), _tools_detached(self.agent):
                response = self.agent.run(prompt)
            summary = response.output or ""
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Could not compact: {e}")
            return
        if not summary.strip():
            self._status("warning", "The summary came back empty; memory is unchanged.")
            return
        try:
            self.agent.reset_memory()
            self.agent.short_term_memory.add_assistant_message(
                f"[compacted context]\n{summary}"
            )
        except Exception:  # noqa: BLE001
            pass
        self._status("success", "Conversation compacted into a summary.")
        console = self.cli._human()
        if console:
            from effgen.ui.render import answer_surface

            answer_surface(summary, framed=True, title="Compacted context", console=console)
        else:
            self._say(summary)

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

    def _cmd_save(self, arg: str) -> None:
        history = self._dump_history()
        if not history:
            self._say("Nothing to save yet.")
            return
        name = arg.split()[0] if arg else datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.endswith(".json"):
            name += ".json"
        path = _history_dir() / f"code_{name}"
        payload = {
            "model": self.model_id,
            "workspace": str(self.workspace),
            "mode": self.mode.value,
            "files_in_context": list(self.context_files),
            "files_written": list(self.session_files),
            "messages": history,
        }
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            self._status("error", f"Could not save: {e}")
            return
        self._status("success", f"Saved to {path}")

    def _report_coding_suitability(self) -> None:
        """Say once, before the first turn, when this model is a poor fit for coding.

        Runs after the agent is built, so what the model reports about its own
        tool calling is answerable. It never blocks: a session on a model that
        cannot write files still starts, having said so.
        """
        if self.quiet:
            return
        from effgen.models._coding import coding_suitability

        note = coding_suitability(
            self.model_id or "", self.provider, getattr(self.agent, "model", None)
        ).note()
        if note:
            self._status("warning", note)

    # ------------------------------------------------------------------
    # The resumable session
    # ------------------------------------------------------------------
    def _adopt_session_state(self, name: str) -> None:
        """Adopt the stored coding state of session *name*, under the restore rules.

        Called before the model is picked and the engine is built, so a stored
        model is the one the session starts on. The conversation itself is
        restored separately, by attaching the session to the agent.

        Nothing here silently widens what the run may do: the workspace always
        comes from the command line or the current directory, and a stored
        permission mode is adopted only on a terminal and only when the caller
        named no permission flag.
        """
        from effgen.cli.code.engine import CODING_METADATA_KEY

        try:
            from effgen.core.session import Session

            session = Session.load_or_create(name)
        except Exception as e:  # noqa: BLE001 - a bad store never blocks a session
            self._status("warning", f"Could not read session '{name}': {e}")
            return

        turns = len(session.messages) // 2
        state = session.metadata.get(CODING_METADATA_KEY) or {}
        if not session.messages and not state:
            self._say(f"Starting a new session '{name}'; it is saved as it grows.")
            return

        stored_ws = str(state.get("workspace") or "")
        if stored_ws and Path(stored_ws) != self.workspace:
            self._status(
                "warning",
                f"Session '{name}' was last used in {stored_ws}; this run works "
                f"in {self.workspace} and keeps it.",
            )

        if not self.model_id:
            model = state.get("model") or session.metadata.get("model")
            if model:
                self.model_id = model
                self.provider = state.get("provider") or self.provider
                self._model_defaulted = False
                self._say(f"Using the session's model {model}; swap with /model.")

        kept, missing = self._existing_context_files(state.get("files_in_context") or [])
        self.context_files = kept
        if missing:
            self._status(
                "warning",
                f"{len(missing)} file(s) in the stored context no longer exist "
                f"and were dropped: {', '.join(missing)}",
            )
        self.session_files = [
            rel for rel in (state.get("files_written") or []) if isinstance(rel, str)
        ]

        stored_mode = state.get("permission_mode")
        if stored_mode and not self.mode_explicit and self.interactive:
            try:
                self.mode = PermissionMode(stored_mode)
                self._say(f"Permission mode restored: {self.mode.value}")
            except ValueError:
                pass
        elif stored_mode and stored_mode != self.mode.value:
            self._say(
                f"The session was last in {stored_mode} mode; this run uses "
                f"{self.mode.value}. Change it with /mode."
            )

        self._say(
            f"Resumed '{name}' ({turns} turn(s), {len(kept)} file(s) in context)."
        )

    def _existing_context_files(self, stored: list) -> tuple[list[str], list[str]]:
        """Split *stored* context paths into those still on disk and those gone."""
        kept: list[str] = []
        missing: list[str] = []
        for rel in stored:
            if not isinstance(rel, str):
                continue
            try:
                exists = (self.workspace / rel).is_file()
            except OSError:  # pragma: no cover - unreadable path
                exists = False
            (kept if exists else missing).append(rel)
        return kept, missing

    def _save_coding_state(self) -> None:
        """Record where this session is working, under which permissions, on what.

        The conversation is stored by :meth:`_record_turn`; this is the coding
        half of the record, refreshed after every turn and after ``/add``,
        ``/drop`` and ``/mode`` so it is never a stale first snapshot.
        """
        if self.engine is None or not self.session_id:
            return
        self.engine.save_coding_state(
            files_in_context=self.context_files, files_written=self.session_files
        )

    def _record_turn(self, result: Any) -> None:
        """Store this turn on the session, and store the line the user typed.

        A turn that runs to completion before it is rendered is written back by
        the agent itself; a streamed turn renders the answer as it arrives and
        writes nothing, so the conversation has to be recorded here or a resumed
        session would carry its coding state and none of what was said.

        Either way the stored user message is the line the person typed, not the
        task the model received — that one has the files-in-context inlined in
        front of it, and a resumed conversation should not replay file bodies as
        the user's words. The files that were sent are recorded beside it.
        """
        if self.engine is None or not self.session_id:
            return
        session = getattr(self.agent, "session", None)
        if session is None or not getattr(result, "success", False):
            return
        answer = getattr(result, "answer", "") or ""
        if not answer:
            return
        messages = session.messages
        if len(messages) >= self._stored_before + 2 and messages[-1].get("role") in (
            "assistant", "agent"
        ):
            # The agent stored the turn; replace the composed task it recorded.
            user = messages[-2]
            if user.get("role") != "user":
                return
            user["content"] = self._last_task
        else:
            session.add_message("user", self._last_task)
            session.add_message("assistant", answer)
        meta = messages[-2].setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["context_files"] = list(self.context_files)
        try:
            session.save()
        except Exception as exc:  # noqa: BLE001 - a store failure never fails a turn
            self._status("warning", f"Could not save the session: {exc}")

    def _cmd_session(self, arg: str) -> None:
        """Show, resume, or begin persisting under, the resumable session id."""
        if not arg:
            if self.session_id:
                self._say(f"Session id: {self.session_id}")
                self._say(
                    "It is listed by `effgen sessions`; the conversation is saved as it grows."
                )
                self._say(f"Continue it later with:  effgen code --session-id {self.session_id}")
            else:
                self._say("This session is not being saved to the session store.")
                self._say("Start persisting:  /session <id>")
            return
        name = arg.split()[0]
        if self.session_id == name:
            self._say(f"Already on session '{name}'.")
            return
        if self.session_id:
            self._status("warning", f"Already saving to '{self.session_id}'.")
            return
        try:
            from effgen.core.session import Session

            session = Session.load_or_create(name)
            if session.messages:
                self._resume_stored_session(name, session)
            else:
                self._begin_session(name, session)
        except Exception as e:  # noqa: BLE001
            self._status("error", f"Could not start session '{name}': {e}")

    def _begin_session(self, name: str, session) -> None:
        """Start persisting the live conversation under a session id with no turns."""
        for msg in self._dump_history():
            if msg["role"] == "user":
                session.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                session.add_assistant_message(msg["content"])
        session.metadata["model"] = self.model_id
        self._attach_session(name, session)
        self._save_coding_state()
        self._status("success", f"Saving this session as '{name}'.")

    def _resume_stored_session(self, name: str, session) -> None:
        """Load a stored session's turns into memory instead of appending to them.

        Appending the live conversation to a session that already has one would
        store both twice, so the stored turns replace live memory and anything
        said in this process before the command is reported as dropped.
        """
        live = len(self._dump_history())
        try:
            self.agent.reset_memory()
            for msg in session.messages:
                if msg.get("role") == "user":
                    self.agent.short_term_memory.add_user_message(msg.get("content", ""))
                elif msg.get("role") in ("assistant", "agent"):
                    self.agent.short_term_memory.add_assistant_message(msg.get("content", ""))
        except Exception:  # noqa: BLE001 - memory hydration is best-effort
            pass
        self._attach_session(name, session)
        self._adopt_session_state(name)
        if live:
            self._status(
                "warning",
                f"{live} message(s) from this process were replaced by the "
                f"stored conversation of '{name}'.",
            )

    def _attach_session(self, name: str, session) -> None:
        """Point this session and its agent at *session*, saving as it grows."""
        session.metadata.setdefault("model", self.model_id)
        session.save()
        self.session_id = name
        if self.agent is not None:
            self.agent.session = session
            self.agent._session_id = name

    def _cmd_load(self, arg: str) -> None:
        files = sorted(_history_dir().glob("code_*.json"), reverse=True)
        if not files:
            self._say("No saved coding sessions found.")
            return
        if not arg:
            self._say("Saved coding sessions:")
            for i, f in enumerate(files[:20], 1):
                self._say(f"  {i}. {f.name}  ({f.stat().st_size} bytes)")
            self._say("Load with:  /load <name|number>")
            return
        token = arg.split()[0]
        target: Path | None = None
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(files):
                target = files[idx]
        else:
            cand = token if token.endswith(".json") else token + ".json"
            for f in files:
                if f.name in (cand, f"code_{cand}"):
                    target = f
                    break
        if target is None:
            self._status("warning", f"No saved session matching '{token}'.")
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self._status("error", f"Could not read {target.name}: {e}")
            return
        messages = data.get("messages", [])
        saved_model = data.get("model")
        if saved_model and saved_model != self.model_id:
            self.model_id = saved_model
            try:
                self.engine = self._build_engine()
            except Exception as e:  # noqa: BLE001
                self._status("error", f"Could not load model '{saved_model}': {e}")
                return
        self.context_files = list(data.get("files_in_context", []) or [])
        self.session_files = list(data.get("files_written", []) or [])
        saved_mode = data.get("mode")
        if saved_mode:
            try:
                self.mode = PermissionMode(saved_mode)
                if self.engine is not None:
                    self.engine.gate.mode = self.mode
            except ValueError:
                pass
        try:
            self.agent.reset_memory()
            for msg in messages:
                if msg.get("role") == "user":
                    self.agent.short_term_memory.add_user_message(msg.get("content", ""))
                elif msg.get("role") in ("assistant", "agent"):
                    self.agent.short_term_memory.add_assistant_message(msg.get("content", ""))
        except Exception:  # noqa: BLE001
            pass
        self._status(
            "success",
            f"Loaded {target.name} ({len(messages)} messages, "
            f"{len(self.context_files)} file(s) in context, model {self.model_id})."
        )

    def _cmd_doctor(self) -> None:
        """Report the provider keys, the system, and this session's readiness.

        The workspace is handed to ``doctor`` so its coding section describes
        the directory this session is confined to rather than the one the
        process was started in.
        """
        from effgen.cli._main import _handle_doctor_command

        class _DoctorArgs:
            doctor_provider = self.provider
            live = False
            output_json = False
            workspace = str(self.workspace)

        try:
            _handle_doctor_command(_DoctorArgs())
        except Exception as e:  # noqa: BLE001
            self._status("error", f"doctor failed: {e}")


def _is_not_a_repository(stderr: str) -> bool:
    """Whether git's own message says the directory is not a repository.

    Git phrases it as ``fatal: not a git repository (or any of the parent
    directories): .git``. Matched on the sentence rather than on an empty
    stderr, which never happens and which is why the session's own wording for
    this case used to be unreachable.
    """
    lowered = (stderr or "").lower()
    return "not a git repository" in lowered or "not a working tree" in lowered


@contextmanager
def _tools_detached(agent: Any) -> "Iterator[None]":
    """Run a turn on *agent* with no tools attached, then put them back.

    Used by ``/compact``: a command described as "summarize" must not be able to
    modify the workspace while it summarizes. With no tools the agent takes its
    direct-inference path, so there is no loop to dispatch a call from — the
    guarantee is structural rather than a permission check that a later turn
    boundary could reopen.

    Args:
        agent: The agent whose tools are detached for the duration.

    Yields:
        None, with the agent holding no tools.
    """
    saved = getattr(agent, "tools", None)
    try:
        agent.tools = {}
        yield
    finally:
        if saved is not None:
            agent.tools = saved
