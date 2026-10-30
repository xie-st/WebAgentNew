"""One coding turn, from the composed task to the notes printed after it.

:mod:`effgen.cli.code.repl` owns the session itself and mixes these methods into
:class:`~effgen.cli.code.repl.CodeREPL`, so ``effgen.cli.code.repl`` stays the
import path callers use. Holds what happens between the user's line and the next
prompt: the files-in-context are folded into the task, the agent runs under the
live status line, the token and cost deltas are measured, and the turn's written
files, refusals, withheld actions and iteration cap are surfaced.
"""

from __future__ import annotations

import time
from typing import Any

from effgen.cli.code.engine import workspace_env
from effgen.cli.code.permissions import PermissionMode


class CodeTurnMixin:
    """The turn half of :class:`~effgen.cli.code.repl.CodeREPL`.

    Every method reads state ``CodeREPL.__init__`` sets — ``self.engine``,
    ``self.context_files``, ``self.workspace`` and the session counters — and
    prints through the view methods.
    """

    def _compose_task(self, message: str) -> str:
        """Prepend the live content of any files-in-context to *message*.

        While a ``/review`` turn is running, the change under review follows,
        because a read-only turn holds no tool that could go and find it.
        """
        composed = self._with_context_files(message)
        engine = self.engine
        if engine is not None and engine.read_only:
            return engine.compose_review_task(composed)
        return composed

    def _with_context_files(self, message: str) -> str:
        """Return *message* behind the live content of the files in context."""
        if not self.context_files:
            return message
        blocks: list[str] = []
        for rel in self.context_files:
            content = self._read_context_file(rel)
            if content is None:
                continue
            blocks.append(f"=== {rel} ===\n{content}")
        if not blocks:
            return message
        joined = "\n\n".join(blocks)
        return (
            "Files currently in context (their content on disk right now):\n\n"
            f"{joined}\n\n---\n\n{message}"
        )

    def _read_context_file(self, rel: str) -> str | None:
        try:
            return (self.workspace / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _snapshot(self) -> tuple[int, float]:
        m = getattr(self.agent, "model", None)
        return (
            int(getattr(m, "total_tokens", 0) or 0),
            float(getattr(m, "total_cost", 0.0) or 0.0),
        )

    def _do_turn(self, message: str) -> None:
        """Run one coding turn and render the result."""
        assert self.engine is not None
        task = self._compose_task(message)
        self._last_task = message
        # How much of the conversation was already stored, so the turn is
        # recorded once whether the agent wrote it back or the streamed path ran.
        session = getattr(self.agent, "session", None)
        self._stored_before = len(session.messages) if session is not None else 0
        self.engine.gate.begin_turn()
        tok0, cost0 = self._snapshot()
        t0 = time.monotonic()
        self.last_trace = None
        # What the turn already put on screen as it was written, so the answer
        # is not printed a second time when the record is rendered.
        self._streamed_answer = ""

        try:
            with workspace_env(self.workspace):
                response = self._run_agent(task)
        except KeyboardInterrupt:
            self._status("warning", "Interrupted; no further actions were taken.")
            return
        except Exception as e:  # noqa: BLE001
            self._status("error", f"{e}")
            return

        result = self.engine.result_from_response(task, response)
        self.last_result = result
        self._note_written(result.files_written)
        try:
            self.last_trace = response.execution_trace or None
        except Exception:  # noqa: BLE001
            self.last_trace = None

        self._render_answer(result, already_streamed=bool(self._streamed_answer))
        self.turns += 1
        elapsed = time.monotonic() - t0
        tok1, cost1 = self._snapshot()
        rtok = int(getattr(response, "tokens_used", 0) or 0) or max(tok1 - tok0, 0)
        rcost = result.cost_usd if result.cost_usd is not None else max(cost1 - cost0, 0.0)
        self.session_tokens += max(rtok, 0)
        self.session_cost += max(rcost or 0.0, 0.0)
        if not self.quiet:
            self._print_footer(elapsed, rtok, rcost or 0.0)
        self._record_turn(result)
        self._save_coding_state()
        self._post_turn_notes(result)

    def _note_written(self, paths: list[str]) -> None:
        """Record the paths just written and re-read the project from disk.

        The paths are remembered for ``/git commit``, and the layout and status
        the agent is working from are rebuilt so the next turn sees the files the
        last one created rather than a listing that predates them.
        """
        if not paths:
            return
        for rel in paths:
            if rel not in self.session_files:
                self.session_files.append(rel)
        if self.engine is not None:
            self.project = self.engine.load_project(refresh=True)

    def _run_agent(self, task: str) -> Any:
        """Run one turn under the live status line ``chat`` uses.

        The turn drives the tool loop with the model's native tool calling where
        the provider supports it. The status line follows the execution events
        (``Running file_operations…``), and each write's diff and each decided
        action print through it as they happen, so the session shows its work
        while the loop runs.

        On an animating terminal, a model whose adapter streams its tool calls
        takes the streamed loop, so the answer appears as it is written rather
        than in one block when the turn ends. Everything else — a piped or
        redirected session, ``--quiet``, ``--json``, a model without streamed
        tool calls — runs the same blocking turn as before.
        """
        from effgen.cli import progress as _progress
        from effgen.core.agent import AgentMode

        console = self.cli._human()
        animate = console is not None and _progress.animation_enabled(
            quiet=self.quiet,
            no_animation=bool(getattr(self.args, "no_animation", False)),
            stream=self.cli._human_stream(),
        )
        if not animate:
            if not self.quiet and self.interactive:
                self._banner_line("· working…", style="dim")
            return self.agent.run(task, mode=AgentMode.AUTO)

        if self._can_stream_turn():
            response = self._stream_agent(task, console)
            if response is not None:
                return response

        with _progress.LiveStatus(
            console,
            model_label=_progress.short_model_label(self.model_id),
            reasoning=_progress.is_reasoning_agent(self.agent),
            tracker=getattr(self.agent, "execution_tracker", None),
            hint="Ctrl-C to cancel",
        ):
            return self.agent.run(task, mode=AgentMode.AUTO)

    def _can_stream_turn(self) -> bool:
        """True when this turn's answer can be rendered as it is written."""
        probe = getattr(self.agent, "_can_stream_native_tools", None)
        try:
            return bool(probe and probe())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a turn
            return False

    def _stream_agent(self, task: str, console: Any) -> Any:
        """Stream one turn, handing the terminal between status and answer.

        Only one live region may own the terminal, so the status line runs while
        the model is thinking and dispatching tools, and the answer region takes
        over from the first answer delta. A tool call after the answer has begun
        closes the region — leaving what was printed — and hands back to the
        status line until the answer resumes.

        Returns the turn's reconstructed record, or ``None`` when the stream
        produced none, which tells the caller to run the turn the blocking way.
        """
        from effgen.cli import progress as _progress

        status = _progress.LiveStatus(
            console,
            model_label=_progress.short_model_label(self.model_id),
            reasoning=_progress.is_reasoning_agent(self.agent),
            tracker=getattr(self.agent, "execution_tracker", None),
            hint="Ctrl-C to cancel",
        )
        answer = _progress.LiveAnswer(console)
        status.__enter__()
        status_open = True
        try:
            for event in self.agent.stream(task, include_events=True):
                if event.kind == "answer" and event.text:
                    if status_open:
                        status.__exit__(None, None, None)
                        status_open = False
                    if not answer.is_open:
                        answer.open()
                    answer.push(event.text)
                elif event.kind in ("tool_call", "thought", "observation"):
                    if answer.is_open:
                        answer.close()
                        status.__enter__()
                        status_open = True
        finally:
            answer.close()
            if status_open:
                status.__exit__(None, None, None)
        response = getattr(self.agent, "last_stream_response", None)
        if response is None:
            # The turn is about to be run again the blocking way, so nothing is
            # claimed to be on screen already and its answer prints normally.
            return None
        self._streamed_answer = answer.text
        return response

    def _post_turn_notes(self, result: Any) -> None:
        """Surface files written, refusals, withheld actions and the iteration cap."""
        if self.quiet:
            return
        # Named once, on the first turn that reaches the tool loop: which path
        # the model's tool calls travel on decides what a turn that called no
        # tool means, and it is the same for the rest of the session.
        path = getattr(result, "tool_calling", "")
        if path and not self._named_tool_calling:
            from .render import tool_calling_label

            self._named_tool_calling = True
            self._say(f"Tool calling: {tool_calling_label(path)}")
        from .render import recovered_answer_note

        note = recovered_answer_note(result)
        if note:
            self._status("warning", note)
        if result.files_written:
            self._say(f"Files written: {', '.join(result.files_written)}")
        for diff in result.diffs:
            failed = int(diff.get("hunks_failed", 0) or 0)
            if failed:
                self._status(
                    "warning",
                    f"{diff.get('path')}: {failed} hunk(s) did not apply (the file "
                    "changed since it was read); the rest were applied."
                )
        for record in result.refused:
            self._status("warning", record.reason)
        if result.withheld:
            kinds = ", ".join(sorted({a.kind for a in result.withheld}))
            # Name the mode the turn actually ran under: ``/plan`` runs one turn
            # in plan mode without changing the session's mode.
            turn_mode = getattr(result, "permission_mode", "") or self.mode.value
            if turn_mode == PermissionMode.PLAN.value:
                self._say(
                    f"Plan mode: {len(result.withheld)} action(s) ({kinds}) were "
                    "proposed, not carried out. /apply to write staged edits, or "
                    "/mode auto-edit."
                )
            else:
                self._say(
                    f"{len(result.withheld)} action(s) ({kinds}) were not permitted "
                    f"in {turn_mode} mode. /mode auto-edit (or /mode yes) to allow."
                )
        if result.hit_iteration_cap:
            # The answer already names the cap and what it cost; this adds the
            # flag that raises it for the session.
            self._status(
                "warning", "Raise the cap for this session with --max-iterations N."
            )
