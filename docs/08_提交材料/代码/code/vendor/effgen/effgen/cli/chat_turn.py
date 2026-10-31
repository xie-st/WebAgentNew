"""One chat turn, from the user's line to the footer under the answer.

:mod:`effgen.cli.chat` owns the session itself and mixes these methods into
:class:`~effgen.cli.chat.ChatREPL`, so ``effgen.cli.chat`` stays the import path
callers use.

A turn takes one of two paths. With no tools attached the model's answer is
streamed straight through, which reads best live, and the turn is written to the
persistent session here once its tokens, cost and latency are known. Once a tool
is attached the turn streams too, when the model's tool calls travel on a path
that can be dispatched mid-stream — the status line while it thinks and calls
tools, the answer written as it arrives — falling back to a blocking
``agent.run()`` under the live-status spinner otherwise. Either way that path
persists the turn itself and reports exact per-turn accounting, which the footer
uses in preference to the difference between two model counters.

A turn that raises is reported and the session continues. A turn stopped with
Ctrl-C keeps what it had reached and is not persisted.
"""

from __future__ import annotations

import sys
import time

from effgen.cli import progress as _progress


class ChatTurnMixin:
    """The turn half of :class:`~effgen.cli.chat.ChatREPL`."""

    #: Per-turn accounting the footer prefers over the difference between two
    #: model counters, and the answer this turn already wrote to the terminal.
    #: Declared here because the composed class reads them.
    _turn_response_tokens: int = 0
    _turn_response_cost: float = 0.0
    _streamed_answer: str = ""

    def _snapshot(self) -> tuple[int, float]:
        m = getattr(self.agent, "model", None)
        return (
            int(getattr(m, "total_tokens", 0) or 0),
            float(getattr(m, "total_cost", 0.0) or 0.0),
        )

    def _do_turn(self, user_input: str) -> None:
        tok0, cost0 = self._snapshot()
        t0 = time.monotonic()
        answer = ""
        interrupted = False
        plain_turn = False
        self.last_trace = None

        try:
            if self.tool_count > 0:
                # agent.run() persists to the session itself.
                answer = self._run_with_tools(user_input)
            else:
                answer = self._stream_plain(user_input)
                # The streaming path bypasses agent.run(), so the turn is
                # persisted below once its tokens/cost/latency are known.
                plain_turn = True
        except KeyboardInterrupt:
            interrupted = True
            self.cli.print("")
            if self._color and self.console:
                self.console.print("[yellow]Stopped.[/yellow]", highlight=False)
            else:
                self.cli.print("Stopped.")
        except Exception as e:  # noqa: BLE001
            self.cli.print_error(f"{e}")
            self._teach_model_error(e)
            return

        self.last_answer = answer
        self.turns += 1

        # Per-turn footer: · 1.2s · 318 tok · $0.0003 (accurate about what we know).
        elapsed = time.monotonic() - t0
        tok1, cost1 = self._snapshot()
        dtok = tok1 - tok0
        dcost = cost1 - cost0
        if dtok <= 0 and answer:
            dtok = self._count_tokens(answer)  # local models: count the output
        self.session_tokens += max(dtok, 0)
        self.session_cost += max(dcost, 0.0)
        if plain_turn and not interrupted:
            self._persist_session_turn(
                user_input, answer,
                tokens=max(dtok, 0), cost=max(dcost, 0.0), elapsed=elapsed,
            )
        if not self.quiet:
            self._print_footer(elapsed, dtok, dcost, interrupted)

    def _persist_session_turn(
        self,
        user_input: str,
        answer: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
        elapsed: float | None = None,
    ) -> None:
        """Append a plain (streamed) turn to the persistent session, if any.

        The turn carries the model, token count, cost and latency it was
        answered with, matching what `agent.run()` stores for tool turns.
        """
        session = getattr(self.agent, "session", None)
        if session is None or not answer:
            return
        meta: dict[str, object] = {"model": self.model_id}
        if tokens:
            meta["tokens_used"] = tokens
        if cost:
            meta["cost_usd"] = round(cost, 8)
        if elapsed is not None:
            meta["latency_ms"] = round(elapsed * 1000, 1)
        try:
            session.add_message("user", user_input, **meta)
            session.add_message("assistant", answer, **meta)
            session.metadata["model"] = self.model_id
            session.save()
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

    # ------------------------------------------------------------------
    # The two answer paths
    # ------------------------------------------------------------------
    def _stream_plain(self, user_input: str) -> str:
        """Stream the model's answer, rendering markdown the way tool turns do.

        A piped/non-TTY run emits the raw answer only; an interactive terminal
        renders lists/headings/code the same whether or not a tool ran — a live
        markdown region on a colour TTY, a single rendered block otherwise.
        """
        gen = self.agent.stream(user_input)
        if not self.interactive:
            return _progress.stream_answer(
                self.console, gen,
                animate=False, interactive=False, quiet=self.quiet,
                trailing_newline=True,
            ).strip()

        return _progress.stream_answer(
            self.console, gen,
            animate=bool(self.console and self.animate and self._color),
            interactive=True, quiet=self.quiet,
            label="assistant", render_plain=True,
        ).strip()

    def _run_with_tools(self, user_input: str) -> str:
        """Run a tool-enabled turn under the live-status spinner, then show the answer.

        The turn uses the agent's own configured mode (single-agent unless the
        caller set another), the same default `effgen run` uses. Forcing
        automatic routing here would let the router decompose an ordinary
        question into sub-agents, which costs several extra model calls and
        answers from a synthesis step rather than from the tool result.
        """
        streamed = ""
        if self.animate and self.console and self._can_stream_tool_turn():
            response = self._stream_tool_turn(user_input)
            if response is not None:
                streamed = self._streamed_answer
        else:
            response = None
        if response is not None:
            pass
        elif self.animate and self.console:
            reasoning = _progress.is_reasoning_agent(self.agent)
            with _progress.LiveStatus(
                self.console,
                model_label=_progress.short_model_label(self.model_id),
                reasoning=reasoning,
                tracker=self.agent.execution_tracker,
                hint="Ctrl-C to cancel",
            ):
                response = self.agent.run(user_input)
        else:
            if not self.quiet and self.interactive:
                print("Thinking…")
            response = self.agent.run(user_input)

        try:
            self.last_trace = response.execution_trace or None
        except Exception:  # noqa: BLE001
            self.last_trace = None

        answer = response.output or ""
        meta = response.metadata or {}
        # A turn the loop stopped has no answer: the reply says what stopped it,
        # and what the turn had reached follows under its own label, so the tool
        # output it recovered is not read as the reply.
        progress = (
            str(meta.get("partial_output") or "")
            if not response.success and meta.get("partial")
            else ""
        )
        if not self.interactive:
            sys.stdout.write((answer or "") + "\n")
            if progress:
                self._show_progress(progress)
            sys.stdout.flush()
        elif streamed and streamed.strip() == (answer or "").strip():
            # The answer is already on screen, written as it arrived. Printing
            # it again would show it twice.
            if progress:
                self._show_progress(progress)
        else:
            self._show_answer(answer)
            if progress:
                self._show_progress(progress)

        # Prefer the response's own accounting for the footer when available.
        self._turn_response_tokens = int(getattr(response, "tokens_used", 0) or 0)
        self._turn_response_cost = _progress._extract_cost(meta) or 0.0
        if not response.success:
            reason = meta.get("reason", "failed")
            self.cli.print_warning(f"(turn did not fully succeed: {reason})")
        return answer

    # ------------------------------------------------------------------
    # Streaming a tool turn
    # ------------------------------------------------------------------
    def _can_stream_tool_turn(self) -> bool:
        """True when this turn's tool calls travel on a path that can stream.

        A model whose adapter records the calls it streams can dispatch them
        mid-stream, so the answer is written as it arrives. A model that emits
        its calls as text in the prompt (a local chat template) cannot: the call
        syntax would reach the terminal before anything knows it is a call, so
        those turns keep the blocking path.
        """
        probe = getattr(self.agent, "_can_stream_native_tools", None)
        try:
            return bool(probe and probe())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a turn
            return False

    def _stream_tool_turn(self, user_input: str):
        """Stream a tool-using turn, handing the terminal between two regions.

        Only one live region may own the terminal, so the status line runs while
        the model is thinking and dispatching tools, and the answer region takes
        over from the first answer delta. A tool call after the answer has begun
        closes the region — leaving what was printed — and hands back to the
        status line until the answer resumes. The same handoff the coding
        session uses.

        Returns:
            The turn's response, or ``None`` when the stream produced none,
            which tells the caller to run the turn the blocking way.
        """
        self._streamed_answer = ""
        status = _progress.LiveStatus(
            self.console,
            model_label=_progress.short_model_label(self.model_id),
            reasoning=_progress.is_reasoning_agent(self.agent),
            tracker=getattr(self.agent, "execution_tracker", None),
            hint="Ctrl-C to cancel",
        )
        answer = _progress.LiveAnswer(self.console)
        status.__enter__()
        status_open = True
        try:
            for event in self.agent.stream(user_input, include_events=True):
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
            # The turn runs again the blocking way, so nothing is claimed to be
            # on screen already and its answer prints normally.
            return None
        self._streamed_answer = answer.text
        return response

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.agent.model.count_tokens(text).count)
        except Exception:  # noqa: BLE001
            return 0
