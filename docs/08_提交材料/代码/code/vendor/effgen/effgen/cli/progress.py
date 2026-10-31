"""Live progress and status presentation for the command-line interface.

This is a *presentation layer only*: it never changes what an agent does, only
how the agent's existing execution events are surfaced to a human watching an
interactive terminal. It turns the silent "dead pause" while a model thinks into
a live, ticking status line — ``Thinking…`` → ``Calling <model>…`` →
``Running <tool>…`` — followed by a single glanceable summary line.

Everything here is **opt-out and TTY-aware**. Animation is shown only when:

* output is an interactive terminal (``stdout.isatty()``),
* the ``rich`` library is installed,
* and none of the following ask for plain output:
  ``--quiet``, ``--no-animation``, ``EFFGEN_NO_ANIM=1``, ``NO_COLOR``, or a CI
  environment.

When animation is off the caller falls back to plain, single-line text that is
safe to pipe, redirect, or capture in logs. The live status is *transient* (it
erases itself when the work finishes) so it never corrupts the final output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:  # rich is an install dependency, but degrade to plain output if it is absent.
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is normally present
    _RICH_AVAILABLE = False


# Braille shimmer frames for the manual spinner (kept in sync with the icons
# used elsewhere in the CLI so the whole product feels like one tool), plus the
# stand-in frames used when the console cannot encode braille.
_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BRAILLE_ASCII = "|/-\\"


def _spinner_frame(stream: Any = None) -> str:
    """Return the current spinner cell for *stream*'s encoding."""
    from effgen.ui.palette import supports_unicode

    frames = _BRAILLE if supports_unicode(stream) else _BRAILLE_ASCII
    return frames[int(time.monotonic() * 10) % len(frames)]


def _console_stream(console: Any) -> Any:
    """Return the file a rich console writes to, or ``None`` if unknown."""
    return getattr(console, "file", None)


# Environment variables that indicate a non-interactive CI runner. Animation is
# pointless (and noisy) in CI logs, so we fall back to plain output there.
_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
)


def _env_flag(name: str) -> bool:
    """Return True if an environment flag is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def color_enabled() -> bool:
    """Honor the ``NO_COLOR`` convention (https://no-color.org/).

    Re-exported from the shared theme module so the CLI has one source of truth.
    """
    from effgen.ui.theme import color_enabled as _color_enabled

    return _color_enabled()


def is_ci() -> bool:
    """True when running under a known continuous-integration environment."""
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


def animation_enabled(
    *,
    quiet: bool = False,
    no_animation: bool = False,
    stream: Any = None,
) -> bool:
    """Decide whether to show live animated status for the current invocation.

    Args:
        quiet: The user asked for quiet output (errors/answer only).
        no_animation: The user passed ``--no-animation``.
        stream: The output stream to test for interactivity (defaults to
            ``sys.stdout``).

    Returns:
        True only for an interactive terminal that has not opted out.
    """
    if not _RICH_AVAILABLE:
        return False
    if quiet or no_animation:
        return False
    if _env_flag("EFFGEN_NO_ANIM"):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if is_ci():
        return False
    out = stream if stream is not None else sys.stdout
    try:
        if not out.isatty():
            return False
    except Exception:  # noqa: BLE001 - a stream without isatty() is non-interactive
        return False
    return True


def is_reasoning_agent(agent: Any) -> bool:
    """Best-effort detection of a reasoning model (pauses before any token).

    Reasoning models (OpenAI o-series / gpt-5 reasoning, thinking-budget Gemini,
    DeepSeek-R, …) can sit silent for many seconds before emitting output. We
    surface a ``🧠 reasoning…`` indicator for them so the wait feels alive rather
    than hung. Detection prefers the adapter's own capability flag and falls back
    to a conservative model-id heuristic.
    """
    model = getattr(agent, "model", None)
    if model is None:
        return False
    if getattr(model, "_is_reasoning_model", False):
        return True
    name = (getattr(model, "model_name", "") or "").lower()
    return any(
        marker in name
        for marker in ("o1-", "o3-", "o4-mini", "reasoner", "deepseek-r", "-thinking")
    )


def short_model_label(model_name: str | None, limit: int = 28) -> str:
    """A compact, human-friendly model label for status lines.

    Strips a ``provider/`` or ``org/`` prefix so ``Qwen/Qwen2.5-1.5B-Instruct``
    reads as ``Qwen2.5-1.5B-Instruct`` and ``openai:gpt-5-nano`` as ``gpt-5-nano``.
    """
    if not model_name:
        return "model"
    label = str(model_name)
    for sep in (":", "/"):
        if sep in label:
            label = label.split(sep)[-1]
    if len(label) > limit:
        label = label[: limit - 1] + "…"
    return label


# ---------------------------------------------------------------------------
# Live status line
# ---------------------------------------------------------------------------


class _StatusState:
    """Mutable status shared between the agent thread and the Live refresher.

    The agent's :class:`ExecutionTracker` invokes :meth:`LiveStatus.on_event`
    (synchronously, on the agent thread) which mutates this object; rich's
    background refresh thread reads it ~12×/s to redraw the line. Plain
    attribute reads/writes of immutable values are safe across the two threads
    for this purely cosmetic use.
    """

    # A tool/sub-step label stays visible at least this long so instant tools
    # (e.g. the calculator, which finishes in well under a frame) still flash a
    # readable "Running <tool>…" rather than being skipped between refreshes.
    _STICKY_SECONDS = 0.6

    def __init__(
        self,
        model_label: str,
        reasoning: bool,
        hint: str | None = None,
        stream: Any = None,
    ):
        self.start = time.monotonic()
        self.model_label = model_label
        self.reasoning = reasoning
        # The stream the label will be printed to, so the reasoning marker is
        # chosen for that terminal's encoding rather than for ``sys.stdout``.
        self.stream = stream
        self.base_label = self._idle_label()
        self.sticky_label: str | None = None
        self.sticky_until = 0.0
        # An optional trailing hint (e.g. "Ctrl-C to cancel") shown dim after the
        # status label and cleared with the line when the run finishes.
        self.hint = hint

    def _idle_label(self) -> str:
        if self.reasoning:
            from effgen.ui.palette import glyph

            mark = glyph("reasoning", self.stream)
            prefix = f"{mark} " if mark else ""
            return f"{prefix}{self.model_label} reasoning…"
        return f"Calling {self.model_label}…"

    def set_step(self, label: str) -> None:
        """Show *label* now and hold it for at least the sticky window."""
        self.sticky_label = label
        self.sticky_until = time.monotonic() + self._STICKY_SECONDS

    def set_base(self, label: str) -> None:
        """Set the steady-state label shown once any sticky step has expired."""
        self.base_label = label

    def effective_label(self) -> str:
        if self.sticky_label is not None and time.monotonic() < self.sticky_until:
            return self.sticky_label
        return self.base_label


class _StatusRenderable:
    """A rich renderable rebuilt on every refresh so time/spinner keep ticking."""

    def __init__(self, state: _StatusState, stream: Any = None):
        self.state = state
        self.stream = stream

    def __rich__(self) -> Any:
        from effgen.ui.render import ascii_fold

        s = self.state
        frame = _spinner_frame(self.stream)
        elapsed = time.monotonic() - s.start
        text = Text()
        text.append(frame + " ", style="cyan")
        text.append(ascii_fold(s.effective_label(), self.stream))
        text.append(ascii_fold(f"   ⏱ {elapsed:4.1f}s", self.stream), style="dim")
        if s.hint:
            text.append(ascii_fold(f"   ({s.hint})", self.stream), style="dim")
        return text


# The live status currently animating, if any. A live region owns the terminal
# and repaints over whatever else is written there, so anything that needs to
# ask the human a question mid-run suspends it through
# :func:`suspend_live_status` first.
_ACTIVE_STATUS: "LiveStatus | None" = None


@contextmanager
def suspend_live_status() -> "Iterator[None]":
    """Stop the animating status line for the duration of the block.

    A prompt written while a live region is refreshing is erased by the next
    frame, so a question asked mid-run would never be readable. Nothing happens
    when no status is animating.
    """
    status = _ACTIVE_STATUS
    if status is None:
        yield
        return
    with status.paused():
        yield


class LiveStatus:
    """Context manager that animates a single live status line during a run.

    Usage::

        with LiveStatus(console, model_label="gpt-5-nano", tracker=agent.execution_tracker):
            response = agent.run(task)

    It is transient: when the ``with`` block exits, the status line is erased so
    the caller can print the answer and a frozen summary on a clean slate. When
    ``rich`` is unavailable or ``console`` is ``None`` it becomes a no-op (the
    caller should already have decided not to animate via :func:`animation_enabled`).
    """

    def __init__(
        self,
        console: Any,
        *,
        model_label: str = "model",
        reasoning: bool = False,
        tracker: Any = None,
        hint: str | None = None,
    ) -> None:
        self.console = console
        self.tracker = tracker
        self.state = _StatusState(
            model_label, reasoning, hint=hint, stream=_console_stream(console)
        )
        self._live: Any = None
        self._active = bool(_RICH_AVAILABLE and console is not None)

    def __enter__(self) -> "LiveStatus":
        global _ACTIVE_STATUS
        if not self._active:
            return self
        _ACTIVE_STATUS = self
        if self.tracker is not None:
            try:
                self.tracker.add_listener(self.on_event)
            except Exception:  # noqa: BLE001 - never let cosmetics break a run
                pass
        self._live = Live(
            _StatusRenderable(self.state, _console_stream(self.console)),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> bool:
        global _ACTIVE_STATUS
        if _ACTIVE_STATUS is self:
            _ACTIVE_STATUS = None
        if self.tracker is not None:
            try:
                self.tracker.remove_listener(self.on_event)
            except Exception:  # noqa: BLE001
                pass
        if self._live is not None:
            try:
                self._live.__exit__(*exc)
            except Exception:  # noqa: BLE001
                pass
            self._live = None
        return False  # never suppress exceptions (Ctrl-C must propagate)

    @contextmanager
    def paused(self) -> "Iterator[None]":
        """Take the status line down for the duration of the block, then resume.

        Used when the run has to hand the terminal back — to ask the human a
        question, for instance — because the live region repaints over anything
        else written to the same stream.
        """
        live, self._live = self._live, None
        if live is not None:
            try:
                live.stop()
            except Exception:  # noqa: BLE001 - never let cosmetics break a run
                live = None
        try:
            yield
        finally:
            if live is not None:
                try:
                    live.start(refresh=True)
                    self._live = live
                except Exception:  # noqa: BLE001
                    self._live = None

    def on_event(self, event: Any) -> None:
        """Update the status label from an execution event (tracker listener)."""
        try:
            etype = getattr(event.type, "value", str(event.type))
        except Exception:  # noqa: BLE001
            return
        if etype == "tool_call_start":
            name = (event.data or {}).get("tool_name", "tool")
            self.state.set_step(f"Running {name}…")
        elif etype in ("tool_call_complete", "tool_call_failed"):
            # The model is called again next; revert the steady-state label but
            # let any still-visible "Running <tool>…" finish its sticky window.
            self.state.set_base(self.state._idle_label())
        elif etype == "sub_agent_start":
            name = (event.data or {}).get("agent_name") or event.agent_id or "sub-agent"
            self.state.set_step(f"Delegating to {name}…")
        elif etype == "task_decomposition":
            self.state.set_base("Planning…")


# ---------------------------------------------------------------------------
# Thinking spinner + streaming answer renderer
# ---------------------------------------------------------------------------

# One shared refresh rate for the live streaming region. Too low reads as choppy,
# too high flickers; 8–12/s is the usable band and 10/s is the shared constant so
# every streaming surface redraws at the same cadence rather than per-call guesses.
STREAM_REFRESH_PER_SECOND = 10


class _ThinkingRenderable:
    """A spinning 'Thinking…' line rebuilt each refresh so the spinner ticks."""

    def __init__(self, label: str, start: float, stream: Any = None):
        self.label = label
        self.start = start
        self.stream = stream

    def __rich__(self) -> Any:
        from effgen.ui.render import ascii_fold

        frame = _spinner_frame(self.stream)
        text = Text()
        text.append(frame + " ", style="cyan")
        text.append(ascii_fold(self.label, self.stream))
        return text


class thinking_status:  # noqa: N801 - used as a lowercase context-manager factory
    """Transient pre-first-token / indeterminate spinner shared by every surface.

    One spinner, one frames set (:data:`_BRAILLE`), one label — used for the
    streaming pre-token wait and the setup wizard so a "thinking" pause looks the
    same everywhere. :class:`LiveStatus` remains the richer model/tool-aware
    variant used whenever an execution tracker is available. A no-op when ``rich``
    is unavailable, ``console`` is ``None``, or ``animate`` is false.
    """

    def __init__(self, console: Any, *, animate: bool = True, label: str = "Thinking…") -> None:
        self.console = console
        self.label = label
        self._active = bool(animate and _RICH_AVAILABLE and console is not None)
        self._live: Any = None

    def __enter__(self) -> "thinking_status":
        if self._active:
            self._live = Live(
                _ThinkingRenderable(self.label, time.monotonic(), _console_stream(self.console)),
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._live is not None:
            try:
                self._live.__exit__(*exc)
            except Exception:  # noqa: BLE001
                pass
            self._live = None
        return False


class LiveAnswer:
    """A live region that re-renders an answer as its deltas arrive.

    :func:`stream_answer` renders one answer from start to finish; this is the
    same region under the caller's control, for a surface that has to hand the
    terminal back and forth — a coding turn opens it when the model starts
    answering, closes it when a tool call interrupts, and opens it again when
    the answer resumes. Only one live region may own the terminal at a time, so
    the caller closes this before starting a status line and vice versa.

    Closing leaves the rendered answer on screen (the region is not transient),
    and :attr:`text` is everything pushed so far.

    A no-op when ``rich`` is unavailable or *console* is ``None``: the deltas
    still accumulate in :attr:`text`, nothing is drawn.
    """

    def __init__(self, console: Any) -> None:
        self.console = console
        self._parts: list[str] = []
        self._live: Any = None
        self._active = bool(_RICH_AVAILABLE and console is not None)

    @property
    def is_open(self) -> bool:
        """True while the region owns the terminal."""
        return self._live is not None

    @property
    def text(self) -> str:
        """Everything pushed into the region, in order."""
        return "".join(self._parts)

    def open(self) -> None:
        """Start drawing. Does nothing if the region is already open."""
        if not self._active or self._live is not None:
            return
        self._live = Live(
            console=self.console,
            refresh_per_second=STREAM_REFRESH_PER_SECOND,
            transient=False,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        self._render()

    def push(self, delta: str) -> None:
        """Add *delta* to the answer and redraw."""
        if not delta:
            return
        self._parts.append(delta)
        self._render()

    def close(self) -> None:
        """Stop drawing, leaving what has been rendered on screen."""
        live, self._live = self._live, None
        if live is None:
            return
        try:
            live.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - never let cosmetics break a turn
            pass

    def _render(self) -> None:
        if self._live is None:
            return
        from effgen.ui.render import answer_renderable

        try:
            self._live.update(answer_renderable(self.text))
        except Exception:  # noqa: BLE001 - never let cosmetics break a turn
            pass


def stream_answer(
    console: Any,
    token_iter: Any,
    *,
    animate: bool,
    interactive: bool = True,
    quiet: bool = False,
    label: str | None = None,
    render_plain: bool = False,
    trailing_newline: bool = False,
) -> str:
    """Render a streamed answer once, chosen by capability; return the full text.

    Three mutually exclusive presentations, one function:

    * **Animating** (interactive colour TTY with ``rich``): show the shared
      :class:`thinking_status` spinner until the first non-empty token, then hand
      off to a live region that re-renders ``answer_renderable`` at
      :data:`STREAM_REFRESH_PER_SECOND`, so streamed markdown (lists, code,
      tables) renders live instead of dumping raw source. An optional *label*
      (e.g. ``assistant``) is printed on its own line first.
    * **Interactive, not animating** with ``render_plain=True`` (e.g. a
      ``NO_COLOR`` terminal): collect the tokens behind a lightweight wait
      indicator, then render the finished answer once via
      :func:`effgen.ui.render.answer_surface` so it reads as markdown, not raw
      text.
    * **Otherwise** (piped / redirected / non-TTY / ``rich`` absent / ``--quiet``):
      write raw tokens straight to ``stdout`` as they arrive — byte-identical to
      a plain token passthrough — optionally followed by a trailing newline.

    The returned text is the concatenation of the streamed tokens (unstripped),
    so the caller can persist the turn or stamp a summary.

    Args:
        console: The console the answer is drawn on.
        token_iter: The tokens to render as they arrive.
        animate: Use the live-rendering presentation.
        interactive: Whether the surface is an interactive terminal.
        quiet: Write raw tokens with no decoration.
        label: A label printed on its own line before the answer.
        render_plain: Collect the tokens and render once at the end.
        trailing_newline: Write a newline after the raw-token path.
    """
    it = iter(token_iter)
    collected: list[str] = []

    if animate and _RICH_AVAILABLE and console is not None:
        from effgen.ui.render import answer_renderable

        if label:
            console.print(f"[effgen.success]{label}[/effgen.success]", highlight=False)
        first: str | None = None
        with thinking_status(console, animate=not quiet):
            for tok in it:
                if tok:
                    first = tok
                    break
        if first:
            collected.append(first)
        with Live(
            console=console,
            refresh_per_second=STREAM_REFRESH_PER_SECOND,
            transient=False,
            vertical_overflow="visible",
        ) as live:
            if collected:
                live.update(answer_renderable("".join(collected)))
            for tok in it:
                if not tok:
                    continue
                collected.append(tok)
                live.update(answer_renderable("".join(collected)))
        return "".join(collected)

    if interactive and render_plain:
        # Collect behind a lightweight wait indicator, then render once so the
        # answer is markdown-formatted like the animated / tool paths.
        show = not quiet
        if show:
            from effgen.ui.render import ascii_fold

            sys.stdout.write(ascii_fold("Thinking…", sys.stdout))
            sys.stdout.flush()
        for tok in it:
            if tok:
                collected.append(tok)
        if show:
            sys.stdout.write("\r" + " " * 10 + "\r")
            sys.stdout.flush()
        from effgen.ui.render import answer_surface

        answer_surface("".join(collected).strip(), framed=False, label=label, console=console)
        return "".join(collected)

    # Raw passthrough: piped / redirected / non-TTY / rich-absent / quiet.
    for tok in it:
        if tok:
            sys.stdout.write(tok)
            collected.append(tok)
    if trailing_newline:
        sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(collected)


# ---------------------------------------------------------------------------
# Final summary line
# ---------------------------------------------------------------------------


def _extract_cost(metadata: dict[str, Any] | None) -> float | None:
    """Pull a positive USD cost out of response metadata (one shared extractor)."""
    from effgen.ui.render import _extract_cost as _e

    return _e(metadata)


def summary_line(response: Any, *, stream: Any = None) -> tuple[str, str]:
    """Build the one-glance final summary for a finished run.

    Returns ``(plain, markup)`` where *plain* is safe for non-rich output and
    *markup* carries rich color tags. Delegates to the shared builder in
    :mod:`effgen.ui.render` so run/chat share one summary vocabulary.

    Example (success)::

        ✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006
    """
    from effgen.ui.render import summary_line as _summary_line

    return _summary_line(response, mode="run", stream=stream)


def print_summary(cli: Any, response: Any) -> None:
    """Print the frozen summary line via the CLI's rich-or-plain print."""
    from effgen.ui.render import ascii_fold

    stream = cli._human_stream() if hasattr(cli, "_human_stream") else None
    plain, markup = summary_line(response, stream=stream)
    console = getattr(cli, "console", None)
    if console is not None:
        console.print(ascii_fold(markup, stream), highlight=False)
    else:
        print(ascii_fold(plain, stream))


def _truncate(value: Any, limit: int) -> str:
    text = str(value).strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_tool_call(name: str, tool_input: Any, limit: int = 72) -> str:
    """Render a tool call as ``name(key="value", …)`` on one balanced line.

    ``tool_input`` may be a dict of arguments or a JSON string of one; either
    is shown as compact ``key="value"`` pairs. Anything else is shown as a
    single truncated argument. The parentheses always close — a call trimmed to
    fit ends in ``…)`` rather than a dangling ``,`` or an unbalanced brace.

    Args:
        name: The tool that was called.
        tool_input: The call arguments, as a dict or a JSON string.
        limit: Most characters the rendered line may take.

    Returns:
        The one-line rendering, always with balanced parentheses.
    """
    args: Any = tool_input
    if isinstance(args, str):
        text = args.strip()
        if text[:1] in ("{", "["):
            try:
                args = json.loads(text)
            except (ValueError, TypeError):
                args = text
        else:
            args = text

    if isinstance(args, dict):
        parts = []
        for key, val in args.items():
            if isinstance(val, str):
                rendered = f'{key}="{val}"'
            else:
                rendered = f"{key}={val}"
            parts.append(rendered.replace("\n", " "))
        inner = ", ".join(parts)
    else:
        inner = str(args).replace("\n", " ")

    if len(inner) > limit:
        inner = inner[: limit - 1].rstrip(", ") + "…"
    return f"{name}({inner})"


def _fmt_duration(seconds: float) -> str:
    """Human duration: ``820ms`` under a second, ``14.4s`` / ``1m03s`` above."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m{s:02d}s"


def _event_gaps(trace: list[dict[str, Any]]) -> list[float]:
    """Wall-clock seconds elapsed *before* each event (gap from the previous one).

    The gap before a ``tool_call_start`` is the model's think time for that
    step — the time the flat trace throws away. The first event's gap is 0.
    """
    gaps: list[float] = []
    prev: float | None = None
    for ev in trace:
        ts = ev.get("timestamp")
        if isinstance(ts, int | float) and prev is not None:
            gaps.append(max(0.0, float(ts) - prev))
        else:
            gaps.append(0.0)
        if isinstance(ts, int | float):
            prev = float(ts)
    return gaps


def execution_trace_lines(
    trace: list[dict[str, Any]] | None, *, stream: Any = None
) -> list[tuple[str, str]]:
    """Turn an :class:`ExecutionTracker` event trace into readable ``(style, text)``.

    The agent's ``execution_trace`` is a list of event dicts (``type``,
    ``message``, ``data``) — not a ReAct ``thought/action/observation`` shape —
    so a plain ``step.get("thought")`` renders blank. This formatter walks the
    events and produces one human line each (reasoning, tool call + result,
    delegation, …), annotating each step with the wall-clock time it took, and
    is shared by ``effgen run --explain`` and chat's ``/trace`` so they agree.

    Step glyphs resolve through :func:`palette.glyph` (against *stream*, default
    ``stdout``) so a non-UTF-8 console falls back to an ASCII form instead of
    raising ``UnicodeEncodeError``.
    """
    from effgen.ui.palette import glyph

    g_thought = glyph("thought", stream)
    g_tool = glyph("tool", stream)
    g_clock = glyph("clock", stream)
    g_ok = glyph("success", stream)
    g_err = glyph("error", stream)
    g_delegate = glyph("delegate", stream)
    g_plan = glyph("plan", stream)

    events = list(trace or [])
    gaps = _event_gaps(events)
    # Pair each tool_call_start with its terminal event to time the tool itself.
    out: list[tuple[str, str]] = []
    for i, ev in enumerate(events):
        etype = str(ev.get("type", "") or "")
        msg = str(ev.get("message", "") or "")
        data = ev.get("data") or {}
        if etype == "reasoning_step":
            out.append(("cyan", f"{g_thought} {msg or 'reasoning…'}"))
        elif etype == "tool_call_start":
            name = data.get("tool_name", "tool")
            tool_input = data.get("tool_input", data.get("input", ""))
            if tool_input:
                detail = f"{g_tool} {format_tool_call(name, tool_input)}"
            else:
                detail = f"{g_tool} {name}"
            # Attribute the model's think time (the gap into this call) so the
            # step reads with the time it cost, not as if it were instant.
            if gaps[i] >= 0.1:
                detail += f"  {g_clock} {_fmt_duration(gaps[i])}"
            out.append(("green", detail))
        elif etype == "tool_call_complete":
            result = data.get("result", data.get("output", ""))
            exec_s = gaps[i] if gaps[i] >= 0.5 else 0.0
            suffix = f"  ({_fmt_duration(exec_s)})" if exec_s else ""
            out.append((
                "dim",
                f"   {g_ok} {_truncate(result, 120)}{suffix}" if result else f"   {g_ok} done{suffix}",
            ))
        elif etype == "tool_call_failed":
            err = data.get("error", msg)
            out.append(("red", f"   {g_err} {_truncate(err, 120)}"))
        elif etype in ("sub_agent_start",):
            name = data.get("agent_name") or ev.get("agent_id") or "sub-agent"
            out.append(("magenta", f"{g_delegate} delegating to {name}"))
        elif etype == "task_decomposition":
            out.append(("yellow", f"{g_plan} {msg or 'planning…'}"))
        elif etype in ("task_complete", "answer"):
            if msg:
                out.append(("dim", f"   {_truncate(msg, 120)}"))
    return out


def execution_timeline_lines(
    trace: list[dict[str, Any]] | None, *, stream: Any = None
) -> list[tuple[str, str]]:
    """A compact per-step timeline: each step with a proportional bar + duration.

    Collapses the event stream into one line per meaningful step (a tool call,
    reasoning, or delegation), sizes a bar by the step's wall-clock share, and
    labels it with the elapsed time — so a run's slow steps are visible at a
    glance. Falls back to an empty list when there is nothing timed to show.

    Step glyphs resolve through :func:`palette.glyph` (against *stream*) so a
    non-UTF-8 console falls back to an ASCII form.
    """
    from effgen.ui.palette import glyph

    g_tool = glyph("tool", stream)
    g_delegate = glyph("delegate", stream)
    g_plan = glyph("plan", stream)

    events = list(trace or [])
    if not events:
        return []
    gaps = _event_gaps(events)

    # Build steps: a tool call is timed by the think-gap leading into it; a bare
    # reasoning or delegation step is timed by its own leading gap.
    steps: list[tuple[str, str, float]] = []  # (style, label, seconds)
    for i, ev in enumerate(events):
        etype = str(ev.get("type", "") or "")
        data = ev.get("data") or {}
        if etype == "tool_call_start":
            name = data.get("tool_name", "tool")
            tool_input = data.get("tool_input", data.get("input", ""))
            label = format_tool_call(name, tool_input) if tool_input else name
            steps.append(("green", f"{g_tool} {label}", gaps[i]))
        elif etype == "sub_agent_start":
            name = data.get("agent_name") or ev.get("agent_id") or "sub-agent"
            steps.append(("magenta", f"{g_delegate} {name}", gaps[i]))
        elif etype == "task_decomposition":
            steps.append(("yellow", f"{g_plan} planning", gaps[i]))
    if not steps:
        return []

    longest = max((s for _, _, s in steps), default=0.0) or 1.0
    bar_w = 20
    label_w = min(40, max(len(lbl) for _, lbl, _ in steps))
    out: list[tuple[str, str]] = []
    for style, label, secs in steps:
        filled = int(round((secs / longest) * bar_w)) if longest > 0 else 0
        bar = "█" * filled + "·" * (bar_w - filled)
        lbl = label if len(label) <= label_w else label[: label_w - 1] + "…"
        out.append((style, f"{lbl:<{label_w}}  {bar}  {_fmt_duration(secs):>7}"))
    total = sum(s for _, _, s in steps)
    out.append(("dim", f"{'total':<{label_w}}  {'':<{bar_w}}  {_fmt_duration(total):>7}"))
    return out


# ---------------------------------------------------------------------------
# Progress bars for the long-running, countable commands
# ---------------------------------------------------------------------------


class StepProgress:
    """A count/rate/ETA progress bar for batch, eval, ingest, and downloads.

    Animated rich bar on an interactive terminal; a quiet no-op elsewhere (the
    caller still prints its own plain start/finish lines). Use as a context
    manager and call :meth:`update` / :meth:`advance` as work completes::

        with StepProgress(console, total=len(items), description="Batch", animate=ok) as bar:
            for i, item in enumerate(items, 1):
                ...
                bar.update(i)
    """

    def __init__(
        self,
        console: Any,
        *,
        total: int | None,
        description: str = "Working",
        animate: bool = True,
    ) -> None:
        self.total = total
        self.description = description
        self.console = console
        self.animate = bool(animate and _RICH_AVAILABLE and console is not None)
        self._progress: Any = None
        self._task: Any = None
        self._completed = 0

    def __enter__(self) -> "StepProgress":
        if self.animate:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            self._progress.__enter__()
            self._task = self._progress.add_task(self.description, total=self.total)
        return self

    def update(self, completed: int, total: int | None = None) -> None:
        """Set the absolute number of completed steps (and optionally the total)."""
        self._completed = completed
        if total is not None:
            self.total = total
        if self._progress is not None:
            self._progress.update(self._task, completed=completed, total=self.total)

    def advance(self, n: int = 1) -> None:
        """Advance the bar by *n* steps."""
        self.update(self._completed + n)

    def __exit__(self, *exc: Any) -> bool:
        if self._progress is not None:
            try:
                self._progress.__exit__(*exc)
            except Exception:  # noqa: BLE001
                pass
            self._progress = None
        return False
