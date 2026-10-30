"""Rendering helpers for agent results — terminal (Rich) and notebook (HTML).

These power ``AgentResponse.__str__``/``_repr_html_``/``show()``/``trace()`` and
are intentionally tolerant: a response may carry a rich event trace, a sparse
one, or none at all, and the answer may be plain text, markdown, or contain
fenced code/tables. Nothing here changes what an agent *did* — only how it reads.
"""

from __future__ import annotations

import html as _html
import re
import unicodedata
from typing import Any

from .palette import GLYPHS, GLYPHS_ASCII, glyph, supports_unicode
from .theme import CODE_THEME, color_enabled, get_console, rich_available

# Typographic characters the CLI prints in its own chrome (metric separators,
# bars, an em-dash, an ellipsis, spinner cells, the input chevron). On a
# non-UTF-8 stream these would raise ``UnicodeEncodeError``; :func:`ascii_fold`
# swaps them for ASCII stand-ins. Semantic status glyphs are handled separately
# by ``palette.glyph``.
_TYPO_ASCII: dict[str, str] = {
    "·": "-",
    "—": "-",
    "…": "...",
    "█": "#",
    "▌": "",
    "⏱": "",
    "•": "-",
    "›": ">",
}


def ascii_fold(text: str, stream: Any = None) -> str:
    """Return *text* unchanged on a UTF-8 stream, ASCII-folded otherwise.

    Applied at the human-print boundary for effGen's own presentation strings so
    a ``PYTHONIOENCODING=ascii`` (or other non-UTF-8) console prints a readable
    ASCII form instead of raising. On a UTF-8 stream the text is returned
    verbatim, so byte-for-byte output on a normal terminal is unchanged.
    """
    if not text or supports_unicode(stream):
        return text
    return "".join(_TYPO_ASCII.get(ch, ch) for ch in text)


# Every semantic glyph paired with the ASCII stand-in already chosen for it, so
# a stream-level fold substitutes exactly what ``palette.glyph`` would have.
_GLYPH_ASCII: dict[str, str] = {GLYPHS[name]: GLYPHS_ASCII.get(name, "") for name in GLYPHS}

# Characters that reach the stream from prose rather than from the CLI's own
# chrome: mathematical relations in a prompt template's constraint text, an en
# dash in a description, units in a metrics line.
_PROSE_ASCII: dict[str, str] = {
    "–": "-",
    "≤": "<=",
    "≥": ">=",
    "≈": "~",
    "±": "+/-",
    "×": "x",
    "°": "",
    "§": "S",
    "∈": " in ",
    "↑": "^",
    "↓": "v",
    "↔": "<->",
    "▶": ">",
    "️": "",  # variation selector: carries no width of its own
}

# What :func:`fold_for_encoding` substitutes before falling back to a rule:
# the CLI's typography, its glyph table, and the prose characters above.
_STREAM_ASCII: dict[str, str] = {**_GLYPH_ASCII, **_PROSE_ASCII, **_TYPO_ASCII}


def _ascii_stand_in(ch: str) -> str:
    """An ASCII substitute for a character with no entry in the table.

    Box drawing and block elements are the shapes a table or a bar is made of,
    so they keep their orientation (``-``, ``|``, ``+``, ``#``). A pictograph
    carries decoration rather than content and is dropped. Anything else is
    decomposed if that yields ASCII (``é`` -> ``e``) and otherwise replaced.
    """
    if "─" <= ch <= "╿":  # box drawing
        name = unicodedata.name(ch, "")
        vertical, horizontal = "VERTICAL" in name, "HORIZONTAL" in name
        if vertical and not horizontal:
            return "|"
        if horizontal and not vertical:
            return "-"
        return "+"
    if "▀" <= ch <= "▟":  # block elements (bars, the logo banner)
        return "#"
    if "⠀" <= ch <= "⣿":  # braille cells (spinner frames)
        return "*"
    if unicodedata.category(ch) == "So":  # pictographs and emoji
        return ""
    decomposed = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
    return decomposed or "?"


def fold_for_encoding(text: str, encoding: str) -> str:
    """Return *text* rewritten so *encoding* can encode every character.

    Known typography and glyphs become their ASCII stand-ins; anything left that
    the codec still cannot represent is substituted rather than allowed to
    raise. ASCII text is returned unchanged.
    """
    if text.isascii():
        return text
    folded = "".join(_STREAM_ASCII.get(ch, ch) for ch in text)
    try:
        folded.encode(encoding)
    except UnicodeEncodeError:
        folded = "".join(
            ch if _encodable(ch, encoding) else _ascii_stand_in(ch) for ch in folded
        )
    except LookupError:
        return folded.encode("ascii", "replace").decode("ascii")
    return folded


def _encodable(ch: str, encoding: str) -> bool:
    try:
        ch.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class _AsciiFoldingStream:
    """Text-stream proxy that folds what the wrapped stream cannot encode.

    A terminal forced to a hard-ascii encoding (``PYTHONIOENCODING=ascii``)
    raises ``UnicodeEncodeError`` on the first em-dash or box character, which
    turns a listing command into a failed one. Wrapping the stream moves the
    substitution to the single point where text becomes bytes, so a command --
    including one added later, and including output produced by a library the
    CLI does not control -- prints a readable ASCII form and still exits with
    its real status. Every other stream operation is the wrapped stream's.
    """

    __slots__ = ("_stream",)

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    @property
    def wrapped(self) -> Any:
        """The stream this proxy writes through."""
        return self._stream

    def write(self, text: Any) -> Any:
        if not isinstance(text, str) or not text:
            return self._stream.write(text)
        encoding = getattr(self._stream, "encoding", None) or "ascii"
        return self._stream.write(fold_for_encoding(text, encoding))

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> Any:
        return iter(self._stream)


def ascii_folding_stream(stream: Any) -> Any:
    """Return *stream* itself when it can encode the CLI's glyphs, else a proxy.

    On a UTF-8 terminal the caller keeps the identical stream object, so the
    bytes a normal terminal receives are unchanged.
    """
    if stream is None or isinstance(stream, _AsciiFoldingStream):
        return stream
    if supports_unicode(stream):
        return stream
    return _AsciiFoldingStream(stream)


def json_ensure_ascii(stream: Any = None) -> bool:
    """Whether JSON written to *stream* has to escape its non-ASCII characters.

    ``--json`` output stays raw UTF-8 wherever the stream can carry it. When it
    cannot, the payload is emitted with ``\\uXXXX`` escapes: still valid JSON
    decoding to the same value, rather than a transliterated or dropped
    character on a machine-readable path.
    """
    return not supports_unicode(stream)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

# Icons for trace event types — kept in sync with the live status spinner so the
# whole product feels like one tool.
_EVENT_ICONS: dict[str, str] = {
    "task_start": "▶",
    "task_complete": "✓",
    "task_failed": "✗",
    "routing_decision": "🔀",
    "task_decomposition": "🧩",
    "sub_agent_spawn": "⊕",
    "sub_agent_start": "→",
    "sub_agent_progress": "·",
    "sub_agent_complete": "✓",
    "sub_agent_failed": "✗",
    "tool_call_start": "🔧",
    "tool_call_complete": "✓",
    "tool_call_failed": "✗",
    "result_synthesis": "✎",
    "reasoning_step": "💭",
    "error": "✗",
    "warning": "⚠",
    "info": "·",
}


def _extract_cost(metadata: dict[str, Any] | None) -> float | None:
    """Pull a positive USD cost out of response metadata, if any."""
    if not metadata:
        return None
    for key in ("cost_usd", "cost", "total_cost"):
        val = metadata.get(key)
        if isinstance(val, int | float) and val > 0:
            return float(val)
    return None


def format_cost(cost: float | None) -> str | None:
    """Format a USD cost so a real charge is never hidden behind ``$0.0000``.

    The cheap small models effGen targets cost a tiny fraction of a cent per
    turn, so a fixed ``${:.4f}`` collapses almost every real cost to ``$0.0000``
    and guts the "show what it costs" story. This widens precision only when
    needed:

    - ``None`` → ``None`` (no cost info; the caller omits the field).
    - ``0`` → ``"$0.00"`` (genuinely free / local — real, not fabricated).
    - ``>= $0.0001`` → ``"$0.0006"`` (the familiar 4-decimal form).
    - ``< $0.0001`` (would round to ``$0.0000``) → 6 decimals (``"$0.000049"``),
      falling back to scientific (``"$4.9e-07"``) for vanishingly small costs.
    """
    if cost is None:
        return None
    try:
        c = float(cost)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return "$0.00"
    if c >= 1e-4:
        return f"${c:.4f}"
    if c >= 1e-6:
        return f"${c:.6f}"
    return f"${c:.1e}"


def _summary_parts(response: Any) -> list[str]:
    """The ``3.2s · 2 tools · 1,204 tokens · $0.0006`` metric fragments."""
    secs = float(getattr(response, "execution_time", 0.0) or 0.0)
    tools = int(getattr(response, "tool_calls", 0) or 0)
    tokens = int(getattr(response, "tokens_used", 0) or 0)
    cost = _extract_cost(getattr(response, "metadata", None))
    parts = [f"{secs:.1f}s"]
    if tools:
        parts.append(f"{tools} tool" + ("s" if tools != 1 else ""))
    if tokens:
        parts.append(f"{tokens:,} tokens")
    cost_str = format_cost(cost)
    if cost_str is not None:
        parts.append(cost_str)
    return parts


def _tool_names(response: Any) -> list[str]:
    """Distinct tool names used, read from the execution trace (best effort)."""
    names: list[str] = []
    for ev in getattr(response, "execution_trace", None) or []:
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "tool_call_start":
            name = (ev.get("data") or {}).get("tool_name") or (ev.get("data") or {}).get("tool")
            if name and name not in names:
                names.append(str(name))
    return names


def summary_line(
    response: Any = None,
    *,
    mode: str = "run",
    stream: Any = None,
    elapsed: float = 0.0,
    tokens: int = 0,
    cost: float = 0.0,
    interrupted: bool = False,
    session: tuple[int, int, float] | None = None,
) -> tuple[str, str]:
    """Build the one-glance summary for a finished turn (``(plain, markup)``).

    Two presets share one metric vocabulary and one cost extractor:

    - ``mode="run"`` — a one-shot run/wizard footer read from *response*::

          ✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006

      A partial (iteration-cap) run and an outright failure are marked
      distinctly from a success.
    - ``mode="chat"`` — a conversational per-turn footer from the explicit
      *elapsed*/*tokens*/*cost*, with a running *session* total once a session
      has more than one turn::

          · 3.2s · 318 tok · $0.0003    (session: 4 turns · 1,020 tok · $…)

    The status glyphs resolve through :func:`palette.glyph` so a non-UTF-8
    stream falls back to an ASCII stand-in instead of raising.

    Args:
        response: The finished response a ``run`` footer reads its numbers from.
        mode: ``run`` for a one-shot footer, ``chat`` for a per-turn one.
        stream: The stream the glyphs are chosen for.
        elapsed: Seconds the turn took, for ``chat`` mode.
        tokens: Tokens the turn used, for ``chat`` mode.
        cost: What the turn cost, for ``chat`` mode.
        interrupted: Mark the turn as stopped before it finished.
        session: Running ``(turns, tokens, cost)`` totals for the session.

    Returns:
        The ``(plain, markup)`` pair for the footer.
    """
    if mode == "chat":
        parts = [f"{elapsed:.1f}s"]
        if tokens:
            parts.append(f"{tokens:,} tok")
        if cost and cost > 0:
            c = format_cost(cost)
            if c:
                parts.append(c)
        footer = "· " + " · ".join(parts)
        if interrupted:
            footer += " · stopped"
        if session:
            turns, s_tokens, s_cost = session
            if turns > 1 and (s_tokens or s_cost):
                run = f"{turns} turns · {s_tokens:,} tok"
                if s_cost > 0:
                    rc = format_cost(s_cost)
                    if rc:
                        run += f" · {rc}"
                footer += f"    (session: {run})"
        return footer, footer

    body = " · ".join(_summary_parts(response))
    meta = getattr(response, "metadata", None) or {}
    success = bool(getattr(response, "success", False))
    g_ok = glyph("success", stream)
    g_err = glyph("error", stream)
    g_warn = glyph("warning", stream)
    if success:
        plain = f"{g_ok} Done in {body}"
        markup = f"[green]{g_ok}[/green] Done in {body}"
    elif getattr(response, "outcome", None) == "stopped" or meta.get("partial"):
        # A run the loop stopped is not a completed success, but what it had
        # reached is still shown: mark it distinctly from an outright failure so
        # the progress is not read as finished. The footer names what stopped
        # it, since the iteration cap is only one of the reasons.
        tail = f" · {body}" if body else ""
        reason = getattr(response, "stop_reason", None) or meta.get(
            "reason", "max_iterations_partial"
        )
        plain = f"{g_warn} Stopped ({reason}) — partial result{tail}"
        markup = (
            f"[yellow]{g_warn}[/yellow] Stopped ([yellow]{reason}[/yellow])"
            f" — partial result{tail}"
        )
    else:
        reason = meta.get("reason", "failed")
        tail = f" · {body}" if body else ""
        plain = f"{g_err} Failed ({reason}){tail}"
        markup = f"[red]{g_err}[/red] Failed ([yellow]{reason}[/yellow]){tail}"
    return plain, markup


# ---------------------------------------------------------------------------
# Terminal rendering (Rich)
# ---------------------------------------------------------------------------


# Small (and very common) SLM quirk: the model wraps its *entire* answer in a
# ```markdown … ``` fence. Rendered verbatim that becomes one gray code box.
# We unwrap a sole outer markdown fence at render time only — never mutating the
# stored ``output`` — so such answers read as the markdown they actually are.
_OUTER_MD_FENCE_RE = re.compile(r"^```(?:markdown|md)\s*$", re.IGNORECASE)


def _unwrap_markdown_fence(text: str) -> str:
    """If the whole answer is a single ```markdown``` fence, return its contents."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return text
    lines = s.split("\n")
    if len(lines) >= 2 and _OUTER_MD_FENCE_RE.match(lines[0].strip()) and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def answer_renderable(text: str) -> Any:
    """Return a Rich renderable for an answer, rendering markdown + code + tables.

    Falls back to plain ``Text`` when ``rich`` is unavailable. Honors
    ``NO_COLOR`` by leaving code-block styling to the (colorless) console.
    """
    text = _unwrap_markdown_fence(text or "")
    if not rich_available():  # pragma: no cover - rich normally present
        return text
    from rich.markdown import Markdown
    from rich.text import Text

    # Markdown() renders headings, lists, fenced code (highlighted), and tables.
    # If the text is plainly not markdown-ish, a Markdown render is still fine,
    # but we keep a Text fallback for empties.
    if not text.strip():
        return Text("(no output)", style="effgen.muted")
    code_theme = CODE_THEME if color_enabled() else "ansi_light"
    try:
        return Markdown(text, code_theme=code_theme)
    except Exception:  # noqa: BLE001 - never let formatting break printing
        return Text(text)


def answer_surface(
    text: str,
    *,
    success: bool = True,
    framed: bool = True,
    partial: bool = False,
    title: str | None = None,
    label: str | None = None,
    console: Any = None,
) -> None:
    """Render a finished answer once, using the shared markdown renderer + roles.

    ``framed=True`` draws a bordered panel (the one-shot ``run``/wizard/report
    surface); ``framed=False`` draws an inline labelled block (conversational
    ``chat``). Both go through :func:`answer_renderable`, so markdown, code, and
    tables render the same way on every surface, and both take their colour from
    the ``effgen.*`` theme roles rather than hardcoded style names — so a chosen
    ``--theme`` and ``NO_COLOR`` reach every answer.

    The border/title role follows the outcome: ``effgen.success`` for a success,
    ``effgen.warning`` for a partial (iteration-cap) result, ``effgen.error``
    otherwise. Degrades to plain ``print`` when ``rich`` is unavailable.

    Args:
        text: The answer to render.
        success: Whether the run succeeded, which picks the colour role.
        framed: Draw a bordered panel instead of an inline block.
        partial: Mark the answer as cut short by the iteration cap.
        title: Panel title, for the framed form.
        label: Inline label printed before the block, for the unframed form.
        console: The console to write to, defaulting to the shared one.
    """
    text = text or ""
    if not rich_available():  # pragma: no cover - rich normally present
        if not framed and label:
            print(f"{label} {text}")
        else:
            print(text)
        return

    console = console or get_console()
    if framed:
        from rich.panel import Panel

        if success:
            border = "effgen.success"
        elif partial:
            border = "effgen.warning"
        else:
            border = "effgen.error"
        ttl = title if title is not None else ("Answer" if success else "Failed")
        console.print(
            Panel(
                answer_renderable(text),
                title=f"[{border}]{ttl}[/{border}]",
                border_style=border,
                expand=True,
            )
        )
    else:
        if label:
            lbl = f"[effgen.success]{label}[/effgen.success] " if color_enabled() else f"{label} "
            console.print(lbl, end="", highlight=False)
        console.print(answer_renderable(text))


def response_show(response: Any, console: Any = None) -> None:
    """Print a compact human view: the answer, then a one-line metric footer.

    Uses the shared theme; degrades to plain ``print`` without ``rich``.
    """
    output = getattr(response, "output", "") or ""
    success = bool(getattr(response, "success", False))
    parts = _summary_parts(response)
    body = " · ".join(parts)

    if not rich_available():  # pragma: no cover - rich normally present
        print(output)
        mark = glyph("success") if success else glyph("error")
        print(ascii_fold(f"{mark} {body}"))
        return

    console = console or get_console()

    answer_surface(output, success=success, framed=True, console=console)

    tool_names = _tool_names(response)
    ok = glyph("success")
    err = glyph("error")
    icon = f"[effgen.success]{ok}[/effgen.success]" if success else f"[effgen.error]{err}[/effgen.error]"
    footer_bits = [icon, f"[effgen.muted]{body}[/effgen.muted]"]
    if tool_names:
        footer_bits.append("[effgen.muted]·[/effgen.muted]")
        footer_bits.append("[effgen.tool]" + ", ".join(tool_names) + "[/effgen.tool]")
    console.print(ascii_fold(" ".join(footer_bits)), highlight=False)


def response_trace(response: Any, console: Any = None) -> None:
    """Print the full reasoning trace (every recorded step) as a tree.

    Falls back to a plain numbered list without ``rich`` or when there is no
    structured trace.
    """
    events = [e for e in (getattr(response, "execution_trace", None) or []) if isinstance(e, dict)]

    if not rich_available():  # pragma: no cover - rich normally present
        if not events:
            print("(no execution trace recorded)")
            return
        for i, ev in enumerate(events, 1):
            etype = ev.get("type", "step")
            msg = ev.get("message", "") or _trace_data_summary(ev.get("data"))
            print(f"{i:>2}. [{etype}] {msg}")
        return

    from rich.text import Text
    from rich.tree import Tree

    console = console or get_console()
    if not events:
        console.print("[effgen.muted]No step-by-step trace was recorded for this run.[/effgen.muted]", highlight=False)
        # Still show the answer so trace() is never empty-handed.
        console.print(answer_renderable(getattr(response, "output", "") or ""))
        return

    tree = Tree("[effgen.heading]Reasoning trace[/effgen.heading]")
    for ev in events:
        etype = str(ev.get("type", "step"))
        icon = _EVENT_ICONS.get(etype, "·")
        msg = ev.get("message", "") or ""
        label = Text()
        style = (
            "effgen.error"
            if etype in ("error", "task_failed", "tool_call_failed", "sub_agent_failed")
            else "effgen.muted"
        )
        label.append(f"{icon} ", style=style)
        label.append(etype, style="effgen.label")
        if msg:
            label.append(f"  {msg}")
        node = tree.add(label)
        detail = _trace_data_summary(ev.get("data"))
        if detail:
            node.add(Text(detail, style="effgen.muted"))
    console.print(tree)


def _trace_data_summary(data: Any) -> str:
    """One-line, redaction-safe summary of an event's ``data`` payload."""
    if not isinstance(data, dict) or not data:
        return ""
    interesting = ("tool_name", "tool", "arguments", "args", "input", "result", "output", "error", "model")
    bits = []
    for key in interesting:
        if key in data and data[key] not in (None, "", {}, []):
            val = str(data[key]).replace("\n", " ")
            if len(val) > 120:
                val = val[:117] + "…"
            bits.append(f"{key}={val}")
    return "  ".join(bits)


# ---------------------------------------------------------------------------
# Notebook rendering (HTML card)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
# A GitHub-flavored table separator row, e.g. ``|---|:--:|`` (alignment colons ok).
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")


def _split_table_row(line: str) -> list[str]:
    """Split a ``| a | b |`` row into trimmed cell strings."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _md_inline(text: str) -> str:
    """Escape HTML then apply a tiny subset of inline markdown (code/bold/italic)."""
    out = _html.escape(text)
    out = _INLINE_CODE_RE.sub(
        lambda m: f"<code>{m.group(1)}</code>", out
    )
    out = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    return out


def _markdown_to_html(text: str) -> str:
    """Minimal, dependency-free markdown→HTML for the notebook answer card.

    Handles fenced code blocks, headings, unordered/ordered list items, inline
    code/bold/italic, and paragraphs. Everything is HTML-escaped first, so model
    output cannot inject markup into the notebook.
    """
    if not text or not text.strip():
        return '<em style="color:#888">(no output)</em>'

    text = _unwrap_markdown_fence(text)
    lines = text.split("\n")
    html_parts: list[str] = []
    in_code = False
    code_buf: list[str] = []
    list_buf: list[str] = []
    list_kind: str | None = None  # "ul" or "ol"

    def flush_list() -> None:
        nonlocal list_kind
        if list_buf:
            tag = list_kind or "ul"
            html_parts.append(
                f"<{tag} style='margin:4px 0 8px 22px'>"
                + "".join(f"<li>{item}</li>" for item in list_buf)
                + f"</{tag}>"
            )
            list_buf.clear()
            list_kind = None

    def emit_table(header: list[str], rows: list[list[str]]) -> None:
        cell = "border:1px solid #ddd;padding:4px 10px;text-align:left"
        thead = "".join(
            f"<th style='{cell};background:#f3f3f3'>{_md_inline(c)}</th>" for c in header
        )
        body = "".join(
            "<tr>" + "".join(f"<td style='{cell}'>{_md_inline(c)}</td>" for c in r) + "</tr>"
            for r in rows
        )
        html_parts.append(
            "<table style='border-collapse:collapse;margin:8px 0;font-size:0.9em'>"
            f"<thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>"
        )

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        fence = _FENCE_RE.match(line.strip())
        if fence:
            if in_code:
                flush_list()
                code = _html.escape("\n".join(code_buf))
                html_parts.append(
                    "<pre style='background:#1e1e1e;color:#e6e6e6;padding:10px;"
                    "border-radius:6px;overflow-x:auto;font-size:0.85em'>"
                    f"<code>{code}</code></pre>"
                )
                code_buf.clear()
                in_code = False
            else:
                flush_list()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_list()
            level = min(len(heading.group(1)) + 2, 6)  # h1→h3 so it sits in a card
            html_parts.append(
                f"<h{level} style='margin:8px 0 4px'>{_md_inline(heading.group(2))}</h{level}>"
            )
            i += 1
            continue

        # GitHub-flavored table: a header row followed by a |---|---| separator.
        if (
            "|" in line
            and i + 1 < n
            and _TABLE_SEP_RE.match(lines[i + 1])
            and "|" in lines[i + 1]
        ):
            flush_list()
            header = _split_table_row(line)
            rows: list[list[str]] = []
            i += 2  # consume header + separator
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_table_row(lines[i]))
                i += 1
            emit_table(header, rows)
            continue

        stripped = line.strip()
        ul = re.match(r"^[-*+]\s+(.*)$", stripped)
        ol = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if ul:
            if list_kind not in (None, "ul"):
                flush_list()
            list_kind = "ul"
            list_buf.append(_md_inline(ul.group(1)))
            i += 1
            continue
        if ol:
            if list_kind not in (None, "ol"):
                flush_list()
            list_kind = "ol"
            list_buf.append(_md_inline(ol.group(1)))
            i += 1
            continue

        if not stripped:
            flush_list()
            i += 1
            continue

        flush_list()
        html_parts.append(f"<p style='margin:4px 0'>{_md_inline(stripped)}</p>")
        i += 1

    if in_code:  # unterminated fence — render what we have
        code = _html.escape("\n".join(code_buf))
        html_parts.append(
            "<pre style='background:#1e1e1e;color:#e6e6e6;padding:10px;"
            f"border-radius:6px;overflow-x:auto;font-size:0.85em'><code>{code}</code></pre>"
        )
    flush_list()
    return "\n".join(html_parts)


# Shared, theme-aware card chrome for the Jupyter `_repr_html_` cards below.
# JupyterLab/VS Code/classic Jupyter render this HTML directly in the page
# (not a sandboxed iframe), so a `prefers-color-scheme` media query follows
# the notebook's own OS/browser theme. The class name is repeated verbatim
# across every card instance in a notebook, which is safe — CSS rules are
# idempotent, not additive state.
_CARD_STYLE = """<style>
.effgen-card {
  --effgen-bg: transparent;
  --effgen-border: #ddd;
  --effgen-text: #1a1a1a;
  --effgen-muted: #666;
  --effgen-muted-2: #888;
  --effgen-divider: #eee;
  --effgen-success: #2e7d32;
  --effgen-fail: #c62828;
}
@media (prefers-color-scheme: dark) {
  .effgen-card {
    --effgen-bg: #1e1e1e;
    --effgen-border: #444;
    --effgen-text: #e8e8e8;
    --effgen-muted: #aaa;
    --effgen-muted-2: #999;
    --effgen-divider: #3a3a3a;
    --effgen-success: #66bb6a;
    --effgen-fail: #ef5350;
  }
}
</style>"""


def response_html(response: Any) -> str:
    """Build a self-contained HTML card for a result (Jupyter ``_repr_html_``).

    The card shows the markdown-rendered answer plus a compact metric strip
    (status · time · tools · tokens · cost) and, when present, a collapsible
    step trace. All model-derived text is HTML-escaped.
    """
    output = getattr(response, "output", "") or ""
    success = bool(getattr(response, "success", False))
    answer_html = _markdown_to_html(output)

    accent = "var(--effgen-success)" if success else "var(--effgen-fail)"
    badge = "✓ success" if success else "✗ failed"
    metrics = " &nbsp;·&nbsp; ".join(_summary_parts(response))
    tool_names = _tool_names(response)
    tools_html = (
        f" &nbsp;·&nbsp; tools: {_html.escape(', '.join(tool_names))}" if tool_names else ""
    )

    # Optional collapsible trace.
    events = [e for e in (getattr(response, "execution_trace", None) or []) if isinstance(e, dict)]
    trace_html = ""
    if events:
        rows = []
        for ev in events:
            etype = str(ev.get("type", "step"))
            icon = _EVENT_ICONS.get(etype, "·")
            msg = _html.escape(str(ev.get("message", "") or ""))
            detail = _html.escape(_trace_data_summary(ev.get("data")))
            line = f"{icon} <strong>{_html.escape(etype)}</strong>"
            if msg:
                line += f" — {msg}"
            if detail:
                line += (
                    "<br><span style='color:var(--effgen-muted-2);font-size:0.85em'>"
                    f"&nbsp;&nbsp;{detail}</span>"
                )
            rows.append(f"<li style='margin:3px 0'>{line}</li>")
        trace_html = (
            "<details style='margin-top:8px'>"
            f"<summary style='cursor:pointer;color:{accent}'>Step trace "
            f"({len(events)} steps)</summary>"
            "<ul style='margin:6px 0 0 18px;padding:0;list-style:none'>"
            + "".join(rows)
            + "</ul></details>"
        )

    return f"""
{_CARD_STYLE}
<div class="effgen-card" style="background:var(--effgen-bg);border:1px solid var(--effgen-border);
            border-left:4px solid {accent};border-radius:8px;
            padding:12px 16px;margin:6px 0;font-family:-apple-system,Segoe UI,sans-serif;
            max-width:920px">
  <div style="font-size:0.8em;color:{accent};font-weight:600;margin-bottom:6px">
    effGen result &nbsp;·&nbsp; {badge}
  </div>
  <div style="color:var(--effgen-text);line-height:1.5">{answer_html}</div>
  <div style="font-size:0.8em;color:var(--effgen-muted);margin-top:10px;
              border-top:1px solid var(--effgen-divider);padding-top:6px">
    {metrics}{tools_html}
  </div>
  {trace_html}
</div>
""".strip()


def _generation_metric_parts(result: Any) -> list[str]:
    """Metric fragments for a raw model ``GenerationResult`` card/footer."""
    meta = getattr(result, "metadata", None) or {}
    parts: list[str] = []
    model = getattr(result, "model_name", None)
    if model:
        parts.append(str(model))
    secs = meta.get("duration_s")
    if secs is None and isinstance(meta.get("latency_ms"), int | float):
        secs = meta["latency_ms"] / 1000.0
    if isinstance(secs, int | float) and secs > 0:
        parts.append(f"{secs:.1f}s")
    tokens = meta.get("total_tokens")
    if not isinstance(tokens, int | float) or tokens <= 0:
        tokens = getattr(result, "tokens_used", 0) or 0
    if tokens:
        parts.append(f"{int(tokens):,} tokens")
    cost_str = format_cost(_extract_cost(meta))
    if cost_str is not None:
        parts.append(cost_str)
    return parts


def generation_result_html(result: Any) -> str:
    """Compact HTML card for a raw model ``GenerationResult`` (Jupyter).

    Mirrors the agent result card at the model layer so
    ``load_model(...).generate(...)`` renders a tidy card in a notebook instead
    of a dataclass repr. Simpler than :func:`response_html` — no tools or step
    trace exist at this layer — but consistent in look and metric strip.
    """
    text = getattr(result, "text", "") or ""
    finish = str(getattr(result, "finish_reason", "") or "")
    ok = finish != "error"
    accent = "var(--effgen-success)" if ok else "var(--effgen-fail)"
    answer_html = _markdown_to_html(text)
    metrics = " &nbsp;·&nbsp; ".join(_generation_metric_parts(result))
    finish_html = (
        f" &nbsp;·&nbsp; finish: {_html.escape(finish)}" if finish and finish not in ("stop", "") else ""
    )
    return f"""
{_CARD_STYLE}
<div class="effgen-card" style="background:var(--effgen-bg);border:1px solid var(--effgen-border);
            border-left:4px solid {accent};border-radius:8px;
            padding:12px 16px;margin:6px 0;font-family:-apple-system,Segoe UI,sans-serif;
            max-width:920px">
  <div style="font-size:0.8em;color:{accent};font-weight:600;margin-bottom:6px">
    effGen model output
  </div>
  <div style="color:var(--effgen-text);line-height:1.5">{answer_html}</div>
  <div style="font-size:0.8em;color:var(--effgen-muted);margin-top:10px;
              border-top:1px solid var(--effgen-divider);padding-top:6px">
    {metrics}{finish_html}
  </div>
</div>
""".strip()


def generation_result_markdown(result: Any) -> str:
    """Plain-markdown view of a ``GenerationResult`` (Jupyter ``_repr_markdown_``)."""
    text = getattr(result, "text", "") or ""
    metrics = " · ".join(_generation_metric_parts(result))
    footer = f"\n\n<sub>{metrics}</sub>" if metrics else ""
    return f"{text}{footer}"
