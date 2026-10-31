"""Shared terminal rendering for the coding agent's diffs, ticks and summary.

Both the single-shot command (:mod:`effgen.cli.commands.code`) and the
interactive REPL (:mod:`effgen.cli.code.repl`) show the same things — a pending
edit as a colorized unified diff, a decided action as a one-line tick, and the
one-glance run footer — so those renderers live here and are called from both.

Every function routes through the CLI's human stream (``cli._human()`` /
``cli._human_stream()``), so a piped or ``--json`` invocation keeps this chatter
on stderr and leaves stdout for the answer or the JSON document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from effgen.ui.render import ascii_fold
from effgen.ui.theme import color_enabled

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

    from .edits import ProposedEdit
    from .engine import CodeRunResult
    from .permissions import ActionRecord

# Glyph per decision, so an action list reads without color.
_DECISION_GLYPH: dict[str, str] = {
    "allowed": "success",
    "withheld": "warning",
    "declined": "warning",
    "refused": "error",
}

#: Theme role per status line kind.
_STATUS_ROLES: dict[str, str] = {
    "success": "effgen.success",
    "warning": "effgen.warning",
    "error": "effgen.error",
}


#: How each tool-calling path reads in the run report. A turn that produced no
#: tool call at all is the one a user most needs this for, so the label says how
#: the call was meant to travel rather than just naming the strategy.
_TOOL_CALLING_LABELS: dict[str, str] = {
    "react": "react — tool calls read out of the model's text",
    "native": "native — tool definitions sent to the provider's tool API",
    "hybrid": "hybrid — provider tool API first, falling back to the text",
    "openai_native": "openai native tools — run through the Responses API",
    "gemini_native": "gemini native tools — run through the Gemini tool API",
}


def tool_calling_label(name: str) -> str:
    """Return the run-report description of tool-calling path *name*."""
    return _TOOL_CALLING_LABELS.get(name, name)


def print_plain(cli: "CLIInterface", text: str = "", *, style: str | None = None) -> None:
    """Print *text* exactly as given, with no console markup parsing.

    Most of what the coding agent prints is content it did not author — a
    command's output, a file path, a git line, model text, a usage example such
    as ``/apply [n]``. A console reads square brackets as markup, which drops
    ``[n]`` from a help line and raises on an unbalanced tag such as
    ``[/usr/bin]`` in a command's output, so these lines go out verbatim.

    Args:
        cli: The interface the line is written through.
        text: The line to print, used verbatim.
        style: A theme style name, applied only when color is enabled.
    """
    cli.print_line(text, style=style if color_enabled() else None)


def print_status(cli: "CLIInterface", kind: str, text: str) -> None:
    """Print a glyph-prefixed success/warning/error line, without markup.

    Args:
        cli: The interface the line is written through.
        kind: ``success``, ``warning`` or ``error``, which picks the glyph.
        text: The message shown after the glyph.
    """
    from effgen.ui.palette import glyph

    print_plain(cli, f"{glyph(kind, cli._human_stream())} {text}", style=_STATUS_ROLES.get(kind))


def print_action(cli: "CLIInterface", record: "ActionRecord") -> None:
    """Print one decided action as a progress tick."""
    from effgen.ui.palette import glyph

    stream = cli._human_stream()
    if record.decision == "allowed" and record.outcome is None:
        # The tick is printed once the outcome is known, not twice.
        return
    role = _DECISION_GLYPH.get(record.decision, "muted")
    text = record.summary
    if record.decision == "allowed":
        if record.outcome == "error":
            # An action that was permitted and then failed is not a success:
            # the glyph has to say so where there is no color to say it.
            role = "error"
            text = f"{text} — failed: {record.detail}"
    else:
        text = f"{text} — {record.decision}: {record.reason}"
    line = ascii_fold(f"  {glyph(role, stream)} {text}", stream)
    console = cli._human()
    if console and color_enabled():
        style = "effgen.success" if record.decision == "allowed" else "effgen.warning"
        if record.decision == "refused" or record.outcome == "error":
            style = "effgen.error"
        console.print(f"[{style}]{line}[/{style}]", highlight=False)
    else:
        # NO_COLOR (or no rich console): plain text, no escape codes.
        print(line, file=stream)


def print_diff(cli: "CLIInterface", edit: "ProposedEdit") -> None:
    """Show a pending edit as a colorized unified diff before it is decided."""
    from .diffs import render_diff

    stream = cli._human_stream()
    kind = "new file" if edit.is_new else "edit"
    header = f"{kind} {edit.rel_path} ({edit.stat()})"
    plain, markup = render_diff(edit.diff_text, stream)
    console = cli._human()
    if console is not None and color_enabled():
        console.print(f"[effgen.heading]{header}[/effgen.heading]", highlight=False)
        console.print(markup, highlight=False)
    else:
        # NO_COLOR (or no rich console): plain diff text, no escape codes.
        print(ascii_fold(header, stream), file=stream)
        print(ascii_fold(plain, stream), file=stream)


def print_summary(cli: "CLIInterface", result: "CodeRunResult") -> None:
    """Print the one-glance run summary via the CLI's human stream.

    Uses :func:`effgen.ui.render.summary_line` (the same footer ``effgen run``
    prints) but routes it through ``cli._human()`` so a piped run keeps it off
    stdout. The tool-calling path the turn ran through is named under it, so a
    run that called no tool shows how the call was meant to travel.
    """
    from effgen.ui.render import summary_line

    stream = cli._human_stream()
    plain, markup = summary_line(result, stream=stream)
    console = cli._human()
    if console is not None and color_enabled():
        console.print(ascii_fold(markup, stream), highlight=False)
    else:
        print(ascii_fold(plain, stream), file=stream)
    path = getattr(result, "tool_calling", "")
    if path:
        print_plain(cli, f"Tool calling: {tool_calling_label(path)}")
    note = recovered_answer_note(result)
    if note:
        print_status(cli, "warning", note)


def recovered_answer_note(result: "CodeRunResult") -> str:
    """Return the line naming what a stopped run held, or ``""`` otherwise.

    When a model will not write a final answer the run stops holding what its
    tools returned — the last tool result, or the text it had reached. That is
    progress, not an answer; this line says which, so a repeated file read is
    not mistaken for a review.
    """
    from .engine import RECOVERED_ANSWER_LABELS

    if not getattr(result, "recovered_answer", False):
        return ""
    source = getattr(result, "answer_source", "")
    return (
        f"The model did not write an answer; what the run had is "
        f"{RECOVERED_ANSWER_LABELS.get(source, source)}."
    )
