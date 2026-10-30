"""Aligned table rendering shared by the CLI listing/results commands.

Renders a Rich table on an interactive terminal and a clean, aligned
plain-text table when output is piped or Rich is unavailable, so the results
commands (``eval``, ``compare``, ``cost``, ``sessions list``) read as one
family without adding box-drawing or color to piped/``--json`` output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .theme import rich_available


def console_is_interactive(console: Any) -> bool:
    """True when *console* writes to a real terminal (not a pipe or file)."""
    return bool(console is not None and getattr(console, "is_terminal", False))


def check_mark(present: bool, stream: Any = None) -> str:
    """A capability cell: the success glyph when *present*, else empty.

    Routes the ``✓`` used across the catalog tables through the one glyph table
    so every listing marks a capability the same way and a non-UTF-8 console
    gets an ASCII stand-in instead of raising. On a UTF-8 terminal the cell is
    the same ``✓`` as before.
    """
    if not present:
        return ""
    from .palette import glyph

    return glyph("success", stream)


def empty_state(
    console: Any,
    *,
    title: str,
    message: str,
    hints: Sequence[str] = (),
    stream: Any = None,
) -> None:
    """Print a consistent "nothing here yet" block on an interactive terminal.

    A muted heading, the *message*, and any next-step *hints* each prefixed with
    an arrow, all through the shared theme roles and glyph table so every
    results command reports an empty result the same way. Callers gate this on
    an interactive console and keep their existing plain text for piped or
    redirected output, so the machine-readable bytes are unchanged.

    Args:
        console: The console to write to.
        title: The heading above the message.
        message: What is empty, in one line.
        hints: Next steps, each printed on its own line.
        stream: The stream the glyphs are chosen for.
    """
    from .palette import glyph

    arrow = glyph("arrow", stream) or "->"
    console.print(f"[effgen.heading]{title}[/effgen.heading]", highlight=False)
    console.print(f"[effgen.muted]{message}[/effgen.muted]", highlight=False)
    for hint in hints:
        console.print(f"  [effgen.accent]{arrow}[/effgen.accent] {hint}", highlight=False)


def render_table(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    console: Any = None,
    title: str | None = None,
    justify: Sequence[str] | None = None,
    styles: Sequence[str | None] | None = None,
    footer: Sequence[str] | None = None,
    caption: str | None = None,
    file: Any = None,
) -> None:
    """Print *rows* under *columns* as a table.

    A Rich table (with the shared theme) is used on an interactive terminal;
    output that is piped or redirected gets an aligned plain-text table with no
    box-drawing characters or color, so a ``| command | grep`` pipeline stays
    readable. ``justify`` is a per-column ``"left"``/``"right"``/``"center"``
    list; ``styles`` a per-column Rich style (ignored in the plain path);
    ``footer`` a per-column footer row; ``caption`` a trailing note.

    ``file`` forces the plain-text path to a specific stream (e.g. ``stderr``);
    a caller routing human output away from stdout under ``--json`` passes it so
    the table never lands on the JSON stream.

    Args:
        columns: The column headers.
        rows: The rows, each as many values as there are columns.
        console: The console to write to, defaulting to the shared one.
        title: A heading above the table.
        justify: Per-column alignment.
        styles: Per-column style, used only on the rich path.
        footer: A trailing row of per-column values.
        caption: A note printed under the table.
        file: Force the plain-text path to this stream.
    """
    cell_rows = [["" if c is None else str(c) for c in row] for row in rows]
    if file is None and rich_available() and console_is_interactive(console):
        _render_rich(console, columns, cell_rows, title, justify, styles, footer, caption)
    else:
        _render_plain(columns, cell_rows, title, justify, footer, caption, file)


def _render_rich(
    console: Any,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    title: str | None,
    justify: Sequence[str] | None,
    styles: Sequence[str | None] | None,
    footer: Sequence[str] | None,
    caption: str | None,
) -> None:
    from rich.table import Table

    table = Table(title=title, caption=caption, show_footer=footer is not None)
    for i, col in enumerate(columns):
        table.add_column(
            col,
            justify=(justify[i] if justify and i < len(justify) else "left"),
            style=(styles[i] if styles and i < len(styles) else None),
            footer=(footer[i] if footer and i < len(footer) else ""),
            overflow="fold",
        )
    for row in rows:
        table.add_row(*row)
    console.print(table)


def _render_plain(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    title: str | None,
    justify: Sequence[str] | None,
    footer: Sequence[str] | None,
    caption: str | None,
    file: Any = None,
) -> None:
    n = len(columns)
    body = list(rows) + ([list(footer)] if footer else [])
    widths = [len(str(columns[i])) for i in range(n)]
    for row in body:
        for i in range(n):
            cell = str(row[i]) if i < len(row) else ""
            widths[i] = max(widths[i], len(cell))

    def _fmt(row: Sequence[str]) -> str:
        cells = []
        for i in range(n):
            cell = str(row[i]) if i < len(row) else ""
            right = bool(justify and i < len(justify) and justify[i] == "right")
            cells.append(cell.rjust(widths[i]) if right else cell.ljust(widths[i]))
        return "  ".join(cells).rstrip()

    if title:
        print(title, file=file)
    print(_fmt(list(columns)), file=file)
    print("  ".join("-" * w for w in widths), file=file)
    for row in rows:
        print(_fmt(row), file=file)
    if footer:
        print("  ".join("-" * w for w in widths), file=file)
        print(_fmt(list(footer)), file=file)
    if caption:
        print(caption, file=file)
