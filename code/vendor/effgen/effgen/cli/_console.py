"""Terminal output for the ``effgen`` CLI.

:mod:`effgen.cli._main` owns argument parsing and dispatch and re-exports these
names, so ``effgen.cli._main`` stays the import path callers use. Holds the
methods :class:`effgen.cli._main.CLIInterface` prints through: the stdout/stderr
split for machine-readable output, the markup-free data and line writers, the
success/warning/error marks, the failure panel, and the statistics table. Each
renders through rich when it is installed and falls back to plain text when it
is not.
"""

from __future__ import annotations

import sys
from typing import Any

from effgen.cli import progress as _progress
from effgen.ui.palette import glyph as _glyph
from effgen.ui.theme import get_console as _get_console


class CLIConsoleMixin:
    """The output half of ``CLIInterface``.

    Every method reads ``self.console``, ``self._human_to_stderr`` and
    ``self._err_console``, which ``CLIInterface.__init__`` sets. The rich
    availability flag and the ``Table`` class are read off
    :mod:`effgen.cli._main` at call time, so an install without rich takes the
    plain-text branch and a caller replacing either one on that module is
    honored here.
    """

    def _human(self):
        """Return the console human-facing output should go to (stdout/stderr)."""
        from effgen.cli import _main

        if self._human_to_stderr:
            if self._err_console is None and _main.RICH_AVAILABLE:
                self._err_console = _get_console(stderr=True)
            return self._err_console
        return self.console

    def _animate(self, args) -> bool:
        """Whether to show live animation for this invocation (TTY-aware, opt-out)."""
        return _progress.animation_enabled(
            quiet=getattr(args, 'quiet', False),
            no_animation=getattr(args, 'no_animation', False),
        )

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print with rich formatting if available.

        A line written here is a message to a person, so it renders in the one
        style its markup asks for: value highlighting is off unless the caller
        turns it on. With it on, the console recolours numbers, brackets, paths
        and quoted words inside the line, and a version, a duration or a
        parenthesised count comes out split across several colours. Passing a
        table, panel or other renderable is unaffected — highlighting applies to
        strings.

        The plain branch flushes, because a rich console writes through on every
        call and the built-in ``print`` does not. Without it, an install with no
        rich left the output of a long-running command in the interpreter's
        block buffer: ``effgen serve > server.log`` wrote nothing at all — the
        API key it had just minted included — until the server was stopped.
        """
        console = self._human()
        if console:
            kwargs.setdefault("highlight", False)
            console.print(*args, **kwargs)
        else:
            kwargs.pop("highlight", None)
            kwargs.setdefault("flush", True)
            print(*args, file=sys.stderr if self._human_to_stderr else None, **kwargs)

    def print_data(self, text: str) -> None:
        """Write *text* to stdout byte-for-byte.

        Used for output that is data rather than chatter — an exported session,
        a rendered prompt, a model answer. It bypasses the markup renderer, so
        square-bracket content such as ``[bold]`` or ``[1]`` survives instead of
        being read as console markup and dropped.
        """
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def print_line(self, text: str, style: str | None = None) -> None:
        """Print one line with no markup parsing and no value highlighting.

        Stored values such as model ids, timestamps and durations stay intact
        instead of being read as console markup, split into separately colored
        tokens, or (for text like ``root:x:0:0``) rewritten into emoji.
        """
        console = self._human()
        if console:
            console.print(text, highlight=False, markup=False, emoji=False, style=style)
        else:
            print(text, file=sys.stderr if self._human_to_stderr else None, flush=True)

    def print_header(self, text: str) -> None:
        """Print a header."""
        console = self._human()
        if console:
            console.print(f"\n[effgen.heading]{text}[/effgen.heading]", highlight=False)
        else:
            print(f"\n=== {text} ===", file=sys.stderr if self._human_to_stderr else None,
                  flush=True)

    def _human_stream(self):
        """The underlying text stream human output goes to (for glyph fallback)."""
        return sys.stderr if self._human_to_stderr else sys.stdout

    def print_success(self, text: str) -> None:
        """Print success message."""
        console = self._human()
        mark = _glyph("success", self._human_stream())
        if console:
            console.print(f"[green]{mark}[/green] {text}", highlight=False)
        else:
            print(f"{mark} {text}", file=sys.stderr if self._human_to_stderr else None,
                  flush=True)

    def print_error(self, text: str) -> None:
        """Print error message."""
        console = self._human()
        mark = _glyph("error", self._human_stream())
        if console:
            console.print(f"[red]{mark}[/red] {text}", highlight=False)
        else:
            print(f"{mark} {text}", file=sys.stderr if self._human_to_stderr else None,
                  flush=True)

    def print_warning(self, text: str) -> None:
        """Print warning message."""
        console = self._human()
        mark = _glyph("warning", self._human_stream())
        if console:
            console.print(f"[effgen.warning]{mark}[/effgen.warning] {text}", highlight=False)
        else:
            print(f"{mark} {text}", file=sys.stderr if self._human_to_stderr else None,
                  flush=True)

    def print_error_panel(self, message: str, *, title: str = "Error") -> None:
        """Render a failure as a red-bordered panel, or a styled line without rich.

        A run that fails at generation and a run that fails to load its model
        then read the same way — one red panel — instead of a panel in one case
        and a bare line in the other. The message is shown as plain text so
        provider error strings can't inject console markup.
        """
        from effgen.cli import _main

        console = self._human()
        if console and _main.RICH_AVAILABLE:
            from rich.panel import Panel
            from rich.text import Text
            console.print(
                Panel(Text(message or ""), title=f"[red]{title}[/red]", border_style="red")
            )
        else:
            self.print_error(message)

    def _create_stats_table(self, stats: dict[str, Any]) -> Any:
        """Create statistics table."""
        from effgen.cli import _main

        if not self.console:
            return stats

        table = _main.Table(title="Execution Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in stats.items():
            table.add_row(key, str(value))

        return table
