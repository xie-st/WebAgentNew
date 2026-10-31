"""The single Rich theme and console factory for effGen.

One palette drives every table, spinner, panel, and status line so the whole
product reads as one tool. The named themes (``default``, ``high-contrast``,
``monochrome``, ``light``) come from :mod:`effgen.ui.palette`, the shared source
of truth for both the terminal and the dashboard. A theme is selected with the
``EFFGEN_THEME`` environment variable or the top-level ``--theme`` flag; with
neither set the look is the unchanged default.

``NO_COLOR`` (https://no-color.org/) always wins: when it is set the console
renders structure (tables, panels, layout) with no ANSI color regardless of the
selected theme.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .palette import (
    DEFAULT_THEME_NAME,
    cli_palette,
    normalize_theme_name,
)

try:  # rich is a base dependency, but degrade to plain output if it is absent.
    from rich.console import Console
    from rich.theme import Theme

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich is normally present
    Console = None  # type: ignore[assignment,misc]
    Theme = None  # type: ignore[assignment,misc]
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Palette — semantic names, not raw colors, so the whole CLI stays consistent.
# ---------------------------------------------------------------------------
#
# Use the semantic keys (``effgen.success`` etc.) in markup throughout the CLI
# rather than hardcoding ``[green]``/``[red]`` so the palette is one table in
# :mod:`effgen.ui.palette`. ``_PALETTE`` is the default theme, exposed here for
# callers that expect the historical name.
_PALETTE: dict[str, str] = cli_palette(DEFAULT_THEME_NAME)

# Syntax-highlighting theme used for code blocks everywhere (CLI + notebook).
# Kept in one place so code rendering is uniform.
CODE_THEME = "monokai"

# One-time warning guard so an unknown EFFGEN_THEME is reported once, not per call.
_warned_unknown: set[str] = set()


def _build_theme(palette: dict[str, str] | None = None) -> Any:
    if not _RICH_AVAILABLE:
        return None
    return Theme(palette if palette is not None else _PALETTE)


EFFGEN_THEME = _build_theme()


def rich_available() -> bool:
    """True when the optional ``rich`` library is importable."""
    return _RICH_AVAILABLE


def color_enabled() -> bool:
    """Honor the ``NO_COLOR`` convention (https://no-color.org/).

    Returns ``False`` when ``NO_COLOR`` is set to any value, so callers can both
    build a colorless console and skip ANSI markup entirely.
    """
    return not os.environ.get("NO_COLOR")


def resolve_theme_name(explicit: str | None = None) -> str:
    """Resolve the active theme name from *explicit* or ``EFFGEN_THEME``.

    Falls back to ``default`` when nothing is set, and — for an unrecognized
    name — warns once to stderr and returns ``default`` rather than failing.
    """
    requested = explicit if explicit is not None else os.environ.get("EFFGEN_THEME")
    if not requested:
        return DEFAULT_THEME_NAME
    resolved = normalize_theme_name(requested)
    if resolved is None:
        key = str(requested).strip().lower()
        if key not in _warned_unknown:
            _warned_unknown.add(key)
            from .palette import CLI_THEMES

            names = ", ".join(sorted(CLI_THEMES))
            print(
                f"Unknown theme '{requested}'; using '{DEFAULT_THEME_NAME}'. "
                f"Available themes: {names}.",
                file=sys.stderr,
            )
        return DEFAULT_THEME_NAME
    return resolved


def get_theme(name: str | None = None) -> Any:
    """Build a Rich :class:`~rich.theme.Theme` for a named theme (or the active one)."""
    resolved = resolve_theme_name(name)
    if resolved == DEFAULT_THEME_NAME:
        return EFFGEN_THEME
    return _build_theme(cli_palette(resolved))


if _RICH_AVAILABLE:

    class _EffGenConsole(Console):  # type: ignore[misc,valid-type]
        """A console that reports a closed pipe with the conventional status.

        ``effgen models list | head`` leaves the writer with nowhere to send the
        rest of its output. Rich's default is to exit ``1``, which a caller
        cannot tell apart from a command that genuinely failed; the plain-print
        paths report ``141`` (``128 + SIGPIPE``), so this reports it too.
        """

        def on_broken_pipe(self) -> None:
            self.quiet = True
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
            except (OSError, ValueError):  # pragma: no cover - stdout already gone
                pass
            raise SystemExit(141)

else:  # pragma: no cover - rich is normally present
    _EffGenConsole = None  # type: ignore[assignment,misc]


def get_console(*, theme_name: str | None = None, **kwargs: Any) -> Any:
    """Build a :class:`~rich.console.Console` carrying the effGen theme.

    The theme is chosen from *theme_name*, else the ``EFFGEN_THEME`` environment
    variable, else the unchanged default. Honors ``NO_COLOR`` (renders without
    color, whatever the theme) and accepts any extra ``Console`` keyword
    arguments. Returns ``None`` when ``rich`` is unavailable so callers can fall
    back to plain ``print``.
    """
    if not _RICH_AVAILABLE:
        return None
    kwargs.setdefault("theme", get_theme(theme_name))
    if not color_enabled():
        kwargs["no_color"] = True
    return _EffGenConsole(**kwargs)
