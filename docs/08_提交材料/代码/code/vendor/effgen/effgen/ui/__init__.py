"""Shared presentation layer — one consistent look across CLI and notebooks.

This package is a *thin, dependency-light* home for how effGen renders things to
a human: the single Rich color theme, a themed :class:`~rich.console.Console`
factory, and the helpers that turn an answer (markdown, code, tables) into a
nice terminal view or a Jupyter HTML card.

It deliberately imports only :mod:`rich` and the standard library so that core
objects (e.g. ``AgentResponse``) can render themselves without pulling in the
heavy CLI module. ``NO_COLOR`` is honored everywhere.
"""

from __future__ import annotations

from .render import (
    answer_renderable,
    format_cost,
    generation_result_html,
    generation_result_markdown,
    response_html,
    response_show,
    response_trace,
)
from .tables import console_is_interactive, render_table
from .theme import EFFGEN_THEME, color_enabled, get_console, rich_available

__all__ = [
    "EFFGEN_THEME",
    "color_enabled",
    "console_is_interactive",
    "get_console",
    "render_table",
    "rich_available",
    "answer_renderable",
    "format_cost",
    "generation_result_html",
    "generation_result_markdown",
    "response_html",
    "response_show",
    "response_trace",
]
