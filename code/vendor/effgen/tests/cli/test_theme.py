"""Terminal theme selection, the shared palette, and the NO_COLOR guarantee.

Covers the named CLI themes (``default``/``high-contrast``/``monochrome``/
``light``), selection via ``EFFGEN_THEME`` and the ``--theme`` flag, the promise
that ``NO_COLOR`` turns color off whatever the theme, and the single status→role
map shared with the workflow diagram.
"""

from __future__ import annotations

import io

import pytest

from effgen.ui import palette
from effgen.ui.theme import get_console, resolve_theme_name


def _render(markup: str, *, theme_name=None, width=60) -> str:
    """Render markup on a forced-terminal console and return the raw ANSI."""
    buf = io.StringIO()
    console = get_console(theme_name=theme_name, file=buf, force_terminal=True, width=width)
    assert console is not None  # rich is present in the test env
    console.print(markup)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The default theme is unchanged
# ---------------------------------------------------------------------------


class TestDefaultUnchanged:
    def test_default_palette_matches_historical(self):
        expected = {
            "effgen.success": "bold green",
            "effgen.error": "bold red",
            "effgen.warning": "yellow",
            "effgen.info": "cyan",
            "effgen.accent": "magenta",
            "effgen.muted": "dim",
            "effgen.heading": "bold cyan",
            "effgen.title": "bold magenta",
            "effgen.label": "bold",
            "effgen.value": "default",
            "effgen.cost": "green",
            "effgen.tool": "magenta",
            "effgen.metric": "cyan",
            "effgen.model": "cyan",
            "effgen.prompt": "bold cyan",
            "table.header": "bold cyan",
            "repr.number": "cyan",
        }
        assert palette.cli_palette("default") == expected

    def test_no_theme_env_is_default(self, monkeypatch):
        monkeypatch.delenv("EFFGEN_THEME", raising=False)
        assert resolve_theme_name() == "default"

    def test_default_output_has_green_success(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        out = _render("[effgen.success]OK[/effgen.success]", theme_name="default")
        assert "\x1b[1;32m" in out  # bold green


# ---------------------------------------------------------------------------
# Theme selection
# ---------------------------------------------------------------------------


class TestSelection:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("high-contrast", "high-contrast"),
            ("hc", "high-contrast"),
            ("HIGH_CONTRAST", "high-contrast"),
            ("mono", "monochrome"),
            ("MONOCHROME", "monochrome"),
            ("light", "light"),
            ("dark", "default"),
        ],
    )
    def test_aliases_resolve(self, name, expected):
        assert resolve_theme_name(name) == expected

    def test_env_selects_theme(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_THEME", "monochrome")
        assert resolve_theme_name() == "monochrome"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_THEME", "monochrome")
        assert resolve_theme_name("high-contrast") == "high-contrast"

    def test_unknown_theme_falls_back_and_warns_once(self, monkeypatch, capsys):
        monkeypatch.delenv("EFFGEN_THEME", raising=False)
        # Reset the one-time-warning guard so this test observes the warning.
        from effgen.ui import theme as theme_mod

        theme_mod._warned_unknown.discard("neon")
        assert resolve_theme_name("neon") == "default"
        first = capsys.readouterr().err
        assert "Unknown theme 'neon'" in first
        assert "default, high-contrast, light, monochrome" in first
        # A second lookup does not repeat the warning.
        assert resolve_theme_name("neon") == "default"
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# High-contrast and monochrome behavior
# ---------------------------------------------------------------------------


class TestNamedThemes:
    def test_high_contrast_uses_bright_and_no_dim(self):
        pal = palette.cli_palette("high-contrast")
        assert "dim" not in pal["effgen.muted"]  # low-vision readers keep muted text
        out = _render("[effgen.success]OK[/effgen.success]", theme_name="high-contrast")
        assert "\x1b[1;92m" in out  # bold bright green

    def test_high_contrast_lifts_hardcoded_color(self):
        out = _render("[green]x[/green]", theme_name="high-contrast")
        assert "\x1b[92m" in out  # plain [green] markup becomes bright green

    def test_monochrome_drops_hue_keeps_structure(self):
        out = _render("[effgen.success]OK[/effgen.success]", theme_name="monochrome")
        # Bold structure survives; no color SGR code (3x/9x) present.
        assert "\x1b[1m" in out
        assert "\x1b[1;3" not in out and "\x1b[3" not in out.replace("\x1b[39m", "")

    def test_monochrome_neutralizes_hardcoded_color(self):
        out = _render("[green]x[/green]", theme_name="monochrome")
        assert "\x1b[32m" not in out  # no green


# ---------------------------------------------------------------------------
# NO_COLOR wins over every theme
# ---------------------------------------------------------------------------


class TestNoColorWins:
    @pytest.mark.parametrize("theme_name", ["default", "high-contrast", "monochrome", "light"])
    def test_no_color_strips_color(self, monkeypatch, theme_name):
        monkeypatch.setenv("NO_COLOR", "1")
        out = _render("[effgen.accent]run[/effgen.accent] [green]g[/green]", theme_name=theme_name)
        # No color SGR sequences at all (30-37, 90-97, 38/48 truecolor).
        for code in ("\x1b[3", "\x1b[9", "\x1b[38", "\x1b[48"):
            assert code not in out
        assert "run" in out and "g" in out  # text still rendered


# ---------------------------------------------------------------------------
# Shared status → role map
# ---------------------------------------------------------------------------


class TestStatusMap:
    def test_status_style_and_glyph(self):
        assert palette.status_style("completed") == "effgen.success"
        assert palette.status_style("failed") == "effgen.error"
        assert palette.status_style("running") == "effgen.accent"
        assert palette.status_style("skipped") == "effgen.warning"
        assert palette.status_glyph("completed") == "●"
        assert palette.status_glyph("failed") == "✗"

    def test_unknown_status_is_muted(self):
        assert palette.status_style("whoknows") == "effgen.muted"
        assert palette.status_glyph("whoknows") == "○"

    def test_workflow_diagram_uses_shared_map(self):
        from effgen.ui.workflow_viz import workflow_diagram_lines

        lines = workflow_diagram_lines(
            "w", ["a"], [], [["a"]],
            node_results=[{"id": "a", "status": "completed"}],
        )
        styles = {style for style, _ in lines}
        assert "effgen.success" in styles
