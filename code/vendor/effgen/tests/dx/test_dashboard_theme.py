"""Dashboard theming — palette lockstep, label contrast, and first-paint scheme.

Confirms the dashboard stylesheet stays in lockstep with the shared palette
source of truth, that every waterfall bar label clears WCAG AA on every bar
color in both themes, that first paint follows the OS color scheme with no
JavaScript, and that the declared font stack is a real system stack (no
aspirational webfont that never loads).
"""

from __future__ import annotations

import re
from pathlib import Path

from effgen.ui.palette import DASHBOARD_DARK, DASHBOARD_LIGHT

STATIC_DIR = Path(__file__).parents[2] / "effgen" / "dashboard" / "static"


def _css() -> str:
    return (STATIC_DIR / "style.css").read_text()


def _vars_in_block(css: str, selector: str) -> dict[str, str]:
    """Extract ``--name: value;`` pairs from the first block matching *selector*."""
    idx = css.index(selector)
    start = css.index("{", idx) + 1
    end = css.index("}", start)
    body = css[start:end]
    out: dict[str, str] = {}
    for m in re.finditer(r"--([\w-]+)\s*:\s*([^;]+);", body):
        out[m.group(1)] = m.group(2).strip()
    return out


# ---------------------------------------------------------------------------
# Palette lockstep — style.css matches the Python source of truth
# ---------------------------------------------------------------------------


class TestPaletteLockstep:
    def test_dark_root_matches_palette(self):
        css_vars = _vars_in_block(_css(), ":root {")
        for key, value in DASHBOARD_DARK.items():
            assert css_vars.get(key) == value, f"dark --{key}: {css_vars.get(key)} != {value}"

    def test_light_toggle_block_matches_palette(self):
        css_vars = _vars_in_block(_css(), ':root[data-theme="light"]')
        for key, value in DASHBOARD_LIGHT.items():
            assert css_vars.get(key) == value, f"light --{key}: {css_vars.get(key)} != {value}"

    def test_light_media_block_matches_palette(self):
        # The no-JS first-paint light block must carry the same values.
        css_vars = _vars_in_block(_css(), ':root:not([data-theme="dark"])')
        for key, value in DASHBOARD_LIGHT.items():
            assert css_vars.get(key) == value, f"media --{key}: {css_vars.get(key)} != {value}"


# ---------------------------------------------------------------------------
# Waterfall label contrast (WCAG AA on every bar, both themes)
# ---------------------------------------------------------------------------


def _lin(c: int) -> float:
    x = c / 255
    return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


class TestWaterfallLabelContrast:
    # The bar fill colors the label sits on (wf-model/tool/run/router/is-error).
    BAR_KEYS = ["accent", "accent2", "text-faint", "warn", "err"]

    def test_dark_labels_meet_aa(self):
        ink = DASHBOARD_DARK["wf-label"]
        for key in self.BAR_KEYS:
            ratio = _contrast(ink, DASHBOARD_DARK[key])
            assert ratio >= 4.5, f"dark label on --{key}: {ratio:.1f}"

    def test_light_labels_meet_aa(self):
        ink = DASHBOARD_LIGHT["wf-label"]
        for key in self.BAR_KEYS:
            ratio = _contrast(ink, DASHBOARD_LIGHT[key])
            assert ratio >= 4.5, f"light label on --{key}: {ratio:.1f}"

    def test_label_uses_theme_variable_not_hardcoded_white(self):
        css = _css()
        block = css[css.index(".wf-bar-label"):css.index(".wf-bar-label") + 200]
        assert "var(--wf-label)" in block
        assert "color: #fff" not in block

    def test_tool_bar_does_not_use_success_green(self):
        # Green (--ok) stays reserved for success meaning; the tool bar is cyan.
        css = _css()
        block = css[css.index(".wf-bar.wf-tool"):css.index(".wf-bar.wf-tool") + 60]
        assert "var(--accent2)" in block
        assert "var(--ok)" not in block


# ---------------------------------------------------------------------------
# First paint follows the OS scheme with no JavaScript
# ---------------------------------------------------------------------------


class TestFirstPaintScheme:
    def test_html_does_not_hardcode_theme(self):
        html = (STATIC_DIR / "index.html").read_text()
        m = re.search(r"<html[^>]*>", html)
        assert m is not None
        assert "data-theme" not in m.group(0)

    def test_js_overrides_only_on_stored_choice(self):
        js = (STATIC_DIR / "app.js").read_text()
        # Explicit-choice branch sets the attribute; the no-choice branch does not.
        assert 'stored === "dark" || stored === "light"' in js
        assert "syncThemeButton" in js


# ---------------------------------------------------------------------------
# Font declaration matches reality
# ---------------------------------------------------------------------------


class TestFontStack:
    def test_font_leads_with_system_stack(self):
        css_vars = _vars_in_block(_css(), ":root {")
        assert css_vars["font"].startswith("system-ui")
        assert css_vars["mono"].startswith("ui-monospace")

    def test_no_unbundled_lead_webfont(self):
        # No @font-face is shipped, so the stack must not *lead* with a webfont
        # that can never load; a trailing named font (only if installed) is fine.
        css = _css()
        assert "@font-face" not in css
        css_vars = _vars_in_block(css, ":root {")
        assert not css_vars["font"].startswith("'Inter'")
