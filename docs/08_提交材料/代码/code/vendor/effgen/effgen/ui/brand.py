"""Visual identity assets for the effGen terminal experience.

One module owns the product's terminal identity so nothing else hardcodes art:

- the product glyph (a lightning bolt, matching the dashboard wordmark),
- the one-line wordmark used in headers,
- the block-letter logo banner (Unicode half-blocks, with a pure-ASCII fallback),
- the indigo→cyan gradient applied across the banner.

Everything routes through :mod:`effgen.ui.palette` (colors, glyph fallback) and
:mod:`effgen.ui.theme` (the console factory, ``NO_COLOR`` handling). The banner is
a string constant — there is no figlet/font dependency. The module imports only
the standard library, ``rich`` (a base dependency), and the two ``ui`` helpers,
so it loads without pulling in the tool registry, model catalog, or any adapter.

Color is applied only when the console reports it and ``NO_COLOR`` is unset; the
``monochrome`` theme renders the banner bold with no hue. When ``rich`` is absent
the renderers degrade to the plain one-line wordmark.
"""

from __future__ import annotations

from typing import Any

from . import palette
from .palette import supports_unicode  # re-exported: one implementation
from .theme import color_enabled, resolve_theme_name, rich_available

__all__ = [
    "BOLT",
    "BOLT_ASCII",
    "LOGO_BANNER",
    "LOGO_BANNER_ASCII",
    "BANNER_WIDTH",
    "supports_unicode",
    "wordmark",
    "logo_renderable",
    "print_logo",
]

# The product glyph — a lightning bolt, the same mark the dashboard wordmark uses.
BOLT = palette.GLYPHS["bolt"]
BOLT_ASCII = palette.GLYPHS_ASCII["bolt"]

# The block-letter banner spelling "effGen" in Unicode half-blocks. Five rows,
# well under the 64-column budget. Rows are padded to a common width at import so
# the gradient can address every column and the render is a clean rectangle.
_LOGO_ROWS = (
    "        ██  ██   ▄████▄",
    " ▄███▄ ▀██▀▀██▀ ██▀  ▀▀  ▄███▄  ██▄███▄",
    " ██▄██  ██  ██  ██  ███  ██▄██  ██▀ ▀██",
    " ██▄▄▄  ██  ██  ▀██▄▄██  ██▄▄▄  ██   ██",
    "  ▀▀▀▀  ▀▀  ▀▀    ▀▀▀▀    ▀▀▀▀  ▀▀   ▀▀ ",
)


def _pad(rows: tuple[str, ...]) -> tuple[str, ...]:
    width = max(len(r) for r in rows)
    return tuple(r.ljust(width) for r in rows)


LOGO_BANNER = "\n".join(_pad(_LOGO_ROWS))

# Pure-ASCII banner for streams that cannot encode the half-block glyphs. The
# block cells become '#'; the layout (and width) is preserved.
_ASCII_MAP = str.maketrans({"█": "#", "▀": "#", "▄": "#", "▐": "#", "▌": "#"})
LOGO_BANNER_ASCII = "\n".join(r.translate(_ASCII_MAP) for r in _pad(_LOGO_ROWS))

BANNER_WIDTH = max(len(r) for r in _pad(_LOGO_ROWS))


def wordmark(version: str | None = None, *, stream: Any = None) -> str:
    """The one-line product mark, e.g. ``⚡ effGen`` or ``⚡ effGen v1.0.0``.

    Plain text, safe to print on any stream: the bolt degrades to an ASCII
    stand-in when *stream* cannot encode it.
    """
    bolt = BOLT if supports_unicode(stream) else BOLT_ASCII
    mark = f"{bolt} effGen".strip()
    if version:
        mark = f"{mark} v{version}"
    return mark


def _lerp_hex(a: str, b: str, t: float) -> str:
    """Linear interpolation between two ``#rrggbb`` colors at fraction *t*."""
    ai = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    bi = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(ai[k] + (bi[k] - ai[k]) * t) for k in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _ramp_color(stops: tuple[str, ...], t: float) -> str:
    """The color at fraction *t* (0..1) along a multi-stop hex ramp."""
    if len(stops) == 1:
        return stops[0]
    scaled = t * (len(stops) - 1)
    lo = int(scaled)
    if lo >= len(stops) - 1:
        return stops[-1]
    return _lerp_hex(stops[lo], stops[lo + 1], scaled - lo)


def _column_colors(console: Any, width: int) -> tuple[list[str] | None, str | None]:
    """Per-column styles for the banner, plus a whole-banner fallback style.

    Returns ``(colors, fallback)``:

    - ``colors`` is a list of length *width* of per-column style strings when the
      terminal can show the gradient; ``None`` otherwise.
    - ``fallback`` is a single style applied to the whole banner when there is no
      per-column gradient (``"bold"`` for the ``monochrome`` theme, else ``None``
      for plain, escape-free text under ``NO_COLOR`` / no color support).
    """
    if not color_enabled() or getattr(console, "no_color", False):
        return None, None
    theme = resolve_theme_name()
    if theme == "monochrome":
        return None, "bold"
    system = getattr(console, "color_system", None)
    if system is None:
        return None, None
    if width <= 0:
        return None, None
    if system in ("truecolor", "256"):
        stops = palette.brand_gradient_stops(theme)
        if not stops:
            return None, "bold"
        denom = max(1, width - 1)
        return [_ramp_color(stops, c / denom) for c in range(width)], None
    # 16-color terminal: a two-tone split across the named ANSI stand-ins.
    ramp = palette.brand_named_ramp(theme)
    if not ramp:
        return None, "bold"
    lo, hi = ramp[0], ramp[-1]
    half = width / 2
    return [lo if c < half else hi for c in range(width)], None


def logo_renderable(*, console: Any, compact: bool = False) -> Any:
    """Build the logo for *console*: the gradient banner, or the one-line wordmark.

    The one-line wordmark is used when *compact* is set or the console is
    narrower than the banner. The ASCII banner is used when the console stream
    cannot encode the half-block art. Without ``rich`` the plain wordmark string
    is returned so callers can ``print`` it.
    """
    if not rich_available() or console is None:
        return wordmark()

    from rich.text import Text

    stream = getattr(console, "file", None)
    unicode_ok = supports_unicode(stream)
    width = getattr(console, "width", 80) or 80

    if compact or width < BANNER_WIDTH:
        mark = wordmark(stream=stream)
        colors, fallback = _column_colors(console, len(mark))
        text = Text()
        for i, ch in enumerate(mark):
            style = None
            if colors is not None and ch != " ":
                style = colors[i]
            elif fallback is not None:
                style = fallback
            text.append(ch, style=style)
        return text

    banner = LOGO_BANNER if unicode_ok else LOGO_BANNER_ASCII
    lines = banner.split("\n")
    colors, fallback = _column_colors(console, BANNER_WIDTH)
    text = Text()
    for row, line in enumerate(lines):
        if row:
            text.append("\n")
        for col, ch in enumerate(line):
            if ch == " ":
                text.append(ch)
                continue
            if colors is not None:
                text.append(ch, style=colors[col])
            elif fallback is not None:
                text.append(ch, style=fallback)
            else:
                text.append(ch)
    return text


def print_logo(console: Any = None, *, compact: bool = False) -> None:
    """Render the logo to *console* (or a freshly built one)."""
    if console is None:
        from .theme import get_console

        console = get_console()
    if console is None:  # rich unavailable
        print(wordmark())
        return
    console.print(logo_renderable(console=console, compact=compact))
