"""Escaping, value formatting, and document normalizing for HTML reports.

Every value a report shows passes through here first. Text from a model, a
tool, a provider error, or a user's own task reaches the page escaped, and a
metric the document does not carry renders as an em dash rather than a zero, so
an absent measurement is never read as a measured one.

The normalizers (:func:`_mapping`, :func:`_number`, :func:`_sequence`) let a
report builder read a document that another build wrote or a person edited by
hand: a field of the wrong type renders as empty rather than raising.
"""

from __future__ import annotations

import html
import math
from typing import Any

_DASH = "—"


class ReportError(ValueError):
    """Raised when a result document cannot be rendered as a report."""


# ---------------------------------------------------------------------------
# Formatting helpers. A metric that is absent renders as an em dash; it is
# never substituted with a zero.
# ---------------------------------------------------------------------------

def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value * 100:.{digits}f}%"


def _secs(value: Any, digits: int = 3) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value:.{digits}f}s"


def _ms(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value * 1000:.1f} ms"


def _usd(value: Any, digits: int = 6) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"${value:.{digits}f}"


def _rps(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{value:,.2f} req/s"


def _int(value: Any) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return _DASH
    return f"{int(value):,}"


def _truncate(text: str, limit: int = 240) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fraction(value: Any, maximum: float) -> float:
    """Return *value* as a 0–1 fraction of *maximum* for bar widths."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return 0.0
    if maximum <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / maximum))


def _mapping(value: Any) -> dict[str, Any]:
    """*value* when it is a mapping, an empty mapping otherwise."""
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float:
    """*value* as a float, treating anything that is not a number as zero.

    A cost or a count read out of a saved document may be text; a total built
    from it is a sum of the values that are numbers, not a failed render.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value) if math.isfinite(value) else 0.0


def _sequence(value: Any) -> list[Any]:
    """*value* as a list of items, treating a non-sequence as empty.

    A bare string is not a one-item sequence here: iterating it would render
    one entry per character.
    """
    return list(value) if isinstance(value, list | tuple) else []
