"""Official accessibility-tree (AX) adaptation for deterministic checks.

The official platform does not send the local extractor's ``localization.value``.
Its element names keep the accessibility attribute suffix, for example::

    'Biography' required: False describedby: user_biography_biography_help

and the field's current value is the quoted child ``StaticText`` printed on a
more deeply indented line under that element::

    [3647] textbox 'Biography' required: False describedby: ...
        [3451] StaticText 't2_5adwlxvnI am a robot'

This module parses only that saved AX text. It never re-observes a page, reads a
site database or creates a browser. Matching uses the quoted field name, so the
attribute suffix does not have to appear in a verification condition, and value
lookup scans only descendants of the matched element -- never a same-named
``StaticText`` label that is a sibling of the input.
"""

from __future__ import annotations

import re
from typing import Any

_LEADING_QUOTED = re.compile(r"""^\s*(['"])(.*?)\1""")
_AX_LINE = re.compile(r"^[ \t]*\[([0-9]{1,12})\](?:[ \t]|$)(.*)$")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def extract_quoted_name(raw: Any) -> str:
    """Return the quoted name at the start of an AX name, or the raw name.

    The official tree stores names as ``'<visible name>' <attributes>``. Local
    extractors store the visible name already separated (usually without the
    outer quotes). Both forms reduce to the visible name here.
    """

    text = "" if raw is None else str(raw).strip()
    match = _LEADING_QUOTED.match(text)
    if match is not None:
        return match.group(2).strip()
    return text


def name_matches(element_name: Any, wanted_name: Any) -> bool:
    """Compare two accessible names after removing official attribute suffixes."""

    return (
        _norm(extract_quoted_name(element_name)).casefold()
        == _norm(extract_quoted_name(wanted_name)).casefold()
    )


def find_element_line(ax_text: str, number: str) -> str:
    """Return the numbered AX line for ``number``, including leading indentation.

    Official lines are nested with tabs, so the existing phase-1 check that only
    accepted a line starting at column zero missed every real official element.
    The number must be the line's own bracket token, not a prefix of a longer id.
    """

    if not isinstance(ax_text, str):
        return ""
    wanted = str(number)
    if not wanted.isdigit():
        return ""
    pattern = re.compile(rf"^[ \t]*\[{re.escape(wanted)}\](?:[ \t]|$)")
    for line in ax_text.splitlines():
        if pattern.match(line):
            return line
    return ""


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def parse_ax_line(line: str) -> tuple[str, str, str] | None:
    """Split one numbered AX line into ``(number, role, name_with_attributes)``."""

    match = _AX_LINE.match(line)
    if match is None:
        return None
    number = match.group(1)
    rest = match.group(2).strip()
    if not rest:
        return number, "generic", ""
    if rest[0] in "'\"":
        return number, "generic", rest
    parts = rest.split(None, 1)
    role = parts[0].lower()
    name = parts[1].strip() if len(parts) > 1 else ""
    return number, role, name


def child_static_text_value(ax_text: str, number: str) -> str | None:
    """Return the first descendant ``StaticText`` value of one AX element.

    The scan starts on the line after the element and stops at the first line
    whose indentation is not greater than the element's. A same-named label that
    precedes the field, or is a sibling at the same depth, is therefore not a
    value source. ``None`` means the element has no value-bearing child; an
    empty string is a real observed empty value.
    """

    if not isinstance(ax_text, str):
        return None
    lines = ax_text.splitlines()
    wanted = str(number)
    index = -1
    for position, line in enumerate(lines):
        parsed = parse_ax_line(line)
        if parsed is not None and parsed[0] == wanted:
            index = position
            break
    if index < 0:
        return None
    base_indent = _indent_width(lines[index])
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        if _indent_width(line) <= base_indent:
            break
        parsed = parse_ax_line(line)
        if parsed is None:
            continue
        _child_number, role, name = parsed
        if role != "statictext":
            continue
        return extract_quoted_name(name)
    return None


__all__ = [
    "child_static_text_value",
    "extract_quoted_name",
    "find_element_line",
    "name_matches",
    "parse_ax_line",
]
