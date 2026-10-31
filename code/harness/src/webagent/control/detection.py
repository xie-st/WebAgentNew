"""Visible-state takeover detection.

Only what the model can already see is inspected: the current observation's AX
text, its headings/visible text blocks and its visible interactive elements.
The detector never reads the local site's SQLite state, the fault plan, the
judge configuration or any private API. A test can therefore prove that the
trigger really is a rendered overlay by constructing an ``Observation`` whose
visible text contains the overlay and changing nothing else.

The rule is intentionally conservative:

* a login/verification heading is enough; or
* the overlay copy plus its visible action button must both be present.

An ordinary page that merely links to ``Sign in`` does not trigger a handoff.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..protocol import Observation

_HEADING_LINE = re.compile(r"^\s*-\s*(?P<tag>h[1-6]):\s*(?P<text>.+?)\s*$", re.IGNORECASE)
_JSON_ELEMENT = re.compile(
    r'^\s*\[(?P<number>\d+)\]\s+role=(?P<role>\S+)\s+name=(?P<name>"(?:[^"\\]|\\.)*")'
)
_FAKE_ELEMENT = re.compile(r"^\s*\[(?P<number>\d+)\]\s+(?P<role>\S+)\s+(?P<name>.+?)\s*$")
_METADATA_PREFIXES = (
    "title:",
    "url:",
    "viewport:",
    "headings:",
    "text:",
    "interactive elements",
    "elements:",
    "scroll_position:",
)

_LOGIN_HEADINGS = (
    "sign in required",
    "sign-in required",
    "signin required",
    "login required",
    "log in required",
    "authentication required",
    "session expired",
)
_LOGIN_COPY = (
    "session has expired",
    "sign in in this browser",
    "sign in to restore",
    "sign in to continue",
    "log in in this browser",
    "log in to restore",
    "please sign in",
    "please log in",
)
_LOGIN_CONTROLS = ("sign in", "sign-in", "signin", "log in", "login")

_VERIFICATION_HEADINGS = (
    "verification required",
    "human verification",
    "verify you are human",
    "security check",
    "captcha",
)
_VERIFICATION_COPY = (
    "human verification step",
    "complete verification in this browser",
    "verification step before continuing",
    "complete the verification",
    "verify that you are human",
    "checking your browser",
)
_VERIFICATION_CONTROLS = (
    "complete verification",
    "verify now",
    "i'm not a robot",
    "i am not a robot",
    "verify",
)

#: The official accessibility tree prints visible copy as numbered ``StaticText``
#: lines instead of ``TEXT:`` blocks, and section titles as heading-role elements.
#: Those lines are visible page text and must be read as such; a page action
#: (button/link/textbox) stays a control.
_TEXT_ROLES = frozenset({"statictext", "text", "paragraph", "caption"})
_HEADING_ROLE = re.compile(r"^h[1-6]$|^heading$", re.IGNORECASE)


@dataclass(frozen=True)
class TakeoverSignal:
    """The visible overlay that made the task pause."""

    kind: str
    reason: str
    headings: tuple[str, ...]
    matched_copy: tuple[str, ...]
    matched_controls: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "headings": list(self.headings),
            "matched_copy": list(self.matched_copy),
            "matched_controls": list(self.matched_controls),
            "detection_source": "visible_observation_only",
        }


def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _element_name(raw: str) -> str:
    text = raw.strip().rstrip(",")
    if text.startswith('"') and text.endswith('"'):
        try:
            value = json.loads(text)
            return str(value)
        except json.JSONDecodeError:
            return text.strip('"')
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _visible_parts(observation: Observation) -> tuple[list[str], list[str], list[str]]:
    headings: list[str] = []
    text_lines: list[str] = []
    controls: list[str] = []
    for line in observation.ax_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = _HEADING_LINE.match(stripped)
        if heading is not None:
            headings.append(_normalize(heading.group("text")))
            continue
        element = _JSON_ELEMENT.match(stripped)
        if element is not None:
            _classify_element(
                element.group("role"),
                _normalize(_element_name(element.group("name"))),
                headings,
                text_lines,
                controls,
            )
            continue
        fake_element = _FAKE_ELEMENT.match(stripped)
        if fake_element is not None:
            _classify_element(
                fake_element.group("role"),
                _normalize(_element_name(fake_element.group("name"))),
                headings,
                text_lines,
                controls,
            )
            continue
        lowered = stripped.casefold()
        if any(lowered.startswith(prefix) for prefix in _METADATA_PREFIXES):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        text_lines.append(_normalize(stripped))
    for element in observation.elements.values():
        name = _normalize(element.name)
        if not name:
            continue
        # Visible copy is not an action control: a page that merely mentions
        # "sign in" in body text must not trigger a handoff.
        if str(getattr(element, "role", "") or "").strip().casefold() in _TEXT_ROLES:
            continue
        controls.append(name)
    return headings, text_lines, controls


def _classify_element(
    role: str,
    name: str,
    headings: list[str],
    text_lines: list[str],
    controls: list[str],
) -> None:
    """One numbered AX line: visible copy, a section heading or a control."""

    if not name:
        return
    normalized_role = str(role or "").strip().casefold()
    if _HEADING_ROLE.match(str(role or "").strip()):
        headings.append(name)
    elif normalized_role in _TEXT_ROLES:
        text_lines.append(name)
    else:
        controls.append(name)


def _first_matches(values: Iterable[str], needles: Iterable[str]) -> tuple[str, ...]:
    found: list[str] = []
    for value in values:
        for needle in needles:
            if needle in value:
                found.append(value)
                break
    return tuple(dict.fromkeys(found))


def _has_control(controls: Iterable[str], needles: Iterable[str]) -> tuple[str, ...]:
    return _first_matches(controls, needles)


def detect_takeover(observation: Observation | None) -> TakeoverSignal | None:
    """Return the visible login/verification overlay, if one is rendered."""

    if not isinstance(observation, Observation):
        return None
    headings, text_lines, controls = _visible_parts(observation)
    visible_text = (*headings, *text_lines)

    login_heading = _first_matches(headings, _LOGIN_HEADINGS)
    login_copy = _first_matches(visible_text, _LOGIN_COPY)
    login_controls = _has_control(controls, _LOGIN_CONTROLS)
    if login_heading or (login_copy and login_controls):
        return TakeoverSignal(
            kind="login",
            reason="login_overlay_visible",
            headings=tuple(headings),
            matched_copy=login_copy,
            matched_controls=login_controls,
        )

    verification_heading = _first_matches(headings, _VERIFICATION_HEADINGS)
    verification_copy = _first_matches(visible_text, _VERIFICATION_COPY)
    verification_controls = _has_control(controls, _VERIFICATION_CONTROLS)
    if verification_heading or (verification_copy and verification_controls):
        return TakeoverSignal(
            kind="verification",
            reason="verification_overlay_visible",
            headings=tuple(headings),
            matched_copy=verification_copy,
            matched_controls=verification_controls,
        )
    return None


def takeover_reason(signal: TakeoverSignal | None) -> str | None:
    return None if signal is None else signal.reason


__all__ = ["TakeoverSignal", "detect_takeover", "takeover_reason"]
