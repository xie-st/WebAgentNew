"""Bound real-browser prompt views without changing recorded observations.

EffGen protects the initial question (which includes the initial page) and its
last two tool cycles. Reserve space for those three views, token calibration,
the task and tool instructions by giving each view one eighth of the input limit.
The native final-prompt check and history compaction remain authoritative.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .budget import render_observation
from .errors import ContextOverflow
from .protocol import Observation


@dataclass(frozen=True)
class ObservationView:
    text: str
    original_tokens: int
    tokens: int
    max_tokens: int
    shown_elements: int
    total_elements: int


def render_browser_view(
    observation: Observation, *, max_tokens: int, count_tokens: Callable[[str], int]
) -> ObservationView:
    """Keep whole control rows, prioritizing inputs over the link feed.

    Only the model view is projected. Original AX, localization and evidence
    stay in the session. No element is renumbered and no partial URL is emitted.
    Small pages retain their existing rendering byte for byte.
    """
    original = render_observation(observation)
    original_tokens = count_tokens(original)
    total = len(observation.elements)
    if original_tokens <= max_tokens:
        return ObservationView(original, original_tokens, original_tokens, max_tokens, total, total)

    header = [
        "CURRENT OBSERVATION",
        f"page_id={observation.page_id}",
        f"observation_id={observation.observation_id}",
        f"url={observation.url}",
    ]
    if observation.feedback is not None:
        header.append(
            "PREVIOUS FEEDBACK: "
            f"{observation.feedback.status.value}: {observation.feedback.message}"
        )

    # Real extractors already serialize value/state/href in these rows. Reuse
    # each row once rather than repeating it in a second ELEMENTS table.
    prefix, separator, table = observation.ax_text.partition("INTERACTIVE ELEMENTS (")
    rows: dict[str, str] = {}
    if separator:
        for line in table.splitlines():
            match = re.match(r"\[(\d+)\] ", line)
            if match and match[1] in observation.elements:
                rows[match[1]] = line
    for number, element in observation.elements.items():
        if number in rows:
            continue
        bits = [f"[{number}] role={element.role}", f"name={json.dumps(element.name, ensure_ascii=False)}"]
        for key in ("value", "state", "href"):
            value = element.localization.get(key)
            if value:
                bits.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
        rows[number] = " ".join(bits)

    selected: dict[str, str] = {}
    text_lines: list[str] = []

    def assemble() -> str:
        marker = (
            f"PARTIAL OBSERVATION: {len(selected)}/{total} elements shown; "
            "page text may be omitted. Full observation is saved in evidence. "
            "Use shown original element numbers; scroll or navigate for more."
        )
        return "\n".join([
            *header, marker, "AX TEXT:", *text_lines,
            "INTERACTIVE ELEMENTS:",
            *(selected[n] for n in sorted(selected, key=int)),
        ])

    base_tokens = count_tokens(assemble())
    if base_tokens > max_tokens:
        raise ContextOverflow(
            "observation identity and action feedback exceed the view budget",
            required_tokens=base_tokens, available_tokens=max_tokens,
            token_count_source="observation_view_counter",
        )

    # Give one quarter of the available view to title/headings/page prose. Do
    # not cut through a line, a control, a field value, or a link destination.
    text_allowance = (max_tokens - base_tokens) // 4
    used_text = 0
    for line in dict.fromkeys(prefix.splitlines()):
        cost = count_tokens(line + "\n")
        if used_text + cost <= text_allowance:
            text_lines.append(line)
            used_text += cost

    def priority(number: str) -> tuple[int, int, int]:
        element = observation.elements[number]
        role = element.role
        rank = 0 if role in {"textbox", "combobox", "searchbox"} else 1
        if role == "link":
            rank = 2 if observation.elements[number].name else 3
        return int(element.localization.get("in_viewport") is False), rank, int(number)

    used = count_tokens(assemble())
    for number in sorted(rows, key=priority):
        row = rows[number]
        cost = count_tokens(row + "\n")
        href_suffix = re.search(r' href="(?:[^"\\]|\\.)*"$', row)
        if used + cost > max_tokens and href_suffix is not None:
            # A huge href should not hide an otherwise useful control. Omit
            # its destination as a whole; clicking still uses the real index.
            row = row[:href_suffix.start()] + " (href omitted; use element number)"
            cost = count_tokens(row + "\n")
        if used + cost <= max_tokens:
            selected[number] = row
            used += cost

    text = assemble()
    # Token boundaries and the count in the marker can differ from the sum of
    # line counts. Check the assembled view, not just the packing estimate.
    while count_tokens(text) > max_tokens:
        if text_lines:
            text_lines.pop()
        elif selected:
            selected.pop(next(reversed(selected)))
        else:  # pragma: no cover - base view was checked above
            raise ContextOverflow(
                "observation view cannot fit the input budget",
                required_tokens=count_tokens(text), available_tokens=max_tokens,
                token_count_source="observation_view_counter",
            )
        text = assemble()
    return ObservationView(text, original_tokens, count_tokens(text), max_tokens, len(selected), total)
