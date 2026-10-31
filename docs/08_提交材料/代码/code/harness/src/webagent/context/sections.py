"""Deterministic necessary-context regions for the G-wave final-send check.

Four region classes are protected on every model send:

* the task text the platform supplied (``session.task``),
* the current accessibility-tree observation as the harness renders it,
* the harness-maintained ledger short view,
* the unconfirmed side-effect records.

Everything in this module is plain text rendering over already-recorded state:
there is no model call, no clock reading, no random id and no page access, so the
same state always produces the same bytes. The ledger and effect blocks are the
only content the harness adds to the prompt; their mandatory part is never
shortened, and only the optional ledger detail is dropped when its stated size
limit is reached (that drop is counted, never silent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from ..budget import render_observation
from ..protocol import Observation
from ..state.records import EffectStatus
from ..state.store import StateStore

LEDGER_HEADER = "LEDGER VIEW (harness-maintained; change it only through state tools)"
EFFECTS_HEADER = "PENDING SIDE EFFECTS (unconfirmed; never treat them as done)"
NO_LEDGER = "no task ledger is active for this task"
NO_EFFECTS = "none recorded"

#: Long requirement text is compressed to one bounded line in the short view.
#: The compression is a view decision made before the send; the bytes that the
#: guard protects are the view bytes below, not the untrimmed requirement.
REQUIREMENT_SNIPPET_CHARS = 120
FACT_SNIPPET_CHARS = 80
NOTE_SNIPPET_CHARS = 80

#: An unconfirmed side effect whose result is not yet known. ``failed`` and
#: ``confirmed`` are recorded outcomes and are not listed here.
UNCONFIRMED_EFFECT_STATUSES = frozenset({EffectStatus.PENDING, EffectStatus.UNKNOWN})


class SectionKind(str, Enum):
    """The four region classes the final send must keep byte-complete."""

    TASK = "task"
    OBSERVATION = "observation"
    LEDGER = "ledger"
    EFFECTS = "effects"


#: Canonical order for reports and events; never reordered by the caller.
SECTION_ORDER: tuple[SectionKind, ...] = (
    SectionKind.TASK,
    SectionKind.OBSERVATION,
    SectionKind.LEDGER,
    SectionKind.EFFECTS,
)

SECTION_LABELS: dict[SectionKind, str] = {
    SectionKind.TASK: "task text",
    SectionKind.OBSERVATION: "current AX observation",
    SectionKind.LEDGER: "state ledger short view",
    SectionKind.EFFECTS: "unconfirmed side effects",
}


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _clip(text: Any, limit: int) -> str:
    """One bounded line; deterministic and never empty for non-empty input."""

    value = _one_line(text)
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _quote(text: Any, limit: int) -> str:
    return json.dumps(_clip(text, limit), ensure_ascii=False)


def byte_length(text: str) -> int:
    """UTF-8 bytes of one rendered region; the unit the guard reports."""

    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class ContextBlock:
    """One harness block: a mandatory part plus optionally shortened detail.

    ``required`` is the whole protected region: the guard requires those exact
    bytes in the final send. ``optional`` is only ledger detail; when it does not
    fit ``max_chars`` it is dropped line by line and ``trimmed_optional_chars``
    records how much was left out. ``over_required`` marks a block whose
    mandatory part alone already exceeds its stated size; it is still sent as
    built and the final budget check reports the overflow explicitly.
    """

    kind: SectionKind
    required: str
    optional: str = ""
    trimmed_optional_chars: int = 0
    over_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SectionKind):
            raise TypeError("context block kind must be a SectionKind")
        if not isinstance(self.required, str) or not self.required.strip():
            raise ValueError("context block required text must be a non-empty string")
        if not isinstance(self.optional, str):
            raise TypeError("context block optional text must be a string")
        if self.trimmed_optional_chars < 0:
            raise ValueError("trimmed_optional_chars must be >= 0")

    @property
    def text(self) -> str:
        """The block exactly as it is appended to the prompt."""

        if not self.optional:
            return self.required
        return self.required + "\n" + self.optional

    @property
    def byte_length(self) -> int:
        return byte_length(self.text)

    @property
    def optional_chars(self) -> int:
        return len(self.optional)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "required_bytes": byte_length(self.required),
            "optional_bytes": byte_length(self.optional),
            "trimmed_optional_chars": self.trimmed_optional_chars,
            "over_required": self.over_required,
        }


@dataclass(frozen=True)
class NecessarySection:
    """One protected region as it must appear in the final send."""

    kind: SectionKind
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SectionKind):
            raise TypeError("necessary section kind must be a SectionKind")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("necessary section text must be a non-empty string")

    @property
    def label(self) -> str:
        return SECTION_LABELS[self.kind]

    @property
    def byte_length(self) -> int:
        return byte_length(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "bytes": self.byte_length,
        }


def _append_optional(
    required: str,
    candidates: Sequence[str],
    *,
    max_chars: int,
) -> tuple[str, int]:
    """Append optional lines while the block stays within ``max_chars``.

    A line is either kept whole or dropped whole, so the kept part is a stable
    prefix of the candidate list and the same state always keeps the same lines.
    """

    budget = max_chars - len(required) - 1
    kept: list[str] = []
    used = 0
    dropped = 0
    for line in candidates:
        cost = len(line) + (1 if kept else 0)
        if used + cost > budget:
            dropped += len(line) + 1
            continue
        kept.append(line)
        used += cost
    return "\n".join(kept), dropped


def ledger_block(
    store: StateStore | None,
    *,
    max_chars: int = 2_000,
    branch: str | None = None,
) -> ContextBlock:
    """The ledger short view: stats, goals, plan nodes and an optional fact list."""

    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    lines = [LEDGER_HEADER]
    if store is None:
        lines.append(NO_LEDGER)
        return ContextBlock(SectionKind.LEDGER, "\n".join(lines))

    state = store.state
    lines.append(
        "revision="
        f"{state.revision} plan_version={state.plan_version} "
        f"completion={state.completion_status()}"
    )
    if branch is not None:
        lines.append(f"branch={branch}")
    goals = sorted(state.goals.values(), key=lambda goal: goal.goal_id)
    lines.append(f"goals={len(goals)}")
    for goal in goals:
        lines.append(
            f"- {goal.goal_id} [{goal.status.value}]"
            f" criteria={len(goal.acceptance_criteria)}"
            f" depends={list(goal.depends_on)}"
            f" requirement={_quote(goal.requirement, REQUIREMENT_SNIPPET_CHARS)}"
        )
    plan = sorted(state.plan.values(), key=lambda node: node.node_id)
    lines.append(f"plan_nodes={len(plan)}")
    for node in plan:
        lines.append(f"- {node.node_id} -> {node.goal_id} [{node.status.value}]")
    verified = list(state.verified_goal_ids)
    if verified:
        lines.append(f"verified_goal_ids={verified}")
    facts = sorted(state.facts.values(), key=lambda fact: fact.fact_id)
    lines.append(f"facts={len(facts)}")

    required = "\n".join(lines)
    if len(required) > max_chars:
        return ContextBlock(SectionKind.LEDGER, required, over_required=True)
    candidates = [
        f"- fact {fact.fact_id} {fact.object_id}.{fact.field}"
        f"={_quote(fact.value, FACT_SNIPPET_CHARS)} [{fact.kind.value}]"
        for fact in facts
    ]
    optional, dropped = _append_optional(required, candidates, max_chars=max_chars)
    return ContextBlock(
        SectionKind.LEDGER,
        required,
        optional=optional,
        trimmed_optional_chars=dropped,
    )


def effects_block(
    store: StateStore | None,
    *,
    max_chars: int = 1_200,
) -> ContextBlock:
    """Every unconfirmed side-effect record, as one mandatory block."""

    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    lines = [EFFECTS_HEADER]
    if store is None:
        lines.append(NO_EFFECTS)
        return ContextBlock(SectionKind.EFFECTS, "\n".join(lines))

    effects = [
        effect
        for effect in sorted(
            store.state.pending_effects.values(), key=lambda item: item.action_id
        )
        if effect.status in UNCONFIRMED_EFFECT_STATUSES
    ]
    if not effects:
        lines.append(NO_EFFECTS)
    for effect in effects:
        lines.append(
            f"- {effect.action_id} target={effect.target}"
            f" object={effect.object_id} status={effect.status.value}"
            f" goal={effect.goal_id or '-'}"
            f" observation={effect.observation_id or '-'}"
            f" created_revision={effect.created_revision}"
            f" note={_quote(effect.note, NOTE_SNIPPET_CHARS)}"
        )
    required = "\n".join(lines)
    return ContextBlock(
        SectionKind.EFFECTS,
        required,
        over_required=len(required) > max_chars,
    )


def build_necessary_sections(
    *,
    task: str,
    observation: Observation,
    ledger: ContextBlock,
    effects: ContextBlock,
) -> tuple[NecessarySection, ...]:
    """The four protected regions in canonical order, byte-for-byte as rendered."""

    if not isinstance(task, str) or not task.strip():
        raise ValueError("the task text must be a non-empty string")
    if not isinstance(observation, Observation):
        raise TypeError("the current observation must be an Observation")
    if ledger.kind is not SectionKind.LEDGER or effects.kind is not SectionKind.EFFECTS:
        raise ValueError("ledger and effects blocks must carry their own kinds")
    return (
        NecessarySection(SectionKind.TASK, task),
        NecessarySection(SectionKind.OBSERVATION, render_observation(observation)),
        NecessarySection(SectionKind.LEDGER, ledger.text),
        NecessarySection(SectionKind.EFFECTS, effects.text),
    )


def total_bytes(sections: Sequence[NecessarySection]) -> int:
    return sum(section.byte_length for section in sections)


__all__ = [
    "EFFECTS_HEADER",
    "LEDGER_HEADER",
    "NO_EFFECTS",
    "NO_LEDGER",
    "SECTION_LABELS",
    "SECTION_ORDER",
    "ContextBlock",
    "NecessarySection",
    "SectionKind",
    "UNCONFIRMED_EFFECT_STATUSES",
    "build_necessary_sections",
    "byte_length",
    "effects_block",
    "ledger_block",
    "total_bytes",
]
