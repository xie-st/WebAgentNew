"""Recovery records for phase 2, batch E1.

Recovery is a program-side correction of the ledger and plan, not a browser
undo. It classifies one observed failure with evidence, finds the earliest
supported conflict, computes the reverse dependency closure, and invalidates only
the affected completion claims. Unrelated verified progress is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..protocol import Observation
from ..state.records import EvidenceRef, PendingEffect, TaskState


class RecoveryClass(str, Enum):
    """Failure vocabulary from the implementation note, including unknown."""

    PROTOCOL = "protocol"
    TRANSIENT = "transient"
    UNKNOWN_EFFECT = "unknown_effect"
    CONDITION_FAILED = "condition_failed"
    NO_PROGRESS = "no_progress"
    AUTHENTICATION = "authentication"
    USER_PAUSE = "user_pause"
    UNKNOWN = "unknown"


class EffectOutcome(str, Enum):
    CONFIRMED = "confirmed"
    NOT_APPLIED = "not_applied"
    UNKNOWN = "unknown"


_UNSAFE_ID = frozenset("\r\n\t")


class RecoveryError(RuntimeError):
    """A recovery request violates the ledger or evidence contract."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryError(f"{name} must be a non-empty string")
    return value


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if any(char in text for char in _UNSAFE_ID):
        raise RecoveryError(f"{name} must not contain control line breaks")
    return text


def _optional_token(value: Any, name: str) -> str | None:
    return None if value is None else _token(value, name)


def _refs(value: Any, name: str) -> tuple[EvidenceRef, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RecoveryError(f"{name} must be a list of EvidenceRef records")
    refs: list[EvidenceRef] = []
    seen: set[str] = set()
    for ref in value:
        if not isinstance(ref, EvidenceRef):
            raise RecoveryError(f"{name} must contain EvidenceRef records")
        if ref.evidence_id in seen:
            raise RecoveryError(f"{name} repeats evidence id {ref.evidence_id!r}")
        seen.add(ref.evidence_id)
        refs.append(ref)
    return tuple(refs)


@dataclass(frozen=True)
class FailureEvidence:
    """One observed failure, with the evidence that supports its classification."""

    action_id: str
    sequence: int
    failure_class: RecoveryClass
    goal_id: str | None = None
    message: str = ""
    observation_id: str | None = None
    object_id: str | None = None
    fact_id: str | None = None
    page_url: str | None = None
    exception_type: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _token(self.action_id, "failure.action_id"))
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise RecoveryError("failure.sequence must be an integer >= 0")
        object.__setattr__(
            self, "failure_class", RecoveryClass(self.failure_class)
        )
        object.__setattr__(self, "goal_id", _optional_token(self.goal_id, "failure.goal_id"))
        object.__setattr__(self, "observation_id", _optional_token(self.observation_id, "failure.observation_id"))
        object.__setattr__(self, "object_id", _optional_token(self.object_id, "failure.object_id"))
        object.__setattr__(self, "fact_id", _optional_token(self.fact_id, "failure.fact_id"))
        object.__setattr__(self, "page_url", _optional_token(self.page_url, "failure.page_url"))
        object.__setattr__(self, "exception_type", _optional_token(self.exception_type, "failure.exception_type"))
        object.__setattr__(self, "evidence", _refs(self.evidence, "failure.evidence"))
        if not isinstance(self.message, str):
            raise RecoveryError("failure.message must be a string")
        if not isinstance(self.metadata, Mapping):
            raise RecoveryError("failure.metadata must be a mapping")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType({str(key): value for key, value in self.metadata.items()}),
        )

    @property
    def has_anchor_evidence(self) -> bool:
        return bool(self.observation_id or self.fact_id or self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "failure_class": self.failure_class.value,
            "goal_id": self.goal_id,
            "message": self.message,
            "observation_id": self.observation_id,
            "object_id": self.object_id,
            "fact_id": self.fact_id,
            "page_url": self.page_url,
            "exception_type": self.exception_type,
            "evidence_ids": [ref.evidence_id for ref in self.evidence],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecoveryAnchor:
    """The earliest action/fact with enough evidence to start recovery."""

    anchor_type: str
    anchor_id: str
    goal_id: str | None
    reason: str
    sequence: int
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.anchor_type not in ("action", "fact"):
            raise RecoveryError("anchor_type must be 'action' or 'fact'")
        object.__setattr__(self, "anchor_id", _token(self.anchor_id, "anchor.id"))
        object.__setattr__(self, "goal_id", _optional_token(self.goal_id, "anchor.goal_id"))
        object.__setattr__(self, "evidence", _refs(self.evidence, "anchor.evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_type": self.anchor_type,
            "anchor_id": self.anchor_id,
            "goal_id": self.goal_id,
            "reason": self.reason,
            "sequence": self.sequence,
            "evidence_ids": [ref.evidence_id for ref in self.evidence],
        }


@dataclass(frozen=True)
class RecoveryPlan:
    """A correction limited to the anchor's reverse dependency closure."""

    anchor: RecoveryAnchor | None
    affected_goal_ids: tuple[str, ...]
    invalidated_goal_ids: tuple[str, ...]
    preserved_verified_goal_ids: tuple[str, ...]
    next_step: str
    reason: str
    evidence_sufficient: bool

    def __post_init__(self) -> None:
        for name in (
            "affected_goal_ids",
            "invalidated_goal_ids",
            "preserved_verified_goal_ids",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise RecoveryError(f"{name} must be a tuple")
            for value in values:
                _token(value, name)
            if len(set(values)) != len(values):
                raise RecoveryError(f"{name} repeats an id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": None if self.anchor is None else self.anchor.to_dict(),
            "affected_goal_ids": list(self.affected_goal_ids),
            "invalidated_goal_ids": list(self.invalidated_goal_ids),
            "preserved_verified_goal_ids": list(self.preserved_verified_goal_ids),
            "next_step": self.next_step,
            "reason": self.reason,
            "evidence_sufficient": self.evidence_sufficient,
        }


@dataclass(frozen=True)
class RecoveryAttempt:
    """Whether a bounded recovery attempt is allowed now."""

    allowed: bool
    reason: str
    goal_attempts: Mapping[str, int] = field(default_factory=dict)
    total_attempts: int = 0
    remaining_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.goal_attempts, Mapping):
            raise RecoveryError("attempt.goal_attempts must be a mapping")
        object.__setattr__(
            self,
            "goal_attempts",
            MappingProxyType({str(key): int(value) for key, value in self.goal_attempts.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "goal_attempts": dict(self.goal_attempts),
            "total_attempts": self.total_attempts,
            "remaining_seconds": self.remaining_seconds,
        }


@dataclass(frozen=True)
class RecoveryResult:
    """The committed effect of a recovery plan, if it was applied."""

    applied: bool
    plan: RecoveryPlan
    attempt: RecoveryAttempt
    state_revision_before: int
    state_revision_after: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "plan": self.plan.to_dict(),
            "attempt": self.attempt.to_dict(),
            "state_revision_before": self.state_revision_before,
            "state_revision_after": self.state_revision_after,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EffectExpectation:
    """What a fresh normal observation should show if a write took effect."""

    field_role: str
    field_name: str
    expected_value: str
    absence_is_decisive: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_role", _text(self.field_role, "expectation.field_role"))
        object.__setattr__(self, "field_name", _text(self.field_name, "expectation.field_name"))
        if not isinstance(self.expected_value, str):
            raise RecoveryError("expectation.expected_value must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_role": self.field_role,
            "field_name": self.field_name,
            "expected_value": self.expected_value,
            "absence_is_decisive": self.absence_is_decisive,
        }


@dataclass(frozen=True)
class EffectResolution:
    """Outcome of probing an unknown side effect before any retry decision."""

    outcome: EffectOutcome
    retry_allowed: bool
    reason: str
    observation_id: str | None = None
    visible_value: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", EffectOutcome(self.outcome))
        object.__setattr__(self, "evidence", _refs(self.evidence, "resolution.evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "retry_allowed": self.retry_allowed,
            "reason": self.reason,
            "observation_id": self.observation_id,
            "visible_value": self.visible_value,
            "evidence_ids": [ref.evidence_id for ref in self.evidence],
        }


@dataclass(frozen=True)
class RestartResult:
    """A restart loaded a snapshot, registered a new branch and re-observed.

    There is deliberately no generator field: a Python generator stack cannot be
    restored. Pending effects remain pending and are never re-sent here.
    """

    root_run_id: str
    source_branch: str
    new_branch_id: str
    new_run_id: str
    state: TaskState
    observation: Observation
    pending_effects: tuple[PendingEffect, ...]
    generator_stack_restored: bool = False
    resent_action_ids: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "source_branch": self.source_branch,
            "new_branch_id": self.new_branch_id,
            "new_run_id": self.new_run_id,
            "state": self.state.view(),
            "observation_id": self.observation.observation_id,
            "observation_url": self.observation.url,
            "pending_effects": [effect.to_dict() for effect in self.pending_effects],
            "generator_stack_restored": self.generator_stack_restored,
            "resent_action_ids": list(self.resent_action_ids),
            "reason": self.reason,
        }


__all__ = [
    "EffectExpectation",
    "EffectOutcome",
    "EffectResolution",
    "FailureEvidence",
    "RecoveryAnchor",
    "RecoveryAttempt",
    "RecoveryClass",
    "RecoveryError",
    "RecoveryPlan",
    "RecoveryResult",
    "RestartResult",
]
