"""Verification records for phase 2, batch E1.

A verification request names one goal, the object being checked, the current
normal page observation, the evidence already in scope and the trigger that
asked for the check. A result keeps the three-valued outcome (``pass`` /
``fail`` / ``unknown``) plus the per-condition outcomes, cited evidence, method
and model-call count.

The types deliberately contain no browser, database or hidden-truth handle.
Semantic checks are performed by :mod:`webagent.verification.semantic` through
the same model entry the task already uses; deterministic checks read only the
observation and the run's existing evidence files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..protocol import Observation
from ..state.records import EvidenceRef


class VerificationStatus(str, Enum):
    """Three-valued result; ``unknown`` is not a weak ``fail``."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class CheckKind(str, Enum):
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class VerificationTrigger(str, Enum):
    """The only occasions verification is requested by the phase-2 wiring."""

    ACTION_FEEDBACK = "action_feedback"
    SUBGOAL_END = "subgoal_end"
    TASK_END = "task_end"
    UNKNOWN_EFFECT = "unknown_effect"
    COMPLETION_CHECK = "completion_check"


DETERMINISTIC_CHECKS = frozenset(
    {
        "visible_value_equals",
        "visible_value_contains",
        "page_text_contains",
        "page_url_contains",
        "screenshot_saved",
        "evidence_saved",
        "fact_value_equals",
        "goal_objects_complete",
    }
)
SEMANTIC_CHECKS = frozenset({"semantic"})
SUPPORTED_CHECKS = DETERMINISTIC_CHECKS | SEMANTIC_CHECKS

_UNSAFE_ID = frozenset("\r\n\t")


class VerificationError(RuntimeError):
    """A verification request or result violates the contract."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerificationError(f"{name} must be a non-empty string")
    return value


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if any(char in text for char in _UNSAFE_ID):
        raise VerificationError(f"{name} must not contain control line breaks")
    return text


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise VerificationError(f"{name} must be a list of strings")
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _token(item, name)
        if text in seen:
            raise VerificationError(f"{name} repeats {text!r}")
        seen.add(text)
        items.append(text)
    return tuple(items)


@dataclass(frozen=True)
class VerificationCondition:
    """One independently checkable condition of a goal.

    ``check`` must be one of :data:`SUPPORTED_CHECKS`. Deterministic checks use
    ``role``/``name``/``expected`` and ``params``; a semantic condition carries
    the plain-language ``expected`` judgement and may cite evidence ids.
    """

    condition_id: str
    description: str
    check: str
    expected: Any = None
    role: str | None = None
    name: str | None = None
    evidence_ids: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_id", _token(self.condition_id, "condition.id"))
        object.__setattr__(self, "description", _text(self.description, "condition.description"))
        check = _text(self.check, "condition.check")
        if check not in SUPPORTED_CHECKS:
            raise VerificationError(
                f"condition {self.condition_id!r} uses unsupported check {check!r}; "
                f"allowed: {', '.join(sorted(SUPPORTED_CHECKS))}"
            )
        object.__setattr__(self, "check", check)
        object.__setattr__(self, "evidence_ids", _text_tuple(self.evidence_ids, "condition.evidence_ids"))
        if self.role is not None:
            object.__setattr__(self, "role", _text(self.role, "condition.role"))
        if self.name is not None:
            object.__setattr__(self, "name", _text(self.name, "condition.name"))
        if not isinstance(self.params, Mapping):
            raise VerificationError("condition.params must be a mapping")
        object.__setattr__(
            self,
            "params",
            MappingProxyType({str(key): value for key, value in self.params.items()}),
        )
        if check in ("visible_value_equals", "visible_value_contains"):
            if self.role is None or self.name is None:
                raise VerificationError(
                    f"{check} needs role and name of the visible field"
                )

    @property
    def kind(self) -> CheckKind:
        return CheckKind.SEMANTIC if self.check in SEMANTIC_CHECKS else CheckKind.DETERMINISTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "description": self.description,
            "check": self.check,
            "kind": self.kind.value,
            "expected": self.expected,
            "role": self.role,
            "name": self.name,
            "evidence_ids": list(self.evidence_ids),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class VerificationRequest:
    """Everything a verifier is allowed to use for one goal check."""

    goal_id: str
    subject_id: str
    conditions: tuple[VerificationCondition, ...]
    observation: Observation
    evidence_scope: tuple[EvidenceRef, ...] = ()
    trigger: VerificationTrigger = VerificationTrigger.TASK_END
    branch: str = "root"
    completed_object_ids: tuple[str, ...] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _token(self.goal_id, "request.goal_id"))
        object.__setattr__(self, "subject_id", _token(self.subject_id, "request.subject_id"))
        if not isinstance(self.observation, Observation):
            raise VerificationError("request.observation must be a normal page Observation")
        conditions = tuple(self.conditions)
        if not conditions:
            raise VerificationError("a verification request needs at least one condition")
        seen: set[str] = set()
        for condition in conditions:
            if not isinstance(condition, VerificationCondition):
                raise VerificationError("request.conditions must be VerificationCondition records")
            if condition.condition_id in seen:
                raise VerificationError("request repeats a condition id")
            seen.add(condition.condition_id)
        object.__setattr__(self, "conditions", conditions)
        refs = tuple(self.evidence_scope)
        seen_refs: set[str] = set()
        for ref in refs:
            if not isinstance(ref, EvidenceRef):
                raise VerificationError("request.evidence_scope must contain EvidenceRef records")
            if ref.evidence_id in seen_refs:
                raise VerificationError("request.evidence_scope repeats an evidence id")
            seen_refs.add(ref.evidence_id)
        object.__setattr__(self, "evidence_scope", refs)
        object.__setattr__(self, "trigger", VerificationTrigger(self.trigger))
        object.__setattr__(self, "branch", _token(self.branch, "request.branch"))
        if not isinstance(self.note, str):
            raise VerificationError("request.note must be a string")
        if self.completed_object_ids is not None:
            object.__setattr__(
                self,
                "completed_object_ids",
                _text_tuple(self.completed_object_ids, "request.completed_object_ids"),
            )

    def evidence_index(self) -> dict[str, EvidenceRef]:
        return {ref.evidence_id: ref for ref in self.evidence_scope}

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "subject_id": self.subject_id,
            "trigger": self.trigger.value,
            "branch": self.branch,
            "observation_id": self.observation.observation_id,
            "url": self.observation.url,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "evidence_ids": [ref.evidence_id for ref in self.evidence_scope],
            "note": self.note,
        }


@dataclass(frozen=True)
class ConditionResult:
    """The result of one condition, retained even when the goal fails."""

    condition_id: str
    status: VerificationStatus
    detail: str
    evidence: tuple[EvidenceRef, ...] = ()
    method: str = "deterministic"
    model_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "status": self.status.value,
            "detail": self.detail,
            "evidence_ids": [ref.evidence_id for ref in self.evidence],
            "method": self.method,
            "model_calls": self.model_calls,
        }


@dataclass(frozen=True)
class VerificationResult:
    """The goal-level result returned by :class:`GoalVerifier`."""

    goal_id: str
    subject_id: str
    status: VerificationStatus
    conditions: tuple[ConditionResult, ...]
    evidence: tuple[EvidenceRef, ...]
    method: str
    model_calls: int
    reason: str
    verified: bool
    trigger: VerificationTrigger
    state_revision_before: int | None = None
    state_revision_after: int | None = None

    @property
    def passed(self) -> bool:
        return self.status is VerificationStatus.PASS

    def failed_condition_ids(self) -> tuple[str, ...]:
        return tuple(
            item.condition_id
            for item in self.conditions
            if item.status is not VerificationStatus.PASS
        )

    def unknown_condition_ids(self) -> tuple[str, ...]:
        return tuple(
            item.condition_id
            for item in self.conditions
            if item.status is VerificationStatus.UNKNOWN
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "subject_id": self.subject_id,
            "status": self.status.value,
            "verified": self.verified,
            "conditions": [item.to_dict() for item in self.conditions],
            "evidence_ids": [ref.evidence_id for ref in self.evidence],
            "method": self.method,
            "model_calls": self.model_calls,
            "reason": self.reason,
            "trigger": self.trigger.value,
            "state_revision_before": self.state_revision_before,
            "state_revision_after": self.state_revision_after,
        }


@dataclass(frozen=True)
class CompletionDecision:
    """The answer to "may this task finish as complete?".

    ``accepted`` is true only when every requested goal is verified. A rejected
    decision always names the remaining goals and conditions so the same EffGen
    loop can continue instead of relabelling partial progress as complete.
    """

    accepted: bool
    remaining_goal_ids: tuple[str, ...]
    remaining_conditions: tuple[str, ...]
    reason: str
    results: tuple[VerificationResult, ...] = ()
    state_completion_status: str = "incomplete"
    objects_remaining: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "remaining_goal_ids": list(self.remaining_goal_ids),
            "remaining_conditions": list(self.remaining_conditions),
            "reason": self.reason,
            "results": [result.to_dict() for result in self.results],
            "state_completion_status": self.state_completion_status,
            "objects_remaining": list(self.objects_remaining),
        }


__all__ = [
    "CheckKind",
    "CompletionDecision",
    "ConditionResult",
    "DETERMINISTIC_CHECKS",
    "SEMANTIC_CHECKS",
    "SUPPORTED_CHECKS",
    "VerificationCondition",
    "VerificationError",
    "VerificationRequest",
    "VerificationResult",
    "VerificationStatus",
    "VerificationTrigger",
]
