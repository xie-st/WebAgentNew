"""Task-ledger record types and integrity checks (phase 2, batch D).

One :class:`TaskState` is the single project-side business-completion ledger for
a root run. It is deliberately not a second conversation transcript: it keeps
requirements, facts, the plan and unconfirmed side effects by id, and cites
evidence stored by :mod:`webagent.evidence` rather than copying artifacts.

This module has no I/O. :mod:`webagent.state.store` owns persistence and the
versioned commit path; :mod:`webagent.state.ledger` owns mutation rules.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EvidenceError, WebAgentError
from ..protocol import ARTIFACT_KINDS, validate_run_id

TASK_STATE_SCHEMA_VERSION = "webagent.task_state.v1"

_SHA256_LENGTH = 64
_ID_UNSAFE = frozenset("\r\n\t")


class StateError(WebAgentError):
    """Base class for task-ledger errors."""


class StateValidationError(StateError):
    """A record or plan violates the ledger contract."""


class EvidenceReferenceError(StateValidationError):
    """A cited evidence reference is missing, altered or owned elsewhere."""


class PlanRejected(StateValidationError):
    """A proposed plan revision is not a valid, loss-free revision."""


class StateConflict(StateError):
    """An update was based on an older revision or plan version.

    ``current`` is the committed state the caller must re-read before retrying.
    Returning it is what lets a branch re-check instead of overwriting values it
    has not observed.
    """

    def __init__(
        self,
        message: str,
        *,
        current: "TaskState",
        reason: str,
        base_revision: int | None = None,
        plan_version: int | None = None,
    ) -> None:
        super().__init__(message)
        self.current = current
        self.reason = reason
        self.base_revision = base_revision
        self.plan_version = plan_version

    def current_view(self) -> dict[str, Any]:
        return self.current.view()

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "reason": self.reason,
            "base_revision": self.base_revision,
            "plan_version": self.plan_version,
            "current": self.current.view(),
        }


class GoalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_VERIFICATION = "awaiting_verification"
    VERIFIED = "verified"
    BLOCKED = "blocked"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"


#: Plan nodes use the same vocabulary as goals; a node cannot be more complete
#: than the goal it serves.
PlanNodeStatus = GoalStatus


class FactKind(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"


class FactValidity(str, Enum):
    VALID = "valid"
    INVALIDATED = "invalidated"


class EffectStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: Any, name: str) -> str:
    """Human-readable text; line breaks are allowed (tasks and criteria)."""

    if not isinstance(value, str):
        raise StateValidationError(f"{name} must be a string")
    if not value.strip():
        raise StateValidationError(f"{name} must not be empty")
    return value


def _token(value: Any, name: str) -> str:
    """An id-like token; it must not smuggle control line breaks into JSON logs."""

    text = _text(value, name)
    if any(char in text for char in _ID_UNSAFE):
        raise StateValidationError(f"{name} must not contain control line breaks")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _optional_token(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _token(value, name)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise StateValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _tuple_of_ids(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise StateValidationError(f"{name} must be a list of ids")
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _token(item, f"{name} item")
        if text in seen:
            raise StateValidationError(f"{name} contains duplicate id {text!r}")
        seen.add(text)
        items.append(text)
    return tuple(items)


def _tuple_of_text(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise StateValidationError(f"{name} must be a list of strings")
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _token(item, f"{name} item")
        if text in seen:
            raise StateValidationError(f"{name} contains duplicate item {text!r}")
        seen.add(text)
        items.append(text)
    return tuple(items)


def _string_map(value: Any, name: str) -> Mapping[str, tuple[str, ...]]:
    """Resource declaration: kind -> ordered values.

    A string value is accepted as a one-item list so a tool payload can say
    ``{"tab": "tab-1"}`` without inventing a second wire shape.
    """

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise StateValidationError(f"{name} must be an object of resource -> values")
    normalized: dict[str, tuple[str, ...]] = {}
    for key, raw in value.items():
        resource = _token(key, f"{name} resource")
        values = _tuple_of_ids(raw, f"{name}.{resource}") if not isinstance(raw, str) else (
            _token(raw, f"{name}.{resource}"),
        )
        normalized[resource] = values
    return MappingProxyType(normalized)


def _plain_json(value: Any, name: str) -> Any:
    """Validate a fact value as JSON data and return its plain form."""

    def check(item: Any, where: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise StateValidationError(f"{where} must be a finite number")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                check(child, f"{where}[{index}]")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise StateValidationError(f"{where} keys must be strings")
                check(child, f"{where}.{key}")
            return
        raise StateValidationError(f"{where} is not JSON data")

    check(value, name)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _coerce_enum(enum_cls: type[Enum], value: Any, name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(str(item.value) for item in enum_cls)
        raise StateValidationError(f"{name} must be one of: {allowed}") from exc


def _check_reference(reference: str) -> str:
    if not isinstance(reference, str) or not reference.strip():
        raise EvidenceReferenceError("evidence reference must be a non-empty relative path")
    from pathlib import PurePosixPath

    if "\\" in reference:
        raise EvidenceReferenceError("evidence reference must use '/' separators")
    path = PurePosixPath(reference)
    if path.is_absolute() or any(part in ("..", "") for part in path.parts):
        raise EvidenceReferenceError("evidence reference must stay inside its run directory")
    return reference


@dataclass(frozen=True)
class EvidenceRef:
    """One existing evidence artifact cited by the ledger.

    The ledger stores only identity, ownership and file metadata; the bytes stay
    in the run's evidence directory.
    """

    evidence_id: str
    run_id: str
    reference: str
    sha256: str
    byte_size: int
    branch_id: str = "root"
    kind: str | None = None
    observation_id: str | None = None
    action_id: str | None = None

    def __post_init__(self) -> None:
        _token(self.evidence_id, "evidence_id")
        validate_run_id(self.run_id)
        _token(self.branch_id, "evidence.branch_id")
        object.__setattr__(self, "reference", _check_reference(self.reference))
        if not isinstance(self.sha256, str) or len(self.sha256) != _SHA256_LENGTH:
            raise EvidenceReferenceError("evidence.sha256 must be a 64-character hex digest")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise EvidenceReferenceError("evidence.sha256 must be hexadecimal") from exc
        if self.sha256.lower() != self.sha256:
            raise EvidenceReferenceError("evidence.sha256 must be lowercase")
        _integer(self.byte_size, "evidence.byte_size")
        if self.kind is not None and self.kind not in ARTIFACT_KINDS:
            raise EvidenceReferenceError(
                f"evidence.kind must be one of {sorted(ARTIFACT_KINDS)}"
            )
        _optional_token(self.observation_id, "evidence.observation_id")
        _optional_token(self.action_id, "evidence.action_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "reference": self.reference,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "kind": self.kind,
            "observation_id": self.observation_id,
            "action_id": self.action_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRef":
        if not isinstance(data, Mapping):
            raise EvidenceReferenceError("evidence reference must be an object")
        try:
            return cls(
                evidence_id=data["evidence_id"],
                run_id=data["run_id"],
                reference=data["reference"],
                sha256=data["sha256"],
                byte_size=data["byte_size"],
                branch_id=data.get("branch_id", "root"),
                kind=data.get("kind"),
                observation_id=data.get("observation_id"),
                action_id=data.get("action_id"),
            )
        except KeyError as exc:
            raise EvidenceReferenceError(f"evidence reference is missing {exc.args[0]}") from exc


def _evidence_tuple(value: Any, name: str) -> tuple[EvidenceRef, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise StateValidationError(f"{name} must be a list of evidence references")
    refs: list[EvidenceRef] = []
    seen: set[str] = set()
    for raw in value:
        ref = raw if isinstance(raw, EvidenceRef) else EvidenceRef.from_dict(raw)
        if ref.evidence_id in seen:
            raise StateValidationError(f"{name} repeats evidence id {ref.evidence_id!r}")
        seen.add(ref.evidence_id)
        refs.append(ref)
    return tuple(refs)


@dataclass(frozen=True)
class Goal:
    """One requirement the user asked for, with its verification state."""

    goal_id: str
    requirement: str
    acceptance_criteria: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    object_scope: tuple[str, ...] = ()
    status: GoalStatus = GoalStatus.PENDING
    evidence: tuple[EvidenceRef, ...] = ()
    branch_id: str = "root"
    completed_object_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _token(self.goal_id, "goal.id"))
        object.__setattr__(self, "requirement", _text(self.requirement, "goal.requirement"))
        criteria = _tuple_of_text(
            self.acceptance_criteria, f"goal {self.goal_id}.acceptance_criteria"
        )
        if not criteria:
            raise StateValidationError(f"goal {self.goal_id!r} needs acceptance criteria")
        object.__setattr__(self, "acceptance_criteria", criteria)
        object.__setattr__(
            self, "depends_on", _tuple_of_ids(self.depends_on, f"goal {self.goal_id}.depends_on")
        )
        object.__setattr__(
            self,
            "object_scope",
            _tuple_of_ids(self.object_scope, f"goal {self.goal_id}.object_scope"),
        )
        object.__setattr__(self, "status", _coerce_enum(GoalStatus, self.status, "goal.status"))
        object.__setattr__(
            self, "evidence", _evidence_tuple(self.evidence, f"goal {self.goal_id}.evidence")
        )
        object.__setattr__(self, "branch_id", _token(self.branch_id, "goal.branch_id"))
        completed = _tuple_of_ids(
            self.completed_object_ids, f"goal {self.goal_id}.completed_object_ids"
        )
        if not set(completed).issubset(self.object_scope):
            raise StateValidationError(
                f"goal {self.goal_id!r} completed objects must be inside its object scope"
            )
        object.__setattr__(self, "completed_object_ids", completed)
        if self.status is GoalStatus.VERIFIED:
            if not self.evidence:
                raise StateValidationError(
                    f"goal {self.goal_id!r} cannot be verified without supporting evidence"
                )
            if self.object_scope and set(completed) != set(self.object_scope):
                raise StateValidationError(
                    f"goal {self.goal_id!r} cannot be verified while objects are unaccounted for"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.goal_id,
            "requirement": self.requirement,
            "acceptance_criteria": list(self.acceptance_criteria),
            "depends_on": list(self.depends_on),
            "object_scope": list(self.object_scope),
            "status": self.status.value,
            "evidence": [ref.to_dict() for ref in self.evidence],
            "branch_id": self.branch_id,
            "completed_object_ids": list(self.completed_object_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Goal":
        if not isinstance(data, Mapping):
            raise StateValidationError("goal must be an object")
        return cls(
            goal_id=data.get("id", data.get("goal_id")),
            requirement=data.get("requirement"),
            acceptance_criteria=tuple(data.get("acceptance_criteria") or ()),
            depends_on=tuple(data.get("depends_on") or data.get("dependent_goals") or ()),
            object_scope=tuple(data.get("object_scope") or ()),
            status=data.get("status", GoalStatus.PENDING),
            evidence=tuple(data.get("evidence") or ()),
            branch_id=data.get("branch_id", "root"),
            completed_object_ids=tuple(data.get("completed_object_ids") or ()),
        )


@dataclass(frozen=True)
class Fact:
    """One object field value, explicitly observed or only inferred."""

    fact_id: str
    object_id: str
    field: str
    value: Any
    kind: FactKind
    source: str
    validity: FactValidity = FactValidity.VALID
    revision: int = 1
    evidence: tuple[EvidenceRef, ...] = ()
    branch_id: str = "root"
    goal_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _token(self.fact_id, "fact.id"))
        object.__setattr__(self, "object_id", _token(self.object_id, "fact.object"))
        object.__setattr__(self, "field", _text(self.field, "fact.field"))
        object.__setattr__(self, "value", _plain_json(self.value, f"fact {self.fact_id}.value"))
        kind = _coerce_enum(FactKind, self.kind, "fact.kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", _text(self.source, "fact.source"))
        object.__setattr__(
            self, "validity", _coerce_enum(FactValidity, self.validity, "fact.validity")
        )
        object.__setattr__(self, "revision", _integer(self.revision, "fact.revision", minimum=1))
        object.__setattr__(
            self, "evidence", _evidence_tuple(self.evidence, f"fact {self.fact_id}.evidence")
        )
        object.__setattr__(self, "branch_id", _token(self.branch_id, "fact.branch_id"))
        object.__setattr__(self, "goal_id", _optional_token(self.goal_id, "fact.goal_id"))
        if kind is FactKind.OBSERVED and not self.evidence:
            raise StateValidationError(
                f"observed fact {self.fact_id!r} needs supporting evidence; "
                "an inference is recorded as inferred"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fact_id,
            "object": self.object_id,
            "field": self.field,
            "value": self.value,
            "kind": self.kind.value,
            "source": self.source,
            "validity": self.validity.value,
            "revision": self.revision,
            "evidence": [ref.to_dict() for ref in self.evidence],
            "branch_id": self.branch_id,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Fact":
        if not isinstance(data, Mapping):
            raise StateValidationError("fact must be an object")
        return cls(
            fact_id=data.get("id", data.get("fact_id")),
            object_id=data.get("object", data.get("object_id")),
            field=data.get("field", data.get("proposition")),
            value=data.get("value"),
            kind=data.get("kind", FactKind.INFERRED),
            source=data.get("source"),
            validity=data.get("validity", FactValidity.VALID),
            revision=data.get("revision", 1),
            evidence=tuple(data.get("evidence") or ()),
            branch_id=data.get("branch_id", "root"),
            goal_id=data.get("goal_id"),
        )


@dataclass(frozen=True)
class PlanNode:
    """One executable step in the plan EffGen proposed."""

    node_id: str
    goal_id: str
    depends_on: tuple[str, ...] = ()
    input_fact_ids: tuple[str, ...] = ()
    object_scope: tuple[str, ...] = ()
    resources: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    status: PlanNodeStatus = PlanNodeStatus.PENDING
    plan_version: int = 0
    branch_id: str = "root"

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _token(self.node_id, "plan_node.id"))
        object.__setattr__(self, "goal_id", _token(self.goal_id, "plan_node.goal_id"))
        object.__setattr__(
            self, "depends_on", _tuple_of_ids(self.depends_on, f"plan_node {self.node_id}.depends_on")
        )
        object.__setattr__(
            self,
            "input_fact_ids",
            _tuple_of_ids(self.input_fact_ids, f"plan_node {self.node_id}.input_fact_ids"),
        )
        object.__setattr__(
            self,
            "object_scope",
            _tuple_of_ids(self.object_scope, f"plan_node {self.node_id}.object_scope"),
        )
        object.__setattr__(self, "resources", _string_map(self.resources, "plan_node.resources"))
        object.__setattr__(
            self, "status", _coerce_enum(GoalStatus, self.status, "plan_node.status")
        )
        object.__setattr__(
            self, "plan_version", _integer(self.plan_version, "plan_node.plan_version")
        )
        object.__setattr__(self, "branch_id", _token(self.branch_id, "plan_node.branch_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "goal_id": self.goal_id,
            "depends_on": list(self.depends_on),
            "input_fact_ids": list(self.input_fact_ids),
            "object_scope": list(self.object_scope),
            "resources": {key: list(values) for key, values in self.resources.items()},
            "status": self.status.value,
            "plan_version": self.plan_version,
            "branch_id": self.branch_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanNode":
        if not isinstance(data, Mapping):
            raise StateValidationError("plan node must be an object")
        return cls(
            node_id=data.get("id", data.get("node_id")),
            goal_id=data.get("goal_id", data.get("goal")),
            depends_on=tuple(data.get("depends_on") or ()),
            input_fact_ids=tuple(data.get("input_fact_ids") or ()),
            object_scope=tuple(data.get("object_scope") or ()),
            resources=data.get("resources") or {},
            status=data.get("status", GoalStatus.PENDING),
            plan_version=data.get("plan_version", 0),
            branch_id=data.get("branch_id", "root"),
        )


@dataclass(frozen=True)
class PendingEffect:
    """A side effect that was started but not yet confirmed.

    The record is written and saved *before* the action is dispatched. A residue
    loaded after a restart stays pending; nothing in this module re-sends it.
    """

    action_id: str
    target: str
    object_id: str
    status: EffectStatus = EffectStatus.PENDING
    goal_id: str | None = None
    branch_id: str = "root"
    observation_id: str | None = None
    created_revision: int = 0
    updated_revision: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _token(self.action_id, "effect.action_id"))
        object.__setattr__(self, "target", _token(self.target, "effect.target"))
        object.__setattr__(self, "object_id", _token(self.object_id, "effect.object"))
        object.__setattr__(
            self, "status", _coerce_enum(EffectStatus, self.status, "effect.status")
        )
        object.__setattr__(self, "goal_id", _optional_token(self.goal_id, "effect.goal_id"))
        object.__setattr__(self, "branch_id", _token(self.branch_id, "effect.branch_id"))
        object.__setattr__(
            self, "observation_id", _optional_token(self.observation_id, "effect.observation_id")
        )
        object.__setattr__(
            self, "created_revision", _integer(self.created_revision, "effect.created_revision")
        )
        if self.updated_revision is not None:
            _integer(self.updated_revision, "effect.updated_revision")
        if not isinstance(self.note, str):
            raise StateValidationError("effect.note must be a string")

    @property
    def is_pending(self) -> bool:
        return self.status is EffectStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "target": self.target,
            "object": self.object_id,
            "status": self.status.value,
            "goal_id": self.goal_id,
            "branch_id": self.branch_id,
            "observation_id": self.observation_id,
            "created_revision": self.created_revision,
            "updated_revision": self.updated_revision,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingEffect":
        if not isinstance(data, Mapping):
            raise StateValidationError("pending effect must be an object")
        return cls(
            action_id=data.get("action_id"),
            target=data.get("target"),
            object_id=data.get("object", data.get("object_id")),
            status=data.get("status", EffectStatus.PENDING),
            goal_id=data.get("goal_id"),
            branch_id=data.get("branch_id", "root"),
            observation_id=data.get("observation_id"),
            created_revision=data.get("created_revision", 0),
            updated_revision=data.get("updated_revision"),
            note=data.get("note", ""),
        )


def _assert_acyclic(edges: Mapping[str, Sequence[str]], label: str) -> None:
    """Depth-first cycle check over an id -> dependency mapping."""

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in done:
            return
        if node in visiting:
            cycle = " -> ".join(path + (node,))
            raise PlanRejected(f"{label} dependency cycle: {cycle}")
        visiting.add(node)
        for dependency in edges.get(node, ()):
            visit(dependency, path + (node,))
        visiting.remove(node)
        done.add(node)

    for node in edges:
        visit(node, ())


@dataclass
class TaskState:
    """The single task ledger for one root run.

    Dictionaries are keyed by each record's id, and dependencies are id lists,
    so no second dependency index exists. Records are frozen; a candidate copy
    is replaced only after its snapshot has been written successfully.
    """

    root_run_id: str
    task: str
    revision: int = 0
    plan_version: int = 0
    goals: dict[str, Goal] = field(default_factory=dict)
    facts: dict[str, Fact] = field(default_factory=dict)
    plan: dict[str, PlanNode] = field(default_factory=dict)
    pending_effects: dict[str, PendingEffect] = field(default_factory=dict)
    branches: dict[str, str] = field(default_factory=dict)
    updated_at: str | None = None
    schema_version: str = TASK_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_run_id(self.root_run_id)
        self.task = _text(self.task, "task")
        self.revision = _integer(self.revision, "revision")
        self.plan_version = _integer(self.plan_version, "plan_version")
        if not self.branches:
            self.branches = {"root": self.root_run_id}
        else:
            normalized: dict[str, str] = {}
            for branch, run_id in self.branches.items():
                normalized[_token(branch, "branch id")] = validate_run_id(run_id)
            normalized.setdefault("root", self.root_run_id)
            if normalized["root"] != self.root_run_id:
                raise StateValidationError("the root branch must map to this task's root run")
            self.branches = normalized
        for mapping, name in (
            (self.goals, "goals"),
            (self.facts, "facts"),
            (self.plan, "plan"),
            (self.pending_effects, "pending_effects"),
        ):
            if not isinstance(mapping, dict):
                raise StateValidationError(f"{name} must be an id-keyed object")
        if self.updated_at is not None and not isinstance(self.updated_at, str):
            raise StateValidationError("updated_at must be a string or null")
        if self.schema_version != TASK_STATE_SCHEMA_VERSION:
            raise StateValidationError(
                f"unsupported task state schema_version {self.schema_version!r}"
            )

    # ------------------------------------------------------------- inspection
    def copy(self) -> "TaskState":
        return TaskState(
            root_run_id=self.root_run_id,
            task=self.task,
            revision=self.revision,
            plan_version=self.plan_version,
            goals=dict(self.goals),
            facts=dict(self.facts),
            plan=dict(self.plan),
            pending_effects=dict(self.pending_effects),
            branches=dict(self.branches),
            updated_at=self.updated_at,
        )

    @property
    def unconfirmed_effects(self) -> tuple[PendingEffect, ...]:
        return tuple(
            effect
            for effect in sorted(self.pending_effects.values(), key=lambda item: item.action_id)
            if effect.status in (EffectStatus.PENDING, EffectStatus.UNKNOWN)
        )

    @property
    def verified_goal_ids(self) -> tuple[str, ...]:
        return tuple(
            goal_id
            for goal_id, goal in sorted(self.goals.items())
            if goal.status is GoalStatus.VERIFIED
        )

    def completion_status(self) -> str:
        """``complete`` only when every requirement is verified.

        A blocked requirement stays visible as blocked; everything else that is
        not verified is incomplete. This is what a plan revision must not hide
        by deleting the unfinished requirement.
        """

        if not self.goals:
            return "incomplete"
        if any(goal.status is GoalStatus.BLOCKED for goal in self.goals.values()):
            return "blocked"
        if all(goal.status is GoalStatus.VERIFIED for goal in self.goals.values()):
            return "complete"
        return "incomplete"

    def object_progress(self, goal_ids: Iterable[str] | None = None) -> dict[str, Any]:
        """Count distinct objects, never page visits or repeated evidence.

        Completion is per object id in ``completed_object_ids``. Across goals the
        union is reported, so the same object reached by two requirements is one
        object in the total.
        """

        selected = (
            set(self.goals)
            if goal_ids is None
            else {goal_id for goal_id in goal_ids}
        )
        missing = selected - set(self.goals)
        if missing:
            raise StateValidationError(
                "unknown goal ids for object progress: " + ", ".join(sorted(missing))
            )
        total: set[str] = set()
        completed: set[str] = set()
        for goal_id in sorted(selected):
            goal = self.goals[goal_id]
            total.update(goal.object_scope)
            completed.update(goal.completed_object_ids)
        completed &= total
        remaining = total - completed
        return {
            "objects_total": len(total),
            "objects_completed": len(completed),
            "objects_remaining": len(remaining),
            "completed_object_ids": sorted(completed),
            "remaining_object_ids": sorted(remaining),
            "goals_total": len(selected),
            "goals_verified": sum(
                1 for goal_id in selected if self.goals[goal_id].status is GoalStatus.VERIFIED
            ),
            "complete": bool(total) and not remaining,
        }

    def evidence_refs(self) -> tuple[tuple[str, EvidenceRef], ...]:
        """Every cited reference with the id of the record citing it."""

        refs: list[tuple[str, EvidenceRef]] = []
        for goal in self.goals.values():
            refs.extend((f"goal:{goal.goal_id}", ref) for ref in goal.evidence)
        for fact in self.facts.values():
            refs.extend((f"fact:{fact.fact_id}", ref) for ref in fact.evidence)
        return tuple(refs)

    def view(self, *, goal_id: str | None = None, include_facts: bool = True) -> dict[str, Any]:
        """A compact JSON view for state tools and conflict responses."""

        goals = self.goals
        facts = self.facts
        if goal_id is not None:
            if goal_id not in self.goals:
                raise StateValidationError(f"unknown goal id {goal_id!r}")
            goals = {goal_id: self.goals[goal_id]}
            facts = {key: fact for key, fact in self.facts.items() if fact.goal_id == goal_id}
        return {
            "root_run_id": self.root_run_id,
            "task": self.task,
            "revision": self.revision,
            "plan_version": self.plan_version,
            "completion_status": self.completion_status(),
            "verified_goal_ids": list(self.verified_goal_ids),
            "branches": dict(self.branches),
            "goals": {key: value.to_dict() for key, value in sorted(goals.items())},
            "facts": (
                {key: value.to_dict() for key, value in sorted(facts.items())}
                if include_facts
                else {}
            ),
            "plan": {key: value.to_dict() for key, value in sorted(self.plan.items())},
            "pending_effects": {
                key: value.to_dict()
                for key, value in sorted(self.pending_effects.items())
            },
        }

    # -------------------------------------------------------------- validation
    def validate(self, evidence: "EvidenceResolver | None" = None) -> None:
        """Check ids, ownership, dependency integrity, DAG acyclicity and evidence."""

        if self.schema_version != TASK_STATE_SCHEMA_VERSION:
            raise StateValidationError("unsupported task state schema version")
        validate_run_id(self.root_run_id)
        _text(self.task, "task")
        _integer(self.revision, "revision")
        _integer(self.plan_version, "plan_version")
        if set(self.branches) and "root" not in self.branches:
            raise StateValidationError("task state has no root branch")
        if self.branches.get("root") != self.root_run_id:
            raise StateValidationError("root branch does not point at the root run")
        root_run = self.branches.get("root")
        for branch, run_id in self.branches.items():
            _token(branch, "branch id")
            validate_run_id(run_id)
        if len(set(self.branches.values())) != len(self.branches):
            # The root run may legitimately host only the root branch; a branch
            # must still not silently alias another branch's evidence directory.
            raise StateValidationError("two branches cannot share one run id")

        for collection, label, id_attribute in (
            (self.goals, "goal", "goal_id"),
            (self.facts, "fact", "fact_id"),
            (self.plan, "plan node", "node_id"),
            (self.pending_effects, "pending effect", "action_id"),
        ):
            for key, record in collection.items():
                expected = getattr(record, id_attribute)
                if expected != key:
                    raise StateValidationError(
                        f"{label} map key {key!r} does not match record id {expected!r}"
                    )

        goal_edges: dict[str, tuple[str, ...]] = {}
        for goal in self.goals.values():
            self._check_branch(goal.branch_id, f"goal {goal.goal_id}")
            for dependency in goal.depends_on:
                if dependency not in self.goals:
                    raise StateValidationError(
                        f"goal {goal.goal_id!r} depends on unknown goal {dependency!r}"
                    )
            goal_edges[goal.goal_id] = goal.depends_on
            self._check_evidence(goal.evidence, goal.branch_id, f"goal {goal.goal_id}", evidence)

        for fact in self.facts.values():
            self._check_branch(fact.branch_id, f"fact {fact.fact_id}")
            if fact.goal_id is not None and fact.goal_id not in self.goals:
                raise StateValidationError(
                    f"fact {fact.fact_id!r} refers to unknown goal {fact.goal_id!r}"
                )
            self._check_evidence(fact.evidence, fact.branch_id, f"fact {fact.fact_id}", evidence)

        plan_edges: dict[str, tuple[str, ...]] = {}
        for node in self.plan.values():
            self._check_branch(node.branch_id, f"plan node {node.node_id}")
            if node.goal_id not in self.goals:
                raise StateValidationError(
                    f"plan node {node.node_id!r} serves unknown goal {node.goal_id!r}"
                )
            for dependency in node.depends_on:
                if dependency not in self.plan:
                    raise StateValidationError(
                        f"plan node {node.node_id!r} depends on unknown node {dependency!r}"
                    )
            for fact_id in node.input_fact_ids:
                if fact_id not in self.facts:
                    raise StateValidationError(
                        f"plan node {node.node_id!r} reads unknown fact {fact_id!r}"
                    )
            if node.plan_version > self.plan_version:
                raise StateValidationError(
                    f"plan node {node.node_id!r} claims a future plan version"
                )
            goal = self.goals[node.goal_id]
            if goal.object_scope and not set(node.object_scope).issubset(goal.object_scope):
                raise StateValidationError(
                    f"plan node {node.node_id!r} touches objects outside goal scope"
                )
            plan_edges[node.node_id] = node.depends_on

        for effect in self.pending_effects.values():
            self._check_branch(effect.branch_id, f"pending effect {effect.action_id}")
            if effect.goal_id is not None and effect.goal_id not in self.goals:
                raise StateValidationError(
                    f"pending effect {effect.action_id!r} refers to unknown goal "
                    f"{effect.goal_id!r}"
                )

        _assert_acyclic(goal_edges, "goal")
        _assert_acyclic(plan_edges, "plan")

    def _check_branch(self, branch_id: str, where: str) -> None:
        if branch_id not in self.branches:
            raise StateValidationError(
                f"{where} belongs to unregistered branch {branch_id!r}"
            )

    def _check_evidence(
        self,
        refs: Sequence[EvidenceRef],
        owner_branch: str,
        where: str,
        resolver: "EvidenceResolver | None",
    ) -> None:
        for ref in refs:
            expected_run = self.branches.get(ref.branch_id)
            if expected_run is None:
                raise EvidenceReferenceError(
                    f"{where} cites evidence from unregistered branch {ref.branch_id!r}"
                )
            if expected_run != ref.run_id:
                raise EvidenceReferenceError(
                    f"{where} cites evidence {ref.evidence_id!r} whose run does not "
                    f"belong to branch {ref.branch_id!r}"
                )
            if owner_branch != "root" and ref.branch_id != owner_branch:
                raise EvidenceReferenceError(
                    f"{where} cites evidence from another branch"
                )
            if resolver is not None:
                resolver.verify(ref)

    # -------------------------------------------------------------- JSON forms
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root_run_id": self.root_run_id,
            "task": self.task,
            "revision": self.revision,
            "plan_version": self.plan_version,
            "branches": dict(self.branches),
            "goals": {key: value.to_dict() for key, value in sorted(self.goals.items())},
            "facts": {key: value.to_dict() for key, value in sorted(self.facts.items())},
            "plan": {key: value.to_dict() for key, value in sorted(self.plan.items())},
            "pending_effects": {
                key: value.to_dict() for key, value in sorted(self.pending_effects.items())
            },
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskState":
        if not isinstance(data, Mapping):
            raise StateValidationError("task state snapshot must be an object")
        allowed = {
            "schema_version",
            "root_run_id",
            "task",
            "revision",
            "plan_version",
            "branches",
            "goals",
            "facts",
            "plan",
            "pending_effects",
            "updated_at",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise StateValidationError(
                "unknown task state fields: " + ", ".join(unknown)
            )
        try:
            goals = {
                str(key): Goal.from_dict(value)
                for key, value in (data.get("goals") or {}).items()
            }
            facts = {
                str(key): Fact.from_dict(value)
                for key, value in (data.get("facts") or {}).items()
            }
            plan = {
                str(key): PlanNode.from_dict(value)
                for key, value in (data.get("plan") or {}).items()
            }
            effects = {
                str(key): PendingEffect.from_dict(value)
                for key, value in (data.get("pending_effects") or {}).items()
            }
        except (KeyError, TypeError, AttributeError) as exc:
            raise StateValidationError(f"malformed task state snapshot: {exc}") from exc
        return cls(
            root_run_id=data.get("root_run_id"),
            task=data.get("task"),
            revision=data.get("revision", 0),
            plan_version=data.get("plan_version", 0),
            goals=goals,
            facts=facts,
            plan=plan,
            pending_effects=effects,
            branches=dict(data.get("branches") or {}),
            updated_at=data.get("updated_at"),
            schema_version=data.get("schema_version", TASK_STATE_SCHEMA_VERSION),
        )


class EvidenceResolver:
    """Verify cited evidence against the run's own evidence directory.

    Only paths inside ``runtime/runs/<run_id>/`` are resolved, and only branch
    runs registered in the task state are accepted. The file's bytes are hashed
    and counted; a missing file, changed bytes or wrong owner is rejected.
    """

    def __init__(self, runtime_dir: Any, branches: Mapping[str, str]) -> None:
        from pathlib import Path

        self.runtime_dir = Path(runtime_dir)
        self.runs_root = (self.runtime_dir / "runs").resolve()
        self.branches = dict(branches)

    def run_dir(self, run_id: str) -> Any:
        validate_run_id(run_id)
        run_dir = (self.runs_root / run_id).resolve()
        if run_dir.parent != self.runs_root:
            raise EvidenceReferenceError("evidence run directory escapes the runtime root")
        return run_dir

    def verify(self, ref: EvidenceRef) -> None:
        expected_run = self.branches.get(ref.branch_id)
        if expected_run is None or expected_run != ref.run_id:
            raise EvidenceReferenceError(
                f"evidence {ref.evidence_id!r} does not belong to this task branch"
            )
        run_dir = self.run_dir(ref.run_id)
        if not run_dir.is_dir():
            raise EvidenceReferenceError(
                f"evidence run directory for {ref.run_id!r} does not exist"
            )
        from pathlib import Path

        candidate = Path(ref.reference)
        if candidate.is_absolute():
            raise EvidenceReferenceError("evidence reference must be relative")
        target = (run_dir / candidate).resolve()
        if target == run_dir or not target.is_relative_to(run_dir):
            raise EvidenceReferenceError("evidence reference escapes the run directory")
        if not target.is_file():
            raise EvidenceReferenceError(
                f"evidence file {ref.reference!r} does not exist in run {ref.run_id!r}"
            )
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise EvidenceReferenceError(
                f"cannot read evidence {ref.reference!r}: {exc}"
            ) from exc
        if len(data) != ref.byte_size:
            raise EvidenceReferenceError(
                f"evidence {ref.evidence_id!r} byte size changed "
                f"({ref.byte_size} != {len(data)})"
            )
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        if digest != ref.sha256:
            raise EvidenceReferenceError(
                f"evidence {ref.evidence_id!r} hash does not match the stored file"
            )


__all__ = [
    "EffectStatus",
    "EvidenceRef",
    "EvidenceReferenceError",
    "EvidenceResolver",
    "Fact",
    "FactKind",
    "FactValidity",
    "Goal",
    "GoalStatus",
    "PendingEffect",
    "PlanNode",
    "PlanNodeStatus",
    "PlanRejected",
    "StateConflict",
    "StateError",
    "StateValidationError",
    "TASK_STATE_SCHEMA_VERSION",
    "TaskState",
]
