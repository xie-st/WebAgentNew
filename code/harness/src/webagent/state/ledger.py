"""Versioned state and plan mutations for the task ledger.

Every request carries the revision and plan version it was based on and the
branch that produced it. A stale request is refused with the committed state so
the branch can re-read before submitting again; nothing here merges by guess.

The rules below are deliberately narrower than the record types:

* a model-facing update cannot mark a goal or fact ``verified``;
* observed and inferred facts stay separate;
* a plan revision cannot delete an unfinished requirement, blank its acceptance
  criteria or shrink its object scope;
* evidence references are append-only and validated before commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..protocol import validate_run_id
from .records import (
    EffectStatus,
    EvidenceRef,
    EvidenceResolver,
    Fact,
    FactKind,
    Goal,
    GoalStatus,
    PendingEffect,
    PlanNode,
    PlanRejected,
    StateConflict,
    StateValidationError,
    TaskState,
)


@dataclass(frozen=True)
class StateUpdate:
    """An ordinary ledger update: requirements and facts, not the plan."""

    base_revision: int
    plan_version: int
    source_branch: str = "root"
    goals: tuple[Goal, ...] = ()
    facts: tuple[Fact, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        _check_request(self.base_revision, self.plan_version, self.source_branch)
        _record_tuple(self.goals, Goal, "StateUpdate.goals")
        _record_tuple(self.facts, Fact, "StateUpdate.facts")


@dataclass(frozen=True)
class PlanUpdate:
    """A plan revision, plus any goals that revision introduces or updates."""

    base_revision: int
    plan_version: int
    source_branch: str = "root"
    goals: tuple[Goal, ...] = ()
    plan_nodes: tuple[PlanNode, ...] = ()
    removed_goal_ids: tuple[str, ...] = ()
    removed_plan_node_ids: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        _check_request(self.base_revision, self.plan_version, self.source_branch)
        _record_tuple(self.goals, Goal, "PlanUpdate.goals")
        _record_tuple(self.plan_nodes, PlanNode, "PlanUpdate.plan_nodes")
        _id_tuple(self.removed_goal_ids, "PlanUpdate.removed_goal_ids")
        _id_tuple(self.removed_plan_node_ids, "PlanUpdate.removed_plan_node_ids")
        overlap = set(self.removed_goal_ids) & {goal.goal_id for goal in self.goals}
        if overlap:
            raise PlanRejected(
                "plan update both removes and upserts goals: " + ", ".join(sorted(overlap))
            )
        overlap = set(self.removed_plan_node_ids) & {
            node.node_id for node in self.plan_nodes
        }
        if overlap:
            raise PlanRejected(
                "plan update both removes and upserts plan nodes: "
                + ", ".join(sorted(overlap))
            )


def _check_request(base_revision: Any, plan_version: Any, source_branch: Any) -> None:
    if not isinstance(base_revision, int) or isinstance(base_revision, bool) or base_revision < 0:
        raise StateValidationError("base_revision must be an integer >= 0")
    if not isinstance(plan_version, int) or isinstance(plan_version, bool) or plan_version < 0:
        raise StateValidationError("plan_version must be an integer >= 0")
    if not isinstance(source_branch, str) or not source_branch.strip():
        raise StateValidationError("source_branch must be a non-empty string")


def _record_tuple(value: Sequence[Any], cls: type, name: str) -> None:
    if not isinstance(value, tuple):
        raise StateValidationError(f"{name} must be a tuple of {cls.__name__} records")
    for record in value:
        if not isinstance(record, cls):
            raise StateValidationError(f"{name} must contain only {cls.__name__} records")


def _id_tuple(value: Sequence[str], name: str) -> None:
    if not isinstance(value, tuple):
        raise StateValidationError(f"{name} must be a tuple of ids")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise StateValidationError(f"{name} must contain non-empty id strings")


def _check_basis(state: TaskState, base_revision: int, plan_version: int) -> None:
    if state.revision != base_revision:
        raise StateConflict(
            f"update is based on revision {base_revision}, current revision is "
            f"{state.revision}; re-read the state before retrying",
            current=state,
            reason="stale_base_revision",
            base_revision=base_revision,
            plan_version=plan_version,
        )
    if state.plan_version != plan_version:
        raise StateConflict(
            f"update is based on plan_version {plan_version}, current plan_version is "
            f"{state.plan_version}; re-read the plan before retrying",
            current=state,
            reason="stale_plan_version",
            base_revision=base_revision,
            plan_version=plan_version,
        )


def _branch(state: TaskState, branch_id: str, where: str) -> None:
    if branch_id not in state.branches:
        raise StateValidationError(f"{where} belongs to unregistered branch {branch_id!r}")


def _evidence_ids(goal: Goal) -> set[str]:
    return {ref.evidence_id for ref in goal.evidence}


def _upsert_goal(candidate: TaskState, incoming: Goal, *, allow_verified: bool) -> None:
    _branch(candidate, incoming.branch_id, f"goal {incoming.goal_id}")
    existing = candidate.goals.get(incoming.goal_id)
    if existing is None:
        if incoming.status is GoalStatus.VERIFIED and not allow_verified:
            raise StateValidationError(
                f"goal {incoming.goal_id!r} cannot be created as verified by a model update; "
                "verification is a separate evidence-backed transition"
            )
        candidate.goals[incoming.goal_id] = incoming
        return

    if existing == incoming:
        return
    if existing.branch_id != incoming.branch_id:
        raise StateValidationError(
            f"goal {incoming.goal_id!r} cannot move to another branch via an update"
        )
    if existing.status is GoalStatus.VERIFIED and not allow_verified:
        raise StateValidationError(
            f"goal {incoming.goal_id!r} is already verified; a model update cannot "
            "change it or its evidence"
        )
    if incoming.status is GoalStatus.VERIFIED and not allow_verified:
        raise StateValidationError(
            f"goal {incoming.goal_id!r} cannot become verified through a model update"
        )
    if incoming.requirement != existing.requirement:
        raise PlanRejected(
            f"goal {incoming.goal_id!r} cannot rewrite the user requirement"
        )
    if not set(existing.acceptance_criteria).issubset(incoming.acceptance_criteria):
        raise PlanRejected(
            f"goal {incoming.goal_id!r} cannot drop an acceptance condition"
        )
    if not set(existing.object_scope).issubset(incoming.object_scope):
        raise PlanRejected(f"goal {incoming.goal_id!r} cannot drop objects from its scope")
    if not set(existing.completed_object_ids).issubset(incoming.completed_object_ids):
        raise PlanRejected(
            f"goal {incoming.goal_id!r} cannot drop its completed object records"
        )
    if not _evidence_ids(existing).issubset(_evidence_ids(incoming)):
        raise StateValidationError(
            f"goal {incoming.goal_id!r} cannot remove supporting evidence references"
        )
    candidate.goals[incoming.goal_id] = incoming


def _upsert_fact(candidate: TaskState, incoming: Fact) -> None:
    _branch(candidate, incoming.branch_id, f"fact {incoming.fact_id}")
    existing = candidate.facts.get(incoming.fact_id)
    if existing is None:
        if incoming.revision != 1:
            raise StateValidationError(
                f"new fact {incoming.fact_id!r} must start at revision 1"
            )
        candidate.facts[incoming.fact_id] = incoming
        return
    if existing == incoming:
        return
    if existing.branch_id != incoming.branch_id:
        raise StateValidationError(
            f"fact {incoming.fact_id!r} cannot move to another branch via an update"
        )
    if existing.object_id != incoming.object_id or existing.field != incoming.field:
        raise StateValidationError(
            f"fact {incoming.fact_id!r} identifies another object or field; "
            "use a new fact id for another proposition"
        )
    if existing.kind is not incoming.kind:
        raise StateValidationError(
            f"fact {incoming.fact_id!r} cannot change kind from {existing.kind.value} "
            f"to {incoming.kind.value}; an inference is not promoted to an observation"
        )
    if incoming.revision != existing.revision + 1:
        raise StateValidationError(
            f"fact {incoming.fact_id!r} revision must increase by exactly one "
            f"(expected {existing.revision + 1})"
        )
    if not {ref.evidence_id for ref in existing.evidence}.issubset(
        {ref.evidence_id for ref in incoming.evidence}
    ):
        raise StateValidationError(
            f"fact {incoming.fact_id!r} cannot remove supporting evidence references"
        )
    if existing.goal_id != incoming.goal_id and existing.goal_id is not None:
        raise StateValidationError(
            f"fact {incoming.fact_id!r} cannot be retargeted to another goal"
        )
    candidate.facts[incoming.fact_id] = incoming


def apply_state_update(
    state: TaskState,
    update: StateUpdate,
    *,
    allow_verified: bool = False,
    evidence: EvidenceResolver | None = None,
) -> TaskState:
    """Return a committed-shape candidate for an ordinary state update."""

    _check_basis(state, update.base_revision, update.plan_version)
    _branch(state, update.source_branch, "source branch")
    candidate = state.copy()
    for goal in update.goals:
        _upsert_goal(candidate, goal, allow_verified=allow_verified)
    for fact in update.facts:
        _upsert_fact(candidate, fact)
    candidate.revision = state.revision + 1
    candidate.validate(evidence=evidence)
    return candidate


def apply_plan_update(
    state: TaskState,
    update: PlanUpdate,
    *,
    allow_verified: bool = False,
    evidence: EvidenceResolver | None = None,
) -> TaskState:
    """Return a candidate with a new plan version or reject the revision."""

    _check_basis(state, update.base_revision, update.plan_version)
    _branch(state, update.source_branch, "source branch")
    candidate = state.copy()
    new_plan_version = state.plan_version + 1

    for goal_id in update.removed_goal_ids:
        goal = candidate.goals.get(goal_id)
        if goal is None:
            raise PlanRejected(f"plan update removes unknown goal {goal_id!r}")
        if goal.status is not GoalStatus.VERIFIED:
            raise PlanRejected(
                f"plan update cannot remove unfinished goal {goal_id!r}; "
                "mark it blocked or incomplete instead"
            )
        del candidate.goals[goal_id]

    for node_id in update.removed_plan_node_ids:
        if node_id not in candidate.plan:
            raise PlanRejected(f"plan update removes unknown plan node {node_id!r}")
        del candidate.plan[node_id]

    for goal in update.goals:
        _upsert_goal(candidate, goal, allow_verified=allow_verified)

    for node in update.plan_nodes:
        _branch(candidate, node.branch_id, f"plan node {node.node_id}")
        existing = candidate.plan.get(node.node_id)
        if existing is not None:
            if existing.branch_id != node.branch_id:
                raise PlanRejected(
                    f"plan node {node.node_id!r} cannot move to another branch"
                )
            if existing.goal_id != node.goal_id:
                raise PlanRejected(
                    f"plan node {node.node_id!r} cannot be retargeted to another goal"
                )
            if (
                existing.status is GoalStatus.VERIFIED
                and not allow_verified
                and existing != node
            ):
                raise StateValidationError(
                    f"plan node {node.node_id!r} is already verified; a model plan "
                    "revision cannot change it"
                )
        if node.status is GoalStatus.VERIFIED and not allow_verified and node != existing:
            raise StateValidationError(
                f"plan node {node.node_id!r} cannot become verified through a model update"
            )
        if node.plan_version != new_plan_version:
            raise PlanRejected(
                f"plan node {node.node_id!r} must declare the new plan version "
                f"{new_plan_version}"
            )
        candidate.plan[node.node_id] = node

    candidate.plan_version = new_plan_version
    candidate.revision = state.revision + 1
    candidate.validate(evidence=evidence)
    return candidate


def make_effect(
    *,
    action_id: str,
    target: str,
    object_id: str,
    status: EffectStatus = EffectStatus.PENDING,
    goal_id: str | None = None,
    branch_id: str = "root",
    observation_id: str | None = None,
    created_revision: int,
    note: str = "",
) -> PendingEffect:
    return PendingEffect(
        action_id=action_id,
        target=target,
        object_id=object_id,
        status=status,
        goal_id=goal_id,
        branch_id=branch_id,
        observation_id=observation_id,
        created_revision=created_revision,
        note=note,
    )


def evidence_resolver(state: TaskState, runtime_dir: Any) -> EvidenceResolver:
    return EvidenceResolver(runtime_dir, state.branches)


__all__ = [
    "PlanUpdate",
    "StateUpdate",
    "apply_plan_update",
    "apply_state_update",
    "evidence_resolver",
    "make_effect",
]
