"""Small builders shared by the phase-2 task-state tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from webagent.evidence import EvidenceStore
from webagent.state import (
    EvidenceRef,
    Fact,
    Goal,
    PendingEffect,
    PlanNode,
    StateStore,
)


def ref_for(
    runtime: Path,
    *,
    run_id: str = "run-a",
    branch_id: str = "root",
    evidence_id: str = "ev-1",
    payload: bytes = b"<html><body>complete</body></html>",
    observation_id: str = "obs-1",
    kind: str = "html",
) -> EvidenceRef:
    artifact = EvidenceStore(runtime, run_id).save_bytes(
        kind, payload, observation_id=observation_id
    )
    return EvidenceRef(
        evidence_id=evidence_id,
        run_id=run_id,
        branch_id=branch_id,
        reference=artifact.reference,
        sha256=artifact.sha256,
        byte_size=artifact.byte_size,
        kind=kind,
        observation_id=observation_id,
    )


def new_store(runtime: Path, *, task: str = "collect the requested objects") -> StateStore:
    store = StateStore(runtime, "run-a")
    store.initialize(task)
    return store


def goal(
    goal_id: str = "g1",
    *,
    requirement: str = "collect the requested objects",
    criteria: Sequence[str] = ("each object is present",),
    depends_on: Sequence[str] = (),
    scope: Sequence[str] = (),
    status: str = "pending",
    evidence: Iterable[EvidenceRef] = (),
    branch_id: str = "root",
    completed: Sequence[str] = (),
) -> Goal:
    return Goal(
        goal_id=goal_id,
        requirement=requirement,
        acceptance_criteria=tuple(criteria),
        depends_on=tuple(depends_on),
        object_scope=tuple(scope),
        status=status,
        evidence=tuple(evidence),
        branch_id=branch_id,
        completed_object_ids=tuple(completed),
    )


def fact(
    fact_id: str = "f1",
    *,
    object_id: str = "obj-1",
    field: str = "name",
    value: Any = "Alpha",
    kind: str = "inferred",
    source: str = "model inference",
    validity: str = "valid",
    revision: int = 1,
    evidence: Iterable[EvidenceRef] = (),
    goal_id: str | None = "g1",
    branch_id: str = "root",
) -> Fact:
    return Fact(
        fact_id=fact_id,
        object_id=object_id,
        field=field,
        value=value,
        kind=kind,
        source=source,
        validity=validity,
        revision=revision,
        evidence=tuple(evidence),
        goal_id=goal_id,
        branch_id=branch_id,
    )


def node(
    node_id: str = "n1",
    *,
    goal_id: str = "g1",
    depends_on: Sequence[str] = (),
    input_fact_ids: Sequence[str] = (),
    scope: Sequence[str] = (),
    resources: dict[str, Any] | None = None,
    status: str = "pending",
    plan_version: int = 1,
    branch_id: str = "root",
) -> PlanNode:
    return PlanNode(
        node_id=node_id,
        goal_id=goal_id,
        depends_on=tuple(depends_on),
        input_fact_ids=tuple(input_fact_ids),
        object_scope=tuple(scope),
        resources=resources or {},
        status=status,
        plan_version=plan_version,
        branch_id=branch_id,
    )


def effect(
    action_id: str = "act-1",
    *,
    target: str = "page-1",
    object_id: str = "obj-1",
    status: str = "pending",
    goal_id: str | None = "g1",
    branch_id: str = "root",
    created_revision: int = 1,
) -> PendingEffect:
    return PendingEffect(
        action_id=action_id,
        target=target,
        object_id=object_id,
        status=status,
        goal_id=goal_id,
        branch_id=branch_id,
        created_revision=created_revision,
    )
