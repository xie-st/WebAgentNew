"""D03: proposed actions and model updates never create verification.

Checks observed/inferred separation, evidence existence/hash/ownership, and that
a saved pending side effect does not raise the verified count.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from state_helpers import fact, goal, new_store, ref_for

from webagent.state import (
    EvidenceReferenceError,
    Fact,
    Goal,
    GoalStatus,
    PlanNode,
    StateUpdate,
    StateValidationError,
)


def test_observed_fact_without_evidence_is_refused() -> None:
    with pytest.raises(StateValidationError, match="observed fact"):
        Fact(
            fact_id="f1",
            object_id="obj-1",
            field="name",
            value="Alpha",
            kind="observed",
            source="model claim",
        )


def test_verified_goal_requires_evidence() -> None:
    with pytest.raises(StateValidationError, match="without supporting evidence"):
        Goal(
            goal_id="g1",
            requirement="collect the object",
            acceptance_criteria=("object present",),
            status=GoalStatus.VERIFIED,
        )


def test_model_update_cannot_mark_goal_verified(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = ref_for(runtime)
    pending = goal(evidence=(ref,))
    store.apply_update(StateUpdate(base_revision=0, plan_version=0, goals=(pending,)))
    assert store.state.verified_goal_ids == ()

    verified_claim = goal(
        status="verified",
        evidence=(ref,),
        scope=("obj-1",),
        completed=("obj-1",),
    )
    with pytest.raises(StateValidationError, match="cannot become verified"):
        store.apply_update(
            StateUpdate(base_revision=1, plan_version=0, goals=(verified_claim,))
        )
    assert store.state.revision == 1
    assert store.state.verified_goal_ids == ()
    assert store.state.completion_status() == "incomplete"


def test_action_preparation_does_not_verify_goal(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(scope=("obj-1",)),))
    )
    before = store.state.verified_goal_ids
    effect = store.prepare_effect(
        action_id="act-1",
        target="page-1",
        object_id="obj-1",
        goal_id="g1",
        observation_id="obs-1",
    )
    assert effect.status.value == "pending"
    assert store.state.verified_goal_ids == before == ()
    assert store.state.completion_status() == "incomplete"


def test_missing_evidence_file_is_refused(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = replace(ref_for(runtime), reference="html/missing.html")
    with pytest.raises(EvidenceReferenceError, match="does not exist"):
        store.apply_update(
            StateUpdate(
                base_revision=0,
                plan_version=0,
                goals=(goal(evidence=(ref,)),),
            )
        )
    assert store.state.revision == 0


def test_bad_evidence_hash_is_refused(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = replace(ref_for(runtime), sha256="0" * 64)
    with pytest.raises(EvidenceReferenceError, match="hash"):
        store.apply_update(
            StateUpdate(
                base_revision=0,
                plan_version=0,
                goals=(goal(evidence=(ref,)),),
            )
        )
    assert store.state.revision == 0


def test_cross_task_evidence_is_refused(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    foreign = ref_for(runtime, run_id="run-b", evidence_id="ev-foreign")
    with pytest.raises(EvidenceReferenceError, match="does not belong"):
        store.apply_update(
            StateUpdate(
                base_revision=0,
                plan_version=0,
                goals=(goal(evidence=(foreign,)),),
            )
        )
    assert store.state.revision == 0


def test_cross_branch_evidence_is_refused_for_branch_record(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.register_branch("b1", "run-b1")
    store.register_branch("b2", "run-b2")
    foreign = ref_for(
        runtime,
        run_id="run-b2",
        branch_id="b2",
        evidence_id="ev-b2",
    )
    with pytest.raises(EvidenceReferenceError, match="another branch"):
        store.apply_update(
            StateUpdate(
                base_revision=2,
                plan_version=0,
                source_branch="b1",
                goals=(goal(branch_id="b1", evidence=(foreign,)),),
            )
        )
    assert store.state.revision == 2


def test_observed_and_inferred_facts_stay_separate(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = ref_for(runtime)
    inferred = fact(goal_id=None, kind="inferred", source="model inference")
    observed = fact(
        fact_id="f2",
        object_id="obj-2",
        kind="observed",
        source="observation obs-1",
        evidence=(ref,),
        goal_id=None,
    )
    store.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            facts=(inferred, observed),
        )
    )
    assert store.state.facts["f1"].kind.value == "inferred"
    assert store.state.facts["f2"].kind.value == "observed"
    assert store.state.facts["f1"].evidence == ()
    # An observation is not a completed goal: no verified count moved.
    assert store.state.verified_goal_ids == ()
    assert store.state.completion_status() == "incomplete"

    upgrade = fact(
        goal_id=None,
        kind="observed",
        source="now treated as observed",
        revision=2,
        evidence=(ref,),
    )
    with pytest.raises(StateValidationError, match="cannot change kind"):
        store.apply_update(
            StateUpdate(base_revision=1, plan_version=0, facts=(upgrade,))
        )
    assert store.state.facts["f1"].kind.value == "inferred"
    assert store.state.revision == 1


def test_bad_plan_node_does_not_commit(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(scope=("obj-1",)),))
    )
    bad = PlanNode(
        node_id="n1",
        goal_id="g1",
        object_scope=("obj-other",),
        plan_version=1,
    )
    from webagent.state import PlanUpdate

    with pytest.raises(StateValidationError, match="outside goal scope"):
        store.apply_plan_update(
            PlanUpdate(base_revision=1, plan_version=0, plan_nodes=(bad,))
        )
    assert store.state.revision == 1
    assert store.state.plan_version == 0
