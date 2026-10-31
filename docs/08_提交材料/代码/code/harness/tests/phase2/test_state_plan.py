"""D05: plan DAG integrity, requirement preservation and distinct-object counts."""

from __future__ import annotations

from pathlib import Path

import pytest

from state_helpers import fact, goal, new_store, node, ref_for

from webagent.state import (
    PlanRejected,
    PlanUpdate,
    StateStore,
    StateUpdate,
    StateValidationError,
)


def _seed(tmp_path: Path):
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            goals=(
                goal(
                    criteria=("each object is present", "five distinct objects"),
                    scope=("obj-1", "obj-2", "obj-3", "obj-4", "obj-5"),
                    completed=("obj-1", "obj-2"),
                ),
            ),
        )
    )
    return runtime, store


def test_plan_cycle_is_rejected(tmp_path: Path) -> None:
    _, store = _seed(tmp_path)
    with pytest.raises(PlanRejected, match="cycle"):
        store.apply_plan_update(
            PlanUpdate(
                base_revision=1,
                plan_version=0,
                plan_nodes=(
                    node("n1", depends_on=("n2",)),
                    node("n2", depends_on=("n1",)),
                ),
            )
        )
    assert store.state.plan == {}
    assert store.state.plan_version == 0
    assert store.state.revision == 1


def test_unknown_goal_and_node_dependencies_are_rejected(tmp_path: Path) -> None:
    _, store = _seed(tmp_path)
    with pytest.raises(StateValidationError, match="unknown goal"):
        store.apply_update(
            StateUpdate(
                base_revision=1,
                plan_version=0,
                goals=(goal(goal_id="g2", depends_on=("g-missing",)),),
            )
        )
    with pytest.raises(StateValidationError, match="unknown node"):
        store.apply_plan_update(
            PlanUpdate(
                base_revision=1,
                plan_version=0,
                plan_nodes=(node("n1", depends_on=("n-missing",)),),
            )
        )
    assert store.state.revision == 1
    assert store.state.plan == {}


def test_plan_update_cannot_remove_unfinished_goal(tmp_path: Path) -> None:
    _, store = _seed(tmp_path)
    with pytest.raises(PlanRejected, match="unfinished goal"):
        store.apply_plan_update(
            PlanUpdate(
                base_revision=1,
                plan_version=0,
                removed_goal_ids=("g1",),
            )
        )
    assert set(store.state.goals) == {"g1"}
    assert store.state.revision == 1
    assert store.state.completion_status() == "incomplete"


def test_plan_update_cannot_drop_acceptance_or_object_scope(tmp_path: Path) -> None:
    _, store = _seed(tmp_path)
    with pytest.raises(PlanRejected, match="acceptance condition"):
        store.apply_plan_update(
            PlanUpdate(
                base_revision=1,
                plan_version=0,
                goals=(
                    goal(
                        criteria=("each object is present",),
                        scope=("obj-1", "obj-2", "obj-3", "obj-4", "obj-5"),
                        completed=("obj-1", "obj-2"),
                    ),
                ),
            )
        )
    with pytest.raises(PlanRejected, match="cannot drop objects"):
        store.apply_plan_update(
            PlanUpdate(
                base_revision=1,
                plan_version=0,
                goals=(
                    goal(
                        criteria=("each object is present", "five distinct objects"),
                        scope=("obj-1", "obj-2"),
                        completed=("obj-1", "obj-2"),
                    ),
                ),
            )
        )
    assert store.state.goals["g1"].object_scope == (
        "obj-1",
        "obj-2",
        "obj-3",
        "obj-4",
        "obj-5",
    )
    assert len(store.state.goals["g1"].acceptance_criteria) == 2
    assert store.state.revision == 1


def test_valid_plan_persists_new_version_and_dependencies(tmp_path: Path) -> None:
    runtime, store = _seed(tmp_path)
    store.apply_update(
        StateUpdate(
            base_revision=1,
            plan_version=0,
            facts=(
                # A fact is required for a plan input; the inferred kind needs no
                # page evidence and is explicitly not verification.
                fact(goal_id="g1"),
            ),
        )
    )
    store.apply_plan_update(
        PlanUpdate(
            base_revision=2,
            plan_version=0,
            plan_nodes=(
                node(
                    "n1",
                    scope=("obj-1", "obj-2"),
                    resources={"tab": "tab-1", "write_target": "draft-7"},
                ),
                node("n2", depends_on=("n1",), input_fact_ids=("f1",), scope=("obj-3",)),
            ),
        )
    )
    assert store.state.plan_version == 1
    assert store.state.revision == 3
    loaded = StateStore(runtime, "run-a").load()
    assert loaded.plan["n2"].depends_on == ("n1",)
    assert loaded.plan["n2"].plan_version == 1
    assert loaded.plan["n1"].resources["write_target"] == ("draft-7",)
    assert set(loaded.plan) == {"n1", "n2"}


def test_distinct_objects_are_counted_once_across_goals(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            goals=(
                goal(
                    goal_id="g1",
                    scope=("obj-1", "obj-2", "obj-3", "obj-4", "obj-5"),
                    completed=("obj-1", "obj-2"),
                ),
                goal(
                    goal_id="g2",
                    requirement="collect a second view",
                    criteria=("object is complete",),
                    scope=("obj-2", "obj-3"),
                    completed=("obj-2",),
                ),
            ),
        )
    )
    progress = store.state.object_progress()
    assert progress["objects_total"] == 5
    assert progress["objects_completed"] == 2
    assert progress["objects_remaining"] == 3
    assert progress["completed_object_ids"] == ["obj-1", "obj-2"]
    assert progress["remaining_object_ids"] == ["obj-3", "obj-4", "obj-5"]

    per_goal = store.state.object_progress(["g2"])
    assert per_goal["objects_total"] == 2
    assert per_goal["objects_completed"] == 1
    assert per_goal["objects_remaining"] == 1

    # A second visit or repeated evidence for the same object is not a second
    # object. Progress comes from the completed-id set, not a visit counter.
    store.apply_update(
        StateUpdate(
            base_revision=1,
            plan_version=0,
            facts=(
                fact(
                    "f-other", object_id="obj-1", goal_id="g1", kind="inferred"
                ),
            ),
        )
    )
    assert store.state.object_progress()["objects_completed"] == 2


def test_verified_requirement_may_be_removed_but_blocked_is_kept(tmp_path: Path) -> None:
    runtime, store = _seed(tmp_path)
    ref = ref_for(runtime)
    store.verify_goal(
        "g1",
        evidence=(ref,),
        base_revision=1,
        plan_version=0,
    )
    assert store.state.goals["g1"].status.value == "verified"
    store.apply_plan_update(
        PlanUpdate(
            base_revision=2,
            plan_version=0,
            removed_goal_ids=("g1",),
        )
    )
    assert store.state.goals == {}

    # A blocked requirement stays; completion is not rewritten as complete.
    store2 = new_store(tmp_path / "runtime2")
    store2.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            goals=(goal(status="blocked"),),
        )
    )
    assert store2.state.completion_status() == "blocked"
    with pytest.raises(PlanRejected, match="unfinished goal"):
        store2.apply_plan_update(
            PlanUpdate(base_revision=1, plan_version=0, removed_goal_ids=("g1",))
        )
    assert set(store2.state.goals) == {"g1"}
