"""D04: atomic snapshots, rollback on save failure, consistency after load.

The interruption tests write a half temporary snapshot directly; the loader must
ignore it and keep the last complete ``state.json``. A failed snapshot write must
leave both the on-disk and in-memory state at the previous revision.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from state_helpers import fact, goal, new_store, node, ref_for

from webagent.state import (
    PlanUpdate,
    StateConflict,
    StatePersistenceError,
    StateStore,
    StateUpdate,
)


def _break_write(store: StateStore, data: bytes) -> None:
    """Simulate a crash halfway through the temporary file."""

    store.state_dir.mkdir(parents=True, exist_ok=True)
    partial = store.state_dir / f".state.json.{uuid.uuid4().hex}.part"
    partial.write_bytes(data[: max(1, len(data) // 2)])
    raise OSError("simulated interrupted flush")


def test_interrupted_write_keeps_old_complete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = ref_for(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(evidence=(ref,)),))
    )
    baseline = store.state.to_dict()
    old_bytes = store.state_path.read_bytes()

    monkeypatch.setattr(store, "_atomic_write", lambda data: _break_write(store, data))
    with pytest.raises(StatePersistenceError):
        store.apply_update(
            StateUpdate(
                base_revision=1,
                plan_version=0,
                goals=(goal(goal_id="g2", requirement="second requirement"),),
            )
        )

    # The failed candidate is not installed.
    assert store.state.to_dict() == baseline
    assert store.state_path.read_bytes() == old_bytes

    # A half-written ``.part`` file exists but loading ignores it.
    parts = list(store.state_dir.glob(".state.json.*.part"))
    assert parts
    assert all(part != store.state_path for part in parts)

    reloaded = StateStore(runtime, "run-a")
    loaded = reloaded.load()
    assert loaded.to_dict() == baseline
    assert set(loaded.goals) == {"g1"}


def test_save_failure_does_not_commit_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = ref_for(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(),))
    )
    committed = store.state
    file_before = store.state_path.read_bytes()

    monkeypatch.setattr(store, "_atomic_write", lambda data: _break_write(store, data))
    with pytest.raises(StatePersistenceError):
        store.apply_update(
            StateUpdate(
                base_revision=1,
                plan_version=0,
                facts=(fact(evidence=(ref,)),),
            )
        )

    assert store.state is committed
    assert store.state.revision == 1
    assert store.state.facts == {}
    assert store.state_path.read_bytes() == file_before
    assert StateStore(runtime, "run-a").load().revision == 1


def test_roundtrip_preserves_goals_facts_dependencies_and_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    ref = ref_for(runtime)
    goal_one = goal(
        criteria=("each object is present", "five distinct objects"),
        scope=("obj-1", "obj-2"),
        evidence=(ref,),
        completed=("obj-1",),
    )
    store.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            goals=(goal_one,),
        )
    )
    store.apply_update(
        StateUpdate(
            base_revision=1,
            plan_version=0,
            facts=(fact(evidence=(ref,), goal_id="g1"),),
        )
    )
    store.apply_plan_update(
        PlanUpdate(
            base_revision=2,
            plan_version=0,
            plan_nodes=(node(scope=("obj-1",), input_fact_ids=("f1",)),),
        )
    )
    store.prepare_effect(
        action_id="act-1",
        target="page-1",
        object_id="obj-1",
        goal_id="g1",
    )
    saved = store.state
    assert saved.revision == 4
    # The actual write path (not the fault injector) was used for this snapshot.
    assert store.state_path.is_file()
    payload = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert payload["revision"] == saved.revision

    loaded_store = StateStore(runtime, "run-a")
    loaded = loaded_store.load()
    loaded_store.verify_evidence(ref)
    assert loaded.to_dict() == saved.to_dict()
    assert loaded.goals["g1"].depends_on == saved.goals["g1"].depends_on
    assert loaded.goals["g1"].object_scope == ("obj-1", "obj-2")
    assert loaded.facts["f1"].evidence[0].sha256 == ref.sha256
    assert loaded.plan["n1"].input_fact_ids == ("f1",)
    assert loaded.pending_effects["act-1"].status.value == "pending"
    assert loaded.view()["pending_effects"]["act-1"]["status"] == "pending"


def test_multiline_task_text_is_preserved(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    task = "Find five matching posts.\nFor each: edit a draft, screenshot it, then cancel.\nDo not submit."
    store = StateStore(runtime, "run-a")
    store.initialize(task)
    loaded = StateStore(runtime, "run-a").load()
    assert loaded.task == task


def test_stale_base_revision_is_rejected_with_current_state(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(),))
    )
    with pytest.raises(StateConflict) as caught:
        store.apply_update(
            StateUpdate(
                base_revision=0,
                plan_version=0,
                goals=(goal(goal_id="g2", requirement="second"),),
            )
        )
    conflict = caught.value
    assert conflict.reason == "stale_base_revision"
    assert conflict.current.revision == 1
    assert set(conflict.current_view()["goals"]) == {"g1"}
    assert store.state.revision == 1
    # Re-check and submit against the reported revision.
    store.apply_update(
        StateUpdate(
            base_revision=conflict.current.revision,
            plan_version=conflict.current.plan_version,
            goals=(goal(goal_id="g2", requirement="second"),),
        )
    )
    assert store.state.revision == 2
    assert set(store.state.goals) == {"g1", "g2"}


def test_stale_plan_version_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(),))
    )
    store.apply_plan_update(
        PlanUpdate(base_revision=1, plan_version=0, plan_nodes=(node(),))
    )
    assert store.state.plan_version == 1
    with pytest.raises(StateConflict) as caught:
        store.apply_plan_update(
            PlanUpdate(base_revision=2, plan_version=0, plan_nodes=(node(),))
        )
    assert caught.value.reason == "stale_plan_version"
    assert store.state.plan_version == 1


def test_pending_effect_survives_reload_and_is_not_resent(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(),))
    )
    store.prepare_effect(
        action_id="act-side-effect",
        target="page-1",
        object_id="obj-1",
        observation_id="obs-1",
    )
    revision = store.state.revision

    loaded = StateStore(runtime, "run-a").load()
    effect = loaded.pending_effects["act-side-effect"]
    assert effect.status.value == "pending"
    assert effect.action_id == "act-side-effect"
    assert effect.target == "page-1"
    assert effect.object_id == "obj-1"
    assert loaded.revision == revision
    assert loaded.unconfirmed_effects == (effect,)
    # Loading is read-only; no action id was emitted or cleared by loading.
    assert set(loaded.unconfirmed_effects[0].__dict__) == set(effect.__dict__)
