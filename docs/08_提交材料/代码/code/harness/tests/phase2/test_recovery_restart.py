from __future__ import annotations

from pathlib import Path

from webagent.recovery import RestartRecovery
from webagent.state import EffectStatus, StateStore

from e1_helpers import bootstrap_store, make_goal, make_observation


def test_restart_loads_snapshot_registers_branch_and_never_resends(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", scope=("P-TEST",))])
    store.prepare_effect(
        action_id="act-effective",
        target="5",
        object_id="P-TEST",
        goal_id="g1",
        observation_id="obs-before",
        note="effective save whose feedback was interrupted",
    )
    effect_before = store.pending_effects[0]
    assert effect_before.status is EffectStatus.PENDING

    observation = make_observation(
        runtime,
        run_id="run-b",
        observation_id="obs-restart",
        values=(("textbox", "Review draft", "Saved profile note"),),
    )
    result = RestartRecovery(store).recover(
        new_branch_id="recovery-1",
        new_run_id="run-b",
        observer=lambda: observation,
    )
    assert result.generator_stack_restored is False
    assert result.resent_action_ids == ()
    assert result.observation.observation_id == "obs-restart"
    assert result.pending_effects[0].action_id == "act-effective"
    assert result.pending_effects[0].status is EffectStatus.PENDING
    assert store.state.branches == {"root": "run-a", "recovery-1": "run-b"}
    assert store.state.pending_effects["act-effective"].status is EffectStatus.PENDING

    # A new store loading the same complete snapshot sees the same pending record.
    reloaded = StateStore(runtime, "run-a")
    state = reloaded.load()
    assert state.pending_effects["act-effective"].status is EffectStatus.PENDING
    assert state.branches["recovery-1"] == "run-b"
