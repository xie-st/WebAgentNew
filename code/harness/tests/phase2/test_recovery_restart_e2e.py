from __future__ import annotations

from pathlib import Path

from webagent.backends.playwright import PlaywrightBackend
from webagent.evidence import EvidenceStore
from webagent.recovery import RestartRecovery
from webagent.state import EffectStatus, StateStore

from e1_helpers import interrupt_l04_after_effective_save, judge_l04


def test_restart_after_effective_save_keeps_pending_and_does_not_resend(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    interrupted = interrupt_l04_after_effective_save(
        runtime, seed=101, run_id="e1-restart-101"
    )
    assert interrupted.save_action_id
    assert interrupted.state["pending_effects"][interrupted.save_action_id]["status"] == "unknown"

    store = StateStore(runtime, interrupted.run_id)
    loaded = store.load()
    pending = loaded.pending_effects[interrupted.save_action_id]
    assert pending.status is EffectStatus.UNKNOWN

    # The process is gone. Restart the persistent local site in a new process-like
    # branch and re-observe; this must not touch the pending effect or send the
    # save action again.
    interrupted.prepared.site.start()
    evidence = EvidenceStore(runtime, "run-b")
    backend = PlaywrightBackend(
        "run-b", evidence=evidence, headless=True, runtime_dir=runtime
    )
    try:
        backend.start(f"{interrupted.prepared.base_url}/product/{interrupted.target_id}")
        restart = RestartRecovery(store).recover(
            new_branch_id="recovery-1",
            new_run_id="run-b",
            observer=lambda: backend.observe(),
        )
        assert restart.generator_stack_restored is False
        assert restart.resent_action_ids == ()
        assert restart.pending_effects[0].action_id == interrupted.save_action_id
        assert restart.pending_effects[0].status is EffectStatus.UNKNOWN
        assert store.state.pending_effects[interrupted.save_action_id].status is EffectStatus.UNKNOWN
        assert store.state.branches["recovery-1"] == "run-b"
        assert restart.observation.url.endswith(f"/product/{interrupted.target_id}")

        judge = judge_l04(
            interrupted.prepared.db_path,
            target_id=interrupted.target_id,
            expected_text=interrupted.note,
        )
        assert judge.passed is True, judge.to_dict()
        assert judge.detail["save_count"] == 1
    finally:
        backend.close()
        interrupted.prepared.stop()

    reloaded = StateStore(runtime, interrupted.run_id)
    state = reloaded.load()
    assert state.pending_effects[interrupted.save_action_id].status is EffectStatus.UNKNOWN
    assert state.branches["recovery-1"] == "run-b"
