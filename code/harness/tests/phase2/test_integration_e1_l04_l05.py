from __future__ import annotations

from pathlib import Path

from webagent.recovery import EffectOutcome

from e1_helpers import run_l04, run_l05


def test_l04_lost_feedback_probe_saves_once_and_verifies(tmp_path: Path) -> None:
    run = run_l04(tmp_path / "runtime", seed=101, run_id="e1-l04-101")
    assert run.step is not None and run.step.status.value == "completed"
    assert run.lost_feedback is True
    assert run.resolution is not None
    assert run.resolution.outcome is EffectOutcome.CONFIRMED
    assert run.resolution.retry_allowed is False
    assert run.judge.passed is True, run.judge.to_dict()
    assert run.save_count == 1
    assert run.final_value == str(run.prepared.case.task_params["l04_draft_text"])
    assert run.verification is not None and run.verification.verified is True
    assert run.store.state.goals["g1"].status.value == "verified"
    assert run.store.state.completion_status() == "complete"
    effects = list(run.store.state.pending_effects.values())
    assert len(effects) == 1
    assert effects[0].status.value == "confirmed"
    events = {event["event_type"] for event in run.events}
    assert "verification" in events


def test_l05_second_chain_recovers_without_redoing_first_or_third(tmp_path: Path) -> None:
    run = run_l05(tmp_path / "runtime", seed=101, run_id="e1-l05-101")
    assert run.step is not None and run.step.status.value == "completed"
    assert run.failed_object == run.targets[1]["id"]
    assert run.judge.passed is True, run.judge.to_dict()
    assert run.recovery and run.recovery[0].applied is True
    plan = run.recovery[0].plan
    assert plan.preserved_verified_goal_ids == ("g1", "g3")
    assert plan.invalidated_goal_ids == ("g2", "g2-check")
    assert [(item.goal_id, item.status.value) for item in run.verifications] == [
        ("g1", "pass"),
        ("g3", "pass"),
        ("g2", "pass"),
        ("g2-check", "pass"),
    ]
    assert run.store.state.goals["g1"].status.value == "verified"
    assert run.store.state.goals["g3"].status.value == "verified"
    assert run.store.state.goals["g2"].status.value == "verified"
    assert run.store.state.goals["g2-check"].status.value == "verified"
    assert run.store.state.completion_status() == "complete"
    assert run.judge.detail["save_counts"][run.targets[0]["id"]] == 1
    assert run.judge.detail["save_counts"][run.targets[2]["id"]] == 1
    events = [event["event_type"] for event in run.events]
    assert "recovery" in events and "verification" in events
