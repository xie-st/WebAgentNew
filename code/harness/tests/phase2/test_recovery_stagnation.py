from __future__ import annotations

from pathlib import Path

from webagent.clock import FakeClock
from webagent.recovery import (
    RecoveryController,
    RecoveryCoordinator,
    RecoveryLimits,
    RecoveryPlanner,
    StagnationDetector,
)

from e1_helpers import bootstrap_store, make_goal


def _record(detector: StagnationDetector, *, evidence=(), gap=None) -> None:
    before = None if gap is None else gap + 1
    detector.record(
        action_key="click-save",
        state_key="same-page",
        observation_id=f"obs-{len(detector.records) + 1}",
        new_evidence_ids=evidence,
        gap_before=before,
        gap_after=gap,
    )


def test_full_window_with_repeated_state_triggers_a_signal() -> None:
    detector = StagnationDetector(window=6, clock=FakeClock())
    for _ in range(5):
        _record(detector)
    assert detector.evaluate() is None
    _record(detector)
    signal = detector.evaluate()
    assert signal is not None
    assert signal.window_size == 6
    assert signal.repeated_state_count == 6
    assert signal.action_key == "click-save"
    assert detector.last_progress_at is None


def test_same_action_with_new_evidence_is_not_stagnation() -> None:
    detector = StagnationDetector(window=6, clock=FakeClock())
    for _ in range(5):
        _record(detector)
    _record(detector, evidence=("ev-new",))
    assert detector.evaluate() is None
    assert detector.last_progress_at is not None
    # The progress reset the window; five more no-progress actions still do not
    # reach six.
    for _ in range(5):
        _record(detector)
    assert detector.evaluate() is None


def test_same_page_alone_is_not_enough_when_the_state_does_not_repeat() -> None:
    detector = StagnationDetector(window=6)
    for index in range(6):
        detector.record(
            action_key=f"click-{index}",
            state_key=f"state-{index}",
            observation_id=f"obs-{index}",
        )
    assert detector.evaluate() is None


def test_signal_prompt_is_bounded_to_one_next_adjustment_and_three_reflections() -> None:
    detector = StagnationDetector(window=2, min_repeated_states=2)
    for _ in range(2):
        detector.record(
            action_key="click-save",
            state_key="same-page",
            observation_id=f"obs-{len(detector.records)}",
        )
    signal = detector.evaluate()
    assert signal is not None
    prompt = detector.prompt(
        signal,
        goal_requirement="save the profile note",
        evidence_conflict="the page still shows the old note",
        next_adjustment="re-observe before clicking save again",
        reflections=("r1", "r2", "r3", "r4"),
    )
    assert "Current goal" in prompt
    assert "Evidence conflict" in prompt
    assert prompt.count("One next adjustment") == 1
    assert "r1" not in prompt
    assert "r2" in prompt and "r3" in prompt and "r4" in prompt


def test_invalid_loop_triggers_bounded_recovery_and_then_refuses(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(
        runtime, [make_goal("g1", status="awaiting_verification")]
    )
    detector = StagnationDetector(window=6, clock=FakeClock())
    for _ in range(6):
        detector.record(
            action_key="click-save",
            state_key="same-page",
            observation_id=f"obs-{len(detector.records)}",
        )
    signal = detector.evaluate()
    assert signal is not None
    coordinator = RecoveryCoordinator(
        RecoveryPlanner(store),
        RecoveryController(
            store,
            limits=RecoveryLimits(max_recoveries_per_goal=2, max_recoveries_total=5),
        ),
        detector,
    )
    first = coordinator.handle_stagnation(
        signal, goal_id="g1", action_id="act-loop", observation_id="obs-6"
    )
    assert first is not None and first.recovery.applied is True
    assert store.state.goals["g1"].status.value == "invalidated"
    second = coordinator.handle_stagnation(
        signal, goal_id="g1", action_id="act-loop", observation_id="obs-6"
    )
    assert second is not None and second.recovery.applied is True
    third = coordinator.handle_stagnation(
        signal, goal_id="g1", action_id="act-loop", observation_id="obs-6"
    )
    assert third is not None
    assert third.recovery.applied is False
    assert "per-goal" in third.recovery.attempt.reason
