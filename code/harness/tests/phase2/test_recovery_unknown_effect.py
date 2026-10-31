from __future__ import annotations

from pathlib import Path

from webagent.recovery import (
    EffectExpectation,
    EffectOutcome,
    RecoveryClass,
    UnknownEffectResolver,
)
from webagent.state import EffectStatus, StateStore

from e1_helpers import make_observation, make_goal, bootstrap_store, value_element


def _pending_effect(runtime: Path) -> StateStore:
    store = bootstrap_store(runtime, [make_goal("g1", scope=("P-TEST",))])
    store.prepare_effect(
        action_id="act-save",
        target="5",
        object_id="P-TEST",
        goal_id="g1",
        observation_id="obs-before",
        note="save profile note",
    )
    return store


def test_unknown_effect_is_probed_once_and_confirmed_without_retry(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = _pending_effect(runtime)
    observation = make_observation(
        runtime,
        observation_id="obs-after",
        values=(value_element("Saved profile note"),),
    )
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return observation

    resolution = UnknownEffectResolver(store).resolve(
        store.state.pending_effects["act-save"],
        probe=probe,
        expectation=EffectExpectation(
            field_role="textbox",
            field_name="Review draft",
            expected_value="Saved profile note",
        ),
    )
    assert calls["n"] == 1
    assert resolution.outcome is EffectOutcome.CONFIRMED
    assert resolution.retry_allowed is False
    assert store.state.pending_effects["act-save"].status is EffectStatus.CONFIRMED
    assert store.state.pending_effects["act-save"].updated_revision is not None


def test_decisive_difference_marks_failed_and_allows_one_controlled_retry(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = _pending_effect(runtime)
    observation = make_observation(
        runtime,
        observation_id="obs-after",
        values=(value_element("old value"),),
    )
    resolution = UnknownEffectResolver(store).resolve(
        store.state.pending_effects["act-save"],
        probe=lambda: observation,
        expectation=EffectExpectation(
            field_role="textbox",
            field_name="Review draft",
            expected_value="new value",
        ),
    )
    assert resolution.outcome is EffectOutcome.NOT_APPLIED
    assert resolution.retry_allowed is True
    assert store.state.pending_effects["act-save"].status is EffectStatus.FAILED


def test_ambiguous_page_keeps_unknown_and_forbids_retry(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = _pending_effect(runtime)
    observation = make_observation(runtime, observation_id="obs-after", values=())
    resolution = UnknownEffectResolver(store).resolve(
        store.state.pending_effects["act-save"],
        probe=lambda: observation,
        expectation=EffectExpectation(
            field_role="textbox",
            field_name="Review draft",
            expected_value="new value",
        ),
    )
    assert resolution.outcome is EffectOutcome.UNKNOWN
    assert resolution.retry_allowed is False
    assert store.state.pending_effects["act-save"].status is EffectStatus.UNKNOWN


def test_classification_distinguishes_unknown_effect_from_condition_failure() -> None:
    from webagent.recovery import classify_condition, classify_failure
    from webagent.protocol import FeedbackStatus

    assert (
        classify_failure(status=FeedbackStatus.UNKNOWN, action_op="click")
        is RecoveryClass.UNKNOWN_EFFECT
    )
    assert classify_condition(passed=False) is RecoveryClass.CONDITION_FAILED
    assert classify_condition(passed=None) is RecoveryClass.UNKNOWN
