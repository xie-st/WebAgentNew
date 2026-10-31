"""E06: recovery consumes the fresh observation, never a discarded proposal.

The control layer drops an unconsumed proposal when the user pauses and later
hands the Agent a fresh observation. These tests pin the recovery-side property:
the decision is based on the fresh observation, a discarded proposal is never
reported as an executed action and never buys another attempt, and repeating the
callback for the same observation does not spend a second round.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from e1_helpers import (
    bootstrap_store,
    evidence_ref,
    make_goal,
    make_observation,
    value_element,
)

from webagent.control import ControlCoordinator
from webagent.protocol import ActionOp, WebAction
from webagent.recovery import (
    FeedbackIntake,
    RecoveryClass,
    RecoveryController,
    RecoveryError,
    RecoveryIntake,
    RecoveryPlanner,
)
from webagent.state import GoalStatus


class CountingPlanner(RecoveryPlanner):
    """Counts planning rounds so a test can prove no extra round was spent."""

    def __init__(self, store) -> None:
        super().__init__(store)
        self.calls = 0

    def plan(self, failures):
        self.calls += 1
        return super().plan(failures)


def _discard(
    coordinator: ControlCoordinator,
    *,
    action_id: str,
    observation_id: str = "obs-old",
):
    action = WebAction(
        action_id=action_id,
        run_id="run-a",
        page_id="page-1",
        observation_id=observation_id,
        op=ActionOp.CLICK,
        args={"target": "1"},
    )
    coordinator.discard_proposal(action, reason="paused_before_dispatch:user_pause")
    return coordinator.discarded_proposals


def test_cancelled_proposal_and_fresh_observation_spend_no_attempt(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", status="running")])
    planner = CountingPlanner(store)
    controller = RecoveryController(store)
    intake = RecoveryIntake(planner, controller)

    fresh = make_observation(
        runtime,
        observation_id="obs-fresh",
        url="http://127.0.0.1/products",
        ax_text="Products page after the user left; no Agent action ran",
        values=(value_element("user-visible draft", name="Review draft"),),
    )
    ref = evidence_ref(runtime, fresh)
    coordinator = ControlCoordinator("run-a")
    discarded = _discard(coordinator, action_id="act-old")
    payload = FeedbackIntake(
        observation=fresh,
        failure_class=RecoveryClass.USER_PAUSE,
        goal_id="g1",
        executed_action_id=None,
        evidence=(ref,),
        discarded_proposals=discarded,
    )
    snapshot = store.state.to_dict()

    result = intake.consider(payload)

    assert result.consumed is True
    assert result.reused_observation is False
    assert result.plan is None
    assert result.recovery is None
    assert result.failure is None
    assert result.replayed_action_ids == ()
    assert result.discarded_proposals == discarded
    assert planner.calls == 0
    assert controller.total_attempts == 0
    assert store.state.to_dict() == snapshot
    assert fresh.observation_id in result.next_step
    assert "obs-old" not in result.next_step
    assert "act-old" not in result.next_step
    assert "not replayed" in result.next_step
    assert result.to_dict()["observation_id"] == "obs-fresh"

    # The same fresh callback cannot spend a second attempt or planning round.
    repeated = intake.consider(payload)
    assert repeated.consumed is False
    assert repeated.reused_observation is True
    assert repeated.recovery is None
    assert planner.calls == 0
    assert controller.total_attempts == 0
    assert intake.consumed_observation_ids == ("obs-fresh",)


def test_recovery_decision_anchors_on_the_fresh_observation(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", status="awaiting_verification")])
    planner = CountingPlanner(store)
    controller = RecoveryController(store)
    intake = RecoveryIntake(planner, controller)

    fresh = make_observation(
        runtime,
        observation_id="obs-fresh-2",
        url="http://127.0.0.1/product/P-TEST",
        ax_text="Saved profile note: rechecked value",
        values=(value_element("rechecked value"),),
    )
    ref = evidence_ref(runtime, fresh)
    coordinator = ControlCoordinator("run-a")
    discarded = _discard(coordinator, action_id="act-old", observation_id="obs-old")
    payload = FeedbackIntake(
        observation=fresh,
        failure_class=RecoveryClass.CONDITION_FAILED,
        goal_id="g1",
        executed_action_id="act-new",
        sequence=3,
        message="the saved value did not match the expected condition",
        evidence=(ref,),
        discarded_proposals=discarded,
    )

    result = intake.consider(payload)

    assert result.consumed is True
    assert result.failure is not None
    assert result.failure.action_id == "act-new"
    assert result.failure.observation_id == fresh.observation_id
    assert result.failure.evidence == (ref,)
    assert result.plan is not None
    assert result.plan.anchor is not None
    assert result.plan.anchor.anchor_id == "act-new"
    assert result.plan.invalidated_goal_ids == ("g1",)
    assert result.recovery is not None
    assert result.recovery.applied is True
    assert store.state.goals["g1"].status is GoalStatus.INVALIDATED
    assert controller.total_attempts == 1
    assert planner.calls == 1
    assert fresh.observation_id in result.next_step
    assert result.replayed_action_ids == ()
    assert result.discarded_proposals == discarded
    assert result.to_dict()["recovery"]["applied"] is True

    repeated = intake.consider(payload)
    assert repeated.consumed is False
    assert repeated.reused_observation is True
    assert repeated.recovery is result.recovery
    assert controller.total_attempts == 1
    assert planner.calls == 1
    assert store.state.goals["g1"].status is GoalStatus.INVALIDATED


def test_a_discarded_proposal_cannot_be_reported_as_executed(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", status="running")])
    fresh = make_observation(runtime, observation_id="obs-fresh-3")
    coordinator = ControlCoordinator("run-a")
    discarded = _discard(coordinator, action_id="act-old")

    with pytest.raises(RecoveryError, match="discarded proposal"):
        FeedbackIntake(
            observation=fresh,
            failure_class=RecoveryClass.PROTOCOL,
            goal_id="g1",
            executed_action_id="act-old",
            discarded_proposals=discarded,
        )

    # The guard does not touch the ledger.
    assert store.state.goals["g1"].status is GoalStatus.RUNNING


def test_intake_requires_a_planner_and_controller(tmp_path: Path) -> None:
    store = bootstrap_store(tmp_path / "runtime", [make_goal("g1")])
    with pytest.raises(RecoveryError, match="RecoveryPlanner"):
        RecoveryIntake(object(), RecoveryController(store))
    with pytest.raises(RecoveryError, match="RecoveryController"):
        RecoveryIntake(RecoveryPlanner(store), object())
