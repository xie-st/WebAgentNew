from __future__ import annotations

from pathlib import Path

from webagent.clock import FakeClock
from webagent.protocol import FeedbackStatus
from webagent.recovery import (
    FailureEvidence,
    RecoveryClass,
    RecoveryController,
    RecoveryLimits,
    RecoveryPlanner,
    classify_failure,
    reverse_dependency_closure,
)

from e1_helpers import bootstrap_store, make_goal, verified_goal


def test_failure_classes_keep_unknown_distinct_from_transient() -> None:
    assert (
        classify_failure(status=FeedbackStatus.UNKNOWN, action_op="click")
        is RecoveryClass.UNKNOWN_EFFECT
    )
    assert (
        classify_failure(status=FeedbackStatus.ERROR, error_type="action_timeout")
        is RecoveryClass.TRANSIENT
    )
    assert (
        classify_failure(
            status=FeedbackStatus.ERROR,
            message="Sign in required",
            error_type="action_failed",
        )
        is RecoveryClass.AUTHENTICATION
    )
    assert (
        classify_failure(status=FeedbackStatus.ERROR, error_type="mystery")
        is RecoveryClass.UNKNOWN
    )


def test_earliest_evidenced_error_and_reverse_closure_are_used(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(
        runtime,
        [
            verified_goal(runtime, "g1"),
            make_goal("g2", status="running"),
            make_goal("g3", status="running", depends_on=("g2",)),
            make_goal("g4", status="running"),
        ],
    )
    # A synthetic evidence record for the earliest action is sufficient to
    # anchor the plan; no site database is read.
    failures = [
        FailureEvidence(
            action_id="act-late",
            sequence=9,
            failure_class=RecoveryClass.TRANSIENT,
            goal_id="g4",
            observation_id="obs-9",
        ),
        FailureEvidence(
            action_id="act-early",
            sequence=2,
            failure_class=RecoveryClass.TRANSIENT,
            goal_id="g2",
            observation_id="obs-2",
        ),
        FailureEvidence(
            action_id="act-no-evidence",
            sequence=1,
            failure_class=RecoveryClass.TRANSIENT,
            goal_id="g4",
        ),
    ]
    plan = RecoveryPlanner(store).plan(failures)
    assert plan.evidence_sufficient is True
    assert plan.anchor is not None and plan.anchor.anchor_id == "act-early"
    assert plan.affected_goal_ids == ("g2", "g3")
    assert plan.preserved_verified_goal_ids == ("g1",)
    assert reverse_dependency_closure(store.state, ["g2"]) == ("g2", "g3")
    assert "re-observing" in plan.next_step or "inspect" in plan.next_step


def test_insufficient_evidence_does_not_invent_a_cause(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", status="running")])
    plan = RecoveryPlanner(store).plan(
        [
            FailureEvidence(
                action_id="act-1",
                sequence=0,
                failure_class=RecoveryClass.UNKNOWN,
                goal_id="g1",
            )
        ]
    )
    assert plan.evidence_sufficient is False
    assert plan.affected_goal_ids == ()
    controller = RecoveryController(store)
    result = controller.apply(plan)
    assert result.applied is False
    assert store.state.revision == 1


def test_controller_invalidates_only_affected_nonverified_closure(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(
        runtime,
        [
            verified_goal(runtime, "g1"),
            make_goal("g2", status="awaiting_verification"),
            make_goal("g3", status="awaiting_verification", depends_on=("g2",)),
            make_goal("g4", status="running"),
        ],
    )
    failures = [
        FailureEvidence(
            action_id="act-g2",
            sequence=3,
            failure_class=RecoveryClass.CONDITION_FAILED,
            goal_id="g2",
            observation_id="obs-3",
        )
    ]
    planner = RecoveryPlanner(store)
    plan = planner.plan(failures)
    controller = RecoveryController(store)
    result = controller.apply(plan)
    assert result.applied is True
    assert store.state.goals["g2"].status.value == "invalidated"
    assert store.state.goals["g3"].status.value == "invalidated"
    assert store.state.goals["g1"].status.value == "verified"
    assert store.state.goals["g4"].status.value == "running"
    assert store.state.verified_goal_ids == ("g1",)


def test_recovery_per_goal_limit_and_fake_clock_deadline_really_stop_attempts(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", status="running")])
    clock = FakeClock()
    controller = RecoveryController(
        store,
        limits=RecoveryLimits(max_recoveries_per_goal=2, max_recoveries_total=5),
        clock=clock,
    )
    assert controller.authorize(["g1"]).allowed is True
    assert controller.authorize(["g1"]).allowed is True
    denied = controller.authorize(["g1"])
    assert denied.allowed is False
    assert "per-goal" in denied.reason

    deadline_controller = RecoveryController(
        store,
        limits=RecoveryLimits(max_recoveries_total=5, deadline_epoch=clock.now() + 5),
        clock=clock,
    )
    assert deadline_controller.authorize(["g1"]).allowed is True
    clock.advance(6)
    expired = deadline_controller.authorize(["g1"])
    assert expired.allowed is False
    assert "deadline" in expired.reason


def test_verified_affected_goal_is_reported_as_missing_ledger_support(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(runtime, [make_goal("g1", status="running")])
    state = store.state
    from e1_helpers import evidence_ref, make_observation

    observation = make_observation(runtime, observation_id="obs-v")
    store.verify_goal(
        "g1",
        evidence=(evidence_ref(runtime, observation),),
        base_revision=state.revision,
        plan_version=state.plan_version,
        source_branch="root",
    )
    failures = [
        FailureEvidence(
            action_id="act-g1",
            sequence=1,
            failure_class=RecoveryClass.CONDITION_FAILED,
            goal_id="g1",
            observation_id="obs-x",
        )
    ]
    plan = RecoveryPlanner(store).plan(failures)
    assert plan.invalidated_goal_ids == ("g1",)
    result = RecoveryController(store).apply(plan)
    assert result.applied is False
    assert "StateStore API" in result.reason
    assert store.state.goals["g1"].status.value == "verified"
