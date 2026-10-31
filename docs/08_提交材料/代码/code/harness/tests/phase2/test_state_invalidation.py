"""E03: the programmatic withdrawal of an already verified completion claim.

Recovery needs one ledger transition the model-facing tools never reach. These
tests pin the contract: status change, preserved requirement/objects, old and new
evidence sources, a queryable audit record, idempotent repetition and errors for
invalid input. Nothing here claims that the production recovery controller already
consumes the entry; that wiring is the later R-wave step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from state_helpers import goal, new_store, ref_for

from webagent.clock import FakeClock
from webagent.events import EventLogger, read_events
from webagent.state import (
    EvidenceReferenceError,
    GoalStatus,
    StateStore,
    StateToolRuntime,
    StateUpdate,
    StateValidationError,
    STATE_TOOL_NAMES,
    read_invalidations,
)


def _verified_store(runtime: Path, *, scope: tuple[str, ...] = ("obj-1",)):
    store = new_store(runtime)
    old = ref_for(
        runtime,
        evidence_id="ev-old",
        payload=b"<html><body>old support</body></html>",
        observation_id="obs-old",
    )
    store.apply_update(
        StateUpdate(base_revision=0, plan_version=0, goals=(goal(scope=scope),))
    )
    state = store.state
    store.verify_goal(
        "g1",
        evidence=(old,),
        base_revision=state.revision,
        plan_version=state.plan_version,
        source_branch="root",
    )
    return store, old


def _conflict(runtime: Path, evidence_id: str = "ev-conflict"):
    return ref_for(
        runtime,
        evidence_id=evidence_id,
        payload=b"<html><body>value changed</body></html>",
        observation_id="obs-conflict",
    )


def test_verified_goal_is_invalidated_and_keeps_both_sources(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, old = _verified_store(runtime)
    before = store.state.revision
    conflict = _conflict(runtime)

    result = store.invalidate_goal(
        "g1",
        "the visible saved value changed after the earlier check",
        (conflict,),
        base_revision=before,
        plan_version=0,
    )

    assert result.changed is True
    assert result.from_status is GoalStatus.VERIFIED
    assert result.to_status is GoalStatus.INVALIDATED
    assert result.state_revision_before == before
    assert result.state_revision_after == before + 1
    assert result.previous_evidence_ids == ("ev-old",)
    assert result.appended_evidence_ids == ("ev-conflict",)
    assert result.evidence_ids == ("ev-old", "ev-conflict")
    assert result.plan_version == 0
    assert result.source_branch == "root"

    withdrawn = store.state.goals["g1"]
    assert withdrawn.status is GoalStatus.INVALIDATED
    assert [ref.evidence_id for ref in withdrawn.evidence] == ["ev-old", "ev-conflict"]
    assert withdrawn.completed_object_ids == ("obj-1",)
    assert withdrawn.requirement == "collect the requested objects"
    assert withdrawn.acceptance_criteria == ("each object is present",)
    assert store.state.verified_goal_ids == ()
    assert store.state.completion_status() == "incomplete"

    # The record is not deleted on withdrawal.
    assert "g1" in store.state.goals

    records = store.read_invalidations()
    assert len(records) == 1
    record = records[0]
    assert record.goal_id == "g1"
    assert record.reason == "the visible saved value changed after the earlier check"
    assert record.from_status is GoalStatus.VERIFIED
    assert record.to_status is GoalStatus.INVALIDATED
    assert record.evidence_ids == ("ev-old", "ev-conflict")
    assert read_invalidations(store.state_dir) == records

    reloaded = StateStore(runtime, "run-a").load()
    assert reloaded.goals["g1"].status is GoalStatus.INVALIDATED
    assert [ref.evidence_id for ref in reloaded.goals["g1"].evidence] == [
        "ev-old",
        "ev-conflict",
    ]
    assert reloaded.completion_status() == "incomplete"
    assert StateStore(runtime, "run-a").read_invalidations() == records


def test_withdrawal_without_new_evidence_awaits_verification(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, old = _verified_store(runtime)

    result = store.invalidate_goal(
        "g1",
        "the earlier claim cannot be trusted yet; no cause is supported",
    )

    assert result.changed is True
    assert result.to_status is GoalStatus.AWAITING_VERIFICATION
    assert result.appended_evidence_ids == ()
    withdrawn = store.state.goals["g1"]
    assert withdrawn.status is GoalStatus.AWAITING_VERIFICATION
    assert withdrawn.evidence == (old,)
    assert withdrawn.completed_object_ids == ("obj-1",)
    assert store.state.verified_goal_ids == ()

    explicit = store.invalidate_goal(
        "g1",
        "keep the claim withdrawn until the condition is checked again",
        to_status="awaiting_verification",
        base_revision=result.state_revision_after,
    )
    assert explicit.changed is False
    assert explicit.to_status is GoalStatus.AWAITING_VERIFICATION
    assert store.state.revision == result.state_revision_after


def test_awaiting_goal_can_be_invalidated_with_conflicting_evidence(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(
        StateUpdate(
            base_revision=0,
            plan_version=0,
            goals=(goal(status="awaiting_verification", scope=("obj-1",)),),
        )
    )
    conflict = _conflict(runtime)
    before = store.state.revision

    result = store.invalidate_goal(
        "g1", "the check returned a contradicting value", (conflict,)
    )

    assert result.changed is True
    assert result.from_status is GoalStatus.AWAITING_VERIFICATION
    assert result.to_status is GoalStatus.INVALIDATED
    assert store.state.goals["g1"].status is GoalStatus.INVALIDATED
    assert store.state.revision == before + 1


def test_explicit_invalidation_without_conflicting_evidence_is_refused(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)
    before = store.state.to_dict()

    with pytest.raises(StateValidationError, match="conflicting evidence"):
        store.invalidate_goal(
            "g1", "an unsupported cause", to_status="invalidated"
        )

    assert store.state.to_dict() == before
    assert store.read_invalidations() == ()


def test_repeated_withdrawal_is_idempotent_and_not_counted_again(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)
    conflict = _conflict(runtime)
    first = store.invalidate_goal("g1", "value changed", (conflict,))
    snapshot = store.state.to_dict()

    second = store.invalidate_goal("g1", "value changed", (conflict,))

    assert second.changed is False
    assert second.reason == first.reason
    assert second.recorded_at == first.recorded_at
    assert second.state_revision_before == first.state_revision_before
    assert second.state_revision_after == first.state_revision_after
    assert store.state.to_dict() == snapshot
    assert store.state.revision == first.state_revision_after
    assert len(store.read_invalidations()) == 1

    # An already invalidated claim is not withdrawn twice, even by a call that
    # names no further evidence: it stays a revision-neutral repeat.
    third = store.invalidate_goal("g1", "the claim is already withdrawn")
    assert third.changed is False
    assert third.to_status is GoalStatus.INVALIDATED
    assert third.state_revision_after == first.state_revision_after
    assert store.state.revision == first.state_revision_after
    assert len(store.read_invalidations()) == 1


def test_unknown_goal_and_empty_reason_are_refused(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)
    conflict = _conflict(runtime)
    before = store.state.to_dict()

    with pytest.raises(StateValidationError, match="unknown goal"):
        store.invalidate_goal("missing", "no such requirement", (conflict,))
    for bad_reason in ("", "   ", None):
        with pytest.raises(StateValidationError, match="reason"):
            store.invalidate_goal("g1", bad_reason, (conflict,))

    assert store.state.to_dict() == before
    assert store.read_invalidations() == ()


def test_non_completion_claim_cannot_be_withdrawn(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = new_store(runtime)
    store.apply_update(StateUpdate(base_revision=0, plan_version=0, goals=(goal(),)))
    conflict = _conflict(runtime)

    with pytest.raises(StateValidationError, match="completion claim"):
        store.invalidate_goal("g1", "pending work is not a completion claim", (conflict,))

    assert store.state.goals["g1"].status is GoalStatus.PENDING
    assert store.read_invalidations() == ()


def test_bad_target_or_bad_evidence_is_refused(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)
    before = store.state.to_dict()

    with pytest.raises(StateValidationError, match="one of"):
        store.invalidate_goal(
            "g1", "wrong target", (_conflict(runtime),), to_status="blocked"
        )
    with pytest.raises(StateValidationError, match="EvidenceRef"):
        store.invalidate_goal("g1", "wrong evidence shape", ["ev-old"])
    foreign = ref_for(runtime, run_id="run-b", evidence_id="ev-foreign")
    with pytest.raises(EvidenceReferenceError, match="does not belong"):
        store.invalidate_goal("g1", "evidence from another run", (foreign,))

    assert store.state.to_dict() == before
    assert store.read_invalidations() == ()


def test_withdrawal_writes_a_queryable_recovery_event(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)
    clock = FakeClock()
    logger = EventLogger("run-a", runtime / "events.jsonl", clock=clock)
    store.attach_event_log(logger)
    try:
        result = store.invalidate_goal(
            "g1", "the saved value changed", (_conflict(runtime),)
        )
    finally:
        logger.close()

    events = read_events(runtime / "events.jsonl")
    recovery_events = [event for event in events if event["event_type"] == "recovery"]
    assert len(recovery_events) == 1
    event = recovery_events[0]
    assert event["goal_id"] == "g1"
    assert event["state_revision"] == result.state_revision_after
    assert event["result"]["invalidation"]["to_status"] == "invalidated"
    assert (
        event["result"]["invalidation"]["reason"] == "the saved value changed"
    )


def test_reverification_after_withdrawal_keeps_old_and_new_sources(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)
    first = store.invalidate_goal("g1", "value changed", (_conflict(runtime),))
    recheck = ref_for(
        runtime,
        evidence_id="ev-recheck",
        payload=b"<html><body>rechecked value</body></html>",
        observation_id="obs-recheck",
    )
    state = store.state

    store.verify_goal(
        "g1",
        evidence=(recheck,),
        base_revision=state.revision,
        plan_version=state.plan_version,
        source_branch="root",
    )

    verified = store.state.goals["g1"]
    assert verified.status is GoalStatus.VERIFIED
    assert [ref.evidence_id for ref in verified.evidence] == [
        "ev-old",
        "ev-conflict",
        "ev-recheck",
    ]
    assert verified.completed_object_ids == ("obj-1",)
    assert store.state.revision == first.state_revision_after + 1
    assert store.state.completion_status() == "complete"


def test_model_tools_do_not_expose_withdrawal(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store, _ = _verified_store(runtime)

    assert STATE_TOOL_NAMES == frozenset({"state_read", "state_update", "plan_update"})
    assert not hasattr(StateToolRuntime(store), "invalidate_goal")
