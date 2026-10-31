"""E05/E06 end-to-end on the local site with a real headless browser.

Scripted model + local rendered site; no real model, public network or official
platform. Each run records its own control events and evidence under the pytest
temporary runtime; ``capture_e2_evidence.py`` writes the same scenarios to
``runtime/phase2_e2/``.
"""

from __future__ import annotations

import pytest

from e2_helpers import run_l06, run_l07

from webagent.control import ControlState
from webagent.effgen_bridge import DriverStatus


def _feedback_ids(run) -> list[str]:
    return [
        event["action_id"]
        for event in run.events
        if event["event_type"] == "feedback"
    ]


def _event_types(run) -> list[str]:
    return [event["event_type"] for event in run.control_events]


def test_l06_login_overlay_pauses_without_a_feedback_turn_and_resumes(
    tmp_path,
) -> None:
    run = run_l06(tmp_path / "runtime", seed=101, auth_kind="login")
    try:
        assert run.step is not None and run.step.status is DriverStatus.COMPLETED
        assert run.source.user_action_error is None
        assert run.judge_result is not None and run.judge_result.passed
        assert run.verified is True

        # Nothing ran while the user owned the browser: no model call, no page
        # action and no session event after the handoff started.
        assert run.source.probe_at_handoff == run.source.probe_after_handoff
        assert run.source.probe_at_handoff["backend_executed_actions"] == 4

        types = _event_types(run)
        assert "takeover_detected" in types
        assert "pause_requested" in types and "draining" in types
        assert "handoff" in types and "resume_signal" in types
        takeover = next(
            event for event in run.control_events if event["event_type"] == "takeover_detected"
        )
        assert takeover["details"]["kind"] == "login"
        assert takeover["details"]["detection_source"] == "visible_observation_only"
        handoff = next(event for event in run.control_events if event["event_type"] == "handoff")
        assert handoff["details"]["browser_drained"] is True

        # The deferred feedback turn is truthful: the executed save is reported
        # after resume, and the proposal built on the pre-handoff observation is
        # dropped without being dispatched.
        assert len(run.coordinator.discarded_proposals) == 1
        dropped = run.coordinator.discarded_proposals[0]
        assert dropped["op"] == "scroll"
        assert dropped["executed"] is False and dropped["consumed"] is False
        assert dropped["action_id"] not in _feedback_ids(run)

        # Continue re-observes: the saved draft is visible in the fresh page and
        # the ledger fact records it; the later cancel clears the draft.
        resumed = next(
            event for event in run.control_events if event["event_type"] == "resumed"
        )
        assert resumed["page_epoch"] == 1
        saved_records = [
            record
            for record in run.runner.recorded_observations
            if record["fields"].get("draft_value")
            and record["observation_id"] == resumed["observation_id"]
        ]
        assert saved_records, "the resumed observation must show the saved draft"
        expected = str(run.prepared.case.task_params["l06_draft_text"])
        assert saved_records[0]["fields"]["draft_value"] == expected
        assert run.fact_fields["draft_ever_saved_value"] == expected
        assert run.fact_fields["draft_cleared"] is True

        snapshot = run.coordinator.snapshot().to_dict()
        assert snapshot["total_seconds"] == pytest.approx(
            snapshot["human_wait_seconds"] + snapshot["execution_seconds"], abs=0.01
        )
        assert snapshot["deadline_extended"] is False
        assert snapshot["platform_deadline_epoch"] == pytest.approx(
            run.session.deadline.epoch_seconds
        )
        assert run.reloaded_equal
    finally:
        run.prepared.stop()


def test_l06_verification_overlay_is_handled_the_same_way(tmp_path) -> None:
    run = run_l06(tmp_path / "runtime", seed=202, auth_kind="verification")
    try:
        assert run.step is not None and run.step.status is DriverStatus.COMPLETED
        assert run.source.user_action_error is None
        assert run.judge_result is not None and run.judge_result.passed
        assert run.source.probe_at_handoff == run.source.probe_after_handoff
        takeover = next(
            event for event in run.control_events if event["event_type"] == "takeover_detected"
        )
        assert takeover["details"]["kind"] == "verification"
        assert run.coordinator.snapshot().state == ControlState.RUNNING.value
    finally:
        run.prepared.stop()


def test_l07_user_pause_drops_proposal_and_continues_on_new_observation(tmp_path) -> None:
    run = run_l07(tmp_path / "runtime", seed=303)
    try:
        assert run.step is not None and run.step.status is DriverStatus.COMPLETED
        assert run.source.user_action_error is None
        assert run.judge_result is not None and run.judge_result.passed

        assert run.source.probe_at_handoff == run.source.probe_after_handoff
        types = _event_types(run)
        assert types.index("proposal_dropped") < types.index("handoff")
        assert "resume_signal" in types

        dropped = run.coordinator.discarded_proposals[0]
        assert dropped["op"] == "click"
        assert dropped["executed"] is False and dropped["consumed"] is False
        assert dropped["reason"].startswith("paused_before_dispatch")
        assert dropped["action_id"] not in _feedback_ids(run)

        executed = [
            event["action_id"]
            for event in run.events
            if event["event_type"] == "feedback"
        ]
        assert dropped["action_id"] not in executed
        resumed = next(
            event for event in run.control_events if event["event_type"] == "resumed"
        )
        after = [
            event
            for event in run.events
            if event["event_type"] == "dispatched"
            and event["observation_id"] == resumed["observation_id"]
        ]
        assert after, "a new action must be dispatched from the resumed observation"
        assert all(
            event["observation_id"] != dropped["observation_id"]
            for event in run.events
            if event["event_type"] == "dispatched"
        )
        assert dropped["observation_id"] != resumed["observation_id"]

        expected = str(run.prepared.case.task_params["l07_draft_text"])
        assert run.fact_fields["draft_ever_saved_value"] == expected
        snapshot = run.coordinator.snapshot().to_dict()
        assert snapshot["page_epoch"] == 1
        assert snapshot["human_wait_seconds"] > 0
        assert snapshot["total_seconds"] == pytest.approx(
            snapshot["human_wait_seconds"] + snapshot["execution_seconds"], abs=0.01
        )
        assert run.reloaded_equal
    finally:
        run.prepared.stop()
