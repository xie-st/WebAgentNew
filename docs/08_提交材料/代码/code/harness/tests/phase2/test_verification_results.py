from __future__ import annotations

import json
from pathlib import Path

from webagent.budget import Budget, Deadline
from webagent.clock import FakeClock
from webagent.models.fake import ScriptedModel
from webagent.session import Session
from webagent.verification import (
    CompletionChecker,
    CompletionToolRuntime,
    GoalVerifier,
    ModelSemanticJudge,
    RequestFinishTool,
    VerificationCondition,
    VerificationRequest,
    VerificationStatus,
    VerificationTrigger,
    build_completion_tools,
)

from e1_helpers import (
    bootstrap_store,
    evidence_ref,
    make_goal,
    page_with_value,
    value_element,
)


def _request(
    observation,
    *,
    goal_id: str = "g1",
    value: str = "Harbor Garden profile",
    conditions=None,
    evidence_scope=(),
    trigger: VerificationTrigger = VerificationTrigger.TASK_END,
):
    return VerificationRequest(
        goal_id=goal_id,
        subject_id="P-TEST",
        conditions=tuple(
            conditions
            or (
                VerificationCondition(
                    "value",
                    "the visible profile value equals the saved note",
                    "visible_value_equals",
                    expected=value,
                    role="textbox",
                    name="Review draft",
                ),
                VerificationCondition(
                    "screenshot",
                    "a screenshot was saved for the observation",
                    "screenshot_saved",
                ),
            )
        ),
        observation=observation,
        evidence_scope=evidence_scope,
        trigger=trigger,
    )


def test_all_conditions_pass_commits_verified_with_real_evidence(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "Harbor Garden profile")
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
    )
    verifier = GoalVerifier(store)
    result = verifier.verify(_request(observation))
    assert result.status is VerificationStatus.PASS
    assert result.verified is True
    assert result.model_calls == 0
    assert result.evidence
    assert store.state.goals["g1"].status.value == "verified"
    assert store.state.completion_status() == "complete"
    assert {item.condition_id for item in result.conditions} == {"value", "screenshot"}


def test_failed_condition_does_not_verify_or_change_revision(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "a different value")
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
    )
    before = store.state.revision
    result = GoalVerifier(store).verify(_request(observation))
    assert result.status is VerificationStatus.FAIL
    assert result.verified is False
    assert result.failed_condition_ids() == ("value",)
    assert store.state.revision == before
    assert store.state.goals["g1"].status.value != "verified"


def test_unknown_semantic_without_judge_is_distinct_from_fail(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "Harbor Garden profile")
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
    )
    condition = VerificationCondition(
        "tone",
        "the visible note is neutral in tone",
        "semantic",
        expected="neutral tone",
        evidence_ids=("ev-obs-1-screenshot",),
    )
    request = _request(
        observation,
        conditions=(condition,),
        evidence_scope=(evidence_ref(runtime, observation),),
    )
    result = GoalVerifier(store).verify(request)
    assert result.status is VerificationStatus.UNKNOWN
    assert result.status is not VerificationStatus.FAIL
    assert result.unknown_condition_ids() == ("tone",)
    assert result.verified is False
    assert store.state.goals["g1"].status.value != "verified"


def test_semantic_model_is_billed_and_its_pass_fail_unknown_are_kept(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "Harbor Garden profile")
    ref = evidence_ref(runtime, observation)
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
    )
    condition = VerificationCondition(
        "relevant",
        "the note is a product profile note",
        "semantic",
        expected="profile note",
        evidence_ids=(ref.evidence_id,),
    )
    request = _request(
        observation,
        conditions=(condition,),
        evidence_scope=(ref,),
    )

    for answer, expected in (
        (
            json.dumps({"status": "pass", "reason": "it is a profile", "evidence_ids": [ref.evidence_id]}),
            VerificationStatus.PASS,
        ),
        (json.dumps({"status": "fail", "reason": "wrong kind"}), VerificationStatus.FAIL),
        ("not a json answer", VerificationStatus.UNKNOWN),
    ):
        clock = FakeClock()
        session = Session("run-a", runtime_dir=runtime, clock=clock, budget=Budget(clock=clock))
        session.start_task("verify a profile note")
        session.ingest_observation(observation)
        judge = ModelSemanticJudge(ScriptedModel([answer]), session=session)
        result = GoalVerifier(store, semantic_judge=judge).verify(
            request, commit=False
        )
        assert result.status is expected
        assert result.model_calls == 1
        assert judge.calls == 1
        assert session.logger.model_calls_total == 1
        assert session.budget.model_calls == 1
        session.close()


def test_screenshot_success_only_satisfies_the_screenshot_condition(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "Harbor Garden profile")
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
    )
    screenshot = VerificationCondition("shot", "screenshot exists", "screenshot_saved")
    text = VerificationCondition(
        "text",
        "the text inside the screenshot matches",
        "semantic",
        expected="the screenshot text",
    )
    result = GoalVerifier(store).verify(
        _request(observation, conditions=(screenshot, text), evidence_scope=())
    )
    by_id = {item.condition_id: item for item in result.conditions}
    assert by_id["shot"].status is VerificationStatus.PASS
    assert by_id["text"].status is VerificationStatus.UNKNOWN
    assert result.status is VerificationStatus.UNKNOWN
    assert result.verified is False


def test_verification_does_not_download_or_capture_again(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "other")
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
    )
    run_dir = runtime / "runs" / "run-a"
    before = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))
    GoalVerifier(store).verify(_request(observation))
    after = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*"))
    assert before == after


def test_partial_completion_is_refused_with_remaining_requirements(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "Harbor Garden profile")
    ref = evidence_ref(runtime, observation)
    store = bootstrap_store(
        runtime,
        [
            make_goal("g1", status="verified", evidence=(ref,)),
            make_goal("g2", scope=("P-TEST",), completed=("P-TEST",)),
            make_goal("g3"),
        ],
    )
    checker = CompletionChecker(GoalVerifier(store))
    decision = checker.check(
        [_request(observation, goal_id="g2")],
        commit=False,
    )
    assert decision.accepted is False
    assert decision.remaining_goal_ids == ("g2", "g3") or decision.remaining_goal_ids == ("g3",)
    assert "g3" in decision.remaining_goal_ids
    assert decision.state_completion_status == "incomplete"

    runtime_tool = CompletionToolRuntime(checker, lambda: [_request(observation, goal_id="g2")], commit=False)
    tool = RequestFinishTool(runtime_tool)
    tool_result = tool.execute()
    assert tool_result.success is False
    payload = json.loads(tool_result.output)
    assert payload["accepted"] is False
    assert "g3" in payload["remaining_goal_ids"]
    assert len(build_completion_tools(runtime_tool)) == 1


def test_completion_accepts_only_after_every_goal_is_verified(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    observation = page_with_value(runtime, "Harbor Garden profile")
    store = bootstrap_store(
        runtime,
        [
            make_goal("g1", scope=("P-TEST",), completed=("P-TEST",)),
            make_goal("g2", scope=("P-TEST",), completed=("P-TEST",)),
        ],
    )
    checker = CompletionChecker(GoalVerifier(store))
    requests = [_request(observation, goal_id="g1"), _request(observation, goal_id="g2")]
    decision = checker.check(requests, commit=True)
    assert decision.accepted is True
    assert decision.remaining_goal_ids == ()
    assert store.state.completion_status() == "complete"
