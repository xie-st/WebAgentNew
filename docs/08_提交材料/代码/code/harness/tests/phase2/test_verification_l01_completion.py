from __future__ import annotations

from pathlib import Path

from webagent.budget import Budget
from webagent.clock import FakeClock
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.models.fake import ScriptedModel
from webagent.session import Session
from webagent.state import Goal, GoalStatus, StateUpdate
from webagent.verification import (
    CompletionChecker,
    CompletionDecision,
    CompletionToolRuntime,
    GoalVerifier,
    VerificationCondition,
    VerificationRequest,
    VerificationStatus,
    VerificationTrigger,
    build_completion_tools,
)

from d3_helpers import run_l01
from e1_helpers import bootstrap_store, make_goal, make_observation


def _goal(goal_id: str, target_id: str) -> Goal:
    return Goal(
        goal_id=goal_id,
        requirement=f"L01 follow-up requirement {goal_id}",
        acceptance_criteria=("visible target page and screenshot",),
        object_scope=(target_id,),
        status=GoalStatus.RUNNING,
        completed_object_ids=(target_id,),
    )


def test_l01_writeback_path_pass_fail_unknown_and_partial_completion(tmp_path: Path) -> None:
    run = run_l01(tmp_path / "runtime", seed=101, run_id="e1-verify-l01")
    target = run.target_id
    observation = run.session.observation
    assert observation is not None
    store = run.store
    state = store.state
    store.apply_update(
        StateUpdate(
            base_revision=state.revision,
            plan_version=state.plan_version,
            source_branch="root",
            goals=(_goal("g2", target), _goal("g3", target), _goal("g4", target)),
            note="E1 L01 follow-up requirements",
        )
    )
    verifier = GoalVerifier(store)

    pass_request = VerificationRequest(
        goal_id="g2",
        subject_id=target,
        conditions=(
            VerificationCondition(
                "url",
                "the final page is the target product",
                "page_url_contains",
                expected=target,
            ),
            VerificationCondition("screenshot", "screenshot saved", "screenshot_saved"),
        ),
        observation=observation,
        trigger=VerificationTrigger.TASK_END,
        completed_object_ids=(target,),
    )
    pass_result = verifier.verify(pass_request)
    assert pass_result.status is VerificationStatus.PASS
    assert pass_result.verified is True

    fail_request = VerificationRequest(
        goal_id="g3",
        subject_id=target,
        conditions=(
            VerificationCondition(
                "url",
                "the final page is a different page",
                "page_url_contains",
                expected="definitely-not-the-target",
            ),
        ),
        observation=observation,
        trigger=VerificationTrigger.TASK_END,
        completed_object_ids=(target,),
    )
    fail_result = verifier.verify(fail_request)
    assert fail_result.status is VerificationStatus.FAIL
    assert fail_result.verified is False

    unknown_request = VerificationRequest(
        goal_id="g4",
        subject_id=target,
        conditions=(
            VerificationCondition(
                "tone",
                "the page tone is neutral",
                "semantic",
                expected="neutral",
            ),
        ),
        observation=observation,
        trigger=VerificationTrigger.TASK_END,
        completed_object_ids=(target,),
    )
    unknown_result = verifier.verify(unknown_request)
    assert unknown_result.status is VerificationStatus.UNKNOWN
    assert unknown_result.status is not VerificationStatus.FAIL
    assert unknown_result.verified is False

    checker = CompletionChecker(verifier)
    decision = checker.check(
        [pass_request, fail_request, unknown_request],
        commit=False,
        goal_ids=("g2", "g3", "g4"),
    )
    assert decision.accepted is False
    assert decision.remaining_goal_ids == ("g3", "g4")
    assert decision.remaining_conditions
    assert store.state.completion_status() == "incomplete"


def test_finish_gate_turns_a_completed_answer_into_honest_incomplete(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    clock = FakeClock()
    session = Session("run-gate", runtime_dir=runtime, clock=clock, budget=Budget(clock=clock))
    observation = make_observation(runtime, run_id="run-gate", observation_id="obs-gate")
    session.start_task("finish the task")
    session.ingest_observation(observation)
    denied = CompletionDecision(
        accepted=False,
        remaining_goal_ids=("g2",),
        remaining_conditions=("g2:the second object is verified",),
        reason="the second goal is not verified",
    )
    driver = create_driver(
        session,
        ScriptedModel(["Final Answer: partial work is done"]),
        completion_gate=lambda _step: denied,
    )
    step = driver.start()
    assert step.status is DriverStatus.INCOMPLETE
    assert "g2" in step.message
    assert session.termination is not None and session.termination.value == "incomplete"
    driver.close()
    session.close()


def test_request_finish_refusal_keeps_the_same_generator_alive(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    store = bootstrap_store(
        runtime, [make_goal("g2", scope=("P-TEST",), completed=("P-TEST",))]
    )
    session = Session(
        "run-loop",
        runtime_dir=runtime,
        clock=FakeClock(),
        budget=Budget(clock=FakeClock()),
    )
    observation = make_observation(
        runtime,
        run_id="run-loop",
        observation_id="obs-loop",
        values=(("textbox", "Review draft", "saved"),),
    )
    session.start_task("finish the task")
    session.ingest_observation(observation)
    verifier = GoalVerifier(store)
    checker = CompletionChecker(verifier)
    tool_runtime = CompletionToolRuntime(checker, lambda: [])
    model = ScriptedModel(
        [
            "Thought: check before answering.\nAction: request_finish\nAction Input: {}",
            "Thought: still not complete; check once more.\nAction: request_finish\nAction Input: {}",
            "Final Answer: reported partial progress",
        ]
    )
    driver = create_driver(
        session,
        model,
        extra_tools=build_completion_tools(tool_runtime),
        completion_gate=checker.gate([]),
    )
    step = driver.start()
    assert step.status is DriverStatus.INCOMPLETE
    assert model.call_count >= 3
    assert "g2" in step.message
    assert session.termination is not None and session.termination.value == "incomplete"
    driver.close()
    session.close()
