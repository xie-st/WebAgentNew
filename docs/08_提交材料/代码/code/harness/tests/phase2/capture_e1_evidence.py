"""Generate reviewable offline E1 evidence under ``runtime/phase2_e1/``.

Runs the real local site in headless Chromium with scripted fake models:
L04 lost feedback, L05 affected-chain recovery, an L01 verification write-back,
and a restart after an effective save. Also records fake-clock stagnation and
recovery-limit checks. No real model, public site or official platform is used.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "code" / "harness" / "src"
PHASE2 = Path(__file__).resolve().parent
for path in (str(SRC), str(PHASE2)):
    if path not in sys.path:
        sys.path.insert(0, path)

from webagent.backends.playwright import PlaywrightBackend  # noqa: E402
from webagent.budget import Budget  # noqa: E402
from webagent.clock import FakeClock  # noqa: E402
from webagent.evidence import EvidenceStore  # noqa: E402
from webagent.models.fake import ScriptedModel  # noqa: E402
from webagent.session import Session  # noqa: E402
from webagent.recovery import (  # noqa: E402
    RecoveryController,
    RecoveryLimits,
    RestartRecovery,
    StagnationDetector,
)
from webagent.state import EffectStatus, Goal, GoalStatus, StateStore, StateUpdate  # noqa: E402
from webagent.verification import (  # noqa: E402
    CompletionChecker,
    GoalVerifier,
    ModelSemanticJudge,
    VerificationCondition,
    VerificationRequest,
    VerificationTrigger,
)

from d3_helpers import run_l01  # noqa: E402
from e1_helpers import (  # noqa: E402
    bootstrap_store,
    evidence_ref,
    interrupt_l04_after_effective_save,
    judge_l04,
    make_goal,
    page_with_value,
    run_l04,
    run_l05,
)


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _l01_verification(runtime: Path, out: Path) -> dict[str, object]:
    run = run_l01(runtime, seed=101, run_id="e1-l01-101")
    target = run.target_id
    observation = run.session.observation
    assert observation is not None
    store = run.store
    state = store.state

    def goal(goal_id: str) -> Goal:
        return Goal(
            goal_id=goal_id,
            requirement=f"capture requirement {goal_id}",
            acceptance_criteria=("target page and screenshot",),
            object_scope=(target,),
            status=GoalStatus.RUNNING,
            completed_object_ids=(target,),
        )

    store.apply_update(
        StateUpdate(
            base_revision=state.revision,
            plan_version=state.plan_version,
            source_branch="root",
            goals=(goal("g2"), goal("g3"), goal("g4")),
            note="E1 capture follow-up requirements",
        )
    )
    verifier = GoalVerifier(store)
    pass_request = VerificationRequest(
        goal_id="g2",
        subject_id=target,
        conditions=(
            VerificationCondition("url", "target product URL", "page_url_contains", expected=target),
            VerificationCondition("shot", "screenshot saved", "screenshot_saved"),
        ),
        observation=observation,
        trigger=VerificationTrigger.TASK_END,
        completed_object_ids=(target,),
    )
    fail_request = VerificationRequest(
        goal_id="g3",
        subject_id=target,
        conditions=(
            VerificationCondition("url", "different URL", "page_url_contains", expected="missing-page"),
        ),
        observation=observation,
        trigger=VerificationTrigger.TASK_END,
        completed_object_ids=(target,),
    )
    unknown_request = VerificationRequest(
        goal_id="g4",
        subject_id=target,
        conditions=(
            VerificationCondition("tone", "neutral tone", "semantic", expected="neutral"),
        ),
        observation=observation,
        trigger=VerificationTrigger.TASK_END,
        completed_object_ids=(target,),
    )
    results = [
        verifier.verify(pass_request),
        verifier.verify(fail_request),
        verifier.verify(unknown_request),
    ]
    decision = CompletionChecker(verifier).check(
        [pass_request, fail_request, unknown_request],
        commit=False,
        goal_ids=("g2", "g3", "g4"),
    )
    payload = {
        "run_id": run.run_id,
        "target_id": target,
        "judge": None if run.judge_result is None else run.judge_result.to_dict(),
        "verification": [result.to_dict() for result in results],
        "completion_refused": decision.to_dict(),
        "event_path": str(run.session.evidence.events_path),
        "state": store.state.to_dict(),
    }
    _dump(out / "l01_verification.json", payload)
    return payload


def _restart(runtime: Path, out: Path) -> dict[str, object]:
    interrupted = interrupt_l04_after_effective_save(
        runtime, seed=101, run_id="e1-restart-101"
    )
    store = StateStore(runtime, interrupted.run_id)
    store.load()
    interrupted.prepared.site.start()
    evidence = EvidenceStore(runtime, "e1-restart-branch")
    backend = PlaywrightBackend(
        "e1-restart-branch", evidence=evidence, headless=True, runtime_dir=runtime
    )
    try:
        backend.start(f"{interrupted.prepared.base_url}/product/{interrupted.target_id}")
        restart = RestartRecovery(store).recover(
            new_branch_id="recovery-1",
            new_run_id="e1-restart-branch",
            observer=lambda: backend.observe(),
        )
        judge = judge_l04(
            interrupted.prepared.db_path,
            target_id=interrupted.target_id,
            expected_text=interrupted.note,
        )
    finally:
        backend.close()
        interrupted.prepared.stop()
    payload = {
        "saved_action_id": interrupted.save_action_id,
        "restart": restart.to_dict(),
        "judge": judge.to_dict(),
        "snapshot_pending_status": store.state.pending_effects[
            interrupted.save_action_id
        ].status.value,
    }
    _dump(out / "restart.json", payload)
    return payload


def _stagnation_fake_clock(out: Path) -> dict[str, object]:
    clock = FakeClock()
    detector = StagnationDetector(window=6, clock=clock)
    for _ in range(6):
        detector.record(
            action_key="click-save",
            state_key="same-page",
            observation_id=f"obs-{len(detector.records)}",
        )
    signal = detector.evaluate()
    progressed = StagnationDetector(window=6, clock=clock)
    for _ in range(5):
        progressed.record(
            action_key="click-save",
            state_key="same-page",
            observation_id=f"p-{len(progressed.records)}",
        )
    progressed.record(
        action_key="click-save",
        state_key="same-page",
        observation_id="p-new",
        new_evidence_ids=("ev-new",),
    )

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        store = bootstrap_store(
            Path(directory) / "runtime",
            [make_goal("g1", status="awaiting_verification")],
        )
        controller = RecoveryController(
            store,
            limits=RecoveryLimits(max_recoveries_per_goal=2, max_recoveries_total=5),
            clock=clock,
        )
        attempts = [controller.authorize(["g1"]).allowed for _ in range(3)]
        deadline_controller = RecoveryController(
            store,
            limits=RecoveryLimits(max_recoveries_total=5, deadline_epoch=clock.now() + 5),
            clock=clock,
        )
        deadline_before = deadline_controller.authorize(["g1"]).allowed
        clock.advance(6)
        deadline_after = deadline_controller.authorize(["g1"]).allowed
    payload = {
        "signal_after_six_no_progress": None if signal is None else signal.to_dict(),
        "same_action_with_new_evidence_signal": progressed.evaluate(),
        "attempts_allowed_then_denied": attempts,
        "deadline_attempt_before": deadline_before,
        "deadline_attempt_after": deadline_after,
    }
    _dump(out / "fake_clock.json", payload)
    return payload


def _semantic_billing(runtime: Path, out: Path) -> dict[str, object]:
    import json as _json

    observation = page_with_value(
        runtime,
        "Harbor Garden profile",
        run_id="e1-semantic",
        observation_id="semantic-obs-1",
    )
    ref = evidence_ref(runtime, observation)
    store = bootstrap_store(
        runtime,
        [make_goal("g1", scope=("P-TEST",), completed=("P-TEST",))],
        run_id="e1-semantic",
    )
    clock = FakeClock()
    session = Session(
        "e1-semantic", runtime_dir=runtime, clock=clock, budget=Budget(clock=clock)
    )
    session.start_task("verify one semantic condition")
    session.ingest_observation(observation)
    model = ScriptedModel(
        [
            _json.dumps(
                {
                    "status": "pass",
                    "reason": "the visible note is a profile note",
                    "evidence_ids": [ref.evidence_id],
                }
            )
        ]
    )
    judge = ModelSemanticJudge(model, session=session)
    result = GoalVerifier(store, semantic_judge=judge).verify(
        VerificationRequest(
            goal_id="g1",
            subject_id="P-TEST",
            conditions=(
                VerificationCondition(
                    "profile_kind",
                    "the visible note is a product profile note",
                    "semantic",
                    expected="profile note",
                    evidence_ids=(ref.evidence_id,),
                ),
            ),
            observation=observation,
            evidence_scope=(ref,),
            trigger=VerificationTrigger.SUBGOAL_END,
            completed_object_ids=("P-TEST",),
        )
    )
    payload = {
        "verification": result.to_dict(),
        "model_calls_total": session.logger.model_calls_total,
        "budget_model_calls": session.budget.model_calls,
        "semantic_judge_calls": judge.calls,
    }
    session.close()
    _dump(out / "semantic_billing.json", payload)
    return payload


def main() -> int:
    runtime = ROOT / "runtime" / "phase2_e1"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    out = runtime / "evidence"
    out.mkdir(parents=True, exist_ok=True)

    l04 = run_l04(runtime, seed=101, run_id="e1-l04-101")
    l05 = run_l05(runtime, seed=101, run_id="e1-l05-101")
    l01 = _l01_verification(runtime, out)
    restart = _restart(runtime, out)
    fake = _stagnation_fake_clock(out)
    semantic = _semantic_billing(runtime, out)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": "offline local site + headless Chromium + scripted fake model",
        "seeds": [101],
        "l04": {
            "run_id": l04.run_id,
            "judge": l04.judge.to_dict(),
            "lost_feedback": l04.lost_feedback,
            "resolution": None if l04.resolution is None else l04.resolution.to_dict(),
            "verification": None if l04.verification is None else l04.verification.to_dict(),
            "save_count": l04.save_count,
            "completion_status": l04.store.state.completion_status(),
            "run_dir": str(l04.run_dir),
            "events": str(l04.run_dir / "events.jsonl"),
        },
        "l05": {
            "run_id": l05.run_id,
            "judge": l05.judge.to_dict(),
            "recovery": [item.to_dict() for item in l05.recovery],
            "verifications": [item.to_dict() for item in l05.verifications],
            "failed_object": l05.failed_object,
            "completion_status": l05.store.state.completion_status(),
            "run_dir": str(l05.run_dir),
            "events": str(l05.run_dir / "events.jsonl"),
        },
        "l01": {
            "run_id": l01["run_id"],
            "statuses": [item["status"] for item in l01["verification"]],
            "completion_refused": l01["completion_refused"]["accepted"] is False,
        },
        "restart": {
            "pending_status": restart["snapshot_pending_status"],
            "generator_stack_restored": restart["restart"]["generator_stack_restored"],
            "resent_action_ids": restart["restart"]["resent_action_ids"],
            "branch": restart["restart"]["new_branch_id"],
        },
        "fake_clock": fake,
        "semantic_billing": {
            "status": semantic["verification"]["status"],
            "model_calls_total": semantic["model_calls_total"],
            "budget_model_calls": semantic["budget_model_calls"],
        },
        "evidence_dir": str(out),
    }
    _dump(runtime / "summary.json", summary)
    _dump(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
