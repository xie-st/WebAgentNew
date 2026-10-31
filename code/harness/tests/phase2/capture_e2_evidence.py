"""Capture offline E2 evidence under ``runtime/phase2_e2/``.

Scenarios:
* L06 seed 101: sign-in overlay during a saved draft; automatic takeover.
* L06 seed 202: verification overlay variant.
* L07 seed 303: explicit user pause, user page change, explicit resume.
* fake-clock timing reconciliation and restart-from-snapshot control evidence.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe code\\harness\\tests\\phase2\\capture_e2_evidence.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for entry in (str(ROOT / "code" / "harness" / "src"), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from e2_helpers import run_l06, run_l07  # noqa: E402

from webagent.budget import Budget, Deadline  # noqa: E402
from webagent.clock import FakeClock  # noqa: E402
from webagent.control import (  # noqa: E402
    ControlCoordinator,
    ControlEventLog,
    MemoryControlSource,
    restart_from_snapshot,
)
from webagent.protocol import ToolRequest  # noqa: E402
from webagent.session import Session  # noqa: E402
from webagent.state import EffectStatus, StateStore  # noqa: E402
from webagent.testing import FakeBackend, FakeElement, FakePage, FakeSite  # noqa: E402
from webagent.tools import build_web_action  # noqa: E402

OUTPUT = ROOT / "runtime" / "phase2_e2"


class AdvancingResumeSource(MemoryControlSource):
    def __init__(self, clock: FakeClock, seconds: float) -> None:
        super().__init__()
        self.clock = clock
        self.seconds = seconds
        self.sent = False

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        if not self.sent:
            self.sent = True
            self.clock.advance(self.seconds)
            return "resume"
        return None


def _run_summary(run: Any, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "case_id": run.prepared.case.case_id,
        "seed": run.prepared.seed,
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "target_id": run.target_id,
        "task": run.task,
        "step_status": None if run.step is None else run.step.status.value,
        "judge": None if run.judge_result is None else run.judge_result.to_dict(),
        "verified": run.verified,
        "control": run.coordinator.snapshot().to_dict(),
        "control_events": run.control_events,
        "dropped_proposals": list(run.coordinator.discarded_proposals),
        "wait_probe_before": run.source.probe_at_handoff,
        "wait_probe_after": run.source.probe_after_handoff,
        "user_action_error": (
            None
            if run.source.user_action_error is None
            else f"{type(run.source.user_action_error).__name__}: {run.source.user_action_error}"
        ),
        "fact_fields": run.fact_fields,
        "recorded_observations": run.runner.recorded_observations,
        "executed_actions": [
            {
                "action_id": event["action_id"],
                "action_type": event["action_type"],
                "observation_id": event["observation_id"],
                "result": event["result"],
            }
            for event in run.events
            if event["event_type"] == "feedback"
        ],
        "session_counts": run.session.result_summary(),
        "model_accounting": run.coordinator.accounting.snapshot(),
        "reloaded_equal": run.reloaded_equal,
        "state_before_judge": run.state_before_judge,
        "state_after": run.state_after,
    }


def capture_local_runs() -> dict[str, Any]:
    results: dict[str, Any] = {}

    run = run_l06(OUTPUT, seed=101, auth_kind="login", run_id="e2-l06-101-login")
    results["l06_login"] = _run_summary(run, "L06 login overlay")
    run.prepared.stop()

    run = run_l06(OUTPUT, seed=202, auth_kind="verification", run_id="e2-l06-202-verification")
    results["l06_verification"] = _run_summary(run, "L06 verification overlay")
    run.prepared.stop()

    run = run_l07(OUTPUT, seed=303, run_id="e2-l07-303-pause")
    results["l07_pause"] = _run_summary(run, "L07 explicit pause and page change")
    run.prepared.stop()
    return results


def capture_timing() -> dict[str, Any]:
    clock = FakeClock()
    deadline = clock.now() + 60
    source = AdvancingResumeSource(clock, 30)
    coordinator = ControlCoordinator(
        "e2-timing",
        clock=clock,
        source=source,
        event_log=ControlEventLog("e2-timing", OUTPUT / "e2-timing-control.jsonl", clock=clock),
        platform_deadline_epoch=deadline,
    )
    coordinator.start()
    clock.advance(2)
    coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()
    signal = coordinator.wait_for_user_signal()
    coordinator.begin_refresh()
    clock.advance(3)
    coordinator.finish_refresh()
    clock.advance(1)
    snapshot = coordinator.finish("completed").to_dict()
    coordinator.close()
    return {
        "signal": signal,
        "platform_deadline_epoch": deadline,
        "snapshot": snapshot,
        "reconciled": abs(
            snapshot["total_seconds"]
            - snapshot["human_wait_seconds"]
            - snapshot["execution_seconds"]
        )
        <= 1e-6,
    }


def capture_restart() -> dict[str, Any]:
    old_run = "e2-e08-old"
    new_run = "e2-e08-new"
    site = FakeSite(
        [
            FakePage(
                url="http://fixture.local/product/P-1",
                title="Fixture product",
                ax_text="A fixture product page.",
                elements=(FakeElement(role="button", name="Cancel draft"),),
            )
        ]
    )
    clock = FakeClock()
    backend = FakeBackend(old_run, site, start_url="http://fixture.local/product/P-1")
    session = Session(old_run, runtime_dir=OUTPUT, clock=clock)
    store = StateStore(OUTPUT, old_run, clock=clock)
    executed_action_id = ""
    old_executed = 0
    try:
        session.start_task("save then cancel the draft")
        observation = backend.observe()
        session.ingest_observation(observation)
        store.initialize("save then cancel the draft")
        action = build_web_action(ToolRequest("web_click", {"target": "1"}), observation)
        executed_action_id = action.action_id
        effect = store.prepare_effect(
            action_id=action.action_id,
            target="1",
            object_id="P-1",
            branch_id="root",
            observation_id=observation.observation_id,
            note="simulated write took effect before the interruption",
        )
        assert effect.status is EffectStatus.PENDING
        session.propose(action)
        session.consume(action.action_id)
        backend.execute(action)
        old_executed = len(backend.executed_actions)
    finally:
        backend.close()
        session.close()

    new_backend = FakeBackend(new_run, site, start_url="http://fixture.local/product/P-1")
    try:
        result = restart_from_snapshot(
            OUTPUT,
            old_run,
            branch_id="restart-1",
            new_run_id=new_run,
            observer=lambda: new_backend.observe(),
        )
    finally:
        new_backend.close()
    payload = result.to_dict()
    payload["pending_statuses"] = [effect.status.value for effect in result.pending_effects]
    payload["pending_action_id"] = executed_action_id
    payload["old_executed_actions"] = old_executed
    payload["new_executed_actions"] = len(new_backend.executed_actions)
    payload["replayed_actions"] = list(result.replayed_actions)
    payload["generator_restored"] = result.generator_restored
    return payload


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary = {
        "batch": "phase2-E2",
        "run_root": str(OUTPUT),
        "local_runs": capture_local_runs(),
        "timing": capture_timing(),
        "restart": capture_restart(),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "timing.json").write_text(
        json.dumps(summary["timing"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "restart.json").write_text(
        json.dumps(summary["restart"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUTPUT / 'summary.json'}")
    for key, value in summary["local_runs"].items():
        judge = value["judge"] or {}
        print(
            f"{key}: step={value['step_status']} judge={judge.get('verdict')} "
            f"dropped={len(value['dropped_proposals'])}"
        )
    print(
        "timing:",
        summary["timing"]["snapshot"]["total_seconds"],
        summary["timing"]["snapshot"]["human_wait_seconds"],
        summary["timing"]["snapshot"]["execution_seconds"],
        "reconciled=",
        summary["timing"]["reconciled"],
    )
    print(
        "restart:",
        summary["restart"]["branches"],
        "pending=",
        summary["restart"]["pending_statuses"],
        "generator_restored=",
        summary["restart"]["generator_restored"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
