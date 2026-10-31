"""Capture D3 end-to-end evidence under ``runtime/phase2_d3``.

Offline only: headless Chromium against the local site, the scripted public-plan
model, and the local judge. No real model, official platform or public website is
contacted. Run from the repository root::

    .\\.venv\\Scripts\\python.exe code\\harness\\tests\\phase2\\capture_d3_evidence.py
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from d3_helpers import run_l01  # noqa: E402


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _run_summary(run) -> dict:
    checks = None if run.judge_result is None else [
        check.to_dict() for check in run.judge_result.checks
    ]
    effect_action_id = (
        run.runner.effects_prepared[0].action_id
        if run.runner.effects_prepared
        else None
    )
    effect_events = [
        {
            "event_type": event["event_type"],
            "action_id": event["action_id"],
            "state_revision": event["state_revision"],
            "goal_id": event["goal_id"],
            "branch_id": event["branch_id"],
            "plan_version": event["plan_version"],
            "result": event["result"],
        }
        for event in run.events
        if effect_action_id is not None and event.get("action_id") == effect_action_id
    ]
    return {
        "case_id": "L01",
        "seed": run.prepared.seed,
        "run_id": run.run_id,
        "target": {
            "id": run.target_id,
            "name": run.target_name,
            "sku": run.target_sku,
            "detail_url": run.target_page_url,
            "selected_by": "public product filters only, not hidden judge truth",
        },
        "run_dir": str(run.run_dir),
        "state_path": str(run.run_dir / "state" / "state.json"),
        "events_path": str(run.session.evidence.events_path),
        "judge_path": str(run.run_dir / "judge_result.json"),
        "driver_status": None if run.step is None else run.step.status.value,
        "termination": (
            None
            if run.session.finish_result is None
            else run.session.finish_result.reason.value
        ),
        "submission_count": run.prepared.site.submission_count(),
        "before_judge": {
            "revision": run.state_before_judge["revision"],
            "plan_version": run.state_before_judge["plan_version"],
            "goal_status": run.state_before_judge["goals"]["g1"]["status"],
            "verified_goal_ids": run.view_before_judge["verified_goal_ids"],
            "completion_status": run.view_before_judge["completion_status"],
            "object_progress": run.object_progress_before_judge,
            "fact_revisions": {
                key: value["revision"]
                for key, value in run.state_before_judge["facts"].items()
            },
        },
        "effects": [
            {
                "prepared": effect.to_dict(),
                "confirmed": run.runner.effects_confirmed[index].to_dict(),
            }
            for index, effect in enumerate(run.runner.effects_prepared)
        ],
        "observations": {
            "detail_page_observations": len(run.detail_observations),
            "distinct_objects": sorted(set(run.detail_observations)),
            "executed_actions": len(run.executed_action_ids),
        },
        "judge": None if run.judge_result is None else run.judge_result.to_dict(),
        "judge_checks": checks,
        "verified_writeback": run.verified,
        "after": {
            "revision": run.state_after["revision"],
            "plan_version": run.state_after["plan_version"],
            "goal_status": run.state_after["goals"]["g1"]["status"],
            "verified_goal_ids": run.view_after["verified_goal_ids"],
            "completion_status": run.view_after["completion_status"],
        },
        "snapshot_reload_equal": run.reload_equal,
        "effect_action_events": effect_events,
        "event_state_fields": {
            event["event_type"]: {
                "state_revision": event["state_revision"],
                "goal_id": event["goal_id"],
                "branch_id": event["branch_id"],
                "plan_version": event["plan_version"],
            }
            for event in run.events
            if event["event_type"] in {"task_start", "proposed", "dispatched", "feedback", "agent_claim"}
        },
        "environment": {
            "playwright": importlib.metadata.version("playwright"),
            "browser": "headless Chromium",
            "site": "local Harbor Market (127.0.0.1, isolated SQLite)",
            "model": "scripted public-plan fake model",
        },
    }


def _kill_summary(run) -> dict:
    effect = run.runner.effects_prepared[0]
    before_events = len(run.events)
    return {
        "case_id": "L01",
        "seed": run.prepared.seed,
        "run_id": run.run_id,
        "killed": run.killed,
        "kill_error": None if run.kill_error is None else str(run.kill_error),
        "run_dir": str(run.run_dir),
        "state_path": str(run.run_dir / "state" / "state.json"),
        "pending_effect": effect.to_dict(),
        "executed_action_ids": run.executed_action_ids,
        "effect_action_dispatched_feedback": effect.action_id in run.executed_action_ids,
        "state_after": run.state_after,
        "snapshot_reload_equal": run.reload_equal,
        "reloaded_state_revision": run.reloaded_state["revision"],
        "event_count_at_kill": before_events,
    }


def main() -> dict:
    runtime = Path("runtime") / "phase2_d3"
    suffix = uuid.uuid4().hex[:6]

    passing = run_l01(
        runtime,
        seed=101,
        run_id=f"d3-l01-101-{suffix}",
    )
    passing_summary = _run_summary(passing)
    _write(runtime / "summary.json", passing_summary)
    _write(runtime / "state_before_judge.json", passing.state_before_judge)
    _write(runtime / "state_after.json", passing.state_after)
    if passing.judge_result is not None:
        _write(runtime / "judge_result.json", passing.judge_result.to_dict())
    (runtime / "events.jsonl").write_text(
        passing.session.evidence.events_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    killed = run_l01(
        runtime,
        seed=202,
        run_id=f"d3-l01-202-kill-{suffix}",
        kill_before_cancel=True,
    )
    kill_summary = _kill_summary(killed)
    _write(runtime / "kill_summary.json", kill_summary)
    _write(runtime / "kill_state.json", killed.state_after)

    result = {
        "passing": {
            "run_id": passing.run_id,
            "verdict": "pass" if passing.judge_result and passing.judge_result.passed else "fail",
            "verified": passing.verified,
            "reload_equal": passing.reload_equal,
            "effects": len(passing.runner.effects_prepared),
            "state_path": passing_summary["state_path"],
        },
        "kill": {
            "run_id": killed.run_id,
            "pending_status": kill_summary["pending_effect"]["status"],
            "reload_equal": killed.reload_equal,
            "state_path": kill_summary["state_path"],
        },
        "summary_path": str(runtime / "summary.json"),
    }
    _write(runtime / "capture.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
