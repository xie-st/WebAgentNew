"""CLI and task-file tests. No real model or public network."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from webagent.cli import doctor, load_task_file, main
from webagent.config import ConfigError
from webagent.control import RunControl
from webagent.events import read_events

HARNESS = Path(__file__).parents[1]
REAL_CONFIG = HARNESS / "configs" / "real.yaml"
TASK = HARNESS / "tests" / "fixtures" / "tasks" / "local_chain.json"
SCRIPT = HARNESS / "tests" / "fixtures" / "fake_scripts" / "local_chain.json"


def _config_for(tmp_path: Path) -> Path:
    text = REAL_CONFIG.read_text(encoding="utf-8")
    text = text.replace("runtime_dir: runtime", f"runtime_dir: {(tmp_path / 'runtime').as_posix()}")
    target = tmp_path / "real.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def _task_for(tmp_path: Path, base_url: str, run_id: str) -> Path:
    payload = json.loads(TASK.read_text(encoding="utf-8"))
    payload["start_url"] = f"{base_url}/index.html?seed=alpha-17"
    payload["run_id"] = run_id
    target = tmp_path / f"{run_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def test_task_file_is_external_and_draft_is_a_condition(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "task": "Fill the draft with {draft_text} and preview it.",
                "draft_text": "condition only",
                "start_url": "http://127.0.0.1:9/index.html",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_task_file(path)
    assert loaded["task"].endswith("preview it.")
    assert "condition only" in loaded["task"]


def test_task_file_rejects_executor_action_mappings(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps({"task": "do it", "actions": [{"op": "click", "target": "1"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_task_file(path)


def test_control_file_reads_pause_resume_cancel(tmp_path: Path) -> None:
    control = RunControl(tmp_path / "control.jsonl")
    path = tmp_path / "control.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("pause\n")
    assert control.poll() == "pause"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"command": "resume"}\n')
    assert control.poll() == "resume"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("cancel\n")
    assert control.poll() == "cancel"


def test_control_file_tolerates_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "control.jsonl"
    path.write_bytes("\ufeffpause\n".encode("utf-8"))
    assert RunControl(path).poll() == "pause"


def test_doctor_checks_names_not_values(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    config = _config_for(tmp_path)
    env = {
        "WEBAGENT_MODEL_BASE_URL": "https://fake.invalid/v1",
        "WEBAGENT_MODEL_API_KEY": "sk-FAKE-DOCTOR-VALUE",
    }
    old = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        code = doctor(str(config))
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    output = capsys.readouterr().out
    assert code == 0
    assert "WEBAGENT_MODEL_BASE_URL" in output
    assert "https://fake.invalid/v1" not in output
    assert "sk-FAKE-DOCTOR-VALUE" not in output
    assert "doctor ok" in output


def test_run_playwright_local_chain_through_cli(
    site_server: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config_for(tmp_path)
    task = _task_for(tmp_path, site_server, "cli-local-chain")
    code = main(
        [
            "run",
            "--config",
            str(config),
            "--backend",
            "playwright",
            "--task-file",
            str(task),
            "--model-script",
            str(SCRIPT),
            "--headless",
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--max-seconds",
            "120",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0, output
    assert "loop: completed" in output
    run_dir = tmp_path / "runtime" / "runs" / "cli-local-chain"
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "completed"
    events = read_events(run_dir / "events.jsonl")
    assert any(event["event_type"] == "artifact_saved" for event in events)
    observations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "observations").glob("obs-*.json"))
    ]
    final = observations[-1]
    assert final["url"].endswith("detail.html?seed=alpha-17&q=quantum&id=1")
    assert "Preview: Draft for quantum note" in final["ax_text"]
    screenshot = final["artifacts"]["screenshot"]
    assert screenshot["available"] is True
    assert screenshot["metadata"]["url"] == final["url"]
    assert screenshot["metadata"]["page_id"] == final["page_id"]
    assert screenshot["metadata"]["observation_id"] == final["observation_id"]
    assert (run_dir / screenshot["reference"]).is_file()
    assert len(list((run_dir / "screenshots").glob("*.png"))) == 6


@pytest.mark.timeout(180)
def test_cli_pause_then_resume_reobserves_and_does_not_replay(
    site_server: str, tmp_path: Path
) -> None:
    config = _config_for(tmp_path)
    task = _task_for(tmp_path, site_server, "cli-pause")
    run_dir = tmp_path / "runtime" / "runs" / "cli-pause"
    run_dir.mkdir(parents=True)
    (run_dir / "control.jsonl").write_text("pause\n", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "webagent",
        "run",
        "--config",
        str(config),
        "--backend",
        "playwright",
        "--task-file",
        str(task),
        "--model-script",
        str(SCRIPT),
        "--headless",
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--max-seconds",
        "120",
        "--pause-timeout",
        "60",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(HARNESS.parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )
    try:
        deadline = time.monotonic() + 90
        paused = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            events_path = run_dir / "events.jsonl"
            if events_path.exists():
                events = read_events(events_path)
                if any(event["event_type"] == "paused" for event in events):
                    paused = True
                    break
            time.sleep(0.25)
        assert paused, "run did not reach the pause point"
        with (run_dir / "control.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("resume\n")
        output, _ = process.communicate(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert process.returncode == 0, output
    events = read_events(run_dir / "events.jsonl")
    paused_events = [event for event in events if event["event_type"] == "paused"]
    resumed_events = [event for event in events if event["event_type"] == "resumed"]
    assert len(paused_events) == 1
    assert len(resumed_events) == 1
    assert resumed_events[0]["observation_id"] != paused_events[0]["observation_id"]
    assert resumed_events[0]["result"]["old_observation_id"] != resumed_events[0]["observation_id"]
    cancelled = [event for event in events if event["event_type"] == "cancelled"]
    assert any(event["result"]["reason"] == "paused" for event in cancelled)
    dispatched = [event for event in events if event["event_type"] == "dispatched"]
    assert len(dispatched) == 4, "the paused first action is not dispatched"
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "completed"
    assert len(list((run_dir / "screenshots").glob("*.png"))) >= 5

