"""Demo 面板组件级测试：校验、进程锁、控制状态机、结果映射、证据白名单与脱敏。

不启动真实浏览器；子进程用假进程或可控的慢子进程模拟，验证面板与
``control.jsonl`` 行协议及 events/result 产物之间的真实契约。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import app as demo_app_module  # noqa: E402
from app import DemoApp, DemoError, PRESET_TASK  # noqa: E402

HARNESS = Path(__file__).parents[1]
SLOW_CHILD = Path(__file__).parent / "fixtures" / "slow_child.py"
REAL_POPEN = subprocess.Popen  # monkeypatch 前保存原始引用


class FakeProc:
    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.pid = 4242
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DemoApp:
    launched: list[FakeProc] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> FakeProc:
        proc = FakeProc(argv)
        launched.append(proc)
        return proc

    monkeypatch.setattr(demo_app_module.subprocess, "Popen", fake_popen)
    application = DemoApp(
        repo_root=Path(os.environ.get("WEBAGENT_REPO_ROOT", str(Path.cwd()))),
        runtime_dir=tmp_path / "runtime",
        headless=True,
        port=8787,
    )
    application._launched = launched  # type: ignore[attr-defined]
    return application


def _write_events(run_dir: Path, events: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
    (run_dir / "events.jsonl").write_text(payload, encoding="utf-8")


def _event(kind: str, **extra: Any) -> dict[str, Any]:
    base = {
        "schema_version": "webagent.events.v1",
        "event_type": kind,
        "time": "2026-10-31T08:00:00.000+00:00",
        "elapsed_ms": 1000,
        "url": None,
    }
    base.update(extra)
    return base


# ---------- 输入校验 ----------

def test_start_rejects_empty_task_and_bad_url(
    app: DemoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEBAGENT_MODEL_BASE_URL", "http://model.example")
    monkeypatch.setenv("WEBAGENT_MODEL_API_KEY", "sk-test")
    with pytest.raises(DemoError, match="任务描述不能为空"):
        app.start("real", "   ", "http://example.com", 120)
    with pytest.raises(DemoError, match="网址"):
        app.start("real", "任务", "not-a-url", 120)
    with pytest.raises(DemoError, match="网址"):
        app.start("real", "任务", "ftp://example.com", 120)
    assert app.current is None


def test_start_rejects_non_positive_seconds(app: DemoApp) -> None:
    for value in (0, -5, "abc", None):
        with pytest.raises(DemoError, match="正数"):
            app.start("real", "任务", "http://example.com", value)
    with pytest.raises(DemoError, match="不能超过"):
        app.start("real", "任务", "http://example.com", 999999)


def test_real_mode_blocked_without_env(
    app: DemoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WEBAGENT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("WEBAGENT_MODEL_API_KEY", raising=False)
    with pytest.raises(DemoError, match="WEBAGENT_MODEL"):
        app.start("real", "任务", "http://example.com", 120)
    assert app.current is None


# ---------- 模式隔离与任务写入 ----------

def test_offline_mode_locks_task_and_url(app: DemoApp) -> None:
    run_id = app.start(
        "offline", "用户试图修改的任务文本", "http://evil.example.com", 120
    )
    record = app.current
    assert record is not None
    assert record.task == PRESET_TASK
    assert record.start_url == app.fixture_url()
    argv = record.argv
    assert "--model-script" in argv
    task_payload = json.loads(
        (record.demo_dir / "task.json").read_text(encoding="utf-8")
    )
    assert task_payload["task"] == PRESET_TASK


def test_task_file_keeps_unicode_exactly(
    app: DemoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEBAGENT_MODEL_BASE_URL", "http://model.example")
    monkeypatch.setenv("WEBAGENT_MODEL_API_KEY", "sk-test")
    task = '带引号 "x" 与换行\n第二行 🙂 <b>不是HTML</b>'
    run_id = app.start("real", task, "https://example.com/page?a=1", 300)
    record = app.current
    assert record is not None
    payload = json.loads((record.demo_dir / "task.json").read_text(encoding="utf-8"))
    assert payload["task"] == task  # 逐字对应，UTF-8 正确
    assert payload["start_url"] == "https://example.com/page?a=1"
    assert payload["headless"] is True
    assert "--no-headless" not in record.argv and "--headless" in record.argv


# ---------- 单活动 run 锁 ----------

def test_single_active_run_lock(app: DemoApp) -> None:
    app.start("offline", "", "", 120)
    with pytest.raises(DemoError, match="已有任务"):
        app.start("offline", "", "", 120)
    assert len(app._launched) == 1  # type: ignore[attr-defined]


# ---------- 控制协议与状态机 ----------

def _spawn_slow_child(
    argv: list[str], runtime_dir: Path
) -> subprocess.Popen[Any]:
    """把面板生成的 argv 映射到慢子进程：runtime_dir 与 run_id 来自真实参数。"""

    run_id = argv[argv.index("--run-id") + 1]
    demo_dir = runtime_dir / "demo" / run_id
    demo_dir.mkdir(parents=True, exist_ok=True)
    out = (demo_dir / "stdout.log").open("wb")
    err = (demo_dir / "stderr.log").open("wb")
    return REAL_POPEN(
        [
            sys.executable,
            "-u",
            str(SLOW_CHILD),
            str(runtime_dir),
            run_id,
        ],
        cwd=str(runtime_dir.parent),
        stdout=out,
        stderr=err,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


def test_control_state_machine_with_slow_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用可控慢子进程验证：暂停请求 -> 结构化 paused 确认 -> 恢复 -> 完成。"""

    runtime_dir = tmp_path / "runtime"

    def real_popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        return _spawn_slow_child(argv, runtime_dir)

    monkeypatch.setattr(demo_app_module.subprocess, "Popen", real_popen)
    application = DemoApp(runtime_dir=runtime_dir, headless=True, port=8787)
    application.start("offline", "", "", 120)
    record = application.current
    assert record is not None

    deadline = time.monotonic() + 15
    while application.current is None or not application.current.has_events:
        application.refresh()
        assert time.monotonic() < deadline, "慢子进程未及时产出事件"
        time.sleep(0.05)

    application.control("pause")
    assert application.current.state == "pause_requested"
    with pytest.raises(DemoError):
        application.control("pause")  # 防重复提交
    with pytest.raises(DemoError):
        application.control("resume")  # 未确认暂停前不能继续

    deadline = time.monotonic() + 15
    while application.current.state != "paused":
        application.refresh()
        assert time.monotonic() < deadline, "未收到结构化 paused 确认"
        time.sleep(0.05)

    application.control("resume")
    assert application.current.state == "resume_requested"
    deadline = time.monotonic() + 15
    while application.current.state != "ended":
        application.refresh()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert application.current.outcome == "completed"
    assert application.current.result is not None
    assert application.current.exit_code == 0


def test_cancel_from_running_and_force_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path / "runtime"

    def real_popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        return _spawn_slow_child(argv, runtime_dir)

    monkeypatch.setattr(demo_app_module.subprocess, "Popen", real_popen)
    application = DemoApp(runtime_dir=runtime_dir, headless=True, port=8787)
    application.start("offline", "", "", 120)
    record = application.current
    assert record is not None

    deadline = time.monotonic() + 15
    while not application.current.has_events:  # type: ignore[union-attr]
        application.refresh()
        assert time.monotonic() < deadline
        time.sleep(0.05)

    application.control("cancel")
    assert application.current.state == "cancel_requested"
    # 取消确认前仍不能开始新任务
    with pytest.raises(DemoError, match="已有任务"):
        application.start("offline", "", "", 120)

    deadline = time.monotonic() + 20
    while application.current.state != "ended":
        application.refresh()
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert application.current.outcome == "cancelled"


# ---------- 结果映射 ----------

def _finish(app: DemoApp, result: dict[str, Any], exit_code: int) -> None:
    record = app.current
    assert record is not None
    _write_events(
        record.run_dir,
        [_event("run_start"), _event("observation", url="http://x/", observation_id="obs-0")],
    )
    record.run_dir.mkdir(parents=True, exist_ok=True)
    (record.run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )
    launched = app._launched[-1]  # type: ignore[attr-defined]
    launched.returncode = exit_code


@pytest.mark.parametrize(
    ("termination", "exit_code", "expected"),
    [
        ("completed", 0, "completed"),
        ("deadline", 2, "timeout"),
        ("cancelled", 130, "cancelled"),
        ("error", 2, "failed"),
        ("incomplete", 2, "incomplete"),
        ("budget_exhausted", 2, "incomplete"),
    ],
)
def test_outcome_mapping(
    app: DemoApp, termination: str, exit_code: int, expected: str
) -> None:
    app.start("offline", "", "", 120)
    _finish(
        app,
        {"termination": termination, "finish": {"message": "回答", "claim": "回答"}},
        exit_code,
    )
    app.refresh()
    assert app.current is not None
    assert app.current.state == "ended"
    assert app.current.outcome == expected


def test_exit_without_result_maps_to_start_failed(app: DemoApp) -> None:
    app.start("offline", "", "", 120)
    launched = app._launched[-1]  # type: ignore[attr-defined]
    launched.returncode = 2
    app.refresh()
    assert app.current is not None
    assert app.current.outcome == "start_failed"


def test_partial_jsonl_tail_does_not_crash(app: DemoApp) -> None:
    app.start("offline", "", "", 120)
    record = app.current
    assert record is not None
    good = json.dumps(_event("run_start"), ensure_ascii=False)
    partial = json.dumps(_event("observation"), ensure_ascii=False)[:-8]  # 截断尾行
    record.run_dir.mkdir(parents=True, exist_ok=True)
    (record.run_dir / "events.jsonl").write_text(good + "\n" + partial, encoding="utf-8")
    app.refresh()  # 不抛异常
    snapshot = app.snapshot()
    assert snapshot["run"] is not None


def test_event_log_is_incremental_not_duplicated(app: DemoApp) -> None:
    app.start("offline", "", "", 120)
    record = app.current
    assert record is not None
    record.run_dir.mkdir(parents=True, exist_ok=True)
    _write_events(
        record.run_dir,
        [_event("run_start"), _event("observation", url="http://x/")],
    )
    app.refresh()
    count_after_first = len(record.log_entries)
    assert count_after_first == 2
    # 追加新事件后再次刷新：只新增，不重复
    with (record.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_event("agent_claim"), ensure_ascii=False) + "\n")
    app.refresh()
    app.refresh()
    kinds = [e["kind"] for e in record.log_entries]
    assert kinds == ["run_start", "observation", "agent_claim"]
    snapshot = app.snapshot()
    assert len(snapshot["run"]["events"]) == 3


# ---------- 证据访问白名单 ----------

def test_download_whitelist(app: DemoApp) -> None:
    app.start("offline", "", "", 120)
    record = app.current
    assert record is not None
    run_id = record.run_id

    path, name = app.resolve_download(run_id, "task.json")
    assert path.name == "task.json"
    with pytest.raises(DemoError, match="不允许"):
        app.resolve_download(run_id, "model_calls.json")
    with pytest.raises(DemoError, match="拒绝"):
        app.resolve_download("other-run", "task.json")

    screenshots = record.run_dir / "screenshots"
    screenshots.mkdir(parents=True)
    (screenshots / "obs-0-abc.png").write_bytes(b"\x89PNG\r\n")
    assert app.resolve_screenshot(run_id, "obs-0-abc.png").is_file()
    with pytest.raises(DemoError):
        app.resolve_screenshot(run_id, "..\\secret.png")
    with pytest.raises(DemoError):
        app.resolve_screenshot(run_id, "sub/obs.png")
    with pytest.raises(DemoError):
        app.resolve_screenshot(run_id, "not-a-screenshot.txt")


# ---------- 脱敏 ----------

def test_console_redaction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WEBAGENT_MODEL_API_KEY", "sk-secret-value-123")
    monkeypatch.setenv("WEBAGENT_MODEL_BASE_URL", "https://api.example.com/v1")
    application = DemoApp(runtime_dir=tmp_path / "runtime", headless=True, port=8787)
    text = "url=https://api.example.com/v1 key=sk-secret-value-123 token=abcdefgh1234"
    redacted = application.redactor.redact_text(text)
    assert "sk-secret-value-123" not in redacted
    assert "abcdefgh1234" not in redacted
    assert "api.example.com" in redacted  # base_url 本身保留（非密钥）


def test_console_tail_is_redacted_in_snapshot(
    app: DemoApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEBAGENT_MODEL_API_KEY", "sk-secret-value-123")
    app.redactor = demo_app_module.Redactor(secrets=["sk-secret-value-123"])
    app.start("offline", "", "", 120)
    record = app.current
    assert record is not None
    (record.demo_dir / "stdout.log").write_text(
        "api_key: sk-secret-value-123\nplain line\n", encoding="utf-8"
    )
    app.refresh()
    snapshot = app.snapshot()
    stdout = "\n".join(snapshot["run"]["stdout_tail"])
    assert "sk-secret-value-123" not in stdout
    assert "plain line" in stdout


# ---------- 端口独占 ----------

def test_port_conflict_is_rejected() -> None:
    """第二个面板实例绑定同一端口必须失败（Windows 双绑定回归）。"""

    app1 = DemoApp(port=0, headless=True)
    server1 = demo_app_module.make_server(app1)
    try:
        app2 = DemoApp(port=app1.port, headless=True)
        with pytest.raises(OSError):
            demo_app_module.make_server(app2)
    finally:
        server1.server_close()
