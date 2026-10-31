"""Demo 面板端到端测试（离线脚本模式，真实 CLI + Playwright headless）。

覆盖：通过 HTTP 启动离线运行、等待真实执行完成、结果与截图映射、
取消路径、证据下载白名单与控制台脱敏。运行需要 Playwright Chromium，
用时约 1–2 分钟。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import app as demo_app_module  # noqa: E402
from app import DemoApp, make_server  # noqa: E402

HARNESS = DEMO_DIR.parent
SCRIPT = HARNESS / "tests" / "fixtures" / "fake_scripts" / "local_chain.json"


class PanelServer:
    def __init__(self, tmp_path: Path) -> None:
        self.app = DemoApp(
            runtime_dir=tmp_path / "runtime",
            headless=True,
            port=0,
            offline_script=SCRIPT,
        )
        self.server = make_server(self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.app.port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def wait_finished(self, timeout: float = 100.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.get_json("/api/state")
            run = state.get("run")
            if run and run["state"] == "ended":
                return state
            time.sleep(1.0)
        raise AssertionError("运行超时未结束")

    base: str


@pytest.fixture
def panel(tmp_path: Path) -> PanelServer:
    server = PanelServer(tmp_path)
    server.base = server.start()
    try:
        yield server
    finally:
        server.app.shutdown()
        server.stop()


def test_offline_run_end_to_end(panel: PanelServer) -> None:
    payload = panel.post_json("/api/start", {"mode": "offline"})
    run_id = payload["run_id"]

    state = panel.wait_finished(timeout=100.0)
    run = state["run"]
    assert run["outcome"] == "completed"
    assert run["mode"] == "offline"
    assert run["answer"], "离线脚本运行应返回最终回答"
    assert run["termination"] == "completed"
    assert len(run["screenshots"]) >= 1, "真实浏览器应产生截图"

    # 截图可访问且来自本次 run
    shot = run["screenshots"][-1]
    with urllib.request.urlopen(
        panel.base + f"/api/screenshot?run={run_id}&file={shot['file']}", timeout=30
    ) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/png"

    # 结果文件真实存在，与面板映射一致
    result_path = Path(run["run_dir"]) / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["termination"] == "completed"
    assert result["finish"]["message"] == run["answer"]

    # 离线模式任务与网址锁定为预设
    task = json.loads(
        (Path(run["demo_dir"]) / "task.json").read_text(encoding="utf-8")
    )
    assert task["start_url"].endswith("/fixture/index.html?seed=alpha-17")
    assert task["start_url"].startswith(f"http://127.0.0.1:{panel.app.port}/")

    # 下载白名单：允许的文件可下载，其他被拒绝
    with urllib.request.urlopen(
        panel.base + f"/api/download?run={run_id}&name=result.json", timeout=30
    ) as response:
        assert response.status == 200
    try:
        urllib.request.urlopen(
            panel.base + f"/api/download?run={run_id}&name=model_calls.json", timeout=30
        )
        raise AssertionError("白名单外文件不应可下载")
    except urllib.error.HTTPError as exc:
        assert exc.code == 403

    # 结束后可再次启动（历史保留）
    payload2 = panel.post_json("/api/start", {"mode": "offline"})
    assert payload2["run_id"] != run_id
    state2 = panel.wait_finished(timeout=100.0)
    assert state2["run"]["outcome"] == "completed"


def test_cancel_running_offline_run(panel: PanelServer) -> None:
    panel.post_json("/api/start", {"mode": "offline"})
    # 等待真正进入运行状态（有事件产生）后再取消
    deadline = time.monotonic() + 60
    while True:
        state = panel.get_json("/api/state")
        run = state["run"]
        assert run is not None
        if run["state"] in ("running", "pause_requested"):
            break
        if run["state"] == "ended":
            raise AssertionError("运行过早结束，无法测试取消")
        assert time.monotonic() < deadline
        time.sleep(0.5)

    panel.post_json("/api/control", {"command": "cancel"})
    state = panel.wait_finished(timeout=90.0)
    run = state["run"]
    assert run["outcome"] == "cancelled", f"取消后应为 cancelled，实际 {run['outcome']}"
    assert run["exit_code"] == 130

    # 结束后恢复正常：可再次启动
    panel.post_json("/api/start", {"mode": "offline"})
    state = panel.wait_finished(timeout=100.0)
    assert state["run"]["outcome"] == "completed"


def test_state_endpoint_available_before_any_run(panel: PanelServer) -> None:
    state = panel.get_json("/api/state")
    assert state["run"] is None
    assert state["preset"]["task"]
    assert state["server"]["fixture_url"].endswith("/fixture/index.html?seed=alpha-17")
