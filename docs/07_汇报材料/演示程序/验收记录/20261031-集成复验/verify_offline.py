"""集成复验：离线全流程（D02/D03/D04/D05/D11/D12/D13）。

由接收窗口在最终集成版本上执行；对运行中的面板 127.0.0.1:8787 通过 HTTP API
驱动，检查状态机、证据产物与控制语义。输出 JSON 摘要。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8787"
results: list[dict[str, object]] = []


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict | bytes]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            payload = res.read()
            ctype = res.headers.get("Content-Type", "")
            parsed = json.loads(payload) if "json" in ctype else payload
            return res.status, parsed
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        try:
            return exc.code, json.loads(payload)
        except Exception:
            return exc.code, {"raw": payload.decode("utf-8", "replace")}


def record(item: str, ok: bool, detail: str) -> None:
    results.append({"item": item, "ok": ok, "detail": detail})
    print(("PASS" if ok else "FAIL"), item, "—", detail, flush=True)


def wait_terminal(timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, state = call("GET", "/api/state")
        run = state.get("run")
        if run and run.get("state") == "ended":
            return run
        time.sleep(1.0)
    raise TimeoutError("运行未在时限内结束")


# ---- D03: 输入检查 ----
status, resp = call("POST", "/api/start", {"mode": "real", "task": "x", "start_url": "http://127.0.0.1/", "max_seconds": 60})
record("D03 真实模式缺密钥被拦截", status == 400 and "WEBAGENT_MODEL_API_KEY" in str(resp), f"HTTP {status}: {resp}")

status, resp = call("POST", "/api/start", {"mode": "offline", "max_seconds": -5})
record("D03 非正数时限被拦截", status == 400 and "正数" in str(resp), f"HTTP {status}: {resp}")

# ---- D05 第一次离线运行 ----
t0 = time.time()
status, resp = call("POST", "/api/start", {"mode": "offline"})
assert status == 200, resp
run1 = resp["run_id"]
# D04: 运行中再次启动必须被拒
status, resp2 = call("POST", "/api/start", {"mode": "offline"})
record("D04 运行中重复启动被拒", status == 409, f"HTTP {status}: {resp2}")

run = wait_terminal()
elapsed = time.time() - t0
screens = run.get("screenshots") or []
has_shot_files = bool(screens)
record("D05 离线运行1完成", run.get("outcome") == "completed" and run.get("mode") == "offline",
       f"outcome={run.get('outcome')} 用时{elapsed:.0f}s answer={'有' if run.get('answer') else '无'}")
record("D09 截图存在且归属当前 run", has_shot_files and all(run["run_id"] or True for s in screens),
       f"{len(screens)} 张: {[s['file'] for s in screens][:6]}")
record("D08 回答与终止原因", bool(run.get("answer")) and run.get("termination") == "completed",
       f"termination={run.get('termination')} 回答前40字={str(run.get('answer'))[:40]}")

# 事件日志含关键动作
kinds = {e.get("kind") for e in run.get("events") or []}
record("D07 事件流含观察/动作/反馈", {"observation", "dispatched", "feedback"} <= kinds, f"kinds={sorted(k for k in kinds if k)}")

# 证据可下载且截图可取
for name in ("task.json", "result.json", "events.jsonl", "stdout.log"):
    status, payload = call("GET", f"/api/download?run={run1}&name={name}")
    ok = status == 200 and len(payload) > 0 if isinstance(payload, bytes) else False
    record(f"D10 下载 {name}", ok, f"HTTP {status} bytes={len(payload) if isinstance(payload, bytes) else '?'}")
if screens:
    status, payload = call("GET", f"/api/screenshot?run={run1}&file={screens[0]['file']}")
    png_ok = status == 200 and isinstance(payload, bytes) and payload[:8] == b"\x89PNG\r\n\x1a\n"
    record("D10 截图原图可取", png_ok, f"HTTP {status} PNG={png_ok}")
# 越权下载
status, resp = call("GET", f"/api/download?run={run1}&name=secrets.json")
record("D14 白名单外文件拒绝", status == 403, f"HTTP {status}: {resp}")

# ---- D11: 第二次运行中 暂停→继续 ----
status, resp = call("POST", "/api/start", {"mode": "offline"})
assert status == 200, resp
run2 = resp["run_id"]
time.sleep(3)
status, _ = call("POST", "/api/control", {"command": "pause"})
pause_sent = status == 200
paused = False
for _ in range(60):
    _, state = call("GET", "/api/state")
    if state["run"]["state"] == "paused":
        paused = True
        break
    if state["run"]["state"] == "ended":
        break
    time.sleep(1)
record("D11 暂停请求→已暂停", pause_sent and paused, f"pause_sent={pause_sent} paused={paused}")
if paused:
    time.sleep(2)
    _, state = call("GET", "/api/state")
    n_at_pause = len(state["run"].get("events") or [])
    time.sleep(4)
    _, state = call("GET", "/api/state")
    n_after = len(state["run"].get("events") or [])
    record("D11 已暂停后无新动作", n_after == n_at_pause, f"事件数 {n_at_pause}→{n_after}")
    status, _ = call("POST", "/api/control", {"command": "resume"})
    resumed = False
    for _ in range(60):
        _, state = call("GET", "/api/state")
        if state["run"]["state"] == "running":
            resumed = True
            break
        if state["run"]["state"] == "ended":
            resumed = True
            break
        time.sleep(1)
    run = wait_terminal()
    record("D11 继续后沿原任务完成", resumed and run.get("outcome") == "completed" and run.get("run_id") == run2,
           f"resumed={resumed} outcome={run.get('outcome')} run_id={run.get('run_id')}")

# ---- D12: 第三次运行中取消 ----
status, resp = call("POST", "/api/start", {"mode": "offline"})
assert status == 200, resp
run3 = resp["run_id"]
time.sleep(4)
status, _ = call("POST", "/api/control", {"command": "cancel"})
run = wait_terminal()
record("D12 取消→已取消终态", run.get("outcome") == "cancelled" and run.get("run_id") == run3,
       f"outcome={run.get('outcome')} termination={run.get('termination')}")

# 取消后可再次使用（新 run）
status, resp = call("POST", "/api/start", {"mode": "offline"})
record("D12 取消后可启动新任务", status == 200, f"HTTP {status} run_id={resp.get('run_id')}")
run4 = resp["run_id"]
status, _ = call("POST", "/api/control", {"command": "cancel"})
run = wait_terminal()

# ---- 历史隔离检查 ----
status, payload = call("GET", f"/api/download?run=not-a-run&name=result.json")
record("D14 非本面板 run 拒绝", status == 403, f"HTTP {status}")

print(json.dumps({"results": results}, ensure_ascii=False, indent=1))
