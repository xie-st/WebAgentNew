"""重拍已暂停状态：验证“继续”按钮成为主要操作的修正。"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8787"
OUT = Path(__file__).resolve().parent / "自验证截图" / "视觉"
OUT.mkdir(parents=True, exist_ok=True)


def api(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


api("/api/start", {"mode": "offline"})
time.sleep(3)
api("/api/control", {"command": "pause"})
for _ in range(60):
    run = api("/api/state").get("run") or {}
    if run.get("state") == "paused":
        break
    if run.get("state") == "ended":
        raise SystemExit("运行过早结束，未捕获已暂停状态")
    time.sleep(1)

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1366, "height": 768})
    page = ctx.new_page()
    page.goto(BASE)
    page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT / "1366x768_已暂停_修正.png"))
    print("saved 1366x768_已暂停_修正.png")
    browser.close()

api("/api/control", {"command": "resume"})
deadline = time.time() + 180
while time.time() < deadline:
    run = api("/api/state").get("run") or {}
    if run.get("state") == "ended":
        print("resumed and ended:", run.get("outcome"))
        break
    time.sleep(1)
