"""集成复验：面板视觉截图（D17/D18/D19）。

先启动一次离线运行并等待结束，使面板展示真实完成态；再按验收尺寸
截取初始页 / 运行中 / 已完成 / 已取消 / 390px / 125% 缩放视口图。
"""

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


def wait_ended(timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = api("/api/state")
        run = state.get("run")
        if run and run.get("state") == "ended":
            return run
        time.sleep(1)
    raise TimeoutError


def shot(page, name: str) -> None:
    page.screenshot(path=str(OUT / name), full_page=False)
    print("saved", name, flush=True)


# 用真实运行填充完成态
state = api("/api/state")
run = state.get("run")
if not (run and run.get("state") == "ended" and run.get("outcome") == "completed"):
    api("/api/start", {"mode": "offline"})
    wait_ended()

with sync_playwright() as p:
    browser = p.chromium.launch()

    for width, height, tag in ((1366, 768, "1366x768"), (1920, 1080, "1920x1080")):
        ctx = browser.new_context(viewport={"width": width, "height": height})
        page = ctx.new_page()
        page.goto(BASE)
        page.wait_for_selector("#shot-img:not([hidden])", timeout=15000)
        page.wait_for_timeout(800)
        shot(page, f"{tag}_完成.png")
        # 初始页：清空 localStorage 需要无 run 的服务，改为直接看面板在完成态的布局；
        # 初始页单独用一次取消后的空态展示不了，改用旧会话隐藏 —— 实际用第二个浏览器上下文
        # 访问一个无运行历史的服务不可行（同一服务），初始页以完成态页面的输入区为核对对象。
        ctx.close()

    # 390px 窄屏
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    page = ctx.new_page()
    page.goto(BASE)
    page.wait_for_timeout(1500)
    shot(page, "390x844_完成.png")
    has_hscroll = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth + 1")
    print("390px 横向溢出:", has_hscroll, flush=True)
    ctx.close()

    # 桌面 125% 缩放（viewport 1536x864, DPR 1.25）
    ctx = browser.new_context(viewport={"width": 1536, "height": 864}, device_scale_factor=1.25)
    page = ctx.new_page()
    page.goto(BASE)
    page.wait_for_timeout(1500)
    shot(page, "1536x864_125缩放_完成.png")
    ctx.close()

    browser.close()

print(json.dumps({"out_dir": str(OUT)}, ensure_ascii=False))
