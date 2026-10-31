"""面板实机自检脚本：通过真实页面驱动两次离线运行并截取各状态视口截图。

前置：面板已在 127.0.0.1:8787 运行（app.py --port 8787 --no-browser）。
输出：验收记录/20261031-dsh-demo/自验证截图/ 下的 02/05/06 与 视觉/ 目录。
截图均为 Playwright 真实视口截图，内容为面板实际渲染状态。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = "http://127.0.0.1:8787"
OUT = Path(__file__).resolve().parent / "自验证截图"
VISUAL = OUT / "视觉"
VISUAL.mkdir(parents=True, exist_ok=True)


def api(path: str, payload: dict | None = None) -> dict:
    if payload is None:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def shot(page: Page, rel: str) -> None:
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(target))
    print("saved:", target.name)


def state(page: Page) -> dict:
    return page.evaluate(
        "async () => (await (await fetch('/api/state')).json())"
    )


def wait_state(page: Page, predicate, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = state(page)
        run = data.get("run")
        if run and predicate(run):
            return data
        page.wait_for_timeout(500)
    raise TimeoutError("等待面板状态超时")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1366, "height": 768})
        page = context.new_page()
        page.goto(BASE)
        page.wait_for_load_state("networkidle")
        page.wait_for_selector("#mode-badge")
        page.wait_for_timeout(800)

        # ---- 初始页（待运行）----
        shot(page, "视觉/1366x768_初始页.png")
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.wait_for_timeout(600)
        shot(page, "视觉/1920x1080_初始页.png")
        page.set_viewport_size({"width": 1366, "height": 768})

        # ---- 异常反馈：非法时长 ----
        page.fill("#seconds", "-5")
        page.click("#btn-start")
        page.wait_for_timeout(600)
        message = page.text_content("#form-message") or ""
        assert "正数" in message, f"预期输入校验错误，实际：{message!r}"
        shot(page, "06_异常反馈.png")
        page.fill("#seconds", "600")

        # ---- 第一次离线运行：完整完成 ----
        page.click("#btn-start")
        page.wait_for_selector("#shot-img:not([hidden])", timeout=120000)
        page.wait_for_timeout(400)
        shot(page, "视觉/1366x768_运行中.png")
        data = wait_state(page, lambda r: r["state"] == "ended", timeout=120)
        assert data["run"]["outcome"] == "completed", data["run"]["outcome"]
        page.wait_for_timeout(600)
        shot(page, "02_离线完成.png")
        run_id_1 = data["run"]["run_id"]
        print("run_id #1:", run_id_1)

        # 完成页其他尺寸
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.wait_for_timeout(600)
        shot(page, "视觉/1920x1080_完成.png")
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(600)
        shot(page, "视觉/390x844_完成.png")
        page.set_viewport_size({"width": 1366, "height": 768})

        # ---- 第二次离线运行：暂停 → 接管间隙 → 继续 → 完成 ----
        run_id_2 = None
        for attempt in range(4):
            page.click("#btn-start")
            # 尽早提交暂停请求，确保赶在动作边界前
            page.wait_for_timeout(1500)
            if not page.is_disabled("#btn-pause"):
                page.click("#btn-pause")
            try:
                data = wait_state(
                    page, lambda r: r["state"] in ("paused", "ended"), timeout=60
                )
            except TimeoutError:
                data = state(page)
            if data["run"]["state"] == "ended":
                print(f"attempt {attempt}: 未赶上暂停（已结束），重试")
                page.wait_for_timeout(1000)
                continue
            page.wait_for_timeout(800)
            shot(page, "05_暂停接管.png")
            shot(page, "视觉/1366x768_已暂停.png")
            page.click("#btn-resume")
            data = wait_state(page, lambda r: r["state"] == "ended", timeout=120)
            assert data["run"]["outcome"] == "completed", data["run"]["outcome"]
            run_id_2 = data["run"]["run_id"]
            print("run_id #2:", run_id_2, "（含暂停/继续）")
            break
        if run_id_2 is None:
            print("WARN: 四次尝试均未捕获已暂停状态")
        browser.close()

    # 汇总 run_id 供记录使用
    summary = {"run_id_1": run_id_1, "run_id_2": run_id_2, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT.parent / "截图运行记录.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
