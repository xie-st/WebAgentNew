"""Exercise the migrated HTTP panel, CLI child and actual Chromium locally."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/harness/demo"))
from app import DemoApp, make_server
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
app = DemoApp(repo_root=ROOT, runtime_dir=OUT / "demo", headless=True, port=0)
server = make_server(app)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = f"http://127.0.0.1:{app.port}"


def call(path, body=None):
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            page.goto(base, wait_until="networkidle")
            page.screenshot(path=str(OUT / "01_迁入面板启动.png"), full_page=True)
            run_id = call("/api/start", {"mode": "offline", "max_seconds": 180})["run_id"]
            deadline = time.monotonic() + 200
            while time.monotonic() < deadline:
                state = call("/api/state")
                run = state.get("run")
                if run and run.get("state") == "ended":
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Migrated demo did not terminate")
            assert run["outcome"] == "completed", run
            assert run["screenshots"], "No actual browser screenshot"
            assert run["answer"], "No final answer"
            page.reload(wait_until="networkidle")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "02_迁入面板完成.png"), full_page=True)
            evidence = {
                "status": "passed", "base_url": base, "mode": "offline_script",
                "browser": "actual Chromium, headless", "run_id": run_id,
                "outcome": run["outcome"], "answer": run["answer"],
                "screenshots": run["screenshots"], "state": state,
                "root": str(ROOT),
            }
            (OUT / "demo_smoke.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: evidence[key] for key in
                              ("status", "mode", "run_id", "outcome", "root")}, ensure_ascii=False))
        finally:
            browser.close()
finally:
    app.shutdown()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
