"""可控慢子进程：按真实产物契约模拟 CLI 的暂停 / 继续 / 取消行为。

用法：python slow_child.py <runtime_dir> <run_id>
行为：
* 在 <runtime_dir>/runs/<run_id>/events.jsonl 追加 run_start、observation；
* 等待 control.jsonl 出现 pause，追加结构化 paused 事件；
* 等待 resume 后追加 resumed 事件，写入 result.json（completed）并退出 0；
* 等待期间出现 cancel 则写 result.json（cancelled）并退出 130。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def append(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> int:
    runtime = Path(sys.argv[1])
    run_id = sys.argv[2]
    run_dir = runtime / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events = run_dir / "events.jsonl"
    control = run_dir / "control.jsonl"
    control.write_text("", encoding="utf-8")

    def read_commands() -> list[str]:
        if not control.exists():
            return []
        return [
            line.strip().lower()
            for line in control.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    append(events, {"event_type": "run_start", "elapsed_ms": 0})
    time.sleep(0.3)
    append(events, {"event_type": "observation", "url": "http://fixture.local/", "elapsed_ms": 300})

    def wait_for(commands: set[str], timeout: float = 30.0) -> str | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for command in read_commands():
                if command in commands:
                    return command
            time.sleep(0.05)
        return None

    command = wait_for({"pause", "cancel"})
    if command == "cancel":
        (run_dir / "result.json").write_text(
            json.dumps({"termination": "cancelled"}), encoding="utf-8"
        )
        return 130
    append(events, {"event_type": "paused", "elapsed_ms": 600})

    command = wait_for({"resume", "cancel"})
    if command == "cancel":
        (run_dir / "result.json").write_text(
            json.dumps({"termination": "cancelled"}), encoding="utf-8"
        )
        return 130
    append(events, {"event_type": "resumed", "elapsed_ms": 900})
    (run_dir / "result.json").write_text(
        json.dumps(
            {"termination": "completed", "finish": {"message": "慢子进程完成"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
