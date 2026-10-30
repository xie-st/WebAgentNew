"""Offline C1 demonstration: official adapter through fake SDK callbacks.

This never imports a connection file, starts a platform client or opens a
network connection. It drives the real ``OfficialAgent``/``OfficialBackend``
path with a fake gateway and fake platform observations, then writes a normal
evidence run under the supplied directory.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from webagent.clock import FakeClock
from webagent.config import AppConfig, ModelConfig
from webagent.official_agent import OfficialAgent

RUN = "c1-offline"


class FakeGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(
            text=self.responses.pop(0),
            request_id=f"offline-request-{len(self.calls)}",
        )


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: use the current page.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


def _request(
    *,
    step: int,
    task: str | None,
    url: str,
    text: str,
    deadline: float,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "protocol_version": "1.0",
        "run_id": RUN,
        "step": step,
        "observation": {"type": "accessibility_tree", "url": url, "text": text},
        "history": [],
        "action_history": [],
        "execution_deadline": deadline,
    }
    if task is not None:
        request["task"] = task
    return request


def main() -> int:
    output_root = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/sample_run_c1")
    run_dir = output_root / "runs" / RUN
    if run_dir.exists():
        shutil.rmtree(run_dir)
    clock = FakeClock()
    gateway = FakeGateway(
        [
            _action("web_click", {"target": "2"}),
            _action(
                "web_type",
                {"target": "1", "text": "价格（含税）\n第二行 [备注]", "submit": False},
            ),
            "Thought: finished.\nFinal Answer: done",
        ]
    )
    config = AppConfig(
        profile="official",
        backend="official",
        runtime_dir=str(output_root),
        model=ModelConfig(
            provider="official",
            name="platform-unified",
            context_length=32768,
            tokenizer="heuristic",
            token_count_source="estimated",
            base_url_env=None,
            api_key_env=None,
        ),
    )
    agent = OfficialAgent(
        app_config=config,
        runtime_dir=output_root,
        clock=clock,
        model_api=gateway,
        download_artifacts=frozenset(),
    )
    deadline = clock.now() + 120
    agent.reset({"run_id": RUN, "execution_deadline": deadline})
    outputs = []
    outputs.append(
        agent.step(
            _request(
                step=0,
                task="click search, type the note, then finish",
                url="http://forum.test/search",
                text="[1] textbox 'Query'\n[2] button 'Search'",
                deadline=deadline,
            )
        )
    )
    clock.advance(1)
    outputs.append(
        agent.step(
            _request(
                step=1,
                task=None,
                url="http://forum.test/result",
                text="[1] textbox 'Note'\n[2] button 'Save'",
                deadline=deadline,
            )
        )
    )
    clock.advance(1)
    outputs.append(
        agent.step(
            _request(
                step=2,
                task=None,
                url="http://forum.test/result",
                text="[1] link 'Done'",
                deadline=deadline,
            )
        )
    )
    agent.close()
    for index, output in enumerate(outputs):
        print(f"step {index}: {output['action']}")
    print(f"official offline run: {run_dir}")
    print(f"model calls: {len(gateway.calls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
