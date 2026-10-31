"""browser-use trial: reuse only Browser/Page/Element, never its planning Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

browser_use = pytest.importorskip("browser_use")

from webagent.backends.browser_use import BrowserUseBackend
from webagent.budget import Budget, Deadline
from webagent.clock import SystemClock
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.events import read_events
from webagent.models.fake import ScriptedModel
from webagent.session import Session

DRAFT = "Draft for quantum note"


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: browser-use backend step.\n"
        f"Action: {tool}\nAction Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


@pytest.mark.timeout(180)
def test_browser_use_reuses_browser_layer_and_completes_same_chain(
    site_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_agent(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("browser-use planning Agent must not be constructed")

    monkeypatch.setattr(browser_use, "Agent", no_agent)
    clock = SystemClock()
    session = Session(
        "bu-chain",
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + 120, clock=clock), clock=clock),
    )
    session.start_task("browser-use trial task")
    backend = BrowserUseBackend(
        "bu-chain",
        evidence=session.evidence,
        headless=True,
        runtime_dir=tmp_path / "runtime",
    )
    driver = None
    try:
        backend.start(f"{site_server}/index.html?seed=alpha-17")
        session.ingest_observation(backend.observe())
        model = ScriptedModel(
            [
                _action("web_type", {"target": "1", "text": "quantum"}),
                _action("web_click", {"target": "2"}),
                _action("web_click", {"target": "1"}),
                _action("web_type", {"target": "1", "text": DRAFT}),
                _action("web_click", {"target": "2"}),
                "Thought: preview visible.\nFinal Answer: browser-use browser layer completed",
            ]
        )
        driver = create_driver(session, model)
        step = driver.start()
        while step.status is DriverStatus.ACTION:
            action = driver.take_action()
            result = backend.execute(action)
            step = driver.accept_observation(result.observation)
        assert step.status is DriverStatus.COMPLETED
        final = session.observation
        assert final is not None
        assert "/detail.html" in final.url
        assert f'value="{DRAFT}"' in final.ax_text
        assert f"Preview: {DRAFT}" in final.ax_text
        screenshot = final.artifacts.screenshot
        assert screenshot is not None and screenshot.available
        assert screenshot.metadata["observation_id"] == final.observation_id
        events = read_events(session.evidence.events_path)
        assert len([event for event in events if event["event_type"] == "artifact_saved"]) == 6
    finally:
        if driver is not None:
            driver.close()
        backend.close()
        session.close()
