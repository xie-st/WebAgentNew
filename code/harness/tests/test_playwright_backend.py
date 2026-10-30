"""Offline Playwright acceptance on the seeded local fixture.

Every test uses a headless Chromium, a private temporary profile and a local
HTTP server bound to 127.0.0.1. No public page or real model is contacted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

playwright = pytest.importorskip("playwright")

from webagent.backends.playwright import PlaywrightBackend
from webagent.budget import Budget, Deadline
from webagent.clock import SystemClock
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.errors import ActionRejected
from webagent.events import read_events
from webagent.models.fake import ScriptedModel
from webagent.protocol import ActionOp, TerminationReason, ToolRequest
from webagent.session import Session
from webagent.tools import build_web_action

DRAFT = "Draft for quantum note"
TASK = (
    "In the local fixture catalog, search for 'quantum', open the first result, "
    f"type exactly '{DRAFT}' into the draft text box, click Preview draft, confirm "
    "the preview shows the text, and take a screenshot. Do not submit anything."
)


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        "Thought: next local step.\n"
        f"Action: {tool}\nAction Input: {json.dumps(arguments, ensure_ascii=False)}"
    )


def _chain_script() -> list[str]:
    return [
        _action("web_type", {"target": "1", "text": "quantum"}),
        _action("web_click", {"target": "2"}),
        _action("web_click", {"target": "1"}),
        _action("web_type", {"target": "1", "text": DRAFT}),
        _action("web_click", {"target": "2"}),
        "Thought: preview visible; no submit control was used.\n"
        "Final Answer: local chain completed without submitting",
    ]


def _session(tmp_path: Path, run_id: str, *, screenshot: bool = True) -> Session:
    clock = SystemClock()
    session = Session(
        run_id,
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + 120, clock=clock), clock=clock),
    )
    session.start_task(TASK)
    return session


def test_fake_model_playwright_headless_completes_seeded_task_chain(
    site_server: str, tmp_path: Path
) -> None:
    session = _session(tmp_path, "pw-chain")
    backend = PlaywrightBackend(
        "pw-chain",
        evidence=session.evidence,
        headless=True,
        runtime_dir=tmp_path / "runtime",
    )
    driver = None
    try:
        backend.start(f"{site_server}/index.html?seed=alpha-17")
        session.ingest_observation(backend.observe())
        model = ScriptedModel(_chain_script())
        driver = create_driver(session, model)
        step = driver.start()
        while step.status is DriverStatus.ACTION:
            action = driver.take_action()
            result = backend.execute(action)
            step = driver.accept_observation(result.observation)
        assert step.status is DriverStatus.COMPLETED
        assert session.finish_result is not None
        assert session.finish_result.reason is TerminationReason.COMPLETED
        final = session.observation
        assert final is not None
        assert "/detail.html" in final.url
        assert f'value="{DRAFT}"' in final.ax_text
        assert f"Preview: {DRAFT}" in final.ax_text
        assert 'role=button name="Submit"' not in final.ax_text

        screenshot = final.artifacts.screenshot
        assert screenshot is not None and screenshot.available
        assert screenshot.metadata["url"] == final.url
        assert screenshot.metadata["page_id"] == final.page_id
        assert screenshot.metadata["observation_id"] == final.observation_id
        saved = session.evidence.resolve_reference(screenshot.reference or "")
        assert saved.read_bytes().startswith(b"\x89PNG")
        assert session.evidence.verify(screenshot)

        events = read_events(session.evidence.events_path)
        artifact_events = [event for event in events if event["event_type"] == "artifact_saved"]
        observation_events = [event for event in events if event["event_type"] == "observation"]
        assert len(artifact_events) == len(observation_events) == 6
        for event in artifact_events:
            assert event["url"] is not None
            assert event["page_id"] == final.page_id
            assert event["observation_id"] is not None
            assert event["result"]["sha256"]
        assert [event["result"] for event in events if event["event_type"] == "proposed"] == [
            None
        ] * 5
        assert [event["result"] for event in events if event["event_type"] == "dispatched"] == [
            None
        ] * 5
    finally:
        if driver is not None:
            driver.close()
        backend.close()
        session.close()


def test_typing_appends_and_never_clears(site_server: str, tmp_path: Path) -> None:
    session = _session(tmp_path, "pw-append")
    backend = PlaywrightBackend(
        "pw-append",
        evidence=session.evidence,
        headless=True,
        runtime_dir=tmp_path / "runtime",
    )
    try:
        backend.start(f"{site_server}/index.html?seed=append")
        first = backend.observe()
        session.ingest_observation(first)
        first_action = build_web_action(
            ToolRequest("web_type", {"target": "1", "text": "alpha"}), first
        )
        session.propose(first_action)
        session.consume(first_action.action_id)
        first_result = backend.execute(first_action)
        session.ingest_observation(first_result.observation)
        second_action = build_web_action(
            ToolRequest("web_type", {"target": "1", "text": "beta"}),
            first_result.observation,
        )
        session.propose(second_action)
        session.consume(second_action.action_id)
        result = backend.execute(second_action)
        assert result.feedback.status.value == "observed"
        assert 'value="alphabeta"' in result.observation.ax_text
    finally:
        backend.close()
        session.close()


def test_stale_observation_action_is_rejected_after_new_observation(
    site_server: str, tmp_path: Path
) -> None:
    session = _session(tmp_path, "pw-stale")
    backend = PlaywrightBackend(
        "pw-stale",
        evidence=session.evidence,
        headless=True,
        runtime_dir=tmp_path / "runtime",
    )
    try:
        backend.start(f"{site_server}/index.html?seed=stale")
        first = backend.observe()
        session.ingest_observation(first)
        action = build_web_action(ToolRequest("web_click", {"target": "2"}), first)
        backend.observe()
        with pytest.raises(ActionRejected) as error:
            backend.execute(action)
        assert error.value.reason.value == "stale_observation"
    finally:
        backend.close()
        session.close()


def test_screenshot_refusal_is_visible_not_reused(site_server: str, tmp_path: Path) -> None:
    session = _session(tmp_path, "pw-noshot")
    backend = PlaywrightBackend(
        "pw-noshot",
        evidence=session.evidence,
        headless=True,
        runtime_dir=tmp_path / "runtime",
        screenshot=False,
    )
    try:
        backend.start(f"{site_server}/index.html?seed=noshot")
        observation = backend.observe()
        session.ingest_observation(observation)
        artifact = observation.artifacts.screenshot
        assert artifact is not None and not artifact.available
        assert artifact.reason == "disabled"
        assert list(session.evidence.screenshots_dir.glob("*.png")) == []
    finally:
        backend.close()
        session.close()


def test_close_is_idempotent_and_releases_only_its_own_profile(
    site_server: str, tmp_path: Path
) -> None:
    session = _session(tmp_path, "pw-close")
    backend = PlaywrightBackend(
        "pw-close",
        evidence=session.evidence,
        headless=True,
        runtime_dir=tmp_path / "runtime",
    )
    sentinel = tmp_path / "runtime" / "unrelated-user-file.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("keep me", encoding="utf-8")
    try:
        backend.start(f"{site_server}/index.html?seed=close")
        session.ingest_observation(backend.observe())
        profile = backend.profile_dir
        assert profile.exists()
        backend.close()
        backend.close()
        assert backend.current_url is None
        assert backend._context is None
        assert profile.exists(), "the dedicated profile is not silently deleted"
        assert sentinel.read_text(encoding="utf-8") == "keep me"
    finally:
        backend.close()
        session.close()
