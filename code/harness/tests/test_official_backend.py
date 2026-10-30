"""C02/C03: official observation, action and artifact offline contracts."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from webagent.backends.official import (
    OfficialActionFormatError,
    OfficialBackend,
    parse_ax_elements,
    serialize_stop,
    unescape_text,
)
from webagent.clock import FakeClock
from webagent.effgen_bridge import HarnessToolRuntime
from webagent.errors import ActionRejected, RejectReason
from webagent.events import read_events
from webagent.protocol import ActionOp, ToolRequest, WebAction
from webagent.session import Session
from webagent.tools import build_web_action

START = "http://example.test/start"
NEXT = "http://example.test/next"
OFFICIAL_CLIENT = Path(__file__).resolve().parents[2] / "official_client"


def test_frozen_browser_api_has_the_three_artifact_readers() -> None:
    sys.path.insert(0, str(OFFICIAL_CLIENT))
    try:
        from third_party_agent import browser as sdk_browser
    finally:
        sys.path.remove(str(OFFICIAL_CLIENT))
    assert callable(sdk_browser.get_screenshot)
    assert callable(sdk_browser.save_screenshot)
    assert callable(sdk_browser.get_html)


def _png(width: int = 4, height: int = 3, marker: bytes = b"\x00") -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
    dimensions = struct.pack(">II", width, height)
    return header + dimensions + b"\x08\x06\x00\x00\x00" + marker


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeBrowser:
    def __init__(self, png: bytes | None = None, html: str | None = None) -> None:
        self.png = png
        self.html = html
        self.saved: list[tuple[dict[str, Any], str]] = []
        self.screenshot_ids: list[str] = []
        self.html_ids: list[str] = []
        self.dom_calls = 0
        self._png_marker: bytes = b"\x00"

    def save_screenshot(self, observation: dict[str, Any], directory: str) -> Path:
        self.saved.append((observation, directory))
        image_id = observation["screenshot"]["image_id"]
        target = Path(directory) / f"{image_id}.png"
        target.write_bytes(self.png or b"")
        return target

    def get_screenshot(self, image_id: str) -> bytes:
        self.screenshot_ids.append(image_id)
        return self.png or b""

    def get_html(self, html_id: str) -> str:
        self.html_ids.append(html_id)
        return self.html or ""

    def get_dom(self, dom_id: str) -> Any:  # pragma: no cover - must not be called
        self.dom_calls += 1
        raise AssertionError("DOM must be read from observation.dom.root")


def _metadata(png: bytes, *, image_id: str = "a" * 32) -> dict[str, Any]:
    width, height = struct.unpack(">II", png[16:24])
    return {
        "image_id": image_id,
        "width": width,
        "height": height,
        "mime_type": "image/png",
        "step": 1,
        "byte_size": len(png),
        "sha256": _sha(png),
    }


def _html_metadata(html: str, *, html_id: str = "b" * 32) -> dict[str, Any]:
    encoded = html.encode("utf-8")
    return {
        "html_id": html_id,
        "step": 1,
        "byte_size": len(encoded),
        "sha256": _sha(encoded),
        "mime_type": "text/html",
    }


def _payload(
    *,
    url: str = START,
    text: str = "[1] textbox 'Query'\n[2] button 'Search'",
    **extra: Any,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "type": "accessibility_tree",
        "url": url,
        "text": text,
    }
    observation.update(extra)
    return {
        "protocol_version": "1.0",
        "run_id": "run-a",
        "step": 0,
        "observation": observation,
        "history": [],
        "action_history": [],
    }


@pytest.fixture
def official(tmp_path: Path, clock: FakeClock):
    browser = FakeBrowser()
    session = Session(
        "run-a",
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        page_id="official-page",
    )
    session.start_task("official offline task")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=browser,
        download_artifacts=frozenset(),
    )
    try:
        yield session, backend, browser
    finally:
        session.close()


def _current_action(backend: OfficialBackend, tool: str, args: dict[str, Any]) -> WebAction:
    observation = backend.current_observation
    assert observation is not None
    return build_web_action(ToolRequest(tool, args), observation, action_id=f"act-{tool}")


def test_ax_numbers_are_the_only_action_numbers(official) -> None:
    _session, backend, _browser = official
    observation = backend.from_request(
        _payload(
            dom={
                "root": {
                    "nodeId": 1,
                    "backendNodeId": 99,
                    "nodeName": "body",
                    "children": [],
                }
            }
        )
    )
    assert sorted(observation.elements, key=int) == ["1", "2"]
    assert observation.elements["2"].role == "button"
    assert observation.elements["2"].name == "Search"
    assert "99" not in observation.elements
    assert observation.capabilities.dom is True
    assert observation.artifacts.dom is not None and observation.artifacts.dom.available

    wrong = WebAction(
        action_id="act-99",
        run_id="run-a",
        page_id="official-page",
        observation_id=observation.observation_id,
        op=ActionOp.CLICK,
        args={"target": "99"},
    )
    with pytest.raises(ActionRejected) as error:
        backend.serialize_action(wrong)
    assert error.value.reason is RejectReason.UNKNOWN_ELEMENT


def test_action_commands_are_exact_and_escape_chinese_newlines_and_brackets(official) -> None:
    _session, backend, _browser = official
    observation = backend.from_request(_payload())
    assert observation is not None

    click = _current_action(backend, "web_click", {"target": "2"})
    assert backend.serialize_action(click) == "click [2]"

    original = "价格（含税）\n第二行 [备注] C:\\temp"
    type_action = _current_action(
        backend, "web_type", {"target": "1", "text": original, "submit": False}
    )
    escaped = r"价格（含税）\n第二行 \[备注\] C:\\temp"
    command = backend.serialize_action(type_action)
    assert command == f"type [1] [{escaped}] [0]"
    assert "\n" not in command and "\r" not in command
    assert unescape_text(escaped) == original

    submitted = _current_action(
        backend, "web_type", {"target": "1", "text": "中文", "submit": True}
    )
    assert backend.serialize_action(submitted) == "type [1] [中文] [1]"

    press = _current_action(backend, "web_press", {"keys": "Ctrl+A"})
    assert backend.serialize_action(press) == "press [Ctrl+A]"
    scroll = _current_action(backend, "web_scroll", {"direction": "down"})
    assert backend.serialize_action(scroll) == "scroll [down]"
    back = _current_action(backend, "web_back", {})
    forward = _current_action(backend, "web_forward", {})
    assert backend.serialize_action(back) == "go_back"
    assert backend.serialize_action(forward) == "go_forward"


def test_stop_uses_the_same_escaping_and_keeps_one_bracket_pair() -> None:
    command = serialize_stop("已完成]\n再见")
    assert command == r"stop [已完成\]\n再见]"
    assert command.count("]") == 2  # the escaped text bracket plus the closing one
    assert "\n" not in command


def test_goto_must_use_a_same_origin_url_observed_in_this_task(official) -> None:
    _session, backend, _browser = official
    backend.from_request(_payload(url=NEXT))  # observe the target once
    observation = backend.from_request(_payload(url=START))
    assert sorted(backend.observed_urls) == sorted([NEXT, START])

    allowed = _current_action(backend, "web_goto", {"url": NEXT})
    assert backend.serialize_action(allowed) == f"goto [{NEXT}]"

    unobserved = WebAction(
        action_id="act-unseen",
        run_id="run-a",
        page_id="official-page",
        observation_id=observation.observation_id,
        op=ActionOp.GOTO,
        args={"url": "http://example.test/unseen"},
    )
    with pytest.raises(ActionRejected) as error:
        backend.serialize_action(unobserved)
    assert error.value.reason is RejectReason.EXTERNAL_ORIGIN

    cross_origin = WebAction(
        action_id="act-cross",
        run_id="run-a",
        page_id="official-page",
        observation_id=observation.observation_id,
        op=ActionOp.GOTO,
        args={"url": "http://other.test/"},
    )
    with pytest.raises(ActionRejected):
        backend.serialize_action(cross_origin)


def test_stale_action_is_rejected_after_a_new_observation(official) -> None:
    _session, backend, _browser = official
    first = backend.from_request(_payload())
    old_action = build_web_action(ToolRequest("web_click", {"target": "2"}), first)
    backend.from_request(_payload(url=NEXT, text="[1] link 'Next'"))
    with pytest.raises(ActionRejected) as error:
        backend.serialize_action(old_action)
    assert error.value.reason is RejectReason.STALE_OBSERVATION


def test_unrepresentable_action_content_is_an_explicit_error(official) -> None:
    _session, backend, _browser = official
    backend.from_request(_payload())
    surrogate = _current_action(
        backend, "web_type", {"target": "1", "text": "bad \ud800 text"}
    )
    with pytest.raises(OfficialActionFormatError):
        backend.serialize_action(surrogate)

    backend.from_request(_payload(url="http://example.test/a]b", text="[1] link 'x'"))
    bad_url = _current_action(
        backend, "web_goto", {"url": "http://example.test/a]b"}
    )
    with pytest.raises(OfficialActionFormatError):
        backend.serialize_action(bad_url)


def test_ax_only_still_produces_a_decision_observation(official) -> None:
    _session, backend, _browser = official
    observation = backend.from_request(_payload())
    assert observation.ax_text.startswith("[1]")
    assert observation.artifacts.screenshot is None
    assert observation.artifacts.html is None
    assert observation.artifacts.dom is None
    assert observation.capabilities.screenshot is False
    assert observation.capabilities.html is False
    assert observation.capabilities.dom is False
    action = _current_action(backend, "web_click", {"target": "2"})
    assert backend.serialize_action(action) == "click [2]"


def test_screenshot_html_dom_are_downloaded_verified_and_not_reused(
    tmp_path: Path, clock: FakeClock
) -> None:
    png = _png()
    html = "<html><body>中文 HTML</body></html>"
    browser = FakeBrowser(png=png, html=html)
    session = Session("run-a", runtime_dir=tmp_path / "runtime", clock=clock, page_id="official-page")
    session.start_task("artifacts")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=browser,
        download_artifacts=frozenset({"screenshot", "html"}),
    )
    try:
        first = backend.from_request(
            _payload(
                screenshot=_metadata(png),
                html=_html_metadata(html),
                dom={"root": {"nodeId": 1, "nodeName": "body", "children": []}},
            )
        )
        assert first.artifacts.screenshot is not None and first.artifacts.screenshot.available
        assert first.artifacts.html is not None and first.artifacts.html.available
        assert first.artifacts.dom is not None and first.artifacts.dom.available
        assert session.evidence.verify(first.artifacts.screenshot)
        assert session.evidence.verify(first.artifacts.html)
        assert session.evidence.verify(first.artifacts.dom)
        assert first.artifacts.screenshot.sha256 == _sha(png)
        assert first.artifacts.html.sha256 == _sha(html.encode("utf-8"))
        assert browser.saved
        assert browser.html_ids == ["b" * 32]
        assert browser.dom_calls == 0

        second = backend.from_request(_payload(url=NEXT, text="[1] link 'Next'"))
        assert second.observation_id != first.observation_id
        assert second.artifacts.screenshot is None
        assert second.artifacts.html is None
        assert second.artifacts.dom is None
        reused = backend.download_artifact("screenshot")
        assert reused.available is False
        assert reused.reason == "not_in_this_observation"
    finally:
        session.close()


def test_unavailable_reasons_are_preserved(official) -> None:
    _session, backend, _browser = official
    observation = backend.from_request(
        _payload(
            screenshot={"available": False, "reason": "size_limit"},
            html={"available": False, "reason": "capture_failed"},
            dom={"available": False, "reason": "size_limit"},
        )
    )
    assert observation.artifacts.screenshot is not None
    assert observation.artifacts.screenshot.reason == "size_limit"
    assert observation.artifacts.html is not None
    assert observation.artifacts.html.reason == "capture_failed"
    assert observation.artifacts.dom is not None
    assert observation.artifacts.dom.reason == "size_limit"
    # Metadata presence advertises the capability, while the artifact is unavailable.
    assert observation.capabilities.screenshot is True
    assert observation.capabilities.html is True
    assert observation.capabilities.dom is True


def test_integrity_failure_is_visible_and_never_referenced(
    tmp_path: Path, clock: FakeClock
) -> None:
    png = _png()
    browser = FakeBrowser(png=png)
    session = Session("run-a", runtime_dir=tmp_path / "runtime", clock=clock, page_id="official-page")
    session.start_task("integrity")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=browser,
        download_artifacts=frozenset({"screenshot"}),
    )
    try:
        bad = _metadata(png)
        bad["sha256"] = "0" * 64
        observation = backend.from_request(_payload(screenshot=bad))
        artifact = observation.artifacts.screenshot
        assert artifact is not None
        assert artifact.available is False
        assert artifact.reason == "integrity_mismatch"
        assert artifact.reference is None
        events = read_events(session.evidence.events_path)
        errors = [event for event in events if event["event_type"] == "artifact_error"]
        assert errors
        assert errors[0]["result"]["kind"] == "screenshot"
        assert errors[0]["result"]["reason"] == "integrity_mismatch"
        assert "do not match" in errors[0]["result"]["message"]
    finally:
        session.close()


def test_html_integrity_failure_is_visible(
    tmp_path: Path, clock: FakeClock
) -> None:
    html = "<html>tampered</html>"
    browser = FakeBrowser(html=html)
    session = Session("run-a", runtime_dir=tmp_path / "runtime", clock=clock, page_id="official-page")
    session.start_task("bad html")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=browser,
        download_artifacts=frozenset({"html"}),
    )
    try:
        metadata = _html_metadata(html)
        metadata["byte_size"] += 1
        observation = backend.from_request(_payload(html=metadata))
        artifact = observation.artifacts.html
        assert artifact is not None and artifact.available is False
        assert artifact.reason == "integrity_mismatch"
        assert backend.artifact_errors[-1]["kind"] == "html"
        assert backend.artifact_errors[-1]["reason"] == "integrity_mismatch"
    finally:
        session.close()


def test_html_is_downloaded_on_demand_and_verified(
    tmp_path: Path, clock: FakeClock
) -> None:
    html = "<html>on demand</html>"
    browser = FakeBrowser(html=html)
    session = Session("run-a", runtime_dir=tmp_path / "runtime", clock=clock, page_id="official-page")
    session.start_task("lazy html")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=browser,
        download_artifacts=frozenset(),
    )
    try:
        observation = backend.from_request(_payload(html=_html_metadata(html)))
        pending = observation.artifacts.html
        assert pending is not None and pending.available is False
        assert pending.reason == "not_downloaded"
        loaded = backend.download_artifact("html")
        assert loaded.available is True
        assert loaded.sha256 == _sha(html.encode("utf-8"))
        assert session.evidence.verify(loaded)
        assert browser.html_ids == ["b" * 32]
    finally:
        session.close()


def test_evidence_save_can_load_a_lazy_official_html_artifact(
    tmp_path: Path, clock: FakeClock
) -> None:
    html = "<html>lazy evidence</html>"
    browser = FakeBrowser(html=html)
    session = Session("run-a", runtime_dir=tmp_path / "runtime", clock=clock, page_id="official-page")
    session.start_task("lazy evidence")
    backend = OfficialBackend(
        "run-a",
        session=session,
        evidence=session.evidence,
        browser_api=browser,
        download_artifacts=frozenset(),
    )
    try:
        observation = backend.from_request(_payload(html=_html_metadata(html)))
        session.ingest_observation(observation)
        runtime = HarnessToolRuntime(session, artifact_loader=backend.download_artifact)
        message = runtime.save_evidence("html", "lazy")
        assert "Saved html evidence" in message
        assert browser.html_ids == ["b" * 32]
    finally:
        session.close()


def test_missing_new_fields_still_parse_and_dom_never_calls_get_dom(official) -> None:
    _session, backend, browser = official
    observation = backend.from_request(
        _payload(dom={"root": {"nodeId": 7, "nodeName": "main", "children": []}})
    )
    assert observation.artifacts.dom is not None and observation.artifacts.dom.available
    assert browser.dom_calls == 0
    without = backend.from_request(_payload(url=NEXT, text="[1] link 'Next'"))
    assert without.artifacts.dom is None
    assert without.artifacts.screenshot is None
    assert without.artifacts.html is None


def test_parse_ax_handles_indentation_and_quoted_names() -> None:
    elements = parse_ax_elements(
        "  [3] link 'Result'\n[12] textbox \"Query\"\nnot [99] an id",
        page_id="page",
        observation_id="obs",
    )
    assert sorted(elements, key=int) == ["3", "12"]
    assert elements["3"].name == "Result"
    assert elements["12"].name == "Query"
