"""Headed (default) Playwright browser backend with a private profile.

The backend owns exactly one Chromium persistent context. The profile lives under
``runtime/browser_profiles/<run_id>/`` and is never a user's daily browser
profile. It observes the current rendered DOM, builds a fresh element table for
every observation, and executes actions through Playwright's real input APIs --
never by dispatching synthetic JavaScript click events.

Screenshots are saved through the session's :class:`EvidenceStore`, so each
artifact carries its actual URL, page id and observation id.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import (
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from ..errors import ActionRejected, ProtocolError, RejectReason, SessionStateError
from ..evidence import EvidenceStore
from ..protocol import (
    ActionOp,
    Artifact,
    ArtifactSet,
    Capabilities,
    ElementRef,
    ExecutionResult,
    Feedback,
    FeedbackStatus,
    Observation,
    WebAction,
    validate_run_id,
)
from ..tools import resolve_target
from .browser_js import AX_SNAPSHOT_JS, parse_snapshot

_DEFAULT_VIEWPORT = {"width": 1280, "height": 800}


class PlaywrightBackend:
    backend_id = "playwright"

    def __init__(
        self,
        run_id: str,
        *,
        evidence: EvidenceStore,
        headless: bool = False,
        runtime_dir: str | Path = "runtime",
        profile_dir: str | Path | None = None,
        viewport: dict[str, int] | None = None,
        action_timeout_ms: int = 10_000,
        navigation_timeout_ms: int = 20_000,
        stable_timeout_ms: int = 5_000,
        screenshot: bool = True,
    ) -> None:
        validate_run_id(run_id)
        if not isinstance(evidence, EvidenceStore):
            raise ProtocolError("PlaywrightBackend needs the session EvidenceStore")
        if evidence.run_id != run_id:
            raise ProtocolError("PlaywrightBackend evidence belongs to another run")
        if action_timeout_ms < 1 or navigation_timeout_ms < 1 or stable_timeout_ms < 1:
            raise ProtocolError("browser timeouts must be positive")
        self.run_id = run_id
        self.evidence = evidence
        self.headless = bool(headless)
        self.runtime_dir = Path(runtime_dir)
        self.profile_dir = (
            Path(profile_dir)
            if profile_dir is not None
            else self.runtime_dir / "browser_profiles" / run_id
        )
        self.viewport = dict(viewport or _DEFAULT_VIEWPORT)
        self.action_timeout_ms = int(action_timeout_ms)
        self.navigation_timeout_ms = int(navigation_timeout_ms)
        self.stable_timeout_ms = int(stable_timeout_ms)
        self.screenshot_enabled = bool(screenshot)
        self.page_id = f"page-{uuid.uuid4().hex[:12]}"
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._observation_counter = 0
        self._current_observation: Observation | None = None
        self._closed = False
        self._started = False

    # ---------------------------------------------------------------- lifecycle
    @property
    def current_url(self) -> str | None:
        return None if self._page is None else self._page.url

    @property
    def current_observation(self) -> Observation | None:
        return self._current_observation

    def start(self, url: str | None = None) -> None:
        if self._closed:
            raise SessionStateError("browser backend is closed")
        if self._started:
            raise SessionStateError("browser backend is already started")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        try:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                viewport=self.viewport,
                accept_downloads=False,
                args=["--disable-background-networking", "--disable-extensions"],
            )
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
            self._page.set_default_timeout(self.action_timeout_ms)
            self._page.set_default_navigation_timeout(self.navigation_timeout_ms)
            if url is not None:
                self._navigate(url)
        except Exception:
            self.close()
            raise
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        context = self._context
        self._context = None
        self._page = None
        if context is not None:
            try:
                context.close()
            except Exception:  # noqa: BLE001 - close must be idempotent
                pass
        playwright = self._playwright
        self._playwright = None
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:  # noqa: BLE001 - close must be idempotent
                pass
        self._current_observation = None

    # ---------------------------------------------------------------- observe
    def observe(self) -> Observation:
        return self._observe()

    def _observe(
        self,
        *,
        previous_action_id: str | None = None,
        status: FeedbackStatus = FeedbackStatus.OBSERVED,
        message: str = "",
        error_type: str | None = None,
        page_changed: bool | None = None,
    ) -> Observation:
        page = self._require_page()
        self._wait_for_stable(page)
        snapshot = parse_snapshot(page.evaluate(AX_SNAPSHOT_JS))
        observation_id = f"obs-{self._observation_counter}"
        self._observation_counter += 1
        elements: dict[str, ElementRef] = {}
        for index, raw in enumerate(snapshot["elements"], start=1):
            number = str(index)
            elements[number] = ElementRef(
                page_id=self.page_id,
                observation_id=observation_id,
                number=number,
                role=str(raw.get("role", "generic")),
                name=str(raw.get("name", "")),
                localization={
                    "css_path": str(raw.get("css_path", "")),
                    "tag": str(raw.get("tag", "")),
                    "type": str(raw.get("type", "")),
                    "value": str(raw.get("value", "")),
                    "href": str(raw.get("href", "")),
                },
            )
        feedback = None
        if previous_action_id is not None:
            feedback = Feedback(
                status=status,
                action_id=previous_action_id,
                observation_id=observation_id,
                message=message,
                error_type=error_type,
                page_changed=page_changed,
            )
        artifacts = ArtifactSet(screenshot=self._capture_screenshot(observation_id, snapshot["url"]))
        observation = Observation(
            run_id=self.run_id,
            page_id=self.page_id,
            observation_id=observation_id,
            url=snapshot["url"],
            ax_text=snapshot["ax_text"],
            elements=elements,
            previous_action_id=previous_action_id,
            feedback=feedback,
            artifacts=artifacts,
            capabilities=Capabilities(screenshot=self.screenshot_enabled, html=False, dom=False),
        )
        self._current_observation = observation
        return observation

    def _capture_screenshot(self, observation_id: str, url: str) -> Artifact:
        if not self.screenshot_enabled:
            return Artifact(kind="screenshot", available=False, reason="disabled")
        page = self._require_page()
        try:
            data = page.screenshot(type="png", full_page=False, timeout=self.action_timeout_ms)
            return self.evidence.save_bytes(
                "screenshot",
                data,
                observation_id=observation_id,
                metadata={
                    "url": url,
                    "page_id": self.page_id,
                    "observation_id": observation_id,
                    "viewport": True,
                },
            )
        except Exception as exc:  # noqa: BLE001 - report a real capture failure
            return Artifact(
                kind="screenshot",
                available=False,
                reason=f"capture_failed: {type(exc).__name__}: {str(exc)[:200]}",
                metadata={"url": url, "page_id": self.page_id, "observation_id": observation_id},
            )

    # ---------------------------------------------------------------- execute
    def execute(self, action: WebAction) -> ExecutionResult:
        if self._closed:
            raise SessionStateError("browser backend is closed")
        page = self._require_page()
        observation = self._current_observation
        if observation is None:
            raise SessionStateError("browser backend has no current observation")
        if action.run_id != self.run_id:
            raise ActionRejected(
                RejectReason.WRONG_RUN,
                details=["action belongs to another run"],
                correction="Use the action built for this task.",
            )
        if action.page_id != self.page_id:
            raise ActionRejected(
                RejectReason.WRONG_PAGE,
                details=["action belongs to another page"],
                correction="Use the current page observation.",
            )
        if action.observation_id != observation.observation_id:
            raise ActionRejected(
                RejectReason.STALE_OBSERVATION,
                details=[f"action observation is not current {observation.observation_id!r}"],
                correction=(
                    "Re-observe after pause/resume and build a new action with the new numbers."
                ),
            )
        target = resolve_target(action, observation)
        self._wait_for_stable(page)
        previous_url = page.url
        previous_ax = observation.ax_text
        status = FeedbackStatus.OBSERVED
        error_type: str | None = None
        message = ""
        try:
            if action.op is ActionOp.CLICK:
                locator = self._locator_for(target)
                locator.click(timeout=self.action_timeout_ms)
                message = f"clicked element {action.target} ({target.role} {target.name!r})"
            elif action.op is ActionOp.TYPE:
                locator = self._locator_for(target)
                locator.click(timeout=self.action_timeout_ms)
                # Append semantics: focus, move to the end, type; never clear.
                page.keyboard.press("End")
                page.keyboard.type(action.text or "", delay=0)
                message = f"typed into element {action.target} without clearing"
                if action.submit:
                    page.keyboard.press("Enter")
                    message += " then pressed Enter"
            elif action.op is ActionOp.PRESS:
                page.keyboard.press(action.keys or "")
                message = f"pressed {action.keys!r}"
            elif action.op is ActionOp.SCROLL:
                delta = int(self.viewport["height"] * 0.8)
                page.mouse.wheel(0, delta if action.direction == "down" else -delta)
                message = f"scrolled {action.direction}"
            elif action.op is ActionOp.GOTO:
                self._navigate(action.url or "")
                message = f"navigated to {action.url}"
            elif action.op is ActionOp.GO_BACK:
                response = page.go_back(wait_until="domcontentloaded", timeout=self.navigation_timeout_ms)
                if response is None:
                    raise PlaywrightTimeoutError("no back history entry")
                message = "went back"
            elif action.op is ActionOp.GO_FORWARD:
                response = page.go_forward(wait_until="domcontentloaded", timeout=self.navigation_timeout_ms)
                if response is None:
                    raise PlaywrightTimeoutError("no forward history entry")
                message = "went forward"
            else:  # pragma: no cover - ActionOp is exhaustive
                raise ProtocolError(f"unsupported browser action {action.op!r}")
        except PlaywrightTimeoutError as exc:
            status = FeedbackStatus.ERROR
            error_type = "action_timeout"
            message = f"browser action timed out: {str(exc)[:200]}"
        except Exception as exc:  # noqa: BLE001 - the page feedback must survive
            status = FeedbackStatus.ERROR
            error_type = "action_failed"
            message = f"{type(exc).__name__}: {str(exc)[:300]}"
        if status is FeedbackStatus.OBSERVED:
            self._wait_for_stable(page)
        new_ax = self._quick_ax(page)
        page_changed = (page.url != previous_url) or (new_ax != previous_ax)
        observation = self._observe(
            previous_action_id=action.action_id,
            status=status,
            message=message,
            error_type=error_type,
            page_changed=page_changed,
        )
        assert observation.feedback is not None
        return ExecutionResult(action.action_id, observation.feedback, observation)

    # ---------------------------------------------------------------- internals
    def _require_page(self) -> Page:
        if self._closed:
            raise SessionStateError("browser backend is closed")
        if self._page is None:
            raise SessionStateError("browser backend has not started")
        return self._page

    def _locator_for(self, target: ElementRef | None) -> Any:
        if target is None:
            raise ProtocolError("this action has no element target")
        css_path = str((target.localization or {}).get("css_path", "")).strip()
        if not css_path:
            raise ProtocolError("element localization has no CSS path")
        page = self._require_page()
        locator = page.locator(css_path)
        if locator.count() < 1:
            raise ProtocolError("element is no longer in the rendered DOM")
        return locator.first

    def _quick_ax(self, page: Page) -> str:
        try:
            title = page.title()
        except Exception:  # noqa: BLE001 - comparison only
            title = ""
        try:
            body = page.locator("body").inner_text(timeout=1_000)
        except Exception:  # noqa: BLE001 - comparison only
            body = ""
        return f"{page.url}\n{title}\n{body}"

    def _navigate(self, url: str) -> None:
        page = self._require_page()
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https", "file") or not (parts.netloc or parts.scheme == "file"):
            raise ProtocolError("browser start URL must be an absolute http(s) file URL")
        page.goto(url, wait_until="domcontentloaded", timeout=self.navigation_timeout_ms)

    def _wait_for_stable(self, page: Page) -> None:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=self.stable_timeout_ms)
        except PlaywrightTimeoutError:
            pass
        try:
            page.wait_for_load_state("load", timeout=self.stable_timeout_ms)
        except PlaywrightTimeoutError:
            pass
        try:
            page.wait_for_function(
                "() => document.readyState === 'complete'",
                timeout=self.stable_timeout_ms,
            )
        except PlaywrightTimeoutError:
            pass
        page.wait_for_timeout(120)


__all__ = ["PlaywrightBackend"]
