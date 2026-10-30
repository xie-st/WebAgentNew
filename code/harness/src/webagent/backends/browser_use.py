"""browser-use browser-layer adapter (experimental, no planning Agent).

The adapter uses only ``browser_use.Browser`` (BrowserSession), ``Page``,
``Element`` and ``Mouse``. It never constructs ``browser_use.Agent`` and never
uses browser-use's prompt/tool loop: EffGen still decides every action.

Two facts from the trial are encoded explicitly:

* default extensions are disabled, because browser-use otherwise downloads them
  from the public internet at browser start;
* the cloud client's ``httpx.AsyncClient`` is constructed with ``trust_env=False``
  for this backend. The installed browser-use build cannot parse this Windows
  host's ``NO_PROXY`` shape and would otherwise fail before opening a page.

The browser profile is private to this run. This backend is not selected by
default in B1.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import uuid
from pathlib import Path
from typing import Any, Awaitable, TypeVar

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
_T = TypeVar("_T")


async def _quiet_evaluate(page: Any, script: str) -> Any:
    """Call browser-use's async evaluate without its debug print."""

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return await page.evaluate(script)


class _AsyncLoop:
    """One persistent event loop, driven synchronously without a worker thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run(self, awaitable: Awaitable[_T]) -> _T:
        return self.loop.run_until_complete(awaitable)

    def close(self) -> None:
        try:
            pending = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        except Exception:  # noqa: BLE001 - cleanup is best effort
            pass
        finally:
            asyncio.set_event_loop(None)
            self.loop.close()


@contextlib.contextmanager
def _browser_without_env_proxy() -> Any:
    """Construct Browser even when httpx cannot parse the host proxy variables."""

    from browser_use.browser.cloud import cloud as cloud_module

    original = cloud_module.httpx.AsyncClient

    class _NoEnvClient(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("trust_env", False)
            super().__init__(*args, **kwargs)

    cloud_module.httpx.AsyncClient = _NoEnvClient
    try:
        yield
    finally:
        cloud_module.httpx.AsyncClient = original


class BrowserUseBackend:
    backend_id = "browser-use"

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
            raise ProtocolError("BrowserUseBackend needs the session EvidenceStore")
        if evidence.run_id != run_id:
            raise ProtocolError("BrowserUseBackend evidence belongs to another run")
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
        self.page_id = f"page-bu-{uuid.uuid4().hex[:12]}"
        self._runner: _AsyncLoop | None = None
        self._browser: Any | None = None
        self._page: Any | None = None
        self._observation_counter = 0
        self._current_observation: Observation | None = None
        self._closed = False
        self._started = False
        logging.getLogger("browser_use").setLevel(logging.WARNING)

    # ---------------------------------------------------------------- lifecycle
    @property
    def current_url(self) -> str | None:
        if self._page is None or self._runner is None:
            return None
        try:
            return self._runner.run(self._page.get_url())
        except Exception:  # noqa: BLE001 - diagnostic only
            return None

    @property
    def current_observation(self) -> Observation | None:
        return self._current_observation

    def start(self, url: str | None = None) -> None:
        if self._closed:
            raise SessionStateError("browser backend is closed")
        if self._started:
            raise SessionStateError("browser backend is already started")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._runner = _AsyncLoop()
        try:
            with _browser_without_env_proxy():
                self._browser = self._make_browser()
                self._runner.run(self._start_async(url))
        except Exception:
            self.close()
            raise
        self._started = True

    def _make_browser(self) -> Any:
        from browser_use import Browser

        return Browser(
            headless=self.headless,
            user_data_dir=str(self.profile_dir),
            window_size=self.viewport,
            enable_default_extensions=False,
            highlight_elements=False,
            use_cloud=False,
            keep_alive=False,
        )

    async def _start_async(self, url: str | None) -> None:
        assert self._browser is not None
        await self._browser.start()
        self._page = await self._browser.new_page(url or "about:blank")
        if url:
            await self._wait_stable_async()
        await self._browser.get_state_as_text()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runner = self._runner
        browser = self._browser
        self._page = None
        self._browser = None
        self._runner = None
        if runner is not None and browser is not None:
            try:
                runner.run(browser.stop())
            except Exception:  # noqa: BLE001 - close must be idempotent
                pass
        if runner is not None:
            runner.close()
        self._current_observation = None

    # ---------------------------------------------------------------- observe
    def observe(self) -> Observation:
        if self._runner is None:
            raise SessionStateError("browser backend has not started")
        return self._runner.run(self._observe_async())

    async def _observe_async(
        self,
        *,
        previous_action_id: str | None = None,
        status: FeedbackStatus = FeedbackStatus.OBSERVED,
        message: str = "",
        error_type: str | None = None,
        page_changed: bool | None = None,
    ) -> Observation:
        await self._wait_stable_async()
        page = self._require_page()
        raw = await _quiet_evaluate(page, AX_SNAPSHOT_JS.strip())
        snapshot = parse_snapshot(raw)
        observation_id = f"obs-{self._observation_counter}"
        self._observation_counter += 1
        elements: dict[str, ElementRef] = {}
        for index, raw_element in enumerate(snapshot["elements"], start=1):
            number = str(index)
            elements[number] = ElementRef(
                page_id=self.page_id,
                observation_id=observation_id,
                number=number,
                role=str(raw_element.get("role", "generic")),
                name=str(raw_element.get("name", "")),
                localization={
                    "css_path": str(raw_element.get("css_path", "")),
                    "tag": str(raw_element.get("tag", "")),
                    "type": str(raw_element.get("type", "")),
                    "value": str(raw_element.get("value", "")),
                    "href": str(raw_element.get("href", "")),
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
        artifacts = ArtifactSet(
            screenshot=await self._capture_screenshot_async(observation_id, snapshot["url"])
        )
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

    async def _capture_screenshot_async(self, observation_id: str, url: str) -> Artifact:
        if not self.screenshot_enabled:
            return Artifact(kind="screenshot", available=False, reason="disabled")
        try:
            encoded = await self._require_page().screenshot(format="png")
            data = base64.b64decode(encoded, validate=True)
            if not data.startswith(b"\x89PNG"):
                raise ValueError("screenshot is not a PNG")
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
        if self._runner is None:
            raise SessionStateError("browser backend has not started")
        return self._runner.run(self._execute_async(action))

    async def _execute_async(self, action: WebAction) -> ExecutionResult:
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
        await self._wait_stable_async()
        previous_url = await page.get_url()
        previous_title = await page.get_title()
        status = FeedbackStatus.OBSERVED
        error_type: str | None = None
        message = ""
        try:
            if action.op is ActionOp.CLICK:
                element = await self._element_for_async(target)
                await element.click()
                message = f"clicked element {action.target} ({target.role} {target.name!r})"
            elif action.op is ActionOp.TYPE:
                element = await self._element_for_async(target)
                await element.focus()
                await page.press("End")
                await element.fill(action.text or "", clear=False)
                message = f"typed into element {action.target} without clearing"
                if action.submit:
                    await page.press("Enter")
                    message += " then pressed Enter"
            elif action.op is ActionOp.PRESS:
                await page.press(action.keys or "")
                message = f"pressed {action.keys!r}"
            elif action.op is ActionOp.SCROLL:
                delta = int(self.viewport["height"] * 0.8)
                await page.mouse.scroll(
                    x=0, y=0, delta_y=delta if action.direction == "down" else -delta
                )
                message = f"scrolled {action.direction}"
            elif action.op is ActionOp.GOTO:
                await page.goto(action.url or "")
                message = f"navigated to {action.url}"
            elif action.op is ActionOp.GO_BACK:
                await page.go_back()
                message = "went back"
            elif action.op is ActionOp.GO_FORWARD:
                await page.go_forward()
                message = "went forward"
            else:  # pragma: no cover - ActionOp is exhaustive
                raise ProtocolError(f"unsupported browser action {action.op!r}")
        except Exception as exc:  # noqa: BLE001 - the page feedback must survive
            status = FeedbackStatus.ERROR
            error_type = "action_failed"
            message = f"{type(exc).__name__}: {str(exc)[:300]}"
        if status is FeedbackStatus.OBSERVED:
            await self._wait_stable_async()
        new_url = await page.get_url()
        new_title = await page.get_title()
        page_changed = (new_url != previous_url) or (new_title != previous_title)
        if action.op in (ActionOp.GO_BACK, ActionOp.GO_FORWARD) and (
            status is FeedbackStatus.OBSERVED and new_url == previous_url
        ):
            status = FeedbackStatus.ERROR
            error_type = "no_history"
            message = f"{action.op.value} produced no page change"
        new_observation = await self._observe_async(
            previous_action_id=action.action_id,
            status=status,
            message=message,
            error_type=error_type,
            page_changed=page_changed,
        )
        assert new_observation.feedback is not None
        return ExecutionResult(action.action_id, new_observation.feedback, new_observation)

    # ---------------------------------------------------------------- internals
    def _require_page(self) -> Any:
        if self._closed:
            raise SessionStateError("browser backend is closed")
        if self._page is None:
            raise SessionStateError("browser backend has not started")
        return self._page

    async def _element_for_async(self, target: ElementRef | None) -> Any:
        if target is None:
            raise ProtocolError("this action has no element target")
        css_path = str((target.localization or {}).get("css_path", "")).strip()
        if not css_path:
            raise ProtocolError("element localization has no CSS path")
        elements = await self._require_page().get_elements_by_css_selector(css_path)
        if not elements:
            raise ProtocolError("element is no longer in the rendered DOM")
        return elements[0]

    async def _wait_stable_async(self) -> None:
        if self._page is None:
            return
        await asyncio.sleep(0.15)
        for _ in range(20):
            try:
                state = await _quiet_evaluate(self._page, "() => document.readyState")
            except Exception:  # noqa: BLE001 - retry while the page settles
                await asyncio.sleep(0.1)
                continue
            if str(state).strip() == "complete":
                break
            await asyncio.sleep(0.1)


__all__ = ["BrowserUseBackend"]
