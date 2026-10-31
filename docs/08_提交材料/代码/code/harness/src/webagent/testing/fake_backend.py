"""Deterministic offline backend for A1 tests and later A2 integration tests.

It implements the same backend contract as Playwright/official: observe, execute
one consumed action, return feedback plus a fresh observation. It is not a
browser and must never be used for B/C acceptance runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from ..errors import ActionRejected, ProtocolError, RejectReason, SessionStateError
from ..protocol import (
    ActionOp,
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


@dataclass(frozen=True)
class FakeElement:
    role: str
    name: str
    href: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not isinstance(self.name, str):
            raise ProtocolError("fake element role/name must be strings")
        if self.href is not None:
            parts = urlsplit(self.href)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                raise ProtocolError("fake element href must be an absolute http(s) URL")


@dataclass(frozen=True)
class FakePage:
    url: str
    title: str
    ax_text: str
    elements: tuple[FakeElement, ...] = ()

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ProtocolError("fake page url must be an absolute http(s) URL")
        if not isinstance(self.title, str) or not isinstance(self.ax_text, str):
            raise ProtocolError("fake page title/ax_text must be strings")


class FakeSite:
    def __init__(self, pages: Iterable[FakePage]) -> None:
        ordered = tuple(pages)
        if not ordered:
            raise ProtocolError("fake site needs at least one page")
        self.pages = {page.url: page for page in ordered}
        if len(self.pages) != len(ordered):
            raise ProtocolError("fake site page URLs must be unique")
        self.default_url = ordered[0].url

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FakeSite":
        raw_pages = data.get("pages")
        if not isinstance(raw_pages, list):
            raise ProtocolError("fake site needs a pages list")
        pages = []
        for raw in raw_pages:
            if not isinstance(raw, Mapping):
                raise ProtocolError("fake page must be a mapping")
            elements = tuple(
                FakeElement(
                    role=str(element.get("role", "generic")),
                    name=str(element.get("name", "")),
                    href=element.get("href"),
                )
                for element in raw.get("elements", ())
                if isinstance(element, Mapping)
            )
            pages.append(
                FakePage(
                    url=str(raw.get("url", "")),
                    title=str(raw.get("title", "")),
                    ax_text=str(raw.get("ax_text", "")),
                    elements=elements,
                )
            )
        return cls(pages)

    @classmethod
    def from_json(cls, path: str | Path) -> "FakeSite":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def page(self, url: str) -> FakePage | None:
        return self.pages.get(url)


class FakeBackend:
    backend_id = "fake"

    def __init__(
        self,
        run_id: str,
        site: FakeSite,
        *,
        page_id: str = "page-1",
        start_url: str | None = None,
    ) -> None:
        validate_run_id(run_id)
        self.run_id = run_id
        self.site = site
        self.page_id = page_id
        self._counter = 0
        self._current_url: str | None = None
        self._current_observation: Observation | None = None
        self._history: list[str] = []
        self._forward: list[str] = []
        self._values: dict[tuple[str, str], str] = {}
        self._scroll: dict[str, int] = {}
        self._closed = False
        self.executed_actions: list[WebAction] = []
        if start_url is not None:
            self.start(start_url)

    @property
    def current_url(self) -> str | None:
        return self._current_url

    @property
    def current_observation(self) -> Observation | None:
        return self._current_observation

    def start(self, url: str | None = None) -> Observation:
        if self._closed:
            raise SessionStateError("fake backend is closed")
        target = url or self._current_url or self.site.default_url
        if self.site.page(target) is None:
            raise ProtocolError(f"fake site has no page {target!r}")
        self._current_url = target
        self._history.clear()
        self._forward.clear()
        observation = self._make_observation()
        self._current_observation = observation
        return observation

    def observe(self) -> Observation:
        if self._closed:
            raise SessionStateError("fake backend is closed")
        if self._current_url is None:
            raise SessionStateError("fake backend has not started")
        observation = self._make_observation()
        self._current_observation = observation
        return observation

    def execute(self, action: WebAction) -> ExecutionResult:
        if self._closed:
            raise SessionStateError("fake backend is closed")
        current = self._current_observation
        if current is None:
            raise SessionStateError("fake backend has not started")
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
                correction="Use the current page's observation.",
            )
        if action.observation_id != current.observation_id:
            raise ActionRejected(
                RejectReason.STALE_OBSERVATION,
                details=[
                    f"action observation {action.observation_id!r} is not current "
                    f"{current.observation_id!r}"
                ],
                correction="Re-observe and build the action against the current observation.",
            )
        # Full binding/argument validation (including same-origin goto).
        resolve_target(action, current)
        self.executed_actions.append(action)
        status = FeedbackStatus.OBSERVED
        error_type = None
        page_changed = False
        message = ""
        try:
            if action.op is ActionOp.CLICK:
                element = self._element(action.target or "")
                message = f"clicked {element.name!r}"
                if element.href is not None:
                    self._navigate(element.href)
                    page_changed = True
            elif action.op is ActionOp.TYPE:
                target = action.target or ""
                self._element(target)
                previous = self._values.get((self._current_url or "", target), "")
                self._values[(self._current_url or "", target)] = previous + (action.text or "")
                message = f"typed into element {target}; submit={str(action.submit).lower()}"
            elif action.op is ActionOp.PRESS:
                message = f"pressed {action.keys!r}"
            elif action.op is ActionOp.SCROLL:
                url = self._current_url or ""
                self._scroll[url] = self._scroll.get(url, 0) + (1 if action.direction == "down" else -1)
                message = f"scrolled {action.direction}"
                page_changed = True
            elif action.op is ActionOp.GOTO:
                if self.site.page(action.url or "") is None:
                    status = FeedbackStatus.ERROR
                    error_type = "not_found"
                    message = "navigation target is not part of the fake site"
                else:
                    self._navigate(action.url or "")
                    message = f"navigated to {action.url}"
                    page_changed = True
            elif action.op is ActionOp.GO_BACK:
                if not self._history:
                    status = FeedbackStatus.ERROR
                    error_type = "no_history"
                    message = "no back history"
                else:
                    self._forward.append(self._current_url or "")
                    self._current_url = self._history.pop()
                    message = "went back"
                    page_changed = True
            elif action.op is ActionOp.GO_FORWARD:
                if not self._forward:
                    status = FeedbackStatus.ERROR
                    error_type = "no_history"
                    message = "no forward history"
                else:
                    self._history.append(self._current_url or "")
                    self._current_url = self._forward.pop()
                    message = "went forward"
                    page_changed = True
            else:  # pragma: no cover - enum is exhaustive
                raise ProtocolError(f"unsupported fake action {action.op}")
        except KeyError:
            status = FeedbackStatus.ERROR
            error_type = "unknown_element"
            message = "element is not part of the current fake page"
        observation = self._make_observation(
            previous_action_id=action.action_id,
            status=status,
            message=message,
            error_type=error_type,
            page_changed=page_changed,
        )
        self._current_observation = observation
        assert observation.feedback is not None
        return ExecutionResult(action.action_id, observation.feedback, observation)

    def close(self) -> None:
        self._closed = True
        self._current_observation = None

    def _element(self, target: str) -> FakeElement:
        page = self.site.page(self._current_url or "")
        if page is None:
            raise KeyError(self._current_url)
        try:
            index = int(target)
        except ValueError as exc:
            raise KeyError(target) from exc
        if index < 1 or index > len(page.elements):
            raise KeyError(target)
        return page.elements[index - 1]

    def _navigate(self, url: str) -> None:
        if self.site.page(url) is None:
            raise KeyError(url)
        if self._current_url is not None:
            self._history.append(self._current_url)
        self._forward.clear()
        self._current_url = url

    def _make_observation(
        self,
        *,
        previous_action_id: str | None = None,
        status: FeedbackStatus = FeedbackStatus.OBSERVED,
        message: str = "",
        error_type: str | None = None,
        page_changed: bool | None = None,
    ) -> Observation:
        page = self.site.page(self._current_url or "")
        if page is None:
            raise KeyError(self._current_url)
        observation_id = f"obs-{self._counter}"
        self._counter += 1
        elements: dict[str, ElementRef] = {}
        lines = [f"TITLE: {page.title}", f"URL: {page.url}", page.ax_text, "ELEMENTS:"]
        for index, element in enumerate(page.elements, start=1):
            number = str(index)
            value = self._values.get((page.url, number))
            name = element.name if value is None else f"{element.name} (value={value!r})"
            elements[number] = ElementRef(
                page_id=self.page_id,
                observation_id=observation_id,
                number=number,
                role=element.role,
                name=name,
                localization={"fake_index": index, "url": page.url},
            )
            lines.append(f"[{number}] {element.role} {name!r}")
        scroll = self._scroll.get(page.url, 0)
        lines.append(f"SCROLL_POSITION: {scroll}")
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
        return Observation(
            run_id=self.run_id,
            page_id=self.page_id,
            observation_id=observation_id,
            url=page.url,
            ax_text="\n".join(lines),
            elements=elements,
            previous_action_id=previous_action_id,
            feedback=feedback,
            capabilities=Capabilities(),
        )
