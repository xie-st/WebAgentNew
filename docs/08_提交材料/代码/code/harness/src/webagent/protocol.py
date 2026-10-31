"""Immutable protocol types shared by the official and real backends.

The JSON forms keep all protocol meaning. Backend-local localization data is
kept in the local form (evidence) and deliberately excluded from the prompt /
official form: the model only ever sees page, observation and element numbers.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .errors import ProtocolError

SCHEMA_VERSION = "webagent.protocol.v1"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ELEMENT_RE = re.compile(r"^[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ActionOp(str, Enum):
    CLICK = "click"
    TYPE = "type"
    PRESS = "press"
    SCROLL = "scroll"
    GOTO = "goto"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"


class ScrollDirection(str, Enum):
    UP = "up"
    DOWN = "down"


class FeedbackStatus(str, Enum):
    OBSERVED = "observed"
    ERROR = "error"
    UNKNOWN = "unknown"


class TerminationReason(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    DEADLINE = "deadline"
    ERROR = "error"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


ARTIFACT_KINDS = frozenset({"screenshot", "html", "dom"})
ALL_ACTION_OPS = frozenset(ActionOp)


def validate_run_id(run_id: str) -> str:
    """Run ids become directory names; reject anything that is not path-safe."""

    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ProtocolError(
            "run_id must be 1-64 characters of letters, digits, '_' or '-' and "
            "must start with a letter or digit"
        )
    return run_id


def new_action_id() -> str:
    return "act_" + uuid.uuid4().hex


def _as_text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ProtocolError(f"{name} must not be empty")
    return value


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping")
    return MappingProxyType({str(key): item for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _coerce_enum(enum_cls: type[Enum], value: Any, name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(str(item.value) for item in enum_cls)
        raise ProtocolError(f"{name} must be one of: {allowed}") from exc


def json_dumps(value: Any) -> str:
    """Stable JSON for protocol/evidence files (UTF-8, no ASCII escaping)."""

    return json.dumps(_plain(value), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ElementRef:
    """A single element of one observation.

    ``number`` is the model-facing action number. ``localization`` is a local
    backend hint (CDP id, selector, index) and must never be sent to the
    official platform or into a model prompt.
    """

    page_id: str
    observation_id: str
    number: str
    role: str
    name: str
    localization: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _as_text(self.page_id, "element.page_id")
        _as_text(self.observation_id, "element.observation_id")
        number = _as_text(self.number, "element.number")
        if not _ELEMENT_RE.match(number):
            raise ProtocolError("element.number must be a decimal string")
        _as_text(self.role, "element.role", allow_empty=True)
        _as_text(self.name, "element.name", allow_empty=True)
        if self.localization is not None:
            frozen = _freeze(_as_mapping(self.localization, "element.localization"))
            object.__setattr__(self, "localization", frozen)

    def to_local_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "observation_id": self.observation_id,
            "number": self.number,
            "role": self.role,
            "name": self.name,
            "localization": _plain(self.localization),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        return {"number": self.number, "role": self.role, "name": self.name}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ElementRef":
        if not isinstance(data, Mapping):
            raise ProtocolError("element must be a mapping")
        return cls(
            page_id=data.get("page_id", ""),
            observation_id=data.get("observation_id", ""),
            number=data.get("number", ""),
            role=data.get("role", ""),
            name=data.get("name", ""),
            localization=data.get("localization"),
        )


@dataclass(frozen=True)
class Artifact:
    """Reference to one optional observation artifact.

    ``available=False`` records why the artifact is missing; it never points at
    a previous observation's file.
    """

    kind: str
    available: bool
    reference: str | None = None
    sha256: str | None = None
    byte_size: int | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_KINDS:
            raise ProtocolError(f"artifact.kind must be one of {sorted(ARTIFACT_KINDS)}")
        if not isinstance(self.available, bool):
            raise ProtocolError("artifact.available must be a bool")
        frozen_meta = _freeze(_as_mapping(self.metadata, "artifact.metadata"))
        object.__setattr__(self, "metadata", frozen_meta)
        if self.available:
            _as_text(self.reference, "artifact.reference")
            if not isinstance(self.sha256, str) or not _SHA256_RE.match(self.sha256):
                raise ProtocolError("available artifact needs a lowercase sha256 hex digest")
            if not isinstance(self.byte_size, int) or isinstance(self.byte_size, bool) or self.byte_size < 0:
                raise ProtocolError("available artifact needs a non-negative byte_size")
            if self.reason is not None:
                raise ProtocolError("available artifact must not carry an unavailable reason")
        else:
            _as_text(self.reason, "artifact.reason")
            if self.reference is not None or self.sha256 is not None or self.byte_size is not None:
                raise ProtocolError("unavailable artifact must not reference bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "available": self.available,
            "reference": self.reference,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "reason": self.reason,
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Artifact":
        return cls(
            kind=data.get("kind", ""),
            available=data.get("available", False),
            reference=data.get("reference"),
            sha256=data.get("sha256"),
            byte_size=data.get("byte_size"),
            reason=data.get("reason"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class ArtifactSet:
    screenshot: Artifact | None = None
    html: Artifact | None = None
    dom: Artifact | None = None

    def __post_init__(self) -> None:
        for field_name, artifact in (
            ("screenshot", self.screenshot),
            ("html", self.html),
            ("dom", self.dom),
        ):
            if artifact is not None and artifact.kind != field_name:
                raise ProtocolError(f"artifact set field {field_name} holds kind {artifact.kind!r}")

    def get(self, kind: str) -> Artifact | None:
        if kind == "screenshot":
            return self.screenshot
        if kind == "html":
            return self.html
        if kind == "dom":
            return self.dom
        raise ProtocolError(f"unknown artifact kind {kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "screenshot": None if self.screenshot is None else self.screenshot.to_dict(),
            "html": None if self.html is None else self.html.to_dict(),
            "dom": None if self.dom is None else self.dom.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ArtifactSet":
        if not data:
            return cls()
        return cls(
            screenshot=None if data.get("screenshot") is None else Artifact.from_dict(data["screenshot"]),
            html=None if data.get("html") is None else Artifact.from_dict(data["html"]),
            dom=None if data.get("dom") is None else Artifact.from_dict(data["dom"]),
        )


@dataclass(frozen=True)
class Capabilities:
    ax_text: bool = True
    screenshot: bool = False
    html: bool = False
    dom: bool = False
    actions: frozenset[ActionOp] = field(default_factory=lambda: ALL_ACTION_OPS)

    def __post_init__(self) -> None:
        for name in ("ax_text", "screenshot", "html", "dom"):
            if not isinstance(getattr(self, name), bool):
                raise ProtocolError(f"capabilities.{name} must be a bool")
        actions = self.actions
        if not isinstance(actions, frozenset):
            try:
                actions = frozenset(actions)
            except TypeError as exc:
                raise ProtocolError("capabilities.actions must be iterable") from exc
        normalized = frozenset(
            _coerce_enum(ActionOp, item, "capabilities.actions") for item in actions
        )
        object.__setattr__(self, "actions", normalized)

    def allows(self, op: ActionOp | str) -> bool:
        return _coerce_enum(ActionOp, op, "op") in self.actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "ax_text": self.ax_text,
            "screenshot": self.screenshot,
            "html": self.html,
            "dom": self.dom,
            "actions": sorted(op.value for op in self.actions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Capabilities":
        if not data:
            return cls()
        actions = data.get("actions")
        return cls(
            ax_text=data.get("ax_text", True),
            screenshot=data.get("screenshot", False),
            html=data.get("html", False),
            dom=data.get("dom", False),
            actions=tuple(ALL_ACTION_OPS if actions is None else actions),
        )


@dataclass(frozen=True)
class Feedback:
    """What the environment reported after an action.

    ``observed`` only means a page observation was returned; it is not business
    success. No execution evidence means ``unknown``.
    """

    status: FeedbackStatus
    action_id: str
    observation_id: str | None
    message: str
    error_type: str | None = None
    page_changed: bool | None = None

    def __post_init__(self) -> None:
        status = _coerce_enum(FeedbackStatus, self.status, "feedback.status")
        object.__setattr__(self, "status", status)
        _as_text(self.action_id, "feedback.action_id")
        _as_text(self.message, "feedback.message", allow_empty=True)
        if self.observation_id is not None:
            _as_text(self.observation_id, "feedback.observation_id")
        if self.error_type is not None:
            _as_text(self.error_type, "feedback.error_type")
        if self.page_changed is not None and not isinstance(self.page_changed, bool):
            raise ProtocolError("feedback.page_changed must be a bool or null")
        if status is FeedbackStatus.OBSERVED and self.observation_id is None:
            raise ProtocolError("observed feedback must reference the new observation")
        if status is FeedbackStatus.ERROR and self.error_type is None:
            raise ProtocolError("error feedback must name an error_type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "action_id": self.action_id,
            "observation_id": self.observation_id,
            "message": self.message,
            "error_type": self.error_type,
            "page_changed": self.page_changed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Feedback":
        return cls(
            status=data.get("status"),
            action_id=data.get("action_id", ""),
            observation_id=data.get("observation_id"),
            message=data.get("message", ""),
            error_type=data.get("error_type"),
            page_changed=data.get("page_changed"),
        )


@dataclass(frozen=True)
class Observation:
    run_id: str
    page_id: str
    observation_id: str
    url: str
    ax_text: str
    elements: Mapping[str, ElementRef] = field(default_factory=dict)
    previous_action_id: str | None = None
    feedback: Feedback | None = None
    artifacts: ArtifactSet = field(default_factory=ArtifactSet)
    capabilities: Capabilities = field(default_factory=Capabilities)

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        _as_text(self.page_id, "observation.page_id")
        _as_text(self.observation_id, "observation.observation_id")
        _as_text(self.url, "observation.url")
        if not isinstance(self.ax_text, str):
            raise ProtocolError("observation.ax_text must be a string")
        if not isinstance(self.elements, Mapping):
            raise ProtocolError("observation.elements must be a mapping")
        frozen_elements: dict[str, ElementRef] = {}
        for key, element in self.elements.items():
            if not isinstance(key, str) or not _ELEMENT_RE.match(key):
                raise ProtocolError("observation element keys must be decimal strings")
            if not isinstance(element, ElementRef):
                raise ProtocolError("observation element values must be ElementRef")
            if element.page_id != self.page_id:
                raise ProtocolError(
                    f"element {key} is bound to page {element.page_id!r}, not {self.page_id!r}"
                )
            if element.observation_id != self.observation_id:
                raise ProtocolError(
                    f"element {key} is bound to observation {element.observation_id!r}, "
                    f"not {self.observation_id!r}"
                )
            if element.number != key:
                raise ProtocolError(f"element map key {key!r} does not match element number")
            frozen_elements[key] = element
        object.__setattr__(self, "elements", MappingProxyType(frozen_elements))
        if self.previous_action_id is not None:
            _as_text(self.previous_action_id, "observation.previous_action_id")
        if self.feedback is not None:
            if not isinstance(self.feedback, Feedback):
                raise ProtocolError("observation.feedback must be Feedback or null")
            if self.previous_action_id is None:
                raise ProtocolError("feedback requires previous_action_id")
            if self.feedback.action_id != self.previous_action_id:
                raise ProtocolError("feedback.action_id must match previous_action_id")
            if (
                self.feedback.observation_id is not None
                and self.feedback.observation_id != self.observation_id
            ):
                raise ProtocolError("feedback.observation_id must match this observation")
        elif self.previous_action_id is not None:
            raise ProtocolError("previous_action_id requires feedback")

    def require_element(self, number: str) -> ElementRef:
        if not isinstance(number, str) or number not in self.elements:
            raise ProtocolError(f"element {number!r} is not in this observation")
        return self.elements[number]

    def to_local_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "page_id": self.page_id,
            "observation_id": self.observation_id,
            "url": self.url,
            "ax_text": self.ax_text,
            "elements": {key: self.elements[key].to_local_dict() for key in sorted(self.elements, key=int)},
            "previous_action_id": self.previous_action_id,
            "feedback": None if self.feedback is None else self.feedback.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "capabilities": self.capabilities.to_dict(),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Model-facing view: no localization, no binary artifact bytes."""

        return {
            "url": self.url,
            "observation_id": self.observation_id,
            "ax_text": self.ax_text,
            "elements": [
                self.elements[key].to_prompt_dict()
                for key in sorted(self.elements, key=int)
            ],
            "previous_feedback": None if self.feedback is None else self.feedback.to_dict(),
            "artifacts": {
                kind: {
                    "available": artifact.available,
                    "reason": artifact.reason,
                }
                for kind in ("screenshot", "html", "dom")
                if (artifact := self.artifacts.get(kind)) is not None
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        if not isinstance(data, Mapping):
            raise ProtocolError("observation must be a mapping")
        elements = {}
        raw_elements = data.get("elements") or {}
        if not isinstance(raw_elements, Mapping):
            raise ProtocolError("observation.elements must be a mapping")
        for key, raw in raw_elements.items():
            elements[str(key)] = ElementRef.from_dict(raw)
        return cls(
            run_id=data.get("run_id", ""),
            page_id=data.get("page_id", ""),
            observation_id=data.get("observation_id", ""),
            url=data.get("url", ""),
            ax_text=data.get("ax_text", ""),
            elements=elements,
            previous_action_id=data.get("previous_action_id"),
            feedback=None if data.get("feedback") is None else Feedback.from_dict(data["feedback"]),
            artifacts=ArtifactSet.from_dict(data.get("artifacts")),
            capabilities=Capabilities.from_dict(data.get("capabilities")),
        )


_ACTION_ARG_KEYS: dict[ActionOp, frozenset[str]] = {
    ActionOp.CLICK: frozenset({"target"}),
    ActionOp.TYPE: frozenset({"target", "text", "submit"}),
    ActionOp.PRESS: frozenset({"keys"}),
    ActionOp.SCROLL: frozenset({"direction"}),
    ActionOp.GOTO: frozenset({"url"}),
    ActionOp.GO_BACK: frozenset(),
    ActionOp.GO_FORWARD: frozenset(),
}


@dataclass(frozen=True)
class WebAction:
    action_id: str
    run_id: str
    page_id: str
    observation_id: str
    op: ActionOp
    args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _as_text(self.action_id, "action.action_id")
        validate_run_id(self.run_id)
        _as_text(self.page_id, "action.page_id")
        _as_text(self.observation_id, "action.observation_id")
        op = _coerce_enum(ActionOp, self.op, "action.op")
        object.__setattr__(self, "op", op)
        if not isinstance(self.args, Mapping):
            raise ProtocolError("action.args must be a mapping")
        allowed = _ACTION_ARG_KEYS[op]
        extra = sorted(set(self.args) - allowed)
        if extra:
            raise ProtocolError(f"unsupported {op.value} arguments: {', '.join(extra)}")
        normalized = {str(key): item for key, item in self.args.items()}
        if op is ActionOp.CLICK:
            if "target" not in normalized:
                raise ProtocolError("click requires target")
            normalized["target"] = self._target_text(normalized["target"])
        elif op is ActionOp.TYPE:
            if "target" not in normalized:
                raise ProtocolError("type requires target")
            if "text" not in normalized:
                raise ProtocolError("type requires text")
            normalized["target"] = self._target_text(normalized["target"])
            if not isinstance(normalized["text"], str):
                raise ProtocolError("type.text must be a string")
            submit = normalized.get("submit", False)
            if not isinstance(submit, bool):
                raise ProtocolError("type.submit must be a bool")
            normalized["submit"] = submit
        elif op is ActionOp.PRESS:
            normalized["keys"] = _as_text(normalized.get("keys"), "press.keys")
        elif op is ActionOp.SCROLL:
            normalized["direction"] = _coerce_enum(
                ScrollDirection, normalized.get("direction"), "scroll.direction"
            ).value
        elif op is ActionOp.GOTO:
            url = _as_text(normalized.get("url"), "goto.url")
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                raise ProtocolError("goto.url must be an absolute http(s) URL")
            normalized["url"] = url
        elif normalized:
            raise ProtocolError(f"{op.value} takes no arguments")
        object.__setattr__(self, "args", MappingProxyType(normalized))

    @staticmethod
    def _target_text(value: Any) -> str:
        if not isinstance(value, str):
            raise ProtocolError("target must be the element number as a string")
        if not _ELEMENT_RE.match(value):
            raise ProtocolError("target must be the current observation's element number")
        return value

    @property
    def target(self) -> str | None:
        value = self.args.get("target")
        return value if isinstance(value, str) else None

    @property
    def text(self) -> str | None:
        value = self.args.get("text")
        return value if isinstance(value, str) else None

    @property
    def submit(self) -> bool:
        return bool(self.args.get("submit", False))

    @property
    def keys(self) -> str | None:
        value = self.args.get("keys")
        return value if isinstance(value, str) else None

    @property
    def direction(self) -> str | None:
        value = self.args.get("direction")
        return value if isinstance(value, str) else None

    @property
    def url(self) -> str | None:
        value = self.args.get("url")
        return value if isinstance(value, str) else None

    def to_local_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "run_id": self.run_id,
            "page_id": self.page_id,
            "observation_id": self.observation_id,
            "op": self.op.value,
            "args": _plain(self.args),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WebAction":
        return cls(
            action_id=data.get("action_id", ""),
            run_id=data.get("run_id", ""),
            page_id=data.get("page_id", ""),
            observation_id=data.get("observation_id", ""),
            op=data.get("op", ""),
            args=data.get("args") or {},
        )


@dataclass(frozen=True)
class ExecutionResult:
    """The environment's response to one dispatched action.

    Binding is checked here so a backend cannot return an observation that is
    not the direct successor of the action it claims to answer.
    """

    action_id: str
    feedback: Feedback
    observation: Observation

    def __post_init__(self) -> None:
        _as_text(self.action_id, "execution.action_id")
        if not isinstance(self.feedback, Feedback):
            raise ProtocolError("execution.feedback must be Feedback")
        if not isinstance(self.observation, Observation):
            raise ProtocolError("execution.observation must be Observation")
        if self.feedback.action_id != self.action_id:
            raise ProtocolError("execution feedback does not answer this action")
        if self.observation.previous_action_id != self.action_id:
            raise ProtocolError("execution observation is not bound to this action")
        if self.feedback.observation_id is not None and (
            self.feedback.observation_id != self.observation.observation_id
        ):
            raise ProtocolError("execution feedback references a different observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "feedback": self.feedback.to_dict(),
            "observation": self.observation.to_local_dict(),
        }


@dataclass(frozen=True)
class Finish:
    """The Agent's ending declaration; official score is never invented here."""

    claim: str
    message: str
    reason: TerminationReason
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _as_text(self.claim, "finish.claim")
        _as_text(self.message, "finish.message")
        object.__setattr__(
            self, "reason", _coerce_enum(TerminationReason, self.reason, "finish.reason")
        )
        refs: Iterable[Any] = self.evidence_refs
        if not isinstance(refs, (tuple, list)):
            raise ProtocolError("finish.evidence_refs must be a sequence")
        normalized = tuple(_as_text(ref, "finish.evidence_ref") for ref in refs)
        object.__setattr__(self, "evidence_refs", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "message": self.message,
            "reason": self.reason.value,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ToolRequest:
    """A tool call emitted by the model, before observation binding."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        _as_text(self.tool, "tool_request.tool")
        if not isinstance(self.arguments, Mapping):
            raise ProtocolError("tool_request.arguments must be a mapping")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType({str(key): value for key, value in self.arguments.items()}),
        )
        if self.request_id is not None:
            _as_text(self.request_id, "tool_request.request_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": _plain(self.arguments),
            "request_id": self.request_id,
        }
