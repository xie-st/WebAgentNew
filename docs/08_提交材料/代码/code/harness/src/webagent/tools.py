"""The single web-tool schema and action-binding rules.

Both the official and real backends must consume this module rather than each
defining their own tool names or argument shapes. The model sees the canonical
schema from :func:`render_catalog`; an emitted request is bound to the *current*
observation here, and only then becomes a :class:`WebAction`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .errors import ActionRejected, ProtocolError, RejectReason
from .protocol import (
    ALL_ACTION_OPS,
    ActionOp,
    ElementRef,
    Observation,
    ScrollDirection,
    ToolRequest,
    WebAction,
    new_action_id,
)
from .protocol import _plain  # local JSON conversion only

WEB_TOOL_NAMES = frozenset(
    {
        "web_click",
        "web_type",
        "web_press",
        "web_scroll",
        "web_goto",
        "web_back",
        "web_forward",
    }
)
INTERNAL_TOOL_NAMES = frozenset({"web_observe", "task_note", "evidence_save"})


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    op: ActionOp | None = None
    internal: bool = False

    @property
    def is_web_action(self) -> bool:
        return self.op is not None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": json.loads(json.dumps(self.parameters, ensure_ascii=False)),
        }


def _params(properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_BOOL = {"type": "boolean"}

TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="web_click",
        description=(
            "Click one element in the current observation. Use the element number "
            "shown in this observation only."
        ),
        parameters=_params(
            {
                "target": {
                    **_STRING,
                    "description": "Element number from the current observation.",
                }
            },
            ("target",),
        ),
        op=ActionOp.CLICK,
    ),
    ToolSpec(
        name="web_type",
        description=(
            "Type text into an element from the current observation. Multi-line "
            "text may be given directly, including line breaks; the execution "
            "layer splits it into typing and Enter sub-actions, so do not split "
            "lines yourself. The platform may append to existing content; to "
            "replace, click/select and clear explicitly. There is no hidden clear "
            "flag."
        ),
        parameters=_params(
            {
                "target": {
                    **_STRING,
                    "description": "Element number from the current observation.",
                },
                "text": {
                    **_STRING,
                    "description": (
                        "Text to enter; line breaks are allowed and split by the "
                        "execution layer."
                    ),
                },
                "submit": {
                    **_BOOL,
                    "description": "Press Enter after typing when true; default false.",
                    "default": False,
                },
            },
            ("target", "text"),
        ),
        op=ActionOp.TYPE,
    ),
    ToolSpec(
        name="web_press",
        description="Press a key or key combination on the current page, e.g. 'Ctrl+A'.",
        parameters=_params(
            {"keys": {**_STRING, "description": "Key combination to press."}},
            ("keys",),
        ),
        op=ActionOp.PRESS,
    ),
    ToolSpec(
        name="web_scroll",
        description="Scroll the current page up or down by one viewport.",
        parameters=_params(
            {
                "direction": {
                    "type": "string",
                    "enum": [ScrollDirection.UP.value, ScrollDirection.DOWN.value],
                    "description": "Scroll direction.",
                }
            },
            ("direction",),
        ),
        op=ActionOp.SCROLL,
    ),
    ToolSpec(
        name="web_goto",
        description=(
            "Navigate to an absolute URL that is same-origin with the current "
            "observation. Server-internal URLs cannot be reached from this machine."
        ),
        parameters=_params(
            {"url": {**_STRING, "description": "Absolute same-origin http(s) URL."}},
            ("url",),
        ),
        op=ActionOp.GOTO,
    ),
    ToolSpec(
        name="web_back",
        description="Go back in browser history; a new observation is returned.",
        parameters=_params({}, ()),
        op=ActionOp.GO_BACK,
    ),
    ToolSpec(
        name="web_forward",
        description="Go forward in browser history; a new observation is returned.",
        parameters=_params({}, ()),
        op=ActionOp.GO_FORWARD,
    ),
    ToolSpec(
        name="web_observe",
        description=(
            "Return the latest real observation already recorded for this task. "
            "It does not re-observe the page or execute anything."
        ),
        parameters=_params({}, ()),
        internal=True,
    ),
    ToolSpec(
        name="task_note",
        description="Save a task-local fact. This is an internal tool, not a page action.",
        parameters=_params(
            {"note": {**_STRING, "description": "Fact to keep for this task."}},
            ("note",),
        ),
        internal=True,
    ),
    ToolSpec(
        name="evidence_save",
        description=(
            "Save an artifact already captured for the current observation. "
            "This is an internal tool; it does not re-capture the page."
        ),
        parameters=_params(
            {
                "kind": {
                    "type": "string",
                    "enum": ["screenshot", "html", "dom"],
                    "description": "Artifact kind captured with this observation.",
                },
                "label": {**_STRING, "description": "Optional local label."},
            },
            ("kind",),
        ),
        internal=True,
    ),
)

_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def tool_specs() -> tuple[ToolSpec, ...]:
    return TOOL_SPECS


def get_tool_spec(name: str) -> ToolSpec:
    try:
        return _SPECS_BY_NAME[name]
    except KeyError as exc:
        raise ActionRejected(
            RejectReason.UNKNOWN_TOOL,
            details=[f"unknown tool {name!r}"],
            correction=(
                f"Tool {name!r} is not defined. Use one of: "
                + ", ".join(spec.name for spec in TOOL_SPECS)
            ),
        ) from exc


def is_web_tool(name: str) -> bool:
    return name in WEB_TOOL_NAMES


def tool_schema_catalog() -> list[dict[str, Any]]:
    return [spec.schema() for spec in TOOL_SPECS]


def render_catalog() -> str:
    lines = ["Available tools (one action per step):"]
    for spec in TOOL_SPECS:
        kind = "web action" if spec.is_web_action else "internal"
        lines.append(f"- {spec.name} ({kind}): {spec.description}")
        params = spec.parameters.get("properties", {})
        if params:
            rendered = ", ".join(
                f"{name}"
                + ("*" if name in spec.parameters.get("required", []) else "")
                for name in params
            )
            lines.append(f"  arguments: {rendered}")
    return "\n".join(lines)


def _validate_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    properties = spec.parameters.get("properties", {})
    required = tuple(spec.parameters.get("required", ()))
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ProtocolError(f"{spec.name} does not accept: {', '.join(unknown)}")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ProtocolError(f"{spec.name} requires: {', '.join(missing)}")
    normalized: dict[str, Any] = {}
    for name, value in arguments.items():
        schema = properties[name]
        expected = schema.get("type")
        if expected == "string" and not isinstance(value, str):
            raise ProtocolError(f"{spec.name}.{name} must be a string")
        if expected == "boolean" and not isinstance(value, bool):
            raise ProtocolError(f"{spec.name}.{name} must be a boolean")
        if "enum" in schema and value not in schema["enum"]:
            raise ProtocolError(
                f"{spec.name}.{name} must be one of: {', '.join(map(str, schema['enum']))}"
            )
        normalized[name] = value
    if spec.name == "web_type":
        normalized.setdefault("submit", False)
    return normalized


def _origin(url: str) -> tuple[str, str, int | None] | None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.scheme, parts.hostname.lower(), port


def same_origin(left: str, right: str) -> bool:
    left_origin = _origin(left)
    return left_origin is not None and left_origin == _origin(right)


def _valid_numbers(observation: Observation, limit: int = 30) -> str:
    numbers = sorted(observation.elements, key=int)
    return ", ".join(numbers[:limit]) if numbers else "(none)"


def _correction(reason: RejectReason, observation: Observation | None, detail: str) -> str:
    if observation is None:
        return (
            f"Action refused ({reason.value}): {detail}. "
            "Wait for a real observation before proposing an action."
        )
    return (
        f"Action refused ({reason.value}): {detail}. "
        f"Current observation: page_id={observation.page_id}, "
        f"observation_id={observation.observation_id}, url={observation.url}. "
        f"Valid element numbers in this observation: {_valid_numbers(observation)}. "
        "Return exactly one web action bound to this observation; do not reuse "
        "numbers or actions from an earlier observation."
    )


def resolve_target(action: WebAction, observation: Observation) -> ElementRef | None:
    """Check full observation binding and return the action's element.

    Validity depends on the complete (page, observation, number) reference, not
    on the number having appeared in some earlier observation.
    """

    if action.run_id != observation.run_id:
        raise ActionRejected(
            RejectReason.WRONG_RUN,
            details=[f"action run {action.run_id!r} != observation run {observation.run_id!r}"],
            correction=_correction(
                RejectReason.WRONG_RUN, observation, "the action belongs to another task"
            ),
        )
    if action.page_id != observation.page_id:
        raise ActionRejected(
            RejectReason.WRONG_PAGE,
            details=[f"action page {action.page_id!r} != observation page {observation.page_id!r}"],
            correction=_correction(
                RejectReason.WRONG_PAGE, observation, "the action belongs to another page"
            ),
        )
    if action.observation_id != observation.observation_id:
        raise ActionRejected(
            RejectReason.STALE_OBSERVATION,
            details=[
                f"action observation {action.observation_id!r} != current "
                f"{observation.observation_id!r}"
            ],
            correction=_correction(
                RejectReason.STALE_OBSERVATION,
                observation,
                "the action was built from an older observation",
            ),
        )
    if not observation.capabilities.allows(action.op):
        raise ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[f"operation {action.op.value} is not allowed by current capabilities"],
            correction=_correction(
                RejectReason.INVALID_ARGUMENTS,
                observation,
                f"operation {action.op.value} is unavailable",
            ),
        )
    if action.op is ActionOp.GOTO:
        if not same_origin(action.url or "", observation.url):
            raise ActionRejected(
                RejectReason.EXTERNAL_ORIGIN,
                details=["goto target is not same-origin with the current observation"],
                correction=_correction(
                    RejectReason.EXTERNAL_ORIGIN,
                    observation,
                    "goto is limited to same-origin URLs seen in this task",
                ),
            )
        return None
    if action.op in (ActionOp.GO_BACK, ActionOp.GO_FORWARD):
        return None
    if action.op not in (ActionOp.CLICK, ActionOp.TYPE):
        return None
    target = action.target
    if target is None:
        raise ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[f"{action.op.value} has no element target"],
            correction=_correction(
                RejectReason.INVALID_ARGUMENTS, observation, "this action needs an element target"
            ),
        )
    element = observation.elements.get(target)
    if element is None:
        raise ActionRejected(
            RejectReason.UNKNOWN_ELEMENT,
            details=[f"element {target!r} does not exist in the current observation"],
            correction=_correction(
                RejectReason.UNKNOWN_ELEMENT,
                observation,
                f"element {target!r} is not part of the current observation",
            ),
        )
    return element


def build_web_action(
    request: ToolRequest,
    observation: Observation,
    *,
    action_id: str | None = None,
) -> WebAction:
    """Turn one model tool request into an observation-bound WebAction."""

    try:
        spec = get_tool_spec(request.tool)
    except ActionRejected as exc:
        exc.correction = _correction(exc.reason, observation, "; ".join(exc.details))
        raise
    if not spec.is_web_action:
        raise ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[f"{request.tool} is an internal tool, not a page action"],
            correction=_correction(
                RejectReason.INVALID_ARGUMENTS,
                observation,
                f"{request.tool} cannot be dispatched as a page action",
            ),
        )
    try:
        arguments = _validate_arguments(spec, request.arguments)
    except ProtocolError as exc:
        raise ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[str(exc)],
            correction=_correction(RejectReason.INVALID_ARGUMENTS, observation, str(exc)),
        ) from exc
    if not observation.capabilities.allows(spec.op):
        raise ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[f"operation {spec.op.value} is not allowed by current capabilities"],
            correction=_correction(
                RejectReason.INVALID_ARGUMENTS,
                observation,
                f"operation {spec.op.value} is unavailable",
            ),
        )
    if spec.op is ActionOp.GOTO:
        if not same_origin(arguments["url"], observation.url):
            raise ActionRejected(
                RejectReason.EXTERNAL_ORIGIN,
                details=[f"goto {arguments['url']!r} is not same-origin with {observation.url!r}"],
                correction=_correction(
                    RejectReason.EXTERNAL_ORIGIN,
                    observation,
                    "goto is limited to same-origin URLs seen in this task",
                ),
            )
    elif spec.op in (ActionOp.CLICK, ActionOp.TYPE):
        target = arguments.get("target")
        if target not in observation.elements:
            raise ActionRejected(
                RejectReason.UNKNOWN_ELEMENT,
                details=[f"element {target!r} does not exist in the current observation"],
                correction=_correction(
                    RejectReason.UNKNOWN_ELEMENT,
                    observation,
                    f"element {target!r} is not part of the current observation",
                ),
            )
    try:
        return WebAction(
            action_id=action_id or new_action_id(),
            run_id=observation.run_id,
            page_id=observation.page_id,
            observation_id=observation.observation_id,
            op=spec.op,
            args=arguments,
        )
    except ProtocolError as exc:
        raise ActionRejected(
            RejectReason.INVALID_ARGUMENTS,
            details=[str(exc)],
            correction=_correction(RejectReason.INVALID_ARGUMENTS, observation, str(exc)),
        ) from exc


@dataclass(frozen=True)
class GuardResult:
    """One validated web action (if any) plus internal tool calls."""

    action: WebAction | None
    internal: tuple[ToolRequest, ...] = ()

    @property
    def requires_environment(self) -> bool:
        return self.action is not None


def guard_requests(
    requests: Sequence[ToolRequest],
    observation: Observation,
) -> GuardResult:
    """Reject multiple web actions and bind the single one to this observation.

    Internal tools are validated and returned for direct execution. A model
    output containing more than one web action is refused as a whole; none of
    those actions is queued. Internal calls in the same output are not page
    actions and do not make the output valid if it also has two web actions.
    """

    if observation is None:
        raise ActionRejected(
            RejectReason.NO_OBSERVATION,
            details=["no current observation"],
            correction=_correction(RejectReason.NO_OBSERVATION, None, "no observation"),
        )
    if not requests:
        return GuardResult(action=None, internal=())
    web_requests: list[ToolRequest] = []
    internal_requests: list[ToolRequest] = []
    for request in requests:
        spec = get_tool_spec(request.tool)
        if spec.is_web_action:
            web_requests.append(request)
        else:
            internal_requests.append(request)
    if len(web_requests) > 1:
        names = ", ".join(request.tool for request in web_requests)
        raise ActionRejected(
            RejectReason.MULTIPLE_ACTIONS,
            details=[f"{len(web_requests)} web actions in one model output: {names}"],
            correction=(
                _correction(
                    RejectReason.MULTIPLE_ACTIONS,
                    observation,
                    f"one model output contained {len(web_requests)} web actions",
                )
                + " Emit exactly one web action per step and wait for its observation; "
                "no action from this batch was queued."
            ),
        )
    for request in internal_requests:
        try:
            spec = get_tool_spec(request.tool)
            _validate_arguments(spec, request.arguments)
        except (ActionRejected, ProtocolError) as exc:
            if isinstance(exc, ActionRejected):
                exc.correction = _correction(exc.reason, observation, "; ".join(exc.details))
                raise
            raise ActionRejected(
                RejectReason.INVALID_ARGUMENTS,
                details=[str(exc)],
                correction=_correction(RejectReason.INVALID_ARGUMENTS, observation, str(exc)),
            ) from exc
    action = (
        build_web_action(web_requests[0], observation) if web_requests else None
    )
    return GuardResult(action=action, internal=tuple(internal_requests))


def describe_action(action: WebAction) -> str:
    """Short local rendering for logs/diagnostics; not the official wire form."""

    args = _plain(action.args)
    if args:
        return f"{action.op.value} {json.dumps(args, ensure_ascii=False, sort_keys=True)}"
    return action.op.value


__all__ = [
    "ALL_ACTION_OPS",
    "GuardResult",
    "INTERNAL_TOOL_NAMES",
    "TOOL_SPECS",
    "WEB_TOOL_NAMES",
    "build_web_action",
    "describe_action",
    "get_tool_spec",
    "guard_requests",
    "is_web_tool",
    "render_catalog",
    "resolve_target",
    "same_origin",
    "tool_schema_catalog",
    "tool_specs",
]
