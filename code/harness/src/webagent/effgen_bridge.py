"""Assemble EffGen and drive its interactive loop from one harness session.

The vendored EffGen entry (:meth:`effgen.core.agent.Agent.run_interactive`) is a
generator: it yields one external tool request and resumes when the environment's
result is sent back. This module is the harness side of that contract. It owns
the EffGen agent's tools, converts each request into the A1 tool schema and
binding rules, and walks the session through

``guard_requests -> propose -> consume -> backend.execute -> ingest_observation``

without rebuilding the agent between steps.

It does not call a real model, start a browser or start an event loop. The model
is whichever EffGen ``BaseModel`` the caller passes; the environment is the A1
``Backend``. Internal tools (``web_observe``, ``task_note``, ``evidence_save``)
run through EffGen's ordinary tool path and are backed by the live session.

One driver belongs to one task. The EffGen agent is built once for that task and
kept for its whole life; :meth:`EffGenLoopDriver.start` only opens the generator.

Phase-1 configuration gate: SINGLE agent, ``tool_calling_mode="react"``, no
capability probe, sub-agents, fallback, speculative execution or framework
cross-task memory. EffGen's shipped 10-iteration default is *not* used here: the
interactive entry is bounded by the session deadline (``max_iterations=None``),
and a caller that wants a hard cap says so in :class:`LoopSettings`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from effgen.core.agent import Agent
from effgen.core.agent_config import AgentConfig as EffGenAgentConfig, AgentMode
from effgen.core.agent_interactive import (
    InteractiveToolRequest,
)
from effgen.core.agent_response import AgentResponse
from effgen.prompts.tool_contract import ToolUsePolicy
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)

from .backend import Backend
from .budget import render_observation
from .config import AppConfig
from .errors import (
    ActionRejected,
    DeadlineExceeded,
    EvidenceError,
    ProtocolError,
    SessionStateError,
)
from .protocol import (
    Finish,
    FeedbackStatus,
    Observation,
    TerminationReason,
    ToolRequest,
    WebAction,
)
from .session import Session, SessionState
from .tools import TOOL_SPECS, ToolSpec, WEB_TOOL_NAMES, guard_requests


class BridgeError(RuntimeError):
    """The driver was used outside the state its contract allows."""


@dataclass(frozen=True)
class LoopSettings:
    """Limits and hooks for one interactive task.

    ``max_iterations=None`` is the phase-1 default: the session deadline ends the
    run, not a step count. ``max_turns_without_progress=None`` keeps the
    framework from asking for an answer because turns repeated; on a page walk a
    repeated-looking turn is still a new observation.
    """

    external_tools: frozenset[str] = WEB_TOOL_NAMES
    max_format_corrections: int = 2
    max_iterations: int | None = None
    max_turns_without_progress: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.external_tools, frozenset):
            object.__setattr__(self, "external_tools", frozenset(self.external_tools))
        if not self.external_tools:
            raise BridgeError("at least one external tool is required")
        if self.max_format_corrections < 0:
            raise BridgeError("max_format_corrections must be >= 0")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise BridgeError("max_iterations must be a positive integer or None")


class DriverStatus(str, Enum):
    """What one advance of the loop produced."""

    ACTION = "action"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    DEADLINE = "deadline"
    FORMAT_ERROR = "format_error"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DriverStep:
    """One observable state of the loop after an advance.

    ``ACTION`` means the environment owes exactly one execution: the call is
    yielded by EffGen and the harness has proposed it. It has not been consumed
    or executed yet; call :meth:`EffGenLoopDriver.take_action`.
    """

    status: DriverStatus
    request: InteractiveToolRequest | None = None
    action: WebAction | None = None
    response: AgentResponse | None = None
    message: str = ""
    corrections: int = 0

    @property
    def needs_action(self) -> bool:
        return self.status is DriverStatus.ACTION

    @property
    def terminal(self) -> bool:
        return self.status not in (DriverStatus.ACTION,)


class HarnessToolRuntime:
    """The live session the internal EffGen tools read and write.

    Bound to a driver's session before the loop starts. It never re-observes and
    never acts on the page: ``web_observe`` returns the latest real observation
    already recorded, ``task_note`` stores a task-local fact, and
    ``evidence_save`` copies an artifact this observation already captured.
    """

    def __init__(
        self,
        session: Session | None = None,
        *,
        artifact_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.session = session
        self.artifact_loader = artifact_loader

    def bind(self, session: Session) -> None:
        self.session = session

    def set_artifact_loader(self, loader: Callable[[str], Any] | None) -> None:
        self.artifact_loader = loader

    def _session(self) -> Session:
        if self.session is None:
            raise BridgeError("harness tool runtime is not bound to a session")
        return self.session

    def observe(self) -> str:
        session = self._session()
        observation = session.observation
        if observation is None:
            return (
                "No observation has been recorded for this task yet. "
                "A web action returns one; web_observe does not re-capture the page."
            )
        return render_observation(observation)

    def note(self, note: str = "") -> str:
        session = self._session()
        session.add_task_note(note)
        return f"Saved task note ({len(session.task_notes)} total): {note}"

    def save_evidence(self, kind: str, label: str | None = None) -> str:
        session = self._session()
        observation = session.observation
        if observation is None:
            raise EvidenceError("no observation to save evidence for")
        artifact = observation.artifacts.get(kind)
        if (
            artifact is not None
            and not artifact.available
            and artifact.reason == "not_downloaded"
            and self.artifact_loader is not None
        ):
            # Official HTTP artifacts are downloaded on demand, still inside the
            # SDK callback that supplied this observation. The loader verifies
            # size/hash and stores the bytes before this tool copies them.
            artifact = self.artifact_loader(kind)
        if artifact is None or not artifact.available:
            reason = "not captured with this observation" if artifact is None else artifact.reason
            raise EvidenceError(f"{kind} is unavailable for this observation: {reason}")
        session.evidence.assert_valid(artifact)
        data = session.evidence.resolve_reference(artifact.reference or "").read_bytes()
        saved = session.evidence.save_bytes(
            kind,
            data,
            observation_id=observation.observation_id,
            metadata={"label": label or "", "source_reference": artifact.reference},
        )
        session.evidence.assert_valid(saved)
        if artifact.sha256 is not None and saved.sha256 != artifact.sha256:
            raise EvidenceError("saved evidence bytes do not match the observation artifact")
        return (
            f"Saved {kind} evidence as {saved.reference} "
            f"(sha256={saved.sha256}, bytes={saved.byte_size})"
        )


_CATEGORY_BY_TOOL: dict[str, ToolCategory] = {
    "web_click": ToolCategory.EXTERNAL_API,
    "web_type": ToolCategory.EXTERNAL_API,
    "web_press": ToolCategory.EXTERNAL_API,
    "web_scroll": ToolCategory.EXTERNAL_API,
    "web_goto": ToolCategory.EXTERNAL_API,
    "web_back": ToolCategory.EXTERNAL_API,
    "web_forward": ToolCategory.EXTERNAL_API,
    "web_observe": ToolCategory.INFORMATION_RETRIEVAL,
    "task_note": ToolCategory.DATA_PROCESSING,
    "evidence_save": ToolCategory.FILE_OPERATIONS,
}

_PARAMETER_TYPE = {
    "string": ParameterType.STRING,
    "boolean": ParameterType.BOOLEAN,
    "integer": ParameterType.INTEGER,
    "number": ParameterType.FLOAT,
    "object": ParameterType.OBJECT,
    "array": ParameterType.ARRAY,
}


def _parameter_specs(spec: ToolSpec) -> list[ParameterSpec]:
    """EffGen declarations for the one canonical harness schema."""

    properties = spec.parameters.get("properties", {}) or {}
    required = set(spec.parameters.get("required", ()) or ())
    parameters: list[ParameterSpec] = []
    for name, raw in properties.items():
        parameter_type = _PARAMETER_TYPE.get(str(raw.get("type", "string")), ParameterType.STRING)
        parameters.append(
            ParameterSpec(
                name=name,
                type=parameter_type,
                description=str(raw.get("description", "")),
                required=name in required,
                default=raw.get("default"),
                enum=list(raw["enum"]) if raw.get("enum") else None,
            )
        )
    return parameters


class _HarnessTool(BaseTool):
    """An EffGen tool whose execution is a plain call, never a fresh event loop.

    ``BaseTool.execute`` is a coroutine; driving it would make EffGen start an
    event loop (or a worker thread under one) for every internal page tool. These
    tools answer synchronously instead, and keep the framework's own parameter
    validation, so a call reaches the loop with the same envelope a built-in tool
    produces. The abstract ``_execute`` is required by the base class and is
    never reached through :meth:`execute`.
    """

    async def _execute(self, **kwargs: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError("harness tools execute synchronously")

    def execute(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        ok, error = self.validate_parameters(**kwargs)
        if not ok:
            return ToolResult(
                success=False,
                output=None,
                error=f"Parameter validation failed: {error}",
            )
        try:
            return ToolResult(success=True, output=self._run(**kwargs))
        except Exception as exc:  # noqa: BLE001 - the loop reads failures as text
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution failed: {type(exc).__name__}: {exc}",
                metadata={"error_type": type(exc).__name__},
            )

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


class _ExternalWebTool(_HarnessTool):
    """One page action, declared to EffGen and executed by the caller's environment.

    It must never run inside the process: the interactive loop yields before
    dispatch. An ordinary ``Agent.run`` that reaches it is refused rather than
    answered with a fabricated result.
    """

    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(
            ToolMetadata(
                name=spec.name,
                description=spec.description,
                category=_CATEGORY_BY_TOOL[spec.name],
                parameters=_parameter_specs(spec),
            )
        )

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover - never dispatched
        raise BridgeError("a page action reached in-process execution")

    def execute(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        return ToolResult(
            success=False,
            output=None,
            error=(
                f"Tool '{self.name}' executes in the environment, not in this "
                "process; use the interactive loop entry so the action is "
                "dispatched and its observation returned."
            ),
            metadata={"error_type": "ExternalToolNotExecutable"},
        )


class _InternalTool(_HarnessTool):
    """``web_observe`` / ``task_note`` / ``evidence_save``, backed by the session."""

    def __init__(self, spec: ToolSpec, runner: Callable[..., str]) -> None:
        super().__init__(
            ToolMetadata(
                name=spec.name,
                description=spec.description,
                category=_CATEGORY_BY_TOOL[spec.name],
                parameters=_parameter_specs(spec),
            )
        )
        self._runner = runner

    def _run(self, **kwargs: Any) -> Any:
        return self._runner(**kwargs)


SYSTEM_PROMPT = (
    "You operate one web page to complete the task. Each turn you may call one "
    "tool. A web action (web_click, web_type, web_press, web_scroll, web_goto, "
    "web_back, web_forward) is executed by the environment and returns a new "
    "observation; use element numbers only from the observation you were just "
    "given. web_observe returns the latest observation already recorded, "
    "task_note stores a fact for this task, and evidence_save keeps an artifact "
    "this observation already captured. Never invent an element number, an "
    "observation, or a result. When the task is done, state it as the final "
    "answer."
)


def build_effgen_config(
    model: Any,
    tools: Sequence[BaseTool],
    *,
    app_config: AppConfig | None = None,
) -> EffGenAgentConfig:
    """The phase-1 EffGen configuration, with every disabled feature explicit."""

    app = app_config or AppConfig(profile="fake", backend="fake")
    agent_settings = app.agent
    if agent_settings.strategy != "single":
        raise BridgeError("phase 1 runs a single agent")
    if agent_settings.tool_calling_mode != "react":
        raise BridgeError("phase 1 runs the ReAct text tool mode")
    for disabled in (
        "capability_probe",
        "sub_agents",
        "fallback",
        "speculative_execution",
        "cross_task_memory",
    ):
        if getattr(agent_settings, disabled):
            raise BridgeError(f"phase 1 requires agent.{disabled} to be false")
    return EffGenAgentConfig(
        model=model,
        tools=list(tools),
        system_prompt=SYSTEM_PROMPT,
        mode=AgentMode.SINGLE,
        tool_calling_mode="react",
        capability_probe=False,
        enable_sub_agents=False,
        enable_fallback=False,
        enable_memory=bool(agent_settings.cross_task_memory),
        speculative_execution=False,
        models=None,
        tool_use=ToolUsePolicy.AUTO,
        max_turns_without_progress=None,
        enable_streaming=False,
    )


def build_harness_tools(runtime: HarnessToolRuntime) -> list[BaseTool]:
    """The EffGen tools for the canonical harness schema, in schema order."""

    tools: list[BaseTool] = []
    for spec in TOOL_SPECS:
        if spec.is_web_action:
            tools.append(_ExternalWebTool(spec))
        elif spec.name == "web_observe":
            tools.append(_InternalTool(spec, runtime.observe))
        elif spec.name == "task_note":
            tools.append(_InternalTool(spec, runtime.note))
        elif spec.name == "evidence_save":
            tools.append(_InternalTool(spec, runtime.save_evidence))
        else:  # pragma: no cover - the schema is closed
            raise BridgeError(f"no harness implementation for tool {spec.name!r}")
    return tools


def build_effgen_agent(
    model: Any,
    *,
    runtime: HarnessToolRuntime | None = None,
    app_config: AppConfig | None = None,
    extra_tools: Iterable[BaseTool] = (),
) -> Agent:
    """One EffGen agent holding the canonical tools and the phase-1 config."""

    runtime = runtime or HarnessToolRuntime()
    tools = build_harness_tools(runtime)
    tools.extend(extra_tools)
    config = build_effgen_config(model, tools, app_config=app_config)
    return Agent(config)


def _render_action_result(tool: str, observation: Observation) -> str:
    """The text the model reads as the result of the call it just made."""

    feedback = observation.feedback
    if feedback is None:  # pragma: no cover - Session guarantees the binding
        raise ProtocolError("an executed action's observation must carry feedback")
    if feedback.status is FeedbackStatus.OBSERVED:
        head = f"Result of {tool}: {feedback.message or 'observation returned'}"
    else:
        head = (
            f"Error executing tool '{tool}': {feedback.message or feedback.status.value}"
            f" ({feedback.error_type or feedback.status.value})"
        )
    return f"{head}\n{render_observation(observation)}"


class EffGenLoopDriver:
    """Drive one task's interactive EffGen generator against one session.

    States: ``new`` -> ``running`` (either awaiting an action or advancing) ->
    ``terminal``. ``start`` is called once, ``take_action`` exactly once per
    ``ACTION`` step, and ``accept_observation`` exactly once after the environment
    executed that action. ``close`` and ``cancel`` are safe to repeat.
    """

    def __init__(
        self,
        session: Session,
        *,
        agent: Agent,
        runtime: HarnessToolRuntime,
        model_name: str | None = None,
        settings: LoopSettings | None = None,
    ) -> None:
        if not isinstance(session, Session):
            raise BridgeError("EffGenLoopDriver needs a webagent Session")
        if not isinstance(agent, Agent):
            raise BridgeError("EffGenLoopDriver needs an EffGen Agent")
        if runtime.session is not session:
            raise BridgeError("the agent's tool runtime is bound to another session")
        self.session = session
        self.agent = agent
        self.runtime = runtime
        self.settings = settings or LoopSettings()
        self.model_name = model_name or getattr(agent, "model_name", None)
        self._generator: Any = None
        self._pending_request: InteractiveToolRequest | None = None
        self._pending_action: WebAction | None = None
        self._awaiting_environment: WebAction | None = None
        self._corrections: list[dict[str, Any]] = []
        self._model_turn = 0
        self._terminal: DriverStep | None = None
        self._closed = False
        self._started = False

    # ------------------------------------------------------------------ state
    @property
    def started(self) -> bool:
        return self._started

    @property
    def pending_request(self) -> InteractiveToolRequest | None:
        return self._pending_request

    @property
    def pending_action(self) -> WebAction | None:
        return self._pending_action

    @property
    def awaiting_environment(self) -> WebAction | None:
        return self._awaiting_environment

    @property
    def terminal(self) -> DriverStep | None:
        return self._terminal

    @property
    def corrections(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._corrections)

    # ------------------------------------------------------------------- drive
    def start(self) -> DriverStep:
        """Open the generator and advance to its first request or answer."""

        if self._closed:
            raise BridgeError("driver is closed")
        if self._started:
            raise BridgeError("driver is already started; one task per driver")
        session = self.session
        if session.state is not SessionState.RUNNING:
            raise BridgeError(
                "start needs a running session: call reset/start_task and ingest "
                "the first observation before starting the loop"
            )
        if session.observation is None:
            raise BridgeError("start needs a current observation")
        self._started = True
        task_prompt = session.task or ""
        if session.observation is not None:
            # The model must see the initial page before it can choose a valid
            # element number. EffGen's task string is the only pre-loop content,
            # so attach the canonical observation view exactly once.
            task_prompt = task_prompt + "\n\n" + render_observation(session.observation)
        self._generator = self.agent.run_interactive(
            task_prompt,
            external_tools=self.settings.external_tools,
            before_step=self._before_step,
            before_model_call=self._before_model_call,
            batch_correction=self._batch_correction,
            max_iterations=self.settings.max_iterations,
            max_turns_without_progress=self.settings.max_turns_without_progress,
        )
        return self._pump(None, send=False)

    def take_action(self) -> WebAction:
        """Consume the proposed action and hand it to the environment once."""

        if self._pending_action is None or self._pending_request is None:
            raise BridgeError("no page action is waiting for the environment")
        if self._awaiting_environment is not None:
            raise BridgeError("the previous action has not been answered yet")
        self.session.check_deadline("action dispatch")
        action = self.session.consume(self._pending_action.action_id)
        self._awaiting_environment = action
        self._pending_action = None
        return action

    def accept_observation(self, observation: Observation) -> DriverStep:
        """Ingest the environment's observation and resume the same generator."""

        action = self._awaiting_environment
        if action is None:
            raise BridgeError("no action is awaiting the environment's observation")
        if self._closed:
            raise BridgeError("driver is closed")
        # The action was already executed; recording the observation it returned
        # is local bookkeeping, not new model or action work. The deadline is
        # checked before the loop is resumed, so a task that has run out of time
        # still owes no further turn.
        self.session.ingest_observation(observation)
        tool = self._pending_request.tool if self._pending_request is not None else action.op.value
        self._awaiting_environment = None
        self._pending_action = None
        self._pending_request = None
        try:
            self.session.check_deadline("action feedback")
        except DeadlineExceeded as exc:
            return self._finish_deadline(exc)
        text = _render_action_result(tool, observation)
        return self._pump(text, send=True)

    def resume(self, observation: Observation) -> DriverStep:
        """Resume a paused task from a fresh observation.

        The paused request was never executed. Sending it as a normal tool result
        would imply an execution that did not happen, so the generator receives an
        explicit ``not executed`` message carrying the new observation. The old
        element table and action are discarded; the same generator and agent are
        kept, and no old action is replayed.
        """

        if self._closed:
            raise BridgeError("driver is closed")
        if self.session.state is not SessionState.PAUSED:
            raise BridgeError("resume needs a paused session")
        if self._awaiting_environment is not None:
            raise BridgeError("cannot resume while an action is awaiting feedback")
        if self._pending_request is None:
            raise BridgeError("no paused page request to continue")
        request = self._pending_request
        self.session.resume(observation)
        self._pending_action = None
        self._pending_request = None
        tool = request.tool
        text = (
            f"Result of {tool}: NOT EXECUTED. The user paused the task before this "
            "action was sent to the browser, so it produced no page result. The "
            "user may have changed the page. Re-observe the state below and decide "
            "the next step; do not replay the old action blindly.\n"
            + render_observation(observation)
        )
        return self._pump(text, send=True)

    def close(self) -> None:
        """Close the generator without inventing a result for an unanswered call."""

        if self._closed:
            return
        self._closed = True
        generator = self._generator
        self._generator = None
        if generator is not None:
            try:
                generator.close()
            except RuntimeError:  # pragma: no cover - a generator closed mid-run
                pass
        self._pending_action = None
        self._pending_request = None
        if self._terminal is None:
            self._terminal = DriverStep(
                status=DriverStatus.CANCELLED,
                message="loop closed before the run ended",
                corrections=len(self._corrections),
            )

    def cancel(self, reason: str = "cancelled") -> DriverStep:
        """Stop the task: close the generator, then record the cancellation."""

        if self._terminal is not None and self._terminal.status is not DriverStatus.CANCELLED:
            return self._terminal
        if self.session.state is SessionState.RUNNING:
            self.session.cancel(reason)
        self.close()
        step = DriverStep(
            status=DriverStatus.CANCELLED,
            message=str(reason),
            corrections=len(self._corrections),
        )
        self._terminal = step
        return step

    # ------------------------------------------------------------------ hooks
    def _before_step(self, iteration: int) -> None:
        self.session.check_deadline(f"loop turn {iteration}")

    def _before_model_call(self, prompt: Any) -> None:
        session = self.session
        session.check_deadline("model call")
        self._model_turn += 1
        if getattr(self.agent.model, "records_model_calls", False):
            # The adapter emits one post-request event per HTTP attempt (including
            # retries) with real usage/latency. A pre-call placeholder here would
            # double-count the turn and lose that information.
            return
        text = prompt if isinstance(prompt, str) else json.dumps(
            prompt, ensure_ascii=False, default=str
        )
        try:
            tokens: int | None = session.budget.counter.count_text(text)
            source = session.budget.counter.source
        except Exception:  # noqa: BLE001 - a counter never blocks a run
            tokens, source = None, "unknown"
        session.record_model_call(
            request_id=f"ireq-model-{self._model_turn}",
            input_tokens=tokens,
            output_tokens=None,
            token_count_source=source,
            latency_ms=None,
            model_name=self.model_name,
        )

    def _batch_correction(self, calls: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
        """Refuse a batched page action through the same guard the loop uses."""

        requests = [ToolRequest(tool, dict(arguments)) for tool, arguments in calls]
        reason = "batch_page_action"
        correction = (
            "This model output bundled a page action with other calls. A page "
            "action must be sent alone: re-emit exactly one of "
            f"{', '.join(tool for tool, _ in calls if tool in self.settings.external_tools)} "
            "on its own and wait for its observation."
        )
        observation = self.session.observation
        if observation is not None:
            try:
                guard_requests(requests, observation)
            except ActionRejected as exc:
                reason = exc.reason.value
                correction = exc.correction
        self.session.logger.emit(
            "format_correction",
            page_id=None if observation is None else observation.page_id,
            url=None if observation is None else observation.url,
            observation_id=None if observation is None else observation.observation_id,
            result={
                "reason": reason,
                "scope": "batch",
                "tools": [tool for tool, _ in calls],
                "correction": correction,
            },
        )
        return correction

    # ---------------------------------------------------------------- internal
    def _pump(self, value: Any, *, send: bool) -> DriverStep:
        while True:
            try:
                self.session.check_deadline("loop advance")
            except DeadlineExceeded as exc:
                return self._finish_deadline(exc)
            try:
                if send:
                    item = self._generator.send(value)
                    send = False
                else:
                    item = next(self._generator)
            except StopIteration as stop:
                return self._finish_response(stop.value)
            except DeadlineExceeded as exc:
                return self._finish_deadline(exc)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                return self._finish_error(exc)

            request = item
            observation = self.session.observation
            if observation is None:
                return self._finish_error(
                    ProtocolError("the loop asked for an action with no current observation")
                )
            try:
                guarded = guard_requests(
                    [ToolRequest(request.tool, dict(request.arguments))], observation
                )
                if guarded.action is None:
                    raise BridgeError(
                        f"internal tool {request.tool!r} was yielded as external"
                    )
                self.session.propose(guarded.action)
            except ActionRejected as exc:
                step = self._record_rejection(request, exc)
                if step is not None:
                    return step
                value = exc.correction
                send = True
                continue
            except (ProtocolError, SessionStateError, BridgeError) as exc:
                return self._finish_error(exc)
            self._pending_request = request
            self._pending_action = guarded.action
            return DriverStep(
                status=DriverStatus.ACTION,
                request=request,
                action=guarded.action,
                corrections=len(self._corrections),
            )

    def _record_rejection(self, request: InteractiveToolRequest, exc: ActionRejected) -> DriverStep | None:
        """Send a format correction back, or stop once the allowance is spent."""

        observation = self.session.observation
        record = {
            "reason": exc.reason.value,
            "tool": request.tool,
            "request_id": request.request_id,
            "details": list(exc.details),
        }
        self._corrections.append(record)
        self.session.logger.emit(
            "format_correction",
            page_id=None if observation is None else observation.page_id,
            url=None if observation is None else observation.url,
            observation_id=None if observation is None else observation.observation_id,
            result={
                "reason": exc.reason.value,
                "scope": "action",
                "tool": request.tool,
                "request_id": request.request_id,
                "correction": exc.correction,
                "corrections_so_far": len(self._corrections),
            },
        )
        if len(self._corrections) <= self.settings.max_format_corrections:
            return None
        message = (
            "the model's page actions did not become valid after "
            f"{self.settings.max_format_corrections} correction call(s): "
            f"{exc.reason.value}"
        )
        return self._terminate(
            DriverStatus.FORMAT_ERROR,
            message,
            finish_reason=TerminationReason.ERROR,
        )

    def _finish_response(self, response: AgentResponse | None) -> DriverStep:
        if not isinstance(response, AgentResponse):
            return self._finish_error(
                ProtocolError("the interactive loop ended without an AgentResponse")
            )
        success = bool(getattr(response, "success", False))
        output = str(getattr(response, "output", "") or "")
        if success and output.strip():
            reason = TerminationReason.COMPLETED
            claim = output.strip()
            message = output
            status = DriverStatus.COMPLETED
        else:
            reason = TerminationReason.INCOMPLETE
            detail = ""
            metadata = getattr(response, "metadata", None) or {}
            if isinstance(metadata, dict):
                detail = str(metadata.get("reason") or "")
            claim = f"run ended without a completed answer ({detail or 'no answer'})"
            message = detail or "the run ended without an answer"
            status = DriverStatus.INCOMPLETE
        return self._terminate(
            status,
            message,
            finish_reason=reason,
            response=response,
            claim=claim,
        )

    def _finish_deadline(self, exc: DeadlineExceeded) -> DriverStep:
        try:
            self.session.check_deadline("loop advance")
        except DeadlineExceeded:
            pass
        return self._terminate(
            DriverStatus.DEADLINE,
            str(exc),
            finish_reason=TerminationReason.DEADLINE,
            claim="task deadline reached before the loop finished",
        )

    def _finish_error(self, exc: BaseException) -> DriverStep:
        try:
            self.session.log_exception("effgen_loop", exc)
        except Exception:  # noqa: BLE001 - the log must not hide the failure
            pass
        return self._terminate(
            DriverStatus.ERROR,
            f"{type(exc).__name__}: {exc}",
            finish_reason=TerminationReason.ERROR,
            claim="the interactive loop failed",
        )

    def _terminate(
        self,
        status: DriverStatus,
        message: str,
        *,
        finish_reason: TerminationReason,
        response: AgentResponse | None = None,
        claim: str = "",
    ) -> DriverStep:
        generator = self._generator
        self._generator = None
        if generator is not None:
            try:
                generator.close()
            except RuntimeError:  # pragma: no cover
                pass
        session = self.session
        if session.state is SessionState.RUNNING and self._awaiting_environment is None:
            try:
                session.finish(
                    Finish(
                        claim=claim or status.value,
                        message=message,
                        reason=finish_reason,
                    )
                )
            except SessionStateError:
                pass
        self._pending_action = None
        self._pending_request = None
        step = DriverStep(
            status=status,
            response=response,
            message=message,
            corrections=len(self._corrections),
        )
        self._terminal = step
        return step


def create_driver(
    session: Session,
    model: Any,
    *,
    app_config: AppConfig | None = None,
    settings: LoopSettings | None = None,
    extra_tools: Iterable[BaseTool] = (),
    artifact_loader: Callable[[str], Any] | None = None,
) -> EffGenLoopDriver:
    """Build the runtime, the agent and the driver for one task session."""

    runtime = HarnessToolRuntime(session, artifact_loader=artifact_loader)
    agent = build_effgen_agent(
        model, runtime=runtime, app_config=app_config, extra_tools=extra_tools
    )
    return EffGenLoopDriver(session, agent=agent, runtime=runtime, settings=settings)


class BackendLoopRunner:
    """Drive one task to its end against an A1 backend, for offline integration.

    This is the shape the real and official entry points take: advance, execute
    the one consumed action, ingest the returned observation, advance again. It
    is deliberately small and synchronous; a backend that blocks is the caller's
    problem, not a reason for a background thread here.
    """

    def __init__(self, driver: EffGenLoopDriver, backend: Backend) -> None:
        self.driver = driver
        self.backend = backend

    def run(self, *, on_step: Callable[[DriverStep], None] | None = None) -> DriverStep:
        step = self.driver.start()
        while True:
            if on_step is not None:
                on_step(step)
            if step.status is not DriverStatus.ACTION:
                return step
            action = self.driver.take_action()
            result = self.backend.execute(action)
            step = self.driver.accept_observation(result.observation)


__all__ = [
    "BackendLoopRunner",
    "BridgeError",
    "DriverStatus",
    "DriverStep",
    "EffGenLoopDriver",
    "HarnessToolRuntime",
    "LoopSettings",
    "build_effgen_agent",
    "build_effgen_config",
    "build_harness_tools",
    "create_driver",
]
