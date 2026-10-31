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
from effgen.core.thread_budget import MIN_CONTEXT_BUDGET_TOKENS, count_text_tokens
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
    # The completion-check tool the caller may register for the same loop. When
    # it is named in ``external_tools`` the loop yields the call instead of
    # executing it, so the driver can run the finish gate while the generator is
    # still alive and send the refusal back as the tool's result.
    finish_tool_name: str = "request_finish"
    max_finish_rejections: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.external_tools, frozenset):
            object.__setattr__(self, "external_tools", frozenset(self.external_tools))
        if not self.external_tools:
            raise BridgeError("at least one external tool is required")
        if self.max_format_corrections < 0:
            raise BridgeError("max_format_corrections must be >= 0")
        if self.max_finish_rejections < 0:
            raise BridgeError("max_finish_rejections must be >= 0")
        if not isinstance(self.finish_tool_name, str) or not self.finish_tool_name.strip():
            raise BridgeError("finish_tool_name must be a non-empty string")
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
        observation_input_tokens: int | None = None,
    ) -> None:
        self.session = session
        self.artifact_loader = artifact_loader
        self.observation_input_tokens = observation_input_tokens

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
        return self.render_observation(observation)

    def render_observation(self, observation: Observation) -> str:
        """Use the same bounded real-browser view at every loop entry point."""
        if self.observation_input_tokens is None:
            return render_observation(observation)
        from .observation_view import render_browser_view

        session = self._session()
        # Measure with both the native estimator and the project tokenizer.
        # Neither makes a provider request. Calibration headroom is reserved
        # by the per-view share, not by raising the configured input limit.
        view = render_browser_view(
            observation,
            max_tokens=self.observation_input_tokens // 8,
            count_tokens=lambda text: max(
                count_text_tokens(text), session.budget.counter.count_text(text)
            ),
        )
        if view.text != render_observation(observation):
            session.logger.emit(
                "observation_projection",
                page_id=observation.page_id,
                url=observation.url,
                observation_id=observation.observation_id,
                result={
                    "original_tokens": view.original_tokens,
                    "view_tokens": view.tokens,
                    "max_view_tokens": view.max_tokens,
                    "shown_elements": view.shown_elements,
                    "total_elements": view.total_elements,
                    "source": "max(effgen_estimate,project_counter)",
                    "original_preserved": True,
                },
            )
        return view.text

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
    context_budget = int(app.budget.input_tokens)
    if context_budget < MIN_CONTEXT_BUDGET_TOKENS:
        raise BridgeError(
            f"budget.input_tokens={context_budget} is below EffGen's minimum context "
            f"budget {MIN_CONTEXT_BUDGET_TOKENS}; the project value is passed through "
            "unchanged rather than silently raised"
        )
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
        # The project's per-run prompt budget is the native context budget: a
        # plain int makes EffGen compact its own conversation at this count
        # instead of deriving "auto" from the declared window.
        context_budget=context_budget,
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


def _render_action_result(
    tool: str, observation: Observation,
    renderer: Callable[[Observation], str] = render_observation,
) -> str:
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
    return f"{head}\n{renderer(observation)}"


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
        completion_gate: Callable[[object], Any] | None = None,
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
        # Optional phase-2 finish check. It is called only when the generator
        # produced a completed answer; a refusal turns that answer into an
        # honest incomplete end instead of closing the run as complete.
        self.completion_gate = completion_gate
        self._generator: Any = None
        self._pending_request: InteractiveToolRequest | None = None
        self._pending_action: WebAction | None = None
        self._awaiting_environment: WebAction | None = None
        self._corrections: list[dict[str, Any]] = []
        self._finish_rejections = 0
        self._model_turn = 0
        # The tool that produced the action currently awaiting the environment's
        # observation. Kept so a deferred handoff can name the truthful result.
        self._awaiting_tool: str | None = None
        self._terminal: DriverStep | None = None
        self._closed = False
        self._started = False
        # Optional G-wave context hooks. Both stay None unless the official entry
        # installs a protector, so the driver's text and send path are unchanged.
        # ``context_note_provider`` returns the harness necessary-context block to
        # append to the prompt this turn; ``final_send_hook`` runs once before
        # each model call with the exact prompt about to be sent.
        self.context_note_provider: Callable[[], str] | None = None
        self.final_send_hook: Callable[[Any], Any] | None = None

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

    @property
    def finish_rejections(self) -> int:
        """How many request_finish refusals were returned to the same loop."""

        return self._finish_rejections

    # ------------------------------------------------------------------- drive
    def start(self, *, note: str | None = None) -> DriverStep:
        """Open the generator and advance to its first request or answer.

        ``note`` is an optional program-side context line appended after the
        initial observation. The control/recovery wiring uses it for a restarted
        branch (snapshot loaded, generator stack not restored, unresolved side
        effects). It never replaces the task text or the observation.
        """

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
            task_prompt = task_prompt + "\n\n" + self.runtime.render_observation(session.observation)
        if note is not None and str(note).strip():
            task_prompt = task_prompt + "\n\n" + str(note).strip()
        task_prompt = self._with_context_note(task_prompt)
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

    def take_action(
        self,
        *,
        before_dispatch: Callable[[WebAction], None] | None = None,
    ) -> WebAction:
        """Consume the proposed action and hand it to the environment once.

        ``before_dispatch`` runs after the deadline check and before
        ``Session.consume``, with the still-proposed action. It is the ledger's
        pre-dispatch slot: returning normally licenses the dispatch, raising
        aborts it. The hook must not change the action or execute it; if it
        raises, the proposal stays unconsumed so the caller can cancel or retry
        after re-observing.
        """

        if self._pending_action is None or self._pending_request is None:
            raise BridgeError("no page action is waiting for the environment")
        if self._awaiting_environment is not None:
            raise BridgeError("the previous action has not been answered yet")
        self.session.check_deadline("action dispatch")
        action = self._pending_action
        if before_dispatch is not None:
            before_dispatch(action)
        tool = self._pending_request.tool if self._pending_request is not None else action.op.value
        action = self.session.consume(action.action_id)
        self._awaiting_environment = action
        self._awaiting_tool = tool
        self._pending_action = None
        return action

    def reject_action(self, exc: ActionRejected) -> DriverStep:
        """Send an action-serialization refusal into the same loop.

        The environment adapter can refuse a proposal before it reaches the
        platform (for example an empty non-submitting ``type``). That refusal is
        a format correction, not a terminal task failure: the unconsumed
        proposal is dropped, the correction text becomes the next model turn's
        tool result, and the same generator continues. The shared correction
        allowance still applies; when it is spent the run ends as
        ``FORMAT_ERROR``.
        """

        if self._closed:
            raise BridgeError("driver is closed")
        if self._pending_action is None or self._pending_request is None:
            raise BridgeError("no pending page action to reject")
        if self._awaiting_environment is not None:
            raise BridgeError("cannot reject an action that is already in flight")
        request = self._pending_request
        self.session.drop_proposal("format_rejected")
        self._pending_action = None
        self._pending_request = None
        step = self._record_rejection(request, exc)
        if step is not None:
            return step
        return self._pump(exc.correction, send=True)

    def accept_observation(
        self,
        observation: Observation,
        *,
        ingest: bool = True,
        note: str | None = None,
    ) -> DriverStep:
        """Ingest the environment's observation and resume the same generator.

        ``ingest=False`` is for a caller that already recorded the observation
        truthfully (the control/recovery entry records the in-flight result before
        it decides whether to pause). The caller must then have installed exactly
        this observation as the session's current one.

        ``note`` is an optional program-side line appended to the tool result the
        model reads next. It carries a bounded recovery instruction; it never
        replaces the observation.
        """

        action = self._awaiting_environment
        if action is None:
            raise BridgeError("no action is awaiting the environment's observation")
        if self._closed:
            raise BridgeError("driver is closed")
        # The action was already executed; recording the observation it returned
        # is local bookkeeping, not new model or action work. The deadline is
        # checked before the loop is resumed, so a task that has run out of time
        # still owes no further turn.
        if ingest:
            self.session.ingest_observation(observation)
        else:
            current = self.session.observation
            if current is None or current.observation_id != observation.observation_id:
                raise BridgeError(
                    "accept_observation(ingest=False) needs the entry to install the "
                    "same observation first"
                )
        tool = self._pending_request.tool if self._pending_request is not None else (
            self._awaiting_tool or action.op.value
        )
        self._awaiting_environment = None
        self._awaiting_tool = None
        self._pending_action = None
        self._pending_request = None
        try:
            self.session.check_deadline("action feedback")
        except DeadlineExceeded as exc:
            return self._finish_deadline(exc)
        text = _render_action_result(tool, observation, self.runtime.render_observation)
        if note is not None and str(note).strip():
            text = text + "\n\n" + str(note).strip()
        text = self._with_context_note(text)
        return self._pump(text, send=True)

    def expand_pending(self, sub_actions: Sequence[WebAction]) -> tuple[WebAction, ...]:
        """Replace the pending proposal with an ordered sub-action sequence.

        The caller (the official step path) owns the queue; the driver only
        moves the session from "one proposal" to "the proposal is not dispatched".
        """

        if self._closed:
            raise BridgeError("driver is closed")
        if self._pending_action is None or self._pending_request is None:
            raise BridgeError("no pending page action to expand")
        if self._awaiting_environment is not None:
            raise BridgeError("cannot expand an action that is already in flight")
        expanded = self.session.expand_proposal(self._pending_action.action_id, sub_actions)
        self._pending_action = None
        return expanded

    def dispatch_split(self, action: WebAction) -> WebAction:
        """Propose and consume one sub-action of an expanded proposal."""

        if self._closed:
            raise BridgeError("driver is closed")
        if self._pending_request is None:
            raise BridgeError("no expanded proposal is open")
        self.session.propose(action)
        dispatched = self.session.consume(action.action_id)
        self._awaiting_environment = None
        return dispatched

    def accept_split_observation(
        self,
        observation: Observation,
        *,
        final: bool,
    ) -> DriverStep | None:
        """Ingest one sub-action's observation.

        An intermediate observation is local bookkeeping only: it is logged and
        bound to its sub-action but never sent to the model. The final observation
        is the result of the model's original tool call, so the same generator is
        resumed with it.
        """

        if self._closed:
            raise BridgeError("driver is closed")
        if self._pending_request is None:
            raise BridgeError("no expanded proposal is open")
        tool = self._pending_request.tool
        self.session.ingest_observation(observation)
        self._pending_action = None
        if not final:
            return None
        self._pending_request = None
        text = _render_action_result(tool, observation, self.runtime.render_observation)
        text = self._with_context_note(text)
        return self._pump(text, send=True)

    def abort_split(
        self,
        *,
        status: DriverStatus,
        message: str,
        finish_reason: TerminationReason,
        claim: str,
    ) -> DriverStep:
        """End the task when an expanded action stops part-way through.

        Only the executed prefix is recorded; the original multi-line text is
        never reported as fully entered.
        """

        self._pending_action = None
        self._pending_request = None
        self._awaiting_environment = None
        return self._terminate(
            status,
            message,
            finish_reason=finish_reason,
            claim=claim,
        )

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
            + self.runtime.render_observation(observation)
        )
        text = self._with_context_note(text)
        return self._pump(text, send=True)

    def resume_after_takeover(
        self,
        feedback: Observation,
        fresh: Observation,
        *,
        note: str | None = None,
    ) -> DriverStep:
        """Resume the same loop after a user handoff, without faking the action.

        The action that was in flight when the user took control really executed,
        and its feedback observation was recorded truthfully by the entry before
        the handoff. That result is reported as the tool's result. The page was
        then re-observed after the explicit resume, so the fresh observation is
        the current one and the earlier element numbers are invalid.

        ``feedback`` must already be the session's recorded observation for the
        executed action, and ``fresh`` the session's current observation. No
        action is replayed and no completion is claimed.
        """

        if self._closed:
            raise BridgeError("driver is closed")
        action = self._awaiting_environment
        if action is None:
            raise BridgeError("no executed action is awaiting a handoff resume")
        session = self.session
        if session.state is not SessionState.RUNNING:
            raise BridgeError("resume_after_takeover needs a running session")
        if session.in_flight_action is not None or session.proposed_action is not None:
            raise BridgeError("resume_after_takeover cannot run with an open action")
        if not isinstance(feedback, Observation) or not isinstance(fresh, Observation):
            raise BridgeError("resume_after_takeover needs two normal observations")
        current = session.observation
        if current is None or current.observation_id != fresh.observation_id:
            raise BridgeError("the fresh observation must be the session's current one")
        # The request belongs to the executed action whose feedback is reported
        # here; it is not a live proposal and is closed by this resume.
        request = self._pending_request
        tool = (
            request.tool
            if request is not None
            else (self._awaiting_tool or action.op.value)
        )
        self._awaiting_environment = None
        self._awaiting_tool = None
        self._pending_request = None
        head = (
            f"Result of {tool}: the action was executed before the browser was handed "
            "to the user, and its recorded feedback is below. The user then took "
            "control (login or verification); the page has since been re-observed, so "
            "the element numbers and proposals built from the earlier observation are "
            "no longer valid. Use only the fresh observation below. Do not claim this "
            "action had a different effect and do not replay it blindly.\n"
            + _render_action_result(tool, feedback, self.runtime.render_observation)
        )
        text = (
            head
            + "\n\nFresh observation after the user resumed:\n"
            + self.runtime.render_observation(fresh)
        )
        if note is not None and str(note).strip():
            text = text + "\n\n" + str(note).strip()
        text = self._with_context_note(text)
        return self._pump(text, send=True)

    def finish_incomplete(
        self,
        message: str,
        *,
        claim: str,
        feedback_ingested: bool = False,
    ) -> DriverStep:
        """End the run honestly as incomplete without inventing a completed answer.

        ``feedback_ingested`` says the entry already recorded the executed
        action's feedback, so the driver's deferred-feedback slot is closed and the
        session can finish. No completion is claimed.
        """

        if self._closed:
            raise BridgeError("driver is closed")
        if self._terminal is not None:
            return self._terminal
        if feedback_ingested:
            self._awaiting_environment = None
            self._awaiting_tool = None
            self._pending_action = None
            self._pending_request = None
        return self._terminate(
            DriverStatus.INCOMPLETE,
            message,
            finish_reason=TerminationReason.INCOMPLETE,
            claim=claim,
        )

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

    def _with_context_note(self, text: str) -> str:
        """Append the harness necessary-context block; a no-op when disabled."""

        provider = self.context_note_provider
        if provider is None:
            return text
        note = str(provider() or "")
        if not note:
            return text
        return text + "\n\n" + note

    def _before_model_call(self, prompt: Any) -> None:
        session = self.session
        session.check_deadline("model call")
        # Close out instead of starting a call the platform would kill at the
        # deadline; this raises DeadlineMarginReached (a DeadlineExceeded) which
        # the pump turns into the normal terminal deadline step.
        session.check_model_call_margin("model call")
        self._model_turn += 1
        hook = self.final_send_hook
        if hook is not None:
            # G-wave: the last check before this exact prompt reaches the model
            # adapter. It raises rather than letting a protected region be sent
            # incomplete or an over-budget request go out silently.
            hook(prompt)
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
            if (
                request.tool == self.settings.finish_tool_name
                and self.completion_gate is not None
            ):
                # The completion check runs while the generator is suspended at
                # this call. A refusal is sent back as the tool's result and the
                # same loop continues; acceptance closes the run as complete.
                terminal, feedback = self._request_finish_decision()
                if terminal is not None:
                    return terminal
                assert feedback is not None
                value = feedback
                send = True
                continue
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

    def _request_finish_decision(self) -> tuple[DriverStep | None, str | None]:
        """Run the completion gate for an in-loop ``request_finish`` call.

        ``None`` plus feedback text means the refusal was returned to the same
        generator. A step means the run ended here (accepted, exhausted or the
        check itself failed).
        """

        try:
            decision = self.completion_gate(None)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - never claim completion on a failed check
            self._emit_completion_event(
                stage="request_finish",
                accepted=None,
                decision=None,
                attempt=self._finish_rejections + 1,
                continued=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return (
                self._terminate(
                    DriverStatus.INCOMPLETE,
                    f"completion check failed: {type(exc).__name__}: {exc}; "
                    "the run ends incomplete",
                    finish_reason=TerminationReason.INCOMPLETE,
                    claim="completion was not confirmed",
                ),
                None,
            )
        remaining = tuple(str(item) for item in getattr(decision, "remaining_goal_ids", ()))
        conditions = tuple(
            str(item) for item in getattr(decision, "remaining_conditions", ())
        )
        reason = str(getattr(decision, "reason", "completion refused"))
        state_status = str(getattr(decision, "state_completion_status", "incomplete"))
        accepted = bool(getattr(decision, "accepted", False))
        if accepted:
            self._emit_completion_event(
                stage="request_finish",
                accepted=True,
                decision=decision,
                attempt=self._finish_rejections,
                continued=False,
            )
            return (
                self._terminate(
                    DriverStatus.COMPLETED,
                    "completion verified: every declared requirement is verified",
                    finish_reason=TerminationReason.COMPLETED,
                    claim="every declared requirement is verified",
                ),
                None,
            )

        self._finish_rejections += 1
        continued = self._finish_rejections <= self.settings.max_finish_rejections
        self._emit_completion_event(
            stage="request_finish",
            accepted=False,
            decision=decision,
            attempt=self._finish_rejections,
            continued=continued,
        )
        message = (
            "completion refused; the task is not fully verified. "
            f"Remaining goals: {list(remaining)}. "
            f"Remaining conditions: {list(conditions)}. {reason}"
        )
        if not continued:
            return (
                self._terminate(
                    DriverStatus.INCOMPLETE,
                    "completion was refused "
                    f"{self._finish_rejections} times in this loop; ending honestly "
                    f"as incomplete. {message}",
                    finish_reason=TerminationReason.INCOMPLETE,
                    claim=(
                        "completion was not confirmed after "
                        f"{self._finish_rejections} refusals; partial progress is "
                        "not complete"
                    ),
                ),
                None,
            )
        feedback = (
            "Result of request_finish: completion is NOT confirmed. "
            f"Remaining goals: {list(remaining)}. "
            f"Remaining conditions: {list(conditions)}. "
            f"Reason: {reason}. State completion status: {state_status}. "
            "Continue working in this same loop: use normal web actions to obtain "
            "the missing page evidence, record the state you were asked to record, "
            "then call request_finish again. Do not send a final answer that treats "
            "partial progress as complete."
        )
        return None, feedback

    def _emit_completion_event(
        self,
        *,
        stage: str,
        accepted: bool | None,
        decision: Any,
        attempt: int,
        continued: bool,
        error: str | None = None,
    ) -> None:
        """Record what the finish check decided and whether the loop continued."""

        observation = self.session.observation
        result: dict[str, Any] = {
            "stage": stage,
            "accepted": accepted,
            "continued": bool(continued),
            "attempt": int(attempt),
        }
        if stage == "request_finish":
            result["tool"] = self.settings.finish_tool_name
        if decision is not None:
            to_dict = getattr(decision, "to_dict", None)
            if callable(to_dict):
                try:
                    result["decision"] = to_dict()
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
            for name in (
                "remaining_goal_ids",
                "remaining_conditions",
                "reason",
                "state_completion_status",
                "objects_remaining",
            ):
                value = getattr(decision, name, None)
                if value is not None:
                    result[name] = list(value) if isinstance(value, tuple) else value
        if error is not None:
            result["error"] = error
        try:
            self.session.logger.emit(
                "verification",
                page_id=None if observation is None else observation.page_id,
                url=None if observation is None else observation.url,
                observation_id=None
                if observation is None
                else observation.observation_id,
                result=result,
            )
        except Exception:  # noqa: BLE001 - logging cannot change the decision
            pass

    def _finish_response(self, response: AgentResponse | None) -> DriverStep:
        if not isinstance(response, AgentResponse):
            return self._finish_error(
                ProtocolError("the interactive loop ended without an AgentResponse")
            )
        success = bool(getattr(response, "success", False))
        output = str(getattr(response, "output", "") or "")
        metadata = getattr(response, "metadata", None) or {}
        detail = str(metadata.get("reason") or "") if isinstance(metadata, dict) else ""
        if success and output.strip():
            reason = TerminationReason.COMPLETED
            claim = output.strip()
            message = output
            status = DriverStatus.COMPLETED
            if self.completion_gate is not None:
                try:
                    decision = self.completion_gate(None)
                except Exception as exc:  # noqa: BLE001 - never claim completion on a failed check
                    self._emit_completion_event(
                        stage="finish_gate",
                        accepted=None,
                        decision=None,
                        attempt=0,
                        continued=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    remaining = (
                        f"completion check failed: {type(exc).__name__}: {exc}; "
                        "run ends incomplete"
                    )
                    return self._terminate(
                        DriverStatus.INCOMPLETE,
                        remaining,
                        finish_reason=TerminationReason.INCOMPLETE,
                        claim="completion was not confirmed",
                    )
                accepted = bool(getattr(decision, "accepted", False))
                if not accepted:
                    self._emit_completion_event(
                        stage="finish_gate",
                        accepted=False,
                        decision=decision,
                        attempt=0,
                        continued=False,
                    )
                    remaining = getattr(decision, "remaining_goal_ids", ())
                    conditions = getattr(decision, "remaining_conditions", ())
                    detail_text = str(getattr(decision, "reason", "completion refused"))
                    text = (
                        "Completion refused; the task is not fully verified. "
                        f"Remaining goals: {list(remaining)}. "
                        f"Remaining conditions: {list(conditions)}. "
                        f"{detail_text}"
                    )
                    return self._terminate(
                        DriverStatus.INCOMPLETE,
                        text,
                        finish_reason=TerminationReason.INCOMPLETE,
                        claim="completion was not confirmed; partial progress is not complete",
                    )
                self._emit_completion_event(
                    stage="finish_gate",
                    accepted=True,
                    decision=decision,
                    attempt=0,
                    continued=False,
                )
        elif detail == "generation_failed":
            # EffGen reports a failed model generation as a normal return, not an
            # exception. That is an error, not an incomplete answer.
            error = metadata.get("error") if isinstance(metadata, dict) else None
            error_message = ""
            if isinstance(error, dict):
                error_message = str(error.get("message") or "")
            reason = TerminationReason.ERROR
            claim = "the model generation failed"
            message = error_message or "the model generation failed"
            status = DriverStatus.ERROR
        else:
            reason = TerminationReason.INCOMPLETE
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
    completion_gate: Callable[[object], Any] | None = None,
) -> EffGenLoopDriver:
    """Build the runtime, the agent and the driver for one task session."""

    real_browser = app_config is not None and app_config.backend in {"playwright", "browser-use"}
    runtime = HarnessToolRuntime(
        session, artifact_loader=artifact_loader,
        observation_input_tokens=(
            min(app_config.budget.input_tokens, session.budget.limits.input_tokens)
            if real_browser else None
        ),
    )
    agent = build_effgen_agent(
        model, runtime=runtime, app_config=app_config, extra_tools=extra_tools
    )
    return EffGenLoopDriver(
        session,
        agent=agent,
        runtime=runtime,
        settings=settings,
        completion_gate=completion_gate,
    )


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
