"""SDK entry point for the official WebArena platform.

The frozen ``third_party_agent`` client imports ``webagent.official_agent:
OfficialAgent`` and drives it with ``reset`` / ``step`` / ``close``. Each task
gets one EffGen Agent and one interactive generator; the platform, not this
process, executes page actions.

Drive order (matching the phase-1 contract):

* ``reset`` creates an empty task container and closes any previous task.
* the first ``step`` reads the complete task, observation and deadline, then
  creates the Session, OfficialBackend, OfficialModel and EffGen driver;
* each later ``step`` turns that request's observation into the next observation,
  binds it to the previous action, sends feedback into the same generator and
  returns at most one action;
* a terminal generator state becomes a ``stop [说明]`` command, never an
  "official passed" claim;
* ``close`` is idempotent and releases only local task resources.

No real connection, credential file or network call is made by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)

from .backends.official import (
    DEFAULT_TEXT_MODE,
    OfficialBackend,
    TextMode,
    normalize_text_mode,
    serialize_stop,
)
from .budget import Budget, Deadline, build_prompt_messages
from .clock import Clock, SystemClock
from .config import AppConfig, OfficialBehaviors, load_config, parse_config
from .context import ContextProtector
from .control import (
    ControlCoordinator,
    ControlState,
    ControlWaitTimeout,
    FileControlSource,
)
from .control.detection import TakeoverSignal
from .control.restart import load_restart_snapshot
from .control.signals import ControlSource
from .effgen_bridge import (
    DriverStatus,
    DriverStep,
    EffGenLoopDriver,
    LoopSettings,
    create_driver,
)
from .errors import ActionRejected, DeadlineExceeded, SessionStateError
from .events import EventLogger
from .experience import ExperienceLoader, ExperienceSelection
from .models.official import OfficialCallRecord, OfficialModel
from .protocol import (
    ActionOp,
    FeedbackStatus,
    Finish,
    Observation,
    TerminationReason,
    WebAction,
    new_action_id,
    validate_run_id,
)
from .recovery.wiring import OfficialRecoveryRuntime, RecoveryNote
from .session import Session, SessionState
from .state.integration import EffectIntent, EffectPolicy
from .state.records import (
    EffectStatus,
    EvidenceRef,
    GoalStatus,
    PendingEffect,
    StateConflict,
    StateError,
    StateValidationError,
)
from .state.store import StateStore
from .state.tools import StateToolRuntime
from .tools import WEB_TOOL_NAMES
from .verification import (
    DETERMINISTIC_CHECKS,
    CompletionChecker,
    CompletionToolRuntime,
    GoalVerifier,
    VerificationCondition,
    VerificationError,
    VerificationRequest,
    VerificationTrigger,
    build_completion_tools,
    make_finish_gate,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "official.yaml"


@dataclass(frozen=True)
class PlannedSubAction:
    """One wire action of a decomposed multi-line ``type`` proposal."""

    op: ActionOp
    args: Mapping[str, Any]


ENTER = "Enter"


def plan_type_text(text: str, *, submit: bool) -> tuple[PlannedSubAction, ...]:
    """Split multi-line text into ``type`` / ``press [Enter]`` sub-actions.

    The platform has no unescape routine, so a line break must not travel inside
    one ``type`` field. Each line break becomes an explicit Enter press between
    typed segments; the model's ``submit`` flag stays on the last typed segment.
    CRLF and CR are treated as one line break.
    """

    if not isinstance(text, str):
        raise ValueError("type text must be a string")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    parts: list[PlannedSubAction] = []
    for index, line in enumerate(lines):
        if index:
            parts.append(PlannedSubAction(ActionOp.PRESS, {"keys": ENTER}))
        if line:
            last = index == len(lines) - 1
            parts.append(
                PlannedSubAction(
                    ActionOp.TYPE,
                    {"text": line, "submit": bool(submit and last)},
                )
            )
    if submit and lines and not lines[-1]:
        # The last line is empty, so submit needs its own Enter after it.
        parts.append(PlannedSubAction(ActionOp.PRESS, {"keys": ENTER}))
    return tuple(parts)


def _default_config() -> AppConfig:
    if _DEFAULT_CONFIG_PATH.is_file():
        return load_config(_DEFAULT_CONFIG_PATH)
    return parse_config(
        {
            "schema_version": "webagent.config.v1",
            "profile": "official",
            "backend": "official",
            "runtime_dir": "runtime",
            "agent": {
                "strategy": "single",
                "tool_calling_mode": "react",
                "max_iterations": None,
                "capability_probe": False,
                "sub_agents": False,
                "fallback": False,
                "speculative_execution": False,
                "cross_task_memory": False,
            },
            "model": {
                "provider": "official",
                "name": "platform-unified",
                "context_length": 32768,
            },
        }
    )


def _safe_local_run_id(platform_run_id: str) -> str:
    """A path-safe local run id that still maps deterministically to the task."""

    try:
        return validate_run_id(platform_run_id)
    except Exception:  # noqa: BLE001 - the official id can contain other characters
        digest = hashlib.sha256(platform_run_id.encode("utf-8")).hexdigest()[:40]
        return "official-" + digest


def _deadline_value(request: Mapping[str, Any], fallback: float | None) -> float:
    value = request.get("execution_deadline")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if fallback is not None:
        return float(fallback)
    raise ValueError("step request has no execution_deadline")


# ------------------------------------------------------- official ledger wiring
GOAL_DECLARE_TOOL = "goal_declare"


def _parameter(
    name: str,
    description: str,
    *,
    type: ParameterType = ParameterType.STRING,
    required: bool = False,
    default: Any = None,
    enum: Sequence[Any] | None = None,
    items_type: ParameterType | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        type=type,
        description=description,
        required=required,
        default=default,
        enum=list(enum) if enum is not None else None,
        items_type=items_type,
    )


def _state_error_text(exc: StateError) -> str:
    if isinstance(exc, StateConflict):
        return json.dumps(exc.as_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"{type(exc).__name__}: {exc}"


class _OfficialLedgerTool(BaseTool):
    """A synchronous ledger tool; calling it never starts an event loop."""

    async def _execute(self, **kwargs: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError("official ledger tools execute synchronously")

    def execute(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        ok, error = self.validate_parameters(**kwargs)
        if not ok:
            return ToolResult(
                success=False, output=None, error=f"Parameter validation failed: {error}"
            )
        try:
            return ToolResult(success=True, output=self._run(**kwargs))
        except StateError as exc:
            metadata: dict[str, Any] = {"error_type": type(exc).__name__}
            current = getattr(exc, "current", None)
            if current is not None:
                metadata["current_revision"] = current.revision
                metadata["current_plan_version"] = current.plan_version
            return ToolResult(
                success=False,
                output=None,
                error=_state_error_text(exc),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - the loop reads failures as text
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution failed: {type(exc).__name__}: {exc}",
                metadata={"error_type": type(exc).__name__},
            )

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


class _BoundStateReadTool(_OfficialLedgerTool):
    """The compact ledger view; reading never takes a revision from the model."""

    def __init__(self, runtime: StateToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="state_read",
                description=(
                    "Read the task ledger: requirements, facts, plan, pending side "
                    "effects and the current completion status. The harness keeps "
                    "revision, plan version and branch current, so you do not need to "
                    "read before every update."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    _parameter(
                        "goal_id",
                        "Optional goal id to limit the view to one requirement.",
                    ),
                    _parameter(
                        "include_facts",
                        "Include recorded facts; default true.",
                        type=ParameterType.BOOLEAN,
                        default=True,
                    ),
                ],
            )
        )
        self._runtime = runtime

    def _run(self, goal_id: str | None = None, include_facts: bool = True) -> str:
        return self._runtime.read(goal_id=goal_id, include_facts=include_facts)


class _BoundStateUpdateTool(_OfficialLedgerTool):
    """``state_update`` with revision, plan version and branch bound by the run.

    The model supplies business content only. Engineering fields default to the
    current ledger values; if the model does supply one, the normal stale
    revision/plan version checks still refuse it and return the current state.
    """

    def __init__(self, runtime: StateToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="state_update",
                description=(
                    "Record requirement/fact updates for this task. The current "
                    "revision, plan version and branch are filled in by the harness; "
                    "you only provide the business content. goals[]: id, requirement, "
                    "acceptance_criteria[], depends_on[], object_scope[], status, "
                    "completed_object_ids[]; facts[]: id, object, field, value, "
                    "kind=observed|inferred, source, validity, goal_id. This tool "
                    "never verifies a goal; success only means the update was saved."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    _parameter(
                        "goals",
                        "Goal records to add or update.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.OBJECT,
                    ),
                    _parameter(
                        "facts",
                        "Fact records to add or update.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.OBJECT,
                    ),
                    _parameter(
                        "base_revision",
                        "Optional; the harness binds the current revision.",
                        type=ParameterType.INTEGER,
                    ),
                    _parameter(
                        "plan_version",
                        "Optional; the harness binds the current plan version.",
                        type=ParameterType.INTEGER,
                    ),
                    _parameter(
                        "branch",
                        "Optional; must equal the branch this task is running.",
                    ),
                    _parameter("note", "Optional local note for this revision."),
                ],
            )
        )
        self._runtime = runtime

    def _run(self, **kwargs: Any) -> str:
        state = self._runtime.store.state
        payload = dict(kwargs)
        payload.setdefault("base_revision", state.revision)
        payload.setdefault("plan_version", state.plan_version)
        payload.setdefault("branch", self._runtime.branch)
        return self._runtime.update(payload)


class _BoundPlanUpdateTool(_OfficialLedgerTool):
    """``plan_update`` with the same program-bound engineering fields."""

    def __init__(self, runtime: StateToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="plan_update",
                description=(
                    "Propose a plan revision. The harness fills in the current "
                    "revision, plan version and branch. plan[] nodes: id, goal_id, "
                    "depends_on[], input_fact_ids[], object_scope[], resources{}, "
                    "status. goals may add requirements or update unfinished ones. "
                    "remove_goal_ids can only remove verified requirements; an "
                    "unfinished requirement is refused and should be marked blocked. "
                    "Success only means the new plan version was saved."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    _parameter(
                        "goals",
                        "Goal records introduced or updated by this revision.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.OBJECT,
                    ),
                    _parameter(
                        "plan",
                        "Plan nodes to add or update in the new plan version.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.OBJECT,
                    ),
                    _parameter(
                        "remove_goal_ids",
                        "Verified goal ids the revision removes.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.STRING,
                    ),
                    _parameter(
                        "remove_plan_node_ids",
                        "Plan node ids the revision removes.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.STRING,
                    ),
                    _parameter(
                        "base_revision",
                        "Optional; the harness binds the current revision.",
                        type=ParameterType.INTEGER,
                    ),
                    _parameter(
                        "plan_version",
                        "Optional; the harness binds the current plan version.",
                        type=ParameterType.INTEGER,
                    ),
                    _parameter(
                        "branch",
                        "Optional; must equal the branch this task is running.",
                    ),
                    _parameter("note", "Optional local note for this revision."),
                ],
            )
        )
        self._runtime = runtime

    def _run(self, **kwargs: Any) -> str:
        state = self._runtime.store.state
        payload = dict(kwargs)
        payload.setdefault("base_revision", state.revision)
        payload.setdefault("plan_version", state.plan_version)
        payload.setdefault("branch", self._runtime.branch)
        return self._runtime.plan_update(payload)


class GoalDeclarationRuntime:
    """The model-declared goals and the deterministic conditions to check.

    The ledger keeps the business requirement and readable acceptance criteria;
    this runtime keeps the executable condition specification. Evidence
    attribution is bound by the program from the current normal observation, so
    a model cannot cite an evidence file it did not see or move evidence between
    runs.
    """

    def __init__(
        self,
        store: StateStore,
        session: Session,
        *,
        branch: str = "root",
    ) -> None:
        if not isinstance(store, StateStore):
            raise StateValidationError("GoalDeclarationRuntime needs a StateStore")
        self.store = store
        self.session = session
        self.branch = branch
        self._conditions: dict[str, tuple[VerificationCondition, ...]] = {}

    @property
    def goal_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._conditions))

    def declare(
        self,
        goal_id: str,
        requirement: str,
        raw_conditions: Sequence[Mapping[str, Any]],
        object_scope: Sequence[str] | None = None,
        depends_on: Sequence[str] | None = None,
    ) -> tuple[VerificationCondition, ...]:
        if not isinstance(goal_id, str) or not goal_id.strip():
            raise StateValidationError("goal_declare needs a non-empty goal_id")
        if not isinstance(requirement, str) or not requirement.strip():
            raise StateValidationError("goal_declare needs a non-empty requirement")
        conditions = self._build_conditions(raw_conditions)
        dependencies = tuple(str(item) for item in (depends_on or ()))
        state = self.store.state
        payload = {
            "base_revision": state.revision,
            "plan_version": state.plan_version,
            "branch": self.branch,
            "goals": [
                {
                    "id": goal_id,
                    "requirement": requirement,
                    "acceptance_criteria": [item.description for item in conditions],
                    "object_scope": list(object_scope or ()),
                    "depends_on": list(dependencies),
                    "status": GoalStatus.RUNNING.value,
                }
            ],
            "note": "official goal declaration",
        }
        StateToolRuntime(self.store, branch=self.branch).update(payload)
        self._conditions[goal_id] = conditions
        return conditions

    def _build_conditions(
        self, raw_conditions: Sequence[Mapping[str, Any]]
    ) -> tuple[VerificationCondition, ...]:
        if isinstance(raw_conditions, (str, bytes)) or not isinstance(
            raw_conditions, Sequence
        ) or not raw_conditions:
            raise StateValidationError(
                "goal_declare needs a non-empty conditions list"
            )
        conditions: list[VerificationCondition] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_conditions, start=1):
            if not isinstance(raw, Mapping):
                raise StateValidationError("each condition must be an object")
            check = str(raw.get("check") or "").strip()
            if check not in DETERMINISTIC_CHECKS:
                raise StateValidationError(
                    f"goal_declare condition check must be one of "
                    f"{', '.join(sorted(DETERMINISTIC_CHECKS))}; got {check!r}"
                )
            condition_id = str(raw.get("id") or f"c{index}").strip()
            if condition_id in seen:
                raise StateValidationError(
                    f"goal_declare repeats condition id {condition_id!r}"
                )
            seen.add(condition_id)
            expected = raw.get("expected")
            description = str(
                raw.get("description")
                or f"{check}: {expected if expected is not None else condition_id}"
            )
            try:
                condition = VerificationCondition(
                    condition_id=condition_id,
                    description=description,
                    check=check,
                    expected=expected,
                    role=raw.get("role"),
                    name=raw.get("name"),
                )
            except VerificationError as exc:
                raise StateValidationError(f"invalid condition {condition_id!r}: {exc}") from exc
            conditions.append(condition)
        return tuple(conditions)

    def _bound_evidence(self, observation: Observation) -> tuple[EvidenceRef, ...]:
        for kind in ("screenshot", "html", "dom"):
            artifact = observation.artifacts.get(kind)
            if artifact is None or not artifact.available:
                continue
            if not artifact.reference or not artifact.sha256 or artifact.byte_size is None:
                continue
            return (
                EvidenceRef(
                    evidence_id=f"ev-{observation.observation_id}-{kind}",
                    run_id=observation.run_id,
                    branch_id=self.branch,
                    reference=artifact.reference,
                    sha256=artifact.sha256,
                    byte_size=int(artifact.byte_size),
                    kind=artifact.kind,
                    observation_id=observation.observation_id,
                ),
            )
        # No binary artifact came with this observation. The run still saved the
        # exact observation snapshot the deterministic check reads; citing it
        # keeps the check evidence-backed without inventing a screenshot.
        try:
            reference = self.session.evidence.save_observation(observation)
            data = self.session.evidence.resolve_reference(reference).read_bytes()
        except Exception:  # noqa: BLE001 - an unverifiable citation is simply absent
            return ()
        return (
            EvidenceRef(
                evidence_id=f"ev-{observation.observation_id}-observation",
                run_id=observation.run_id,
                branch_id=self.branch,
                reference=reference,
                sha256=hashlib.sha256(data).hexdigest(),
                byte_size=len(data),
                observation_id=observation.observation_id,
            ),
        )

    def requests(self) -> tuple[VerificationRequest, ...]:
        """One request per declared, not-yet-verified goal, for this observation."""

        observation = self.session.observation
        if observation is None:
            return ()
        evidence = self._bound_evidence(observation)
        evidence_ids = tuple(ref.evidence_id for ref in evidence)
        state = self.store.state
        requests: list[VerificationRequest] = []
        for goal_id in sorted(self._conditions):
            goal = state.goals.get(goal_id)
            if goal is None or goal.status is GoalStatus.VERIFIED:
                continue
            conditions = self._conditions[goal_id]
            if evidence_ids:
                conditions = tuple(
                    replace(item, evidence_ids=evidence_ids) for item in conditions
                )
            requests.append(
                VerificationRequest(
                    goal_id=goal_id,
                    subject_id=(
                        goal.object_scope[0] if goal.object_scope else goal_id
                    ),
                    conditions=conditions,
                    observation=observation,
                    evidence_scope=evidence,
                    trigger=VerificationTrigger.COMPLETION_CHECK,
                    branch=self.branch,
                    note="official entry completion check",
                )
            )
        return tuple(requests)


class _GoalDeclareTool(_OfficialLedgerTool):
    """Declare a business requirement and the deterministic checks it needs."""

    def __init__(self, runtime: GoalDeclarationRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name=GOAL_DECLARE_TOOL,
                description=(
                    "Declare one requirement the user asked for, with the visible "
                    "conditions that would prove it. Goal ids are yours; the harness "
                    "binds revision, branch and evidence. conditions[]: id, check "
                    f"({'|'.join(sorted(DETERMINISTIC_CHECKS))}), description, "
                    "expected, role, name. Declare a goal with visible_value_equals "
                    "when a field's value must be exactly a certain text."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    _parameter("goal_id", "Short id for this requirement.", required=True),
                    _parameter(
                        "requirement",
                        "The requirement in the user's words.",
                        required=True,
                    ),
                    _parameter(
                        "conditions",
                        "Deterministic conditions that prove the requirement.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.OBJECT,
                        required=True,
                    ),
                    _parameter(
                        "object_scope",
                        "Optional business object ids this requirement covers.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.STRING,
                    ),
                    _parameter(
                        "depends_on",
                        "Optional goal ids this requirement depends on.",
                        type=ParameterType.ARRAY,
                        items_type=ParameterType.STRING,
                    ),
                ],
            )
        )
        self._runtime = runtime

    def _run(
        self,
        goal_id: str,
        requirement: str,
        conditions: Sequence[Mapping[str, Any]],
        object_scope: Sequence[str] | None = None,
        depends_on: Sequence[str] | None = None,
    ) -> str:
        declared = self._runtime.declare(
            goal_id, requirement, conditions, object_scope, depends_on
        )
        return json.dumps(
            {
                "declared": True,
                "goal_id": goal_id,
                "conditions": [item.to_dict() for item in declared],
                "state_revision": self._runtime.store.state.revision,
                "message": (
                    "The requirement was recorded. This only records what must be "
                    "proved; request_finish performs the evidence-backed check."
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def build_official_state_tools(
    runtime: StateToolRuntime,
    plan: GoalDeclarationRuntime,
) -> list[BaseTool]:
    """The ledger tools registered for the official entry when state_tools is on."""

    return [
        _BoundStateReadTool(runtime),
        _BoundStateUpdateTool(runtime),
        _BoundPlanUpdateTool(runtime),
        _GoalDeclareTool(plan),
    ]


class OfficialAgent:
    """One task's official SDK adapter, instantiated by the SDK factory."""

    def __init__(
        self,
        *,
        app_config: AppConfig | None = None,
        runtime_dir: str | Path | None = None,
        clock: Clock | None = None,
        model_api: Any | None = None,
        browser_api: Any | None = None,
        download_artifacts: frozenset[str] | set[str] | tuple[str, ...] | None = None,
        control_source: ControlSource | None = None,
        control_wait_timeout: float | None = None,
    ) -> None:
        self.app_config = app_config or _default_config()
        if self.app_config.profile != "official" or self.app_config.backend != "official":
            raise ValueError("OfficialAgent needs the official profile and backend")
        self.text_mode: TextMode = normalize_text_mode(
            self.app_config.official.text_mode
            if self.app_config.official.text_mode
            else DEFAULT_TEXT_MODE
        )
        self.runtime_dir = (
            Path(runtime_dir)
            if runtime_dir is not None
            else _PROJECT_ROOT / "runtime"
        )
        self.clock = clock or SystemClock()
        self.model_api = model_api
        self.browser_api = browser_api
        self.download_artifacts = (
            frozenset({"screenshot"})
            if download_artifacts is None
            else frozenset(download_artifacts)
        )
        self._platform_run_id: str | None = None
        self._local_run_id: str | None = None
        self._reset_deadline: float | None = None
        self._session: Session | None = None
        self._backend: OfficialBackend | None = None
        self._model: OfficialModel | None = None
        self._driver: EffGenLoopDriver | None = None
        self._store: StateStore | None = None
        self._verification_plan: GoalDeclarationRuntime | None = None
        # G wave: approved local text experience, loaded once per task. It only
        # contributes prompt text; it never registers a tool or an action path.
        self._experience: ExperienceSelection | None = None
        self._pending_action: WebAction | None = None
        self._pending_observation: Observation | None = None
        self._terminal_step: DriverStep | None = None
        self._last_stop_action: dict[str, str] | None = None
        self._finished = False
        self._step_index = 0
        self._closed = False
        self._split_plan: deque[PlannedSubAction] = deque()
        self._split_total = 0
        self._split_index = 0
        self._split_in_flight: WebAction | None = None
        self._split_target: str | None = None
        self._split_active = False
        self._request_limit_detail: str | None = None
        # R wave: control/recovery wiring. Every path below stays inactive while
        # ``official.behaviors.recovery`` is false.
        self._control_source = control_source
        self._control_wait_timeout = control_wait_timeout
        self._control: ControlCoordinator | None = None
        self._recovery: OfficialRecoveryRuntime | None = None
        self._effect_policy: EffectPolicy | None = None
        self._prepared_effect: tuple[EffectIntent, PendingEffect] | None = None
        self._awaiting_reobserve: WebAction | None = None
        self._reobserve_start: bool = False
        self._deferred_feedback: tuple[WebAction, Observation] | None = None
        self._start_note: str | None = None
        self._recovery_note: str | None = None
        self._restart_from_run: str | None = None
        self._restart_branch_id: str | None = None
        self._state_root_run_id: str | None = None
        self._branch_id: str = "root"
        self._last_local_run_id: str | None = None
        self._resume_note: str | None = None

    # ---------------------------------------------------------------- properties
    @property
    def platform_run_id(self) -> str | None:
        return self._platform_run_id

    @property
    def local_run_id(self) -> str | None:
        return self._local_run_id

    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def backend(self) -> OfficialBackend | None:
        return self._backend

    @property
    def model(self) -> OfficialModel | None:
        return self._model

    @property
    def driver(self) -> EffGenLoopDriver | None:
        return self._driver

    @property
    def store(self) -> StateStore | None:
        return self._store

    @property
    def behaviors(self) -> OfficialBehaviors:
        return self.app_config.official.behaviors

    @property
    def context_protector(self) -> ContextProtector | None:
        """The G-wave necessary-context protector, or None when the switch is off."""

        driver = self._driver
        hook = None if driver is None else getattr(driver, "final_send_hook", None)
        return getattr(hook, "__self__", None) if hook is not None else None

    @property
    def control(self) -> ControlCoordinator | None:
        """The task's root control state when recovery wiring is enabled."""

        return self._control

    @property
    def recovery(self) -> OfficialRecoveryRuntime | None:
        """The task's bounded recovery runtime when recovery wiring is enabled."""

        return self._recovery

    @property
    def branch_id(self) -> str:
        """The ledger branch this task writes (``root`` unless restarted)."""

        return self._branch_id

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def terminal_step(self) -> DriverStep | None:
        return self._terminal_step

    # ---------------------------------------------------------------- SDK hooks
    def reset(self, request: Mapping[str, Any]) -> None:
        """Start an empty task container; no task or observation is assumed."""

        if not isinstance(request, Mapping):
            raise ValueError("reset request must be a mapping")
        platform_run_id = request.get("run_id")
        if not isinstance(platform_run_id, str) or not platform_run_id.strip():
            raise ValueError("reset request has no run_id")
        self._shutdown("reset")
        self._platform_run_id = platform_run_id
        local_run_id = _safe_local_run_id(platform_run_id)
        self._restart_from_run = None
        self._restart_branch_id = None
        if self.behaviors.recovery and self._last_local_run_id != local_run_id:
            restart = self._detect_restart(local_run_id)
            if restart is not None:
                root_run_id, restarted_run_id, restart_index = restart
                self._restart_from_run = root_run_id
                self._restart_branch_id = f"restart-{restart_index}"
                local_run_id = restarted_run_id
        self._local_run_id = local_run_id
        self._last_local_run_id = local_run_id
        reset_deadline = request.get("execution_deadline")
        self._reset_deadline = (
            float(reset_deadline)
            if isinstance(reset_deadline, (int, float)) and not isinstance(reset_deadline, bool)
            else None
        )
        self._session = None
        self._backend = None
        self._model = None
        self._driver = None
        self._store = None
        self._verification_plan = None
        self._experience = None
        self._pending_action = None
        self._pending_observation = None
        self._terminal_step = None
        self._last_stop_action = None
        self._finished = False
        self._step_index = 0
        self._closed = False
        self._clear_split()
        self._request_limit_detail = None
        self._control = None
        self._recovery = None
        self._effect_policy = None
        self._prepared_effect = None
        self._awaiting_reobserve = None
        self._reobserve_start = False
        self._deferred_feedback = None
        self._start_note = None
        self._recovery_note = None
        self._resume_note = None
        self._state_root_run_id = None
        self._branch_id = "root"

    def _detect_restart(
        self, local_run_id: str
    ) -> tuple[str, str, int] | None:
        """Return (root run, restarted run, index) when a complete snapshot exists."""

        state_path = self.runtime_dir / "runs" / local_run_id / "state" / "state.json"
        if not state_path.is_file():
            return None
        for index in range(1, 100):
            candidate = f"{local_run_id}-rs{index}"
            if len(candidate) > 64:
                return None
            run_dir = self.runtime_dir / "runs" / candidate
            if (run_dir / "events.jsonl").exists() or (
                run_dir / "state" / "state.json"
            ).exists():
                continue
            return local_run_id, candidate, index
        return None

    def step(self, request: Mapping[str, Any]) -> dict[str, str]:
        """Advance one platform step and return exactly one action string."""

        if self._closed:
            raise RuntimeError("OfficialAgent is closed; reset is required")
        try:
            return self._step(request)
        except DeadlineExceeded as exc:
            return self._stop_after_error("deadline", exc)
        except Exception as exc:  # noqa: BLE001 - return a visible stop, log the type
            return self._stop_after_error("step", exc)

    def close(self) -> None:
        """Release this task's local resources; repeated calls are safe."""

        self._shutdown("close")

    # ------------------------------------------------------------------ step flow
    def _step(self, request: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(request, Mapping):
            raise ValueError("step request must be a mapping")
        if self._platform_run_id is None:
            raise ValueError("reset is required before step")
        run_id = request.get("run_id")
        if run_id != self._platform_run_id:
            raise ValueError("run_id mismatch; reset is required")
        self._step_index += 1
        if self._finished:
            if self._last_stop_action is not None:
                return dict(self._last_stop_action)
            return {"action": "stop [task already ended]"}
        if self._split_active:
            return self._continue_split(request)
        if self._session is None:
            return self._start_task(request)
        if self._awaiting_reobserve is not None:
            return self._continue_reobserve(request)
        return self._continue_task(request)

    def _start_task(self, request: Mapping[str, Any]) -> dict[str, str]:
        # The SDK's own worker also falls back to 300 seconds when a request
        # omits a deadline; mirror that only as a compatibility fallback.
        fallback = self._reset_deadline
        if fallback is None:
            fallback = self.clock.now() + 300
        deadline = _deadline_value(request, fallback)
        if self.clock.now() >= deadline:
            action = self._stop_action("task deadline reached before local work began")
            self._finished = True
            self._last_stop_action = action
            return dict(action)
        task = request.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("first step has no complete task text")
        assert self._local_run_id is not None
        session = Session(
            self._local_run_id,
            runtime_dir=self.runtime_dir,
            clock=self.clock,
            budget=Budget(
                limits=self.app_config.budget_limits(),
                deadline=Deadline(deadline, clock=self.clock),
                clock=self.clock,
            ),
            page_id="official-page",
        )
        self._session = session
        session.start_task(task)
        behaviors = self.app_config.official.behaviors
        recovery_on = behaviors.recovery

        # G wave: the approved text experience is selected from the configured
        # allowlist only. Off (the default) opens no file and yields empty text.
        # The selection is recorded by the existing behaviors event and the run
        # summary; no new event type or tool path is introduced.
        self._experience = self._load_experience(task)

        # R wave control layer. It is created before the ledger so a restarted
        # branch's control events land in this run's log, and the dispatch gate is
        # installed before any action can be returned.
        if recovery_on:
            source = self._control_source or FileControlSource(
                session.evidence.run_dir / "control.jsonl"
            )
            self._control_source = source
            coordinator = ControlCoordinator(
                self._local_run_id,
                clock=self.clock,
                source=source,
                wait_timeout_seconds=self._control_wait_timeout,
            )
            coordinator.bind_session(session)
            coordinator.start()
            session.set_action_gate(coordinator.gate)
            self._control = coordinator

        store: StateStore | None = None
        plan: GoalDeclarationRuntime | None = None
        restart_load = None
        self._state_root_run_id = self._local_run_id
        self._branch_id = "root"
        if recovery_on and self._restart_from_run is not None:
            self._state_root_run_id = self._restart_from_run
            self._branch_id = self._restart_branch_id or "restart-1"
            restart_load = load_restart_snapshot(
                self.runtime_dir,
                self._restart_from_run,
                branch_id=self._branch_id,
                new_run_id=self._local_run_id,
                clock=self.clock,
                control_log=None if self._control is None else self._control.event_log,
            )
            store = restart_load.store
            store.attach_event_log(session.logger)
        elif behaviors.verification or behaviors.state_tools or recovery_on:
            store = StateStore(
                self.runtime_dir,
                self._local_run_id,
                clock=self.clock,
                event_log=session.logger,
            )
            store.load_or_initialize(task)
            store.attach_event_log(session.logger)
        if store is not None:
            branch = self._branch_id
            session.logger.set_state_context(
                lambda: {
                    "state_revision": store.state.revision,
                    "branch_id": branch,
                    "plan_version": store.state.plan_version,
                }
            )
            plan = GoalDeclarationRuntime(store, session, branch=branch)
        self._store = store
        self._verification_plan = plan
        if recovery_on and store is not None:
            self._recovery = OfficialRecoveryRuntime(
                store,
                session.logger,
                branch=self._branch_id,
                settings=self.app_config.recovery,
                clock=self.clock,
                deadline_epoch=(
                    None if session.deadline is None else session.deadline.epoch_seconds
                ),
            )
            self._effect_policy = EffectPolicy()

        backend = OfficialBackend(
            self._local_run_id,
            session=session,
            evidence=session.evidence,
            page_id="official-page",
            browser_api=self.browser_api,
            download_artifacts=self.download_artifacts,
            deadline_check=session.check_deadline,
            text_mode=self.text_mode,
        )
        self._backend = backend
        initial = backend.from_request(request)
        session.ingest_observation(initial)
        self._pending_observation = initial
        model = OfficialModel(
            self.app_config.model,
            model_api=self.model_api,
        )
        model.set_request_limits(
            self.app_config.budget_limits(),
            self._mandatory_prompt_messages,
            self._report_request_limit,
        )
        model.set_call_sink(self._record_model_call)
        self._model = model

        # G wave: necessary-context protection and the final input-budget check.
        # The protector is built only when ``context.enabled`` is on; a disabled
        # run has no protector, appends nothing and installs no send hook.
        context_protector: ContextProtector | None = None
        if self.app_config.context.enabled:
            context_protector = ContextProtector(
                session,
                store,
                settings=self.app_config.context,
                counter=self.app_config.token_counter(),
                limits=self.app_config.budget_limits(),
                branch=self._branch_id,
            )

        extra_tools: list[BaseTool] = []
        settings: LoopSettings | None = None
        completion_gate = None
        if behaviors.state_tools and store is not None and plan is not None:
            extra_tools.extend(
                build_official_state_tools(
                    StateToolRuntime(store, branch=self._branch_id), plan
                )
            )
        if behaviors.verification and store is not None and plan is not None:
            # The completion checker reads the current normal observation and
            # already-saved evidence; it never navigates or captures anything.
            checker = CompletionChecker(GoalVerifier(store, logger=session.logger))
            base_gate = make_finish_gate(checker, plan.requests)
            completion_gate = (
                self._recovery_completion_gate(base_gate)
                if recovery_on and self._recovery is not None
                else base_gate
            )
            extra_tools.extend(
                build_completion_tools(CompletionToolRuntime(checker, plan.requests))
            )
            settings = LoopSettings(
                external_tools=frozenset(set(WEB_TOOL_NAMES) | {"request_finish"})
            )
        driver = create_driver(
            session,
            model,
            app_config=self.app_config,
            settings=settings,
            extra_tools=extra_tools,
            artifact_loader=backend.download_artifact,
            completion_gate=completion_gate,
        )
        self._driver = driver
        if context_protector is not None:
            # Installed on the driver the loop already uses, so the note reaches
            # the prompt being assembled and the check runs before that exact
            # prompt is sent. No create_driver signature change is needed.
            driver.context_note_provider = context_protector.turn_note
            driver.final_send_hook = context_protector.check_final_send
        if restart_load is not None:
            self._start_note = self._restart_note(restart_load)
            self._log_recovery_event(
                "recovery",
                {
                    "stage": "restart",
                    "root_run_id": restart_load.root_run_id,
                    "branch_id": restart_load.branch_id,
                    "new_run_id": restart_load.new_run_id,
                    "loaded_revision": restart_load.loaded_revision,
                    "plan_version": restart_load.plan_version,
                    "branches": dict(restart_load.branches),
                    "pending_effect_ids": list(restart_load.pending_effect_ids),
                    "generator_stack_restored": False,
                    "replayed_actions": [],
                    "note": self._start_note,
                },
            )
        # G wave: combine the restart note (if any) with the approved experience
        # text into the one program-side note the existing start path reads.
        self._start_note = self._start_note_text()
        session.set_runtime_summary(
            self._runtime_summary_payload(
                tools=[tool.name for tool in extra_tools],
                completion_gate=completion_gate is not None,
                restart_load=restart_load,
            )
        )
        try:
            session.logger.emit(
                "behaviors", result=session.result_summary()["runtime"]
            )
        except Exception:  # noqa: BLE001 - recording a switch cannot fail the run
            pass
        session.check_deadline("before loop")
        if self._control is not None:
            # A takeover visible before the loop opens pauses before any model
            # turn or page action.
            signal = self._control.detect(initial)
            if signal is not None:
                return self._handoff(
                    signal=signal,
                    action=None,
                    feedback=None,
                    feedback_observation=None,
                    start_pending=True,
                )
        self._refresh_runtime_summary()
        return self._consume_step(driver.start(note=self._start_note))

    def _continue_task(self, request: Mapping[str, Any]) -> dict[str, str]:
        session = self._session
        backend = self._backend
        driver = self._driver
        action = self._pending_action
        if session is None or backend is None or driver is None:
            raise RuntimeError("official task container is incomplete")
        session.check_deadline("official step")
        if action is None:
            raise RuntimeError("no action is awaiting platform feedback")
        observation = backend.from_request(
            request,
            previous_action=action,
            previous_observation=self._pending_observation,
        )
        self._pending_observation = observation
        if self._control is None:
            return self._consume_step(driver.accept_observation(observation))
        return self._continue_controlled(observation)

    def _continue_controlled(self, observation: Observation) -> dict[str, str]:
        """One platform step with the root control state and recovery decisions.

        The executed action's result is recorded truthfully first; the generator
        is advanced only after the control layer allows it. A detected takeover or
        an explicit user pause hands the browser to the user before the generator
        advances, so no model call or page action runs during the wait.
        """

        session = self._session
        coordinator = self._control
        driver = self._driver
        action = self._pending_action
        if session is None or coordinator is None or driver is None or action is None:
            raise RuntimeError("controlled official step has no session/driver/action")
        feedback = observation.feedback
        if feedback is None:
            raise RuntimeError("controlled official step has no action feedback")
        session.ingest_observation(observation)
        self._confirm_effect(action, observation)
        coordinator.note_action_finished(feedback.status.value)

        signal = coordinator.detect(observation)
        note: str | None = None
        terminal: str | None = None
        if signal is None and self._recovery is not None:
            outcome = self._recovery.after_feedback(action, observation)
            note = outcome.text or None
            terminal = outcome.terminal_reason
            if outcome.handoff_reason is not None:
                signal = TakeoverSignal(
                    kind="authentication",
                    reason=outcome.handoff_reason,
                    headings=(),
                    matched_copy=(feedback.message,),
                    matched_controls=(),
                )
        command = coordinator.poll_external_command()
        if command == "cancel":
            return self._cancel_after_handoff("user_cancel")
        if terminal is not None:
            self._recovery_note = note
            return self._honest_stop(terminal, note=note)
        if signal is not None or command == "pause":
            return self._handoff(
                signal=signal,
                action=action,
                feedback=feedback,
                feedback_observation=observation,
                note=note,
            )
        # The finish call inside this advance writes result.json; record the
        # reconciled control timing before it, not after.
        self._refresh_runtime_summary()
        return self._consume_step(
            driver.accept_observation(observation, ingest=False, note=note)
        )

    # --------------------------------------------------- takeover / pause flow
    def _handoff(
        self,
        *,
        signal: TakeoverSignal | None,
        action: WebAction | None,
        feedback: Any | None,
        feedback_observation: Observation | None,
        note: str | None = None,
        start_pending: bool = False,
    ) -> dict[str, str]:
        """Close the dispatch gate, hand the browser to the user, wait, re-observe.

        Nothing is advanced between the handoff and the explicit resume: no model
        call, no page action, no generator turn. The executed action's feedback was
        already recorded truthfully when present.
        """

        session = self._session
        coordinator = self._control
        if session is None or coordinator is None:
            raise RuntimeError("handoff needs the control layer")
        reason = "user_pause" if signal is None else signal.reason
        source = "user" if signal is None else "auto"
        details = {} if signal is None else signal.to_dict()
        coordinator.request_pause(reason, source=source, details=details)
        coordinator.begin_drain(
            executed_action_id=None if action is None else action.action_id,
            feedback_status=None if feedback is None else feedback.status.value,
            note=(
                "the executed action's feedback was recorded; no model turn or page "
                "action runs while the user owns the browser"
            ),
        )
        self._deferred_feedback = (
            (action, feedback_observation)
            if action is not None and feedback_observation is not None
            else None
        )
        if session.proposed_action is not None:
            coordinator.discard_proposal(
                session.proposed_action, reason=f"paused_before_dispatch:{reason}"
            )
            session.drop_proposal("paused")
        coordinator.begin_user_control()
        self._log_recovery_event(
            "recovery",
            {
                "stage": "handoff",
                "waiting_reason": reason,
                "pause_source": source,
                "signal": details,
                "waiting_for": "explicit_user_resume",
                "control_state": coordinator.state.value,
                "control_epoch": coordinator.control_epoch,
                "page_epoch": coordinator.page_epoch,
                "action_id": None if action is None else action.action_id,
                "executed_action_feedback_recorded": feedback_observation is not None,
                "old_element_numbers_valid": False,
            },
        )
        command = self._wait_for_signal()
        if command == "cancel":
            return self._cancel_after_handoff("user_cancel")
        if command != "resume":
            return self._honest_stop(
                "no explicit user resume arrived; the task stays handed off",
            )
        coordinator.check_platform_deadline("resume_after_handoff")
        coordinator.begin_refresh()
        reobserve_action, command_text = self._propose_reobserve()
        self._awaiting_reobserve = reobserve_action
        self._reobserve_start = start_pending
        self._resume_note = note
        self._log_recovery_event(
            "recovery",
            {
                "stage": "reobserve",
                "reason": "fresh_observation_after_handoff",
                "action_id": reobserve_action.action_id,
                "url": None if session.observation is None else session.observation.url,
                "control_state": coordinator.state.value,
                "control_epoch": coordinator.control_epoch,
                "page_epoch": coordinator.page_epoch,
                "old_element_numbers_valid": False,
            },
        )
        return {"action": command_text}

    def _wait_for_signal(self) -> str:
        coordinator = self._control
        if coordinator is None:
            return "cancel"
        try:
            return coordinator.wait_for_user_signal(
                timeout_seconds=coordinator.wait_timeout_seconds
            )
        except ControlWaitTimeout as exc:
            if self._session is not None:
                try:
                    self._session.log_exception("control_wait", exc)
                except Exception:  # noqa: BLE001
                    pass
            coordinator.block("no_explicit_resume")
            return "timeout"
        except Exception as exc:  # noqa: BLE001 - a failed wait is not a resume
            if self._session is not None:
                try:
                    self._session.log_exception("control_wait", exc)
                except Exception:  # noqa: BLE001
                    pass
            coordinator.block("control_wait_failed")
            return "timeout"

    def _propose_reobserve(self) -> tuple[WebAction, str]:
        """Dispatch one same-URL navigation so the page is captured again."""

        session = self._session
        backend = self._backend
        if session is None or backend is None or session.observation is None:
            raise RuntimeError("re-observe needs a session, backend and observation")
        observation = session.observation
        action = WebAction(
            action_id=new_action_id(),
            run_id=observation.run_id,
            page_id=observation.page_id,
            observation_id=observation.observation_id,
            op=ActionOp.GOTO,
            args={"url": observation.url},
        )
        session.propose(action)
        dispatched = session.consume(action.action_id)
        return dispatched, backend.serialize_action(dispatched)

    def _continue_reobserve(self, request: Mapping[str, Any]) -> dict[str, str]:
        session = self._session
        backend = self._backend
        driver = self._driver
        coordinator = self._control
        action = self._awaiting_reobserve
        if (
            session is None
            or backend is None
            or driver is None
            or coordinator is None
            or action is None
        ):
            raise RuntimeError("re-observe state is incomplete")
        observation = backend.from_request(
            request,
            previous_action=action,
            previous_observation=self._pending_observation,
        )
        self._pending_observation = observation
        session.ingest_observation(observation)
        coordinator.note_action_finished(observation.feedback.status.value)
        coordinator.finish_refresh(observation=observation)
        self._log_recovery_event(
            "recovery",
            {
                "stage": "resumed",
                "resume_signal": "explicit_user_resume",
                "observation_id": observation.observation_id,
                "url": observation.url,
                "control_state": coordinator.state.value,
                "control_epoch": coordinator.control_epoch,
                "page_epoch": coordinator.page_epoch,
                "old_element_numbers_valid": False,
                "timing": coordinator.timing.snapshot().to_dict(),
            },
        )
        start_pending = self._reobserve_start
        deferred = self._deferred_feedback
        note = self._resume_note
        self._awaiting_reobserve = None
        self._reobserve_start = False
        self._deferred_feedback = None
        self._resume_note = None
        self._refresh_runtime_summary()
        if start_pending:
            step = driver.start(note=self._start_note)
        else:
            if deferred is None:
                raise RuntimeError("handoff resume lost the executed action's feedback")
            step = driver.resume_after_takeover(
                deferred[1], observation, note=note
            )
        return self._consume_step(step)

    def _cancel_after_handoff(self, reason: str) -> dict[str, str]:
        coordinator = self._control
        driver = self._driver
        if coordinator is not None:
            coordinator.cancel(reason)
        self._refresh_runtime_summary()
        if driver is not None:
            return self._terminal_response(driver.cancel(reason))
        return self._stop_after_error("control", RuntimeError(reason))

    def _honest_stop(self, reason: str, *, note: str | None = None) -> dict[str, str]:
        """End the run without claiming completion when it cannot continue."""

        coordinator = self._control
        driver = self._driver
        if coordinator is not None and coordinator.state not in (
            ControlState.CANCELLED,
            ControlState.BLOCKED,
        ):
            coordinator.block(reason)
        if driver is None:
            action = self._stop_action(f"task ended without completion: {reason}")
            self._finished = True
            self._last_stop_action = action
            self._finalize_run()
            return dict(action)
        message = (
            f"task ended without claiming completion: {reason}. "
            "Partial progress is not complete."
        )
        if note:
            message = message + " " + note
        self._refresh_runtime_summary()
        step = driver.finish_incomplete(
            message,
            claim="the task stopped without completing its requirements",
            feedback_ingested=True,
        )
        return self._terminal_response(step)

    def _recovery_completion_gate(self, base_gate: Any) -> Any:
        """Run the completion check, then withdraw claims the refusal invalidated."""

        def gate(_step: object = None) -> Any:
            decision = base_gate(None)
            if (
                not getattr(decision, "accepted", False)
                and self._recovery is not None
            ):
                note = self._recovery.after_completion_refusal(
                    decision, None if self._session is None else self._session.observation
                )
                if note.text:
                    self._recovery_note = note.text
                    decision = replace(
                        decision, reason=f"{decision.reason} {note.text}"
                    )
            return decision

        return gate

    def _load_experience(self, task: str) -> ExperienceSelection:
        """Select approved local experience for this task's prompt.

        The loader reads only the configured allowlist and the local entry files
        that allowlist names; no observation, URL or page text is an input. A
        relative allowlist path is resolved against the configuration file's
        directory, falling back to the project root for an in-memory config.
        """

        settings = self.app_config.experience
        source_path = self.app_config.source_path
        base_dir = (
            Path(source_path).resolve().parent if source_path else _PROJECT_ROOT
        )
        loader = ExperienceLoader(
            enabled=settings.enabled,
            allowlist=settings.allowlist,
            base_dir=base_dir,
            max_entries=settings.max_entries,
            max_chars=settings.max_chars,
        )
        return loader.load(
            task=task,
            profile=self.app_config.profile,
            backend=self.app_config.backend,
        )

    def _start_note_text(self) -> str | None:
        """The program-side note prefixed to the task when the loop starts.

        It carries the restart note (if any) and then approved experience text.
        Both are prompt text only: nothing here registers a tool, adds a schema
        or changes action dispatch.
        """

        parts: list[str] = []
        if self._start_note is not None and self._start_note.strip():
            parts.append(self._start_note.strip())
        if self._experience is not None and self._experience.text:
            parts.append(self._experience.text)
        return "\n\n".join(parts) if parts else None

    def _restart_note(self, restart_load: Any) -> str:
        pending = ", ".join(restart_load.pending_effect_ids) or "none"
        task = self._session.task if self._session is not None else ""
        return (
            "Restart note: the previous process was interrupted. The task ledger "
            f"was loaded from a complete snapshot (root run {restart_load.root_run_id}, "
            f"branch {restart_load.branch_id}, run {restart_load.new_run_id}, "
            f"revision {restart_load.loaded_revision}). The Python generator stack "
            "was NOT restored and no page action was replayed. Unresolved side "
            f"effects: {pending}. Before repeating any write, verify its current "
            "visible result on the page; do not resend the same write blindly. "
            f"Task: {task}"
        )

    def _runtime_summary_payload(
        self,
        *,
        tools: Sequence[str],
        completion_gate: bool,
        restart_load: Any = None,
    ) -> dict[str, Any]:
        behaviors = self.app_config.official.behaviors
        summary: dict[str, Any] = {
            "behaviors": {
                "verification": behaviors.verification,
                "state_tools": behaviors.state_tools,
                "recovery": behaviors.recovery,
            },
            "tools": list(tools),
            "completion_gate": completion_gate,
            "deadline_margin_seconds": self.app_config.budget.deadline_margin_seconds,
        }
        if self._experience is not None:
            summary["experience"] = self._experience.summary()
        if self._control is not None:
            source = self._control.source
            summary["control"] = {
                "gate_installed": True,
                "source": (
                    "file" if isinstance(source, FileControlSource)
                    else "none" if source is None else type(source).__name__
                ),
                "wait_timeout_seconds": self._control.wait_timeout_seconds,
                "timing": self._control.timing.snapshot().to_dict(),
            }
        if self._recovery is not None:
            summary["recovery"] = self._recovery.summary()
        if restart_load is not None:
            summary["restart"] = {
                "root_run_id": restart_load.root_run_id,
                "branch_id": restart_load.branch_id,
                "new_run_id": restart_load.new_run_id,
                "loaded_revision": restart_load.loaded_revision,
                "pending_effect_ids": list(restart_load.pending_effect_ids),
                "generator_stack_restored": False,
                "replayed_actions": [],
            }
        # G wave: record the project switch next to the native value it shares, so
        # a run states whether necessary-context protection was on and which token
        # ceiling the native budget/compaction path was given.
        context = self.app_config.context
        summary["context"] = {
            "enabled": context.enabled,
            "max_input_tokens": context.max_input_tokens,
            "max_request_bytes": context.max_request_bytes,
            "ledger_max_chars": context.ledger_max_chars,
            "effects_max_chars": context.effects_max_chars,
            "native_context_budget_tokens": int(self.app_config.budget.input_tokens),
        }
        return summary

    def _refresh_runtime_summary(self) -> None:
        session = self._session
        if session is None:
            return
        current = session.result_summary().get("runtime") or {}
        summary = dict(current)
        if self._control is not None:
            control = dict(summary.get("control") or {})
            # The snapshot is the reconciled control record: state, epochs,
            # pause reason, discard count and the final ``finished`` flag.
            control.update(self._control.snapshot().to_dict())
            control["gate_installed"] = True
            control["timing"] = self._control.timing.snapshot().to_dict()
            summary["control"] = control
        if self._recovery is not None:
            summary["recovery"] = self._recovery.summary()
        session.set_runtime_summary(summary)

    def _finish_control(self, status: str) -> None:
        coordinator = self._control
        if coordinator is None:
            return
        try:
            if coordinator.state not in (
                ControlState.CANCELLED,
                ControlState.BLOCKED,
            ):
                coordinator.finish(status)
        except Exception:  # noqa: BLE001 - control bookkeeping cannot fail the run
            pass

    def _log_recovery_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        session = self._session
        if session is None:
            return
        try:
            session.logger.emit(
                event_type,
                page_id=None if session.observation is None else session.observation.page_id,
                url=None if session.observation is None else session.observation.url,
                observation_id=(
                    None
                    if session.observation is None
                    else session.observation.observation_id
                ),
                result=dict(payload),
            )
        except Exception:  # noqa: BLE001 - recording cannot change the decision
            pass

    # ------------------------------------------------------------ pending effects
    def _prepare_effect(self, action: WebAction) -> None:
        """Save a declared side effect before the action reaches the platform."""

        session = self._session
        store = self._store
        policy = self._effect_policy
        if session is None or store is None or policy is None:
            return
        observation = session.observation
        if observation is None:
            return
        intent = policy.classify(action, observation)
        if intent is None:
            return
        effect = store.prepare_effect(
            action_id=action.action_id,
            target=intent.target,
            object_id=intent.object_id,
            goal_id=None,
            branch_id=self._branch_id,
            observation_id=observation.observation_id,
            note=intent.note,
        )
        self._prepared_effect = (intent, effect)

    def _confirm_effect(self, action: WebAction, observation: Observation) -> None:
        prepared = self._prepared_effect
        self._prepared_effect = None
        store = self._store
        if prepared is None or store is None or prepared[1].action_id != action.action_id:
            return
        feedback = observation.feedback
        status = EffectStatus.UNKNOWN
        if feedback is not None:
            if feedback.status is FeedbackStatus.OBSERVED:
                status = EffectStatus.CONFIRMED
            elif feedback.status is FeedbackStatus.ERROR:
                status = EffectStatus.FAILED
        try:
            store.confirm_effect(
                action.action_id,
                status=status,
                observation_id=observation.observation_id,
                note=(
                    "no feedback was returned for this official action"
                    if feedback is None
                    else f"feedback={feedback.status.value}"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - keep the first failure visible
            if self._session is not None:
                try:
                    self._session.log_exception("official_effect_confirm", exc)
                except Exception:  # noqa: BLE001
                    pass

    def _consume_step(self, step: DriverStep) -> dict[str, str]:
        if step.status is DriverStatus.ACTION:
            backend = self._backend
            driver = self._driver
            if backend is None or driver is None or step.action is None:
                raise RuntimeError("action step has no official backend or action")
            if self._should_split(step.action):
                return self._start_split(step.action)
            # Serialize before consuming: an unrepresentable command is a local
            # refusal, not a dispatched platform action with no feedback. A
            # refusal that carries a model-facing correction (an empty typing
            # text, for example) is fed back into the same loop instead of
            # ending the task.
            try:
                command = backend.serialize_action(step.action)
            except ActionRejected as exc:
                self._refresh_runtime_summary()
                return self._consume_step(driver.reject_action(exc))
            if self._store is not None and self._effect_policy is not None:
                try:
                    self._prepare_effect(step.action)
                except Exception as exc:  # noqa: BLE001 - saving failed, do not send
                    return self._stop_after_error("effect_save", exc)
            action = driver.take_action()
            if self._control is not None:
                self._control.note_action_in_flight(action)
            self._pending_action = action
            return {"action": command}
        return self._terminal_response(step)

    # --------------------------------------------------------- split text mode
    def _should_split(self, action: WebAction) -> bool:
        if self.text_mode is not TextMode.DECOMPOSE or action.op is not ActionOp.TYPE:
            return False
        text = action.text or ""
        return "\n" in text or "\r" in text

    def _start_split(self, action: WebAction) -> dict[str, str]:
        """Expand one multi-line type proposal into a sub-action queue."""

        session = self._session
        driver = self._driver
        if session is None or driver is None:
            raise RuntimeError("split type action has no session or driver")
        observation = session.observation
        if observation is None:
            raise RuntimeError("split type action has no current observation")
        plan = plan_type_text(action.text or "", submit=action.submit)
        if not plan:
            raise RuntimeError("split type action produced no sub-action")
        self._split_target = action.target
        # Validate the whole plan against the observation the model saw before
        # anything is dispatched; nothing is sent if the plan is invalid.
        driver.expand_pending(tuple(self._materialize(part, observation) for part in plan))
        self._split_plan = deque(plan)
        self._split_total = len(plan)
        self._split_index = 0
        self._split_in_flight = None
        self._split_active = True
        return self._dispatch_split_next()

    def _materialize(self, part: PlannedSubAction, observation: Observation) -> WebAction:
        args = dict(part.args)
        if part.op is ActionOp.TYPE:
            if self._split_target is None:
                raise RuntimeError("split type sub-action has no target element")
            args["target"] = self._split_target
        return WebAction(
            action_id=new_action_id(),
            run_id=observation.run_id,
            page_id=observation.page_id,
            observation_id=observation.observation_id,
            op=part.op,
            args=args,
        )

    def _dispatch_split_next(self) -> dict[str, str]:
        session = self._session
        backend = self._backend
        driver = self._driver
        if session is None or backend is None or driver is None:
            raise RuntimeError("split type action has no session, backend or driver")
        observation = session.observation
        if observation is None:
            raise RuntimeError("split type action has no current observation")
        part = self._split_plan[0]
        sub_action = self._materialize(part, observation)
        # Serialize before dispatch: a refused command never becomes a platform
        # action the harness is waiting to answer.
        command = backend.serialize_action(sub_action)
        dispatched = driver.dispatch_split(sub_action)
        self._split_plan.popleft()
        self._split_index += 1
        self._split_in_flight = dispatched
        return {"action": command}

    def _continue_split(self, request: Mapping[str, Any]) -> dict[str, str]:
        """Answer the last sub-action's observation and hand out the next one."""

        session = self._session
        backend = self._backend
        driver = self._driver
        in_flight = self._split_in_flight
        if session is None or backend is None or driver is None or in_flight is None:
            raise RuntimeError("split type action state is incomplete")
        observation = backend.from_request(
            request,
            previous_action=in_flight,
            previous_observation=self._pending_observation,
        )
        self._pending_observation = observation
        final = not self._split_plan
        self._log_split_step(observation, in_flight, final=final)
        try:
            step = driver.accept_split_observation(observation, final=final)
        except DeadlineExceeded as exc:
            return self._abort_split("deadline", exc)
        except Exception as exc:  # noqa: BLE001 - reported with the executed prefix
            return self._abort_split("step", exc)
        if final:
            self._clear_split()
            if step is None:
                raise RuntimeError("split type action ended without a driver step")
            return self._consume_step(step)
        try:
            session.check_deadline("split action dispatch")
            return self._dispatch_split_next()
        except DeadlineExceeded as exc:
            return self._abort_split("deadline", exc)
        except Exception as exc:  # noqa: BLE001 - reported with the executed prefix
            return self._abort_split("step", exc)

    def _log_split_step(
        self,
        observation: Observation,
        action: WebAction,
        *,
        final: bool,
    ) -> None:
        session = self._session
        if session is None:
            return
        try:
            session.logger.emit(
                "subaction",
                page_id=observation.page_id,
                url=observation.url,
                observation_id=observation.observation_id,
                action_id=action.action_id,
                action_type=action.op.value,
                result={
                    "index": self._split_index,
                    "total": self._split_total,
                    "remaining": len(self._split_plan),
                    "delivered_to_model": final,
                },
            )
        except Exception:  # noqa: BLE001 - logging cannot change the page state
            pass

    def _abort_split(self, stage: str, exc: BaseException) -> dict[str, str]:
        """Stop with the truthfully executed prefix of a split proposal."""

        session = self._session
        driver = self._driver
        executed = self._split_index
        total = self._split_total
        remaining = len(self._split_plan)
        deadline = stage == "deadline"
        reason = TerminationReason.DEADLINE if deadline else TerminationReason.ERROR
        detail = str(exc) if deadline else f"{type(exc).__name__}: {exc}"
        message = (
            "the split type action stopped before the full text was entered "
            f"({executed} of {total} sub-actions executed, {remaining} not sent); "
            f"{detail}"
        )
        claim = (
            "multi-line text was not fully entered; "
            f"executed {executed} of {total} sub-actions"
        )
        if session is not None:
            if not deadline:
                try:
                    session.log_exception("official_split", exc)
                except Exception:  # noqa: BLE001
                    pass
            try:
                session.drop_proposal("split_aborted")
                if (
                    session.proposed_action is not None
                    or session.in_flight_action is not None
                ):
                    session.finish_incomplete(
                        Finish(claim=claim, message=message, reason=reason)
                    )
                if self._split_in_flight is not None:
                    session.logger.emit(
                        "subaction",
                        page_id=None
                        if session.observation is None
                        else session.observation.page_id,
                        url=None if session.observation is None else session.observation.url,
                        observation_id=None
                        if session.observation is None
                        else session.observation.observation_id,
                        action_id=self._split_in_flight.action_id,
                        action_type=self._split_in_flight.op.value,
                        result={
                            "aborted": True,
                            "reason": "deadline" if deadline else "error",
                            "executed": executed,
                            "total": total,
                            "remaining": remaining,
                            "delivered_to_model": False,
                            "detail": detail,
                        },
                    )
            except Exception:  # noqa: BLE001 - the stop must still be returned
                pass
        self._clear_split()
        if driver is not None:
            self._refresh_runtime_summary()
            step = driver.abort_split(
                status=DriverStatus.DEADLINE if deadline else DriverStatus.ERROR,
                message=message,
                finish_reason=reason,
                claim=claim,
            )
            return self._terminal_response(step)
        self._finished = True
        action = self._stop_action(message)
        self._last_stop_action = action
        self._finalize_run()
        return dict(action)

    def _clear_split(self) -> None:
        self._split_plan = deque()
        self._split_total = 0
        self._split_index = 0
        self._split_in_flight = None
        self._split_target = None
        self._split_active = False

    def _terminal_response(self, step: DriverStep) -> dict[str, str]:
        self._finish_control(step.status.value)
        self._terminal_step = step
        self._finished = True
        if step.status is DriverStatus.COMPLETED:
            message = step.message or "task completed"
        elif step.status is DriverStatus.DEADLINE:
            message = step.message or "task deadline reached before completion"
        elif step.status is DriverStatus.CANCELLED:
            message = step.message or "task cancelled"
        else:
            message = step.message or f"task not completed ({step.status.value})"
        if self._request_limit_detail and step.status in (
            DriverStatus.INCOMPLETE,
            DriverStatus.ERROR,
        ):
            message = f"{self._request_limit_detail}; {message}"
        action = self._stop_action(message)
        self._last_stop_action = action
        self._finalize_run()
        return dict(action)

    # --------------------------------------------------------------- model sink
    def _record_model_call(self, record: OfficialCallRecord) -> None:
        session = self._session
        if session is None:
            return
        session.record_model_call(
            request_id=record.request_id,
            input_tokens=None,
            output_tokens=None,
            token_count_source="platform_usage_unavailable",
            latency_ms=record.latency_ms,
            failure_type=record.failure_type,
            model_name=self.app_config.model.name,
        )

    # ------------------------------------------------------------------ failure
    def _stop_after_error(self, stage: str, exc: BaseException) -> dict[str, str]:
        session = self._session
        if session is not None:
            try:
                session.log_exception(f"official_{stage}", exc)
            except Exception:  # noqa: BLE001 - logging must not replace the stop
                pass
            try:
                if self._driver is not None and self._driver.terminal is None:
                    self._refresh_runtime_summary()
                    self._driver.cancel(f"official_{stage}")
                elif session.state in (SessionState.RUNNING, SessionState.PAUSED):
                    session.cancel(f"official_{stage}")
            except Exception:  # noqa: BLE001
                pass
        self._finished = True
        self._finish_control(str(stage))
        message = (
            "task deadline reached" if stage == "deadline"
            else f"task not completed due to local error ({type(exc).__name__})"
        )
        action = self._stop_action(message)
        self._last_stop_action = action
        self._finalize_run()
        return dict(action)

    def _mandatory_prompt_messages(self) -> list[dict[str, str]] | None:
        """The must-keep prompt view for the final request-limit check."""

        session = self._session
        if session is None or session.task is None or session.observation is None:
            return None
        return build_prompt_messages(
            task=session.task,
            observation=session.observation,
        )

    def _report_request_limit(self, exc: BaseException) -> None:
        """Keep the specific limit visible when EffGen turns it into a failure."""

        limit_name = getattr(exc, "limit_name", "request_limit")
        limit = getattr(exc, "limit", None)
        actual = getattr(exc, "actual", None)
        self._request_limit_detail = (
            f"{limit_name} limit exceeded (actual={actual}, limit={limit})"
        )
        session = self._session
        if session is not None:
            try:
                session.log_exception("official_request_limit", exc)
            except Exception:  # noqa: BLE001 - logging never replaces the refusal
                pass

    def _finalize_run(self) -> None:
        """Write the terminal evidence before the SDK can end this process.

        ``close`` is idempotent; doing it at the terminal step guarantees
        ``result.json`` and ``run_end`` even when the SDK never calls ``close``.
        """

        # Final control/recovery reconciliation is persisted with the result
        # summary: total/human-wait/execution seconds and attempt counters.
        self._refresh_runtime_summary()
        driver = self._driver
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001
                pass
        session = self._session
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass

    def _stop_action(self, message: str) -> dict[str, str]:
        try:
            command = serialize_stop(message, text_mode=self.text_mode)
        except Exception:  # noqa: BLE001 - a scalar fallback must always exist
            command = "stop [task stopped]"
        return {"action": command}

    # --------------------------------------------------------------- lifecycle
    def _shutdown(self, reason: str) -> None:
        driver = self._driver
        if driver is not None:
            try:
                if driver.terminal is None:
                    driver.cancel(reason)
                driver.close()
            except Exception:  # noqa: BLE001
                pass
        backend = self._backend
        if backend is not None:
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                pass
        session = self._session
        if session is not None:
            try:
                if session.state not in (
                    SessionState.FINISHED,
                    SessionState.CANCELLED,
                    SessionState.CLOSED,
                ):
                    try:
                        session.cancel(reason)
                    except SessionStateError:
                        pass
                session.close()
            except Exception:  # noqa: BLE001
                pass
        model = self._model
        if model is not None:
            try:
                model.unload()
            except Exception:  # noqa: BLE001
                pass
        coordinator = self._control
        if coordinator is not None:
            try:
                coordinator.close()
            except Exception:  # noqa: BLE001
                pass
        self._driver = None
        self._backend = None
        self._session = None
        self._model = None
        self._store = None
        self._verification_plan = None
        self._pending_action = None
        self._pending_observation = None
        self._closed = True


__all__ = ["OfficialAgent", "PlannedSubAction", "plan_type_text"]
