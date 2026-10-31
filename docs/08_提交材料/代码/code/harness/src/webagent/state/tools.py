"""EffGen tools for the task ledger: ``state_read``, ``state_update``, ``plan_update``.

They follow the harness's existing :class:`effgen.tools.base_tool.BaseTool`
registration path. The caller passes them to the EffGen loop as extra tools;
this module does not replace the canonical web-tool schema.

Tool success means "the update was recorded at a new revision". It never means
a goal was verified. A model can cite existing evidence ids from
``state_read`` but cannot invent evidence metadata.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)

from .ledger import PlanUpdate, StateUpdate
from .records import (
    EffectStatus,
    EvidenceRef,
    Fact,
    FactKind,
    FactValidity,
    Goal,
    GoalStatus,
    PendingEffect,
    PlanNode,
    PlanNodeStatus,
    StateConflict,
    StateError,
    StateValidationError,
)
from .store import StateStore

STATE_TOOL_NAMES = frozenset({"state_read", "state_update", "plan_update"})

_GOAL_STATUS_VALUES = [status.value for status in GoalStatus]
_FACT_KIND_VALUES = [kind.value for kind in FactKind]
_FACT_VALIDITY_VALUES = [validity.value for validity in FactValidity]


def _parameter(
    name: str,
    description: str,
    *,
    type: ParameterType = ParameterType.STRING,
    required: bool = False,
    default: Any = None,
    enum: list[Any] | None = None,
    items_type: ParameterType | None = None,
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        type=type,
        description=description,
        required=required,
        default=default,
        enum=enum,
        items_type=items_type,
    )


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StateValidationError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise StateValidationError(f"{where} must be a list")
    return list(value)


def _id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateValidationError(f"{where} must be a non-empty string")
    return value


class StateToolRuntime:
    """The ledger and branch one agent's state tools write through."""

    def __init__(self, store: StateStore, *, branch: str = "root") -> None:
        if not isinstance(store, StateStore):
            raise StateValidationError("StateToolRuntime needs a StateStore")
        if not isinstance(branch, str) or not branch.strip():
            raise StateValidationError("StateToolRuntime branch must be non-empty")
        self.store = store
        self.branch = branch

    # ---------------------------------------------------------------- helpers
    def _check_branch(self, payload: Mapping[str, Any]) -> str:
        requested = payload.get("branch", self.branch)
        if not isinstance(requested, str) or requested != self.branch:
            raise StateValidationError(
                f"this tool writes branch {self.branch!r}; it cannot write {requested!r}"
            )
        return self.branch

    def _evidence_index(self) -> dict[str, EvidenceRef]:
        index: dict[str, EvidenceRef] = {}
        for _, ref in self.store.state.evidence_refs():
            index.setdefault(ref.evidence_id, ref)
        return index

    def _evidence_refs(
        self,
        value: Any,
        *,
        existing: Sequence[EvidenceRef],
        where: str,
    ) -> tuple[EvidenceRef, ...]:
        if value is None:
            return tuple(existing)
        known = self._evidence_index()
        refs: list[EvidenceRef] = []
        seen: set[str] = set()
        for item in _list(value, where):
            if isinstance(item, str):
                ref = known.get(item)
                if ref is None:
                    raise StateValidationError(
                        f"{where} cites unknown evidence id {item!r}; state_read lists "
                        "the ids already recorded for this task"
                    )
            elif isinstance(item, EvidenceRef):
                ref = item
            elif isinstance(item, Mapping):
                ref = EvidenceRef.from_dict(item)
            else:
                raise StateValidationError(
                    f"{where} entries must be evidence ids or reference objects"
                )
            if ref.evidence_id in seen:
                raise StateValidationError(f"{where} repeats evidence id {ref.evidence_id!r}")
            seen.add(ref.evidence_id)
            refs.append(ref)
        return tuple(refs)

    def _goal(self, raw: Any, existing: Goal | None, branch: str) -> Goal:
        data = _mapping(raw, "goal")
        goal_id = data.get("id", data.get("goal_id"))
        if goal_id is None and existing is not None:
            goal_id = existing.goal_id
        goal_id = _id(goal_id, "goal.id")
        requirement = data.get("requirement", existing.requirement if existing else None)
        criteria = data.get(
            "acceptance_criteria",
            existing.acceptance_criteria if existing else (),
        )
        depends_on = data.get("depends_on", existing.depends_on if existing else ())
        object_scope = data.get("object_scope", existing.object_scope if existing else ())
        status = data.get("status", existing.status if existing else GoalStatus.PENDING)
        evidence = self._evidence_refs(
            data.get("evidence"),
            existing=existing.evidence if existing else (),
            where=f"goal {goal_id}.evidence",
        )
        completed = data.get(
            "completed_object_ids",
            existing.completed_object_ids if existing else (),
        )
        return Goal(
            goal_id=goal_id,
            requirement=requirement,
            acceptance_criteria=tuple(_list(criteria, f"goal {goal_id}.acceptance_criteria")),
            depends_on=tuple(_list(depends_on, f"goal {goal_id}.depends_on")),
            object_scope=tuple(_list(object_scope, f"goal {goal_id}.object_scope")),
            status=status,
            evidence=evidence,
            branch_id=branch,
            completed_object_ids=tuple(
                _list(completed, f"goal {goal_id}.completed_object_ids")
            ),
        )

    def _fact(self, raw: Any, existing: Fact | None, branch: str) -> Fact:
        data = _mapping(raw, "fact")
        fact_id = data.get("id", data.get("fact_id"))
        if fact_id is None and existing is not None:
            fact_id = existing.fact_id
        fact_id = _id(fact_id, "fact.id")
        object_id = data.get(
            "object", data.get("object_id", existing.object_id if existing else None)
        )
        field = data.get(
            "field", data.get("proposition", existing.field if existing else None)
        )
        value = data.get("value", existing.value if existing else None)
        kind = data.get("kind", existing.kind if existing else FactKind.INFERRED)
        source = data.get("source", existing.source if existing else None)
        validity = data.get("validity", existing.validity if existing else FactValidity.VALID)
        revision = data.get(
            "revision", (existing.revision + 1) if existing else 1
        )
        goal_id = data.get("goal_id", existing.goal_id if existing else None)
        evidence = self._evidence_refs(
            data.get("evidence"),
            existing=existing.evidence if existing else (),
            where=f"fact {fact_id}.evidence",
        )
        return Fact(
            fact_id=fact_id,
            object_id=object_id,
            field=field,
            value=value,
            kind=kind,
            source=source,
            validity=validity,
            revision=revision,
            evidence=evidence,
            branch_id=branch,
            goal_id=goal_id,
        )

    def _plan_node(self, raw: Any, existing: PlanNode | None, branch: str, version: int) -> PlanNode:
        data = _mapping(raw, "plan node")
        node_id = data.get("id", data.get("node_id"))
        if node_id is None and existing is not None:
            node_id = existing.node_id
        node_id = _id(node_id, "plan_node.id")
        goal_id = data.get(
            "goal_id", data.get("goal", existing.goal_id if existing else None)
        )
        return PlanNode(
            node_id=node_id,
            goal_id=_id(goal_id, f"plan_node {node_id}.goal_id"),
            depends_on=tuple(
                _list(
                    data.get("depends_on", existing.depends_on if existing else ()),
                    f"plan_node {node_id}.depends_on",
                )
            ),
            input_fact_ids=tuple(
                _list(
                    data.get(
                        "input_fact_ids",
                        existing.input_fact_ids if existing else (),
                    ),
                    f"plan_node {node_id}.input_fact_ids",
                )
            ),
            object_scope=tuple(
                _list(
                    data.get("object_scope", existing.object_scope if existing else ()),
                    f"plan_node {node_id}.object_scope",
                )
            ),
            resources=data.get("resources", existing.resources if existing else {}),
            status=data.get("status", existing.status if existing else PlanNodeStatus.PENDING),
            plan_version=version,
            branch_id=branch,
        )

    # ------------------------------------------------------------- operations
    def read(self, *, goal_id: str | None = None, include_facts: bool = True) -> str:
        view = self.store.state.view(goal_id=goal_id, include_facts=include_facts)
        return json.dumps(view, ensure_ascii=False, separators=(",", ":"))

    def update(self, payload: Mapping[str, Any]) -> str:
        branch = self._check_branch(payload)
        state = self.store.state
        goals: list[Goal] = []
        for raw in _list(payload.get("goals"), "goals"):
            data = _mapping(raw, "goal")
            gid = data.get("id", data.get("goal_id"))
            existing = state.goals.get(gid) if isinstance(gid, str) else None
            goals.append(self._goal(data, existing, branch))
        facts: list[Fact] = []
        for raw in _list(payload.get("facts"), "facts"):
            data = _mapping(raw, "fact")
            fid = data.get("id", data.get("fact_id"))
            existing = state.facts.get(fid) if isinstance(fid, str) else None
            facts.append(self._fact(data, existing, branch))
        update = StateUpdate(
            base_revision=payload.get("base_revision"),
            plan_version=payload.get("plan_version"),
            source_branch=branch,
            goals=tuple(goals),
            facts=tuple(facts),
            note=str(payload.get("note") or ""),
        )
        recorded = self.store.apply_update(update)
        return _recorded_message(recorded, "state_update")

    def plan_update(self, payload: Mapping[str, Any]) -> str:
        branch = self._check_branch(payload)
        state = self.store.state
        version = state.plan_version + 1
        goals: list[Goal] = []
        for raw in _list(payload.get("goals"), "goals"):
            data = _mapping(raw, "goal")
            gid = data.get("id", data.get("goal_id"))
            existing = state.goals.get(gid) if isinstance(gid, str) else None
            goals.append(self._goal(data, existing, branch))
        nodes: list[PlanNode] = []
        for raw in _list(payload.get("plan"), "plan"):
            data = _mapping(raw, "plan node")
            nid = data.get("id", data.get("node_id"))
            existing = state.plan.get(nid) if isinstance(nid, str) else None
            nodes.append(self._plan_node(data, existing, branch, version))
        removed_goals = tuple(
            _id(item, "remove_goal_ids item")
            for item in _list(payload.get("remove_goal_ids"), "remove_goal_ids")
        )
        removed_nodes = tuple(
            _id(item, "remove_plan_node_ids item")
            for item in _list(payload.get("remove_plan_node_ids"), "remove_plan_node_ids")
        )
        update = PlanUpdate(
            base_revision=payload.get("base_revision"),
            plan_version=payload.get("plan_version"),
            source_branch=branch,
            goals=tuple(goals),
            plan_nodes=tuple(nodes),
            removed_goal_ids=removed_goals,
            removed_plan_node_ids=removed_nodes,
            note=str(payload.get("note") or ""),
        )
        recorded = self.store.apply_plan_update(update)
        return _recorded_message(recorded, "plan_update")


def _recorded_message(state: Any, tool: str) -> str:
    return json.dumps(
        {
            "recorded": True,
            "tool": tool,
            "revision": state.revision,
            "plan_version": state.plan_version,
            "completion_status": state.completion_status(),
            "message": (
                f"{tool} recorded at revision {state.revision}. "
                "Success means the update was saved; it does not verify any goal."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class _StateTool(BaseTool):
    """A synchronous ledger tool; calling it never starts an event loop."""

    async def _execute(self, **kwargs: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError("state tools execute synchronously")

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
        except StateError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=_error_text(exc),
                metadata=_error_metadata(exc),
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


def _error_text(exc: StateError) -> str:
    if isinstance(exc, StateConflict):
        return json.dumps(exc.as_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"{type(exc).__name__}: {exc}"


def _error_metadata(exc: StateError) -> dict[str, Any]:
    metadata: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(exc, StateConflict):
        metadata["current_revision"] = exc.current.revision
        metadata["current_plan_version"] = exc.current.plan_version
    elif getattr(exc, "current", None) is not None:
        metadata["current_revision"] = exc.current.revision
    return metadata


class StateReadTool(_StateTool):
    def __init__(self, runtime: StateToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="state_read",
                description=(
                    "Read the task ledger: requirements, facts, plan, pending side "
                    "effects and current revision/plan_version. Read this before an "
                    "update so the revision you submit is current."
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


class StateUpdateTool(_StateTool):
    def __init__(self, runtime: StateToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="state_update",
                description=(
                    "Record requirement/fact updates. Submit base_revision and "
                    "plan_version from state_read. goals[]: id, requirement, "
                    "acceptance_criteria[], depends_on[], object_scope[], status, "
                    "evidence[] (existing ids), completed_object_ids[]; facts[]: id, "
                    "object, field, value, kind=observed|inferred, source, validity, "
                    "revision, evidence[] (existing ids), goal_id. An observed fact "
                    "needs evidence. This tool never sets a goal or fact to verified; "
                    "success means the update was saved."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    _parameter(
                        "base_revision",
                        "Revision this update is based on (from state_read).",
                        type=ParameterType.INTEGER,
                        required=True,
                    ),
                    _parameter(
                        "plan_version",
                        "Plan version this update is based on (from state_read).",
                        type=ParameterType.INTEGER,
                        required=True,
                    ),
                    _parameter(
                        "branch",
                        "Source branch; the tool runtime's branch is the only accepted value.",
                    ),
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
                    _parameter("note", "Optional local note for this revision."),
                ],
            )
        )
        self._runtime = runtime

    def _run(self, **kwargs: Any) -> str:
        return self._runtime.update(kwargs)


class PlanUpdateTool(_StateTool):
    def __init__(self, runtime: StateToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="plan_update",
                description=(
                    "Propose a plan revision. Submit base_revision and plan_version "
                    "from state_read. plan[] nodes: id, goal_id, depends_on[], "
                    "input_fact_ids[], object_scope[], resources{}, status. goals may "
                    "add requirements or update unfinished ones. remove_goal_ids can "
                    "only remove verified requirements; an unfinished requirement is "
                    "refused and should be marked blocked instead. Success only means "
                    "the new plan version was saved."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    _parameter(
                        "base_revision",
                        "Revision this plan revision is based on.",
                        type=ParameterType.INTEGER,
                        required=True,
                    ),
                    _parameter(
                        "plan_version",
                        "Current plan version this revision replaces.",
                        type=ParameterType.INTEGER,
                        required=True,
                    ),
                    _parameter(
                        "branch",
                        "Source branch; the tool runtime's branch is the only accepted value.",
                    ),
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
                    _parameter("note", "Optional local note for this revision."),
                ],
            )
        )
        self._runtime = runtime

    def _run(self, **kwargs: Any) -> str:
        return self._runtime.plan_update(kwargs)


def build_state_tools(runtime: StateToolRuntime) -> list[BaseTool]:
    """The three ledger tools in a stable registration order."""

    return [
        StateReadTool(runtime),
        StateUpdateTool(runtime),
        PlanUpdateTool(runtime),
    ]


__all__ = [
    "PlanUpdateTool",
    "STATE_TOOL_NAMES",
    "StateReadTool",
    "StateToolRuntime",
    "StateUpdateTool",
    "build_state_tools",
]
