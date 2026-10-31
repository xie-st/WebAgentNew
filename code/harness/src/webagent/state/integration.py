"""D3 wiring: ledger tools, pre-dispatch effects and observed facts.

This module is the single integration point between the phase-1 interactive
loop and the phase-2 :class:`~webagent.state.store.StateStore`:

* :func:`create_stateful_driver` registers ``state_read`` / ``state_update`` /
  ``plan_update`` through the existing ``extra_tools`` parameter of
  :func:`~webagent.effgen_bridge.create_driver`;
* :class:`StatefulLoopRunner` drives the same generator and backend as
  :class:`~webagent.effgen_bridge.BackendLoopRunner`, but saves a pending side
  effect through the driver's ``before_dispatch`` hook and records the action's
  feedback afterwards;
* :class:`EffectPolicy` is the explicit save/submit/comment classification. It
  is a separate declaration table, not a change to the closed ``TOOL_SPECS``
  schema or to :class:`~webagent.protocol.WebAction`;
* :func:`record_observed_object` writes an ``observed`` fact plus the run's
  existing screenshot evidence; the fact alone never verifies a goal.

The local-site object mapping is supplied by the caller as an
``observation_recorder``. Production code therefore does not know L01/L02 URLs,
and tests can use the same wiring against a different site.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from effgen.tools.base_tool import BaseTool

from ..backend import Backend
from ..effgen_bridge import (
    DriverStatus,
    DriverStep,
    EffGenLoopDriver,
    LoopSettings,
    create_driver,
)
from ..protocol import ActionOp, Artifact, FeedbackStatus, Observation, WebAction
from ..session import Session
from .ledger import StateUpdate
from .records import (
    EffectStatus,
    EvidenceRef,
    Fact,
    FactKind,
    Goal,
    GoalStatus,
    PendingEffect,
    StateValidationError,
)
from .store import StateStore
from .tools import StateToolRuntime, build_state_tools


# --------------------------------------------------------------------- effects
class EffectClass(str, Enum):
    """Business meaning of a declared web write."""

    DRAFT_SAVE = "draft_save"
    DRAFT_CLEAR = "draft_clear"
    SUBMIT = "submit"
    COMMENT = "comment"
    PURCHASE = "purchase"


@dataclass(frozen=True)
class EffectRule:
    """One explicit rule in the side-effect declaration table.

    Click rules match the accessible name of the clicked element. ``TYPE`` with
    ``submit_only`` classifies a typing action that presses Enter, because that
    can submit the surrounding form. A rule only fires for the operation it
    names; navigation, scrolling and ordinary typing are not writes.
    """

    effect_class: EffectClass
    op: ActionOp = ActionOp.CLICK
    names: tuple[str, ...] = ()
    submit_only: bool = False

    def matches(self, action: WebAction, observation: Observation) -> bool:
        if action.op is not self.op:
            return False
        if self.op is ActionOp.TYPE:
            return bool(self.submit_only and action.submit)
        if self.op is not ActionOp.CLICK:
            return False
        element = observation.elements.get(action.target or "")
        if element is None:
            return False
        name = " ".join(str(element.name or "").casefold().split())
        return any(pattern in name for pattern in self.names)


# Order is significant: "save draft" is a draft write, not generic "save";
# "send comment" is a comment, not generic "send".
DEFAULT_EFFECT_RULES: tuple[EffectRule, ...] = (
    EffectRule(EffectClass.DRAFT_SAVE, names=("save draft",)),
    EffectRule(EffectClass.DRAFT_CLEAR, names=("cancel draft", "discard draft")),
    EffectRule(
        EffectClass.COMMENT,
        names=("send comment", "post comment", "submit comment", "publish comment"),
    ),
    EffectRule(
        EffectClass.PURCHASE,
        names=("add to cart", "checkout", "place order", "buy now", "purchase"),
    ),
    EffectRule(
        EffectClass.SUBMIT,
        names=("submit", "save changes", "save", "send", "publish", "confirm", "delete"),
    ),
    EffectRule(EffectClass.SUBMIT, op=ActionOp.TYPE, submit_only=True),
)


@dataclass(frozen=True)
class EffectIntent:
    """A side effect the policy asked the runner to save before dispatch."""

    effect_class: EffectClass
    target: str
    object_id: str
    note: str = ""


class EffectPolicy:
    """The explicit write classification used by the stateful runner.

    ``object_resolver`` maps the current observation (and action) to a business
    object id; the default uses the last URL path segment for detail pages and
    the page id otherwise. A resolver should return an id, not infer success.
    """

    def __init__(
        self,
        *,
        rules: Sequence[EffectRule] = DEFAULT_EFFECT_RULES,
        object_resolver: Callable[[Observation, WebAction], str | None] | None = None,
    ) -> None:
        self.rules = tuple(rules)
        self.object_resolver = object_resolver

    def classify(
        self,
        action: WebAction,
        observation: Observation,
    ) -> EffectIntent | None:
        for rule in self.rules:
            if not rule.matches(action, observation):
                continue
            object_id = self._object_id(observation, action)
            element = observation.elements.get(action.target or "")
            if element is not None:
                note = (
                    f"{rule.effect_class.value}: {element.role} "
                    f"{str(element.name or '')!r} on object {object_id}"
                )
            else:
                note = f"{rule.effect_class.value}: {action.op.value} on object {object_id}"
            return EffectIntent(
                effect_class=rule.effect_class,
                target=action.target or action.op.value,
                object_id=object_id,
                note=note,
            )
        return None

    def _object_id(self, observation: Observation, action: WebAction) -> str:
        if self.object_resolver is not None:
            resolved = self.object_resolver(observation, action)
            if isinstance(resolved, str) and resolved.strip():
                return resolved
        parts = [part for part in urlsplit(observation.url).path.split("/") if part]
        if len(parts) >= 2 and parts[-1]:
            return parts[-1]
        return f"page-{observation.page_id}"


# -------------------------------------------------------------------- evidence
def evidence_ref_from_artifact(
    artifact: Artifact,
    *,
    run_id: str,
    branch_id: str = "root",
    evidence_id: str,
    observation_id: str | None = None,
    action_id: str | None = None,
) -> EvidenceRef:
    """Reference an artifact the session already saved.

    The ledger stores identity and metadata only; the bytes remain in the run's
    evidence directory and are re-hashed by ``StateStore`` on commit.
    """

    if not isinstance(artifact, Artifact) or not artifact.available:
        reason = "missing" if not isinstance(artifact, Artifact) else artifact.reason
        raise StateValidationError(
            f"cannot cite unavailable evidence ({reason or 'unknown reason'}); "
            "an observed fact needs a saved artifact"
        )
    if not artifact.reference or not artifact.sha256 or artifact.byte_size is None:
        raise StateValidationError("artifact has no complete reference/hash/size")
    return EvidenceRef(
        evidence_id=evidence_id,
        run_id=run_id,
        branch_id=branch_id,
        reference=artifact.reference,
        sha256=artifact.sha256,
        byte_size=int(artifact.byte_size),
        kind=artifact.kind,
        observation_id=observation_id,
        action_id=action_id,
    )


@dataclass(frozen=True)
class ObservedObject:
    """What an observation recorder recognized on the current page.

    ``fields`` is a small JSON object describing the public values observed. The
    recorder must not use hidden judge truth; it only maps normal observations.
    """

    object_id: str
    fields: Mapping[str, Any] = field(default_factory=dict)


def observed_page_fact_id(object_id: str) -> str:
    return f"fact-{object_id}-page-observed"


def record_observed_object(
    store: StateStore,
    *,
    object_id: str,
    fields: Mapping[str, Any],
    evidence: Sequence[EvidenceRef],
    source: str,
    goal_id: str | None = None,
    branch_id: str = "root",
    base_revision: int | None = None,
    plan_version: int | None = None,
) -> Fact:
    """Save one observed object fact and mark it completed in its goal scope.

    A repeat observation of the same object updates the same fact revision and
    appends the new evidence; ``completed_object_ids`` is a set-like tuple, so a
    second visit never increases the object count. The goal is not verified
    here: only :meth:`StateStore.verify_goal` may do that, with evidence.
    """

    if not isinstance(store, StateStore):
        raise StateValidationError("record_observed_object needs a StateStore")
    if not evidence:
        raise StateValidationError("an observed fact needs at least one evidence reference")
    state = store.state
    fact_id = observed_page_fact_id(object_id)
    existing = state.facts.get(fact_id)
    references: list[EvidenceRef] = []
    if existing is not None:
        if existing.object_id != object_id:
            raise StateValidationError("observed fact id points at another object")
        if existing.kind is not FactKind.OBSERVED:
            raise StateValidationError("an inferred fact is not promoted to observed")
        references.extend(existing.evidence)
        revision = existing.revision + 1
    else:
        revision = 1
    seen = {ref.evidence_id for ref in references}
    for ref in evidence:
        if not isinstance(ref, EvidenceRef):
            raise StateValidationError("observed evidence must be EvidenceRef records")
        if ref.evidence_id not in seen:
            seen.add(ref.evidence_id)
            references.append(ref)

    fact = Fact(
        fact_id=fact_id,
        object_id=object_id,
        field="page_observed",
        value=dict(fields),
        kind=FactKind.OBSERVED,
        source=source,
        revision=revision,
        evidence=tuple(references),
        branch_id=branch_id,
        goal_id=goal_id,
    )

    goals: list[Goal] = []
    goal = state.goals.get(goal_id) if goal_id is not None else None
    if goal is not None and goal.status is not GoalStatus.VERIFIED:
        if object_id in goal.object_scope and object_id not in goal.completed_object_ids:
            next_status = (
                GoalStatus.RUNNING if goal.status is GoalStatus.PENDING else goal.status
            )
            goals.append(
                replace(
                    goal,
                    status=next_status,
                    completed_object_ids=(*goal.completed_object_ids, object_id),
                )
            )

    update = StateUpdate(
        base_revision=state.revision if base_revision is None else base_revision,
        plan_version=state.plan_version if plan_version is None else plan_version,
        source_branch=branch_id,
        goals=tuple(goals),
        facts=(fact,),
        note=f"observation recorded for {object_id}",
    )
    store.apply_update(update)
    return store.state.facts[fact_id]


# --------------------------------------------------------------------- runner
class StatefulLoopRunner:
    """``BackendLoopRunner`` plus ledger checks around every dispatched action.

    The loop is still one EffGen generator and one backend. It differs only at
    two points: before dispatch it saves a pending effect when the explicit
    policy classifies the action as a write, and after the feedback observation
    it records an observed fact/evidence and confirms the effect. A failed
    pending-effect save cancels the run before the backend sees the action.
    """

    def __init__(
        self,
        driver: EffGenLoopDriver,
        backend: Backend,
        store: StateStore,
        *,
        branch: str = "root",
        goal_id: str | None = None,
        effect_policy: EffectPolicy | None = None,
        observation_recorder: (
            Callable[[Observation, WebAction | None], ObservedObject | None] | None
        ) = None,
        on_step: Callable[[DriverStep], None] | None = None,
    ) -> None:
        if not isinstance(store, StateStore):
            raise StateValidationError("StatefulLoopRunner needs a StateStore")
        self.driver = driver
        self.backend = backend
        self.store = store
        self.branch = branch
        self.goal_id = goal_id
        self.effect_policy = effect_policy or EffectPolicy()
        self.observation_recorder = observation_recorder
        self.on_step = on_step
        self.effects_prepared: list[PendingEffect] = []
        self.effects_confirmed: list[PendingEffect] = []
        self.observed_objects: list[str] = []
        self.recorded_fact_ids: list[str] = []
        self.last_error: BaseException | None = None

    def run(self) -> DriverStep:
        session = self.driver.session
        self._record_observation(session.observation, None)
        step = self.driver.start()
        while True:
            if self.on_step is not None:
                self.on_step(step)
            if step.status is not DriverStatus.ACTION:
                return step
            if step.action is None:  # pragma: no cover - DriverStep guarantees it
                raise StateValidationError("action step has no proposed WebAction")

            prepared: tuple[EffectIntent, PendingEffect] | None = None

            def before_dispatch(action: WebAction) -> None:
                nonlocal prepared
                observation = session.observation
                if observation is None:
                    return
                intent = self.effect_policy.classify(action, observation)
                if intent is None:
                    return
                effect = self.store.prepare_effect(
                    action_id=action.action_id,
                    target=intent.target,
                    object_id=intent.object_id,
                    goal_id=self.goal_id,
                    branch_id=self.branch,
                    observation_id=observation.observation_id,
                    note=intent.note,
                )
                prepared = (intent, effect)
                self.effects_prepared.append(effect)

            try:
                action = self.driver.take_action(before_dispatch=before_dispatch)
            except Exception as exc:  # noqa: BLE001 - the ledger refuses the dispatch
                self.last_error = exc
                session.log_exception("ledger_before_dispatch", exc)
                if prepared is not None:
                    try:
                        aborted = self.store.confirm_effect(
                            prepared[1].action_id,
                            status=EffectStatus.FAILED,
                            note="dispatch was aborted after the pending effect was saved",
                        )
                        self.effects_confirmed.append(aborted)
                    except Exception as confirm_exc:  # noqa: BLE001 - keep the first failure
                        self.last_error = confirm_exc
                        session.log_exception("ledger_effect_abort", confirm_exc)
                return self.driver.cancel(
                    f"ledger_dispatch_blocked: {type(exc).__name__}: {exc}"
                )

            result = self.backend.execute(action)
            if prepared is not None:
                confirmed = self.store.confirm_effect(
                    action.action_id,
                    status=_effect_status(result.feedback),
                    observation_id=result.observation.observation_id,
                    note=f"feedback={result.feedback.status.value}",
                )
                self.effects_confirmed.append(confirmed)
            step = self.driver.accept_observation(result.observation)
            self._record_observation(result.observation, action)

    # ----------------------------------------------------------------- private
    def _record_observation(
        self,
        observation: Observation | None,
        action: WebAction | None,
    ) -> None:
        if observation is None or self.observation_recorder is None:
            return
        observed = self.observation_recorder(observation, action)
        if observed is None:
            return
        artifact = observation.artifacts.get("screenshot")
        evidence_id = f"ev-{observation.observation_id}-screenshot"
        ref = evidence_ref_from_artifact(
            artifact,
            run_id=self.driver.session.run_id,
            branch_id=self.branch,
            evidence_id=evidence_id,
            observation_id=observation.observation_id,
            action_id=None if action is None else action.action_id,
        )
        fact = record_observed_object(
            self.store,
            object_id=observed.object_id,
            fields=observed.fields,
            evidence=(ref,),
            source=f"observation:{observation.observation_id}",
            goal_id=self.goal_id,
            branch_id=self.branch,
        )
        self.observed_objects.append(observed.object_id)
        self.recorded_fact_ids.append(fact.fact_id)


def _effect_status(feedback: Any) -> EffectStatus:
    status = getattr(feedback, "status", None)
    if status is FeedbackStatus.OBSERVED:
        return EffectStatus.CONFIRMED
    if status is FeedbackStatus.ERROR:
        return EffectStatus.FAILED
    return EffectStatus.UNKNOWN


# --------------------------------------------------------------- driver entry
def create_stateful_driver(
    session: Session,
    model: Any,
    store: StateStore,
    *,
    branch: str = "root",
    app_config: Any = None,
    settings: LoopSettings | None = None,
    extra_tools: Iterable[BaseTool] = (),
    artifact_loader: Callable[[str], Any] | None = None,
) -> EffGenLoopDriver:
    """Build a normal interactive driver with the ledger tools registered.

    The three state tools go through the existing ``extra_tools`` path; the
    canonical web-tool schema and ``WebAction`` serialization are untouched.
    """

    if not isinstance(store, StateStore):
        raise StateValidationError("create_stateful_driver needs a StateStore")
    tools = list(extra_tools)
    tools.extend(build_state_tools(StateToolRuntime(store, branch=branch)))
    return create_driver(
        session,
        model,
        app_config=app_config,
        settings=settings,
        extra_tools=tools,
        artifact_loader=artifact_loader,
    )


__all__ = [
    "DEFAULT_EFFECT_RULES",
    "EffectClass",
    "EffectIntent",
    "EffectPolicy",
    "EffectRule",
    "ObservedObject",
    "StatefulLoopRunner",
    "create_stateful_driver",
    "evidence_ref_from_artifact",
    "observed_page_fact_id",
    "record_observed_object",
]
