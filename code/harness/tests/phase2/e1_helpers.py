"""Shared fixtures for E1 verification/recovery unit and browser runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from effgen.models.base import GenerationResult

from webagent.backend import Backend
from webagent.backends.playwright import PlaywrightBackend
from webagent.budget import Budget, Deadline
from webagent.evidence import EvidenceStore
from webagent.clock import SystemClock
from webagent.effgen_bridge import (
    DriverStatus,
    DriverStep,
    EffGenLoopDriver,
    LoopSettings,
    create_driver,
)
from webagent.events import read_events
from webagent.models.fake import FakeResponse, ScriptedModel
from webagent.protocol import (
    ArtifactSet,
    Capabilities,
    ElementRef,
    ExecutionResult,
    Feedback,
    FeedbackStatus,
    Observation,
    WebAction,
)
from webagent.recovery import (
    EffectExpectation,
    FailureEvidence,
    RecoveryClass,
    RecoveryController,
    RecoveryLimits,
    RecoveryPlanner,
    UnknownEffectResolver,
)
from webagent.session import Session
from webagent.state import (
    EffectPolicy,
    EvidenceRef,
    Goal,
    GoalStatus,
    ObservedObject,
    PlanNode,
    PlanUpdate,
    StateStore,
    StatefulLoopRunner,
    create_stateful_driver,
    evidence_ref_from_artifact,
    record_observed_object,
)
from webagent.testing import start
from webagent.testing.evaluation import HiddenStateReader
from webagent.verification import (
    GoalVerifier,
    VerificationCondition,
    VerificationRequest,
    VerificationTrigger,
)


def make_observation(
    runtime: Path,
    *,
    run_id: str = "run-a",
    observation_id: str = "obs-1",
    page_id: str = "page-1",
    url: str = "http://127.0.0.1/product/P-TEST",
    ax_text: str = "Product detail",
    values: Sequence[tuple[str, str, str]] = (),
    screenshot: bool = True,
) -> Observation:
    """A valid normal observation with fresh numbered elements."""

    store = EvidenceStore(runtime, run_id)
    elements: dict[str, ElementRef] = {}
    lines = [f"TITLE: fixture", f"URL: {url}", ax_text, "ELEMENTS:"]
    for index, (role, name, value) in enumerate(values, start=1):
        number = str(index)
        elements[number] = ElementRef(
            page_id=page_id,
            observation_id=observation_id,
            number=number,
            role=role,
            name=name,
            localization={"value": value},
        )
        lines.append(
            f"[{number}] role={role} name={json.dumps(name)} value={json.dumps(value)}"
        )
    artifacts = ArtifactSet()
    if screenshot:
        artifact = store.save_bytes(
            "screenshot",
            b"\x89PNG\r\n\x1a\nfixture",
            observation_id=observation_id,
        )
        artifacts = ArtifactSet(screenshot=artifact)
    return Observation(
        run_id=run_id,
        page_id=page_id,
        observation_id=observation_id,
        url=url,
        ax_text="\n".join(lines),
        elements=elements,
        artifacts=artifacts,
        capabilities=Capabilities(screenshot=screenshot),
    )


def evidence_ref(
    runtime: Path,
    observation: Observation,
    *,
    evidence_id: str | None = None,
    run_id: str | None = None,
    branch_id: str = "root",
) -> EvidenceRef:
    artifact = observation.artifacts.screenshot
    if artifact is None or not artifact.available:
        raise AssertionError("observation has no screenshot artifact")
    return EvidenceRef(
        evidence_id=evidence_id or f"ev-{observation.observation_id}",
        run_id=run_id or observation.run_id,
        branch_id=branch_id,
        reference=str(artifact.reference),
        sha256=str(artifact.sha256),
        byte_size=int(artifact.byte_size),
        kind="screenshot",
        observation_id=observation.observation_id,
    )


def make_goal(
    goal_id: str,
    *,
    scope: Sequence[str] = (),
    depends_on: Sequence[str] = (),
    status: str = "running",
    completed: Sequence[str] = (),
    evidence: Iterable[EvidenceRef] = (),
    criteria: Sequence[str] = ("condition verified",),
) -> Goal:
    return Goal(
        goal_id=goal_id,
        requirement=f"requirement {goal_id}",
        acceptance_criteria=tuple(criteria),
        depends_on=tuple(depends_on),
        object_scope=tuple(scope),
        status=status,
        evidence=tuple(evidence),
        branch_id="root",
        completed_object_ids=tuple(completed),
    )


def verified_goal(runtime: Path, goal_id: str, **kwargs: Any) -> Goal:
    """A goal with a real saved screenshot before it is marked verified."""

    observation = make_observation(
        runtime,
        observation_id=f"obs-{goal_id}",
        url="http://127.0.0.1/product/P-TEST",
        values=(value_element("fixture evidence"),),
    )
    return make_goal(
        goal_id,
        status="verified",
        evidence=(evidence_ref(runtime, observation),),
        **kwargs,
    )


def bootstrap_store(
    runtime: Path,
    goals: Sequence[Goal],
    *,
    run_id: str = "run-a",
    task: str = "phase-2 E1 task",
    nodes: Sequence[PlanNode] = (),
) -> StateStore:
    store = StateStore(runtime, run_id)
    store.initialize(task)
    if nodes:
        plan_nodes = tuple(nodes)
    else:
        plan_nodes = tuple(
            PlanNode(
                node_id=f"n-{goal.goal_id}",
                goal_id=goal.goal_id,
                object_scope=goal.object_scope,
                status="pending",
                plan_version=1,
                branch_id="root",
            )
            for goal in goals
        )
    verified = [goal for goal in goals if goal.status.value == "verified"]
    verified = [
        replace(
            goal,
            evidence=(evidence_ref(
                runtime,
                make_observation(
                    runtime,
                    observation_id=f"obs-{goal.goal_id}",
                    url="http://127.0.0.1/product/P-TEST",
                    values=(value_element("fixture evidence"),),
                ),
            ),),
        )
        if not goal.evidence
        else goal
        for goal in verified
    ]
    initial_goals = tuple(
        replace(goal, status="running", evidence=())
        if goal.status.value == "verified"
        else goal
        for goal in goals
    )
    store.apply_plan_update(
        PlanUpdate(
            base_revision=0,
            plan_version=0,
            source_branch="root",
            goals=initial_goals,
            plan_nodes=plan_nodes,
            note="E1 fixture",
        )
    )
    for goal in verified:
        if not goal.evidence:
            raise AssertionError("a fixture verified goal needs evidence")
        state = store.state
        store.verify_goal(
            goal.goal_id,
            evidence=goal.evidence,
            base_revision=state.revision,
            plan_version=state.plan_version,
            source_branch="root",
            completed_object_ids=goal.completed_object_ids,
            note="E1 fixture verified goal",
        )
    return store


def value_element(value: str, *, name: str = "Review draft") -> tuple[str, str, str]:
    return ("textbox", name, value)


def page_with_value(
    runtime: Path,
    value: str,
    *,
    run_id: str = "run-a",
    observation_id: str = "obs-1",
    url: str = "http://127.0.0.1/product/P-TEST",
) -> Observation:
    return make_observation(
        runtime,
        run_id=run_id,
        observation_id=observation_id,
        url=url,
        values=(value_element(value),),
        ax_text=f"Saved profile note: {value}",
    )


def page_map(values: Mapping[str, Any]) -> dict[str, Any]:
    return dict(values)


# --------------------------------------------------------------------- browser
_ELEMENT_LINE = re.compile(
    r'^\[(?P<number>\d+)\] role=(?P<role>\S+) name=(?P<name>"(?:[^"\\]|\\.)*")'
)


def _action(tool: str, arguments: Mapping[str, Any]) -> str:
    return (
        "Thought: next public step.\n"
        f"Action: {tool}\nAction Input: {json.dumps(dict(arguments), ensure_ascii=False)}"
    )


def _latest_elements(prompt: str) -> list[dict[str, str]]:
    tail = prompt.rsplit("INTERACTIVE ELEMENTS", 1)[-1]
    elements: list[dict[str, str]] = []
    for line in tail.splitlines():
        match = _ELEMENT_LINE.match(line.strip())
        if match is None:
            continue
        try:
            name = json.loads(match.group("name"))
        except json.JSONDecodeError:
            continue
        elements.append(
            {
                "number": match.group("number"),
                "role": match.group("role"),
                "name": str(name),
            }
        )
    return elements


def _find(elements: list[dict[str, str]], role: str, name: str) -> dict[str, str]:
    for element in elements:
        if element["role"] == role and element["name"] == name:
            return element
    raise RuntimeError(f"no {role} named {name!r} in current observation")


class PublicPlanModel(ScriptedModel):
    """A scripted model whose plan is derived only from public page data."""

    def _record_generation(self, prompt: str, text: str) -> GenerationResult:
        self._record(prompt, None, None, FakeResponse(text=text), None)
        tokens = self.count_tokens(prompt).count + self.count_tokens(text).count
        return GenerationResult(
            text=text,
            tokens_used=tokens,
            finish_reason="stop",
            model_name=self.model_name,
        )


class L04Model(PublicPlanModel):
    def __init__(
        self,
        *,
        target_name: str,
        target_id: str,
        list_url: str,
        product_url: str,
        note: str,
    ) -> None:
        super().__init__([])
        self.target_name = target_name
        self.target_id = target_id
        self.list_url = list_url
        self.product_url = product_url
        self.note = note
        self.stage = 0

    def generate(self, prompt: str, config: Any = None, **kwargs: Any) -> GenerationResult:
        stage = self.stage
        self.stage += 1
        elements = _latest_elements(prompt)
        if stage == 0:
            text = _action("state_read", {})
        elif stage == 1:
            text = _action("web_goto", {"url": self.list_url})
        elif stage == 2:
            text = _action("web_click", {"target": _find(elements, "link", self.target_name)["number"]})
        elif stage == 3:
            text = _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.note},
            )
        elif stage == 4:
            text = _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        elif stage == 5:
            text = _action("web_goto", {"url": self.product_url})
        elif stage == 6:
            text = (
                "Thought: the lost confirmation was followed by a fresh visible read; "
                "the saved note is visible and no second save was sent.\n"
                f"Final Answer: {self.target_id} saved once; note visible={self.note!r}"
            )
        else:
            raise RuntimeError(f"L04 plan has no stage {stage}")
        return self._record_generation(prompt, text)


class L05Model(PublicPlanModel):
    def __init__(
        self,
        *,
        products: Sequence[Mapping[str, Any]],
        notes: Sequence[str],
        base_url: str,
    ) -> None:
        super().__init__([])
        self.products = [dict(product) for product in products]
        self.notes = list(notes)
        self.base_url = base_url
        self.stage = 0

    def generate(self, prompt: str, config: Any = None, **kwargs: Any) -> GenerationResult:
        stage = self.stage
        self.stage += 1
        elements = _latest_elements(prompt)
        if stage == 0:
            text = _action("state_read", {})
        elif stage == 1:
            text = _action("web_goto", {"url": self._url(0)})
        elif stage == 2:
            text = _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.notes[0]},
            )
        elif stage == 3:
            text = _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        elif stage == 4:
            text = _action("web_goto", {"url": self._url(2)})
        elif stage == 5:
            text = _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.notes[2]},
            )
        elif stage == 6:
            text = _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        elif stage == 7:
            text = _action("web_goto", {"url": self._url(1)})
        elif stage == 8:
            text = _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.notes[1]},
            )
        elif stage == 9:
            text = _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        elif stage == 10:
            text = _action("web_goto", {"url": self._url(1)})
        elif stage == 11:
            text = _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.notes[1]},
            )
        elif stage == 12:
            text = _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        elif stage == 13:
            ids = ", ".join(str(product["id"]) for product in self.products)
            text = (
                "Thought: the first and third independent notes stayed saved; only the "
                "second chain was repaired after its failure.\n"
                f"Final Answer: completed the three distinct products: {ids}"
            )
        else:
            raise RuntimeError(f"L05 plan has no stage {stage}")
        return self._record_generation(prompt, text)

    def _url(self, index: int) -> str:
        return f"{self.base_url}/product/{self.products[index]['id']}"


def _product_recorder():
    def record(observation: Observation, action: WebAction | None) -> ObservedObject | None:
        path = urlsplit(observation.url).path.rstrip("/")
        if not path.startswith("/product/"):
            return None
        object_id = unquote(path.rsplit("/", 1)[-1])
        return ObservedObject(
            object_id=object_id,
            fields={"url": observation.url, "observed_with": "screenshot"},
        )

    return record


class LostFeedbackBackend:
    """Delegates to Playwright, then loses the feedback after a real save."""

    backend_id = "playwright-lost-feedback"

    def __init__(self, inner: PlaywrightBackend, policy: EffectPolicy) -> None:
        self.inner = inner
        self.policy = policy
        self.lost = False
        self.lost_action: WebAction | None = None

    def observe(self) -> Observation:
        return self.inner.observe()

    def execute(self, action: WebAction) -> ExecutionResult:
        current = self.inner.current_observation
        is_save = bool(
            current is not None
            and self.policy.classify(action, current) is not None
            and (current.elements.get(action.target or "") is not None)
            and "save draft" in str(current.elements[action.target or ""].name).casefold()
        )
        result = self.inner.execute(action)
        if not (is_save and not self.lost):
            return result
        self.lost = True
        self.lost_action = action
        observation = result.observation
        feedback = Feedback(
            FeedbackStatus.UNKNOWN,
            action_id=action.action_id,
            observation_id=observation.observation_id,
            message=(
                "the server committed the save but its feedback was lost; "
                "re-observe the visible page before deciding whether to retry"
            ),
            page_changed=True,
        )
        unknown = replace(observation, feedback=feedback)
        return ExecutionResult(action.action_id, feedback, unknown)

    def close(self) -> None:
        self.inner.close()


class FailOnceSaveBackend:
    """Returns an evidenced error before executing one selected save."""

    backend_id = "playwright-fail-once-save"

    def __init__(
        self,
        inner: PlaywrightBackend,
        policy: EffectPolicy,
        *,
        object_id: str,
    ) -> None:
        self.inner = inner
        self.policy = policy
        self.object_id = object_id
        self.failed = False
        self.failed_action: WebAction | None = None

    def observe(self) -> Observation:
        return self.inner.observe()

    def execute(self, action: WebAction) -> ExecutionResult:
        current = self.inner.current_observation
        intent = self.policy.classify(action, current) if current is not None else None
        if (
            not self.failed
            and intent is not None
            and intent.object_id == self.object_id
            and intent.effect_class.value == "draft_save"
        ):
            self.failed = True
            self.failed_action = action
            observation = self.inner.observe()
            feedback = Feedback(
                FeedbackStatus.ERROR,
                action_id=action.action_id,
                observation_id=observation.observation_id,
                message="simulated transient save failure before the write was applied",
                error_type="simulated_transient",
                page_changed=False,
            )
            failed_observation = replace(
                observation,
                previous_action_id=action.action_id,
                feedback=feedback,
            )
            return ExecutionResult(action.action_id, feedback, failed_observation)
        return self.inner.execute(action)

    def close(self) -> None:
        self.inner.close()


@dataclass(frozen=True)
class E1JudgeResult:
    case_id: str
    checks: Mapping[str, bool]
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "checks": dict(self.checks),
            "detail": dict(self.detail),
        }


def judge_l04(
    db_path: Path,
    *,
    target_id: str,
    expected_text: str,
) -> E1JudgeResult:
    hidden = HiddenStateReader(db_path)
    drafts = hidden.draft_texts("product", target_id)
    saves = hidden.activities(kinds=["draft_saved"], item_type="product", item_id=target_id)
    viewed = target_id in hidden.viewed_ids("product")
    checks = {
        "target_viewed": viewed,
        "draft_text": bool(drafts) and drafts[-1] == expected_text,
        "saved_once": len(saves) == 1,
    }
    return E1JudgeResult(
        "L04",
        checks,
        {"drafts": drafts, "save_count": len(saves), "viewed": viewed},
    )


def judge_l05(
    db_path: Path,
    *,
    target_ids: Sequence[str],
    expected_notes: Sequence[str],
) -> E1JudgeResult:
    hidden = HiddenStateReader(db_path)
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {"drafts": {}, "save_counts": {}}
    for target_id, expected in zip(target_ids, expected_notes):
        drafts = hidden.draft_texts("product", target_id)
        saves = hidden.activities(kinds=["draft_saved"], item_type="product", item_id=target_id)
        detail["drafts"][target_id] = drafts
        detail["save_counts"][target_id] = len(saves)
        checks[f"{target_id}:text"] = bool(drafts) and drafts[-1] == expected
        checks[f"{target_id}:saved_once"] = len(saves) == 1
        checks[f"{target_id}:viewed"] = target_id in hidden.viewed_ids("product")
    return E1JudgeResult("L05", checks, detail)


def public_l04_target(prepared: Any) -> tuple[dict[str, Any], str]:
    params = dict(prepared.case.task_params)
    rows, total = prepared.site.store.list_products(
        category=str(params["l04_category"]),
        max_price_cents=int(round(float(params["l04_max_price"]) * 100)),
        min_rating=float(params["l04_min_rating"]),
        in_stock=True,
        page_size=10_000,
    )
    if total != 1:
        raise AssertionError(f"public L04 conditions matched {total} products, expected 1")
    product = dict(rows[0])
    page = int(product["position"]) // int(prepared.site.page_size) + 1
    return product, f"{prepared.base_url}/products?page={page}"


def public_l05_targets(prepared: Any) -> list[dict[str, Any]]:
    params = dict(prepared.case.task_params)
    rows, total = prepared.site.store.list_products(
        category=str(params["l05_category"]),
        max_price_cents=int(round(float(params["l05_max_price"]) * 100)),
        min_rating=float(params["l05_min_rating"]),
        in_stock=True,
        page_size=10_000,
    )
    if total != int(params["l05_count"]):
        raise AssertionError(f"public L05 conditions matched {total} products")
    return [dict(row) for row in sorted(rows, key=lambda row: int(row["position"]))]


def _bootstrap_l04(store: StateStore, prepared: Any, target_id: str, task: str) -> None:
    goal = Goal(
        goal_id="g1",
        requirement=task,
        acceptance_criteria=tuple(str(item) for item in prepared.case.completion["criteria"]),
        object_scope=(target_id,),
        status=GoalStatus.RUNNING,
        branch_id="root",
    )
    node = PlanNode(
        node_id="n1",
        goal_id="g1",
        object_scope=(target_id,),
        status=GoalStatus.PENDING,
        plan_version=1,
        branch_id="root",
    )
    store.apply_plan_update(
        PlanUpdate(
            base_revision=0,
            plan_version=0,
            source_branch="root",
            goals=(goal,),
            plan_nodes=(node,),
            note="E1 L04 bootstrap",
        )
    )


def _bootstrap_l05(
    store: StateStore,
    prepared: Any,
    targets: Sequence[str],
    tasks: str,
) -> None:
    criteria = tuple(str(item) for item in prepared.case.completion["criteria"])
    goals = (
        Goal("g1", tasks, criteria, object_scope=(targets[0],), status=GoalStatus.RUNNING),
        Goal("g2", tasks, criteria, object_scope=(targets[1],), status=GoalStatus.AWAITING_VERIFICATION),
        Goal(
            "g2-check",
            tasks,
            ("the repaired second note is visible",),
            depends_on=("g2",),
            status=GoalStatus.AWAITING_VERIFICATION,
        ),
        Goal("g3", tasks, criteria, object_scope=(targets[2],), status=GoalStatus.RUNNING),
    )
    nodes = tuple(
        PlanNode(
            node_id=f"n-{goal.goal_id}",
            goal_id=goal.goal_id,
            object_scope=goal.object_scope,
            status=GoalStatus.PENDING,
            plan_version=1,
        )
        for goal in goals
    )
    store.apply_plan_update(
        PlanUpdate(
            base_revision=0,
            plan_version=0,
            source_branch="root",
            goals=goals,
            plan_nodes=nodes,
            note="E1 L05 bootstrap",
        )
    )


@dataclass
class L04Run:
    prepared: Any
    target: dict[str, Any]
    task: str
    run_id: str
    store: StateStore
    step: DriverStep | None
    runner: StatefulLoopRunner
    lost_feedback: bool
    resolution: Any
    verification: Any
    judge: E1JudgeResult
    save_count: int
    final_value: str | None
    events: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.prepared.run_dir)


@dataclass
class L05Run:
    prepared: Any
    targets: list[dict[str, Any]]
    notes: list[str]
    task: str
    run_id: str
    store: StateStore
    step: DriverStep | None
    verifications: list[Any]
    recovery: list[Any]
    failed_object: str | None
    executed_objects: list[str]
    judge: E1JudgeResult
    events: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.prepared.run_dir)


def run_l04(
    runtime_dir: str | Path,
    *,
    seed: int = 101,
    run_id: str | None = None,
) -> L04Run:
    runtime_dir = Path(runtime_dir)
    prepared = start("L04", seed, runtime_dir=runtime_dir, run_id=run_id)
    run_name = str(prepared.run_dir.name)
    target, list_url = public_l04_target(prepared)
    target_id = str(target["id"])
    note = str(prepared.case.task_params["l04_draft_text"])
    task = str(prepared.agent_view()["task"])
    store = StateStore(runtime_dir, run_name)
    store.initialize(task)
    _bootstrap_l04(store, prepared, target_id, task)

    clock = SystemClock()
    session = Session(
        run_name,
        runtime_dir=runtime_dir,
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + 180, clock=clock), clock=clock),
    )
    session.logger.set_state_context(
        lambda: {
            "state_revision": store.state.revision,
            "branch_id": "root",
            "plan_version": store.state.plan_version,
        },
        goal_id="g1",
    )
    session.start_task(task)
    backend = PlaywrightBackend(
        run_name,
        evidence=session.evidence,
        headless=True,
        runtime_dir=runtime_dir,
    )
    policy = EffectPolicy()
    lost_backend = LostFeedbackBackend(backend, policy)
    driver = None
    step = None
    resolution = None
    verification = None
    judge = E1JudgeResult("L04", {})
    try:
        backend.start(f"{prepared.base_url}{prepared.case.start_path}")
        session.ingest_observation(backend.observe())
        model = L04Model(
            target_name=str(target["name"]),
            target_id=target_id,
            list_url=list_url,
            product_url=f"{prepared.base_url}/product/{target_id}",
            note=note,
        )
        driver = create_stateful_driver(session, model, store, branch="root")
        runner = StatefulLoopRunner(
            driver,
            lost_backend,
            store,
            branch="root",
            goal_id="g1",
            effect_policy=policy,
            observation_recorder=_product_recorder(),
        )
        step = runner.run()
        effects = [effect for effect in store.pending_effects if effect.status.value == "unknown"]
        if not effects:
            raise AssertionError("L04 did not leave an unknown pending effect")
        resolution = UnknownEffectResolver(store).resolve(
            effects[0],
            probe=lambda: session.observation,
            expectation=EffectExpectation(
                field_role="textbox",
                field_name="Review draft",
                expected_value=note,
            ),
        )
        verifier = GoalVerifier(store, logger=session.logger)
        verification = verifier.verify(
            VerificationRequest(
                goal_id="g1",
                subject_id=target_id,
                conditions=(
                    VerificationCondition(
                        "visible_note",
                        "the final page shows the saved profile note",
                        "visible_value_equals",
                        expected=note,
                        role="textbox",
                        name="Review draft",
                    ),
                    VerificationCondition(
                        "screenshot",
                        "a screenshot was saved for the final observation",
                        "screenshot_saved",
                    ),
                ),
                observation=session.observation,
                evidence_scope=(),
                trigger=VerificationTrigger.ACTION_FEEDBACK,
                branch="root",
                completed_object_ids=(target_id,),
            )
        )
        judge = judge_l04(prepared.db_path, target_id=target_id, expected_text=note)
        save_count = judge.detail["save_count"]
        final_value = None
        if resolution is not None:
            final_value = resolution.visible_value
        events = read_events(session.evidence.events_path)
        state = store.state.to_dict()
    finally:
        if driver is not None:
            driver.close()
        lost_backend.close()
        session.close()
        prepared.stop()
    return L04Run(
        prepared=prepared,
        target=target,
        task=task,
        run_id=run_name,
        store=store,
        step=step,
        runner=runner,
        lost_feedback=lost_backend.lost,
        resolution=resolution,
        verification=verification,
        judge=judge,
        save_count=save_count,
        final_value=final_value,
        events=events,
        state=state,
    )


class _RecoveryAwareRunner:
    def __init__(
        self,
        driver: EffGenLoopDriver,
        backend: Backend,
        store: StateStore,
        *,
        verifier: GoalVerifier,
        notes: Mapping[str, str],
        goal_by_object: Mapping[str, str],
        checks: Mapping[str, tuple[str, ...]],
        planner: RecoveryPlanner,
        controller: RecoveryController,
    ) -> None:
        self.driver = driver
        self.backend = backend
        self.store = store
        self.verifier = verifier
        self.notes = dict(notes)
        self.goal_by_object = dict(goal_by_object)
        self.checks = dict(checks)
        self.policy = EffectPolicy()
        self.planner = planner
        self.controller = controller
        self.sequence = 0
        self.verifications: list[Any] = []
        self.recovery: list[Any] = []
        self.executed_objects: list[str] = []
        self.failed_object: str | None = None

    def run(self) -> DriverStep:
        step = self.driver.start()
        while step.status is DriverStatus.ACTION:
            assert step.action is not None
            action = step.action
            current = self.driver.session.observation
            intent = self.policy.classify(action, current) if current is not None else None
            prepared = None

            def before_dispatch(proposed: WebAction) -> None:
                nonlocal prepared
                observation = self.driver.session.observation
                if observation is None:
                    return
                classified = self.policy.classify(proposed, observation)
                if classified is None:
                    return
                prepared = self.store.prepare_effect(
                    action_id=proposed.action_id,
                    target=classified.target,
                    object_id=classified.object_id,
                    goal_id=self.goal_by_object.get(classified.object_id, "g2"),
                    branch_id="root",
                    observation_id=observation.observation_id,
                    note=classified.note,
                )

            action = self.driver.take_action(before_dispatch=before_dispatch)
            self.sequence += 1
            result = self.backend.execute(action)
            if prepared is not None:
                self.store.confirm_effect(
                    action.action_id,
                    status=(
                        "failed"
                        if result.feedback.status is FeedbackStatus.ERROR
                        else "confirmed"
                        if result.feedback.status is FeedbackStatus.OBSERVED
                        else "unknown"
                    ),
                    observation_id=result.observation.observation_id,
                    note=f"feedback={result.feedback.status.value}",
                )
            object_id = _object_from_url(result.observation.url)
            if object_id is not None:
                self.executed_objects.append(object_id)
            if intent is not None and intent.effect_class.value == "draft_save" and object_id is not None:
                self._record(result.observation, action)
                if result.feedback.status is FeedbackStatus.ERROR:
                    self.failed_object = object_id
                    self._recover(result, object_id)
                else:
                    self._verify(result.observation, object_id)
            step = self.driver.accept_observation(result.observation)
        return step

    def _record(self, observation: Observation, action: WebAction) -> None:
        artifact = observation.artifacts.get("screenshot")
        if artifact is None or not artifact.available:
            return
        object_id = _object_from_url(observation.url)
        if object_id is None:
            return
        ref = evidence_ref_from_artifact(
            artifact,
            run_id=self.driver.session.run_id,
            branch_id="root",
            evidence_id=f"ev-{observation.observation_id}-screenshot",
            observation_id=observation.observation_id,
            action_id=action.action_id,
        )
        record_observed_object(
            self.store,
            object_id=object_id,
            fields={"url": observation.url, "observed_with": "screenshot"},
            evidence=(ref,),
            source=f"observation:{observation.observation_id}",
            goal_id=self.goal_by_object.get(object_id),
            branch_id="root",
        )

    def _verify(self, observation: Observation, object_id: str) -> None:
        note = self.notes[object_id]
        goal_id = self.goal_by_object[object_id]
        requests = [
            self._request(goal_id, object_id, note, observation, visible=True),
        ]
        for dependent in self.checks.get(object_id, ()):
            requests.append(
                self._request(dependent, object_id, note, observation, visible=False)
            )
        for request in requests:
            result = self.verifier.verify(request)
            self.verifications.append(result)

    def _request(
        self,
        goal_id: str,
        object_id: str,
        note: str,
        observation: Observation,
        *,
        visible: bool,
    ) -> VerificationRequest:
        if visible:
            conditions = (
                VerificationCondition(
                    "visible_note",
                    "the visible profile note equals the saved value",
                    "visible_value_equals",
                    expected=note,
                    role="textbox",
                    name="Review draft",
                ),
                VerificationCondition("screenshot", "screenshot saved", "screenshot_saved"),
            )
        else:
            conditions = (
                VerificationCondition(
                    "page_note",
                    "the current page text contains the repaired note",
                    "page_text_contains",
                    expected=note,
                ),
                VerificationCondition("screenshot", "screenshot saved", "screenshot_saved"),
            )
        return VerificationRequest(
            goal_id=goal_id,
            subject_id=object_id,
            conditions=conditions,
            observation=observation,
            trigger=VerificationTrigger.SUBGOAL_END,
            branch="root",
            completed_object_ids=(object_id,) if visible else None,
        )

    def _recover(self, result: ExecutionResult, object_id: str) -> None:
        artifact = result.observation.artifacts.get("screenshot")
        refs = ()
        if artifact is not None and artifact.available:
            refs = (
                evidence_ref_from_artifact(
                    artifact,
                    run_id=self.driver.session.run_id,
                    branch_id="root",
                    evidence_id=f"ev-{result.observation.observation_id}-recovery",
                    observation_id=result.observation.observation_id,
                    action_id=result.action_id,
                ),
            )
        failure = FailureEvidence(
            action_id=result.action_id,
            sequence=self.sequence,
            failure_class=RecoveryClass.TRANSIENT,
            goal_id=self.goal_by_object[object_id],
            message=result.feedback.message,
            observation_id=result.observation.observation_id,
            object_id=object_id,
            evidence=refs,
        )
        plan = self.planner.plan([failure])
        attempt = self.controller.authorize(plan.affected_goal_ids)
        applied = self.controller.apply(plan, attempt=attempt)
        self.recovery.append(applied)
        try:
            self.driver.session.logger.emit(
                "recovery",
                goal_id=None if plan.anchor is None else plan.anchor.goal_id,
                result=applied.to_dict(),
            )
        except Exception:  # noqa: BLE001 - diagnostics must not alter recovery
            pass


def _object_from_url(url: str) -> str | None:
    path = urlsplit(url).path.rstrip("/")
    if not path.startswith("/product/"):
        return None
    return unquote(path.rsplit("/", 1)[-1])


@dataclass
class L04Interrupted:
    prepared: Any
    target: dict[str, Any]
    note: str
    run_id: str
    target_id: str
    store: StateStore
    save_action_id: str
    state: dict[str, Any]


def interrupt_l04_after_effective_save(
    runtime_dir: str | Path,
    *,
    seed: int = 101,
    run_id: str | None = None,
) -> L04Interrupted:
    """Execute the real save, lose feedback, then stop as if the process died.

    The pending effect is already in the snapshot. The driver/generator is closed
    without accepting the lost observation, so this helper does not claim that a
    generator stack survived.
    """

    runtime_dir = Path(runtime_dir)
    prepared = start("L04", seed, runtime_dir=runtime_dir, run_id=run_id)
    run_name = str(prepared.run_dir.name)
    target, list_url = public_l04_target(prepared)
    target_id = str(target["id"])
    note = str(prepared.case.task_params["l04_draft_text"])
    task = str(prepared.agent_view()["task"])
    store = StateStore(runtime_dir, run_name)
    store.initialize(task)
    _bootstrap_l04(store, prepared, target_id, task)
    clock = SystemClock()
    session = Session(
        run_name,
        runtime_dir=runtime_dir,
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + 180, clock=clock), clock=clock),
    )
    session.start_task(task)
    backend = PlaywrightBackend(
        run_name,
        evidence=session.evidence,
        headless=True,
        runtime_dir=runtime_dir,
    )
    policy = EffectPolicy()
    lost_backend = LostFeedbackBackend(backend, policy)
    driver = None
    save_action_id = ""
    try:
        backend.start(f"{prepared.base_url}{prepared.case.start_path}")
        session.ingest_observation(backend.observe())
        model = L04Model(
            target_name=str(target["name"]),
            target_id=target_id,
            list_url=list_url,
            product_url=f"{prepared.base_url}/product/{target_id}",
            note=note,
        )
        driver = create_stateful_driver(session, model, store, branch="root")
        step = driver.start()
        while step.status is DriverStatus.ACTION:
            assert step.action is not None
            action = step.action
            observation = session.observation
            intent = policy.classify(action, observation) if observation is not None else None
            effect = None
            if intent is not None:
                effect = store.prepare_effect(
                    action_id=action.action_id,
                    target=intent.target,
                    object_id=intent.object_id,
                    goal_id="g1",
                    branch_id="root",
                    observation_id=observation.observation_id if observation else None,
                    note=intent.note,
                )
            dispatched = driver.take_action()
            result = lost_backend.execute(dispatched)
            if effect is not None:
                store.confirm_effect(
                    dispatched.action_id,
                    status=(
                        "unknown"
                        if result.feedback.status is FeedbackStatus.UNKNOWN
                        else "confirmed"
                        if result.feedback.status is FeedbackStatus.OBSERVED
                        else "failed"
                    ),
                    observation_id=result.observation.observation_id,
                    note="interrupted after the effective save",
                )
            if (
                intent is not None
                and intent.effect_class.value == "draft_save"
                and result.feedback.status is FeedbackStatus.UNKNOWN
            ):
                save_action_id = dispatched.action_id
                break
            step = driver.accept_observation(result.observation)
        state = store.state.to_dict()
    finally:
        if driver is not None:
            driver.close()
        lost_backend.close()
        session.close()
        prepared.stop()
    return L04Interrupted(
        prepared=prepared,
        target=target,
        note=note,
        run_id=run_name,
        target_id=target_id,
        store=store,
        save_action_id=save_action_id,
        state=state,
    )


def run_l05(
    runtime_dir: str | Path,
    *,
    seed: int = 101,
    run_id: str | None = None,
) -> L05Run:
    runtime_dir = Path(runtime_dir)
    prepared = start("L05", seed, runtime_dir=runtime_dir, run_id=run_id)
    run_name = str(prepared.run_dir.name)
    targets = public_l05_targets(prepared)
    target_ids = [str(target["id"]) for target in targets]
    notes = [
        str(prepared.case.task_params["l05_note_first"]),
        str(prepared.case.task_params["l05_note_second"]),
        str(prepared.case.task_params["l05_note_third"]),
    ]
    task = str(prepared.agent_view()["task"])
    store = StateStore(runtime_dir, run_name)
    store.initialize(task)
    _bootstrap_l05(store, prepared, target_ids, task)

    clock = SystemClock()
    session = Session(
        run_name,
        runtime_dir=runtime_dir,
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + 240, clock=clock), clock=clock),
    )
    session.logger.set_state_context(
        lambda: {
            "state_revision": store.state.revision,
            "branch_id": "root",
            "plan_version": store.state.plan_version,
        },
        goal_id="g2",
    )
    session.start_task(task)
    backend = PlaywrightBackend(
        run_name,
        evidence=session.evidence,
        headless=True,
        runtime_dir=runtime_dir,
    )
    policy = EffectPolicy()
    fail_backend = FailOnceSaveBackend(backend, policy, object_id=target_ids[1])
    driver = None
    step = None
    try:
        backend.start(f"{prepared.base_url}{prepared.case.start_path}")
        session.ingest_observation(backend.observe())
        model = L05Model(products=targets, notes=notes, base_url=prepared.base_url)
        driver = create_stateful_driver(session, model, store, branch="root")
        verifier = GoalVerifier(store, logger=session.logger)
        planner = RecoveryPlanner(store)
        controller = RecoveryController(
            store,
            limits=RecoveryLimits(max_recoveries_per_goal=2, max_recoveries_total=5),
        )
        runner = _RecoveryAwareRunner(
            driver,
            fail_backend,
            store,
            verifier=verifier,
            notes=dict(zip(target_ids, notes)),
            goal_by_object={target_ids[0]: "g1", target_ids[1]: "g2", target_ids[2]: "g3"},
            checks={target_ids[1]: ("g2-check",)},
            planner=planner,
            controller=controller,
        )
        step = runner.run()
        judge = judge_l05(prepared.db_path, target_ids=target_ids, expected_notes=notes)
        events = read_events(session.evidence.events_path)
        state = store.state.to_dict()
    finally:
        if driver is not None:
            driver.close()
        fail_backend.close()
        session.close()
        prepared.stop()
    return L05Run(
        prepared=prepared,
        targets=targets,
        notes=notes,
        task=task,
        run_id=run_name,
        store=store,
        step=step,
        verifications=runner.verifications,
        recovery=runner.recovery,
        failed_object=runner.failed_object,
        executed_objects=runner.executed_objects,
        judge=judge,
        events=events,
        state=state,
    )


__all__ = [
    "bootstrap_store",
    "evidence_ref",
    "make_goal",
    "make_observation",
    "page_map",
    "page_with_value",
    "value_element",
    "verified_goal",
]
