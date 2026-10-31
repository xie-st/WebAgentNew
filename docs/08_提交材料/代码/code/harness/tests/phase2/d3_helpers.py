"""D3 end-to-end helpers: L01 through the real local site and the ledger.

The scripted model is deliberately not a reasoning benchmark. A test-side
planner computes the target from the public product list (the same conditions
the task states), and the model emits that public plan as tool calls. Element
numbers are read from the actual model-visible observation each turn; hidden
judge truth is never used to choose an action.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from effgen.models.base import GenerationResult

from webagent.backends.playwright import PlaywrightBackend
from webagent.budget import Budget, Deadline
from webagent.clock import SystemClock
from webagent.effgen_bridge import DriverStatus, DriverStep
from webagent.events import read_events
from webagent.models.fake import FakeResponse, ScriptedModel
from webagent.protocol import Observation, WebAction
from webagent.session import Session
from webagent.state import (
    EffectPolicy,
    Goal,
    GoalStatus,
    ObservedObject,
    PlanNode,
    PlanUpdate,
    StateStore,
    StatefulLoopRunner,
    create_stateful_driver,
)
from webagent.testing import start
from webagent.testing.evaluation import EvidenceRecord


class SimulatedKill(RuntimeError):
    """Test-only process interruption just before a declared write executes."""


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


class L01PublicPlanModel(ScriptedModel):
    """A scripted model with a public-data plan and prompt-bound element numbers."""

    def __init__(
        self,
        *,
        target_name: str,
        target_id: str,
        target_sku: str,
        list_url: str,
        draft_text: str,
    ) -> None:
        super().__init__([])
        self.target_name = target_name
        self.target_id = target_id
        self.target_sku = target_sku
        self.list_url = list_url
        self.draft_text = draft_text
        self.stage = 0

    def generate(self, prompt: str, config: Any = None, **kwargs: Any) -> GenerationResult:
        text = self._next_text(prompt)
        self._record(prompt, config, None, FakeResponse(text=text), None)
        tokens = self.count_tokens(prompt).count + self.count_tokens(text).count
        return GenerationResult(
            text=text,
            tokens_used=tokens,
            finish_reason="stop",
            model_name=self.model_name,
        )

    def _next_text(self, prompt: str) -> str:
        stage = self.stage
        self.stage += 1
        elements = _latest_elements(prompt)
        if stage == 0:
            return _action("state_read", {})
        if stage == 1:
            return _action("web_goto", {"url": self.list_url})
        if stage == 2:
            link = self._find(elements, "link", self.target_name)
            return _action("web_click", {"target": link["number"]})
        if stage == 3:
            textbox = self._find(elements, "textbox", "Review draft")
            return _action(
                "web_type",
                {"target": textbox["number"], "text": self.draft_text},
            )
        if stage == 4:
            button = self._find(elements, "button", "Cancel draft")
            return _action("web_click", {"target": button["number"]})
        if stage == 5:
            return _action("web_goto", {"url": self.list_url})
        if stage == 6:
            link = self._find(elements, "link", self.target_name)
            return _action("web_click", {"target": link["number"]})
        if stage == 7:
            return (
                "Thought: the target detail was observed twice and the draft was "
                "cancelled without submitting.\n"
                f"Final Answer: {self.target_id} ({self.target_sku}) checked; "
                "draft cancelled; no comment sent"
            )
        raise RuntimeError(f"L01 public plan has no stage {stage}")

    @staticmethod
    def _find(elements: list[dict[str, str]], role: str, name: str) -> dict[str, str]:
        for element in elements:
            if element["role"] == role and element["name"] == name:
                return element
        visible = [f"{item['role']}:{item['name']}" for item in elements]
        raise RuntimeError(f"no {role} named {name!r} in current observation: {visible}")


class KillBeforeEffectBackend:
    """Delegates to Playwright, but interrupts before the declared write runs."""

    backend_id = "playwright-kill-before-effect"

    def __init__(self, inner: PlaywrightBackend, policy: EffectPolicy) -> None:
        self.inner = inner
        self.policy = policy
        self.killed_action: WebAction | None = None

    def observe(self) -> Observation:
        return self.inner.observe()

    def execute(self, action: WebAction) -> Any:
        observation = self.inner.current_observation
        if observation is not None and self.policy.classify(action, observation) is not None:
            self.killed_action = action
            raise SimulatedKill(
                f"simulated kill before executing {action.op.value} {action.action_id}"
            )
        return self.inner.execute(action)

    def close(self) -> None:
        self.inner.close()


@dataclass
class L01Run:
    prepared: Any
    target_id: str
    target_name: str
    target_sku: str
    target_page_url: str
    draft_text: str
    task: str
    run_id: str
    store: StateStore
    session: Session
    runner: StatefulLoopRunner
    step: DriverStep | None
    killed: bool = False
    kill_error: BaseException | None = None
    judge_result: Any = None
    verified: bool = False
    state_before_judge: dict[str, Any] = field(default_factory=dict)
    view_before_judge: dict[str, Any] = field(default_factory=dict)
    object_progress_before_judge: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    view_after: dict[str, Any] = field(default_factory=dict)
    reloaded_state: dict[str, Any] = field(default_factory=dict)
    reload_equal: bool = False
    executed_action_ids: list[str] = field(default_factory=list)
    detail_observations: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return Path(self.prepared.run_dir)


def public_l01_target(prepared: Any) -> tuple[dict[str, Any], str]:
    """Resolve L01's target from the public catalog, never from hidden truth."""

    params = dict(prepared.case.task_params)
    rows, total = prepared.site.store.list_products(
        category=str(params["l01_category"]),
        max_price_cents=int(round(float(params["l01_max_price"]) * 100)),
        min_rating=float(params["l01_min_rating"]),
        in_stock=True,
        page_size=10_000,
    )
    if total != 1:
        raise AssertionError(f"public conditions matched {total} products, expected 1")
    product = dict(rows[0])
    page = int(product["position"]) // int(prepared.site.page_size) + 1
    list_url = f"{prepared.base_url}/products?page={page}"
    return product, list_url


def _bootstrap_goal(store: StateStore, prepared: Any, target_id: str, task: str) -> None:
    criteria = tuple(str(item) for item in prepared.case.completion["criteria"])
    goal = Goal(
        goal_id="g1",
        requirement=task,
        acceptance_criteria=criteria,
        object_scope=(target_id,),
        status=GoalStatus.RUNNING,
        branch_id="root",
    )
    node = PlanNode(
        node_id="n1",
        goal_id="g1",
        object_scope=(target_id,),
        resources={"tab": "root"},
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
            note="D3 bootstrap from the public case view",
        )
    )


def _product_recorder(target_id: str):
    def record(
        observation: Observation, action: WebAction | None
    ) -> ObservedObject | None:
        path = urlsplit(observation.url).path.rstrip("/")
        if not path.startswith("/product/"):
            return None
        object_id = unquote(path.rsplit("/", 1)[-1])
        heading = ""
        for line in observation.ax_text.splitlines():
            if line.startswith("- h1: "):
                heading = line[len("- h1: ") :]
                break
        return ObservedObject(
            object_id=object_id,
            fields={
                "url": observation.url,
                "heading": heading,
                "observed_with": "screenshot",
            },
        )

    return record


def _judge_evidence(state: Any, runtime_dir: Path, object_type: str) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for fact in state.facts.values():
        if fact.field != "page_observed":
            continue
        for ref in fact.evidence:
            if ref.kind != "screenshot":
                continue
            records.append(
                EvidenceRecord(
                    object_type=object_type,
                    object_id=fact.object_id,
                    kind="screenshot",
                    reference=str((runtime_dir / ref.reference).resolve()),
                    sha256=ref.sha256,
                )
            )
    return records


def _evidence_refs_for_object(state: Any, object_id: str) -> tuple[Any, ...]:
    fact = state.facts.get(f"fact-{object_id}-page-observed")
    if fact is None:
        return ()
    refs: list[Any] = []
    seen: set[str] = set()
    for ref in fact.evidence:
        if ref.evidence_id in seen:
            continue
        seen.add(ref.evidence_id)
        refs.append(ref)
    return tuple(refs)


def run_l01(
    runtime_dir: str | Path,
    *,
    seed: int = 101,
    run_id: str | None = None,
    kill_before_cancel: bool = False,
    verify: bool = True,
    judge_evidence: str = "ledger",
) -> L01Run:
    """Run L01 end to end on the local site with the stateful loop.

    Caller owns cleanup of returned ``prepared`` (the helper stops the HTTP site
    only after judging; the browser and session are closed before returning).
    """

    if judge_evidence not in {"ledger", "none"}:
        raise ValueError("judge_evidence must be 'ledger' or 'none'")
    runtime_dir = Path(runtime_dir)
    prepared = start("L01", seed, runtime_dir=runtime_dir, run_id=run_id)
    # Exercise the public reset entry as well; start already created this copy,
    # and the same seed must rebuild identical public data.
    prepared.reset(seed)
    run_id = str(prepared.run_dir.name)
    target, list_url = public_l01_target(prepared)
    target_id = str(target["id"])
    task = str(prepared.agent_view()["task"])

    store = StateStore(runtime_dir, run_id)
    store.initialize(task)
    _bootstrap_goal(store, prepared, target_id, task)

    clock = SystemClock()
    session = Session(
        run_id,
        runtime_dir=runtime_dir,
        clock=clock,
        budget=Budget(
            deadline=Deadline(clock.now() + 180, clock=clock),
            clock=clock,
        ),
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
        run_id,
        evidence=session.evidence,
        headless=True,
        runtime_dir=runtime_dir,
    )
    policy = EffectPolicy()
    effective_backend: Any = backend
    driver = None
    runner = None
    step: DriverStep | None = None
    killed = False
    kill_error: BaseException | None = None
    try:
        backend.start(f"{prepared.base_url}{prepared.case.start_path}")
        session.ingest_observation(backend.observe())
        model = L01PublicPlanModel(
            target_name=str(target["name"]),
            target_id=target_id,
            target_sku=str(target["sku"]),
            list_url=list_url,
            draft_text=str(prepared.case.task_params["l01_draft_text"]),
        )
        driver = create_stateful_driver(session, model, store, branch="root")
        if kill_before_cancel:
            effective_backend = KillBeforeEffectBackend(backend, policy)
        runner = StatefulLoopRunner(
            driver,
            effective_backend,
            store,
            branch="root",
            goal_id="g1",
            effect_policy=policy,
            observation_recorder=_product_recorder(target_id),
        )
        try:
            step = runner.run()
        except SimulatedKill as exc:
            killed = True
            kill_error = exc
    finally:
        if driver is not None:
            driver.close()
        effective_backend.close()
        session.close()

    events = read_events(session.evidence.events_path)
    executed_ids = list(
        dict.fromkeys(
            str(event["action_id"])
            for event in events
            if event["event_type"] == "feedback" and event["action_id"]
        )
    )
    detail_observations = list(runner.observed_objects) if runner is not None else []
    target_page_url = f"{prepared.base_url}/product/{target_id}"

    try:
        state_before_judge = store.state.to_dict()
        view_before_judge = store.state.view()
        object_progress_before = store.state.object_progress()
        judge_result = None
        verified = False
        if not killed:
            evidence = (
                _judge_evidence(
                    store.state,
                    prepared.run_dir,
                    str(prepared.case.judge_config["object_type"]),
                )
                if judge_evidence == "ledger"
                else []
            )
            result = prepared.judge(
                evidence=evidence,
                answer={"product_ids": [target_id]},
            )
            judge_result = result
            (prepared.run_dir / "judge_result.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if (
                verify
                and result.passed
                and step is not None
                and step.status is DriverStatus.COMPLETED
            ):
                state = store.state
                refs = _evidence_refs_for_object(state, target_id)
                store.verify_goal(
                    "g1",
                    evidence=refs,
                    base_revision=state.revision,
                    plan_version=state.plan_version,
                    source_branch="root",
                    completed_object_ids=(target_id,),
                    note=f"local judge {result.verdict}",
                )
                verified = True

        state_after = store.state.to_dict()
        view_after = store.state.view()
        reloaded = StateStore(runtime_dir, run_id)
        reloaded.load()
        reloaded_state = reloaded.state.to_dict()
    finally:
        prepared.stop()

    return L01Run(
        prepared=prepared,
        target_id=target_id,
        target_name=str(target["name"]),
        target_sku=str(target["sku"]),
        target_page_url=target_page_url,
        draft_text=str(prepared.case.task_params["l01_draft_text"]),
        task=task,
        run_id=run_id,
        store=store,
        session=session,
        runner=runner,
        step=step,
        killed=killed,
        kill_error=kill_error,
        judge_result=judge_result,
        verified=verified,
        state_before_judge=state_before_judge,
        view_before_judge=view_before_judge,
        object_progress_before_judge=object_progress_before,
        state_after=state_after,
        view_after=view_after,
        reloaded_state=reloaded_state,
        reload_equal=state_after == reloaded_state,
        executed_action_ids=executed_ids,
        detail_observations=detail_observations,
        events=events,
    )
