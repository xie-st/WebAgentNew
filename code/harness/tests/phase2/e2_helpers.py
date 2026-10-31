"""E2 helpers: L06/L07 through the local site and the controlled loop.

The scripted model follows a public plan built from the public product catalog
and the task text. Element numbers come from the prompt's current observation;
hidden judge truth is never used to choose an action. The "user" is simulated
through the same browser page the Agent owns, which is the faithful offline
stand-in for a human taking the browser during a handoff.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from effgen.models.base import GenerationResult

from webagent.backends.playwright import PlaywrightBackend
from webagent.budget import Budget, Deadline
from webagent.clock import Clock, SystemClock
from webagent.control import (
    ControlAwareModel,
    ControlCoordinator,
    ControlEventLog,
    ControlState,
    ControlledLoopRunner,
    MemoryControlSource,
    read_control_events,
)
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
    create_stateful_driver,
)
from webagent.testing import start
from webagent.testing.evaluation import EvidenceBundle, EvidenceRecord
from webagent.testing.evaluation.hidden import HiddenStateReader
from webagent.testing.evaluation.judge import JudgeResult, LocalJudge

from d3_helpers import _action, _latest_elements

_DRAFT_LINE = re.compile(
    r'^\[(?P<number>\d+)\]\s+role=(?P<role>\S+)\s+name=(?P<name>"(?:[^"\\]|\\.)*")'
    r'(?:\s+value=(?P<value>"(?:[^"\\]|\\.)*"))?'
)


def _quoted(text: str) -> str:
    try:
        return str(json.loads(text))
    except json.JSONDecodeError:
        return text.strip('"')


def public_control_target(prepared: Any, prefix: str) -> tuple[dict[str, Any], str]:
    """Resolve the unique target from public conditions, never hidden truth."""

    params = dict(prepared.case.task_params)
    rows, total = prepared.site.store.list_products(
        category=str(params[f"{prefix}_category"]),
        max_price_cents=int(round(float(params[f"{prefix}_max_price"]) * 100)),
        min_rating=float(params[f"{prefix}_min_rating"]),
        in_stock=True,
        page_size=10_000,
    )
    if total != 1:
        raise AssertionError(
            f"{prepared.case.case_id}: public conditions matched {total} products, expected 1"
        )
    product = dict(rows[0])
    page = int(product["position"]) // int(prepared.site.page_size) + 1
    list_url = f"{prepared.base_url}/products?page={page}"
    return product, list_url


def _find(elements: list[dict[str, str]], role: str, name: str) -> dict[str, str]:
    for element in elements:
        if element["role"] == role and element["name"] == name:
            return element
    visible = [f"{item['role']}:{item['name']}" for item in elements]
    raise RuntimeError(f"no {role} named {name!r} in current observation: {visible}")


class _ScriptedPlanModel(ScriptedModel):
    def __init__(self) -> None:
        super().__init__([])
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

    def _next_text(self, prompt: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class L06PlanModel(_ScriptedPlanModel):
    """goto list -> target -> type -> save (overlay) -> scroll -> cancel -> done.

    The scroll after the overlay is proposed only after the resume has accepted
    the deferred truthful feedback. The controlled runner drops it because it
    uses the pre-handoff observation, then re-observes and continues.
    """

    def __init__(self, *, target_name: str, list_url: str, draft_text: str) -> None:
        super().__init__()
        self.target_name = target_name
        self.list_url = list_url
        self.draft_text = draft_text

    def _next_text(self, prompt: str) -> str:
        stage = self.stage
        self.stage += 1
        elements = _latest_elements(prompt)
        if stage == 0:
            return _action("web_goto", {"url": self.list_url})
        if stage == 1:
            return _action("web_click", {"target": _find(elements, "link", self.target_name)["number"]})
        if stage == 2:
            return _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.draft_text},
            )
        if stage == 3:
            return _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        if stage == 4:
            return _action("web_scroll", {"direction": "down"})
        if stage == 5:
            return _action("web_click", {"target": _find(elements, "button", "Cancel draft")["number"]})
        if stage == 6:
            return (
                "Thought: the saved draft was re-read after the user completed the "
                "sign-in step and it was then cancelled without sending.\n"
                f"Final Answer: draft verified and cancelled on the target product ({self.target_name})"
            )
        raise RuntimeError(f"L06 public plan has no stage {stage}")


class L07PlanModel(_ScriptedPlanModel):
    """goto list -> target -> type -> save (paused before dispatch) -> target -> cancel."""

    def __init__(self, *, target_name: str, list_url: str, draft_text: str) -> None:
        super().__init__()
        self.target_name = target_name
        self.list_url = list_url
        self.draft_text = draft_text

    def _next_text(self, prompt: str) -> str:
        stage = self.stage
        self.stage += 1
        elements = _latest_elements(prompt)
        if stage == 0:
            return _action("web_goto", {"url": self.list_url})
        if stage == 1:
            return _action("web_click", {"target": _find(elements, "link", self.target_name)["number"]})
        if stage == 2:
            return _action(
                "web_type",
                {"target": _find(elements, "textbox", "Review draft")["number"], "text": self.draft_text},
            )
        if stage == 3:
            return _action("web_click", {"target": _find(elements, "button", "Save draft")["number"]})
        if stage == 4:
            return _action("web_click", {"target": _find(elements, "link", self.target_name)["number"]})
        if stage == 5:
            return _action("web_click", {"target": _find(elements, "button", "Cancel draft")["number"]})
        if stage == 6:
            return (
                "Thought: the user changed the page while paused; I re-observed the "
                "new list page and cancelled the saved draft without sending.\n"
                f"Final Answer: draft checked and cancelled on target {self.target_name}"
            )
        raise RuntimeError(f"L07 public plan has no stage {stage}")


class UserSimulatingSource(MemoryControlSource):
    """A control source whose first handoff poll performs the user's edit.

    It only acts after the coordinator reports ``user_control``; it records a
    probe of model/action/event counters at that moment, performs the real
    browser edit, and returns the explicit resume signal.
    """

    def __init__(
        self,
        *,
        user_action: Callable[[], None],
        probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.user_action = user_action
        self.probe = probe
        self.coordinator: ControlCoordinator | None = None
        self.simulated = False
        self.probe_at_handoff: dict[str, Any] = {}
        self.probe_after_handoff: dict[str, Any] = {}
        self.user_action_error: BaseException | None = None

    def attach(self, coordinator: ControlCoordinator) -> None:
        self.coordinator = coordinator

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        coordinator = self.coordinator
        if (
            coordinator is not None
            and coordinator.state is ControlState.USER_CONTROL
            and not self.simulated
        ):
            self.simulated = True
            if self.probe is not None:
                self.probe_at_handoff = dict(self.probe())
            try:
                self.user_action()
            except BaseException as exc:  # noqa: BLE001 - reported by the test
                self.user_action_error = exc
            if self.probe is not None:
                self.probe_after_handoff = dict(self.probe())
            return "resume"
        return None


# ------------------------------------------------------------------ browser user
def _page(backend: PlaywrightBackend) -> Any:
    page = getattr(backend, "_page", None)
    if page is None:
        raise RuntimeError("the browser page is not available for the simulated user")
    return page


def user_complete_auth(backend: PlaywrightBackend, *, kind: str) -> None:
    page = _page(backend)
    name = "Complete verification" if kind == "verification" else "Sign in"
    page.get_by_role("button", name=name).click()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(200)


def user_save_draft_and_open_products(backend: PlaywrightBackend) -> None:
    page = _page(backend)
    page.get_by_role("button", name="Save draft").click()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(150)
    page.get_by_role("link", name="Products", exact=True).click()
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_timeout(150)


# ------------------------------------------------------------------ recorder
def control_product_recorder(
    target_id: str,
    *,
    previous_fields: Callable[[], Mapping[str, Any]] | None = None,
) -> Callable[[Observation, WebAction | None], ObservedObject | None]:
    def record(
        observation: Observation, action: WebAction | None
    ) -> ObservedObject | None:
        path = urlsplit(observation.url).path.rstrip("/")
        if not path.startswith("/product/"):
            return None
        object_id = unquote(path.rsplit("/", 1)[-1])
        ax = observation.ax_text
        lowered = ax.casefold()
        if "sign in required" in lowered or "verification required" in lowered:
            return None
        heading = ""
        notices: list[str] = []
        draft = ""
        for line in ax.splitlines():
            stripped = line.strip()
            if stripped.startswith("- h1: "):
                heading = stripped[len("- h1: ") :]
            if stripped.startswith("- ") and "Draft" in stripped:
                notices.append(stripped[2:])
            draft_line = _DRAFT_LINE.match(stripped)
            if draft_line is not None:
                name = _quoted(draft_line.group("name"))
                if name == "Review draft" and draft_line.group("value"):
                    draft = _quoted(draft_line.group("value"))
        prior = dict(previous_fields() or {}) if previous_fields is not None else {}
        last_saved = prior.get("draft_ever_saved_value") or prior.get("draft_value") or ""
        if draft:
            last_saved = draft
        return ObservedObject(
            object_id=object_id,
            fields={
                "url": observation.url,
                "heading": heading,
                "draft_value": draft,
                "draft_ever_saved_value": last_saved,
                "draft_cleared": bool(last_saved) and not draft,
                "notices": notices,
                "observed_with": "screenshot",
            },
        )

    return record


# ---------------------------------------------------------------------- judge
def ledger_evidence(
    state: Any, runtime_dir: str | Path, object_type: str = "product"
) -> list[EvidenceRecord]:
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
                    reference=str((Path(runtime_dir) / ref.reference).resolve()),
                    sha256=ref.sha256,
                )
            )
    return records


def judge_control_case(
    prepared: Any,
    *,
    evidence: Any = None,
    answer: Any = None,
) -> JudgeResult:
    """Reuse D1's L01 checks with L06/L07's own hidden configuration."""

    case = prepared.case
    hidden = HiddenStateReader(prepared.db_path)
    judge = LocalJudge(case, hidden, evidence_root=prepared.run_dir)
    checks = judge._judge_l01(EvidenceBundle.coerce(evidence), answer)
    return JudgeResult(
        case_id=case.case_id,
        case_version=case.case_version,
        seed=int(hidden.meta().get("seed", 0)),
        checks=tuple(checks),
        evidence_count=len(EvidenceBundle.coerce(evidence)),
        hidden_state_hash=hidden.snapshot_hash(),
    )


# ----------------------------------------------------------------------- run
@dataclass
class ControlCaseRun:
    prepared: Any
    target: dict[str, Any]
    target_id: str
    list_url: str
    task: str
    run_id: str
    store: StateStore
    session: Session
    coordinator: ControlCoordinator
    source: UserSimulatingSource
    model: ControlAwareModel
    runner: ControlledLoopRunner
    step: DriverStep | None
    events: list[dict[str, Any]] = field(default_factory=list)
    control_events: list[dict[str, Any]] = field(default_factory=list)
    judge_result: JudgeResult | None = None
    verified: bool = False
    state_before_judge: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    fact_fields: dict[str, Any] = field(default_factory=dict)
    reloaded_equal: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.prepared.run_dir)

    @property
    def target_page_url(self) -> str:
        return f"{self.prepared.base_url}/product/{self.target_id}"


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
            note="E2 bootstrap from the public case view",
        )
    )


def _run(
    case_id: str,
    seed: int,
    *,
    runtime_dir: str | Path,
    prefix: str,
    model_factory: Callable[[dict[str, Any], str], ScriptedModel],
    user_action: Callable[[PlaywrightBackend], None],
    auth_kind: str | None = None,
    pause_before_dispatch: bool = False,
    run_id: str | None = None,
    clock: Clock | None = None,
) -> ControlCaseRun:
    runtime = Path(runtime_dir)
    prepared = start(case_id, seed, runtime_dir=runtime, run_id=run_id)
    if auth_kind == "verification":
        # Replace the manifest's login fault (same fault id) so exactly one
        # overlay type is active in this variant.
        prepared.inject_fault(
            {
                "fault_id": "sign-in-mid-edit",
                "type": "verification_prompt",
                "enabled": True,
                "trigger": {"event": "draft_saved", "after": 1},
                "params": {},
                "note": "E2 verification variant replaces the sign-in overlay",
            }
        )
    target, list_url = public_control_target(prepared, prefix)
    target_id = str(target["id"])
    task = str(prepared.agent_view()["task"])
    run_id = str(prepared.run_dir.name)

    store = StateStore(runtime, run_id)
    store.initialize(task)
    _bootstrap_goal(store, prepared, target_id, task)

    clock = clock or SystemClock()
    budget_seconds = int(prepared.case.budget_seconds)
    session = Session(
        run_id,
        runtime_dir=runtime,
        clock=clock,
        budget=Budget(deadline=Deadline(clock.now() + min(budget_seconds, 180), clock=clock), clock=clock),
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
        runtime_dir=runtime,
    )
    coordinator = ControlCoordinator(
        run_id,
        clock=clock,
        event_log=ControlEventLog(
            run_id, prepared.run_dir / "control_events.jsonl", clock=clock
        ),
        platform_deadline_epoch=session.deadline.epoch_seconds,
    )
    source = UserSimulatingSource(user_action=lambda: user_action(backend))
    source.attach(coordinator)
    coordinator.source = source
    model = ControlAwareModel(
        model_factory(target, list_url),
        session=session,
        coordinator=coordinator,
        clock=clock,
    )

    driver = None
    runner = None
    step = None
    try:
        backend.start(f"{prepared.base_url}{prepared.case.start_path}")
        session.ingest_observation(backend.observe())
        driver = create_stateful_driver(session, model, store, branch="root")

        def on_step(current: DriverStep) -> None:
            if not pause_before_dispatch or current.action is None:
                return
            element = session.observation.elements.get(current.action.target or "") if session.observation else None
            if element is not None and " ".join(element.name.casefold().split()) == "save draft":
                source.send("pause")

        def _probe() -> dict[str, Any]:
            current_events = read_events(session.evidence.events_path)
            return {
                "model_calls": len(model.calls),
                "session_model_calls_total": session.model_calls_total,
                "backend_executed_actions": sum(
                    1 for event in current_events if event["event_type"] == "dispatched"
                ),
                "events": len(current_events),
            }

        source.probe = _probe
        runner = ControlledLoopRunner(
            driver,
            backend,
            coordinator,
            store=store,
            branch="root",
            goal_id="g1",
            effect_policy=EffectPolicy(),
            observation_recorder=control_product_recorder(
                target_id,
                previous_fields=lambda: (
                    dict(store.state.facts[f"fact-{target_id}-page-observed"].value)
                    if f"fact-{target_id}-page-observed" in store.state.facts
                    else {}
                ),
            ),
            on_step=on_step,
        )
        step = runner.run()
    finally:
        if driver is not None:
            driver.close()
        backend.close()
        session.close()
        coordinator.close()

    events = read_events(session.evidence.events_path)
    control_events = read_control_events(prepared.run_dir / "control_events.jsonl")
    state_before = store.state.to_dict()
    fact_fields = {}
    fact = store.state.facts.get(f"fact-{target_id}-page-observed")
    if fact is not None:
        fact_fields = dict(fact.value)
    evidence = ledger_evidence(store.state, prepared.run_dir)
    judge_result = judge_control_case(
        prepared, evidence=evidence, answer={"product_ids": [target_id]}
    )
    (prepared.run_dir / "judge_result.json").write_text(
        json.dumps(judge_result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verified = False
    if (
        step is not None
        and step.status is DriverStatus.COMPLETED
        and judge_result.passed
        and fact is not None
    ):
        state = store.state
        store.verify_goal(
            "g1",
            evidence=tuple(fact.evidence),
            base_revision=state.revision,
            plan_version=state.plan_version,
            source_branch="root",
            completed_object_ids=(target_id,),
            note=f"local judge {judge_result.verdict}",
        )
        verified = True
    state_after = store.state.to_dict()
    reloaded = StateStore(runtime, run_id)
    reloaded.load()
    reloaded_equal = reloaded.state.to_dict() == state_after

    return ControlCaseRun(
        prepared=prepared,
        target=target,
        target_id=target_id,
        list_url=list_url,
        task=task,
        run_id=run_id,
        store=store,
        session=session,
        coordinator=coordinator,
        source=source,
        model=model,
        runner=runner,
        step=step,
        events=events,
        control_events=control_events,
        judge_result=judge_result,
        verified=verified,
        state_before_judge=state_before,
        state_after=state_after,
        fact_fields=fact_fields,
        reloaded_equal=reloaded_equal,
    )


def run_l06(
    runtime_dir: str | Path,
    *,
    seed: int = 101,
    auth_kind: str = "login",
    run_id: str | None = None,
    clock: Clock | None = None,
) -> ControlCaseRun:
    from webagent.testing.cases import load_case

    draft = str(load_case("L06").task_params["l06_draft_text"])

    def factory(target: dict[str, Any], list_url: str) -> ScriptedModel:
        return L06PlanModel(
            target_name=str(target["name"]),
            list_url=list_url,
            draft_text=draft,
        )

    return _run(
        "L06",
        seed,
        runtime_dir=runtime_dir,
        prefix="l06",
        model_factory=factory,
        user_action=lambda backend: user_complete_auth(backend, kind=auth_kind),
        auth_kind=auth_kind,
        run_id=run_id,
        clock=clock,
    )


def run_l07(
    runtime_dir: str | Path,
    *,
    seed: int = 303,
    run_id: str | None = None,
    clock: Clock | None = None,
) -> ControlCaseRun:
    from webagent.testing.cases import load_case

    draft = str(load_case("L07").task_params["l07_draft_text"])

    def factory(target: dict[str, Any], list_url: str) -> ScriptedModel:
        return L07PlanModel(
            target_name=str(target["name"]),
            list_url=list_url,
            draft_text=draft,
        )

    return _run(
        "L07",
        seed,
        runtime_dir=runtime_dir,
        prefix="l07",
        model_factory=factory,
        user_action=user_save_draft_and_open_products,
        pause_before_dispatch=True,
        run_id=run_id,
        clock=clock,
    )


__all__ = [
    "ControlCaseRun",
    "L06PlanModel",
    "L07PlanModel",
    "UserSimulatingSource",
    "control_product_recorder",
    "judge_control_case",
    "ledger_evidence",
    "public_control_target",
    "run_l06",
    "run_l07",
    "user_complete_auth",
    "user_save_draft_and_open_products",
]
