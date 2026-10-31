"""F-wave acceptance: parallel read-only branches, shared writes, rate, budget.

The tests never use the network or a real model: branches talk to a temporary
local HTTP site started by the fixture and the model is an offline fake gateway
that records its own concurrency. The coordinator's timeline is used as the
overlap evidence, so "parallel" here means two independent page waits really
were in flight at the same time, not that a Tab was switched back and forth.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest
import yaml

from webagent.clock import FakeClock
from webagent.config import (
    AppConfig,
    BudgetConfig,
    ConfigError,
    LimitsConfig,
    ModelConfig,
    OfficialSettings,
    ParallelConfig,
    load_config,
    parse_config,
)
from webagent.official_agent import OfficialAgent
from webagent.protocol import ActionOp, ElementRef, Observation, WebAction, new_action_id
from webagent.errors import ActionRejected, RejectReason
from webagent.budget import BudgetLimits
from webagent.scheduling import (
    CONTROL_HOOK_EVENTS,
    BranchContext,
    BranchSpec,
    ControlHookEvent,
    RateLimited,
    SharedRateLimiter,
    TabLease,
    TabPool,
    TabUnavailable,
    TaskCoordinator,
    WaitCancelled,
)
from webagent.tools import tool_specs

CONFIGS = Path(__file__).parents[2] / "configs"


# --------------------------------------------------------------------------- site
class _SiteState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.counter = 0
        self.limited_hits = 0
        self.active_counter_writes = 0
        self.max_counter_writes = 0

    def record(self, path: str, method: str, started: float, finished: float, status: int) -> None:
        with self.lock:
            self.requests.append(
                {
                    "path": path,
                    "method": method,
                    "started_at": started,
                    "finished_at": finished,
                    "status": status,
                }
            )

    def starts(self, path: str) -> list[float]:
        with self.lock:
            return [item["started_at"] for item in self.requests if item["path"] == path]


class _SiteHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the test server
        return

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        self._handle("POST")

    def _handle(self, method: str) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        state: _SiteState = self.server.state  # type: ignore[attr-defined]
        started = time.monotonic()
        status = 200
        headers: dict[str, str] = {}
        payload: dict[str, Any]
        try:
            if parsed.path == "/read":
                time.sleep(float(query.get("delay", ["0"])[0]))
                payload = {"value": query.get("key", [""])[0]}
            elif parsed.path == "/limited":
                with state.lock:
                    state.limited_hits += 1
                    hit = state.limited_hits
                if hit == 1:
                    status = 429
                    headers["Retry-After"] = query.get("retry_after", ["0.3"])[0]
                    payload = {"error": "rate limited"}
                else:
                    payload = {"ok": True}
            elif parsed.path == "/counter" and method == "GET":
                with state.lock:
                    payload = {"count": state.counter}
            elif parsed.path == "/counter" and method == "POST":
                delta = int(query.get("delta", ["1"])[0])
                # Deliberately non-atomic read-modify-write: overlapping writes
                # lose an update, so serialization is what keeps the final value.
                with state.lock:
                    state.active_counter_writes += 1
                    state.max_counter_writes = max(
                        state.max_counter_writes, state.active_counter_writes
                    )
                    current = state.counter
                try:
                    time.sleep(0.08)
                    with state.lock:
                        state.counter = current + delta
                        new_value = state.counter
                finally:
                    with state.lock:
                        state.active_counter_writes -= 1
                payload = {"count": new_value}
            elif parsed.path == "/ping":
                payload = {"ok": True}
            else:
                status = 404
                payload = {"error": "not found"}
        finally:
            state.record(parsed.path, method, started, time.monotonic(), status)
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


@dataclass(frozen=True)
class _Response:
    status: int
    headers: dict[str, str]
    payload: dict[str, Any]


class LocalSite:
    def __init__(self, server: ThreadingHTTPServer, state: _SiteState) -> None:
        self.server = server
        self.state = state
        self.base = f"http://127.0.0.1:{server.server_address[1]}"

    def fetch(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        method: str = "GET",
        timeout: float = 5.0,
    ) -> _Response:
        query = urlencode(params or {})
        url = f"{self.base}{path}" + (f"?{query}" if query else "")
        request = Request(url, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local fixture only
                body = response.read()
                return _Response(
                    status=response.status,
                    headers=dict(response.headers),
                    payload=json.loads(body or b"{}"),
                )
        except HTTPError as exc:
            body = exc.read()
            return _Response(
                status=exc.code,
                headers=dict(exc.headers),
                payload=json.loads(body) if body else {},
            )


@pytest.fixture()
def site() -> LocalSite:
    state = _SiteState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    server.state = state  # type: ignore[attr-defined]
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield LocalSite(server, state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# --------------------------------------------------------------------------- model
class FakeModelGateway:
    """Offline model double recording call concurrency and per-call delay."""

    def __init__(self, *, delay: float = 0.04) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0
        self.calls: list[str] = []

    def generate(self, prompt: str = "", *, delay: float | None = None) -> str:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            time.sleep(self.delay if delay is None else delay)
            with self._lock:
                self.calls.append(prompt)
            return f"answer:{prompt}"
        finally:
            with self._lock:
                self._active -= 1


# --------------------------------------------------------------------------- helpers
def make_coordinator(
    tmp_path: Path,
    *,
    enabled: bool = True,
    capacity: int = 2,
    rate: float = 20.0,
    limits: BudgetLimits | None = None,
    root_run_id: str | None = None,
    max_rate_retries: int = 3,
    backoff: tuple[float, ...] = (0.2, 0.3),
) -> TaskCoordinator:
    settings = ParallelConfig(
        enabled=enabled,
        capacity=capacity,
        site_ops_per_second=rate,
        max_rate_retries=max_rate_retries,
        backoff_seconds=backoff,
    )
    return TaskCoordinator(
        settings,
        runtime_dir=tmp_path / "runtime",
        root_run_id=root_run_id or f"f-root-{time.monotonic_ns()}",
        limits=limits,
    )


def fake_config(**parallel: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile": "fake",
        "backend": "fake",
        "model": {"provider": "fake", "name": "fake-scripted", "context_length": 32768},
    }
    if parallel:
        payload["parallel"] = parallel
    return payload


def read_plan() -> list[BranchSpec]:
    return [BranchSpec("read-a"), BranchSpec("read-b")]


def observation_for(ctx: BranchContext) -> Observation:
    observation_id = f"obs-{ctx.branch_id}"
    return Observation(
        run_id=ctx.run_id,
        page_id=ctx.page_id,
        observation_id=observation_id,
        url="http://fixture.local/page",
        ax_text=f"{ctx.branch_id} page",
        elements={
            "1": ElementRef(
                page_id=ctx.page_id,
                observation_id=observation_id,
                number="1",
                role="button",
                name="Read",
            )
        },
    )


# ------------------------------------------------------------------- F01 / F08 / F02
def test_capacity_two_readonly_branches_have_real_wait_overlap(
    site: LocalSite, tmp_path: Path
) -> None:
    gateway = FakeModelGateway(delay=0.05)
    coordinator = make_coordinator(tmp_path, capacity=2, root_run_id="f-cap2")
    barrier = threading.Barrier(2)

    def work(ctx: BranchContext) -> str:
        ctx.model_call(lambda: gateway.generate(ctx.branch_id))
        # Both independent page waits are entered together; the interval overlap
        # below is then a property of the coordinator, not of thread start jitter.
        barrier.wait(timeout=5)
        return ctx.page_action(
            lambda: site.fetch("/read", params={"delay": 0.35, "key": ctx.branch_id}).payload[
                "value"
            ]
        )

    report = coordinator.run(read_plan(), work)

    assert report.mode == "parallel" and report.execution == "parallel"
    assert report.capacity == 2 and report.max_active == 2
    assert report.statuses == {"read-a": "succeeded", "read-b": "succeeded"}
    assert [branch.output for branch in report.branches] == ["read-a", "read-b"]

    first, second = report.branches
    overlap = min(first.finished_at, second.finished_at) - max(  # type: ignore[operator]
        first.started_at, second.started_at  # type: ignore[arg-type]
    )
    assert overlap > 0.2
    assert report.overlap_seconds > 0.2

    # A branch owns its loop, its Tab and its own evidence record.
    assert len({branch.page_id for branch in report.branches}) == 2
    assert len({branch.run_id for branch in report.branches}) == 2
    for branch in report.branches:
        assert (tmp_path / "runtime" / "runs" / str(branch.run_id) / "events.jsonl").is_file()
    assert gateway.max_concurrent == 1
    print(
        "F01 capacity=2 timeline:",
        [
            (branch.branch_id, round(branch.started_at - report.started_at, 3),
             round(branch.finished_at - report.started_at, 3))
            for branch in report.branches
        ],
        "overlap=", round(overlap, 3),
    )


def test_capacity_four_queues_extra_ready_tasks_and_never_exceeds_capacity(
    site: LocalSite, tmp_path: Path
) -> None:
    coordinator = make_coordinator(tmp_path, capacity=4, root_run_id="f-cap4")
    specs = [BranchSpec(f"read-{index}") for index in range(6)]

    def work(ctx: BranchContext) -> str:
        time.sleep(0.3)
        return ctx.page_action(lambda: site.fetch("/read", params={"delay": 0.5}).payload["value"])

    report = coordinator.run(specs, work)

    assert report.execution == "parallel" and report.capacity == 4
    assert report.max_active == 4  # peak never exceeds the configured capacity
    print(
        "F08 capacity=4 branches:",
        [(branch.branch_id, branch.status) for branch in report.branches],
    )
    assert all(branch.status == "succeeded" for branch in report.branches)
    assert len(report.branches) == 6
    assert report.max_queue_wait_seconds > 0.2
    assert len({branch.page_id for branch in report.branches}) == 4
    queued = [branch for branch in report.branches if branch.started_at is not None]
    earliest_finish = min(
        branch.finished_at for branch in report.branches if branch.finished_at is not None
    )
    started_behind_queue = [
        branch
        for branch in queued
        if branch.started_at is not None and branch.started_at >= earliest_finish - 0.05
    ]
    assert len(started_behind_queue) >= 2
    print(
        "F08 capacity=4 timeline:",
        [
            (branch.branch_id, round(branch.started_at - report.started_at, 3),
             round(branch.finished_at - report.started_at, 3))
            for branch in report.branches
        ],
    )


def test_branch_element_ownership_does_not_cross_tabs(tmp_path: Path) -> None:
    coordinator = make_coordinator(tmp_path, capacity=2, root_run_id="f-owner")
    barrier = threading.Barrier(2)
    rejected: list[str] = []
    accepted: dict[str, str] = {}

    def work(ctx: BranchContext) -> str:
        session = ctx.session
        session.start_task(f"read {ctx.branch_id}")
        observation = observation_for(ctx)
        session.ingest_observation(observation)
        barrier.wait(timeout=5)
        other_page = "tab-2" if ctx.page_id == "tab-1" else "tab-1"
        cross = WebAction(
            new_action_id(),
            ctx.run_id,
            other_page,
            observation.observation_id,
            ActionOp.CLICK,
            {"target": "1"},
        )
        try:
            session.propose(cross)
        except ActionRejected as exc:
            assert exc.reason is RejectReason.WRONG_PAGE
            rejected.append(ctx.branch_id)
        own = WebAction(
            new_action_id(), ctx.run_id, ctx.page_id, observation.observation_id, ActionOp.CLICK, {"target": "1"}
        )
        session.propose(own)
        session.consume()
        accepted[ctx.branch_id] = ctx.page_id
        return ctx.page_id

    report = coordinator.run(read_plan(), work)

    assert report.execution == "parallel"
    assert sorted(rejected) == ["read-a", "read-b"]
    assert set(accepted) == {"read-a", "read-b"}
    assert len(set(accepted.values())) == 2
    assert [branch.status for branch in report.branches] == ["succeeded", "succeeded"]
    for branch in report.branches:
        events = (tmp_path / "runtime" / "runs" / str(branch.run_id) / "events.jsonl").read_text(
            "utf-8"
        )
        assert branch.page_id in events
        assert f"run_id\":\"{branch.run_id}" in events


def test_tab_lease_allows_only_one_in_flight_action() -> None:
    lease = TabLease("tab-1", "read-a")
    lock = threading.Lock()
    active = 0
    peak = 0

    def act() -> None:
        nonlocal active, peak
        with lease.action_slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.1)
            with lock:
                active -= 1

    threads = [threading.Thread(target=act) for _ in range(2)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    elapsed = time.monotonic() - started

    assert peak == 1
    assert lease.max_in_flight == 1
    assert lease.action_count == 2
    assert elapsed >= 0.18


def test_tab_pool_refuses_a_tab_owned_by_another_branch() -> None:
    pool = TabPool(2)
    lease = pool.acquire("read-a")
    with pytest.raises(TabUnavailable):
        pool.acquire("read-b", requested_page_id=lease.page_id)
    pool.release(lease)
    other = pool.acquire("read-b", requested_page_id=lease.page_id)
    assert other.page_id == lease.page_id and other.branch_id == "read-b"
    pool.release(other)


# ------------------------------------------------------------------- F02 shared write
def test_shared_write_plan_degrades_to_serial_without_lost_updates(
    site: LocalSite, tmp_path: Path
) -> None:
    coordinator = make_coordinator(tmp_path, capacity=2, root_run_id="f-write")

    def work(ctx: BranchContext) -> int:
        response = ctx.page_action(
            lambda: site.fetch("/counter", params={"delta": 1}, method="POST"),
            write_target="counter",
        )
        return response.payload["count"]

    specs = [
        BranchSpec("write-a", read_only=False, write_targets=("counter",)),
        BranchSpec("write-b", read_only=False, write_targets=("counter",)),
    ]
    report = coordinator.run(specs, work)

    assert report.mode == "parallel"
    assert report.execution == "serial"
    assert report.degraded_reason == "shared_write_serialization"
    assert report.capacity == 1 and report.max_active == 1
    assert report.overlap_seconds == 0.0
    assert [branch.status for branch in report.branches] == ["succeeded", "succeeded"]
    assert site.state.counter == 2  # both increments survived
    assert site.state.max_counter_writes == 1  # the non-atomic endpoint saw no race
    assert len(report.lock_waits) == 2
    print(
        "F02 shared-write timeline:",
        [(branch.branch_id, branch.started_at, branch.finished_at) for branch in report.branches],
        "counter=", site.state.counter,
    )


# ------------------------------------------------------------------------ F04 rate
def _limited_fetch(site: LocalSite) -> int:
    response = site.fetch("/limited", params={"retry_after": "0.35"})
    if response.status == 429:
        header = response.headers.get("Retry-After")
        raise RateLimited("429 from the local site", retry_after=float(header) if header else None)
    return response.status


def test_cooldown_triggered_by_one_branch_backs_off_every_branch(
    site: LocalSite, tmp_path: Path
) -> None:
    coordinator = make_coordinator(
        tmp_path, capacity=2, rate=4.0, root_run_id="f-rate", max_rate_retries=1
    )

    def work(ctx: BranchContext) -> int:
        if ctx.branch_id == "read-a":
            ctx.page_action(lambda: _limited_fetch(site))
            ctx.page_action(lambda: site.fetch("/read", params={"delay": 0.02}))
            return 1
        ctx.wait_cancellable(0.1)
        ctx.page_action(lambda: site.fetch("/read", params={"delay": 0.02}))
        ctx.page_action(lambda: site.fetch("/read", params={"delay": 0.02}))
        return 2

    report = coordinator.run(read_plan(), work)

    assert [branch.status for branch in report.branches] == ["succeeded", "succeeded"]
    cooldown_end = coordinator.limiter.cooldown_until
    assert cooldown_end > report.started_at
    waits = [wait for wait in report.rate_waits if wait.kind == "cooldown"]
    assert {wait.branch_id for wait in waits} == {"read-a", "read-b"}
    assert all(wait.cancelled is False for wait in report.rate_waits)
    # Both branches resumed after the shared cooldown and then kept the shared
    # pacing, so recovery carries no burst of requests.
    after = sorted(
        started
        for started in site.state.starts("/read") + site.state.starts("/limited")
        if started >= cooldown_end - 0.02
    )
    assert len(after) >= 3
    gaps = [after[index + 1] - after[index] for index in range(len(after) - 1)]
    assert min(gaps) >= 0.18  # one shared schedule, interval 0.25s
    print(
        "F04 cooldown waits:",
        [(wait.branch_id, round(wait.seconds, 3)) for wait in report.rate_waits],
        "post-recovery gaps=", [round(gap, 3) for gap in gaps],
    )


def test_rate_wait_is_cancellable_and_ledger_keeps_cancelled_calls(
    site: LocalSite, tmp_path: Path
) -> None:
    coordinator = make_coordinator(
        tmp_path, capacity=2, rate=4.0, root_run_id="f-cancel", max_rate_retries=3
    )
    gateway = FakeModelGateway(delay=0.02)
    holder: dict[str, Any] = {}

    def work(ctx: BranchContext) -> int:
        ctx.model_call(lambda: gateway.generate(ctx.branch_id))
        if ctx.branch_id == "read-a":
            ctx.page_action(lambda: _limited_fetch_with_delay(site, 5.0))
        else:
            ctx.wait_cancellable(0.1)
            ctx.page_action(lambda: site.fetch("/read", params={"delay": 0.02}))
        return 0

    def run() -> None:
        holder["report"] = coordinator.run(read_plan() + [BranchSpec("read-c")], work)

    thread = threading.Thread(target=run, daemon=True)
    started = time.monotonic()
    thread.start()
    time.sleep(0.4)
    coordinator.cancel_all("user cancelled the run")
    thread.join(timeout=5)
    elapsed = time.monotonic() - started

    assert not thread.is_alive()
    assert elapsed < 2.0  # a 5s cooldown wait must not block cancellation
    report = holder["report"]
    assert report.statuses == {
        "read-a": "cancelled",
        "read-b": "cancelled",
        "read-c": "abandoned",
    }
    # Discarded queued work is recorded without inventing consumption; the two
    # branches that did call the model stay charged.
    assert report.ledger["total_calls"] == 2
    assert set(report.ledger["by_branch"]) == {"read-a", "read-b"}
    assert report.rate_waits
    assert all(wait.cancelled for wait in report.rate_waits)
    assert all(wait.kind == "cooldown" for wait in report.rate_waits)


def _limited_fetch_with_delay(site: LocalSite, retry_after: float) -> int:
    response = site.fetch("/limited", params={"retry_after": str(retry_after)})
    if response.status == 429:
        raise RateLimited("429 from the local site", retry_after=retry_after)
    return response.status


# ------------------------------------------------------------- F05 model and budget
def test_single_model_slot_and_ledger_charges_failed_and_cancelled_branches(
    tmp_path: Path,
) -> None:
    coordinator = make_coordinator(tmp_path, capacity=3, root_run_id="f-ledger")
    gateway = FakeModelGateway(delay=0.1)
    holder: dict[str, Any] = {}

    def work(ctx: BranchContext) -> str:
        ctx.model_call(lambda: gateway.generate(ctx.branch_id))
        if ctx.branch_id == "branch-fail":
            raise RuntimeError("branch failed after its model call")
        if ctx.branch_id == "branch-cancel":
            ctx.wait_cancellable(10.0)
        return ctx.branch_id

    specs = [BranchSpec("branch-ok"), BranchSpec("branch-fail"), BranchSpec("branch-cancel")]

    def run() -> None:
        holder["report"] = coordinator.run(specs, work)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if len(coordinator.ledger.snapshot()["records"]) == 3:
            break
        time.sleep(0.02)
    coordinator.cancel_branch("branch-cancel", "test cancellation")
    thread.join(timeout=5)

    report = holder["report"]
    assert report.statuses == {
        "branch-ok": "succeeded",
        "branch-fail": "failed",
        "branch-cancel": "cancelled",
    }
    assert gateway.max_concurrent == 1
    assert report.ledger["max_concurrent"] == 1
    assert report.ledger["total_calls"] == 3
    assert report.ledger["by_branch"] == {
        "branch-ok": 1,
        "branch-fail": 1,
        "branch-cancel": 1,
    }
    failed_records = [
        record for record in report.ledger["records"] if record["branch_id"] == "branch-fail"
    ]
    # The model call itself completed; the branch failed afterwards, and the
    # charge stays in the ledger instead of being recomputed from successes.
    assert failed_records and failed_records[0]["outcome"] == "ok"
    assert report.ledger["refused_claims"] == 0


def test_root_model_budget_is_shared_and_refusals_are_recorded(tmp_path: Path) -> None:
    gateway = FakeModelGateway(delay=0.02)
    coordinator = make_coordinator(
        tmp_path,
        capacity=2,
        root_run_id="f-budget",
        limits=BudgetLimits(max_model_calls=2),
    )

    def work(ctx: BranchContext) -> str:
        ctx.model_call(lambda: gateway.generate(ctx.branch_id))
        return ctx.branch_id

    report = coordinator.run(
        [BranchSpec("budget-a"), BranchSpec("budget-b"), BranchSpec("budget-c")], work
    )

    assert report.ledger["total_calls"] == 2
    assert report.ledger["refused_claims"] == 1
    assert list(report.statuses.values()).count("succeeded") == 2
    failed = [branch for branch in report.branches if branch.status == "failed"]
    assert len(failed) == 1 and failed[0].error_type == "BudgetExceeded"


# -------------------------------------------------------- F06 official degradation
def test_official_profile_rejects_parallel_and_runs_the_same_plan_serially(
    site: LocalSite, tmp_path: Path
) -> None:
    official = load_config(CONFIGS / "official.yaml")
    assert official.parallel.enabled is False
    assert official.parallel.capacity == 1
    assert official.summary()["parallel"]["enabled"] is False

    raw = yaml.safe_load((CONFIGS / "official.yaml").read_text(encoding="utf-8"))
    raw["parallel"]["enabled"] = True
    with pytest.raises(ConfigError):
        parse_config(raw)

    # The official tool catalog has no Tab or parallel tool at all.
    names = {spec.name for spec in tool_specs()}
    assert not any(
        token in name.lower() for name in names for token in ("tab", "parallel", "switch")
    )

    official_coordinator = TaskCoordinator.from_config(
        official, runtime_dir=tmp_path / "official-runtime", root_run_id="official-run"
    )
    assert official_coordinator.mode == "serial"
    assert official_coordinator.supports_tabs is False
    assert official_coordinator.capacity == 1

    def work(ctx: BranchContext) -> str:
        return ctx.page_action(
            lambda: site.fetch("/read", params={"delay": 0.05, "key": ctx.branch_id}).payload[
                "value"
            ]
        )

    serial = official_coordinator.run(read_plan(), work)
    assert serial.execution == "serial"
    assert serial.max_active == 1 and serial.overlap_seconds == 0.0
    assert [branch.page_id for branch in serial.branches] == ["tab-1", "tab-1"]
    assert [branch.output for branch in serial.branches] == ["read-a", "read-b"]

    parallel = make_coordinator(tmp_path, capacity=2, root_run_id="f-again")
    parallel_report = parallel.run(read_plan(), work)
    assert {branch.output for branch in parallel_report.branches} == {
        branch.output for branch in serial.branches
    }
    assert parallel_report.max_active == 2


def test_official_final_request_text_carries_no_parallel_or_tab_trace(tmp_path: Path) -> None:
    class FakeGateway:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def generate(self, messages: list[dict[str, str]]) -> Any:
            self.calls.append(messages)
            if len(self.calls) == 1:
                import types

                return types.SimpleNamespace(
                    text='Thought: one action.\nAction: web_click\nAction Input: {"target": "2"}',
                    request_id="req-1",
                )
            import types

            return types.SimpleNamespace(
                text="Thought: done.\nFinal Answer: done", request_id="req-2"
            )

    config = AppConfig(
        profile="official",
        backend="official",
        runtime_dir=str(tmp_path / "runtime"),
        model=ModelConfig(
            provider="official",
            name="platform-unified",
            context_length=32768,
            tokenizer="heuristic",
            token_count_source="estimated",
        ),
        budget=BudgetConfig(deadline_margin_seconds=0.0),
        limits=LimitsConfig(max_messages=256, max_request_bytes=1024 * 1024),
        official=OfficialSettings(),
    )
    clock = FakeClock()
    gateway = FakeGateway()
    agent = OfficialAgent(
        app_config=config,
        runtime_dir=tmp_path / "runtime",
        clock=clock,
        model_api=gateway,
        download_artifacts=frozenset(),
    )
    try:
        agent.reset({"run_id": "official-parallel-check", "execution_deadline": clock.now() + 120})
        first = agent.step(
            {
                "protocol_version": "1.0",
                "run_id": "official-parallel-check",
                "step": 0,
                "task": "click search then finish",
                "observation": {
                    "type": "accessibility_tree",
                    "url": "http://forum.test/search",
                    "text": "[1] textbox 'Query'\n[2] button 'Search'",
                },
                "history": [],
                "action_history": [],
                "execution_deadline": clock.now() + 120,
            }
        )
        second = agent.step(
            {
                "protocol_version": "1.0",
                "run_id": "official-parallel-check",
                "step": 1,
                "observation": {
                    "type": "accessibility_tree",
                    "url": "http://forum.test/result",
                    "text": "[1] link 'Done'",
                },
                "history": [],
                "action_history": [],
                "execution_deadline": clock.now() + 120,
            }
        )
    finally:
        agent.close()

    assert first == {"action": "click [2]"}
    assert second["action"].startswith("stop")
    assert len(gateway.calls) == 2
    sent = "\n".join(
        message.get("content", "")
        for call in gateway.calls
        for message in call
        if isinstance(message, dict)
    ).lower()
    for forbidden in ("web_tab", "open_tab", "switch_tab", "parallel", "branch_id"):
        assert forbidden not in sent


# --------------------------------------------------- serial equivalence / config
def test_unconfigured_parallel_keeps_serial_behavior(site: LocalSite, tmp_path: Path) -> None:
    raw = fake_config()
    config = parse_config(raw)
    assert config.parallel.enabled is False
    assert config.parallel.capacity == 2  # documented default when not configured

    coordinator = TaskCoordinator.from_config(
        config, runtime_dir=tmp_path / "runtime", root_run_id="f-serial"
    )
    assert coordinator.mode == "serial" and coordinator.capacity == 1

    def work(ctx: BranchContext) -> str:
        return ctx.page_action(
            lambda: site.fetch("/read", params={"delay": 0.1, "key": ctx.branch_id}).payload["value"]
        )

    specs = [BranchSpec("read-a"), BranchSpec("read-b"), BranchSpec("read-c")]
    report = coordinator.run(specs, work)

    assert report.mode == "serial" and report.execution == "serial"
    assert report.capacity == 1 and report.max_active == 1
    assert report.overlap_seconds == 0.0
    assert [branch.page_id for branch in report.branches] == ["tab-1", "tab-1", "tab-1"]
    assert [branch.output for branch in report.branches] == ["read-a", "read-b", "read-c"]
    started = [branch.started_at for branch in report.branches]
    assert started == sorted(started)  # plan order, no interleaving

    parallel = make_coordinator(tmp_path, capacity=3, root_run_id="f-serial-par")
    parallel_report = parallel.run(specs, work)
    assert {branch.branch_id: branch.output for branch in parallel_report.branches} == {
        branch.branch_id: branch.output for branch in report.branches
    }
    assert parallel_report.max_active == 3


def test_invalid_capacity_and_parallel_values_are_config_errors() -> None:
    for capacity in (0, -1, 1.5, "2", True):
        with pytest.raises(ConfigError):
            parse_config(fake_config(enabled=True, capacity=capacity))

    assert parse_config(fake_config(enabled=True)).parallel.capacity == 2
    with pytest.raises(ConfigError):
        parse_config(fake_config(enabled=True, capacity=2, model_concurrency=2))
    with pytest.raises(ConfigError):
        parse_config(fake_config(enabled=True, site_ops_per_second=0))
    with pytest.raises(ConfigError):
        parse_config(fake_config(enabled=True, not_a_key=True))

    serial = parse_config(fake_config())
    assert serial.parallel.enabled is False
    assert serial.summary()["parallel"]["enabled"] is False


def test_shared_rate_limiter_paces_one_token_and_shares_cooldown() -> None:
    limiter = SharedRateLimiter(interval_seconds=0.1)
    first = limiter.acquire("a")
    started = time.monotonic()
    limiter.note_retry_after(0.15)
    wait = limiter.acquire("b")
    elapsed = time.monotonic() - started
    assert first is None
    assert wait is not None and wait.kind == "cooldown"
    assert elapsed >= 0.14

    class Clock:
        def __init__(self) -> None:
            self.value = 100.0

        def now(self) -> float:
            return self.value

        def monotonic(self) -> float:
            return self.value

        def iso(self, epoch_seconds: float) -> str:
            return ""

    clock = Clock()
    limiter = SharedRateLimiter(interval_seconds=1.0, clock=clock)  # type: ignore[arg-type]
    assert limiter.acquire("a") is None
    clock.value += 0.2
    event = threading.Event()
    event.set()
    with pytest.raises(WaitCancelled):
        limiter.acquire("b", cancel_event=event)


def test_f07_control_hook_surface_is_declared_but_not_implemented(tmp_path: Path) -> None:
    assert CONTROL_HOOK_EVENTS == (
        "pause_requested",
        "resume_requested",
        "drain_requested",
    )
    coordinator = make_coordinator(tmp_path, enabled=False, root_run_id="f-hook")
    seen: list[ControlHookEvent] = []
    coordinator.set_control_hook(seen.append)
    for call in (
        lambda: coordinator.pause("user"),
        coordinator.resume,
        lambda: coordinator.drain("user"),
    ):
        with pytest.raises(NotImplementedError):
            call()
    assert [event.kind for event in seen] == list(CONTROL_HOOK_EVENTS)
