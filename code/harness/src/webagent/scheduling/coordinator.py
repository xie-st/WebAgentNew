"""F-wave scheduling skeleton: serial default, read-only branches in parallel.

Scope of this first release (see the F-batch acceptance table):

* only independent read-only tasks may run in parallel; one branch owns one Tab
  and one loop, and the Tab sees at most one in-flight action;
* a plan that declares any shared write degrades the whole run to serial
  execution, so a shared write cannot race and cannot lose an update;
* every branch shares one site schedule and one model-call gate, and all claims
  land in one root ledger including failed, cancelled and discarded branches;
* the official profile never exposes Tabs or parallel execution and always runs
  one action / one model call at a time.

The coordinator only adds coordination. It does not implement a second plan, a
second loop or a second retry stack: each branch is the owner of its own
interaction loop and receives a :class:`BranchContext` with the scheduling
primitives (Tab slot, shared rate schedule, write lock, root model slot).

F07 (global pause/drain while branches, lock waits and rate waits are active)
is deliberately a hook only: ``pause``, ``resume`` and ``drain`` exist with the
documented signatures and the :class:`ControlHookEvent` type, and each raises
``NotImplementedError`` until the R-wave control layer is merged.
"""

from __future__ import annotations

import queue
import re
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..budget import BudgetLimits, Deadline
from ..clock import Clock, SystemClock
from ..config import AppConfig, ParallelConfig
from ..errors import BudgetExceeded, DeadlineExceeded
from ..protocol import validate_run_id
from ..redaction import DEFAULT_REDACTOR
from ..session import Session
from .errors import CoordinatorError, RateLimited, SchedulingError, TabUnavailable, WaitCancelled
from .ledger import RootBudgetLedger
from .limiter import RateWait, SharedRateLimiter
from .tabs import TabLease, TabPool

F07_NOT_IMPLEMENTED = (
    "F07 global pause/drain is not implemented in this release: it depends on the "
    "R-wave control layer (pause, takeover and resume) and will be completed after "
    "that layer is merged. The coordinator exposes the hook signature and event type only."
)

CONTROL_HOOK_EVENTS: tuple[str, ...] = (
    "pause_requested",
    "resume_requested",
    "drain_requested",
)


@dataclass(frozen=True)
class ControlHookEvent:
    """Event shape the R wave will receive once global pause/drain is implemented."""

    kind: str
    reason: str | None
    at: float
    active_branches: int
    waiting_branches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "at": self.at,
            "active_branches": self.active_branches,
            "waiting_branches": self.waiting_branches,
        }


@dataclass(frozen=True)
class BranchSpec:
    """One independent unit the coordinator may schedule.

    ``read_only`` and ``write_targets`` are the resource declaration used for
    the first-release safety rule: any declared write degrades the whole run to
    serial execution. ``page_id`` optionally requests a specific Tab; a Tab that
    is owned by another branch is refused rather than stolen.
    """

    branch_id: str
    task: str = ""
    read_only: bool = True
    write_targets: tuple[str, ...] = ()
    page_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, str) or not self.branch_id.strip():
            raise SchedulingError("branch_id must be a non-empty string")
        if not isinstance(self.task, str):
            raise SchedulingError("branch task must be a string")
        if not isinstance(self.read_only, bool):
            raise SchedulingError("branch read_only must be a boolean")
        targets = self.write_targets
        if isinstance(targets, str):
            raise SchedulingError("branch write_targets must be a sequence of strings")
        if any(not isinstance(target, str) or not target.strip() for target in targets):
            raise SchedulingError("branch write_targets must be non-empty strings")
        if self.page_id is not None and (
            not isinstance(self.page_id, str) or not self.page_id.strip()
        ):
            raise SchedulingError("branch page_id must be a non-empty string or null")

    @property
    def requires_serial(self) -> bool:
        return (not self.read_only) or bool(self.write_targets)


@dataclass(frozen=True)
class LockWait:
    branch_id: str
    target: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {"branch_id": self.branch_id, "target": self.target, "seconds": self.seconds}


@dataclass(frozen=True)
class BranchReport:
    branch_id: str
    task: str
    status: str  # succeeded | failed | cancelled | abandoned
    output: Any
    error_type: str | None
    error_message: str | None
    page_id: str | None
    run_id: str | None
    read_only: bool
    write_targets: tuple[str, ...]
    queued_at: float
    started_at: float | None
    finished_at: float | None
    duration_seconds: float | None
    model_calls: int
    page_actions: int
    rate_retries: int
    lock_waits: tuple[LockWait, ...]
    rate_waits: tuple[RateWait, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "task": self.task,
            "status": self.status,
            "output": self.output,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "page_id": self.page_id,
            "run_id": self.run_id,
            "read_only": self.read_only,
            "write_targets": list(self.write_targets),
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "model_calls": self.model_calls,
            "page_actions": self.page_actions,
            "rate_retries": self.rate_retries,
            "lock_waits": [wait.to_dict() for wait in self.lock_waits],
            "rate_waits": [wait.to_dict() for wait in self.rate_waits],
        }


def _peak_concurrency(intervals: Sequence[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    # At the same instant releases happen before starts, so touching intervals
    # (end == start) are not counted as overlapping.
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _max_pair_overlap(intervals: Sequence[tuple[float, float]]) -> float:
    best = 0.0
    for index, (start, end) in enumerate(intervals):
        for other_start, other_end in intervals[index + 1 :]:
            overlap = min(end, other_end) - max(start, other_start)
            if overlap > best:
                best = overlap
    return best


@dataclass(frozen=True)
class CoordinatorReport:
    root_run_id: str
    mode: str  # configured: serial | parallel
    execution: str  # actual: serial | parallel
    requested_capacity: int
    capacity: int
    degraded_reason: str | None
    started_at: float
    finished_at: float
    duration_seconds: float
    branches: tuple[BranchReport, ...]
    ledger: Mapping[str, Any]
    rate_waits: tuple[RateWait, ...]
    lock_waits: tuple[LockWait, ...]
    max_active: int
    overlap_seconds: float
    max_queue_wait_seconds: float

    @property
    def statuses(self) -> dict[str, str]:
        return {branch.branch_id: branch.status for branch in self.branches}

    def timeline(self) -> list[dict[str, Any]]:
        return [
            {
                "branch_id": branch.branch_id,
                "page_id": branch.page_id,
                "status": branch.status,
                "started_at": branch.started_at,
                "finished_at": branch.finished_at,
            }
            for branch in self.branches
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "mode": self.mode,
            "execution": self.execution,
            "requested_capacity": self.requested_capacity,
            "capacity": self.capacity,
            "degraded_reason": self.degraded_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "branches": [branch.to_dict() for branch in self.branches],
            "ledger": dict(self.ledger),
            "rate_waits": [wait.to_dict() for wait in self.rate_waits],
            "lock_waits": [wait.to_dict() for wait in self.lock_waits],
            "max_active": self.max_active,
            "overlap_seconds": self.overlap_seconds,
            "max_queue_wait_seconds": self.max_queue_wait_seconds,
            "timeline": self.timeline(),
        }


def _safe_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "branch"


class BranchContext:
    """Scheduling primitives handed to one branch's own interaction loop."""

    def __init__(
        self,
        coordinator: "TaskCoordinator",
        spec: BranchSpec,
        lease: TabLease,
        session: Session,
        cancel_event: threading.Event,
    ) -> None:
        self.coordinator = coordinator
        self.spec = spec
        self.lease = lease
        self.session = session
        self.cancel_event = cancel_event
        self.branch_id = spec.branch_id
        self.task = spec.task
        self.page_id = lease.page_id
        self.run_id = session.run_id
        self.model_calls = 0
        self.page_actions = 0
        self.rate_retries = 0
        self.lock_waits: list[LockWait] = []
        self.started_at: float | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def read_only(self) -> bool:
        return self.spec.read_only and not self.spec.write_targets

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise WaitCancelled(f"branch {self.branch_id} was cancelled")

    def wait_cancellable(self, seconds: float) -> None:
        """Sleep up to ``seconds`` but wake immediately on branch cancellation."""

        if seconds < 0:
            raise ValueError("wait seconds must be non-negative")
        if self.cancel_event.wait(timeout=float(seconds)):
            raise WaitCancelled(f"branch {self.branch_id} was cancelled")

    def page_action(
        self,
        action: Callable[[], Any],
        *,
        write_target: str | None = None,
    ) -> Any:
        """Run one page action under the Tab, site and (when declared) write locks.

        The action is retried when it raises :class:`RateLimited`; the cooldown is
        applied to the shared schedule so every branch backs off together. Waits
        are cancellable. The Tab holds at most this one action while the body
        runs.
        """

        if not callable(action):
            raise SchedulingError("page_action needs a callable")
        self.check_cancelled()
        target = write_target or (None if self.read_only else self.branch_id)
        needs_write_lock = target is not None
        if needs_write_lock:
            self.coordinator._acquire_write_lock(self, target)  # noqa: SLF001 - same package
        try:
            attempt = 0
            while True:
                with self.lease.action_slot(cancel_event=self.cancel_event):
                    self.coordinator.limiter.acquire(  # noqa: SLF001 - same package
                        self.branch_id, cancel_event=self.cancel_event
                    )
                    try:
                        result = action()
                    except RateLimited as exc:
                        delay = exc.retry_after
                        if delay is None:
                            backoff = self.coordinator.settings.backoff_seconds
                            delay = backoff[min(attempt, len(backoff) - 1)]
                        self.rate_retries += 1
                        self.coordinator.limiter.note_retry_after(delay)  # noqa: SLF001
                        if attempt >= self.coordinator.settings.max_rate_retries:
                            raise
                        attempt += 1
                        continue
                self.page_actions += 1
                return result
        finally:
            if needs_write_lock:
                self.coordinator._release_write_lock()  # noqa: SLF001 - same package

    def model_call(self, call: Callable[[], Any]) -> Any:
        """Run one model call through the single root slot and root ledger."""

        if not callable(call):
            raise SchedulingError("model_call needs a callable")
        self.check_cancelled()
        started = self.coordinator.clock.monotonic()
        failure_type: str | None = None
        claimed = False
        try:
            with self.coordinator.ledger.model_call(  # noqa: SLF001 - same package
                self.branch_id, cancel_event=self.cancel_event
            ):
                claimed = True
                self.model_calls += 1
                try:
                    return call()
                except BaseException as exc:
                    failure_type = type(exc).__name__
                    raise
        finally:
            if claimed:
                latency_ms = (self.coordinator.clock.monotonic() - started) * 1000.0
                try:
                    # Branch-local event evidence; the root ledger above stays the
                    # authority for accounting, including failed/cancelled calls.
                    self.session.record_model_call(
                        token_count_source="unknown",
                        latency_ms=latency_ms,
                        failure_type=failure_type,
                    )
                except Exception:  # noqa: BLE001 - event evidence must not replace the call result
                    pass


class TaskCoordinator:
    def __init__(
        self,
        settings: ParallelConfig | None = None,
        *,
        runtime_dir: str | Path = "runtime",
        root_run_id: str = "parallel",
        clock: Clock | None = None,
        deadline: Deadline | None = None,
        limits: BudgetLimits | None = None,
        profile: str = "real",
    ) -> None:
        self.settings = settings or ParallelConfig()
        self.clock = clock or SystemClock()
        self.runtime_dir = Path(runtime_dir)
        self.root_run_id = validate_run_id(str(root_run_id))
        self.deadline = deadline
        self.limits = limits
        self.profile = str(profile)
        self.supports_tabs = self.profile != "official"
        parallel_enabled = bool(self.settings.enabled) and self.supports_tabs
        self.mode = "parallel" if parallel_enabled else "serial"
        self.requested_capacity = int(self.settings.capacity)
        self.capacity = self.requested_capacity if parallel_enabled else 1
        self.limiter = SharedRateLimiter(
            interval_seconds=self.settings.interval_seconds, clock=self.clock
        )
        self.ledger = RootBudgetLedger(
            limits=limits, deadline=deadline, clock=self.clock
        )
        self._write_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = False
        self._contexts: dict[str, BranchContext] = {}
        self._cancel_all_event = threading.Event()
        self._cancel_reason: str | None = None
        self._control_hook: Callable[[ControlHookEvent], None] | None = None
        self._used_run_ids: set[str] = set()
        self._waits_start = 0
        self._resource_lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        runtime_dir: str | Path | None = None,
        root_run_id: str = "parallel",
        clock: Clock | None = None,
        deadline: Deadline | None = None,
    ) -> "TaskCoordinator":
        return cls(
            config.parallel,
            runtime_dir=config.runtime_dir if runtime_dir is None else runtime_dir,
            root_run_id=root_run_id,
            clock=clock,
            deadline=deadline,
            limits=config.budget_limits(),
            profile=config.profile,
        )

    # ------------------------------------------------------------- control hook
    def set_control_hook(self, hook: Callable[[ControlHookEvent], None] | None) -> None:
        with self._state_lock:
            self._control_hook = hook

    def _emit_control_hook(self, kind: str, reason: str | None) -> None:
        if kind not in CONTROL_HOOK_EVENTS:
            raise SchedulingError(f"unknown control hook event {kind!r}")
        with self._state_lock:
            hook = self._control_hook
            active = len(self._contexts)
        event = ControlHookEvent(
            kind=kind,
            reason=reason,
            at=self.clock.monotonic(),
            active_branches=active,
            waiting_branches=len(self.limiter.waiting_branches),
        )
        if hook is not None:
            hook(event)

    def pause(self, reason: str = "user") -> None:
        self._emit_control_hook("pause_requested", reason)
        raise NotImplementedError(F07_NOT_IMPLEMENTED)

    def resume(self) -> None:
        self._emit_control_hook("resume_requested", None)
        raise NotImplementedError(F07_NOT_IMPLEMENTED)

    def drain(self, reason: str = "user") -> None:
        self._emit_control_hook("drain_requested", reason)
        raise NotImplementedError(F07_NOT_IMPLEMENTED)

    # ---------------------------------------------------------------- cancel API
    def cancel_all(self, reason: str = "cancelled") -> None:
        with self._state_lock:
            if not self._running:
                raise CoordinatorError("no coordinator run is in progress")
            if self._cancel_reason is None:
                self._cancel_reason = str(reason)
            contexts = list(self._contexts.values())
            self._cancel_all_event.set()
        for context in contexts:
            context.cancel_event.set()
        self.limiter.wake()

    def cancel_branch(self, branch_id: str, reason: str = "cancelled") -> None:
        with self._state_lock:
            context = self._contexts.get(branch_id)
        if context is None:
            raise CoordinatorError(f"branch {branch_id!r} is not active")
        context.cancel_event.set()
        self.limiter.wake()

    # ------------------------------------------------------------------ run API
    def run(
        self,
        specs: Sequence[BranchSpec],
        worker: Callable[[BranchContext], Any],
        *,
        report_path: str | Path | None = None,
    ) -> CoordinatorReport:
        plan = tuple(specs)
        if not plan:
            raise CoordinatorError("the coordinator needs at least one branch")
        if not callable(worker):
            raise CoordinatorError("worker must be callable")
        seen: set[str] = set()
        for spec in plan:
            if not isinstance(spec, BranchSpec):
                raise CoordinatorError("specs must be BranchSpec values")
            if spec.branch_id in seen:
                raise CoordinatorError(f"duplicate branch id {spec.branch_id!r}")
            seen.add(spec.branch_id)

        with self._run_lock:
            if self._running:
                raise CoordinatorError("the coordinator is already running")
            self._running = True

        try:
            self._cancel_all_event.clear()
            self._cancel_reason = None
            self._contexts = {}
            self._used_run_ids = set()
            self._waits_start = len(self.limiter.waits)
            (self.runtime_dir / "runs").mkdir(parents=True, exist_ok=True)
            self.ledger = RootBudgetLedger(
                limits=self.limits, deadline=self.deadline, clock=self.clock
            )
            started_at = self.clock.monotonic()
            execution = "serial" if self.mode == "serial" else "parallel"
            degraded_reason: str | None = None
            if self.mode == "serial" and self.profile == "official":
                degraded_reason = "official_profile"
            write_plan = [spec for spec in plan if spec.requires_serial]
            if execution == "parallel" and write_plan:
                execution = "serial"
                degraded_reason = "shared_write_serialization"
            capacity = 1 if execution == "serial" else self.capacity
            tab_pool = TabPool(capacity)
            if execution == "serial":
                collected: list[BranchReport] = []
                for index, spec in enumerate(plan):
                    if self._cancel_all_event.is_set():
                        collected.extend(
                            self._abandoned_report(pending_spec, started_at)
                            for pending_spec in plan[index:]
                        )
                        break
                    collected.append(
                        self._run_one(spec, worker, tab_pool, queued_at=started_at)
                    )
                reports = tuple(collected)
            else:
                reports = self._run_parallel(plan, worker, tab_pool, capacity, started_at)
            reports = tuple(reports)
            finished_at = self.clock.monotonic()
            intervals = [
                (branch.started_at, branch.finished_at)
                for branch in reports
                if branch.started_at is not None and branch.finished_at is not None
            ]
            rate_waits = tuple(
                wait for branch in reports for wait in branch.rate_waits
            )
            lock_waits = tuple(
                wait for branch in reports for wait in branch.lock_waits
            )
            report = CoordinatorReport(
                root_run_id=self.root_run_id,
                mode=self.mode,
                execution=execution,
                requested_capacity=self.requested_capacity,
                capacity=capacity,
                degraded_reason=degraded_reason,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=max(0.0, finished_at - started_at),
                branches=reports,
                ledger=self.ledger.snapshot(),
                rate_waits=rate_waits,
                lock_waits=lock_waits,
                max_active=_peak_concurrency(intervals),
                overlap_seconds=_max_pair_overlap(intervals),
                max_queue_wait_seconds=max(
                    (
                        max(0.0, (branch.started_at or branch.queued_at) - branch.queued_at)
                        for branch in reports
                    ),
                    default=0.0,
                ),
            )
            if report_path is not None:
                self._write_report(report, report_path)
            return report
        finally:
            with self._state_lock:
                self._contexts = {}
                self._running = False

    def _run_parallel(
        self,
        plan: Sequence[BranchSpec],
        worker: Callable[[BranchContext], Any],
        tab_pool: TabPool,
        capacity: int,
        queued_at: float,
    ) -> tuple[BranchReport, ...]:
        pending = deque(plan)
        completed: queue.Queue[BranchReport] = queue.Queue()
        active: dict[str, threading.Thread] = {}
        collected: dict[str, BranchReport] = {}
        while pending or active:
            while pending and len(active) < capacity and not self._cancel_all_event.is_set():
                spec = pending.popleft()
                thread = threading.Thread(
                    target=self._parallel_entry,
                    args=(spec, worker, tab_pool, completed, queued_at),
                    name=f"branch-{spec.branch_id}",
                    daemon=True,
                )
                active[spec.branch_id] = thread
                thread.start()
            if self._cancel_all_event.is_set():
                while pending:
                    spec = pending.popleft()
                    collected[spec.branch_id] = self._abandoned_report(spec, queued_at)
            if not active:
                continue
            try:
                report = completed.get(timeout=0.05)
            except queue.Empty:
                continue
            active.pop(report.branch_id, None)
            collected[report.branch_id] = report
        return tuple(collected[spec.branch_id] for spec in plan)

    def _parallel_entry(
        self,
        spec: BranchSpec,
        worker: Callable[[BranchContext], Any],
        tab_pool: TabPool,
        completed: "queue.Queue[BranchReport]",
        queued_at: float,
    ) -> None:
        try:
            report = self._run_one(spec, worker, tab_pool, queued_at=queued_at)
        except BaseException as exc:  # noqa: BLE001 - a branch must always be reported
            report = BranchReport(
                branch_id=spec.branch_id,
                task=spec.task,
                status="failed",
                output=None,
                error_type=type(exc).__name__,
                error_message=DEFAULT_REDACTOR.redact_exception(exc),
                page_id=None,
                run_id=None,
                read_only=spec.read_only,
                write_targets=spec.write_targets,
                queued_at=queued_at,
                started_at=None,
                finished_at=self.clock.monotonic(),
                duration_seconds=None,
                model_calls=0,
                page_actions=0,
                rate_retries=0,
                lock_waits=(),
                rate_waits=(),
            )
        completed.put(report)

    def _abandoned_report(self, spec: BranchSpec, queued_at: float) -> BranchReport:
        return BranchReport(
            branch_id=spec.branch_id,
            task=spec.task,
            status="abandoned",
            output=None,
            error_type=None,
            error_message=self._cancel_reason,
            page_id=None,
            run_id=None,
            read_only=spec.read_only,
            write_targets=spec.write_targets,
            queued_at=queued_at,
            started_at=None,
            finished_at=None,
            duration_seconds=None,
            model_calls=0,
            page_actions=0,
            rate_retries=0,
            lock_waits=(),
            rate_waits=(),
        )

    def _branch_run_id(self, branch_id: str) -> str:
        token = _safe_token(branch_id)
        prefix = self.root_run_id[: 64 - len(token) - 1]
        candidate = f"{prefix}-{token}"[:64] if prefix else token[:64]
        suffix = 2
        while candidate in self._used_run_ids:
            tail = f"-{suffix}"
            candidate = f"{candidate[: 64 - len(tail)]}{tail}"
            suffix += 1
        self._used_run_ids.add(candidate)
        return candidate

    def _run_one(
        self,
        spec: BranchSpec,
        worker: Callable[[BranchContext], Any],
        tab_pool: TabPool,
        *,
        queued_at: float,
    ) -> BranchReport:
        started_at = self.clock.monotonic()
        lease: TabLease | None = None
        session: Session | None = None
        context: BranchContext | None = None
        status = "failed"
        output: Any = None
        error_type: str | None = None
        error_message: str | None = None
        cancel_event = threading.Event()
        try:
            # Tab allocation and the branch's evidence directory are created
            # under one lock: they are short local operations, and serializing
            # them keeps the run-directory resolution deterministic on platforms
            # where concurrent directory creation races the filesystem walk.
            with self._resource_lock:
                lease = tab_pool.acquire(spec.branch_id, requested_page_id=spec.page_id)
                run_id = self._branch_run_id(spec.branch_id)
                session = Session(
                    run_id,
                    runtime_dir=self.runtime_dir,
                    clock=self.clock,
                    page_id=lease.page_id,
                )
            context = BranchContext(self, spec, lease, session, cancel_event)
            context.started_at = started_at
            with self._state_lock:
                self._contexts[spec.branch_id] = context
            if self._cancel_all_event.is_set():
                cancel_event.set()
            output = worker(context)
            status = "succeeded"
        except WaitCancelled as exc:
            status = "cancelled"
            error_type = type(exc).__name__
            error_message = DEFAULT_REDACTOR.redact_exception(exc)
        except TabUnavailable as exc:
            status = "failed"
            error_type = type(exc).__name__
            error_message = DEFAULT_REDACTOR.redact_exception(exc)
        except (DeadlineExceeded, BudgetExceeded) as exc:
            status = "failed"
            error_type = type(exc).__name__
            error_message = DEFAULT_REDACTOR.redact_exception(exc)
        except BaseException as exc:  # noqa: BLE001 - the report is the branch's terminal record
            status = "failed"
            error_type = type(exc).__name__
            error_message = DEFAULT_REDACTOR.redact_exception(exc)
        finally:
            finished_at = self.clock.monotonic()
            if session is not None:
                try:
                    if status != "succeeded" and session.state.value not in (
                        "finished",
                        "cancelled",
                        "closed",
                    ):
                        session.cancel(self._cancel_reason or status)
                except Exception:  # noqa: BLE001 - close still has to run
                    pass
                try:
                    session.close()
                except Exception:  # noqa: BLE001 - evidence close must not crash the run
                    pass
            if lease is not None:
                tab_pool.release(lease)
            with self._state_lock:
                self._contexts.pop(spec.branch_id, None)
        return BranchReport(
            branch_id=spec.branch_id,
            task=spec.task,
            status=status,
            output=output,
            error_type=error_type,
            error_message=error_message,
            page_id=None if lease is None else lease.page_id,
            run_id=None if session is None else session.run_id,
            read_only=spec.read_only,
            write_targets=spec.write_targets,
            queued_at=queued_at,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=max(0.0, finished_at - started_at),
            model_calls=0 if context is None else context.model_calls,
            page_actions=0 if context is None else context.page_actions,
            rate_retries=0 if context is None else context.rate_retries,
            lock_waits=() if context is None else tuple(context.lock_waits),
            rate_waits=(
                ()
                if context is None
                else tuple(
                    wait
                    for wait in self.limiter.waits[self._waits_start :]
                    if wait.branch_id == spec.branch_id
                )
            ),
        )

    # ------------------------------------------------------------- write locking
    def _acquire_write_lock(self, context: BranchContext, target: str) -> None:
        started = self.clock.monotonic()
        while not self._write_lock.acquire(timeout=0.02):
            if context.cancel_event.is_set():
                raise WaitCancelled(
                    f"branch {context.branch_id} cancelled while waiting for write lock {target!r}"
                )
        context.lock_waits.append(
            LockWait(
                branch_id=context.branch_id,
                target=target,
                seconds=max(0.0, self.clock.monotonic() - started),
            )
        )

    def _release_write_lock(self) -> None:
        try:
            self._write_lock.release()
        except RuntimeError:
            pass

    def _write_report(self, report: CoordinatorReport, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        import json

        target.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
