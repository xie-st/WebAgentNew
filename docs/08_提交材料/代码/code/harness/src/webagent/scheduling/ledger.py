"""Root-level model-call gate and accounting shared by every branch.

Every branch claims from one ledger before its model call. The claim takes a
unique call index, so a call that later fails, is cancelled or is discarded is
still charged; the coordinator never recomputes the total from successful
branches only. The concurrency gate is one slot, which is the first-release
rule: overlapping page waits must not widen model concurrency. The root deadline
and ``max_model_calls`` are checked at the claim, never per branch.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ..budget import BudgetLimits, Deadline
from ..clock import Clock, SystemClock
from ..errors import BudgetExceeded
from .errors import WaitCancelled


@dataclass(frozen=True)
class ModelCallRecord:
    call_index: int
    branch_id: str
    started_at: float
    finished_at: float | None
    outcome: str
    failure_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "branch_id": self.branch_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "failure_type": self.failure_type,
        }


class RootBudgetLedger:
    def __init__(
        self,
        *,
        limits: BudgetLimits | None = None,
        deadline: Deadline | None = None,
        clock: Clock | None = None,
        poll_seconds: float = 0.02,
    ) -> None:
        self.limits = limits or BudgetLimits()
        self.deadline = deadline
        self.clock = clock or (deadline.clock if deadline is not None else SystemClock())
        self.poll_seconds = float(poll_seconds)
        self._gate = threading.Lock()
        self._lock = threading.Lock()
        self._claimed = 0
        self._inflight = 0
        self.max_concurrent = 0
        self._refused: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []

    @property
    def total_calls(self) -> int:
        with self._lock:
            return self._claimed

    @property
    def refused_claims(self) -> int:
        with self._lock:
            return len(self._refused)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = [dict(record) for record in self._records]
            refused = [dict(record) for record in self._refused]
            by_branch: dict[str, int] = {}
            for record in records:
                if record["outcome"] == "refused":
                    continue
                by_branch[record["branch_id"]] = by_branch.get(record["branch_id"], 0) + 1
            return {
                "total_calls": self._claimed,
                "max_concurrent": self.max_concurrent,
                "by_branch": by_branch,
                "records": records,
                "refused_claims": len(refused),
            }

    def _acquire_gate(self, branch_id: str, cancel_event: threading.Event | None) -> None:
        while not self._gate.acquire(timeout=self.poll_seconds):
            if cancel_event is not None and cancel_event.is_set():
                raise WaitCancelled(
                    f"branch {branch_id} cancelled while waiting for the single model slot"
                )

    def _begin(self, branch_id: str) -> int:
        if self.deadline is not None:
            try:
                self.deadline.require("model call")
            except Exception as exc:
                with self._lock:
                    self._refused.append(
                        {
                            "branch_id": branch_id,
                            "reason": type(exc).__name__,
                            "started_at": self.clock.monotonic(),
                        }
                    )
                raise
        with self._lock:
            if (
                self.limits.max_model_calls is not None
                and self._claimed >= self.limits.max_model_calls
            ):
                self._refused.append(
                    {
                        "branch_id": branch_id,
                        "reason": "BudgetExceeded",
                        "started_at": self.clock.monotonic(),
                    }
                )
                raise BudgetExceeded(
                    f"root model call budget exhausted ({self.limits.max_model_calls})"
                )
            self._claimed += 1
            self._inflight += 1
            self.max_concurrent = max(self.max_concurrent, self._inflight)
            return self._claimed

    @contextmanager
    def model_call(
        self,
        branch_id: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[int]:
        """Hold the single model slot and claim one root-level call.

        Yields the unique call index. The record stays in the ledger with the
        outcome that actually happened (``ok``, ``failed`` or ``cancelled``).
        """

        self._acquire_gate(branch_id, cancel_event)
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise WaitCancelled(
                    f"branch {branch_id} cancelled before its model call was claimed"
                )
            call_index = self._begin(branch_id)
            started = self.clock.monotonic()
            record: dict[str, Any] = {
                "call_index": call_index,
                "branch_id": branch_id,
                "started_at": started,
                "finished_at": None,
                "outcome": "running",
                "failure_type": None,
            }
            with self._lock:
                self._records.append(record)
            outcome = "ok"
            failure_type: str | None = None
            try:
                yield call_index
            except WaitCancelled:
                outcome = "cancelled"
                raise
            except BaseException as exc:  # noqa: BLE001 - charged failures are the point
                outcome = "failed"
                failure_type = type(exc).__name__
                raise
            finally:
                with self._lock:
                    record["finished_at"] = self.clock.monotonic()
                    record["outcome"] = outcome
                    record["failure_type"] = failure_type
                    self._inflight -= 1
        finally:
            self._gate.release()
