"""One shared site schedule for every branch.

The coordinator does not have a per-branch limiter: a single-token pacing
schedule and a global cooldown belong to the whole site. That is what makes a
429 observed by one branch back off every branch, and it also makes a burst
after recovery impossible - the next operation may start only one interval
after the previous one, never several at once.

Waits are cancellable: the loop polls the branch cancellation event on every
iteration, and :meth:`SharedRateLimiter.wake` breaks pacing waiters out of their
condition wait when a cooldown or a cancellation changes the schedule.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ..clock import Clock, SystemClock
from .errors import WaitCancelled

RateWaitKind = str  # "rate" | "cooldown"


@dataclass(frozen=True)
class RateWait:
    """One branch wait on the shared site schedule."""

    branch_id: str
    kind: RateWaitKind
    requested_at: float
    resumed_at: float | None
    seconds: float
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "kind": self.kind,
            "requested_at": self.requested_at,
            "resumed_at": self.resumed_at,
            "seconds": self.seconds,
            "cancelled": self.cancelled,
        }


class SharedRateLimiter:
    def __init__(
        self,
        *,
        interval_seconds: float = 1.0,
        clock: Clock | None = None,
        poll_seconds: float = 0.02,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self.interval_seconds = float(interval_seconds)
        self.clock = clock or SystemClock()
        self.poll_seconds = float(poll_seconds)
        self._condition = threading.Condition()
        now = self.clock.monotonic()
        self._next_allowed = now
        self._cooldown_until = 0.0
        self._waits: list[RateWait] = []
        self._waiting: dict[str, int] = {}
        self.max_waiters = 0

    @property
    def cooldown_until(self) -> float:
        with self._condition:
            return self._cooldown_until

    @property
    def waits(self) -> tuple[RateWait, ...]:
        with self._condition:
            return tuple(self._waits)

    @property
    def waiting_branches(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(sorted(self._waiting))

    def note_retry_after(self, seconds: float) -> float:
        """Apply a cooldown to every branch and return its monotonic end."""

        with self._condition:
            end = self.clock.monotonic() + max(0.0, float(seconds))
            if end > self._cooldown_until:
                self._cooldown_until = end
            self._condition.notify_all()
            return self._cooldown_until

    def wake(self) -> None:
        """Break pacing waiters out of their wait (cooldown or cancellation)."""

        with self._condition:
            self._condition.notify_all()

    def acquire(
        self,
        branch_id: str,
        *,
        cancel_event: threading.Event | None = None,
    ) -> RateWait | None:
        """Claim the next site slot, waiting out pacing or a shared cooldown.

        Returns the recorded wait, or ``None`` when no wait was needed. In the
        first release pacing is one shared token: after an operation the next
        one is allowed only ``interval_seconds`` later.
        """

        with self._condition:
            now = self.clock.monotonic()
            target = max(self._next_allowed, self._cooldown_until)
            if target <= now:
                self._next_allowed = now + self.interval_seconds
                return None
            requested_at = now
            kind: RateWaitKind = (
                "cooldown" if self._cooldown_until >= self._next_allowed else "rate"
            )
            self._waiting[branch_id] = self._waiting.get(branch_id, 0) + 1
            self.max_waiters = max(self.max_waiters, len(self._waiting))
            cancelled = False
            record: RateWait | None = None
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        raise WaitCancelled(
                            f"branch {branch_id} cancelled while waiting for the shared site schedule"
                        )
                    now = self.clock.monotonic()
                    target = max(self._next_allowed, self._cooldown_until)
                    delay = target - now
                    if delay <= 0:
                        self._next_allowed = now + self.interval_seconds
                        break
                    self._condition.wait(min(delay, self.poll_seconds))
            finally:
                remaining = self._waiting.get(branch_id, 0) - 1
                if remaining > 0:
                    self._waiting[branch_id] = remaining
                else:
                    self._waiting.pop(branch_id, None)
                resumed_at = self.clock.monotonic()
                record = RateWait(
                    branch_id=branch_id,
                    kind=kind,
                    requested_at=requested_at,
                    resumed_at=resumed_at,
                    seconds=0.0 if cancelled else max(0.0, resumed_at - requested_at),
                    cancelled=cancelled,
                )
                self._waits.append(record)
        return record
