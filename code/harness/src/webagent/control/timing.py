"""Separate total / human-wait / execution timing for one controlled task.

The platform deadline is an input, never a value this module may move. A local
pause adds to the human-wait bucket and to total elapsed time; it never adds to
the official remaining time. ``deadline_extended`` is computed by comparing the
remembered platform epoch with the value in force, and is always false in this
implementation because nothing writes to the deadline.

All durations use the monotonic clock. The official deadline is a wall-clock
epoch supplied by the platform/caller and is read with ``clock.now``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..budget import Deadline
from ..clock import Clock, SystemClock
from ..errors import DeadlineExceeded

_ROUND = 6


@dataclass(frozen=True)
class TimingSnapshot:
    total_seconds: float
    human_wait_seconds: float
    execution_seconds: float
    platform_deadline_epoch: float | None
    official_remaining_seconds: float | None
    deadline_extended: bool
    handoff_count: int
    open_handoff: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "human_wait_seconds": self.human_wait_seconds,
            "execution_seconds": self.execution_seconds,
            "platform_deadline_epoch": self.platform_deadline_epoch,
            "official_remaining_seconds": self.official_remaining_seconds,
            "deadline_extended": self.deadline_extended,
            "handoff_count": self.handoff_count,
            "open_handoff": self.open_handoff,
        }


class ControlTiming:
    """Reconcilable timing ledger; no wall-clock correction of durations."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        platform_deadline_epoch: float | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        if platform_deadline_epoch is not None and (
            not isinstance(platform_deadline_epoch, (int, float))
            or isinstance(platform_deadline_epoch, bool)
        ):
            raise ValueError("platform_deadline_epoch must be a number or null")
        self.platform_deadline_epoch = (
            None if platform_deadline_epoch is None else float(platform_deadline_epoch)
        )
        self._remembered_deadline = self.platform_deadline_epoch
        self._started_monotonic: float | None = None
        self._started_epoch: float | None = None
        self._finished_monotonic: float | None = None
        self._handoff_start: float | None = None
        self._handoff_reason: str | None = None
        self._closed_handoffs: list[float] = []
        self._handoff_count = 0

    # ------------------------------------------------------------------ clock
    @property
    def started(self) -> bool:
        return self._started_monotonic is not None

    def start(self, *, epoch_seconds: float | None = None) -> None:
        if self.started:
            return
        self._started_monotonic = self.clock.monotonic()
        self._started_epoch = (
            self.clock.now() if epoch_seconds is None else float(epoch_seconds)
        )

    def finish(self) -> None:
        if self._finished_monotonic is None:
            self._finished_monotonic = self.clock.monotonic()
        self.end_handoff()

    # --------------------------------------------------------------- handoffs
    def begin_handoff(self, reason: str = "") -> None:
        self.start()
        if self._handoff_start is not None:
            raise RuntimeError("a handoff interval is already open")
        self._handoff_start = self.clock.monotonic()
        self._handoff_reason = str(reason)
        self._handoff_count += 1

    def end_handoff(self) -> None:
        if self._handoff_start is None:
            return
        self._closed_handoffs.append(self.clock.monotonic() - self._handoff_start)
        self._handoff_start = None

    @property
    def open_handoff(self) -> bool:
        return self._handoff_start is not None

    # ---------------------------------------------------------------- reading
    def _end_monotonic(self) -> float:
        if self._finished_monotonic is not None:
            return self._finished_monotonic
        return self.clock.monotonic()

    def total_seconds(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        return max(0.0, self._end_monotonic() - self._started_monotonic)

    def human_wait_seconds(self) -> float:
        total = sum(self._closed_handoffs)
        if self._handoff_start is not None:
            total += max(0.0, self.clock.monotonic() - self._handoff_start)
        return total

    def execution_seconds(self) -> float:
        return max(0.0, self.total_seconds() - self.human_wait_seconds())

    def official_remaining_seconds(self) -> float | None:
        if self.platform_deadline_epoch is None:
            return None
        return self.platform_deadline_epoch - self.clock.now()

    def deadline_extended(self) -> bool:
        if self._remembered_deadline is None:
            return False
        return (
            self.platform_deadline_epoch is None
            or abs(self.platform_deadline_epoch - self._remembered_deadline) > 1e-9
        )

    def check_deadline(self, stage: str = "work") -> None:
        """Refuse work after the platform deadline, including during handoff."""

        if self.platform_deadline_epoch is None:
            return
        if self.clock.now() >= self.platform_deadline_epoch:
            raise DeadlineExceeded(
                f"platform deadline reached before {stage}; the local pause does not "
                "extend the official task deadline"
            )

    def deadline_from_session(self, deadline: Deadline | None) -> float | None:
        if deadline is None:
            return None
        if self.platform_deadline_epoch is None:
            self.platform_deadline_epoch = float(deadline.epoch_seconds)
            self._remembered_deadline = float(deadline.epoch_seconds)
        return self.platform_deadline_epoch

    def snapshot(self) -> TimingSnapshot:
        return TimingSnapshot(
            total_seconds=round(self.total_seconds(), _ROUND),
            human_wait_seconds=round(self.human_wait_seconds(), _ROUND),
            execution_seconds=round(self.execution_seconds(), _ROUND),
            platform_deadline_epoch=self.platform_deadline_epoch,
            official_remaining_seconds=(
                None
                if self.official_remaining_seconds() is None
                else round(self.official_remaining_seconds(), _ROUND)
            ),
            deadline_extended=self.deadline_extended(),
            handoff_count=self._handoff_count,
            open_handoff=self.open_handoff,
        )


__all__ = ["ControlTiming", "TimingSnapshot"]
