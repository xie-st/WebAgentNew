"""Stagnation detection and a bounded recovery prompt.

Progress is not "a different action" and not "a different page". It is new
relevant evidence or a smaller gap for the current goal. A repeated action that
returns a new evidence id or completes another object is progress and must not
trigger recovery. A page/same-action repetition alone is only a diagnostic
candidate once the feedback window is full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..clock import Clock, SystemClock
from .types import RecoveryError


@dataclass(frozen=True)
class ProgressRecord:
    """One feedback step as seen by the project-side detector."""

    action_key: str
    state_key: str
    observation_id: str
    new_evidence_ids: tuple[str, ...] = ()
    gap_before: int | None = None
    gap_after: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.action_key, "action_key"),
            (self.state_key, "state_key"),
            (self.observation_id, "observation_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RecoveryError(f"{name} must be a non-empty string")
        if not isinstance(self.new_evidence_ids, tuple):
            object.__setattr__(self, "new_evidence_ids", tuple(self.new_evidence_ids))

    @property
    def reduced_gap(self) -> bool:
        return (
            self.gap_before is not None
            and self.gap_after is not None
            and self.gap_after < self.gap_before
        )

    @property
    def is_progress(self) -> bool:
        return bool(self.new_evidence_ids) or self.reduced_gap

    def to_dict(self) -> dict[str, object]:
        return {
            "action_key": self.action_key,
            "state_key": self.state_key,
            "observation_id": self.observation_id,
            "new_evidence_ids": list(self.new_evidence_ids),
            "gap_before": self.gap_before,
            "gap_after": self.gap_after,
            "is_progress": self.is_progress,
            "note": self.note,
        }


@dataclass(frozen=True)
class StagnationSignal:
    window_size: int
    repeated_state_count: int
    action_key: str
    state_key: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "repeated_state_count": self.repeated_state_count,
            "action_key": self.action_key,
            "state_key": self.state_key,
            "reason": self.reason,
        }


class StagnationDetector:
    """A fixed feedback window with an explicit new-evidence exception."""

    def __init__(
        self,
        *,
        window: int = 6,
        min_repeated_states: int = 2,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window < 2:
            raise RecoveryError("stagnation window must be an integer >= 2")
        if (
            not isinstance(min_repeated_states, int)
            or isinstance(min_repeated_states, bool)
            or min_repeated_states < 2
        ):
            raise RecoveryError("min_repeated_states must be an integer >= 2")
        self.window = window
        self.min_repeated_states = min_repeated_states
        self.clock = clock or SystemClock()
        self.records: list[ProgressRecord] = []
        self.last_progress_at: float | None = None

    def record(
        self,
        *,
        action_key: str,
        state_key: str,
        observation_id: str,
        new_evidence_ids: Iterable[str] = (),
        gap_before: int | None = None,
        gap_after: int | None = None,
        note: str = "",
    ) -> ProgressRecord:
        record = ProgressRecord(
            action_key=action_key,
            state_key=state_key,
            observation_id=observation_id,
            new_evidence_ids=tuple(dict.fromkeys(str(item) for item in new_evidence_ids)),
            gap_before=gap_before,
            gap_after=gap_after,
            note=note,
        )
        if record.is_progress:
            # A new relevant evidence id or a smaller gap resets the no-progress
            # window; the same URL/same click alone is not enough.
            self.records.clear()
            self.last_progress_at = self.clock.now()
        self.records.append(record)
        if len(self.records) > self.window * 2:
            self.records = self.records[-self.window :]
        return record

    def evaluate(self) -> StagnationSignal | None:
        if len(self.records) < self.window:
            return None
        window_records = self.records[-self.window :]
        if any(record.is_progress for record in window_records):
            return None
        latest = window_records[-1]
        repeated = sum(1 for record in window_records if record.state_key == latest.state_key)
        if repeated < self.min_repeated_states:
            return None
        return StagnationSignal(
            window_size=len(window_records),
            repeated_state_count=repeated,
            action_key=latest.action_key,
            state_key=latest.state_key,
            reason=(
                f"{len(window_records)} feedback actions produced no new relevant "
                f"evidence or gap reduction; state repeated {repeated} times"
            ),
        )

    def prompt(
        self,
        signal: StagnationSignal,
        *,
        goal_requirement: str,
        evidence_conflict: str,
        next_adjustment: str,
        reflections: Sequence[str] = (),
    ) -> str:
        """At most three reflection lines plus one bounded next-step line."""

        lines = [
            f"Current goal: {goal_requirement.strip()}",
            f"Evidence conflict: {evidence_conflict.strip()}",
        ]
        recent = [str(item).strip() for item in reflections if str(item).strip()][-3:]
        if recent:
            lines.append("Recent evidence-based reflections: " + " | ".join(recent))
        lines.append(f"One next adjustment: {next_adjustment.strip()}")
        return "\n".join(lines)


__all__ = ["ProgressRecord", "StagnationDetector", "StagnationSignal"]
