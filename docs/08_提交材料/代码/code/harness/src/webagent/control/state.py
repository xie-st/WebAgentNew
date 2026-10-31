"""Root task control states and immutable snapshots.

The root control layer is deliberately separate from
:class:`~webagent.session.SessionState`: a session is about one page action,
while this state covers the whole task handoff. The phase-2 contract names the
running/pause handoff states; ``new`` is the pre-start value and is not a task
state of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ControlState(str, Enum):
    NEW = "new"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    DRAINING = "draining"
    USER_CONTROL = "user_control"
    REFRESHING = "refreshing"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


TERMINAL_CONTROL_STATES = frozenset({ControlState.CANCELLED, ControlState.BLOCKED})

#: States in which the dispatch gate is open. ``refreshing`` is open because the
#: resumed generator may propose while the new observation is installed; the
#: runner still never dispatches during refresh.
ACTION_STATES = frozenset({ControlState.RUNNING, ControlState.REFRESHING})


class HandoffReason(str, Enum):
    """Why the browser was handed to the user."""

    USER_PAUSE = "user_pause"
    LOGIN_REQUIRED = "login_required"
    VERIFICATION_REQUIRED = "verification_required"
    MODEL_RESULT_DURING_PAUSE = "model_result_during_pause"
    CONTROL_GATE = "control_gate"


@dataclass(frozen=True)
class ControlSnapshot:
    """A read-only view used by tests, reports and the control event log."""

    run_id: str
    state: str
    control_epoch: int
    page_epoch: int
    pause_reason: str | None
    pause_source: str | None
    handoff_reason: str | None
    total_seconds: float
    human_wait_seconds: float
    execution_seconds: float
    platform_deadline_epoch: float | None
    official_remaining_seconds: float | None
    deadline_extended: bool
    pause_count: int
    discarded_proposals: int
    last_model_call_id: str | None
    session_state: str | None
    observation_id: str | None
    finished: bool
    accounting: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "state": self.state,
            "control_epoch": self.control_epoch,
            "page_epoch": self.page_epoch,
            "pause_reason": self.pause_reason,
            "pause_source": self.pause_source,
            "handoff_reason": self.handoff_reason,
            "total_seconds": self.total_seconds,
            "human_wait_seconds": self.human_wait_seconds,
            "execution_seconds": self.execution_seconds,
            "platform_deadline_epoch": self.platform_deadline_epoch,
            "official_remaining_seconds": self.official_remaining_seconds,
            "deadline_extended": self.deadline_extended,
            "pause_count": self.pause_count,
            "discarded_proposals": self.discarded_proposals,
            "last_model_call_id": self.last_model_call_id,
            "session_state": self.session_state,
            "observation_id": self.observation_id,
            "finished": self.finished,
            "accounting": dict(self.accounting),
        }
        return payload


__all__ = [
    "ACTION_STATES",
    "ControlSnapshot",
    "ControlState",
    "HandoffReason",
    "TERMINAL_CONTROL_STATES",
]
