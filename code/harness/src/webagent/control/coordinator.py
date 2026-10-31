"""Root task control: automatic takeover, pause/drain/handoff/resume.

This coordinator is the branch-control layer described in the phase-2 design.
It does not change :class:`~webagent.session.SessionState`; it owns the
task-level sequence::

    running -> pause_requested -> draining -> user_control -> refreshing -> running

plus ``cancelled`` and ``blocked``. It also owns the dispatch gate: while the
task is paused or handed off, ``gate()`` refuses a page-action dispatch. The
session consults the same gate in ``Session.consume`` so an action cannot be
dispatched even from a caller that bypasses the controlled runner.

User resume is signal-driven. ``wait_for_user_signal`` never returns ``resume``
on its own; a timeout leaves the state in ``user_control`` and raises
``ControlWaitTimeout``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, TYPE_CHECKING

from ..budget import Deadline
from ..clock import Clock, SystemClock
from ..errors import DeadlineExceeded
from ..protocol import Observation, WebAction
from .accounting import ModelCallAccounting, ModelCallRecord
from .detection import TakeoverSignal, detect_takeover
from .errors import (
    ControlCancelled,
    ControlError,
    ControlPaused,
    ControlStateError,
    ControlWaitTimeout,
)
from .events import ControlEventLog
from .state import ACTION_STATES, ControlSnapshot, ControlState, TERMINAL_CONTROL_STATES
from .timing import ControlTiming

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..session import Session
    from .signals import ControlSource

_DEFAULT_POLL_SECONDS = 0.02


class ControlCoordinator:
    """One task's control state, gate, timing and control event log."""

    def __init__(
        self,
        run_id: str,
        *,
        clock: Clock | None = None,
        source: "ControlSource | None" = None,
        event_log: ControlEventLog | None = None,
        timer: ControlTiming | None = None,
        platform_deadline_epoch: float | None = None,
        wait_timeout_seconds: float | None = None,
        detector: Callable[[Observation | None], TakeoverSignal | None] = detect_takeover,
    ) -> None:
        self.run_id = run_id
        self.clock = clock or SystemClock()
        self.source = source
        self.detector = detector
        self.wait_timeout_seconds = wait_timeout_seconds
        self.timing = timer or ControlTiming(
            clock=self.clock, platform_deadline_epoch=platform_deadline_epoch
        )
        self.event_log = event_log
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._state = ControlState.NEW
        self._control_epoch = 0
        self._page_epoch = 0
        self._pause_reason: str | None = None
        self._pause_source: str | None = None
        self._handoff_reason: str | None = None
        self._pause_count = 0
        self._discarded_proposals: list[dict[str, Any]] = []
        self._accounting = ModelCallAccounting()
        self._last_model_call_id: str | None = None
        self._session: "Session | None" = None
        self._in_flight_action: str | None = None
        self._executed_action_ids: list[str] = []
        self._feedback_statuses: list[str] = []
        self._finished = False
        self._started = False

    # ------------------------------------------------------------------ setup
    def bind_session(self, session: "Session") -> None:
        self._session = session
        self.timing.deadline_from_session(session.deadline)
        if self.event_log is None:
            run_dir = Path(session.evidence.run_dir)
            self.event_log = ControlEventLog(
                self.run_id, run_dir / "control_events.jsonl", clock=self.clock
            )

    @property
    def started(self) -> bool:
        return self._started

    @property
    def state(self) -> ControlState:
        return self._state

    @property
    def control_epoch(self) -> int:
        return self._control_epoch

    @property
    def page_epoch(self) -> int:
        return self._page_epoch

    @property
    def pause_reason(self) -> str | None:
        return self._pause_reason

    @property
    def discarded_proposals(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._discarded_proposals)

    @property
    def accounting(self) -> ModelCallAccounting:
        return self._accounting

    @property
    def last_model_call_id(self) -> str | None:
        return self._last_model_call_id

    # --------------------------------------------------------------- lifecycle
    def start(self) -> ControlSnapshot:
        with self._lock:
            if self._started:
                return self.snapshot()
            self._state = ControlState.RUNNING
            self._control_epoch = 1
            self._started = True
            self.timing.start()
            self._log(
                "control_start",
                details={
                    "platform_deadline_epoch": self.timing.platform_deadline_epoch,
                    "action_gate": "open",
                },
            )
            return self.snapshot()

    def request_pause(
        self,
        reason: str,
        *,
        source: str = "user",
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Close the dispatch gate; returns whether this call changed state."""

        with self._lock:
            if self._state in TERMINAL_CONTROL_STATES:
                return False
            if self._state is not ControlState.RUNNING:
                return False
            previous = self._state
            self._state = ControlState.PAUSE_REQUESTED
            self._control_epoch += 1
            self._pause_reason = str(reason)
            self._pause_source = str(source)
            self._pause_count += 1
            self._log(
                "pause_requested",
                from_state=previous.value,
                to_state=self._state.value,
                reason=str(reason),
                source=str(source),
                details={**dict(details or {}), "action_gate": "closed"},
            )
            return True

    def begin_drain(
        self,
        *,
        executed_action_id: str | None = None,
        feedback_status: str | None = None,
        note: str = "",
    ) -> ControlSnapshot:
        with self._lock:
            if self._state in TERMINAL_CONTROL_STATES:
                return self.snapshot()
            if self._state is ControlState.RUNNING:
                raise ControlStateError("drain needs a pending pause request")
            previous = self._state
            self._state = ControlState.DRAINING
            if executed_action_id is not None:
                self._executed_action_ids.append(str(executed_action_id))
            if feedback_status is not None:
                self._feedback_statuses.append(str(feedback_status))
            self._log(
                "draining",
                from_state=previous.value,
                to_state=self._state.value,
                reason=self._pause_reason,
                source=self._pause_source,
                action_id=executed_action_id,
                details={
                    "feedback_status": feedback_status,
                    "note": str(note),
                    "in_flight_action": self._in_flight_action,
                },
            )
            return self.snapshot()

    def begin_user_control(self) -> ControlSnapshot:
        with self._lock:
            if self._state not in (ControlState.PAUSE_REQUESTED, ControlState.DRAINING):
                raise ControlStateError(
                    f"cannot hand off the browser from {self._state.value}"
                )
            previous = self._state
            self._state = ControlState.USER_CONTROL
            self._handoff_reason = self._pause_reason
            self.timing.begin_handoff(self._pause_reason or "")
            self._log(
                "handoff",
                from_state=previous.value,
                to_state=self._state.value,
                reason=self._pause_reason,
                source=self._pause_source,
                details={
                    "browser_drained": self._in_flight_action is None,
                    "executed_action_ids": list(self._executed_action_ids),
                    "feedback_statuses": list(self._feedback_statuses),
                    "timing": self.timing.snapshot().to_dict(),
                },
            )
            return self.snapshot()

    def wait_for_user_signal(
        self,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
    ) -> str:
        """Block for an explicit resume/cancel; never resume on a timer."""

        if self._state is not ControlState.USER_CONTROL:
            raise ControlStateError(
                f"wait_for_user_signal needs user_control, not {self._state.value}"
            )
        deadline = (
            None if timeout_seconds is None else time.monotonic() + float(timeout_seconds)
        )
        while True:
            if self._state is ControlState.CANCELLED:
                return "cancel"
            command = self.source.poll() if self.source is not None else None
            if command == "resume":
                self._log(
                    "resume_signal",
                    reason="explicit_user_signal",
                    source="user",
                    details={"command": "resume"},
                )
                return "resume"
            if command == "cancel":
                self._log(
                    "cancel_signal",
                    reason="user_cancel",
                    source="user",
                    details={"command": "cancel"},
                )
                return "cancel"
            if deadline is not None and time.monotonic() >= deadline:
                self._log(
                    "wait_timeout",
                    reason="no_explicit_resume",
                    source="control",
                    details={"timeout_seconds": timeout_seconds},
                )
                raise ControlWaitTimeout(
                    "no explicit resume arrived; the task stays in user_control"
                )
            self._wake.wait(poll_seconds)
            self._wake.clear()

    def begin_refresh(self) -> ControlSnapshot:
        with self._lock:
            if self._state is not ControlState.USER_CONTROL:
                raise ControlStateError(
                    f"cannot refresh from {self._state.value}; a handoff is not open"
                )
            previous = self._state
            self._state = ControlState.REFRESHING
            self.timing.end_handoff()
            self._log(
                "refreshing",
                from_state=previous.value,
                to_state=self._state.value,
                reason="reobserve_after_handoff",
                source="control",
                details={"action_gate": "open_for_reobserve"},
            )
            return self.snapshot()

    def finish_refresh(self, *, observation: Observation | None = None) -> ControlSnapshot:
        with self._lock:
            if self._state is not ControlState.REFRESHING:
                raise ControlStateError(
                    f"cannot finish refresh from {self._state.value}"
                )
            previous = self._state
            self._state = ControlState.RUNNING
            self._control_epoch += 1
            self._page_epoch += 1
            self._log(
                "resumed",
                from_state=previous.value,
                to_state=self._state.value,
                reason="fresh_observation_installed",
                source="control",
                observation_id=None if observation is None else observation.observation_id,
                details={
                    "control_epoch": self._control_epoch,
                    "page_epoch": self._page_epoch,
                    "action_gate": "open",
                    "old_element_numbers_valid": False,
                },
            )
            return self.snapshot()

    def cancel(
        self, reason: str = "user_cancel", *, details: Mapping[str, Any] | None = None
    ) -> ControlSnapshot:
        with self._lock:
            if self._state is ControlState.CANCELLED:
                return self.snapshot()
            previous = self._state
            self.timing.end_handoff()
            self._state = ControlState.CANCELLED
            self._finished = True
            self.timing.finish()
            self._log(
                "cancelled",
                from_state=previous.value,
                to_state=self._state.value,
                reason=str(reason),
                source="control",
                details={**dict(details or {}), "action_gate": "closed"},
            )
            self._wake.set()
            return self.snapshot()

    def block(
        self, reason: str, *, details: Mapping[str, Any] | None = None
    ) -> ControlSnapshot:
        with self._lock:
            if self._state is ControlState.BLOCKED:
                return self.snapshot()
            previous = self._state
            self.timing.end_handoff()
            self._state = ControlState.BLOCKED
            self._finished = True
            self.timing.finish()
            self._log(
                "blocked",
                from_state=previous.value,
                to_state=self._state.value,
                reason=str(reason),
                source="control",
                details={**dict(details or {}), "action_gate": "closed"},
            )
            self._wake.set()
            return self.snapshot()

    def finish(self, status: str = "completed") -> ControlSnapshot:
        with self._lock:
            if self._finished:
                return self.snapshot()
            self._finished = True
            self.timing.finish()
            self._log(
                "control_end",
                reason=str(status),
                source="control",
                details={"timing": self.timing.snapshot().to_dict()},
            )
            return self.snapshot()

    # ------------------------------------------------------------------- gate
    def action_allowed(self) -> bool:
        return self._state in ACTION_STATES

    def gate(self, operation: str = "dispatch") -> None:
        """Raise when a page action may not be dispatched."""

        if self.action_allowed():
            return
        if self._state is ControlState.CANCELLED:
            raise ControlCancelled(f"cannot {operation}: the task was cancelled")
        raise ControlPaused(
            f"cannot {operation}: task control state is {self._state.value} "
            f"({self._pause_reason or 'handoff'})"
        )

    # --------------------------------------------------------------- commands
    def poll_external_command(self) -> str | None:
        if self.source is None:
            return None
        command = self.source.poll()
        if command == "pause":
            self.request_pause("user_pause", source="user", details={"command": "pause"})
            return "pause"
        if command in ("cancel", "resume"):
            return command
        return None

    # --------------------------------------------------------------- detector
    def detect(self, observation: Observation | None) -> TakeoverSignal | None:
        signal = self.detector(observation)
        if signal is not None:
            self._log(
                "takeover_detected",
                reason=signal.reason,
                source="visible_observation",
                observation_id=(
                    None if observation is None else observation.observation_id
                ),
                details=signal.to_dict(),
            )
        return signal

    # ---------------------------------------------------------------- actions
    def note_action_in_flight(self, action: WebAction) -> None:
        with self._lock:
            self._in_flight_action = action.action_id
            self._log(
                "action_in_flight",
                action_id=action.action_id,
                observation_id=action.observation_id,
                details={"op": action.op.value},
            )

    def note_action_finished(self, feedback_status: str) -> None:
        with self._lock:
            action_id = self._in_flight_action
            self._in_flight_action = None
            self._log(
                "action_finished",
                action_id=action_id,
                details={"feedback_status": str(feedback_status)},
            )

    def discard_proposal(
        self,
        action: WebAction | None,
        *,
        reason: str,
        model_call_id: str | None = None,
    ) -> None:
        """Record an unconsumed proposal; it is never a successful action."""

        with self._lock:
            record = {
                "action_id": None if action is None else action.action_id,
                "observation_id": None if action is None else action.observation_id,
                "op": None if action is None else action.op.value,
                "reason": str(reason),
                "model_call_id": model_call_id or self._last_model_call_id,
                "consumed": False,
                "executed": False,
            }
            self._discarded_proposals.append(record)
            self._accounting.mark_discarded(record["model_call_id"])
            self._log(
                "proposal_dropped",
                reason=str(reason),
                source="control",
                action_id=record["action_id"],
                observation_id=record["observation_id"],
                details={**record, "not_executed": True},
            )

    # ----------------------------------------------------------------- models
    def record_model_result(
        self,
        *,
        call_id: str,
        control_epoch: int | None,
        request_id: str | None,
        latency_ms: float,
        failure_type: str | None = None,
        paused_at_return: bool = False,
    ) -> ModelCallRecord:
        with self._lock:
            record = self._accounting.record(
                call_id=call_id,
                control_epoch=control_epoch,
                request_id=request_id,
                latency_ms=latency_ms,
                failure_type=failure_type,
                paused_at_return=paused_at_return,
            )
            self._last_model_call_id = record.call_id
            if paused_at_return or (
                control_epoch is not None and control_epoch != self._control_epoch
            ):
                self._log(
                    "model_result_during_pause",
                    reason="control_epoch_changed",
                    source="model",
                    details={
                        **record.to_dict(),
                        "counted": True,
                        "proposal_usable": False,
                    },
                )
            return record

    # --------------------------------------------------------------- deadline
    def check_platform_deadline(self, stage: str = "work") -> None:
        try:
            self.timing.check_deadline(stage)
        except DeadlineExceeded:
            self._log(
                "deadline_blocked",
                reason=stage,
                source="platform",
                details={
                    "platform_deadline_epoch": self.timing.platform_deadline_epoch,
                    "official_remaining_seconds": self.timing.official_remaining_seconds(),
                    "deadline_extended": self.timing.deadline_extended(),
                },
            )
            raise

    # ----------------------------------------------------------------- output
    def snapshot(self) -> ControlSnapshot:
        timer = self.timing.snapshot()
        session = self._session
        observation = None if session is None else session.observation
        return ControlSnapshot(
            run_id=self.run_id,
            state=self._state.value,
            control_epoch=self._control_epoch,
            page_epoch=self._page_epoch,
            pause_reason=self._pause_reason,
            pause_source=self._pause_source,
            handoff_reason=self._handoff_reason,
            total_seconds=timer.total_seconds,
            human_wait_seconds=timer.human_wait_seconds,
            execution_seconds=timer.execution_seconds,
            platform_deadline_epoch=timer.platform_deadline_epoch,
            official_remaining_seconds=timer.official_remaining_seconds,
            deadline_extended=timer.deadline_extended,
            pause_count=self._pause_count,
            discarded_proposals=len(self._discarded_proposals),
            last_model_call_id=self._last_model_call_id,
            session_state=None if session is None else session.state.value,
            observation_id=(
                None if observation is None else observation.observation_id
            ),
            finished=self._finished,
            accounting=self._accounting.snapshot(),
        )

    def close(self) -> None:
        if self.event_log is not None:
            self.event_log.close()

    # ----------------------------------------------------------------- private
    def _log(
        self,
        event_type: str,
        *,
        from_state: str | None = None,
        to_state: str | None = None,
        reason: str | None = None,
        source: str | None = None,
        action_id: str | None = None,
        observation_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self.event_log is None:
            return
        self.event_log.append(
            event_type,
            control_epoch=self._control_epoch,
            page_epoch=self._page_epoch,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            source=source,
            action_id=action_id,
            observation_id=observation_id,
            details=details,
        )


__all__ = ["ControlCoordinator"]
