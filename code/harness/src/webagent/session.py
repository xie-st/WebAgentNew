"""Single-task session lifecycle.

The session owns the current observation, the one proposed action, the one
action awaiting feedback, pause/resume and close. It is deliberately free of
EffGen: A2 drives the generator and calls these transitions; the official SDK
and B/C backends call the same ones.

Invariants enforced here:

* an action is valid only against the current (run, page, observation) binding;
* one action may be proposed and consumed at most once;
* a dispatched action's next observation must carry feedback for exactly that
  action;
* pause is only possible with no in-flight action, and resume installs a fresh
  observation, invalidating the old element table and old action bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .budget import Budget, BudgetLimits, Deadline
from .clock import Clock, SystemClock
from .errors import (
    ActionRejected,
    BudgetExceeded,
    DeadlineExceeded,
    EvidenceError,
    ProtocolError,
    RejectReason,
    SessionStateError,
)
from .events import EventLogger
from .evidence import EvidenceStore
from .protocol import (
    ElementRef,
    FeedbackStatus,
    Finish,
    Observation,
    TerminationReason,
    WebAction,
    validate_run_id,
)
from .redaction import DEFAULT_REDACTOR, Redactor
from .tools import resolve_target


class SessionState(str, Enum):
    NEW = "new"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    CLOSED = "closed"


@dataclass(frozen=True)
class SessionSnapshot:
    run_id: str
    state: SessionState
    task: str | None
    observation_id: str | None
    page_id: str | None
    proposed_action_id: str | None
    in_flight_action_id: str | None
    model_calls_total: int


class Session:
    def __init__(
        self,
        run_id: str,
        *,
        runtime_dir: str | Path = "runtime",
        clock: Clock | None = None,
        logger: EventLogger | None = None,
        evidence: EvidenceStore | None = None,
        redactor: Redactor | None = None,
        budget: Budget | None = None,
        limits: BudgetLimits | None = None,
        page_id: str | None = None,
    ) -> None:
        validate_run_id(run_id)
        self.run_id = run_id
        self.clock = clock or SystemClock()
        self.redactor = redactor or DEFAULT_REDACTOR
        self.evidence = evidence or EvidenceStore(runtime_dir, run_id)
        self.logger = logger or EventLogger(
            run_id,
            self.evidence.events_path,
            clock=self.clock,
            redactor=self.redactor,
            model_calls_total=lambda: self._model_calls,
        )
        self.budget = budget or Budget(limits, clock=self.clock)
        self._page_id = page_id
        self._initial_page_id = page_id
        self.state = SessionState.NEW
        self.task: str | None = None
        self.finish_result: Finish | None = None
        self._observation: Observation | None = None
        self._seen_observations: set[str] = set()
        self._proposed: WebAction | None = None
        self._in_flight: WebAction | None = None
        self._consumed: set[str] = set()
        self._last_action_id: str | None = None
        self._notes: list[str] = []
        self._model_calls = 0
        self._deadline_logged = False
        self._closed = False
        self.logger.emit(
            "run_start",
            result={"schema_version": "webagent.result.v1", "run_id": run_id},
        )

    # ---------------------------------------------------------------- read API
    @property
    def observation(self) -> Observation | None:
        return self._observation

    @property
    def proposed_action(self) -> WebAction | None:
        return self._proposed

    @property
    def in_flight_action(self) -> WebAction | None:
        return self._in_flight

    @property
    def page_id(self) -> str | None:
        return self._page_id

    @property
    def model_calls_total(self) -> int:
        return self.logger.model_calls_total

    @property
    def task_notes(self) -> tuple[str, ...]:
        return tuple(self._notes)

    def add_task_note(self, note: str) -> None:
        """Store a task-local fact for internal ``task_note`` tools."""

        if not isinstance(note, str) or not note.strip():
            raise ProtocolError("task note must be a non-empty string")
        self._notes.append(note)

    @property
    def deadline(self) -> Deadline | None:
        return self.budget.deadline

    def snapshot(self) -> SessionSnapshot:
        observation = self._observation
        return SessionSnapshot(
            run_id=self.run_id,
            state=self.state,
            task=self.task,
            observation_id=None if observation is None else observation.observation_id,
            page_id=None if observation is None else observation.page_id,
            proposed_action_id=None if self._proposed is None else self._proposed.action_id,
            in_flight_action_id=None if self._in_flight is None else self._in_flight.action_id,
            model_calls_total=self.model_calls_total,
        )

    # ------------------------------------------------------------- transitions
    def reset(self) -> None:
        if self.state is SessionState.CLOSED:
            raise SessionStateError("session is closed")
        previous_state = self.state
        if self._in_flight is not None:
            self.logger.emit(
                "cancelled",
                page_id=self._page_id,
                url=None if self._observation is None else self._observation.url,
                observation_id=None if self._observation is None else self._observation.observation_id,
                action_id=self._in_flight.action_id,
                action_type=self._in_flight.op.value,
                result={"reason": "reset", "scope": "in_flight_action"},
            )
        if self._proposed is not None:
            self.logger.emit(
                "cancelled",
                page_id=self._page_id,
                url=None if self._observation is None else self._observation.url,
                observation_id=None if self._observation is None else self._observation.observation_id,
                action_id=self._proposed.action_id,
                action_type=self._proposed.op.value,
                result={"reason": "reset", "scope": "proposed_action"},
            )
        self.logger.emit(
            "reset",
            page_id=self._page_id,
            result={
                "previous_state": previous_state.value,
                "had_task": self.task is not None,
            },
        )
        self.task = None
        self.finish_result = None
        self._page_id = self._initial_page_id
        self._observation = None
        self._seen_observations.clear()
        self._proposed = None
        self._in_flight = None
        self._consumed.clear()
        self._last_action_id = None
        self._notes.clear()
        self._deadline_logged = False
        self.state = SessionState.READY

    def start_task(self, task: str) -> None:
        if self.state in (SessionState.CLOSED, SessionState.CANCELLED):
            raise SessionStateError(f"cannot start a task while session is {self.state.value}")
        if self.state in (SessionState.RUNNING, SessionState.PAUSED):
            raise SessionStateError("a task is already active")
        if self.state is SessionState.FINISHED:
            raise SessionStateError("reset the session before starting another task")
        if not isinstance(task, str) or not task.strip():
            raise SessionStateError("task must be a non-empty string")
        self.task = task
        self.state = SessionState.RUNNING
        self.logger.emit(
            "task_start",
            page_id=self._page_id,
            result={"task": task},
        )

    def ingest_observation(self, observation: Observation) -> Observation:
        if self.state is not SessionState.RUNNING:
            raise SessionStateError(
                f"cannot ingest an observation while session is {self.state.value}"
            )
        self._validate_observation(observation)
        if observation.observation_id in self._seen_observations:
            raise SessionStateError(
                f"observation {observation.observation_id!r} was already consumed in this task"
            )
        if self._proposed is not None:
            raise SessionStateError("ingest a new observation only after dispatching the proposed action")
        if self._in_flight is not None:
            action = self._in_flight
            if observation.previous_action_id != action.action_id:
                raise SessionStateError(
                    "the dispatched action's next observation must be bound to that action"
                )
            if observation.feedback is None:
                raise SessionStateError("the dispatched action's next observation must carry feedback")
            feedback = observation.feedback
            if feedback.status is FeedbackStatus.OBSERVED and feedback.observation_id != observation.observation_id:
                raise SessionStateError("observed feedback must reference this observation")
        else:
            if observation.previous_action_id is not None:
                raise SessionStateError(
                    "observation claims feedback for an action that was not dispatched by this session"
                )
        reference = self._save_observation(observation)
        self._observation = observation
        self._seen_observations.add(observation.observation_id)
        page_id = observation.page_id
        if self._in_flight is not None:
            action = self._in_flight
            feedback = observation.feedback
            assert feedback is not None
            self.logger.emit(
                "feedback",
                page_id=page_id,
                url=observation.url,
                observation_id=observation.observation_id,
                action_id=action.action_id,
                action_type=action.op.value,
                result=feedback.status.value,
                feedback_message=feedback.message,
                error_type=feedback.error_type,
                page_changed=feedback.page_changed,
            )
            self._last_action_id = action.action_id
            self._in_flight = None
        else:
            self._last_action_id = None
        self._emit_observation(observation, reference)
        return observation

    def propose(self, action: WebAction) -> WebAction:
        self._require_running_for_action("propose")
        self.check_deadline("action proposal")
        if self._in_flight is not None:
            raise SessionStateError("cannot propose a new action while one is awaiting feedback")
        if self._proposed is not None:
            raise SessionStateError("an action is already proposed")
        observation = self._observation
        if observation is None:
            raise ActionRejected(
                RejectReason.NO_OBSERVATION,
                details=["no current observation"],
                correction="Wait for a current observation before proposing an action.",
            )
        if action.action_id in self._consumed:
            raise ActionRejected(
                RejectReason.DUPLICATE_CONSUMPTION,
                details=[f"action {action.action_id!r} was already consumed"],
                correction="Do not resubmit an action that was already dispatched; use the new observation.",
            )
        resolve_target(action, observation)
        self.logger.emit(
            "proposed",
            page_id=action.page_id,
            url=observation.url,
            observation_id=action.observation_id,
            action_id=action.action_id,
            action_type=action.op.value,
            result=None,
            action_arguments=action.to_local_dict()["args"],
        )
        self._proposed = action
        return action

    def consume(self, action_id: str | None = None) -> WebAction:
        self._require_running_for_action("consume")
        self.check_deadline("action dispatch")
        if action_id is not None and action_id in self._consumed:
            raise ActionRejected(
                RejectReason.DUPLICATE_CONSUMPTION,
                details=[f"action {action_id!r} was already dispatched"],
                correction="This action was already consumed once; a pending request cannot be consumed twice.",
            )
        if self._proposed is None:
            raise ActionRejected(
                RejectReason.NO_PENDING_ACTION,
                details=["no proposed action to consume"],
                correction="Propose exactly one action from the current observation, then consume it once.",
            )
        if action_id is not None and action_id != self._proposed.action_id:
            raise ActionRejected(
                RejectReason.INVALID_ARGUMENTS,
                details=[f"pending action is {self._proposed.action_id!r}, not {action_id!r}"],
                correction="Consume the currently proposed action id.",
            )
        action = self._proposed
        observation = self._observation
        assert observation is not None
        self.logger.emit(
            "dispatched",
            page_id=action.page_id,
            url=observation.url,
            observation_id=action.observation_id,
            action_id=action.action_id,
            action_type=action.op.value,
            result=None,
        )
        self._consumed.add(action.action_id)
        self._proposed = None
        self._in_flight = action
        return action

    def pause(self) -> None:
        if self.state is not SessionState.RUNNING:
            raise SessionStateError(f"cannot pause while session is {self.state.value}")
        if self._in_flight is not None:
            raise SessionStateError(
                "cannot pause with an action awaiting feedback; wait for it to finish or time out"
            )
        dropped = None
        if self._proposed is not None:
            dropped = self._proposed
            self.logger.emit(
                "cancelled",
                page_id=dropped.page_id,
                url=None if self._observation is None else self._observation.url,
                observation_id=dropped.observation_id,
                action_id=dropped.action_id,
                action_type=dropped.op.value,
                result={"reason": "paused", "scope": "proposed_action"},
            )
            self._proposed = None
        self.state = SessionState.PAUSED
        self.logger.emit(
            "paused",
            page_id=self._page_id,
            url=None if self._observation is None else self._observation.url,
            observation_id=None if self._observation is None else self._observation.observation_id,
            result={"dropped_proposed_action": dropped is not None},
        )

    def resume(self, observation: Observation) -> Observation:
        if self.state is not SessionState.PAUSED:
            raise SessionStateError(f"cannot resume while session is {self.state.value}")
        self.check_deadline("resume")
        if self._in_flight is not None:
            raise SessionStateError("cannot resume while an action is awaiting feedback")
        old_observation_id = None if self._observation is None else self._observation.observation_id
        self._validate_observation(observation)
        if observation.observation_id in self._seen_observations:
            raise SessionStateError("resume requires a fresh observation id")
        if observation.previous_action_id is not None or observation.feedback is not None:
            raise SessionStateError(
                "resume observation must not claim feedback for an action; re-observe the page"
            )
        reference = self._save_observation(observation)
        self._observation = observation
        self._seen_observations.add(observation.observation_id)
        self._last_action_id = None
        self.state = SessionState.RUNNING
        self._emit_observation(observation, reference)
        self.logger.emit(
            "resumed",
            page_id=observation.page_id,
            url=observation.url,
            observation_id=observation.observation_id,
            result={
                "old_observation_id": old_observation_id,
                "new_observation_id": observation.observation_id,
            },
        )
        return observation

    def finish(self, finish: Finish) -> None:
        if self.state is not SessionState.RUNNING:
            raise SessionStateError(f"cannot finish while session is {self.state.value}")
        if self._in_flight is not None:
            raise SessionStateError("cannot finish while an action is awaiting feedback")
        if self._proposed is not None:
            raise SessionStateError("cannot finish with an unconsumed proposed action")
        self.logger.emit(
            "agent_claim",
            page_id=self._page_id,
            url=None if self._observation is None else self._observation.url,
            observation_id=None if self._observation is None else self._observation.observation_id,
            result={
                "claim": finish.claim,
                "message": finish.message,
                "reason": finish.reason.value,
            },
        )
        self.finish_result = finish
        self.state = SessionState.FINISHED
        self._write_result(finish=finish)

    def cancel(self, reason: str = "cancelled") -> None:
        if self.state is SessionState.CLOSED:
            return
        scope = "in_flight_action" if self._in_flight is not None else (
            "proposed_action" if self._proposed is not None else "task"
        )
        action = self._in_flight or self._proposed
        self.logger.emit(
            "cancelled",
            page_id=self._page_id,
            url=None if self._observation is None else self._observation.url,
            observation_id=None if self._observation is None else self._observation.observation_id,
            action_id=None if action is None else action.action_id,
            action_type=None if action is None else action.op.value,
            result={"reason": str(reason), "scope": scope},
        )
        self._proposed = None
        self._in_flight = None
        self.state = SessionState.CANCELLED
        self._write_result(termination=TerminationReason.CANCELLED)

    def close(self) -> None:
        if self._closed:
            return
        if self._in_flight is not None:
            self.logger.emit(
                "cancelled",
                page_id=self._page_id,
                url=None if self._observation is None else self._observation.url,
                observation_id=None if self._observation is None else self._observation.observation_id,
                action_id=self._in_flight.action_id,
                action_type=self._in_flight.op.value,
                result={"reason": "close", "scope": "in_flight_action"},
            )
            self._in_flight = None
        if self._proposed is not None:
            self._proposed = None
        final_state = self.state
        if final_state not in (SessionState.FINISHED, SessionState.CANCELLED):
            self._write_result()
        self.logger.emit(
            "close",
            page_id=self._page_id,
            url=None if self._observation is None else self._observation.url,
            observation_id=None if self._observation is None else self._observation.observation_id,
            result={"final_state": final_state.value},
        )
        self.logger.emit(
            "run_end",
            result={
                "final_state": final_state.value,
                "termination": None
                if self.finish_result is None
                else self.finish_result.reason.value,
            },
        )
        self.logger.close()
        self.evidence.close()
        self.state = SessionState.CLOSED
        self._closed = True

    # --------------------------------------------------------------- utilities
    def resolve_target(self, action: WebAction) -> ElementRef | None:
        """Resolve an action against the current observation with full binding checks."""

        if self._observation is None:
            raise ActionRejected(
                RejectReason.NO_OBSERVATION,
                details=["no current observation"],
                correction="Wait for a current observation before resolving an action.",
            )
        return resolve_target(action, self._observation)

    def check_deadline(self, stage: str = "work") -> None:
        try:
            self.budget.check_deadline(stage)
        except DeadlineExceeded:
            if not self._deadline_logged:
                self._deadline_logged = True
                self.logger.emit(
                    "deadline",
                    page_id=self._page_id,
                    url=None if self._observation is None else self._observation.url,
                    observation_id=None if self._observation is None else self._observation.observation_id,
                    result={"stage": stage, "kind": "task_deadline"},
                )
            raise

    def record_model_call(
        self,
        *,
        request_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        token_count_source: str = "unknown",
        latency_ms: float | None = None,
        failure_type: str | None = None,
        model_name: str | None = None,
    ) -> Mapping[str, Any]:
        try:
            call_index = self.budget.begin_model_call()
        except DeadlineExceeded:
            self.check_deadline("model call")
            raise
        except BudgetExceeded as exc:
            self.log_exception("model_call", exc)
            raise
        self._model_calls = max(self._model_calls, call_index)
        observation = self._observation
        return self.logger.emit_model_call(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_count_source=token_count_source,
            latency_ms=latency_ms,
            failure_type=failure_type,
            model_name=model_name,
            page_id=self._page_id,
            url=None if observation is None else observation.url,
            observation_id=None if observation is None else observation.observation_id,
        )

    def log_exception(self, stage: str, exc: BaseException) -> Mapping[str, Any]:
        return self.logger.emit(
            "error",
            page_id=self._page_id,
            url=None if self._observation is None else self._observation.url,
            observation_id=None if self._observation is None else self._observation.observation_id,
            result={
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": self.redactor.redact_exception(exc),
            },
        )

    # ----------------------------------------------------------------- private
    def _require_running_for_action(self, operation: str) -> None:
        if self.state is SessionState.PAUSED:
            raise ActionRejected(
                RejectReason.PAUSED,
                details=[f"cannot {operation} while paused"],
                correction="The task is paused. Resume with a fresh observation before acting.",
            )
        if self.state is SessionState.CLOSED:
            raise ActionRejected(
                RejectReason.CLOSED,
                details=[f"cannot {operation} after close"],
                correction="The session is closed; no more actions can be proposed.",
            )
        if self.state is not SessionState.RUNNING:
            raise SessionStateError(f"cannot {operation} while session is {self.state.value}")

    def _validate_observation(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise ProtocolError("expected an Observation")
        if observation.run_id != self.run_id:
            raise SessionStateError(
                f"observation run {observation.run_id!r} does not belong to {self.run_id!r}"
            )
        if self._page_id is None:
            self._page_id = observation.page_id
        elif observation.page_id != self._page_id:
            raise SessionStateError(
                f"observation page {observation.page_id!r} does not match session page {self._page_id!r}"
            )

    def _save_observation(self, observation: Observation) -> str:
        try:
            return self.evidence.save_observation(observation)
        except EvidenceError as exc:
            self.log_exception("evidence_save_observation", exc)
            raise

    def _emit_observation(self, observation: Observation, reference: str) -> None:
        artifacts = observation.artifacts
        self.logger.emit(
            "observation",
            page_id=observation.page_id,
            url=observation.url,
            observation_id=observation.observation_id,
            result={
                "observation_ref": reference,
                "ax_chars": len(observation.ax_text),
                "element_count": len(observation.elements),
                "artifacts": {
                    kind: {
                        "available": artifact.available,
                        "reason": artifact.reason,
                        "reference": artifact.reference,
                    }
                    for kind in ("screenshot", "html", "dom")
                    if (artifact := artifacts.get(kind)) is not None
                },
            },
        )
        for kind in ("screenshot", "html", "dom"):
            artifact = artifacts.get(kind)
            if artifact is None or not artifact.available:
                continue
            self.logger.emit(
                "artifact_saved",
                page_id=observation.page_id,
                url=observation.url,
                observation_id=observation.observation_id,
                result={
                    "kind": artifact.kind,
                    "reference": artifact.reference,
                    "sha256": artifact.sha256,
                    "byte_size": artifact.byte_size,
                },
            )

    def _write_result(
        self,
        *,
        finish: Finish | None = None,
        termination: TerminationReason | None = None,
    ) -> None:
        observation = self._observation
        if termination is not None:
            termination_value = termination.value
        elif finish is not None:
            termination_value = finish.reason.value
        else:
            termination_value = None
        payload = {
            "schema_version": "webagent.result.v1",
            "run_id": self.run_id,
            "state": self.state.value,
            "task": self.task,
            "termination": termination_value,
            "finish": None if finish is None else finish.to_dict(),
            "last_observation": None
            if observation is None
            else {
                "page_id": observation.page_id,
                "observation_id": observation.observation_id,
                "url": observation.url,
            },
            "model_calls_total": self.model_calls_total,
            "time": self.clock.iso(self.clock.now()),
        }
        self.evidence.write_result(self.redactor.redact(payload))

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
