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
from typing import Any, Callable, Mapping, Sequence

from .budget import Budget, BudgetLimits, Deadline, DeadlineMarginReached
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
from .events import EventLogger, RunStatusWriter
from .evidence import EvidenceStore
from .protocol import (
    ActionOp,
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
        action_gate: Callable[[str], None] | None = None,
    ) -> None:
        validate_run_id(run_id)
        self.run_id = run_id
        self.clock = clock or SystemClock()
        self.redactor = redactor or DEFAULT_REDACTOR
        self.evidence = evidence or EvidenceStore(runtime_dir, run_id)
        self._status_writer = RunStatusWriter(
            run_id,
            self.evidence.run_dir / "run_status.json",
            clock=self.clock,
            redactor=self.redactor,
        )
        self.logger = logger or EventLogger(
            run_id,
            self.evidence.events_path,
            clock=self.clock,
            redactor=self.redactor,
            model_calls_total=lambda: self._model_calls,
            on_event=self._update_run_status,
        )
        self.budget = budget or Budget(limits, clock=self.clock)
        self._page_id = page_id
        self._initial_page_id = page_id
        self.state = SessionState.NEW
        self.task: str | None = None
        self.finish_result: Finish | None = None
        self.termination: TerminationReason | None = None
        self._termination_note: str | None = None
        self._observation: Observation | None = None
        self._seen_observations: set[str] = set()
        self._proposed: WebAction | None = None
        self._in_flight: WebAction | None = None
        self._consumed: set[str] = set()
        self._last_action_id: str | None = None
        self._last_action_state: dict[str, Any] | None = None
        self._notes: list[str] = []
        self._model_calls = 0
        self._model_attempts = 0
        self._model_failures = 0
        self._platform_request_ids: set[str] = set()
        self._proposed_count = 0
        self._dispatched_count = 0
        self._completed_steps = 0
        self._deadline_logged = False
        self._margin_logged = False
        self._closed = False
        self._action_gate = action_gate
        self._runtime_summary: dict[str, Any] = {}
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

    def set_action_gate(self, gate: Callable[[str], None] | None) -> None:
        """Install the dispatch gate used by the phase-2 control layer.

        The gate is consulted in :meth:`consume` only. A proposal may still be
        recorded when a model result returns during a pause (the design requires
        the unconsumed proposal to be dropped explicitly, not format-corrected),
        but ``consume`` raises before anything reaches a backend. A gate must
        raise :class:`~webagent.errors.ActionRejected` to refuse.
        """

        self._action_gate = gate

    def add_task_note(self, note: str) -> None:
        """Store a task-local fact for internal ``task_note`` tools."""

        if not isinstance(note, str) or not note.strip():
            raise ProtocolError("task note must be a non-empty string")
        self._notes.append(note)

    def set_runtime_summary(self, summary: Mapping[str, Any]) -> None:
        """Attach the entry point's actual runtime switches to the result summary.

        ``result.json`` is written from :meth:`result_summary`; recording the
        switches here means the persisted run record says which phase-2
        behaviors were really enabled rather than only what the config named.
        """

        if not isinstance(summary, Mapping):
            raise ProtocolError("runtime summary must be a mapping")
        self._runtime_summary = dict(summary)

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
        self._last_action_state = None
        self._notes.clear()
        self._deadline_logged = False
        self._margin_logged = False
        self.termination = None
        self._termination_note = None
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
            self._completed_steps += 1
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
        self._proposed_count += 1
        return action

    def expand_proposal(
        self,
        action_id: str,
        sub_actions: Sequence[WebAction],
    ) -> tuple[WebAction, ...]:
        """Replace one proposed action with its ordered sub-action sequence.

        The proposed action is never dispatched: the execution layer hands each
        sub-action to the platform in a later step. Every sub-action is validated
        against the observation the proposal was built from before the proposal is
        dropped, so an invalid plan changes nothing.
        """

        self._require_running_for_action("expand")
        self.check_deadline("action expansion")
        if self._in_flight is not None:
            raise SessionStateError("cannot expand while an action is awaiting feedback")
        proposed = self._proposed
        if proposed is None or proposed.action_id != action_id:
            raise SessionStateError("expand needs the currently proposed action")
        subs = tuple(sub_actions)
        if not subs:
            raise ProtocolError("an expanded proposal needs at least one sub-action")
        observation = self._observation
        if observation is None:
            raise ActionRejected(
                RejectReason.NO_OBSERVATION,
                details=["no current observation"],
                correction="Wait for a current observation before expanding an action.",
            )
        for sub in subs:
            if not isinstance(sub, WebAction):
                raise ProtocolError("sub-actions must be WebAction values")
            if sub.action_id in self._consumed:
                raise ActionRejected(
                    RejectReason.DUPLICATE_CONSUMPTION,
                    details=[f"sub-action {sub.action_id!r} was already dispatched"],
                    correction="Build a fresh expansion; a consumed action is never replayed.",
                )
            resolve_target(sub, observation)
        self.logger.emit(
            "expanded",
            page_id=proposed.page_id,
            url=observation.url,
            observation_id=proposed.observation_id,
            action_id=proposed.action_id,
            action_type=proposed.op.value,
            result={
                "reason": "line_break_split",
                "parent_action_id": proposed.action_id,
                "sub_action_count": len(subs),
                "sub_actions": [
                    {
                        "index": index,
                        "op": sub.op.value,
                        "target": sub.target,
                        "keys": sub.keys,
                        "text_chars": len(sub.text) if sub.text is not None else None,
                    }
                    for index, sub in enumerate(subs)
                ],
            },
        )
        self._proposed = None
        return subs

    def drop_proposal(self, reason: str = "dropped") -> None:
        """Drop the unconsumed proposal without dispatching it."""

        if self.state is not SessionState.RUNNING:
            return
        if self._in_flight is not None or self._proposed is None:
            return
        dropped = self._proposed
        self.logger.emit(
            "cancelled",
            page_id=dropped.page_id,
            url=None if self._observation is None else self._observation.url,
            observation_id=dropped.observation_id,
            action_id=dropped.action_id,
            action_type=dropped.op.value,
            result={"reason": str(reason), "scope": "proposed_action"},
        )
        self._proposed = None

    def consume(self, action_id: str | None = None) -> WebAction:
        self._require_running_for_action("consume")
        if self._action_gate is not None:
            # The phase-2 control gate owns "may this page action be sent now?".
            # It raises ActionRejected(PAUSED/CLOSED) and leaves the proposal
            # unconsumed so the caller can drop it explicitly.
            self._action_gate("consume")
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
        self._dispatched_count += 1
        self._last_action_state = {
            "action_id": action.action_id,
            "op": action.op.value,
            "target": action.target,
            "keys": action.keys,
            "text_chars": len(action.text) if action.text is not None else None,
            "text_preview": (action.text or "")[:120] if action.text is not None else None,
            "submit": action.submit if action.op is ActionOp.TYPE else None,
        }
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
        self.termination = finish.reason
        self._termination_note = finish.message or finish.claim
        self.state = SessionState.FINISHED
        self._write_result(finish=finish)

    def finish_incomplete(self, finish: Finish) -> None:
        """Finish when a dispatched action's observation could not be used.

        Used by the split text path: if the platform's next observation cannot
        be ingested, the dispatched sub-action is recorded as unanswered instead
        of pretending it succeeded, and the task still gets a terminal result.
        """

        if self.state is not SessionState.RUNNING:
            raise SessionStateError(f"cannot finish while session is {self.state.value}")
        action = self._in_flight or self._proposed
        if action is not None:
            self.logger.emit(
                "cancelled",
                page_id=action.page_id,
                url=None if self._observation is None else self._observation.url,
                observation_id=action.observation_id,
                action_id=action.action_id,
                action_type=action.op.value,
                result={
                    "reason": finish.reason.value,
                    "scope": "in_flight_action"
                    if self._in_flight is not None
                    else "proposed_action",
                },
            )
        self._in_flight = None
        self._proposed = None
        self.finish(finish)

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
        self.termination = TerminationReason.CANCELLED
        self._termination_note = str(reason)
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
                if self.termination is None
                else self.termination.value,
                "summary": self.result_summary(),
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

    def check_model_call_margin(self, stage: str = "model call") -> None:
        """Refuse a new model call when the deadline safety margin is reached."""

        try:
            self.budget.require_model_call_margin(stage)
        except DeadlineMarginReached as exc:
            if not self._margin_logged:
                self._margin_logged = True
                self.logger.emit(
                    "deadline_margin",
                    page_id=self._page_id,
                    url=None if self._observation is None else self._observation.url,
                    observation_id=None
                    if self._observation is None
                    else self._observation.observation_id,
                    result={
                        "stage": stage,
                        "remaining_seconds": round(exc.remaining_seconds, 3),
                        "margin_seconds": exc.margin_seconds,
                        "action": "controlled_closeout",
                    },
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
        self._model_attempts += 1
        if failure_type:
            self._model_failures += 1
        if isinstance(request_id, str) and request_id:
            self._platform_request_ids.add(request_id)
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
        elif self.termination is not None:
            termination_value = self.termination.value
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
            "summary": self.result_summary(),
            "model_calls_total": self.model_calls_total,
            "time": self.clock.iso(self.clock.now()),
        }
        self.evidence.write_result(self.redactor.redact(payload))

    def result_summary(self) -> dict[str, Any]:
        """Step/action totals and separate model-call counters for reconciliation."""

        summary = {
            "steps_completed": self._completed_steps,
            "observations": len(self._seen_observations),
            "actions_proposed": self._proposed_count,
            "actions_dispatched": self._dispatched_count,
            "last_action": self._last_action_state,
            "unfinished": self._termination_note,
            "model_calls": self.model_call_counts(),
        }
        if self._runtime_summary:
            summary["runtime"] = dict(self._runtime_summary)
        return summary

    def model_call_counts(self) -> dict[str, int]:
        """Three separate counts; a single total would hide what differs locally.

        ``platform_calls`` is what this process saw the platform accept (a
        distinct platform ``request_id`` was returned). ``local_attempts`` counts
        every attempted SDK call, including refused and failed ones.
        ``failures`` counts attempts with a failure type. The platform's own batch
        number stays authoritative; these are local, reconcilable values.
        """

        return {
            "platform_calls": len(self._platform_request_ids),
            "local_attempts": self._model_attempts,
            "failures": self._model_failures,
        }

    def run_status(self, last_event: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The lightweight state persisted after every event."""

        status: dict[str, Any] = {
            "run_id": self.run_id,
            "state": self.state.value,
            "task_started": self.task is not None,
            "termination": None if self.termination is None else self.termination.value,
            "steps": {
                "completed": self._completed_steps,
                "proposed": self._proposed_count,
                "dispatched": self._dispatched_count,
            },
            "last_action": self._last_action_state,
            "model_calls": self.model_call_counts(),
            "model_calls_total": self.model_calls_total,
        }
        if last_event is not None:
            status["last_event"] = {
                "event_type": last_event.get("event_type"),
                "time": last_event.get("time"),
                "action_id": last_event.get("action_id"),
                "action_type": last_event.get("action_type"),
                "observation_id": last_event.get("observation_id"),
                "model_calls_total": last_event.get("model_calls_total"),
            }
        return status

    def _update_run_status(self, event: Mapping[str, Any]) -> None:
        # A status write must never break the run or the event log it describes.
        try:
            self._status_writer.write(self.run_status(event))
        except Exception:  # noqa: BLE001 - best-effort local state file
            pass

    def status_path(self) -> Path:
        return self._status_writer.path

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
