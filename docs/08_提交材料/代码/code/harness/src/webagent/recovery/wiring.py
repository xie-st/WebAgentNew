"""Production recovery wiring for the official entry (phase-2 R wave, E03/E04).

This module connects the already-implemented recovery pieces to the official
step loop; it does not reimplement them:

* :class:`~webagent.recovery.planner.RecoveryPlanner` finds the earliest
  evidence-anchored failure and the reverse dependency closure;
* :class:`~webagent.recovery.controller.RecoveryController` owns the per-goal and
  whole-task attempt limits plus the absolute deadline;
* :meth:`RecoveryController.withdraw` applies the plan through the programmatic
  ledger transition, so a verified claim inside the closure is withdrawn with its
  evidence kept while unrelated verified progress is preserved;
* :class:`~webagent.recovery.stagnation.StagnationDetector` decides whether a
  repeated action produced new evidence or reduced the outstanding gap.

Everything here is driven by normal observations and the ledger. It never calls a
model, never navigates and never invents a cause: a plan without anchor evidence
changes no completion claim, and a denied attempt ends the recovery path instead
of looping. The caller (``OfficialAgent``) decides how a returned note reaches the
model or how a terminal reason ends the run honestly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..clock import Clock, SystemClock
from ..config import RecoverySettings
from ..protocol import Observation, WebAction, FeedbackStatus
from ..state.integration import evidence_ref_from_artifact
from ..state.records import EvidenceRef, GoalStatus
from ..state.store import StateStore
from ..verification.types import VerificationStatus
from .classify import classify_feedback
from .controller import RecoveryController, RecoveryLimits
from .coordinator import RecoveryCoordinator, RecoveryOutcome
from .planner import RecoveryPlanner
from .stagnation import StagnationDetector
from .types import FailureEvidence, RecoveryClass

_UNVERIFIED = frozenset(
    {
        GoalStatus.PENDING,
        GoalStatus.RUNNING,
        GoalStatus.AWAITING_VERIFICATION,
        GoalStatus.INVALIDATED,
    }
)


@dataclass(frozen=True)
class RecoveryNote:
    """What one recovery decision adds to the run.

    ``text`` is a bounded program note for the next model turn. ``handoff_reason``
    asks the caller to hand the browser to the user (authentication feedback).
    ``terminal_reason`` ends the run honestly: the bounded recovery path cannot
    continue, so no completion is claimed.
    """

    text: str = ""
    handoff_reason: str | None = None
    terminal_reason: str | None = None

    @property
    def terminal(self) -> bool:
        return self.terminal_reason is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "handoff_reason": self.handoff_reason,
            "terminal_reason": self.terminal_reason,
        }


class OfficialRecoveryRuntime:
    """One task's bounded recovery decisions, with a real ledger and deadline."""

    def __init__(
        self,
        store: StateStore,
        logger: Any = None,
        *,
        branch: str = "root",
        settings: RecoverySettings | None = None,
        clock: Clock | None = None,
        deadline_epoch: float | None = None,
    ) -> None:
        if not isinstance(store, StateStore):
            raise TypeError("OfficialRecoveryRuntime needs a StateStore")
        settings = settings or RecoverySettings()
        self.store = store
        self.logger = logger
        self.branch = branch
        self.clock = clock or SystemClock()
        self.settings = settings
        self.limits = RecoveryLimits(
            max_recoveries_per_goal=settings.max_recoveries_per_goal,
            max_recoveries_total=settings.max_recoveries_total,
            deadline_epoch=deadline_epoch,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            reflection_limit=settings.reflection_limit,
        )
        self.controller = RecoveryController(store, limits=self.limits, clock=self.clock)
        self.detector = StagnationDetector(
            window=settings.stagnation_window,
            min_repeated_states=settings.stagnation_repeats,
            clock=self.clock,
        )
        self.coordinator = RecoveryCoordinator(
            RecoveryPlanner(store), self.controller, self.detector
        )
        self._sequence = 0
        self._fact_revisions: dict[str, int] = {}
        self.applied = 0
        self.denied = 0
        self.last_reason = ""

    # ------------------------------------------------------------------ ledger
    @property
    def attempts(self) -> dict[str, Any]:
        return {
            "total": self.controller.total_attempts,
            "per_goal": dict(self.controller.goal_attempts),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "branch": self.branch,
            "takeover": True,
            "pause_control": True,
            "stagnation": {
                "window": self.settings.stagnation_window,
                "repeats": self.settings.stagnation_repeats,
            },
            "limits": {
                "max_recoveries_per_goal": self.settings.max_recoveries_per_goal,
                "max_recoveries_total": self.settings.max_recoveries_total,
                "retry_backoff_seconds": list(self.settings.retry_backoff_seconds),
                "reflection_limit": self.settings.reflection_limit,
                "deadline_epoch": self.limits.deadline_epoch,
            },
            "attempts": self.attempts,
            "applied": self.applied,
            "denied": self.denied,
            "last_reason": self.last_reason,
        }

    # -------------------------------------------------------- completion refusal
    def after_completion_refusal(
        self,
        decision: Any,
        observation: Observation | None,
    ) -> RecoveryNote:
        """Withdraw the completion claims a failed check invalidated."""

        failures: list[FailureEvidence] = []
        evidence: list[EvidenceRef] = []
        for result in getattr(decision, "results", ()) or ():
            if getattr(result, "status", None) is not VerificationStatus.FAIL:
                continue
            self._sequence += 1
            refs = tuple(getattr(result, "evidence", ()) or ())
            evidence.extend(refs)
            failures.append(
                FailureEvidence(
                    action_id=f"verify-{result.goal_id}",
                    sequence=self._sequence,
                    failure_class=RecoveryClass.CONDITION_FAILED,
                    goal_id=result.goal_id,
                    message=str(getattr(result, "reason", "") or "condition failed"),
                    observation_id=(
                        None if observation is None else observation.observation_id
                    ),
                    evidence=refs,
                )
            )
        if not failures:
            return RecoveryNote()
        outcome = self.coordinator.handle_failures(
            failures,
            evidence=tuple(evidence),
            note="completion check refused; withdrawing affected completion claims",
            source_branch=self.branch,
        )
        return self._describe(
            outcome,
            stage="completion_refusal",
            observation=observation,
            prefix=(
                "Recovery: the completion check failed on "
                + ", ".join(sorted({failure.goal_id or "?" for failure in failures}))
                + ". "
            ),
        )

    # ------------------------------------------------------------ new feedback
    def after_feedback(
        self,
        action: WebAction | None,
        observation: Observation,
    ) -> RecoveryNote:
        """Decide what one executed action's feedback means for the ledger."""

        feedback = observation.feedback
        if feedback is None or action is None:
            return RecoveryNote()
        classification = classify_feedback(
            feedback,
            action_op=action.op,
            page_text=observation.ax_text,
            url=observation.url,
        )
        if classification is RecoveryClass.AUTHENTICATION:
            self.last_reason = "authentication feedback requires user takeover"
            self._emit(
                "authentication_feedback",
                observation,
                {
                    "stage": "authentication_feedback",
                    "failure_class": classification.value,
                    "action_id": action.action_id,
                    "message": feedback.message,
                    "requires_user_takeover": True,
                },
            )
            return RecoveryNote(
                text=(
                    "Recovery: the platform reported authentication/verification "
                    "feedback for the last action. The task pauses for the user; no "
                    "credentials or verification codes are filled in by the harness."
                ),
                handoff_reason="authentication_feedback",
            )

        evidence = self._observation_evidence(observation)
        failure_note = RecoveryNote()
        handled_failure = False
        if feedback.status is not FeedbackStatus.OBSERVED:
            handled_failure = True
            self._sequence += 1
            goal_id = self._active_goal()
            failure = FailureEvidence(
                action_id=action.action_id,
                sequence=self._sequence,
                failure_class=classification,
                goal_id=goal_id,
                message=feedback.message,
                observation_id=observation.observation_id,
                evidence=evidence,
            )
            outcome = self.coordinator.handle_failures(
                [failure],
                evidence=evidence,
                note=f"action effect anomaly ({classification.value})",
                source_branch=self.branch,
            )
            failure_note = self._describe(
                outcome,
                stage="action_failure",
                observation=observation,
                prefix=(
                    f"Recovery: the last action's effect is not confirmed "
                    f"({classification.value}). "
                ),
            )
            if failure_note.terminal:
                return failure_note

        progress = self._record_progress(action, observation)
        if handled_failure:
            # A failure already spent this step's recovery decision; the progress
            # record still updated the window so a later stall is measured.
            return failure_note
        signal = self.detector.evaluate()
        if signal is None:
            return failure_note
        stagnation_note = self._handle_stagnation(
            signal, action=action, observation=observation, evidence=evidence
        )
        return stagnation_note if stagnation_note.text or stagnation_note.terminal else failure_note

    # ------------------------------------------------------------- internals
    def _handle_stagnation(self, signal: Any, *, action, observation, evidence) -> RecoveryNote:
        goal_id = self._active_goal()
        if goal_id is None:
            attempt = self.controller.authorize(["task"])
            if not attempt.allowed:
                self.denied += 1
                self.last_reason = attempt.reason
                self._emit(
                    "stagnation_limit",
                    observation,
                    {
                        "stage": "stagnation_limit",
                        "signal": signal.to_dict(),
                        "applied": False,
                        "allowed": False,
                        "reason": attempt.reason,
                    },
                )
                return RecoveryNote(terminal_reason=attempt.reason)
            self._emit(
                "stagnation_unattributed",
                observation,
                {
                    "stage": "stagnation_unattributed",
                    "signal": signal.to_dict(),
                    "applied": False,
                    "reason": "no single unfinished goal can anchor the correction",
                },
            )
            return RecoveryNote(
                text=(
                    "Recovery: the last actions repeated without new evidence. "
                    "No single unfinished requirement anchors a correction, so "
                    "change one step for the current requirement instead of "
                    "repeating the same action."
                )
            )
        self._sequence += 1
        result = self.coordinator.handle_stagnation(
            signal,
            goal_id=goal_id,
            action_id=action.action_id if action is not None else "task",
            observation_id=observation.observation_id,
            sequence=self._sequence,
            evidence=evidence,
        )
        if result is None:
            return RecoveryNote()
        recovery = result.recovery
        self._emit(
            "stagnation",
            observation,
            {
                "stage": (
                    "stagnation" if recovery.attempt.allowed else "stagnation_limit"
                ),
                "signal": signal.to_dict(),
                "goal_id": goal_id,
                "applied": recovery.applied,
                "allowed": recovery.attempt.allowed,
                "attempt": recovery.attempt.to_dict(),
                "plan": recovery.plan.to_dict(),
                "reason": recovery.reason,
            },
        )
        if not recovery.attempt.allowed:
            self.denied += 1
            self.last_reason = recovery.attempt.reason
            return RecoveryNote(terminal_reason=recovery.attempt.reason)
        if not recovery.applied:
            self.last_reason = recovery.reason
            return RecoveryNote(
                text=(
                    "Recovery: the repeated step could not be corrected in the "
                    f"ledger ({recovery.reason}). Re-observe the affected object "
                    "before deciding; do not repeat the same action."
                )
            )
        self.applied += 1
        self.last_reason = recovery.plan.next_step
        self.controller.record_reflection(
            f"{signal.reason}; adjustment: {recovery.plan.next_step}"
        )
        return RecoveryNote(
            text=(
                f"Recovery: {signal.reason}. {recovery.plan.next_step}. "
                f"Requirement {goal_id} must be re-obtained from the page; "
                "unrelated verified requirements are preserved."
            )
        )

    def _describe(
        self,
        outcome: RecoveryOutcome | None,
        *,
        stage: str,
        observation: Observation | None,
        prefix: str,
    ) -> RecoveryNote:
        if outcome is None:
            return RecoveryNote()
        plan = outcome.plan
        result = outcome.result
        payload = {
            "stage": stage,
            "applied": result.applied,
            "allowed": outcome.attempt.allowed,
            "plan": plan.to_dict(),
            "attempt": outcome.attempt.to_dict(),
            "result": result.to_dict(),
            "withdrawn_goal_ids": list(plan.invalidated_goal_ids),
            "preserved_verified_goal_ids": list(plan.preserved_verified_goal_ids),
            "next_step": plan.next_step,
            "evidence_sufficient": plan.evidence_sufficient,
        }
        self._emit(stage, observation, payload)
        if not outcome.attempt.allowed:
            self.denied += 1
            self.last_reason = outcome.attempt.reason
            return RecoveryNote(
                text=(
                    prefix
                    + "the bounded recovery budget refused another attempt "
                    f"({outcome.attempt.reason}). The task ends without claiming "
                    "completion."
                ),
                terminal_reason=outcome.attempt.reason,
            )
        if not result.applied:
            self.last_reason = result.reason
            if not plan.evidence_sufficient:
                return RecoveryNote(
                    text=(
                        prefix
                        + "the available evidence does not support a cause, so no "
                        "completion claim was changed. Collect page evidence for "
                        "the affected requirement before deciding again."
                    )
                )
            return RecoveryNote(
                text=(
                    prefix
                    + f"the ledger did not apply the correction ({result.reason}). "
                    "Re-observe the affected object and do not invent a cause."
                )
            )
        self.applied += 1
        self.last_reason = plan.next_step
        withdrawn = ", ".join(plan.invalidated_goal_ids) or "none"
        preserved = ", ".join(plan.preserved_verified_goal_ids) or "none"
        return RecoveryNote(
            text=(
                prefix
                + f"withdrawn completion claims: {withdrawn}; preserved unrelated "
                f"verified requirements: {preserved}. {plan.next_step}. Redo only "
                "the withdrawn or remaining requirements; do not repeat unrelated "
                "verified work and do not treat the withdrawn evidence as removed "
                "(it stays in the ledger with the new conflicting evidence)."
            )
        )

    def _record_progress(self, action: WebAction, observation: Observation) -> None:
        state = self.store.state
        gap = len([goal for goal in state.goals.values() if goal.status in _UNVERIFIED])
        new_fact_ids: list[str] = []
        for fact_id, fact in state.facts.items():
            previous = self._fact_revisions.get(fact_id, 0)
            if fact.revision > previous:
                new_fact_ids.append(fact_id)
            self._fact_revisions[fact_id] = fact.revision
        digest = hashlib.sha256(observation.ax_text.encode("utf-8")).hexdigest()[:12]
        self.detector.record(
            action_key=self._action_key(action),
            state_key=f"{observation.url}:{digest}",
            observation_id=observation.observation_id,
            new_evidence_ids=new_fact_ids,
            gap_before=gap,
            gap_after=gap,
            note="official step feedback",
        )

    @staticmethod
    def _action_key(action: WebAction) -> str:
        target = action.target or ""
        text = (action.text or "")[:60]
        url = action.url or ""
        return f"{action.op.value}:{target}:{text}:{url}"

    def _active_goal(self) -> str | None:
        unfinished = [
            goal_id
            for goal_id, goal in sorted(self.store.state.goals.items())
            if goal.status in _UNVERIFIED
        ]
        return unfinished[0] if len(unfinished) == 1 else None

    def _observation_evidence(self, observation: Observation) -> tuple[EvidenceRef, ...]:
        refs: list[EvidenceRef] = []
        for kind in ("screenshot", "html", "dom"):
            artifact = observation.artifacts.get(kind)
            if artifact is None or not artifact.available:
                continue
            try:
                refs.append(
                    evidence_ref_from_artifact(
                        artifact,
                        run_id=observation.run_id,
                        branch_id=self.branch,
                        evidence_id=f"ev-{observation.observation_id}-{kind}",
                        observation_id=observation.observation_id,
                    )
                )
            except Exception:  # noqa: BLE001 - an unusable citation is simply absent
                continue
        return tuple(refs)

    def _emit(self, stage: str, observation: Observation | None, payload: Mapping[str, Any]) -> None:
        if self.logger is None:
            return
        try:
            self.logger.emit(
                "recovery",
                page_id=None if observation is None else observation.page_id,
                url=None if observation is None else observation.url,
                observation_id=None
                if observation is None
                else observation.observation_id,
                goal_id=payload.get("goal_id"),
                result={**dict(payload), "wiring_module": "recovery.wiring"},
            )
        except Exception:  # noqa: BLE001 - recording cannot change the decision
            pass


__all__ = ["OfficialRecoveryRuntime", "RecoveryNote"]
