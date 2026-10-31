"""Connect a stagnation signal to a bounded, evidence-based recovery attempt.

A signal is only a diagnostic candidate. This coordinator records it as a
``no_progress`` failure, asks the planner for the affected closure and spends one
bounded recovery attempt. It does not execute web actions; the normal action loop
owns the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..state.records import EvidenceRef
from .controller import RecoveryController
from .planner import RecoveryPlanner
from .stagnation import StagnationDetector, StagnationSignal
from .types import (
    FailureEvidence,
    RecoveryAttempt,
    RecoveryClass,
    RecoveryPlan,
    RecoveryResult,
)


@dataclass(frozen=True)
class StagnationRecoveryResult:
    signal: StagnationSignal
    recovery: RecoveryResult


@dataclass(frozen=True)
class RecoveryOutcome:
    """One planned, authorized and applied recovery application."""

    plan: RecoveryPlan
    attempt: RecoveryAttempt
    result: RecoveryResult

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "attempt": self.attempt.to_dict(),
            "result": self.result.to_dict(),
        }


class RecoveryCoordinator:
    def __init__(
        self,
        planner: RecoveryPlanner,
        controller: RecoveryController,
        detector: StagnationDetector,
    ) -> None:
        self.planner = planner
        self.controller = controller
        self.detector = detector

    def handle_failures(
        self,
        failures: Sequence[FailureEvidence],
        *,
        evidence: Sequence[EvidenceRef] = (),
        note: str = "",
        source_branch: str = "root",
    ) -> RecoveryOutcome | None:
        """Plan and apply one evidence-anchored correction.

        This is the production path shared by a refused completion check and an
        anomalous action effect. Verified claims inside the reverse dependency
        closure are withdrawn through the programmatic ledger transition; an
        unrelated verified claim is preserved. A plan without anchor evidence
        changes no completion claim and spends no attempt.
        """

        if not failures:
            return None
        plan = self.planner.plan(failures)
        targets = plan.affected_goal_ids or plan.invalidated_goal_ids
        attempt = (
            self.controller.authorize(targets)
            if targets
            else RecoveryAttempt(True, "no affected goal to authorize")
        )
        result = self.controller.withdraw(
            plan,
            attempt=attempt,
            evidence=tuple(evidence),
            note=note,
            source_branch=source_branch,
        )
        return RecoveryOutcome(plan, attempt, result)

    def handle_stagnation(
        self,
        signal: StagnationSignal | None,
        *,
        goal_id: str,
        action_id: str,
        observation_id: str,
        sequence: int = 0,
        evidence: Sequence[EvidenceRef] = (),
    ) -> StagnationRecoveryResult | None:
        if signal is None:
            return None
        failure = FailureEvidence(
            action_id=action_id,
            sequence=sequence,
            failure_class=RecoveryClass.NO_PROGRESS,
            goal_id=goal_id,
            message=signal.reason,
            observation_id=observation_id,
            evidence=tuple(evidence),
        )
        plan = self.planner.plan([failure])
        attempt = self.controller.authorize(plan.affected_goal_ids or (goal_id,))
        recovery = self.controller.apply(plan, attempt=attempt)
        return StagnationRecoveryResult(signal, recovery)


__all__ = [
    "RecoveryCoordinator",
    "RecoveryOutcome",
    "StagnationRecoveryResult",
]

