"""Local recovery planning from the earliest supported failure.

The planner never calls the site or a model. It reads the ledger's goal graph,
chooses the earliest failure that has an action/fact/observation anchor, computes
the reverse dependency closure and names only the completion claims that must be
withdrawn. If the evidence does not support a cause, it returns an insufficient
plan instead of inventing one.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..state.records import GoalStatus, TaskState
from ..state.store import StateStore
from .types import (
    FailureEvidence,
    RecoveryAnchor,
    RecoveryClass,
    RecoveryError,
    RecoveryPlan,
)

_NEXT_STEP = {
    RecoveryClass.PROTOCOL: "correct the action/tool format, then re-observe before acting",
    RecoveryClass.TRANSIENT: "inspect the affected page state, then retry only the affected step",
    RecoveryClass.UNKNOWN_EFFECT: (
        "re-observe the affected object's visible state before deciding whether to retry"
    ),
    RecoveryClass.CONDITION_FAILED: "repair only the failed condition and re-run verification",
    RecoveryClass.NO_PROGRESS: "change one step for the current goal; do not repeat the loop",
    RecoveryClass.AUTHENTICATION: "pause the task for user takeover; do not keep probing",
    RecoveryClass.USER_PAUSE: "wait for the user's explicit continue signal",
    RecoveryClass.UNKNOWN: "collect more page evidence before assigning a cause",
}


def reverse_dependency_closure(state: TaskState, seed_goal_ids: Iterable[str]) -> tuple[str, ...]:
    """Goals affected by the seeds, including transitive reverse dependencies."""

    selected = {str(goal_id) for goal_id in seed_goal_ids}
    if not selected:
        return ()
    unknown = selected - set(state.goals)
    if unknown:
        raise RecoveryError("closure names unknown goals: " + ", ".join(sorted(unknown)))
    affected = set(selected)
    changed = True
    while changed:
        changed = False
        for goal_id, goal in state.goals.items():
            if goal_id in affected:
                continue
            if set(goal.depends_on) & affected:
                affected.add(goal_id)
                changed = True
    return tuple(sorted(affected))


class RecoveryPlanner:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def earliest_anchor(
        self,
        failures: Sequence[FailureEvidence],
    ) -> tuple[RecoveryAnchor | None, str]:
        """Pick the earliest failure with an evidenced anchor; else say why not."""

        ordered = sorted(failures, key=lambda item: (item.sequence, item.action_id))
        if not ordered:
            return None, "no failure evidence was supplied"
        for failure in ordered:
            if not failure.has_anchor_evidence:
                continue
            if failure.fact_id is not None:
                anchor = RecoveryAnchor(
                    "fact",
                    failure.fact_id,
                    failure.goal_id,
                    failure.message or f"{failure.failure_class.value} fact conflict",
                    failure.sequence,
                    failure.evidence,
                )
                return anchor, f"earliest supported fact conflict: {failure.fact_id}"
            anchor = RecoveryAnchor(
                "action",
                failure.action_id,
                failure.goal_id,
                failure.message or f"{failure.failure_class.value} action outcome",
                failure.sequence,
                failure.evidence,
            )
            return anchor, f"earliest supported action outcome: {failure.action_id}"
        return None, "failure records have no action, fact, observation or evidence anchor"

    def plan(self, failures: Sequence[FailureEvidence]) -> RecoveryPlan:
        state = self.store.state
        anchor, anchor_reason = self.earliest_anchor(failures)
        verified = set(state.verified_goal_ids)
        if anchor is None:
            return RecoveryPlan(
                anchor=None,
                affected_goal_ids=(),
                invalidated_goal_ids=(),
                preserved_verified_goal_ids=tuple(sorted(verified)),
                next_step="collect more evidence before changing any completion claim",
                reason=anchor_reason,
                evidence_sufficient=False,
            )

        start: set[str] = set()
        if anchor.goal_id is not None:
            start.add(anchor.goal_id)
        if anchor.anchor_type == "fact":
            fact = state.facts.get(anchor.anchor_id)
            if fact is not None and fact.goal_id is not None:
                start.add(fact.goal_id)
        if not start:
            return RecoveryPlan(
                anchor=anchor,
                affected_goal_ids=(),
                invalidated_goal_ids=(),
                preserved_verified_goal_ids=tuple(sorted(verified)),
                next_step="identify which requirement the affected object belongs to",
                reason="anchor has no goal attribution; no completion claim was changed",
                evidence_sufficient=False,
            )

        affected = reverse_dependency_closure(state, start)
        invalidated = tuple(
            goal_id
            for goal_id in affected
            if state.goals[goal_id].status
            in (GoalStatus.VERIFIED, GoalStatus.AWAITING_VERIFICATION)
        )
        preserved = tuple(sorted(verified - set(invalidated)))
        failure_class = next(
            (
                failure.failure_class
                for failure in sorted(failures, key=lambda item: (item.sequence, item.action_id))
                if failure.action_id == anchor.anchor_id or failure.fact_id == anchor.anchor_id
            ),
            RecoveryClass.UNKNOWN,
        )
        return RecoveryPlan(
            anchor=anchor,
            affected_goal_ids=affected,
            invalidated_goal_ids=invalidated,
            preserved_verified_goal_ids=preserved,
            next_step=_NEXT_STEP[failure_class],
            reason=(
                f"{anchor_reason}; affected closure={list(affected)}; "
                f"withdraw={list(invalidated)}; preserve={list(preserved)}"
            ),
            evidence_sufficient=True,
        )


__all__ = ["RecoveryPlanner", "reverse_dependency_closure"]
