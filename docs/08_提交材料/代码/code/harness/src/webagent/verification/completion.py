"""Finish-time completion checking.

A model may stop after handling only some objects. The completion checker runs
the same evidence-backed verification path over the requested goals and refuses
to confirm completion while any requirement remains. The refusal carries the
remaining goal ids and conditions back to the caller so the same EffGen loop can
continue, or the run can end honestly as ``incomplete``.

The checker never navigates, captures or downloads anything. If a condition
needs new page evidence, the normal action loop must obtain it first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..state.store import StateStore
from .types import (
    CompletionDecision,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)
from .verifier import GoalVerifier


@dataclass(frozen=True)
class CompletionCheck:
    """The small callback object accepted by an EffGen driver finish gate."""

    checker: "CompletionChecker"
    requests: tuple[VerificationRequest, ...]
    commit: bool = True

    def __call__(self, step: object = None) -> CompletionDecision:
        return self.checker.check(self.requests, commit=self.commit)


class CompletionChecker:
    """Verify every requirement before allowing a complete finish."""

    def __init__(self, verifier: GoalVerifier, store: StateStore | None = None) -> None:
        self.verifier = verifier
        self.store = store or verifier.store

    def check(
        self,
        requests: Sequence[VerificationRequest],
        *,
        commit: bool = True,
        goal_ids: Sequence[str] | None = None,
    ) -> CompletionDecision:
        state = self.store.state
        selected = (
            tuple(sorted(state.goals))
            if goal_ids is None
            else tuple(dict.fromkeys(str(item) for item in goal_ids))
        )
        unknown_goals = [goal_id for goal_id in selected if goal_id not in state.goals]
        if unknown_goals:
            raise ValueError("completion check names unknown goals: " + ", ".join(unknown_goals))
        request_by_goal = {request.goal_id: request for request in requests}
        results: list[VerificationResult] = []

        for goal_id in selected:
            goal = state.goals[goal_id]
            if goal.status.value == "verified":
                continue
            request = request_by_goal.get(goal_id)
            if request is None:
                continue
            results.append(self.verifier.verify(request, commit=commit))

        state = self.store.state
        remaining: list[str] = []
        remaining_conditions: list[str] = []
        for goal_id in selected:
            goal = state.goals[goal_id]
            if goal.status.value == "verified":
                continue
            remaining.append(goal_id)
            result = next((item for item in results if item.goal_id == goal_id), None)
            if result is not None and result.status is not VerificationStatus.PASS:
                remaining_conditions.extend(
                    f"{goal_id}:{condition_id}"
                    for condition_id in result.failed_condition_ids()
                )
            else:
                remaining_conditions.extend(
                    f"{goal_id}:{criterion}" for criterion in goal.acceptance_criteria
                )

        accepted = not remaining
        reason = (
            "all selected requirements are verified"
            if accepted
            else "completion refused; remaining goals: " + ", ".join(remaining)
        )
        objects_remaining: tuple[str, ...] = ()
        if selected:
            progress = state.object_progress(goal_ids=selected)
            objects_remaining = tuple(progress["remaining_object_ids"])
        return CompletionDecision(
            accepted=accepted,
            remaining_goal_ids=tuple(remaining),
            remaining_conditions=tuple(dict.fromkeys(remaining_conditions)),
            reason=reason,
            results=tuple(results),
            state_completion_status=state.completion_status(),
            objects_remaining=objects_remaining,
        )

    def gate(
        self,
        requests: Sequence[VerificationRequest],
        *,
        commit: bool = True,
    ) -> CompletionCheck:
        return CompletionCheck(self, tuple(requests), commit=commit)


def make_finish_gate(
    checker: CompletionChecker,
    request_factory: Callable[[], Sequence[VerificationRequest]],
    *,
    commit: bool = True,
) -> Callable[[object], CompletionDecision]:
    """A lazily-built gate for a driver whose final observation is current."""

    def gate(_step: object = None) -> CompletionDecision:
        return checker.check(request_factory(), commit=commit)

    return gate


__all__ = ["CompletionCheck", "CompletionChecker", "make_finish_gate"]
