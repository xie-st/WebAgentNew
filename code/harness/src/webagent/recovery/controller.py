"""Bounded recovery application with real limits and a real deadline.

The controller is the only place that writes an invalidation. It first authorizes
an attempt against per-goal and whole-task limits and the absolute deadline,
then commits one version-checked ledger update for the affected goals.

Withdrawing a *verified* goal needs the programmatic invalidation transition
described in the E1 report's shared-integration section. Until that small
ledger/StateStore API exists, this controller refuses to partially apply a plan
that contains a verified goal rather than silently leaving a stale verified
claim in place. Non-verified affected goals are invalidated through the existing
``StateStore.apply_update`` path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..clock import Clock, SystemClock
from ..state.ledger import StateUpdate
from ..state.records import EvidenceRef, Goal, GoalStatus, StateValidationError
from ..state.store import StateStore
from .types import (
    RecoveryAttempt,
    RecoveryError,
    RecoveryPlan,
    RecoveryResult,
)


class RecoveryLimitExceeded(RecoveryError):
    """The bounded recovery budget refused another attempt."""


class RecoveryNeedsLedgerSupport(RecoveryError):
    """A verified goal must be withdrawn but no supported ledger transition exists."""


class RecoveryLimits:
    def __init__(
        self,
        *,
        max_recoveries_per_goal: int = 2,
        max_recoveries_total: int = 5,
        deadline_epoch: float | None = None,
        retry_backoff_seconds: Sequence[float] = (1.0, 2.0),
        reflection_limit: int = 3,
    ) -> None:
        if (
            not isinstance(max_recoveries_per_goal, int)
            or isinstance(max_recoveries_per_goal, bool)
            or max_recoveries_per_goal < 1
        ):
            raise RecoveryError("max_recoveries_per_goal must be a positive integer")
        if (
            not isinstance(max_recoveries_total, int)
            or isinstance(max_recoveries_total, bool)
            or max_recoveries_total < 1
        ):
            raise RecoveryError("max_recoveries_total must be a positive integer")
        if deadline_epoch is not None and (
            not isinstance(deadline_epoch, (int, float))
            or isinstance(deadline_epoch, bool)
        ):
            raise RecoveryError("deadline_epoch must be a timestamp or null")
        if not retry_backoff_seconds or any(
            not isinstance(item, (int, float)) or isinstance(item, bool) or item < 0
            for item in retry_backoff_seconds
        ):
            raise RecoveryError("retry_backoff_seconds must be non-negative numbers")
        if (
            not isinstance(reflection_limit, int)
            or isinstance(reflection_limit, bool)
            or reflection_limit < 1
        ):
            raise RecoveryError("reflection_limit must be a positive integer")
        self.max_recoveries_per_goal = max_recoveries_per_goal
        self.max_recoveries_total = max_recoveries_total
        self.deadline_epoch = None if deadline_epoch is None else float(deadline_epoch)
        self.retry_backoff_seconds = tuple(float(item) for item in retry_backoff_seconds)
        self.reflection_limit = reflection_limit


class RecoveryController:
    def __init__(
        self,
        store: StateStore,
        *,
        limits: RecoveryLimits | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(store, StateStore):
            raise RecoveryError("RecoveryController needs a StateStore")
        self.store = store
        self.limits = limits or RecoveryLimits()
        self.clock = clock or SystemClock()
        self._goal_attempts: dict[str, int] = {}
        self._total_attempts = 0
        self.reflections: list[str] = []

    @property
    def goal_attempts(self) -> Mapping[str, int]:
        return dict(self._goal_attempts)

    @property
    def total_attempts(self) -> int:
        return self._total_attempts

    def remaining_seconds(self) -> float | None:
        if self.limits.deadline_epoch is None:
            return None
        return self.limits.deadline_epoch - self.clock.now()

    def authorize(self, goal_ids: Sequence[str]) -> RecoveryAttempt:
        """Reserve one attempt; a denied attempt does not increment counters."""

        ids = tuple(dict.fromkeys(str(goal_id) for goal_id in goal_ids))
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            return RecoveryAttempt(
                allowed=False,
                reason=f"recovery deadline reached with {remaining:.3f}s remaining",
                goal_attempts=self._goal_attempts,
                total_attempts=self._total_attempts,
                remaining_seconds=remaining,
            )
        if self._total_attempts >= self.limits.max_recoveries_total:
            return RecoveryAttempt(
                allowed=False,
                reason=(
                    "whole-task recovery limit reached "
                    f"({self.limits.max_recoveries_total})"
                ),
                goal_attempts=self._goal_attempts,
                total_attempts=self._total_attempts,
                remaining_seconds=remaining,
            )
        over = [
            goal_id
            for goal_id in ids
            if self._goal_attempts.get(goal_id, 0) >= self.limits.max_recoveries_per_goal
        ]
        if over:
            return RecoveryAttempt(
                allowed=False,
                reason=(
                    "per-goal recovery limit reached for " + ", ".join(sorted(over))
                ),
                goal_attempts=self._goal_attempts,
                total_attempts=self._total_attempts,
                remaining_seconds=remaining,
            )
        for goal_id in ids:
            self._goal_attempts[goal_id] = self._goal_attempts.get(goal_id, 0) + 1
        self._total_attempts += 1
        return RecoveryAttempt(
            allowed=True,
            reason=f"recovery attempt {self._total_attempts} authorized",
            goal_attempts=self._goal_attempts,
            total_attempts=self._total_attempts,
            remaining_seconds=remaining,
        )

    def apply(
        self,
        plan: RecoveryPlan,
        *,
        attempt: RecoveryAttempt | None = None,
        note: str = "",
    ) -> RecoveryResult:
        if not isinstance(plan, RecoveryPlan):
            raise RecoveryError("apply needs a RecoveryPlan")
        if not plan.evidence_sufficient:
            return RecoveryResult(
                applied=False,
                plan=plan,
                attempt=attempt
                or RecoveryAttempt(False, "plan has insufficient evidence"),
                state_revision_before=self.store.state.revision,
                state_revision_after=self.store.state.revision,
                reason=plan.reason,
            )
        ids = plan.invalidated_goal_ids
        if attempt is None:
            attempt = (
                self.authorize(plan.affected_goal_ids or ids)
                if ids
                else RecoveryAttempt(True, "no completion claim needed withdrawal")
            )
        if not attempt.allowed:
            return RecoveryResult(
                applied=False,
                plan=plan,
                attempt=attempt,
                state_revision_before=self.store.state.revision,
                state_revision_after=self.store.state.revision,
                reason=attempt.reason,
            )
        if not ids:
            return RecoveryResult(
                applied=True,
                plan=plan,
                attempt=attempt,
                state_revision_before=self.store.state.revision,
                state_revision_after=self.store.state.revision,
                reason="plan affects no currently verified/awaiting goal",
            )

        state = self.store.state
        verified_affected = [
            goal_id
            for goal_id in ids
            if state.goals[goal_id].status is GoalStatus.VERIFIED
        ]
        if verified_affected:
            return RecoveryResult(
                applied=False,
                plan=plan,
                attempt=attempt,
                state_revision_before=state.revision,
                state_revision_after=state.revision,
                reason=(
                    "verified goals require the programmatic invalidation transition "
                    "not present in the current StateStore API: "
                    + ", ".join(sorted(verified_affected))
                ),
            )

        goals: list[Goal] = []
        for goal_id in ids:
            goal = state.goals[goal_id]
            goals.append(
                replace(
                    goal,
                    status=GoalStatus.INVALIDATED,
                )
            )
        update = StateUpdate(
            base_revision=state.revision,
            plan_version=state.plan_version,
            source_branch="root",
            goals=tuple(goals),
            note=note or "local recovery invalidation",
        )
        try:
            new_state = self.store.apply_update(update)
        except StateValidationError as exc:
            return RecoveryResult(
                applied=False,
                plan=plan,
                attempt=attempt,
                state_revision_before=state.revision,
                state_revision_after=state.revision,
                reason=f"ledger refused invalidation: {exc}",
            )
        return RecoveryResult(
            applied=True,
            plan=plan,
            attempt=attempt,
            state_revision_before=state.revision,
            state_revision_after=new_state.revision,
            reason=(
                f"invalidated {len(goals)} affected goal(s); "
                f"preserved {len(plan.preserved_verified_goal_ids)} unrelated verified goal(s)"
            ),
        )

    def withdraw(
        self,
        plan: RecoveryPlan,
        *,
        attempt: RecoveryAttempt | None = None,
        evidence: Sequence[EvidenceRef] = (),
        note: str = "",
        source_branch: str = "root",
    ) -> RecoveryResult:
        """Apply a plan through the programmatic withdrawal transition.

        Unlike :meth:`apply`, this production path can withdraw an already
        ``verified`` claim. It uses :meth:`StateStore.invalidate_goal`, which keeps
        the goal record, its requirement, acceptance criteria, object scope,
        completed objects and earlier evidence, appends the new conflicting
        evidence, writes one audit record and (with an attached event log) one
        ``recovery`` event per goal. Unaffected verified goals are never touched.

        Without new conflicting evidence the withdrawal target is
        ``awaiting_verification``: the old claim is no longer trusted but no cause
        is invented. A denied attempt, insufficient plan evidence or a refused
        ledger transition returns ``applied=False`` and changes nothing.
        """

        if not isinstance(plan, RecoveryPlan):
            raise RecoveryError("withdraw needs a RecoveryPlan")
        if not plan.evidence_sufficient:
            return RecoveryResult(
                applied=False,
                plan=plan,
                attempt=attempt
                or RecoveryAttempt(False, "plan has insufficient evidence"),
                state_revision_before=self.store.state.revision,
                state_revision_after=self.store.state.revision,
                reason=plan.reason,
            )
        ids = plan.invalidated_goal_ids
        if attempt is None:
            attempt = (
                self.authorize(plan.affected_goal_ids or ids)
                if ids
                else RecoveryAttempt(True, "no completion claim needed withdrawal")
            )
        if not attempt.allowed:
            return RecoveryResult(
                applied=False,
                plan=plan,
                attempt=attempt,
                state_revision_before=self.store.state.revision,
                state_revision_after=self.store.state.revision,
                reason=attempt.reason,
            )
        state = self.store.state
        revision_before = state.revision
        if not ids:
            return RecoveryResult(
                applied=True,
                plan=plan,
                attempt=attempt,
                state_revision_before=revision_before,
                state_revision_after=revision_before,
                reason="plan affects no currently verified/awaiting goal",
            )

        withdrawn: list[str] = []
        unchanged: list[str] = []
        for goal_id in ids:
            goal = self.store.state.goals[goal_id]
            existing = {ref.evidence_id for ref in goal.evidence}
            new_refs = tuple(
                ref for ref in evidence if ref.evidence_id not in existing
            )
            try:
                record = self.store.invalidate_goal(
                    goal_id,
                    note or plan.reason,
                    new_refs or None,
                    source_branch=source_branch,
                )
            except StateValidationError as exc:
                return RecoveryResult(
                    applied=False,
                    plan=plan,
                    attempt=attempt,
                    state_revision_before=revision_before,
                    state_revision_after=self.store.state.revision,
                    reason=f"ledger refused withdrawal of {goal_id!r}: {exc}",
                )
            (withdrawn if record.changed else unchanged).append(goal_id)
        return RecoveryResult(
            applied=True,
            plan=plan,
            attempt=attempt,
            state_revision_before=revision_before,
            state_revision_after=self.store.state.revision,
            reason=(
                f"withdrew {len(withdrawn)} affected completion claim(s) "
                f"({', '.join(sorted(withdrawn)) or 'none'}); "
                f"{len(unchanged)} already withdrawn; preserved "
                f"{len(plan.preserved_verified_goal_ids)} unrelated verified goal(s)"
            ),
        )

    def record_reflection(self, reflection: str) -> None:
        """Keep at most the configured number of evidence-based reflections."""

        if not isinstance(reflection, str) or not reflection.strip():
            raise RecoveryError("reflection must be a non-empty string")
        self.reflections.append(reflection.strip())
        self.reflections = self.reflections[-self.limits.reflection_limit :]


__all__ = [
    "RecoveryController",
    "RecoveryLimitExceeded",
    "RecoveryLimits",
    "RecoveryNeedsLedgerSupport",
]
