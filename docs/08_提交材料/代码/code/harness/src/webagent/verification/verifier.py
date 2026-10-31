"""Goal-level verification: evaluate conditions, then commit only a real pass.

The verifier is a programmatic path, not a model tool. It reads the current
observation, the run's already-saved evidence and the task ledger. It never
creates a browser, re-fetches a page or downloads an artifact that the
observation already captured. A goal becomes ``verified`` only through
:meth:`StateStore.verify_goal`, after every condition passed and every cited
artifact was hash-checked.
"""

from __future__ import annotations

from dataclasses import replace

from ..events import EventLogger
from ..state.records import EvidenceRef, GoalStatus
from ..state.store import StateStore
from .checks import DeterministicChecker
from .semantic import ModelSemanticJudge
from .types import (
    CheckKind,
    ConditionResult,
    VerificationCondition,
    VerificationError,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)


class GoalVerifier:
    """Evaluate one :class:`VerificationRequest` and optionally write a pass."""

    def __init__(
        self,
        store: StateStore,
        *,
        semantic_judge: ModelSemanticJudge | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        if not isinstance(store, StateStore):
            raise VerificationError("GoalVerifier needs a StateStore")
        self.store = store
        self.deterministic = DeterministicChecker(store)
        self.semantic_judge = semantic_judge
        self.logger = logger

    def verify(
        self,
        request: VerificationRequest,
        *,
        commit: bool = True,
    ) -> VerificationResult:
        if not isinstance(request, VerificationRequest):
            raise VerificationError("verify needs a VerificationRequest")
        state = self.store.state
        goal = state.goals.get(request.goal_id)
        if goal is None:
            raise VerificationError(f"unknown goal {request.goal_id!r}")
        revision_before = state.revision
        if goal.status is GoalStatus.VERIFIED:
            result = VerificationResult(
                goal_id=request.goal_id,
                subject_id=request.subject_id,
                status=VerificationStatus.PASS,
                conditions=tuple(
                    ConditionResult(
                        condition_id=condition.condition_id,
                        status=VerificationStatus.PASS,
                        detail="goal was already verified",
                        method="already_verified",
                    )
                    for condition in request.conditions
                ),
                evidence=goal.evidence,
                method="already_verified",
                model_calls=0,
                reason="goal is already verified by evidence-backed transition",
                verified=True,
                trigger=request.trigger,
                state_revision_before=revision_before,
                state_revision_after=revision_before,
            )
            self._emit(result)
            return result

        outcomes: list[ConditionResult] = []
        for condition in request.conditions:
            if condition.check in ("semantic",):
                outcomes.append(self._semantic(condition, request))
            else:
                outcome = self.deterministic.evaluate(condition, request)
                outcomes.append(outcome.as_condition(condition))

        methods = {item.method for item in outcomes}
        method = "mixed" if len(methods) > 1 else next(iter(methods), "deterministic")
        status = self._aggregate(outcomes)
        evidence = self._unique_evidence(outcomes)

        object_problem = self._object_problem(request)
        if object_problem is not None:
            status = VerificationStatus.FAIL
        reason = self._reason(outcomes, status, object_problem)
        verified = False

        if status is VerificationStatus.PASS and commit:
            if not evidence:
                status = VerificationStatus.FAIL
                reason = "all conditions passed but no real evidence artifact was cited"
            else:
                try:
                    for ref in evidence:
                        self.store.verify_evidence(ref)
                    state = self.store.state
                    existing_ids = {ref.evidence_id for ref in state.goals[request.goal_id].evidence}
                    new_refs = tuple(
                        ref for ref in evidence if ref.evidence_id not in existing_ids
                    )
                    if not new_refs:
                        raise VerificationError(
                            "verification evidence is already recorded but the goal is not "
                            "verified; a fresh evidence-backed transition is required"
                        )
                    state = self.store.verify_goal(
                        request.goal_id,
                        evidence=new_refs,
                        base_revision=state.revision,
                        plan_version=state.plan_version,
                        source_branch=request.branch,
                        completed_object_ids=self._completed_objects(request),
                        note=(
                            f"verification trigger={request.trigger.value}; "
                            f"conditions={len(request.conditions)}; note={request.note}"
                        ),
                    )
                    verified = True
                except Exception as exc:  # noqa: BLE001 - a failed write is not a pass claim
                    status = VerificationStatus.UNKNOWN
                    reason = f"conditions passed but the verified transition was not committed: {exc}"
                    if self.logger is not None:
                        try:
                            self.logger.emit(
                                "verification",
                                result={
                                    "goal_id": request.goal_id,
                                    "status": status.value,
                                    "reason": reason,
                                },
                            )
                        except Exception:
                            pass

        result = VerificationResult(
            goal_id=request.goal_id,
            subject_id=request.subject_id,
            status=status,
            conditions=tuple(outcomes),
            evidence=evidence,
            method=method,
            model_calls=sum(item.model_calls for item in outcomes),
            reason=reason,
            verified=verified,
            trigger=request.trigger,
            state_revision_before=revision_before,
            state_revision_after=self.store.state.revision,
        )
        self._emit(result)
        return result

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _aggregate(outcomes: list[ConditionResult]) -> VerificationStatus:
        if any(item.status is VerificationStatus.FAIL for item in outcomes):
            return VerificationStatus.FAIL
        if any(item.status is VerificationStatus.UNKNOWN for item in outcomes):
            return VerificationStatus.UNKNOWN
        return VerificationStatus.PASS

    @staticmethod
    def _unique_evidence(outcomes: list[ConditionResult]) -> tuple[EvidenceRef, ...]:
        index: dict[str, EvidenceRef] = {}
        for outcome in outcomes:
            for ref in outcome.evidence:
                index.setdefault(ref.evidence_id, ref)
        return tuple(index.values())

    def _semantic(
        self,
        condition: VerificationCondition,
        request: VerificationRequest,
    ) -> ConditionResult:
        if self.semantic_judge is None:
            return ConditionResult(
                condition_id=condition.condition_id,
                status=VerificationStatus.UNKNOWN,
                detail="semantic condition requested but no unified model judge is configured",
                method="semantic",
                model_calls=0,
            )
        return self.semantic_judge.evaluate(condition, request)

    def _object_problem(self, request: VerificationRequest) -> str | None:
        goal = self.store.state.goals[request.goal_id]
        if not goal.object_scope:
            return None
        completed = (
            set(goal.completed_object_ids)
            if request.completed_object_ids is None
            else set(request.completed_object_ids)
        )
        missing = sorted(set(goal.object_scope) - completed)
        if missing:
            return f"goal object scope is incomplete; unaccounted objects: {missing}"
        return None

    def _completed_objects(self, request: VerificationRequest) -> tuple[str, ...]:
        if request.completed_object_ids is not None:
            return request.completed_object_ids
        return self.store.state.goals[request.goal_id].completed_object_ids

    def _reason(
        self,
        outcomes: list[ConditionResult],
        status: VerificationStatus,
        object_problem: str | None,
    ) -> str:
        if object_problem is not None:
            return object_problem
        failed = [item.condition_id for item in outcomes if item.status is VerificationStatus.FAIL]
        unknown = [item.condition_id for item in outcomes if item.status is VerificationStatus.UNKNOWN]
        if status is VerificationStatus.PASS:
            return "all conditions passed"
        if status is VerificationStatus.FAIL:
            return "failed conditions: " + ", ".join(failed or unknown)
        return "conditions not determinable: " + ", ".join(unknown)

    def _emit(self, result: VerificationResult) -> None:
        if self.logger is None:
            return
        try:
            self.logger.emit(
                "verification",
                goal_id=result.goal_id,
                result=result.to_dict(),
            )
        except Exception:  # noqa: BLE001 - diagnostics must not change the result
            pass


__all__ = ["GoalVerifier"]
