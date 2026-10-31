"""Recovery feedback intake, decoupled from generator advancement (phase-2 E06).

The control layer cancels a proposal without executing it, then hands the Agent
a fresh observation of the page the user left behind. Recovery must decide from
that *fresh* observation. It must not spend a model turn or a recovery attempt
re-playing the discarded proposal, and it must not anchor a cause on the
proposal's old observation.

:class:`FeedbackIntake` is the single record passed at that seam. It carries the
fresh observation, the truthful feedback of an action that actually ran, and the
control coordinator's discarded-proposal records. :class:`RecoveryIntake`
enforces the two properties programmatically:

* a discarded proposal can never be the executed action of a recovery decision;
* a discarded proposal alone (no executed action) spends no recovery attempt and
  yields the next step from the fresh observation;
* consuming the same fresh observation twice does not authorize a second
  attempt, so a repeated callback cannot buy another round of work.

The intake does not execute web actions itself; the normal action loop owns the
next step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..protocol import Observation
from ..state.records import EvidenceRef
from .controller import RecoveryController
from .planner import RecoveryPlanner
from .types import (
    FailureEvidence,
    RecoveryClass,
    RecoveryError,
    RecoveryPlan,
    RecoveryResult,
)

_UNSAFE_ID = frozenset("\r\n\t")


def _optional_token(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecoveryError(f"{name} must be None or a non-empty string")
    if any(char in value for char in _UNSAFE_ID):
        raise RecoveryError(f"{name} must not contain control line breaks")
    return value


def _discarded_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise RecoveryError("discarded_proposals must be a sequence of control records")
    if not isinstance(value, Sequence):
        raise RecoveryError("discarded_proposals must be a sequence of control records")
    records: list[Mapping[str, Any]] = []
    for record in value:
        if not isinstance(record, Mapping):
            raise RecoveryError("discarded_proposals entries must be mappings")
        action_id = record.get("action_id")
        if action_id is not None:
            _optional_token(action_id, "discarded proposal action_id")
        records.append(dict(record))
    return tuple(records)


@dataclass(frozen=True)
class FeedbackIntake:
    """One fresh observation plus the feedback recovery may legitimately use.

    ``executed_action_id`` is set only when the browser action ran and returned
    this observation. ``discarded_proposals`` is
    :attr:`webagent.control.ControlCoordinator.discarded_proposals` (or the same
    records): proposals that were cancelled or dropped and therefore never ran.
    """

    observation: Observation
    failure_class: RecoveryClass | str = RecoveryClass.UNKNOWN
    goal_id: str | None = None
    executed_action_id: str | None = None
    sequence: int = 0
    message: str = ""
    evidence: tuple[EvidenceRef, ...] = ()
    discarded_proposals: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise RecoveryError("intake.observation must be an Observation")
        try:
            failure_class = RecoveryClass(self.failure_class)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in RecoveryClass)
            raise RecoveryError(f"intake.failure_class must be one of: {allowed}") from exc
        object.__setattr__(self, "failure_class", failure_class)
        object.__setattr__(self, "goal_id", _optional_token(self.goal_id, "intake.goal_id"))
        object.__setattr__(
            self,
            "executed_action_id",
            _optional_token(self.executed_action_id, "intake.executed_action_id"),
        )
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise RecoveryError("intake.sequence must be an integer >= 0")
        if not isinstance(self.message, str):
            raise RecoveryError("intake.message must be a string")
        evidence = self.evidence
        if not isinstance(evidence, tuple):
            raise RecoveryError("intake.evidence must be a tuple of EvidenceRef records")
        for ref in evidence:
            if not isinstance(ref, EvidenceRef):
                raise RecoveryError("intake.evidence must contain EvidenceRef records")
        if len({ref.evidence_id for ref in evidence}) != len(evidence):
            raise RecoveryError("intake.evidence repeats an evidence id")
        object.__setattr__(self, "discarded_proposals", _discarded_records(self.discarded_proposals))
        if (
            self.executed_action_id is not None
            and self.executed_action_id in self.discarded_action_ids
        ):
            raise RecoveryError(
                "intake.executed_action_id is a discarded proposal; "
                "a proposal that never ran cannot be an executed action"
            )

    @property
    def discarded_action_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for record in self.discarded_proposals:
            action_id = record.get("action_id")
            if isinstance(action_id, str) and action_id not in ids:
                ids.append(action_id)
        return tuple(ids)


@dataclass(frozen=True)
class IntakeResult:
    """The recovery decision formed from one fresh observation."""

    consumed: bool
    reused_observation: bool
    observation_id: str
    observation_url: str
    next_step: str
    reason: str
    plan: RecoveryPlan | None = None
    recovery: RecoveryResult | None = None
    failure: FailureEvidence | None = None
    discarded_proposals: tuple[Mapping[str, Any], ...] = ()
    replayed_action_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumed": self.consumed,
            "reused_observation": self.reused_observation,
            "observation_id": self.observation_id,
            "observation_url": self.observation_url,
            "next_step": self.next_step,
            "reason": self.reason,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "recovery": None if self.recovery is None else self.recovery.to_dict(),
            "failure": None if self.failure is None else self.failure.to_dict(),
            "discarded_proposals": [dict(record) for record in self.discarded_proposals],
            "replayed_action_ids": list(self.replayed_action_ids),
        }


class RecoveryIntake:
    """Form at most one recovery decision per fresh observation."""

    def __init__(self, planner: RecoveryPlanner, controller: RecoveryController) -> None:
        if not isinstance(planner, RecoveryPlanner):
            raise RecoveryError("RecoveryIntake needs a RecoveryPlanner")
        if not isinstance(controller, RecoveryController):
            raise RecoveryError("RecoveryIntake needs a RecoveryController")
        self.planner = planner
        self.controller = controller
        self._decisions: dict[str, IntakeResult] = {}

    @property
    def consumed_observation_ids(self) -> tuple[str, ...]:
        return tuple(self._decisions)

    def consider(self, intake: FeedbackIntake) -> IntakeResult:
        """Consume one fresh observation; a repeat never spends another attempt."""

        if not isinstance(intake, FeedbackIntake):
            raise RecoveryError("consider needs a FeedbackIntake")
        observation_id = intake.observation.observation_id
        previous = self._decisions.get(observation_id)
        if previous is not None:
            return replace(
                previous,
                consumed=False,
                reused_observation=True,
                reason=(
                    f"{previous.reason}; observation {observation_id} was already "
                    "consumed, so no second attempt or extra round is spent"
                ),
            )
        result = self._decide(intake)
        self._decisions[observation_id] = result
        return result

    # ----------------------------------------------------------------- private
    def _decide(self, intake: FeedbackIntake) -> IntakeResult:
        observation = intake.observation
        observation_id = observation.observation_id

        if intake.executed_action_id is None:
            if intake.discarded_action_ids:
                detail = (
                    "the discarded proposal(s) were never executed and are not replayed"
                )
            else:
                detail = "no action was executed for this feedback"
            next_step = (
                f"continue from fresh observation {observation_id} "
                f"({observation.url}); {detail}"
            )
            return IntakeResult(
                consumed=True,
                reused_observation=False,
                observation_id=observation_id,
                observation_url=observation.url,
                next_step=next_step,
                reason="no executed action to recover; no recovery attempt was spent",
                plan=None,
                recovery=None,
                discarded_proposals=intake.discarded_proposals,
            )

        failure = FailureEvidence(
            action_id=intake.executed_action_id,
            sequence=intake.sequence,
            failure_class=intake.failure_class,
            goal_id=intake.goal_id,
            message=intake.message or f"{intake.failure_class.value} feedback",
            observation_id=observation_id,
            evidence=intake.evidence,
        )
        plan = self.planner.plan([failure])
        attempt = None
        if plan.evidence_sufficient and plan.affected_goal_ids:
            attempt = self.controller.authorize(plan.affected_goal_ids)
        recovery = self.controller.apply(plan, attempt=attempt)
        next_step = (
            f"{plan.next_step}; continue from fresh observation {observation_id} "
            f"({observation.url})"
        )
        return IntakeResult(
            consumed=True,
            reused_observation=False,
            observation_id=observation_id,
            observation_url=observation.url,
            next_step=next_step,
            reason=(
                f"recovery decision based on the executed action "
                f"{intake.executed_action_id} and fresh observation {observation_id}"
            ),
            plan=plan,
            recovery=recovery,
            failure=failure,
            discarded_proposals=intake.discarded_proposals,
        )


__all__ = ["FeedbackIntake", "IntakeResult", "RecoveryIntake"]
