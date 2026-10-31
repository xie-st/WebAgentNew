"""Unknown side-effect probing before any retry.

When feedback is lost after a write, the pending effect is not evidence that the
write failed. The resolver asks the caller for one fresh normal observation,
reads only the visible page state, and decides:

* the expected visible value is present -> confirmed, retry is forbidden;
* the field is present and decisively differs -> not applied, retry may be
  considered by the caller;
* the field is absent or ambiguous -> unknown, retry remains forbidden.

This module never creates a browser, reads a site database or sends an action.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

from ..protocol import Observation
from ..state.records import EffectStatus, EvidenceRef, PendingEffect
from ..state.store import StateStore
from ..verification.checks import visible_value
from .types import (
    EffectExpectation,
    EffectOutcome,
    EffectResolution,
    RecoveryError,
)


class UnknownEffectResolver:
    def __init__(self, store: StateStore) -> None:
        if not isinstance(store, StateStore):
            raise RecoveryError("UnknownEffectResolver needs a StateStore")
        self.store = store

    def resolve(
        self,
        effect: PendingEffect,
        *,
        probe: Callable[[], Observation],
        expectation: EffectExpectation,
        evidence_scope: Sequence[EvidenceRef] = (),
        note: str = "",
    ) -> EffectResolution:
        """Probe once, then decide; no action is retried by this method."""

        if not isinstance(effect, PendingEffect):
            raise RecoveryError("resolve needs a PendingEffect")
        if effect.status not in (EffectStatus.PENDING, EffectStatus.UNKNOWN):
            return EffectResolution(
                outcome=(
                    EffectOutcome.CONFIRMED
                    if effect.status is EffectStatus.CONFIRMED
                    else EffectOutcome.NOT_APPLIED
                    if effect.status is EffectStatus.FAILED
                    else EffectOutcome.UNKNOWN
                ),
                retry_allowed=False,
                reason=f"effect already has status {effect.status.value}",
                observation_id=effect.observation_id,
            )
        if not callable(probe):
            raise RecoveryError("probe must be a callable returning one fresh Observation")

        observation = probe()
        if not isinstance(observation, Observation):
            raise RecoveryError("probe must return a normal Observation")
        actual = visible_value(
            observation,
            role=expectation.field_role,
            name=expectation.field_name,
        )
        refs = tuple(evidence_scope)
        if not refs:
            refs = self._observation_refs(observation, effect, note=note)

        if actual is not None and actual == expectation.expected_value:
            self.store.confirm_effect(
                effect.action_id,
                status=EffectStatus.CONFIRMED,
                observation_id=observation.observation_id,
                note=note or "unknown-effect probe found the expected visible value; no retry",
            )
            return EffectResolution(
                outcome=EffectOutcome.CONFIRMED,
                retry_allowed=False,
                reason=(
                    f"visible {expectation.field_name!r} equals the saved value; "
                    "the write took effect, so it must not be sent again"
                ),
                observation_id=observation.observation_id,
                visible_value=actual,
                evidence=refs,
            )

        if actual is not None and expectation.absence_is_decisive:
            self.store.confirm_effect(
                effect.action_id,
                status=EffectStatus.FAILED,
                observation_id=observation.observation_id,
                note=note or "unknown-effect probe found a decisive different value; retry allowed",
            )
            return EffectResolution(
                outcome=EffectOutcome.NOT_APPLIED,
                retry_allowed=True,
                reason=(
                    f"visible {expectation.field_name!r} differs from the intended value; "
                    "one controlled retry may be considered"
                ),
                observation_id=observation.observation_id,
                visible_value=actual,
                evidence=refs,
            )

        if effect.status is EffectStatus.PENDING:
            self.store.confirm_effect(
                effect.action_id,
                status=EffectStatus.UNKNOWN,
                observation_id=observation.observation_id,
                note=note or "unknown-effect probe was not decisive",
            )
        return EffectResolution(
            outcome=EffectOutcome.UNKNOWN,
            retry_allowed=False,
            reason="the fresh observation does not decisively show the write's effect",
            observation_id=observation.observation_id,
            visible_value=actual,
            evidence=refs,
        )

    @staticmethod
    def _observation_refs(
        observation: Observation,
        effect: PendingEffect,
        *,
        note: str,
    ) -> tuple[EvidenceRef, ...]:
        artifact = observation.artifacts.get("screenshot")
        if artifact is None or not artifact.available:
            return ()
        from ..state.integration import evidence_ref_from_artifact

        return (
            evidence_ref_from_artifact(
                artifact,
                run_id=observation.run_id,
                branch_id=effect.branch_id,
                evidence_id=f"ev-{observation.observation_id}-screenshot",
                observation_id=observation.observation_id,
                action_id=effect.action_id,
            ),
        )


__all__ = ["UnknownEffectResolver"]
