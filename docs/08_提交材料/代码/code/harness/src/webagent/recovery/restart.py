"""Restart from a complete snapshot with a fresh observation.

Only the ledger snapshot is restored. A Python generator stack is not restored,
no pending effect is automatically confirmed and no action is replayed. The new
branch is registered with its own run id so its future evidence is attributable.
"""

from __future__ import annotations

from typing import Callable

from ..protocol import Observation, validate_run_id
from ..state.store import StateStore
from .types import RecoveryError, RestartResult


class RestartRecovery:
    def __init__(self, store: StateStore) -> None:
        if not isinstance(store, StateStore):
            raise RecoveryError("RestartRecovery needs a StateStore")
        self.store = store

    def recover(
        self,
        *,
        new_branch_id: str,
        new_run_id: str,
        observer: Callable[[], Observation],
        source_branch: str = "root",
    ) -> RestartResult:
        """Load, register a new branch, re-observe and leave pending effects alone."""

        if not isinstance(new_branch_id, str) or not new_branch_id.strip():
            raise RecoveryError("new_branch_id must be a non-empty string")
        validate_run_id(new_run_id)
        if not callable(observer):
            raise RecoveryError("observer must be a callable returning one fresh Observation")
        state = self.store.load()
        if source_branch not in state.branches:
            raise RecoveryError(f"source branch {source_branch!r} is not registered")
        if new_branch_id in state.branches:
            raise RecoveryError(f"branch {new_branch_id!r} is already registered")
        pending_before = self.store.pending_effects

        observation = observer()
        if not isinstance(observation, Observation):
            raise RecoveryError("observer must return a normal Observation")
        if observation.run_id != new_run_id:
            raise RecoveryError(
                f"restart observation belongs to run {observation.run_id!r}, "
                f"not the new branch run {new_run_id!r}"
            )

        state = self.store.register_branch(
            new_branch_id,
            new_run_id,
            base_revision=state.revision,
            plan_version=state.plan_version,
        )
        # Pending effects were read before the branch update. Registration does
        # not touch them; re-assert the captured ids so a caller can prove no
        # automatic resend happened.
        pending_after = tuple(
            effect
            for effect in self.store.pending_effects
            if effect.action_id in {item.action_id for item in pending_before}
        )
        return RestartResult(
            root_run_id=state.root_run_id,
            source_branch=source_branch,
            new_branch_id=new_branch_id,
            new_run_id=new_run_id,
            state=state,
            observation=observation,
            pending_effects=pending_after,
            generator_stack_restored=False,
            resent_action_ids=(),
            reason=(
                "loaded complete snapshot, registered a new branch and re-observed; "
                "pending effects remain unresolved and no action was resent"
            ),
        )


__all__ = ["RestartRecovery"]
