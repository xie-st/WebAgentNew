"""Atomic JSON persistence and the serialized ledger commit path.

Layout::

    runtime/runs/<root_run_id>/state/state.json

The full snapshot is written to a sibling temporary file, flushed and fsynced,
then atomically replaced. A failed write leaves the previous snapshot untouched
and the candidate state is never installed in memory, so an action that depends
on the update cannot be dispatched from a state that was not saved.

Loading reads the one complete ``state.json``; temporary ``.part`` files from an
interrupted write are ignored. There is no journal replay and no database.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..clock import Clock, SystemClock
from ..protocol import json_dumps, validate_run_id
from .invalidation import (
    GoalInvalidation,
    InvalidationLog,
    WITHDRAWABLE_FROM,
    coerce_target_status,
    require_goal_id,
    require_reason,
)
from .ledger import (
    PlanUpdate,
    StateUpdate,
    apply_plan_update,
    apply_state_update,
)
from .records import (
    EffectStatus,
    EvidenceRef,
    EvidenceResolver,
    Goal,
    GoalStatus,
    PendingEffect,
    StateError,
    StateValidationError,
    TASK_STATE_SCHEMA_VERSION,
    TaskState,
)


class StatePersistenceError(StateError):
    """The snapshot could not be written; the candidate was not committed."""


class StateNotFound(StateError):
    """No complete snapshot exists yet, or none has been loaded."""


def _check_basis(
    state: TaskState,
    base_revision: int | None,
    plan_version: int | None,
) -> None:
    from .records import StateConflict

    if base_revision is not None and state.revision != base_revision:
        raise StateConflict(
            f"update is based on revision {base_revision}, current revision is "
            f"{state.revision}; re-read the state before retrying",
            current=state,
            reason="stale_base_revision",
            base_revision=base_revision,
            plan_version=plan_version,
        )
    if plan_version is not None and state.plan_version != plan_version:
        raise StateConflict(
            f"update is based on plan_version {plan_version}, current plan_version is "
            f"{state.plan_version}; re-read the plan before retrying",
            current=state,
            reason="stale_plan_version",
            base_revision=base_revision,
            plan_version=plan_version,
        )


class StateStore:
    """One task's ledger store, with a serialized compare-and-save update path.

    Model calls and browser waits never take this store's lock; only the
    version check plus snapshot write does. The single-process D batch keeps the
    interface ready for a later coordinator without adding a multi-writer
    transaction layer.
    """

    def __init__(
        self,
        runtime_dir: str | Path,
        root_run_id: str,
        *,
        clock: Clock | None = None,
        event_log: Any | None = None,
    ) -> None:
        validate_run_id(root_run_id)
        self.runtime_dir = Path(runtime_dir)
        self.root_run_id = root_run_id
        self.clock = clock or SystemClock()
        self.event_log = event_log
        self._state_dir = (self.runtime_dir / "runs" / root_run_id / "state").resolve()
        self._runs_root = (self.runtime_dir / "runs").resolve()
        if self._state_dir.parent.parent != self._runs_root:
            raise StatePersistenceError("state directory escapes the runtime root")
        self._state_path = self._state_dir / "state.json"
        self._lock = threading.RLock()
        self._state: TaskState | None = None

    # ------------------------------------------------------------------ paths
    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def invalidation_path(self) -> Path:
        return self._state_dir / InvalidationLog.FILENAME

    def attach_event_log(self, event_log: Any | None) -> None:
        """Forward later withdrawals to the run's structured event log."""

        self.event_log = event_log

    def read_invalidations(self) -> tuple[GoalInvalidation, ...]:
        """The withdrawal audit written under this task's ``state/`` directory."""

        return InvalidationLog(self._state_dir).read()

    @property
    def state(self) -> TaskState:
        if self._state is None:
            raise StateNotFound(
                f"task state for {self.root_run_id!r} is not loaded; call load() or initialize()"
            )
        return self._state

    @property
    def pending_effects(self) -> tuple[PendingEffect, ...]:
        return self.state.unconfirmed_effects

    # ------------------------------------------------------------- lifecycle
    def initialize(
        self,
        task: str,
        *,
        branches: dict[str, str] | None = None,
    ) -> TaskState:
        """Create the first snapshot for this run. Never overwrites one."""

        with self._lock:
            if self._state_path.exists():
                raise StateValidationError(
                    f"a task state snapshot already exists at {self._state_path}"
                )
            state = TaskState(
                root_run_id=self.root_run_id,
                task=task,
                branches=dict(branches) if branches else {"root": self.root_run_id},
                updated_at=self.clock.iso(self.clock.now()),
            )
            state.validate(evidence=self._resolver_for(state))
            self._commit(state)
            return state

    def load(self) -> TaskState:
        """Load the latest complete snapshot and make it the committed state."""

        with self._lock:
            if not self._state_path.is_file():
                raise StateNotFound(
                    f"no task state snapshot at {self._state_path}"
                )
            try:
                raw = self._state_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise StatePersistenceError(
                    f"cannot read task state snapshot {self._state_path}: {exc}"
                ) from exc
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise StateValidationError(
                    f"task state snapshot is not complete JSON: {exc}"
                ) from exc
            state = TaskState.from_dict(data)
            if state.root_run_id != self.root_run_id:
                raise StateValidationError(
                    f"snapshot belongs to run {state.root_run_id!r}, not {self.root_run_id!r}"
                )
            if state.schema_version != TASK_STATE_SCHEMA_VERSION:
                raise StateValidationError("unsupported task state schema version")
            state.validate(evidence=self._resolver_for(state))
            self._state = state
            return state

    def load_or_initialize(self, task: str | None = None) -> TaskState:
        if self._state_path.is_file():
            return self.load()
        if task is None:
            raise StateNotFound(
                f"no task state snapshot at {self._state_path} and no task text was given"
            )
        return self.initialize(task)

    # --------------------------------------------------------------- mutation
    def apply_update(self, update: StateUpdate) -> TaskState:
        """Apply an ordinary state update under the version-checked lock."""

        with self._lock:
            state = self.state
            resolver = self._resolver_for(state)
            candidate = apply_state_update(state, update, evidence=resolver)
            self._commit(candidate)
            return candidate

    def apply_plan_update(self, update: PlanUpdate) -> TaskState:
        """Apply a plan revision under the version-checked lock."""

        with self._lock:
            state = self.state
            resolver = self._resolver_for(state)
            candidate = apply_plan_update(state, update, evidence=resolver)
            self._commit(candidate)
            return candidate

    # ------------------------------------------------------ pending effects
    def prepare_effect(
        self,
        *,
        action_id: str,
        target: str,
        object_id: str,
        goal_id: str | None = None,
        branch_id: str | None = None,
        observation_id: str | None = None,
        note: str = "",
        base_revision: int | None = None,
        plan_version: int | None = None,
    ) -> PendingEffect:
        """Save a pending side effect *before* its action is dispatched.

        Returning normally is the caller's license to dispatch. If the snapshot
        cannot be written this raises :class:`StatePersistenceError` and the
        action must not be sent.
        """

        with self._lock:
            state = self.state
            _check_basis(state, base_revision, plan_version)
            branch = branch_id or "root"
            if action_id in state.pending_effects:
                raise StateValidationError(
                    f"action {action_id!r} already has a pending-effect record; "
                    "do not resend it without checking the page"
                )
            new_revision = state.revision + 1
            effect = PendingEffect(
                action_id=action_id,
                target=target,
                object_id=object_id,
                status=EffectStatus.PENDING,
                goal_id=goal_id,
                branch_id=branch,
                observation_id=observation_id,
                created_revision=new_revision,
                note=note,
            )
            candidate = state.copy()
            candidate.pending_effects[action_id] = effect
            candidate.revision = new_revision
            self._commit(candidate)
            return candidate.pending_effects[action_id]

    def confirm_effect(
        self,
        action_id: str,
        *,
        status: EffectStatus | str = EffectStatus.CONFIRMED,
        observation_id: str | None = None,
        note: str | None = None,
    ) -> PendingEffect:
        """Record feedback for a saved pending effect.

        A confirmation is always made after the attempt. ``pending`` is not a
        valid confirmation result, and an already confirmed/failed effect is
        never reopened by an unrelated later call.
        """

        with self._lock:
            state = self.state
            effect = state.pending_effects.get(action_id)
            if effect is None:
                raise StateValidationError(f"unknown pending effect {action_id!r}")
            target_status = (
                status if isinstance(status, EffectStatus) else EffectStatus(status)
            )
            if target_status is EffectStatus.PENDING:
                raise StateValidationError("a confirmation cannot restore the pending status")
            if effect.status is target_status:
                return effect
            allowed = {
                EffectStatus.PENDING: {
                    EffectStatus.CONFIRMED,
                    EffectStatus.FAILED,
                    EffectStatus.UNKNOWN,
                },
                EffectStatus.UNKNOWN: {EffectStatus.CONFIRMED, EffectStatus.FAILED},
                EffectStatus.CONFIRMED: set(),
                EffectStatus.FAILED: set(),
            }[effect.status]
            if target_status not in allowed:
                raise StateValidationError(
                    f"effect {action_id!r} cannot move from {effect.status.value} "
                    f"to {target_status.value}"
                )
            new_revision = state.revision + 1
            updated = PendingEffect(
                action_id=effect.action_id,
                target=effect.target,
                object_id=effect.object_id,
                status=target_status,
                goal_id=effect.goal_id,
                branch_id=effect.branch_id,
                observation_id=observation_id or effect.observation_id,
                created_revision=effect.created_revision,
                updated_revision=new_revision,
                note=effect.note if note is None else note,
            )
            candidate = state.copy()
            candidate.pending_effects[action_id] = updated
            candidate.revision = new_revision
            self._commit(candidate)
            return candidate.pending_effects[action_id]

    # ---------------------------------------------------------- verification
    def verify_goal(
        self,
        goal_id: str,
        *,
        evidence: Sequence[EvidenceRef],
        base_revision: int,
        plan_version: int,
        source_branch: str = "root",
        completed_object_ids: Iterable[str] | None = None,
        note: str = "",
    ) -> TaskState:
        """Evidence-backed transition to ``verified``.

        This is the programmatic path used by verification (batch E). The
        model-facing state tools never reach it, so an action proposal cannot
        raise the verified count.
        """

        with self._lock:
            state = self.state
            _check_basis(state, base_revision, plan_version)
            if source_branch not in state.branches:
                raise StateValidationError(
                    f"verification belongs to unregistered branch {source_branch!r}"
                )
            goal = state.goals.get(goal_id)
            if goal is None:
                raise StateValidationError(f"unknown goal {goal_id!r}")
            if goal.status is GoalStatus.VERIFIED:
                return state
            new_refs = tuple(evidence)
            if not new_refs:
                raise StateValidationError(
                    f"goal {goal_id!r} cannot be verified without evidence"
                )
            if any(not isinstance(ref, EvidenceRef) for ref in new_refs):
                raise StateValidationError("verification evidence must be EvidenceRef records")
            merged: list[EvidenceRef] = list(goal.evidence)
            seen = {ref.evidence_id for ref in merged}
            for ref in new_refs:
                if ref.evidence_id in seen:
                    raise StateValidationError(
                        f"verification repeats evidence id {ref.evidence_id!r}"
                    )
                seen.add(ref.evidence_id)
                merged.append(ref)
            completed = (
                goal.object_scope
                if completed_object_ids is None
                else tuple(completed_object_ids)
            )
            verified = Goal(
                goal_id=goal.goal_id,
                requirement=goal.requirement,
                acceptance_criteria=goal.acceptance_criteria,
                depends_on=goal.depends_on,
                object_scope=goal.object_scope,
                status=GoalStatus.VERIFIED,
                evidence=tuple(merged),
                branch_id=goal.branch_id,
                completed_object_ids=completed,
            )
            candidate = state.copy()
            candidate.goals[goal_id] = verified
            candidate.revision = state.revision + 1
            candidate.updated_at = self.clock.iso(self.clock.now())
            self._commit(candidate)
            return candidate

    def invalidate_goal(
        self,
        goal_id: str,
        reason: str,
        evidence_refs: Sequence[EvidenceRef] | None = None,
        *,
        to_status: GoalStatus | str | None = None,
        base_revision: int | None = None,
        plan_version: int | None = None,
        source_branch: str | None = None,
        event_log: Any | None = None,
    ) -> GoalInvalidation:
        """Withdraw a verified/awaiting goal without deleting its record.

        ``verified`` and ``awaiting_verification`` claims can move to
        ``invalidated`` (new conflicting evidence exists) or back to
        ``awaiting_verification`` (the claim is withdrawn but no cause is
        invented). The requirement, acceptance criteria, object scope, completed
        objects and old evidence stay; new evidence is appended. One audit record
        is written under ``state/invalidations.jsonl`` and, when an event log is
        attached, one ``recovery`` event names the goal and result.

        A repeated call for a goal already in the requested status returns the
        recorded result with ``changed=False``; it does not raise the state
        revision, append a second record or emit a second event. Unknown goals,
        an empty reason and a status that cannot be targeted raise
        :class:`StateValidationError` before any commit.
        """

        with self._lock:
            state = self.state
            _check_basis(state, base_revision, plan_version)
            goal_id = require_goal_id(goal_id)
            reason = require_reason(reason)
            goal = state.goals.get(goal_id)
            if goal is None:
                raise StateValidationError(f"unknown goal {goal_id!r}")

            new_refs: tuple[EvidenceRef, ...]
            if evidence_refs is None:
                new_refs = ()
            elif isinstance(evidence_refs, (str, bytes)) or not isinstance(
                evidence_refs, Iterable
            ):
                raise StateValidationError(
                    "withdrawal evidence must be a list of EvidenceRef records"
                )
            else:
                new_refs = tuple(evidence_refs)
            for ref in new_refs:
                if not isinstance(ref, EvidenceRef):
                    raise StateValidationError(
                        "withdrawal evidence must be EvidenceRef records"
                    )

            target = coerce_target_status(to_status, has_evidence=bool(new_refs))
            branch = source_branch or "root"
            if branch not in state.branches:
                raise StateValidationError(
                    f"withdrawal belongs to unregistered branch {branch!r}"
                )

            if goal.status is target:
                return self._repeat_invalidation(goal, target, reason, branch)
            if goal.status is GoalStatus.INVALIDATED:
                # An already withdrawn claim is never withdrawn twice; a later
                # call for it is a repeat even if it names another target status.
                return self._repeat_invalidation(
                    goal, GoalStatus.INVALIDATED, reason, branch
                )

            if target is GoalStatus.INVALIDATED and not new_refs:
                raise StateValidationError(
                    f"goal {goal_id!r} cannot be invalidated without new conflicting "
                    "evidence; use awaiting_verification when evidence is insufficient"
                )

            if goal.status not in WITHDRAWABLE_FROM:
                raise StateValidationError(
                    f"goal {goal_id!r} is {goal.status.value}; only a verified or "
                    "awaiting_verification completion claim can be withdrawn"
                )

            previous_ids = tuple(ref.evidence_id for ref in goal.evidence)
            seen = set(previous_ids)
            merged: list[EvidenceRef] = list(goal.evidence)
            appended: list[str] = []
            for ref in new_refs:
                if ref.evidence_id in seen:
                    raise StateValidationError(
                        f"withdrawal repeats evidence id {ref.evidence_id!r}"
                    )
                seen.add(ref.evidence_id)
                merged.append(ref)
                appended.append(ref.evidence_id)

            withdrawn = Goal(
                goal_id=goal.goal_id,
                requirement=goal.requirement,
                acceptance_criteria=goal.acceptance_criteria,
                depends_on=goal.depends_on,
                object_scope=goal.object_scope,
                status=target,
                evidence=tuple(merged),
                branch_id=goal.branch_id,
                completed_object_ids=goal.completed_object_ids,
            )
            candidate = state.copy()
            candidate.goals[goal_id] = withdrawn
            candidate.revision = state.revision + 1
            candidate.updated_at = self.clock.iso(self.clock.now())
            self._commit(candidate)

            record = GoalInvalidation(
                goal_id=goal_id,
                reason=reason,
                from_status=goal.status,
                to_status=target,
                changed=True,
                state_revision_before=state.revision,
                state_revision_after=candidate.revision,
                plan_version=candidate.plan_version,
                source_branch=branch,
                previous_evidence_ids=previous_ids,
                appended_evidence_ids=tuple(appended),
                evidence_ids=tuple(ref.evidence_id for ref in merged),
                recorded_at=self.clock.iso(self.clock.now()),
            )
            try:
                InvalidationLog(self._state_dir).append(record)
            except OSError as exc:
                raise StatePersistenceError(
                    f"goal {goal_id!r} was withdrawn at revision "
                    f"{candidate.revision}, but the invalidation audit could not be "
                    f"written: {exc}"
                ) from exc
            self._emit_invalidation(record, event_log or self.event_log)
            return record

    def register_branch(
        self,
        branch_id: str,
        run_id: str,
        *,
        base_revision: int | None = None,
        plan_version: int | None = None,
    ) -> TaskState:
        """Record a branch's own run id so its evidence is attributable."""

        with self._lock:
            state = self.state
            _check_basis(state, base_revision, plan_version)
            if not isinstance(branch_id, str) or not branch_id.strip():
                raise StateValidationError("branch_id must be a non-empty string")
            validate_run_id(run_id)
            existing = state.branches.get(branch_id)
            if existing == run_id:
                return state
            if existing is not None:
                raise StateValidationError(
                    f"branch {branch_id!r} is already registered to run {existing!r}"
                )
            if run_id in state.branches.values():
                raise StateValidationError(
                    f"run {run_id!r} is already registered to another branch"
                )
            candidate = state.copy()
            candidate.branches[branch_id] = run_id
            candidate.revision = state.revision + 1
            candidate.updated_at = self.clock.iso(self.clock.now())
            self._commit(candidate)
            return candidate

    # --------------------------------------------------------------- evidence
    def verify_evidence(self, ref: EvidenceRef) -> None:
        self._resolver_for(self.state).verify(ref)

    def _resolver_for(self, state: TaskState) -> EvidenceResolver:
        return EvidenceResolver(self.runtime_dir, state.branches)

    def _repeat_invalidation(
        self,
        goal: Goal,
        to_status: GoalStatus,
        reason: str,
        branch: str,
    ) -> GoalInvalidation:
        """Return the prior record, or a revision-neutral synthetic one."""

        existing = self._last_invalidation(goal.goal_id, to_status)
        if existing is not None:
            return replace(existing, changed=False)
        state = self.state
        evidence_ids = tuple(ref.evidence_id for ref in goal.evidence)
        return GoalInvalidation(
            goal_id=goal.goal_id,
            reason=reason,
            from_status=goal.status,
            to_status=to_status,
            changed=False,
            state_revision_before=state.revision,
            state_revision_after=state.revision,
            plan_version=state.plan_version,
            source_branch=branch,
            previous_evidence_ids=evidence_ids,
            appended_evidence_ids=(),
            evidence_ids=evidence_ids,
            recorded_at=self.clock.iso(self.clock.now()),
        )

    def _last_invalidation(
        self, goal_id: str, to_status: GoalStatus
    ) -> GoalInvalidation | None:
        try:
            records = InvalidationLog(self._state_dir).read()
        except (OSError, StateValidationError):
            return None
        for record in reversed(records):
            if record.goal_id == goal_id and record.to_status is to_status:
                return record
        return None

    def _emit_invalidation(
        self, record: GoalInvalidation, event_log: Any | None
    ) -> None:
        """Best-effort structured event; diagnostics never undo the ledger write."""

        if event_log is None:
            return
        emit = getattr(event_log, "emit", None)
        if not callable(emit):
            return
        try:
            emit(
                "recovery",
                goal_id=record.goal_id,
                branch_id=record.source_branch,
                state_revision=record.state_revision_after,
                plan_version=record.plan_version,
                result={"invalidation": record.to_dict()},
            )
        except Exception:  # noqa: BLE001 - an event log cannot fail the withdrawal
            return

    # ---------------------------------------------------------------- commit
    def _commit(self, candidate: TaskState) -> None:
        candidate.updated_at = self.clock.iso(self.clock.now())
        candidate.validate(evidence=self._resolver_for(candidate))
        payload = json_dumps(candidate.to_dict()).encode("utf-8")
        try:
            self._atomic_write(payload)
        except StatePersistenceError:
            raise
        except OSError as exc:
            raise StatePersistenceError(
                f"failed to save task state snapshot {self._state_path}: {exc}"
            ) from exc
        self._state = candidate

    def _atomic_write(self, data: bytes) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StatePersistenceError(
                f"cannot create task state directory {self._state_dir}: {exc}"
            ) from exc
        temporary = self._state_dir / f".state.json.{uuid.uuid4().hex}.part"
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise StatePersistenceError(
                f"failed to save task state snapshot {self._state_path}: {exc}"
            ) from exc
        # Best-effort directory durability; Windows does not support opening a
        # directory as a file descriptor, so this is intentionally optional.
        try:
            directory_fd = os.open(self._state_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)


__all__ = [
    "StateNotFound",
    "StatePersistenceError",
    "StateStore",
]
