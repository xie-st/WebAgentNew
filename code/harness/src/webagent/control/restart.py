"""Restart from the complete ledger snapshot (E08 control part).

Only public state/evidence interfaces are used:

* ``StateStore.load`` reads the latest complete ``state.json``;
* ``StateStore.register_branch`` records the restarted run's own run id;
* ``StateStore.pending_effects`` returns the unconfirmed residue.

The restarted run gets a new ``Session`` and a fresh observation. Nothing here
replays a page action, restores a Python generator stack, or claims that the
previous browser state is still current. Integration with E1's recovery policy
uses the same two public entry points and does not require E1's implementation.

:func:`load_restart_snapshot` is the load/register half without a session. The
official entry already owns its session, budget and driver, so it uses that half
and then builds the normal task container around the loaded branch; the
standalone :func:`restart_from_snapshot` remains the complete entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..clock import Clock, SystemClock
from ..evidence import EvidenceStore
from ..protocol import Observation, validate_run_id
from ..session import Session
from ..state.records import PendingEffect, TaskState
from ..state.store import StateStore
from .events import ControlEventLog


@dataclass
class RestartLoad:
    """The loaded snapshot and the newly registered branch, without a session."""

    root_run_id: str
    branch_id: str
    new_run_id: str
    store: StateStore
    state: TaskState
    pending_effects: tuple[PendingEffect, ...]
    loaded_revision: int
    plan_version: int
    branches: dict[str, str]
    task: str = ""

    @property
    def pending_effect_ids(self) -> tuple[str, ...]:
        return tuple(effect.action_id for effect in self.pending_effects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "branch_id": self.branch_id,
            "new_run_id": self.new_run_id,
            "loaded_revision": self.loaded_revision,
            "plan_version": self.plan_version,
            "branches": dict(self.branches),
            "pending_effects": [effect.to_dict() for effect in self.pending_effects],
            "task": self.task,
            "generator_restored": False,
            "replayed_actions": [],
        }


def _check_restart_ids(root_run_id: str, branch_id: str, new_run_id: str) -> None:
    validate_run_id(root_run_id)
    validate_run_id(new_run_id)
    if not isinstance(branch_id, str) or not branch_id.strip():
        raise ValueError("branch_id must be a non-empty string")
    if branch_id == "root":
        raise ValueError("a restarted execution needs a branch id other than 'root'")


def load_restart_snapshot(
    runtime_dir: str | Path,
    root_run_id: str,
    *,
    branch_id: str,
    new_run_id: str,
    clock: Clock | None = None,
    control_log: ControlEventLog | None = None,
) -> RestartLoad:
    """Load the complete snapshot and register the restarted branch's run id."""

    _check_restart_ids(root_run_id, branch_id, new_run_id)
    runtime = Path(runtime_dir)
    clock = clock or SystemClock()
    store = StateStore(runtime, root_run_id, clock=clock)
    state = store.load()
    registered = store.register_branch(
        branch_id,
        new_run_id,
        base_revision=state.revision,
        plan_version=state.plan_version,
    )
    pending = tuple(registered.unconfirmed_effects)
    log = control_log or ControlEventLog(
        new_run_id,
        runtime / "runs" / new_run_id / "control_events.jsonl",
        clock=clock,
    )
    try:
        log.append(
            "restart_loaded",
            control_epoch=1,
            page_epoch=0,
            reason="complete_snapshot",
            source="restart",
            details={
                "root_run_id": root_run_id,
                "branch_id": branch_id,
                "loaded_revision": registered.revision,
                "plan_version": registered.plan_version,
                "pending_effect_ids": [effect.action_id for effect in pending],
                "generator_restored": False,
            },
        )
    finally:
        if control_log is None:
            log.close()
    return RestartLoad(
        root_run_id=root_run_id,
        branch_id=branch_id,
        new_run_id=new_run_id,
        store=store,
        state=registered,
        pending_effects=pending,
        loaded_revision=registered.revision,
        plan_version=registered.plan_version,
        branches=dict(registered.branches),
        task=registered.task,
    )


@dataclass
class RestartResult:
    root_run_id: str
    branch_id: str
    new_run_id: str
    loaded_revision: int
    plan_version: int
    branches: dict[str, str]
    pending_effects: tuple[PendingEffect, ...]
    observation: Observation
    session_state: str
    generator_restored: bool = False
    replayed_actions: tuple[str, ...] = ()
    task: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "branch_id": self.branch_id,
            "new_run_id": self.new_run_id,
            "loaded_revision": self.loaded_revision,
            "plan_version": self.plan_version,
            "branches": dict(self.branches),
            "pending_effects": [effect.to_dict() for effect in self.pending_effects],
            "observation": self.observation.to_local_dict(),
            "session_state": self.session_state,
            "generator_restored": self.generator_restored,
            "replayed_actions": list(self.replayed_actions),
            "task": self.task,
        }


def restart_from_snapshot(
    runtime_dir: str | Path,
    root_run_id: str,
    *,
    branch_id: str,
    new_run_id: str,
    observer: Callable[[], Observation],
    clock: Clock | None = None,
) -> RestartResult:
    """Load the snapshot, register a restarted branch and re-observe once."""

    if not callable(observer):
        raise TypeError("restart needs an observer that returns a fresh Observation")
    runtime = Path(runtime_dir)
    clock = clock or SystemClock()
    load = load_restart_snapshot(
        runtime,
        root_run_id,
        branch_id=branch_id,
        new_run_id=new_run_id,
        clock=clock,
    )
    store = load.store

    session = Session(
        new_run_id,
        runtime_dir=runtime,
        clock=clock,
        evidence=EvidenceStore(runtime, new_run_id),
    )
    log = ControlEventLog(
        new_run_id,
        session.evidence.run_dir / "control_events.jsonl",
        clock=clock,
    )
    try:
        session.start_task(load.task)
        observation = observer()
        if not isinstance(observation, Observation):
            raise TypeError("restart observer must return an Observation")
        session.ingest_observation(observation)
        log.append(
            "reobserved",
            control_epoch=1,
            page_epoch=1,
            reason="restart_reobserve",
            source="restart",
            observation_id=observation.observation_id,
            details={
                "generator_restored": False,
                "replayed_actions": [],
                "old_element_numbers_valid": False,
            },
        )
    finally:
        log.close()

    return RestartResult(
        root_run_id=root_run_id,
        branch_id=branch_id,
        new_run_id=new_run_id,
        loaded_revision=load.loaded_revision,
        plan_version=load.plan_version,
        branches=dict(load.branches),
        pending_effects=load.pending_effects,
        observation=observation,
        session_state=session.state.value,
        generator_restored=False,
        replayed_actions=(),
        task=load.task,
        extra={
            "state_path": str(store.state_path),
            "evidence_dir": str(session.evidence.run_dir),
            "control_events": str(
                session.evidence.run_dir / "control_events.jsonl"
            ),
        },
    )


__all__ = [
    "RestartLoad",
    "RestartResult",
    "load_restart_snapshot",
    "restart_from_snapshot",
]

