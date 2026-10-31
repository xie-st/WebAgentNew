"""
Thread-safe shared state for effGen multi-agent orchestration.

Provides a namespaced key-value store that multiple agents can
read/write concurrently:
- Read/write locks (threading.RLock per namespace)
- State snapshots for rollback
- Event-sourced mutation log
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of state mutations."""
    SET = "set"
    DELETE = "delete"
    CLEAR = "clear"


@dataclass
class StateMutation:
    """
    A single recorded state change.

    Attributes:
        mutation_id: Unique identifier
        namespace: Namespace affected
        mutation_type: Type of change
        key: Key affected (None for CLEAR)
        value: New value (None for DELETE/CLEAR)
        old_value: Previous value (None if key was new)
        timestamp: When the mutation occurred
        agent_id: Agent that performed the mutation
    """
    mutation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace: str = ""
    mutation_type: MutationType = MutationType.SET
    key: str | None = None
    value: Any = None
    old_value: Any = None
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the mutation as a JSON-serializable dict."""
        return {
            "mutation_id": self.mutation_id,
            "namespace": self.namespace,
            "mutation_type": self.mutation_type.value,
            "key": self.key,
            "value": self.value,
            "old_value": self.old_value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
        }


@dataclass
class StateSnapshot:
    """
    Point-in-time snapshot of the shared state.

    Attributes:
        snapshot_id: Unique identifier
        data: Deep copy of the state at snapshot time
        timestamp: When the snapshot was taken
        mutation_index: Index into the mutation log at snapshot time
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    mutation_index: int = 0


class SharedState:
    """
    Thread-safe namespaced key-value store for multi-agent context sharing.

    Each namespace is protected by its own ``threading.RLock`` so that
    agents working in different namespaces do not block each other.

    Usage::

        state = SharedState()
        state.set("workflow", "status", "running", agent_id="agent_1")
        state.get("workflow", "status")  # -> "running"
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()  # protects _data / _locks dicts
        self._mutations: list[StateMutation] = []
        self._snapshots: list[StateSnapshot] = []

    def _get_lock(self, namespace: str) -> threading.RLock:
        """Get or create the lock for a namespace."""
        with self._global_lock:
            if namespace not in self._locks:
                self._locks[namespace] = threading.RLock()
            return self._locks[namespace]

    # -- Read/Write API --

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """
        Read a value from the shared state.

        Args:
            namespace: Namespace (e.g., workflow name, team name)
            key: Key to read
            default: Default if key is missing

        Returns:
            The stored value or *default*
        """
        lock = self._get_lock(namespace)
        with lock:
            return self._data.get(namespace, {}).get(key, default)

    def set(self, namespace: str, key: str, value: Any,
            agent_id: str = "") -> None:
        """
        Write a value to the shared state.

        Args:
            namespace: Namespace
            key: Key to write
            value: Value to store
            agent_id: ID of the agent performing the write
        """
        lock = self._get_lock(namespace)
        with lock:
            ns = self._data.setdefault(namespace, {})
            old_value = ns.get(key)
            ns[key] = value

            self._mutations.append(StateMutation(
                namespace=namespace,
                mutation_type=MutationType.SET,
                key=key,
                value=value,
                old_value=old_value,
                agent_id=agent_id,
            ))

    def delete(self, namespace: str, key: str, agent_id: str = "") -> Any:
        """Delete a key from the shared state.

        Args:
            namespace: The namespace the key lives in.
            key: The key to delete.
            agent_id: The agent recorded as making the change.

        Returns:
            The deleted value, or None if key was missing
        """
        lock = self._get_lock(namespace)
        with lock:
            ns = self._data.get(namespace, {})
            old_value = ns.pop(key, None)

            self._mutations.append(StateMutation(
                namespace=namespace,
                mutation_type=MutationType.DELETE,
                key=key,
                old_value=old_value,
                agent_id=agent_id,
            ))

            return old_value

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        """Return a shallow copy of all keys in a namespace."""
        lock = self._get_lock(namespace)
        with lock:
            return dict(self._data.get(namespace, {}))

    def clear_namespace(self, namespace: str, agent_id: str = "") -> None:
        """Remove all keys in a namespace."""
        lock = self._get_lock(namespace)
        with lock:
            self._data.pop(namespace, None)
            self._mutations.append(StateMutation(
                namespace=namespace,
                mutation_type=MutationType.CLEAR,
                agent_id=agent_id,
            ))

    def namespaces(self) -> list[str]:
        """List all active namespaces."""
        with self._global_lock:
            return list(self._data.keys())

    def has(self, namespace: str, key: str) -> bool:
        """Check if a key exists in a namespace."""
        lock = self._get_lock(namespace)
        with lock:
            return key in self._data.get(namespace, {})

    # -- Snapshots & Rollback --

    def snapshot(self) -> str:
        """
        Take a point-in-time snapshot of the entire state.

        Returns:
            snapshot_id that can be used to rollback
        """
        with self._global_lock:
            snap = StateSnapshot(
                data=copy.deepcopy(self._data),
                mutation_index=len(self._mutations),
            )
            self._snapshots.append(snap)
            logger.debug("State snapshot taken: %s", snap.snapshot_id)
            return snap.snapshot_id

    def rollback(self, snapshot_id: str) -> bool:
        """
        Restore state to a previous snapshot.

        Args:
            snapshot_id: ID returned by ``snapshot()``

        Returns:
            True if rollback succeeded, False if snapshot not found
        """
        target = None
        for snap in self._snapshots:
            if snap.snapshot_id == snapshot_id:
                target = snap
                break

        if target is None:
            logger.warning("Snapshot %s not found", snapshot_id)
            return False

        with self._global_lock:
            self._data = copy.deepcopy(target.data)
            # Truncate mutation log to the snapshot point
            self._mutations = self._mutations[:target.mutation_index]
            logger.debug("State rolled back to snapshot %s", snapshot_id)
            return True

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all available snapshots."""
        return [
            {
                "snapshot_id": s.snapshot_id,
                "timestamp": s.timestamp,
                "mutation_index": s.mutation_index,
            }
            for s in self._snapshots
        ]

    # -- Event Log --

    def get_mutations(self, namespace: str | None = None,
                      agent_id: str | None = None,
                      limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get the mutation event log.

        Args:
            namespace: Filter by namespace
            agent_id: Filter by agent
            limit: Max entries to return (most recent)

        Returns:
            List of mutation dicts
        """
        mutations = self._mutations

        if namespace:
            mutations = [m for m in mutations if m.namespace == namespace]
        if agent_id:
            mutations = [m for m in mutations if m.agent_id == agent_id]
        if limit:
            mutations = mutations[-limit:]

        return [m.to_dict() for m in mutations]

    # -- Utility --

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of all state data."""
        with self._global_lock:
            return copy.deepcopy(self._data)

    def clear(self) -> None:
        """Clear all data, mutations, and snapshots."""
        with self._global_lock:
            self._data.clear()
            self._mutations.clear()
            self._snapshots.clear()
            self._locks.clear()

    def __repr__(self) -> str:
        with self._global_lock:
            ns_count = len(self._data)
            key_count = sum(len(v) for v in self._data.values())
        return f"SharedState(namespaces={ns_count}, keys={key_count}, mutations={len(self._mutations)})"
