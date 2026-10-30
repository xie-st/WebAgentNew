"""Agent lifecycle: state machine, pre-warmed pools, registry, timeouts and cancellation.

Provides:
- AgentLifecycleState: state machine for agent lifecycle
- AgentPool: pre-warmed pool of agents for fast dispatch
- AgentRegistry: central registry of all active agents
- Per-agent timeout and cancellation support
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AgentLifecycleState(Enum):
    """Lifecycle states for an agent."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


# Valid state transitions
_TRANSITIONS: dict[AgentLifecycleState, set[AgentLifecycleState]] = {
    AgentLifecycleState.CREATED: {AgentLifecycleState.INITIALIZING, AgentLifecycleState.TERMINATED},
    AgentLifecycleState.INITIALIZING: {AgentLifecycleState.READY, AgentLifecycleState.FAILED, AgentLifecycleState.TERMINATED},
    AgentLifecycleState.READY: {AgentLifecycleState.RUNNING, AgentLifecycleState.TERMINATED},
    AgentLifecycleState.RUNNING: {AgentLifecycleState.PAUSED, AgentLifecycleState.COMPLETED, AgentLifecycleState.FAILED, AgentLifecycleState.TERMINATED},
    AgentLifecycleState.PAUSED: {AgentLifecycleState.RUNNING, AgentLifecycleState.TERMINATED},
    AgentLifecycleState.COMPLETED: set(),
    AgentLifecycleState.FAILED: {AgentLifecycleState.INITIALIZING, AgentLifecycleState.TERMINATED},
    AgentLifecycleState.TERMINATED: set(),
}


@dataclass
class AgentEntry:
    """
    Registry entry for a tracked agent.

    Attributes:
        agent_id: Unique identifier
        agent: The Agent instance
        state: Current lifecycle state
        created_at: Creation timestamp
        started_at: When execution started
        completed_at: When execution finished
        timeout: Per-agent timeout in seconds (0 = no timeout)
        cancel_event: Threading event to signal cancellation
        metadata: Arbitrary metadata
    """
    agent_id: str
    agent: Any  # Agent instance
    state: AgentLifecycleState = AgentLifecycleState.CREATED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timeout: float = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, new_state: AgentLifecycleState) -> None:
        """
        Transition to a new state, validating the transition.

        Raises ValueError for invalid transitions.
        """
        valid = _TRANSITIONS.get(self.state, set())
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition: {self.state.value} -> {new_state.value}"
            )
        old = self.state
        self.state = new_state

        if new_state == AgentLifecycleState.RUNNING:
            self.started_at = time.time()
        elif new_state in (AgentLifecycleState.COMPLETED,
                           AgentLifecycleState.FAILED,
                           AgentLifecycleState.TERMINATED):
            self.completed_at = time.time()

        logger.debug("Agent %s: %s -> %s", self.agent_id, old.value, new_state.value)

    @property
    def is_active(self) -> bool:
        """True while the agent has not completed, failed or terminated."""
        return self.state in (
            AgentLifecycleState.CREATED,
            AgentLifecycleState.INITIALIZING,
            AgentLifecycleState.READY,
            AgentLifecycleState.RUNNING,
            AgentLifecycleState.PAUSED,
        )

    @property
    def elapsed(self) -> float | None:
        """Seconds since the agent started (``None`` before it starts)."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def is_timed_out(self) -> bool:
        """True when a positive timeout is configured and has elapsed."""
        if self.timeout <= 0 or self.started_at is None:
            return False
        return (time.time() - self.started_at) > self.timeout

    def cancel(self) -> None:
        """Signal cancellation to the agent."""
        self.cancel_event.set()
        if self.state in (AgentLifecycleState.RUNNING, AgentLifecycleState.PAUSED,
                          AgentLifecycleState.READY):
            self.state = AgentLifecycleState.TERMINATED
            self.completed_at = time.time()
        logger.debug("Agent %s cancelled", self.agent_id)

    def to_dict(self) -> dict[str, Any]:
        """Return the entry's state and timing as a JSON-serializable dict."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed": self.elapsed,
            "timeout": self.timeout,
            "is_timed_out": self.is_timed_out,
            "metadata": self.metadata,
        }


class AgentRegistry:
    """
    Central registry of all active agents.

    Thread-safe registry that tracks agent lifecycle states
    and provides lookup/query capabilities.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentEntry] = {}
        self._lock = threading.RLock()

    def register(self, agent_id: str, agent: Any,
                 timeout: float = 0,
                 metadata: dict[str, Any] | None = None) -> AgentEntry:
        """
        Register a new agent.

        Args:
            agent_id: Unique identifier
            agent: Agent instance
            timeout: Execution timeout (0 = no timeout)
            metadata: Optional metadata

        Returns:
            The created AgentEntry
        """
        with self._lock:
            if agent_id in self._agents:
                raise ValueError(f"Agent '{agent_id}' already registered")

            entry = AgentEntry(
                agent_id=agent_id,
                agent=agent,
                timeout=timeout,
                metadata=metadata or {},
            )
            self._agents[agent_id] = entry
            return entry

    def unregister(self, agent_id: str) -> AgentEntry | None:
        """Remove an agent from the registry."""
        with self._lock:
            return self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> AgentEntry | None:
        """Look up an agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_by_state(self, state: AgentLifecycleState) -> list[AgentEntry]:
        """Get all agents in a given state."""
        with self._lock:
            return [e for e in self._agents.values() if e.state == state]

    def active_agents(self) -> list[AgentEntry]:
        """Get all currently active agents."""
        with self._lock:
            return [e for e in self._agents.values() if e.is_active]

    def transition(self, agent_id: str, new_state: AgentLifecycleState) -> None:
        """Transition an agent to a new state."""
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                raise ValueError(f"Agent '{agent_id}' not found in registry")
            entry.transition(new_state)

    def cancel(self, agent_id: str) -> bool:
        """Cancel an agent. Returns True if the agent was found."""
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return False
            entry.cancel()
            return True

    def cancel_all(self) -> int:
        """Cancel all active agents. Returns count of cancelled agents."""
        with self._lock:
            count = 0
            for entry in self._agents.values():
                if entry.is_active:
                    entry.cancel()
                    count += 1
            return count

    def check_timeouts(self) -> list[str]:
        """
        Check for timed-out agents and terminate them.

        Returns:
            List of agent IDs that were terminated due to timeout
        """
        terminated = []
        with self._lock:
            for entry in self._agents.values():
                if entry.is_timed_out and entry.is_active:
                    entry.cancel()
                    terminated.append(entry.agent_id)
                    logger.warning(
                        "Agent %s timed out after %.1fs (limit: %.1fs)",
                        entry.agent_id, entry.elapsed, entry.timeout,
                    )
        return terminated

    def list_agents(self) -> list[dict[str, Any]]:
        """List all registered agents as dicts."""
        with self._lock:
            return [e.to_dict() for e in self._agents.values()]

    def clear(self) -> None:
        """Remove all agents from the registry."""
        with self._lock:
            self._agents.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __repr__(self) -> str:
        with self._lock:
            active = sum(1 for e in self._agents.values() if e.is_active)
        return f"AgentRegistry(total={len(self._agents)}, active={active})"


class AgentPool:
    """
    Pre-warmed pool of agents for fast dispatch.

    Maintains a pool of ready-to-use agents that can be checked
    out and returned. Uses a simple list-based pool.
    """

    def __init__(self, max_size: int = 10) -> None:
        """
        Initialize the agent pool.

        Args:
            max_size: Maximum number of agents in the pool
        """
        self.max_size = max_size
        self._pool: list[Any] = []  # Available agents
        self._checked_out: dict[str, Any] = {}  # agent_id -> agent
        self._lock = threading.RLock()

    def add(self, agent: Any) -> bool:
        """
        Add a pre-warmed agent to the pool.

        Args:
            agent: Agent instance (must have a ``name`` attribute)

        Returns:
            True if added, False if pool is full
        """
        with self._lock:
            if len(self._pool) >= self.max_size:
                return False
            self._pool.append(agent)
            return True

    def acquire(self) -> Any | None:
        """
        Check out an agent from the pool.

        Returns:
            An Agent instance, or None if the pool is empty
        """
        with self._lock:
            if not self._pool:
                return None
            agent = self._pool.pop(0)
            agent_id = getattr(agent, "name", id(agent))
            self._checked_out[str(agent_id)] = agent
            return agent

    def release(self, agent: Any) -> None:
        """
        Return an agent to the pool.

        Args:
            agent: The agent to return
        """
        with self._lock:
            agent_id = str(getattr(agent, "name", id(agent)))
            self._checked_out.pop(agent_id, None)

            if len(self._pool) < self.max_size:
                self._pool.append(agent)

    @property
    def available(self) -> int:
        """Number of agents available in the pool."""
        with self._lock:
            return len(self._pool)

    @property
    def in_use(self) -> int:
        """Number of agents currently checked out."""
        with self._lock:
            return len(self._checked_out)

    def clear(self) -> None:
        """Clear the pool and all checked-out references."""
        with self._lock:
            self._pool.clear()
            self._checked_out.clear()

    def __repr__(self) -> str:
        return f"AgentPool(available={self.available}, in_use={self.in_use}, max={self.max_size})"
