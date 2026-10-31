"""
Execution tracking system for transparent agent operation visibility.

Tracks all execution events including:
- Task start/completion
- Routing decisions
- Task decomposition
- Sub-agent spawning and execution
- Tool calls
- Result synthesis
- Errors and retries
"""

from __future__ import annotations

import itertools  # noqa: F401  re-exported
import json  # noqa: F401  re-exported
import threading  # noqa: F401  re-exported
import time  # noqa: F401  re-exported
from dataclasses import dataclass, field  # noqa: F401  re-exported
from datetime import datetime  # noqa: F401  re-exported
from enum import Enum  # noqa: F401  re-exported
from typing import Any

from effgen.core.execution_tracker_events import (  # noqa: F401  re-exported
    _EVENT_SEQ,
    _EVENT_SEQ_LOCK,
    EventType,
    ExecutionEvent,
    ExecutionNode,
    ExecutionStatus,
    _next_event_id,
)
from effgen.core.execution_tracker_metrics import ExecutionTrackerMetricsMixin
from effgen.core.execution_tracker_render import ExecutionTrackerRenderMixin
from effgen.core.execution_tracker_state import ExecutionTrackerStateMixin


class ExecutionTracker(
    ExecutionTrackerStateMixin,
    ExecutionTrackerMetricsMixin,
    ExecutionTrackerRenderMixin,
):
    """
    Track and display execution progress with full transparency.

    Provides real-time visibility into:
    - Parent agent reasoning
    - Sub-agent spawning
    - Subtask assignments
    - Tool executions
    - Intermediate results
    - Errors and retries
    - Final synthesis
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize execution tracker.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.events: list[ExecutionEvent] = []
        self.nodes: dict[str, ExecutionNode] = {}
        self.root_node_id: str | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.active_agents: set = set()
        self.active_tools: set = set()
        # The sub-agent currently executing, if any. Tool/reasoning events that
        # arrive while a sub-agent is running hang under it in the tree; events
        # outside any delegation hang under the root task.
        self._active_agent_id: str | None = None
        # Optional live observers (e.g. the CLI status line). Each is called
        # synchronously with every tracked event; a misbehaving listener must
        # never break the run, so calls are wrapped in a try/except.
        self._listeners: list[Any] = []

    def __repr__(self) -> str:
        """String representation."""
        return f"ExecutionTracker(events={len(self.events)}, nodes={len(self.nodes)})"


# The mixins are how this module is organised internally, not part of what it
# offers: they are composed into ``ExecutionTracker`` above and stay importable
# from their own modules. Unbinding them here keeps the set of names this module
# exposes the same as when it was a single file.
del ExecutionTrackerMetricsMixin, ExecutionTrackerRenderMixin, ExecutionTrackerStateMixin
