"""The execution-tracker state machine.

Turns the stream of :class:`~effgen.core.execution_tracker_events.ExecutionEvent`
objects into state a consumer can read: the parent link each event gets, the
sets of active agents and tools, the run's start and end, the execution tree,
and the live status snapshot. Event listeners are registered here too, and are
called as each event is tracked.

The class is a mixin: it reads and mutates the attributes
:class:`~effgen.core.execution_tracker.ExecutionTracker` creates in its
``__init__``, and is composed into that class.
"""

from __future__ import annotations

import time
from typing import Any

from effgen.core.execution_tracker_events import (
    EventType,
    ExecutionEvent,
    ExecutionNode,
    ExecutionStatus,
)


class ExecutionTrackerStateMixin:
    """Event intake, hierarchy assignment, tree building and live status."""

    def add_listener(self, callback: Any) -> None:
        """Register a callback invoked with each :class:`ExecutionEvent`.

        Used by the CLI to drive a live status line off the events the agent
        already emits. Safe to call before a run; remove with
        :meth:`remove_listener` when done.
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Any) -> None:
        """Unregister a previously added event listener (no error if absent)."""
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def track_event(self, event: ExecutionEvent) -> None:
        """
        Record execution event.

        Args:
            event: Execution event to track
        """
        # Link the event into the hierarchy before it is stored so a consumer
        # reading the flat trace can reconstruct the tree from parent ids.
        self._assign_parent(event)

        self.events.append(event)

        # Update internal state based on event type
        self._update_state(event)

        # Update execution tree
        self._update_tree(event)

        # Notify any live observers (cosmetic; never let one break the run).
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - observers are best-effort
                pass

    # Event types that describe delegation structure rather than work done
    # inside a step; they hang directly under the root task, not under a
    # currently-running sub-agent.
    _STRUCTURAL_EVENTS = frozenset({
        EventType.SUB_AGENT_SPAWN,
        EventType.SUB_AGENT_START,
        EventType.SUB_AGENT_PROGRESS,
        EventType.SUB_AGENT_COMPLETE,
        EventType.SUB_AGENT_FAILED,
        EventType.ROUTING_DECISION,
        EventType.TASK_DECOMPOSITION,
    })

    def _assign_parent(self, event: ExecutionEvent) -> None:
        """Fill ``parent_event_id`` so the flat trace forms a tree.

        The root task has no parent. A delegation event hangs under the root; a
        tool or reasoning step hangs under the sub-agent running it, or the root
        task when no delegation is active. An explicit ``parent_event_id`` set by
        the caller is left untouched.
        """
        if event.type == EventType.TASK_START or event.parent_event_id is not None:
            return
        if event.type in self._STRUCTURAL_EVENTS:
            event.parent_event_id = self.root_node_id
        else:
            event.parent_event_id = self._active_agent_id or self.root_node_id

    def _update_state(self, event: ExecutionEvent):
        """Update internal state based on event."""
        if event.type == EventType.TASK_START:
            if self.start_time is None:
                self.start_time = event.timestamp

        elif event.type in [EventType.TASK_COMPLETE, EventType.TASK_FAILED]:
            self.end_time = event.timestamp

        elif event.type == EventType.SUB_AGENT_START:
            if event.agent_id:
                self.active_agents.add(event.agent_id)
                self._active_agent_id = event.agent_id

        elif event.type in [EventType.SUB_AGENT_COMPLETE, EventType.SUB_AGENT_FAILED]:
            if event.agent_id:
                self.active_agents.discard(event.agent_id)
                if self._active_agent_id == event.agent_id:
                    self._active_agent_id = None

        elif event.type == EventType.TOOL_CALL_START:
            tool_name = event.data.get("tool_name", "unknown")
            self.active_tools.add(tool_name)

        elif event.type in [EventType.TOOL_CALL_COMPLETE, EventType.TOOL_CALL_FAILED]:
            tool_name = event.data.get("tool_name", "unknown")
            self.active_tools.discard(tool_name)

    def _update_tree(self, event: ExecutionEvent):
        """Update execution tree based on event."""
        if event.type == EventType.TASK_START:
            # Create root node
            node = ExecutionNode(
                node_id=event.event_id,
                node_type="task",
                name=event.data.get("task", "Main Task"),
                status="running",
                started_at=event.timestamp,
                metadata=event.data
            )
            self.nodes[node.node_id] = node
            if self.root_node_id is None:
                self.root_node_id = node.node_id

        elif event.type == EventType.SUB_AGENT_SPAWN:
            # Create sub-agent node
            agent_id = event.agent_id or event.event_id
            node = ExecutionNode(
                node_id=agent_id,
                node_type="agent",
                name=event.data.get("agent_name", agent_id),
                status="pending",
                parent_id=event.parent_event_id or self.root_node_id,
                metadata=event.data
            )
            self.nodes[node.node_id] = node

            # Add to parent's children
            if node.parent_id and node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(node)

        elif event.type == EventType.SUB_AGENT_START:
            # Update agent node status
            if event.agent_id and event.agent_id in self.nodes:
                self.nodes[event.agent_id].status = "running"
                self.nodes[event.agent_id].started_at = event.timestamp

        elif event.type == EventType.SUB_AGENT_COMPLETE:
            # Update agent node status
            if event.agent_id and event.agent_id in self.nodes:
                self.nodes[event.agent_id].status = "completed"
                self.nodes[event.agent_id].completed_at = event.timestamp

        elif event.type == EventType.SUB_AGENT_FAILED:
            # Update agent node status
            if event.agent_id and event.agent_id in self.nodes:
                self.nodes[event.agent_id].status = "failed"
                self.nodes[event.agent_id].completed_at = event.timestamp

        elif event.type == EventType.TOOL_CALL_START:
            # Create a tool node under the step that spawned it. ``agent_id`` on
            # a tool event is the agent's *name*, not a node id, so resolve the
            # parent from the assigned ``parent_event_id`` and fall back to the
            # root task when the referenced node is unknown.
            parent_id = event.parent_event_id
            if parent_id not in self.nodes:
                parent_id = event.agent_id if event.agent_id in self.nodes else self.root_node_id
            node = ExecutionNode(
                node_id=event.event_id,
                node_type="tool",
                name=event.data.get("tool_name", "Tool"),
                status="running",
                started_at=event.timestamp,
                parent_id=parent_id,
                metadata=event.data
            )
            self.nodes[node.node_id] = node

            # Add to parent's children
            if parent_id and parent_id in self.nodes:
                self.nodes[parent_id].children.append(node)

        elif event.type in (EventType.TOOL_CALL_COMPLETE, EventType.TOOL_CALL_FAILED):
            # A completion event carries no id back-reference, so finalize the
            # most recent still-running tool node with a matching name.
            tool_node = self._latest_running_tool(event.data.get("tool_name"))
            if tool_node is not None:
                tool_node.status = (
                    "completed" if event.type == EventType.TOOL_CALL_COMPLETE else "failed"
                )
                tool_node.completed_at = event.timestamp
                for key in ("result", "error"):
                    if event.data.get(key) is not None:
                        tool_node.metadata = {**tool_node.metadata, key: event.data[key]}

        elif event.type in (EventType.TASK_COMPLETE, EventType.TASK_FAILED):
            # Finalize the root task and any tool nodes still marked running so
            # the tree reports a terminal status and a real duration.
            if self.root_node_id and self.root_node_id in self.nodes:
                root = self.nodes[self.root_node_id]
                root.status = (
                    "completed" if event.type == EventType.TASK_COMPLETE else "failed"
                )
                root.completed_at = event.timestamp
                for key in ("execution_time", "tokens_used", "tool_calls"):
                    if key in event.data:
                        root.metadata = {**root.metadata, key: event.data[key]}
            for node in self.nodes.values():
                if node.node_type == "tool" and node.status == "running":
                    node.status = "completed"
                    node.completed_at = event.timestamp

    def _latest_running_tool(self, tool_name: str | None) -> ExecutionNode | None:
        """Return the most recently started running tool node (matching name)."""
        for node in reversed(list(self.nodes.values())):
            if node.node_type != "tool" or node.status != "running":
                continue
            if tool_name is None or node.name == tool_name:
                return node
        return None

    def get_live_status(self) -> ExecutionStatus:
        """
        Get current execution status.

        Returns:
            ExecutionStatus with current state
        """
        # Count subtasks by status
        completed = 0
        failed = 0
        total = 0

        for node in self.nodes.values():
            if node.node_type in ["agent", "subtask"]:
                total += 1
                if node.status == "completed":
                    completed += 1
                elif node.status == "failed":
                    failed += 1

        pending = total - completed - failed

        # Calculate progress
        progress = (completed / total * 100) if total > 0 else 0.0

        # Get current operations
        current_ops = []
        for agent_id in self.active_agents:
            if agent_id in self.nodes:
                current_ops.append(self.nodes[agent_id].name)

        # Calculate elapsed time
        elapsed = 0.0
        if self.start_time:
            end = self.end_time or time.time()
            elapsed = end - self.start_time

        # Estimate remaining time
        estimated_remaining = None
        if progress > 0 and progress < 100:
            estimated_remaining = (elapsed / progress) * (100 - progress)

        return ExecutionStatus(
            active_agents=list(self.active_agents),
            completed_subtasks=completed,
            total_subtasks=total,
            pending_subtasks=pending,
            failed_subtasks=failed,
            current_operations=current_ops,
            progress_percentage=progress,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining
        )

    def generate_execution_tree(self) -> dict[str, Any]:
        """
        Generate visual execution tree.

        Returns:
            Execution tree as nested dictionary
        """
        if self.root_node_id and self.root_node_id in self.nodes:
            return self.nodes[self.root_node_id].to_dict()
        return {}

    def get_trace(self) -> list[dict[str, Any]]:
        """
        Get full execution trace.

        Returns:
            List of all events as dictionaries
        """
        return [event.to_dict() for event in self.events]

    def clear(self) -> None:
        """Clear all tracking data."""
        self.events = []
        self.nodes = {}
        self.root_node_id = None
        self.start_time = None
        self.end_time = None
        self.active_agents = set()
        self.active_tools = set()
        self._active_agent_id = None
