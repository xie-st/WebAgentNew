"""Execution-tracker metrics.

Derives numbers from the tracked events and the execution tree: the run summary,
the timing / throughput / resource / efficiency metric groups, the bottlenecks
they imply, and the critical path through the tree.

Every entry point reports a guard dict rather than raising when there is nothing
to measure — no events, no elapsed time, no subtasks.

The class is a mixin: it reads the attributes
:class:`~effgen.core.execution_tracker.ExecutionTracker` creates in its
``__init__``, and is composed into that class.
"""

from __future__ import annotations

from typing import Any

from effgen.core.execution_tracker_events import EventType


class ExecutionTrackerMetricsMixin:
    """Summary, performance metrics, bottlenecks and the critical path."""

    def get_summary(self) -> dict[str, Any]:
        """
        Get execution summary.

        Returns:
            Summary dictionary with key metrics
        """
        status = self.get_live_status()

        # Count events by type
        event_counts = {}
        for event in self.events:
            event_type = event.type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Get tool usage
        tool_calls = [e for e in self.events if e.type == EventType.TOOL_CALL_START]
        tool_usage = {}
        for call in tool_calls:
            tool_name = call.data.get("tool_name", "unknown")
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "status": status.to_dict(),
            "tool_usage": tool_usage,
            "execution_tree": self.generate_execution_tree(),
            "duration": status.elapsed_time
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get detailed performance metrics from execution.

        Returns:
            Performance metrics dictionary
        """
        if not self.events:
            return {"error": "No execution data available"}

        metrics = {
            "timing": self._calculate_timing_metrics(),
            "throughput": self._calculate_throughput_metrics(),
            "resource_usage": self._calculate_resource_usage(),
            "efficiency": self._calculate_efficiency_metrics(),
            "bottlenecks": self._identify_bottlenecks()
        }

        return metrics

    def _calculate_timing_metrics(self) -> dict[str, Any]:
        """Calculate timing-related metrics."""
        status = self.get_live_status()

        # Agent timing
        agent_times = []
        for node in self.nodes.values():
            if node.node_type == "agent" and node.get_duration():
                agent_times.append(node.get_duration())

        # Tool timing
        tool_times = []
        for node in self.nodes.values():
            if node.node_type == "tool" and node.get_duration():
                tool_times.append(node.get_duration())

        return {
            "total_elapsed": status.elapsed_time,
            "agent_execution": {
                "total": sum(agent_times) if agent_times else 0,
                "average": sum(agent_times) / len(agent_times) if agent_times else 0,
                "min": min(agent_times) if agent_times else 0,
                "max": max(agent_times) if agent_times else 0
            },
            "tool_execution": {
                "total": sum(tool_times) if tool_times else 0,
                "average": sum(tool_times) / len(tool_times) if tool_times else 0,
                "min": min(tool_times) if tool_times else 0,
                "max": max(tool_times) if tool_times else 0
            }
        }

    def _calculate_throughput_metrics(self) -> dict[str, Any]:
        """Calculate throughput metrics."""
        status = self.get_live_status()

        if status.elapsed_time == 0:
            return {"error": "No elapsed time"}

        return {
            "subtasks_per_second": status.completed_subtasks / status.elapsed_time if status.elapsed_time > 0 else 0,
            "events_per_second": len(self.events) / status.elapsed_time if status.elapsed_time > 0 else 0,
            "tool_calls_per_second": len([e for e in self.events if e.type == EventType.TOOL_CALL_START]) / status.elapsed_time if status.elapsed_time > 0 else 0
        }

    def _calculate_resource_usage(self) -> dict[str, Any]:
        """Calculate resource usage metrics."""
        # Count different resource types
        tool_calls = sum(1 for e in self.events if e.type == EventType.TOOL_CALL_START)
        agents_spawned = sum(1 for e in self.events if e.type == EventType.SUB_AGENT_SPAWN)

        # Get peak concurrent agents
        peak_agents = 0
        current_agents = 0
        for event in sorted(self.events, key=lambda e: e.timestamp):
            if event.type == EventType.SUB_AGENT_START:
                current_agents += 1
                peak_agents = max(peak_agents, current_agents)
            elif event.type in [EventType.SUB_AGENT_COMPLETE, EventType.SUB_AGENT_FAILED]:
                current_agents = max(0, current_agents - 1)

        return {
            "total_tool_calls": tool_calls,
            "total_agents_spawned": agents_spawned,
            "peak_concurrent_agents": peak_agents,
            "total_events_tracked": len(self.events)
        }

    def _calculate_efficiency_metrics(self) -> dict[str, Any]:
        """Calculate efficiency metrics."""
        status = self.get_live_status()

        if status.total_subtasks == 0:
            return {"error": "No subtasks"}

        success_rate = status.completed_subtasks / status.total_subtasks if status.total_subtasks > 0 else 0
        failure_rate = status.failed_subtasks / status.total_subtasks if status.total_subtasks > 0 else 0

        # Calculate parallel efficiency (how well we're using parallelism)
        agent_times = [
            node.get_duration() for node in self.nodes.values()
            if node.node_type == "agent" and node.get_duration()
        ]
        total_agent_time = sum(agent_times) if agent_times else 0
        wall_clock_time = status.elapsed_time if status.elapsed_time > 0 else 1

        parallel_efficiency = (total_agent_time / wall_clock_time) / len(agent_times) if agent_times and wall_clock_time > 0 else 0
        parallel_efficiency = min(1.0, parallel_efficiency)  # Cap at 100%

        return {
            "success_rate": round(success_rate, 3),
            "failure_rate": round(failure_rate, 3),
            "parallel_efficiency": round(parallel_efficiency, 3),
            "avg_time_per_subtask": round(status.elapsed_time / status.completed_subtasks, 2) if status.completed_subtasks > 0 else 0
        }

    def _identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Find slowest agents
        agent_durations = [
            (node.name, node.get_duration())
            for node in self.nodes.values()
            if node.node_type == "agent" and node.get_duration()
        ]

        if agent_durations:
            agent_durations.sort(key=lambda x: x[1], reverse=True)
            avg_duration = sum(d for _, d in agent_durations) / len(agent_durations)

            for name, duration in agent_durations[:3]:  # Top 3 slowest
                if duration > avg_duration * 1.5:
                    bottlenecks.append({
                        "type": "slow_agent",
                        "name": name,
                        "duration": round(duration, 2),
                        "severity": "high" if duration > avg_duration * 2 else "medium"
                    })

        # Find failed operations
        failed_events = [e for e in self.events if "failed" in e.type.value]
        if len(failed_events) > 2:
            bottlenecks.append({
                "type": "high_failure_rate",
                "count": len(failed_events),
                "severity": "high"
            })

        # Check for idle time (gaps in execution)
        status = self.get_live_status()
        if status.total_subtasks > 0 and len(status.active_agents) == 0 and status.pending_subtasks > 0:
            bottlenecks.append({
                "type": "idle_with_pending_work",
                "pending": status.pending_subtasks,
                "severity": "medium"
            })

        return bottlenecks

    def get_critical_path(self) -> list[dict[str, Any]]:
        """
        Identify the critical path in execution (longest dependency chain).

        Returns:
            List of nodes on critical path
        """
        if not self.root_node_id or self.root_node_id not in self.nodes:
            return []

        def calculate_path_length(node_id: str, memo: dict[str, float]) -> float:
            if node_id in memo:
                return memo[node_id]

            if node_id not in self.nodes:
                return 0

            node = self.nodes[node_id]
            if not node.children:
                duration = node.get_duration() or 0
                memo[node_id] = duration
                return duration

            max_child_path = max(
                calculate_path_length(child.node_id, memo)
                for child in node.children
            )

            total = (node.get_duration() or 0) + max_child_path
            memo[node_id] = total
            return total

        # Find critical path
        memo = {}
        def find_critical_path(node_id: str) -> list[dict[str, Any]]:
            if node_id not in self.nodes:
                return []

            node = self.nodes[node_id]
            path = [{
                "node_id": node.node_id,
                "name": node.name,
                "duration": node.get_duration(),
                "type": node.node_type
            }]

            if node.children:
                # Find child with longest path
                child_paths = [
                    (child, calculate_path_length(child.node_id, memo))
                    for child in node.children
                ]
                if child_paths:
                    critical_child = max(child_paths, key=lambda x: x[1])[0]
                    path.extend(find_critical_path(critical_child.node_id))

            return path

        return find_critical_path(self.root_node_id)
