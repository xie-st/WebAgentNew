"""Rendering and export for the execution tracker.

Formats a tracked run for a reader — rich, markdown, plain or JSON — and writes
it to a file as JSON, CSV or a self-contained HTML page.

The class is a mixin: it reads the attributes
:class:`~effgen.core.execution_tracker.ExecutionTracker` creates in its
``__init__``, and is composed into that class.
"""

from __future__ import annotations

import json
from datetime import datetime

from effgen.core.execution_tracker_events import EventType


class ExecutionTrackerRenderMixin:
    """Display formatting, the event icon table and trace export."""

    def format_for_display(self, format: str = "rich") -> str:
        """
        Format execution trace for user display.

        Args:
            format: Output format (rich, markdown, plain)

        Returns:
            Formatted string
        """
        if format == "rich":
            return self._format_rich()
        elif format == "markdown":
            return self._format_markdown()
        elif format == "plain":
            return self._format_plain()
        elif format == "json":
            return json.dumps({
                "events": self.get_trace(),
                "tree": self.generate_execution_tree(),
                "status": self.get_live_status().to_dict()
            }, indent=2)
        else:
            return self._format_plain()

    def _format_rich(self) -> str:
        """Format with rich terminal output."""
        lines = []
        lines.append("=" * 60)
        lines.append("EXECUTION TRACE")
        lines.append("=" * 60)

        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            icon = self._get_event_icon(event.type)
            lines.append(f"{timestamp} {icon} [{event.type.value}] {event.message}")
            if event.agent_id:
                lines.append(f"  Agent: {event.agent_id}")

        lines.append("=" * 60)
        status = self.get_live_status()
        lines.append(f"Status: {status.progress_percentage:.1f}% complete")
        lines.append(f"Elapsed: {status.elapsed_time:.2f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_markdown(self) -> str:
        """Format as markdown."""
        lines = []
        lines.append("# Execution Trace\n")

        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            lines.append(f"**{timestamp}** - {event.type.value}")
            lines.append(f"> {event.message}\n")

        status = self.get_live_status()
        lines.append("\n## Summary\n")
        lines.append(f"- Progress: {status.progress_percentage:.1f}%")
        lines.append(f"- Completed: {status.completed_subtasks}/{status.total_subtasks}")
        lines.append(f"- Elapsed: {status.elapsed_time:.2f}s")

        return "\n".join(lines)

    def _format_plain(self) -> str:
        """Format as plain text."""
        lines = []
        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{timestamp}] {event.type.value}: {event.message}")
        return "\n".join(lines)

    def _get_event_icon(self, event_type: EventType) -> str:
        """Get icon for event type."""
        icons = {
            EventType.TASK_START: "🎯",
            EventType.TASK_COMPLETE: "✅",
            EventType.TASK_FAILED: "❌",
            EventType.ROUTING_DECISION: "🧠",
            EventType.TASK_DECOMPOSITION: "🔀",
            EventType.SUB_AGENT_SPAWN: "🚀",
            EventType.SUB_AGENT_START: "🔍",
            EventType.SUB_AGENT_COMPLETE: "✅",
            EventType.SUB_AGENT_FAILED: "❌",
            EventType.TOOL_CALL_START: "🔧",
            EventType.TOOL_CALL_COMPLETE: "✓",
            EventType.TOOL_CALL_FAILED: "✗",
            EventType.RESULT_SYNTHESIS: "✨",
            EventType.REASONING_STEP: "💭",
            EventType.ERROR: "⚠️",
            EventType.WARNING: "⚠️",
            EventType.INFO: "ℹ️"
        }
        return icons.get(event_type, "•")

    def export_trace(self, filepath: str, format: str = "json") -> None:
        """
        Export execution trace to file.

        Args:
            filepath: Path to save trace
            format: Export format (json, csv, html)
        """
        import csv
        import json

        if format == "json":
            data = {
                "trace": self.get_trace(),
                "tree": self.generate_execution_tree(),
                "status": self.get_live_status().to_dict(),
                "summary": self.get_summary(),
                "metrics": self.get_performance_metrics()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            with open(filepath, 'w', newline='') as f:
                fieldnames = ["timestamp", "type", "agent_id", "message"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for event in self.events:
                    writer.writerow({
                        "timestamp": event.timestamp,
                        "type": event.type.value,
                        "agent_id": event.agent_id or "",
                        "message": event.message
                    })

        elif format == "html":
            html = self._generate_html_trace()
            with open(filepath, 'w') as f:
                f.write(html)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_trace(self) -> str:
        """Generate HTML visualization of trace."""
        from datetime import datetime

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>Execution Trace</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".event { padding: 10px; margin: 5px 0; border-left: 3px solid #ddd; }",
            ".task_start { border-color: #4CAF50; }",
            ".task_complete { border-color: #2196F3; }",
            ".task_failed { border-color: #f44336; }",
            ".sub_agent_start { border-color: #FF9800; }",
            ".tool_call { border-color: #9C27B0; }",
            ".timestamp { color: #666; font-size: 0.9em; }",
            ".summary { background: #f5f5f5; padding: 15px; margin: 20px 0; }",
            "</style>",
            "</head><body>",
            "<h1>Execution Trace</h1>"
        ]

        # Add summary
        status = self.get_live_status()
        html_parts.append('<div class="summary">')
        html_parts.append("<h2>Summary</h2>")
        html_parts.append(f"<p>Total Events: {len(self.events)}</p>")
        html_parts.append(f"<p>Completed: {status.completed_subtasks}/{status.total_subtasks}</p>")
        html_parts.append(f"<p>Elapsed Time: {status.elapsed_time:.2f}s</p>")
        html_parts.append(f"<p>Progress: {status.progress_percentage:.1f}%</p>")
        html_parts.append('</div>')

        # Add events
        html_parts.append("<h2>Events</h2>")
        for event in self.events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            event.type.value.replace("_", " ")
            html_parts.append(f'<div class="event {event.type.value}">')
            html_parts.append(f'<span class="timestamp">{timestamp}</span> ')
            html_parts.append(f'<strong>{event.type.value}</strong>: {event.message}')
            if event.agent_id:
                html_parts.append(f' <em>(Agent: {event.agent_id})</em>')
            html_parts.append('</div>')

        html_parts.append("</body></html>")

        return "\n".join(html_parts)
