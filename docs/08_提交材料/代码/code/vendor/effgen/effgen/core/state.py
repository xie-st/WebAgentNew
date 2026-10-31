"""
Agent state management for effGen framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _unsupported_format_msg(format: str) -> str:
    """Build a clear error for an unsupported state serialization format."""
    if format == "pickle":
        return (
            "The 'pickle' state format has been removed: unpickling a state "
            "file can execute arbitrary code (RCE on an untrusted file). Use "
            "the default JSON format (format='json')."
        )
    return f"Unsupported format: {format!r}. Only 'json' is supported."


@dataclass
class AgentState:
    """
    Represents the complete state of an agent.

    This includes conversation history, tool usage, memory, and configuration.
    Can be saved and loaded for persistence.
    """

    agent_id: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to conversation history.

        Args:
            role: Who the message is from.
            content: The message text.
            metadata: Extra context stored with the message.
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.updated_at = datetime.now()

    def add_tool_call(self, tool_name: str, args: dict, result: Any, error: str | None = None) -> None:
        """Record a tool call.

        Args:
            tool_name: The tool that was called.
            args: The arguments it was called with.
            result: What it returned.
            error: The failure message, when the call failed.
        """
        self.tool_history.append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()

    def save(self, filepath: str, format: str = "json") -> None:
        """Save state to file."""
        self.updated_at = datetime.now()

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            raise ValueError(_unsupported_format_msg(format))

    @classmethod
    def load(cls, filepath: str, format: str = "json") -> "AgentState":
        """Load state from file.

        Raises:
            CorruptStateError: The file is not parseable JSON, or holds something
                other than a JSON object of state fields.
        """
        if format == "json":
            from ..errors import CorruptStateError

            with open(filepath) as f:
                try:
                    data = json.load(f)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise CorruptStateError("state", filepath, str(exc)) from exc
            if not isinstance(data, dict):
                raise CorruptStateError(
                    "state",
                    filepath,
                    f"expected a JSON object of fields, got {type(data).__name__}",
                )
            # Convert ISO strings back to datetime. A value that is not a
            # readable timestamp is dropped so the field takes its default,
            # rather than reaching a caller that will subtract dates with it.
            for key in ("created_at", "updated_at"):
                if key not in data or isinstance(data[key], datetime):
                    continue
                try:
                    data[key] = datetime.fromisoformat(data[key])
                except (TypeError, ValueError):
                    data.pop(key)
            # Load forgivingly: state files saved by a different effGen build may carry
            # since-removed fields; drop them (with one warning) rather than crashing.
            from ._compat import load_from_dict

            return load_from_dict(cls, data, label="AgentState")
        else:
            raise ValueError(_unsupported_format_msg(format))

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)

    def clear_history(self) -> None:
        """Clear conversation and tool history."""
        self.conversation_history = []
        self.tool_history = []
        self.updated_at = datetime.now()

    def get_recent_messages(self, n: int = 10) -> list[dict[str, Any]]:
        """Get n most recent messages."""
        return self.conversation_history[-n:]

    def get_token_count_estimate(self) -> int:
        """Estimate total tokens in conversation history."""
        # Rough estimate: 4 characters per token
        total_chars = sum(len(str(msg.get("content", ""))) for msg in self.conversation_history)
        return total_chars // 4
