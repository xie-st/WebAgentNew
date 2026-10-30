"""
Task abstraction for effGen framework.

This module provides task representation and management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SubTask:
    """Represents a subtask in task decomposition."""
    id: str
    description: str
    expected_output: str
    depends_on: list[str] = field(default_factory=list)
    estimated_complexity: float = 0.0  # 0-10 scale
    required_specialization: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate subtask after initialization."""
        if not self.id:
            self.id = f"st_{uuid.uuid4().hex[:8]}"
        if self.estimated_complexity < 0 or self.estimated_complexity > 10:
            raise ValueError("Complexity must be between 0 and 10")


@dataclass
class Task:
    """
    Represents a task to be executed by an agent.

    A task encapsulates the goal, context, constraints, and execution state.
    """

    description: str
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Results
    result: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Sub-task decomposition
    subtasks: list[SubTask] = field(default_factory=list)
    is_decomposed: bool = False

    # Agent assignment
    assigned_agent: str | None = None
    parent_task_id: str | None = None

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, result: Any) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def fail(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def cancel(self) -> None:
        """Cancel the task."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()

    def get_duration(self) -> float | None:
        """Get task execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def is_finished(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]

    def add_subtask(self, subtask: SubTask) -> None:
        """Add a subtask to this task."""
        self.subtasks.append(subtask)
        self.is_decomposed = True

    def get_pending_subtasks(self) -> list[SubTask]:
        """Get all pending subtasks."""
        return [st for st in self.subtasks if st.status == TaskStatus.PENDING]

    def get_completed_subtasks(self) -> list[SubTask]:
        """Get all completed subtasks."""
        return [st for st in self.subtasks if st.status == TaskStatus.COMPLETED]

    def get_failed_subtasks(self) -> list[SubTask]:
        """Get all failed subtasks."""
        return [st for st in self.subtasks if st.status == TaskStatus.FAILED]

    def get_subtask_progress(self) -> float:
        """Get subtask completion progress (0.0 to 1.0)."""
        if not self.subtasks:
            return 0.0
        completed = len(self.get_completed_subtasks())
        return completed / len(self.subtasks)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.name,
            "context": self.context,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "subtasks": [
                {
                    "id": st.id,
                    "description": st.description,
                    "status": st.status.name,
                    "result": st.result,
                    "error": st.error
                } for st in self.subtasks
            ],
            "is_decomposed": self.is_decomposed,
            "assigned_agent": self.assigned_agent,
            "parent_task_id": self.parent_task_id
        }

    def __repr__(self) -> str:
        """String representation of task."""
        duration_str = f", duration={self.get_duration():.2f}s" if self.get_duration() else ""
        return f"Task(id={self.id}, status={self.status.name}, priority={self.priority.name}{duration_str})"
