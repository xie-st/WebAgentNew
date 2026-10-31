"""
Feedback collection system for effGen agents.

Collects user feedback on agent responses (thumbs up/down, ratings, comments)
and exports it as JSONL for analysis and fine-tuning.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    THUMBS = "thumbs"       # up / down
    RATING = "rating"       # 1-5 scale
    TEXT = "text"           # free-text comment


@dataclass
class FeedbackEntry:
    """
    A single piece of user feedback.

    Attributes:
        feedback_id: Unique identifier for this feedback.
        response_id: ID of the agent response being rated.
        feedback_type: Type of feedback.
        value: The feedback value (bool for thumbs, int for rating, str for text).
        timestamp: Unix timestamp when feedback was given.
        agent_name: Name of the agent that produced the response.
        query: The original query (optional).
        metadata: Additional context.
    """
    feedback_id: str
    response_id: str
    feedback_type: FeedbackType
    value: bool | int | str
    timestamp: float
    agent_name: str = ""
    query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        d = {
            "feedback_id": self.feedback_id,
            "response_id": self.response_id,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "query": self.query,
            "metadata": self.metadata,
        }
        return d


class FeedbackCollector:
    """
    Collects and stores user feedback on agent responses.

    Feedback is stored in memory and can be exported as JSONL
    for analysis or fine-tuning datasets.
    """

    def __init__(self, agent_name: str = "") -> None:
        self.agent_name = agent_name
        self._entries: list[FeedbackEntry] = []

    def thumbs(
        self,
        response_id: str,
        thumbs_up: bool,
        query: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Record a thumbs up/down feedback.

        Args:
            response_id: The response being rated.
            thumbs_up: ``True`` for positive feedback, ``False`` for negative.
            query: The query that produced the response.
            metadata: Extra context to store with the entry.

        Returns:
            The stored entry.
        """
        entry = FeedbackEntry(
            feedback_id=uuid.uuid4().hex[:12],
            response_id=response_id,
            feedback_type=FeedbackType.THUMBS,
            value=thumbs_up,
            timestamp=time.time(),
            agent_name=self.agent_name,
            query=query,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.debug("Feedback recorded: %s for response %s", "thumbs_up" if thumbs_up else "thumbs_down", response_id)
        return entry

    def rate(
        self,
        response_id: str,
        rating: int,
        query: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Record a 1-5 rating.

        Args:
            response_id: ID of the response being rated.
            rating: Integer rating from 1 to 5.
            query: Original query (optional).
            metadata: Additional context.

        Returns:
            The stored entry.

        Raises:
            ValueError: If rating is not between 1 and 5.
        """
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be 1-5, got {rating}")
        entry = FeedbackEntry(
            feedback_id=uuid.uuid4().hex[:12],
            response_id=response_id,
            feedback_type=FeedbackType.RATING,
            value=rating,
            timestamp=time.time(),
            agent_name=self.agent_name,
            query=query,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.debug("Rating %d recorded for response %s", rating, response_id)
        return entry

    def comment(
        self,
        response_id: str,
        text: str,
        query: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Record a free-text comment.

        Args:
            response_id: The response being commented on.
            text: The comment.
            query: The query that produced the response.
            metadata: Extra context to store with the entry.

        Returns:
            The stored entry.
        """
        entry = FeedbackEntry(
            feedback_id=uuid.uuid4().hex[:12],
            response_id=response_id,
            feedback_type=FeedbackType.TEXT,
            value=text,
            timestamp=time.time(),
            agent_name=self.agent_name,
            query=query,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        logger.debug("Comment recorded for response %s", response_id)
        return entry

    @property
    def entries(self) -> list[FeedbackEntry]:
        """Get all feedback entries."""
        return list(self._entries)

    def get_by_response(self, response_id: str) -> list[FeedbackEntry]:
        """Get all feedback entries for a specific response."""
        return [e for e in self._entries if e.response_id == response_id]

    def summary(self) -> dict[str, Any]:
        """Get a summary of collected feedback."""
        thumbs = [e for e in self._entries if e.feedback_type == FeedbackType.THUMBS]
        ratings = [e for e in self._entries if e.feedback_type == FeedbackType.RATING]
        comments = [e for e in self._entries if e.feedback_type == FeedbackType.TEXT]

        thumbs_up = sum(1 for e in thumbs if e.value is True)
        avg_rating = (sum(e.value for e in ratings) / len(ratings)) if ratings else 0.0

        return {
            "total": len(self._entries),
            "thumbs_up": thumbs_up,
            "thumbs_down": len(thumbs) - thumbs_up,
            "average_rating": round(avg_rating, 2),
            "total_ratings": len(ratings),
            "total_comments": len(comments),
        }

    def export_jsonl(self, path: str | Path) -> int:
        """
        Export all feedback entries as JSONL.

        Args:
            path: File path to write JSONL output.

        Returns:
            Number of entries exported.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info("Exported %d feedback entries to %s", len(self._entries), path)
        return len(self._entries)

    def clear(self) -> None:
        """Clear all stored feedback."""
        self._entries.clear()
