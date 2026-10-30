"""
Short-term memory system for conversation management.

This module provides short-term memory for managing recent conversation history
with automatic summarization when approaching context limits, efficient retrieval,
and token counting.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    Represents a single message in conversation.

    Attributes:
        role: Role of the message sender
        content: Message content
        timestamp: When the message was created
        metadata: Additional metadata (tool calls, tokens, etc.)
        tokens: Estimated token count for the message
    """
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tokens": self.tokens
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            tokens=data.get("tokens")
        )

    def estimate_tokens(self) -> int:
        """
        Estimate token count for the message.
        Uses rough approximation: 1 token ≈ 4 characters.

        Returns:
            Estimated token count
        """
        if self.tokens is None:
            # Rough estimation: 1 token ≈ 4 characters
            self.tokens = len(self.content) // 4 + len(str(self.metadata)) // 4
        return self.tokens


@dataclass
class ConversationSummary:
    """
    Summary of conversation segment.

    Attributes:
        summary: Condensed summary text
        message_count: Number of messages summarized
        token_count: Total tokens in original messages
        timestamp: When summary was created
        metadata: Additional information about the summary
    """
    summary: str
    message_count: int
    token_count: int
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSummary":
        """Create summary from dictionary (tolerant of older/newer field sets)."""
        from ..core._compat import load_from_dict

        return load_from_dict(cls, data, label="ConversationSummary")


class ShortTermMemory:
    """
    Short-term memory manager for conversation history.

    Features:
    - Store recent conversation messages
    - Automatic summarization when approaching context limits
    - Efficient retrieval of recent context
    - Token counting and management
    - Message filtering and search
    - Rolling window of recent messages

    This class maintains a balance between keeping recent messages
    and older summaries to stay within context limits while preserving
    important conversation history.
    """

    def __init__(self,
                 max_tokens: int = 4096,
                 max_messages: int = 100,
                 summarization_threshold: float = 0.8,
                 summary_length_ratio: float = 0.3,
                 keep_recent_messages: int = 10,
                 summary_budget_ratio: float = 0.4,
                 model: Any = None,
                 compaction_strategy: Any = None,
                 tokenizer: Any = None) -> None:
        """
        Initialize short-term memory.

        Args:
            max_tokens: Maximum total tokens to keep in memory
            max_messages: Maximum number of messages to store
            summarization_threshold: Threshold (0-1) of max_tokens to trigger summarization
            summary_length_ratio: Target ratio of summary length to original (0-1)
            keep_recent_messages: Number of recent messages to always keep unsummarized
            summary_budget_ratio: Share (0-1) of ``max_tokens`` that retained
                summaries may occupy. Summarizing turns older messages into
                summaries, so without a bound on the summaries themselves the
                stored total grows for as long as the conversation runs.
            model: Optional model instance for accurate token counting
            compaction_strategy: How the history is shortened when it
                approaches the window — see
                :mod:`effgen.memory.compaction`. Accepts a strategy, a class or
                a name; None keeps the default of summarizing everything but
                the most recent few.
            tokenizer: Anything with ``count_tokens(text)`` or ``encode(text)``.
                Used in preference to the model's own counter and to the
                character estimate, so the threshold is measured in the units
                the context window is measured in.
        """
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.summarization_threshold = summarization_threshold
        self.summary_length_ratio = summary_length_ratio
        self.keep_recent_messages = keep_recent_messages
        self.summary_budget_ratio = summary_budget_ratio
        self._model = model
        self._tokenizer = tokenizer
        from .compaction import resolve_strategy
        self.compaction_strategy = resolve_strategy(compaction_strategy)

        # Storage
        self.messages: deque[Message] = deque(maxlen=max_messages)
        self.summaries: list[ConversationSummary] = []

        # Statistics
        self.total_messages_added = 0
        self.total_summarizations = 0
        self._current_token_count = 0

    def _count_tokens(self, text: str) -> int:
        """Count tokens with the supplied tokenizer, the model, else a heuristic.

        An explicit tokenizer wins: a caller who supplied one wants the count
        measured the way their model measures it, whatever model object the
        memory happens to hold.
        """
        if self._tokenizer is not None:
            try:
                if hasattr(self._tokenizer, "count_tokens"):
                    result = self._tokenizer.count_tokens(text)
                    return result.count if hasattr(result, "count") else int(result)
                if hasattr(self._tokenizer, "encode"):
                    return len(self._tokenizer.encode(text))
                if callable(self._tokenizer):
                    return int(self._tokenizer(text))
            except Exception:  # best-effort: fall through to the model/heuristic
                logger.debug("Supplied tokenizer could not count; falling back",
                             exc_info=True)
        if self._model is not None:
            try:
                result = self._model.count_tokens(text)
                # TokenCount dataclass has a .count attribute
                return result.count if hasattr(result, 'count') else int(result)
            except Exception:  # best-effort: fall back to the word-count estimate below
                pass
        return len(text) // 4

    def add_message(self,
                   role: MessageRole,
                   content: str,
                   metadata: dict[str, Any] | None = None,
                   tokens: int | None = None) -> Message:
        """
        Add a message to short-term memory.

        Args:
            role: Role of the message sender
            content: Message content
            metadata: Optional metadata
            tokens: Pre-computed token count (will estimate if not provided)

        Returns:
            The created Message object
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tokens=tokens
        )

        # Estimate tokens if not provided
        message_tokens = message.estimate_tokens()

        # Add message
        self.messages.append(message)
        self.total_messages_added += 1
        self._current_token_count += message_tokens

        # Check if we need to summarize
        if self._should_summarize():
            self._summarize_old_messages()

        return message

    def add_user_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience method to add user message."""
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience method to add assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    def add_system_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience method to add system message."""
        return self.add_message(MessageRole.SYSTEM, content, **kwargs)

    def add_tool_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience method to add tool message."""
        return self.add_message(MessageRole.TOOL, content, **kwargs)

    def get_recent_messages(self, n: int | None = None) -> list[Message]:
        """
        Get the n most recent messages.

        Args:
            n: Number of recent messages to retrieve (None = all)

        Returns:
            List of recent messages
        """
        if n is None:
            return list(self.messages)
        return list(self.messages)[-n:]

    def get_messages(self, n: int | None = None) -> list[Message]:
        """
        Get stored messages (the plain counterpart to :meth:`add_message`).

        Args:
            n: Number of most-recent messages to return (None = all).

        Returns:
            List of messages, oldest first.
        """
        return self.get_recent_messages(n)

    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """
        Get all messages by a specific role.

        Args:
            role: Message role to filter by

        Returns:
            List of messages with matching role
        """
        return [msg for msg in self.messages if msg.role == role]

    def get_conversation_context(self,
                                max_tokens: int | None = None) -> list[dict[str, Any]]:
        """
        Get conversation context suitable for LLM input.

        Args:
            max_tokens: Maximum tokens to include (uses instance max if None)

        Returns:
            List of message dictionaries in format expected by LLMs
        """
        max_tokens = max_tokens or self.max_tokens
        context = []
        current_tokens = 0

        # Add summaries first if they exist
        for summary in self.summaries:
            summary_tokens = self._count_tokens(summary.summary)
            if current_tokens + summary_tokens > max_tokens:
                break
            context.append({
                "role": "system",
                "content": f"[Summary of previous conversation: {summary.summary}]"
            })
            current_tokens += summary_tokens

        # Add recent messages
        for message in self.messages:
            msg_tokens = message.estimate_tokens()
            if current_tokens + msg_tokens > max_tokens:
                break
            context.append({
                "role": message.role.value,
                "content": message.content
            })
            current_tokens += msg_tokens

        return context

    def search_messages(self, query: str, case_sensitive: bool = False) -> list[Message]:
        """
        Search messages by content.

        Args:
            query: Search query string
            case_sensitive: Whether search should be case-sensitive

        Returns:
            List of messages matching the query
        """
        results = []
        query_str = query if case_sensitive else query.lower()

        for message in self.messages:
            content = message.content if case_sensitive else message.content.lower()
            if query_str in content:
                results.append(message)

        return results

    def get_token_count(self) -> int:
        """
        Get current total token count (cached).

        Returns the incrementally maintained count. Use
        ``_validate_token_count()`` for a full recalculation.

        Returns:
            Total tokens currently stored
        """
        return self._current_token_count

    def _validate_token_count(self) -> int:
        """
        Full recalculation of token count — call periodically or on demand.

        Updates ``_current_token_count`` and returns the authoritative value.
        """
        total = sum(msg.estimate_tokens() for msg in self.messages)
        total += sum(self._count_tokens(s.summary) for s in self.summaries)
        self._current_token_count = total
        return total

    def _should_summarize(self) -> bool:
        """
        Check if summarization should be triggered.

        Returns:
            True if summarization is needed
        """
        # The configured strategy decides; the default reproduces the
        # long-standing behaviour of summarizing past a fraction of the window.
        return self.compaction_strategy.should_compact(self)

    def _summarize_old_messages(self) -> None:
        """
        Summarize old messages to save space.

        This method:
        1. Identifies messages to summarize (excluding recent ones)
        2. Generates a summary
        3. Removes original messages
        4. Stores the summary
        """
        # Which messages leave is the strategy's decision; the default takes
        # everything but the most recent few, oldest first.
        messages_to_summarize = self.compaction_strategy.messages_to_compact(self)
        if not messages_to_summarize:
            return

        # What replaces them is also the strategy's: summary text, or None to
        # drop them outright with no model call and no tokens spent.
        summary_text = self.compaction_strategy.summarize(self, messages_to_summarize)

        original_tokens = sum(msg.estimate_tokens() for msg in messages_to_summarize)
        if summary_text:
            summary_tokens = self._count_tokens(summary_text)
            self.summaries.append(ConversationSummary(
                summary=summary_text,
                message_count=len(messages_to_summarize),
                token_count=original_tokens,
                metadata={
                    "summary_tokens": summary_tokens,
                    "compression_ratio": (
                        summary_tokens / original_tokens if original_tokens > 0 else 0
                    ),
                    "strategy": type(self.compaction_strategy).__name__,
                },
            ))
            self._compact_summaries()

        # Remove exactly the messages that were compacted. A strategy may take
        # them from the middle, so they are removed by identity rather than by
        # popping a count off the front.
        compacted = {id(msg) for msg in messages_to_summarize}
        remaining = [msg for msg in self.messages if id(msg) not in compacted]
        self.messages.clear()
        self.messages.extend(remaining)

        # Update statistics
        self.total_summarizations += 1
        self._validate_token_count()

    def _summary_budget(self) -> int:
        """Tokens the retained summaries are allowed to occupy."""
        return max(1, int(self.max_tokens * self.summary_budget_ratio))

    def summary_token_count(self) -> int:
        """Tokens currently held by the retained summaries."""
        return sum(self._count_tokens(s.summary) for s in self.summaries)

    def _truncate_to_tokens(self, text: str, limit: int) -> str:
        """Shorten *text* until it counts at most *limit* tokens.

        Slicing on a fixed characters-per-token estimate holds for Latin prose
        and undershoots for scripts that tokenize closer to one token per
        character, so the cut is measured, applied and re-measured instead of
        taken once.
        """
        if limit <= 0:
            return ""
        text = text.rstrip()
        for _ in range(24):
            counted = self._count_tokens(text)
            if counted <= limit:
                return text
            keep = min(int(len(text) * limit / counted), len(text) - 1)
            if keep <= 0:
                return ""
            text = text[:keep].rstrip()
        return ""

    def _compact_summaries(self) -> None:
        """Fold the oldest summaries together until they fit the budget.

        Every summarization adds a summary, and each one is replayed into the
        prompt, so an unbounded list makes the prompt grow for as long as the
        session runs — until it passes the model's context window and every
        further call is refused. Merging the two oldest entries keeps the
        oldest context present in condensed form rather than dropping it, and
        the merged entry records how many summaries it now stands for.
        """
        budget = self._summary_budget()
        while len(self.summaries) > 1 and self.summary_token_count() > budget:
            oldest, second = self.summaries[0], self.summaries[1]
            merged_text = f"{oldest.summary} {second.summary}".strip()
            # Keep the merged entry roughly the size of a single summary, so
            # repeated merges converge instead of growing one long entry.
            merged_text = self._truncate_to_tokens(merged_text, max(1, budget // 2))
            folded = (
                oldest.metadata.get("folded_summaries", 1)
                + second.metadata.get("folded_summaries", 1)
            )
            self.summaries[:2] = [
                ConversationSummary(
                    summary=merged_text,
                    message_count=oldest.message_count + second.message_count,
                    token_count=oldest.token_count + second.token_count,
                    timestamp=oldest.timestamp,
                    metadata={**oldest.metadata, "folded_summaries": folded},
                )
            ]
        # A single summary that alone exceeds the budget is truncated; nothing
        # is left to merge it with.
        if self.summaries and self.summary_token_count() > budget:
            only = self.summaries[0]
            only.summary = self._truncate_to_tokens(only.summary, budget)

    def _generate_summary(self, messages: list[Message]) -> str:
        """
        Generate a structured summary of messages.

        Preserves facts (numbers, names, tool results) with higher priority
        than filler text. Uses a structured format so the agent can recover
        key information from the summary.

        Args:
            messages: Messages to summarize

        Returns:
            Summary text with Facts / Decisions / Pending sections
        """
        import re as _re

        facts: list[str] = []
        decisions: list[str] = []
        pending: list[str] = []

        for msg in messages:
            content = msg.content.strip()
            if not content:
                continue

            # Tool messages are high-value facts — keep them almost verbatim
            if msg.role == MessageRole.TOOL:
                # Truncate very long tool outputs but keep the core result
                snippet = content[:200] + ("..." if len(content) > 200 else "")
                facts.append(f"Tool result: {snippet}")
                continue

            # Extract sentences containing numbers or names (high-priority facts)
            sentences = _re.split(r'(?<=[.!?])\s+', content)
            has_number = False
            for sentence in sentences:
                if _re.search(r'\d', sentence):
                    facts.append(sentence.strip()[:150])
                    has_number = True

            # If no numeric sentences, take first sentence as a fact
            if not has_number and sentences:
                first = sentences[0].strip()
                if len(first) > 150:
                    first = first[:150] + "..."
                if msg.role == MessageRole.USER:
                    pending.append(f"User asked: {first}")
                elif msg.role == MessageRole.ASSISTANT:
                    decisions.append(first)

        # Build structured summary
        parts = []
        if facts:
            parts.append("Facts: " + " | ".join(dict.fromkeys(facts)))  # dedupe
        if decisions:
            parts.append("Decisions: " + " | ".join(dict.fromkeys(decisions)))
        if pending:
            parts.append("Pending: " + " | ".join(dict.fromkeys(pending)))

        summary = "\n".join(parts) if parts else "No significant content."

        # Respect target length
        target_length = int(sum(len(m.content) for m in messages) * self.summary_length_ratio)
        if target_length > 0 and len(summary) > target_length:
            summary = summary[:target_length] + "..."

        return summary

    def clear(self) -> None:
        """Clear all messages and summaries."""
        self.messages.clear()
        self.summaries.clear()
        self._current_token_count = 0

    def get_statistics(self) -> dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with various statistics
        """
        return {
            "current_messages": len(self.messages),
            "current_tokens": self.get_token_count(),
            "max_tokens": self.max_tokens,
            "utilization": self.get_token_count() / self.max_tokens,
            "total_messages_added": self.total_messages_added,
            "total_summarizations": self.total_summarizations,
            "summaries_count": len(self.summaries),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize memory to dictionary.

        Returns:
            Dictionary representation of memory state
        """
        return {
            "config": {
                "max_tokens": self.max_tokens,
                "max_messages": self.max_messages,
                "summarization_threshold": self.summarization_threshold,
                "summary_length_ratio": self.summary_length_ratio,
                "keep_recent_messages": self.keep_recent_messages,
                "summary_budget_ratio": self.summary_budget_ratio,
            },
            "messages": [msg.to_dict() for msg in self.messages],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "statistics": self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortTermMemory":
        """
        Deserialize memory from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ShortTermMemory instance
        """
        from ..core._compat import coerce_kwargs

        # Drop any config keys this build no longer accepts (older/newer snapshot)
        # instead of raising; one quiet warning is emitted on drift.
        config = coerce_kwargs(cls, data.get("config", {}), label="ShortTermMemory")
        memory = cls(**config)

        # Restore messages
        for msg_data in data.get("messages", []):
            message = Message.from_dict(msg_data)
            memory.messages.append(message)

        # Restore summaries
        for summary_data in data.get("summaries", []):
            summary = ConversationSummary.from_dict(summary_data)
            memory.summaries.append(summary)

        # Restore statistics
        stats = data.get("statistics", {})
        memory.total_messages_added = stats.get("total_messages_added", 0)
        memory.total_summarizations = stats.get("total_summarizations", 0)
        memory._validate_token_count()

        return memory

    def save_to_file(self, filepath: str) -> None:
        """
        Save memory to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ShortTermMemory":
        """
        Load memory from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            ShortTermMemory instance
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)
