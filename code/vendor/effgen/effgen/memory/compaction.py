"""How a conversation is shortened when it approaches the context window.

Small models have small windows, so what gets dropped and what survives changes
the answer more for them than for a frontier model. Different tasks want
different answers to that: a long support chat wants a summary, a tool-heavy
run wants the tool results kept and the reasoning dropped, a document workflow
wants the first turn and the last few verbatim.

A strategy answers three questions, and effGen calls them in this order:

1. :meth:`~CompactionStrategy.should_compact` — is it time?
2. :meth:`~CompactionStrategy.messages_to_compact` — which messages leave the
   live window?
3. :meth:`~CompactionStrategy.summarize` — what replaces them, if anything?

:class:`SummarizeOldest` is the default and is what effGen has always done:
summarize everything but the most recent few once the history passes a fraction
of the window. Pass another on the agent, or for one call::

    from effgen.memory.compaction import KeepFirstAndLast

    agent = Agent(AgentConfig(
        model="gpt-5-nano",
        memory_config={"compaction_strategy": KeepFirstAndLast(first=2, last=6)},
    ))

Token counts come from the model's own tokenizer when there is one, so the
threshold is measured in the units the window is measured in rather than in
characters divided by four.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .short_term import Message, ShortTermMemory

logger = logging.getLogger(__name__)

__all__ = [
    "CompactionStrategy",
    "SummarizeOldest",
    "DropOldest",
    "KeepFirstAndLast",
    "KeepToolResults",
    "resolve_strategy",
]


class CompactionStrategy:
    """Base class for a compaction policy. Subclass and override what differs.

    The defaults here are :class:`SummarizeOldest`'s, so a subclass that only
    changes which messages leave keeps the rest of the behaviour.
    """

    #: Shown in statistics and logs. Defaults to the class name.
    name: str = ""

    def __repr__(self) -> str:
        return f"<{self.name or type(self).__name__}>"

    def should_compact(self, memory: "ShortTermMemory") -> bool:
        """Whether the history is long enough to shorten now.

        Args:
            memory: The conversation, already carrying its current token count.

        Returns:
            True to compact before the next message is added.
        """
        threshold = memory.max_tokens * memory.summarization_threshold
        return (
            memory._current_token_count > threshold
            and len(memory.messages) > memory.keep_recent_messages
        )

    def messages_to_compact(self, memory: "ShortTermMemory") -> "list[Message]":
        """Which messages leave the live window, oldest first.

        Returning an empty list cancels this round of compaction.
        """
        keep = memory.keep_recent_messages
        if len(memory.messages) <= keep:
            return []
        return list(memory.messages)[: len(memory.messages) - keep]

    def summarize(
        self, memory: "ShortTermMemory", messages: "list[Message]"
    ) -> str | None:
        """What replaces the messages that left.

        Returns:
            Summary text to keep in their place, or None to drop them with no
            replacement — which costs no model call and no tokens, at the price
            of the conversation forgetting them outright.
        """
        return memory._generate_summary(messages)


class SummarizeOldest(CompactionStrategy):
    """Summarize everything but the most recent few. The default.

    Args:
        threshold: Fraction of the window that triggers compaction. None uses
            the memory's own ``summarization_threshold``.
        keep_recent: How many recent messages stay verbatim. None uses the
            memory's own ``keep_recent_messages``.
    """

    def __init__(self, threshold: float | None = None, keep_recent: int | None = None) -> None:
        self.threshold = threshold
        self.keep_recent = keep_recent

    def should_compact(self, memory: "ShortTermMemory") -> bool:
        """Compact once the history passes the configured fraction of the window."""
        fraction = (
            self.threshold if self.threshold is not None
            else memory.summarization_threshold
        )
        keep = self.keep_recent if self.keep_recent is not None else memory.keep_recent_messages
        return (
            memory._current_token_count > memory.max_tokens * fraction
            and len(memory.messages) > keep
        )

    def messages_to_compact(self, memory: "ShortTermMemory") -> "list[Message]":
        """Everything older than the recent messages this strategy keeps."""
        keep = self.keep_recent if self.keep_recent is not None else memory.keep_recent_messages
        if len(memory.messages) <= keep:
            return []
        return list(memory.messages)[: len(memory.messages) - keep]


class DropOldest(CompactionStrategy):
    """Discard the oldest messages without summarizing them.

    No model call, so nothing to wait for and nothing to pay for, and no risk
    of a summary inventing something. The conversation forgets them. Useful
    when older turns genuinely do not matter, and when a summarization call
    would cost more than the history is worth.

    Args:
        keep_recent: How many recent messages stay. None uses the memory's own
            ``keep_recent_messages``.
    """

    def __init__(self, keep_recent: int | None = None) -> None:
        self.keep_recent = keep_recent

    def messages_to_compact(self, memory: "ShortTermMemory") -> "list[Message]":
        """Everything older than the recent messages this strategy keeps."""
        keep = self.keep_recent if self.keep_recent is not None else memory.keep_recent_messages
        if len(memory.messages) <= keep:
            return []
        return list(memory.messages)[: len(memory.messages) - keep]

    def summarize(
        self, memory: "ShortTermMemory", messages: "list[Message]"
    ) -> str | None:
        """Nothing replaces them."""
        return None


class KeepFirstAndLast(CompactionStrategy):
    """Keep the opening turns and the recent ones; compact the middle.

    The first turns usually carry the task — the document, the instruction, the
    constraint everything else refers to — and a summary of them is a poor
    substitute. The middle is where a long conversation's redundancy is.

    Args:
        first: How many opening messages stay verbatim.
        last: How many recent messages stay verbatim.
        summarize_middle: True to replace the middle with a summary, False to
            drop it.
    """

    def __init__(self, first: int = 2, last: int = 6, summarize_middle: bool = True) -> None:
        self.first = max(0, first)
        self.last = max(0, last)
        self.summarize_middle = summarize_middle

    def messages_to_compact(self, memory: "ShortTermMemory") -> "list[Message]":
        """The messages between the opening and the recent ones."""
        messages = list(memory.messages)
        if len(messages) <= self.first + self.last:
            return []
        return messages[self.first : len(messages) - self.last]

    def summarize(
        self, memory: "ShortTermMemory", messages: "list[Message]"
    ) -> str | None:
        """A summary of the middle, or nothing when the caller asked for none."""
        if not self.summarize_middle:
            return None
        return memory._generate_summary(messages)


class KeepToolResults(CompactionStrategy):
    """Drop the reasoning, keep what the tools returned.

    In a tool-heavy run the reasoning is the bulk of the tokens and the tool
    results are the evidence the answer rests on. Compacting the reasoning and
    keeping the results holds on to the facts while freeing most of the window.

    Args:
        keep_recent: How many recent messages stay verbatim whatever their role.
        summarize_dropped: True to summarize what leaves, False to drop it.
    """

    def __init__(self, keep_recent: int | None = None, summarize_dropped: bool = True) -> None:
        self.keep_recent = keep_recent
        self.summarize_dropped = summarize_dropped

    def messages_to_compact(self, memory: "ShortTermMemory") -> "list[Message]":
        """Older messages that are not a tool result."""
        keep = self.keep_recent if self.keep_recent is not None else memory.keep_recent_messages
        messages = list(memory.messages)
        if len(messages) <= keep:
            return []
        older = messages[: len(messages) - keep]
        return [m for m in older if str(getattr(m.role, "value", m.role)) != "tool"]

    def summarize(
        self, memory: "ShortTermMemory", messages: "list[Message]"
    ) -> str | None:
        """A summary of the reasoning that left, or nothing."""
        if not self.summarize_dropped:
            return None
        return memory._generate_summary(messages)


#: Strategies addressable by name, for a config file or a CLI flag.
STRATEGIES_BY_NAME: dict[str, type[CompactionStrategy]] = {
    "summarize_oldest": SummarizeOldest,
    "drop_oldest": DropOldest,
    "keep_first_and_last": KeepFirstAndLast,
    "keep_tool_results": KeepToolResults,
}


def resolve_strategy(value: Any) -> CompactionStrategy:
    """Return *value* as a strategy instance.

    Accepts an instance, a class, one of :data:`STRATEGIES_BY_NAME`, or None
    for the default.

    Raises:
        ValueError: When *value* names no known strategy.
    """
    if value is None:
        return SummarizeOldest()
    if isinstance(value, CompactionStrategy):
        return value
    if isinstance(value, type) and issubclass(value, CompactionStrategy):
        return value()
    if isinstance(value, str):
        cls = STRATEGIES_BY_NAME.get(value.strip().lower())
        if cls is None:
            known = ", ".join(sorted(STRATEGIES_BY_NAME))
            raise ValueError(
                f"Unknown compaction strategy {value!r}. Known strategies: {known}. "
                "You can also pass a CompactionStrategy instance of your own."
            )
        return cls()
    raise ValueError(
        f"compaction_strategy must be a CompactionStrategy, a class, or one of "
        f"{sorted(STRATEGIES_BY_NAME)} — not {type(value).__name__}."
    )
