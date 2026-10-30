"""
Agent Communication Bus for effGen multi-agent orchestration.

Provides in-memory pub/sub messaging for inter-agent communication:
- Topic-based routing with wildcard subscriptions
- Typed messages (TASK_ASSIGNMENT, RESULT, STATUS_UPDATE, ERROR, HANDOFF)
- Per-agent mailboxes
- Optional message persistence for replay/debugging
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK_ASSIGNMENT = "task_assignment"
    RESULT = "result"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HANDOFF = "handoff"


@dataclass
class AgentMessage:
    """
    A message exchanged between agents.

    Attributes:
        sender: ID of the sending agent
        recipient: ID of the recipient agent (or topic pattern)
        type: Message type
        payload: Message data
        timestamp: When the message was created
        correlation_id: ID linking related messages (e.g., request/response)
        message_id: Unique message identifier
        topic: Topic for pub/sub routing
        metadata: Additional metadata
    """
    sender: str
    recipient: str
    type: MessageType
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "message_id": self.message_id,
            "topic": self.topic,
            "metadata": self.metadata,
        }


# Type alias for message handler callbacks
MessageHandler = Callable[[AgentMessage], None]
AsyncMessageHandler = Callable[[AgentMessage], Any]  # coroutine


class MessageBus:
    """
    In-memory pub/sub message bus for inter-agent communication.

    Features:
    - Per-agent mailboxes (inbox queues)
    - Topic-based publish/subscribe with wildcard patterns
    - Synchronous and asynchronous message delivery
    - Optional message persistence for replay/debugging
    """

    def __init__(self, persist: bool = False) -> None:
        """
        Initialize the message bus.

        Args:
            persist: If True, store all messages for replay/debugging
        """
        self._mailboxes: dict[str, list[AgentMessage]] = defaultdict(list)
        self._subscribers: dict[str, list[MessageHandler | AsyncMessageHandler]] = defaultdict(list)
        self._persist = persist
        self._history: list[AgentMessage] = []

    # -- Mailbox API --

    def send(self, message: AgentMessage) -> None:
        """
        Send a message to a specific agent's mailbox.

        Args:
            message: The message to send
        """
        self._mailboxes[message.recipient].append(message)

        if self._persist:
            self._history.append(message)

        logger.debug(
            "Message %s -> %s [%s] topic=%s",
            message.sender, message.recipient,
            message.type.value, message.topic,
        )

        # Also deliver to topic subscribers
        if message.topic:
            self._deliver_to_subscribers(message)

    def receive(self, agent_id: str) -> list[AgentMessage]:
        """
        Receive and drain all messages from an agent's mailbox.

        Args:
            agent_id: The agent whose mailbox to read

        Returns:
            List of messages (mailbox is emptied)
        """
        messages = self._mailboxes.pop(agent_id, [])
        return messages

    def peek(self, agent_id: str) -> list[AgentMessage]:
        """
        Peek at messages without removing them.

        Args:
            agent_id: The agent whose mailbox to peek

        Returns:
            List of messages (mailbox is NOT emptied)
        """
        return list(self._mailboxes.get(agent_id, []))

    def has_messages(self, agent_id: str) -> bool:
        """Check if an agent has pending messages."""
        return len(self._mailboxes.get(agent_id, [])) > 0

    def broadcast(self, sender: str, agent_ids: list[str],
                  msg_type: MessageType, payload: Any = None,
                  correlation_id: str | None = None) -> str:
        """
        Send the same message to multiple agents.

        Args:
            sender: Sender agent ID
            agent_ids: List of recipient agent IDs
            msg_type: Message type
            payload: Message payload
            correlation_id: Optional shared correlation ID

        Returns:
            The correlation_id used
        """
        cid = correlation_id or str(uuid.uuid4())
        for recipient in agent_ids:
            msg = AgentMessage(
                sender=sender,
                recipient=recipient,
                type=msg_type,
                payload=payload,
                correlation_id=cid,
            )
            self.send(msg)
        return cid

    # -- Pub/Sub API --

    def subscribe(self, topic: str, handler: MessageHandler | AsyncMessageHandler) -> None:
        """
        Subscribe to a topic pattern.

        Supports wildcard patterns using fnmatch:
        - ``"agents.*"`` matches ``"agents.search"``, ``"agents.summarize"``
        - ``"*"`` matches everything

        Args:
            topic: Topic pattern (supports ``*`` and ``?`` wildcards)
            handler: Callback invoked when a matching message is published
        """
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: MessageHandler | AsyncMessageHandler) -> None:
        """Remove a handler from a topic."""
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to a topic.

        The message is delivered to all handlers whose subscription
        pattern matches ``message.topic``.

        Args:
            message: Message to publish (must have ``topic`` set)
        """
        if self._persist:
            self._history.append(message)

        self._deliver_to_subscribers(message)

    def _deliver_to_subscribers(self, message: AgentMessage) -> None:
        """Deliver a message to matching topic subscribers."""
        for pattern, handlers in self._subscribers.items():
            if fnmatch.fnmatch(message.topic, pattern):
                for handler in handlers:
                    try:
                        result = handler(message)
                        # If handler is async, schedule it
                        if asyncio.iscoroutine(result):
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(result)
                            except RuntimeError:
                                # No running loop — run synchronously
                                asyncio.run(result)
                    except Exception:
                        logger.exception(
                            "Error in subscriber handler for topic %s",
                            pattern,
                        )

    # -- History / Replay --

    def get_history(self, agent_id: str | None = None,
                    msg_type: MessageType | None = None,
                    limit: int | None = None) -> list[AgentMessage]:
        """
        Get message history (requires ``persist=True``).

        Args:
            agent_id: Filter by sender or recipient
            msg_type: Filter by message type
            limit: Maximum number of messages to return

        Returns:
            Filtered list of historical messages
        """
        messages = self._history

        if agent_id:
            messages = [
                m for m in messages
                if m.sender == agent_id or m.recipient == agent_id
            ]

        if msg_type:
            messages = [m for m in messages if m.type == msg_type]

        if limit:
            messages = messages[-limit:]

        return messages

    def clear(self) -> None:
        """Clear all mailboxes, subscribers, and history."""
        self._mailboxes.clear()
        self._subscribers.clear()
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"MessageBus(mailboxes={len(self._mailboxes)}, "
            f"subscribers={sum(len(v) for v in self._subscribers.values())}, "
            f"history={len(self._history)})"
        )
