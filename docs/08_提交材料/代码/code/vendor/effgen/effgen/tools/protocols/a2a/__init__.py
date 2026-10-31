"""
Agent-to-Agent (A2A) Protocol Integration.

This package implements Google's A2A protocol for agent-to-agent communication,
providing message protocol handling, task lifecycle management, and context passing.

.. note::
   **Experimental.** effGen ships the A2A *client* (``A2AClient``), the agent-card
   model (``AgentCard``), authentication handlers, and the wire protocol/task
   model (``A2AProtocolHandler``). It does not bundle an A2A *server*; point the
   client at an external A2A-compatible agent. The client and auth paths are
   smoke-tested, but the protocol has not been validated against a third-party
   A2A implementation. Interfaces may change.
"""

from .agent_card import (
    AgentCard,
    AuthScheme,
    Capability,
    CapabilityType,
    EndpointConfig,
)
from .client import (
    A2AClient,
    A2AClientConfig,
    APIKeyAuthHandler,
    AuthHandler,
    BearerAuthHandler,
    OAuth2AuthHandler,
    discover_agents,
)
from .protocol import (
    A2AError,
    A2AMessage,
    A2AProtocolHandler,
    A2AVersion,
    Artifact,
    ErrorCode,
    MessagePart,
    MessagePartType,
    Task,
    TaskRequest,
    TaskState,
    TaskUpdate,
)

__all__ = [
    # Protocol
    "A2AProtocolHandler",
    "Task",
    "TaskRequest",
    "TaskUpdate",
    "TaskState",
    "A2AMessage",
    "A2AError",
    "ErrorCode",
    "Artifact",
    "MessagePart",
    "MessagePartType",
    "A2AVersion",
    # Agent Card
    "AgentCard",
    "Capability",
    "CapabilityType",
    "EndpointConfig",
    "AuthScheme",
    # Client
    "A2AClient",
    "A2AClientConfig",
    "AuthHandler",
    "BearerAuthHandler",
    "OAuth2AuthHandler",
    "APIKeyAuthHandler",
    "discover_agents",
]
