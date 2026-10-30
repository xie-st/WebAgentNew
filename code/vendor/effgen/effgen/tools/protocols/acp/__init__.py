"""
Agent Communication Protocol (ACP) Integration.

This package implements IBM's ACP protocol for agent-to-agent communication,
providing agent manifests, synchronous/asynchronous requests, task tracking,
and OpenTelemetry instrumentation.

.. note::
   **Experimental.** The ACP server (``ACPServer``) and client (``ACPClient``)
   are smoke-tested locally (manifest exchange, synchronous capability execution,
   and capability-token auth), but have not been validated against the external
   BeeAI platform. The HTTP server binds to 127.0.0.1 by default and warns when
   bound to a public address; enable ``require_auth`` before exposing it.
   Interfaces may change.
"""

from .client import (
    ACPAuthHandler,
    ACPClient,
    ACPClientConfig,
    ACPDiscoveryClient,
    APIKeyAuthHandler,
    BearerAuthHandler,
    TokenAuthHandler,
    create_capability_token,
)
from .protocol import (
    ACPError,
    ACPProtocolHandler,
    ACPRequest,
    ACPResponse,
    ACPVersion,
    AgentManifest,
    CapabilityDefinition,
    CapabilityToken,
    ErrorSeverity,
    RequestType,
    SchemaDefinition,
    TaskInfo,
    TaskStatus,
)
from .server import (
    ACPCapabilityRegistry,
    ACPServer,
    ACPServerConfig,
    capability,
)

__all__ = [
    # Protocol
    "ACPProtocolHandler",
    "AgentManifest",
    "ACPRequest",
    "ACPResponse",
    "ACPError",
    "TaskInfo",
    "TaskStatus",
    "RequestType",
    "ErrorSeverity",
    "SchemaDefinition",
    "CapabilityDefinition",
    "CapabilityToken",
    "ACPVersion",
    # Client
    "ACPClient",
    "ACPClientConfig",
    "ACPAuthHandler",
    "TokenAuthHandler",
    "APIKeyAuthHandler",
    "BearerAuthHandler",
    "ACPDiscoveryClient",
    "create_capability_token",
    # Server
    "ACPServer",
    "ACPServerConfig",
    "ACPCapabilityRegistry",
    "capability",
]
