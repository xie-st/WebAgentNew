"""
Model Context Protocol (MCP) implementation for effGen.

This module provides MCP client and server implementations for tool
and resource communication.

Two MCP stacks ship intentionally, for different needs:

- ``effgen.tools.protocols.mcp`` (this package) is a self-contained
  implementation of the MCP wire protocol with **no external SDK dependency**.
  Use it when you cannot install the official ``mcp`` package.
- ``effgen.tools.protocols.mcp_official`` is built on the official MCP Python
  SDK (FastMCP) and is the **recommended** stack for standards-compliant
  interop with other MCP tooling. Prefer it when ``pip install "mcp[cli]"`` is
  available.

Both expose effGen tools over MCP; pick the one that matches your dependency
constraints.
"""

from .client import (
    ConnectionState,
    HTTPTransport,
    MCPClient,
    MCPServerConfig,
    MCPToolBridge,
    MCPTransport,
    SSETransport,
    StdioTransport,
)
from .protocol import (
    ErrorCode,
    MCPCapabilities,
    MCPError,
    MCPNotification,
    MCPProtocolHandler,
    MCPRequest,
    MCPResource,
    MCPResponse,
    MCPTool,
    MCPVersion,
    MessageType,
    TransportType,
)
from .server import (
    MCPServer,
    create_server,
    main_http,
    main_stdio,
)

__all__ = [
    # Protocol
    "MCPProtocolHandler",
    "MCPTool",
    "MCPResource",
    "MCPCapabilities",
    "MCPRequest",
    "MCPResponse",
    "MCPNotification",
    "MCPError",
    "ErrorCode",
    "TransportType",
    "MessageType",
    "MCPVersion",
    # Client
    "MCPClient",
    "MCPServerConfig",
    "MCPTransport",
    "StdioTransport",
    "HTTPTransport",
    "SSETransport",
    "MCPToolBridge",
    "ConnectionState",
    # Server
    "MCPServer",
    "create_server",
    "main_stdio",
    "main_http",
]
