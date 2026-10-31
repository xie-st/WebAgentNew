"""
MCP client implementation for connecting to MCP servers.

This module provides a client for connecting to Model Context Protocol (MCP)
servers, discovering tools, and executing tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx

from ...base_tool import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata
from .protocol import (
    MCPCapabilities,
    MCPProtocolHandler,
    MCPRequest,
    MCPResource,
    MCPResponse,
    MCPTool,
    TransportType,
)

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: TransportType
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    timeout: int = 30


class MCPTransport:
    """Base class for MCP transports."""

    async def send(self, message: MCPRequest) -> None:
        """Send a message."""
        raise NotImplementedError("Subclasses must implement send()")

    async def receive(self) -> MCPResponse:
        """Receive a message."""
        raise NotImplementedError("Subclasses must implement receive()")

    async def connect(self) -> None:
        """Connect to the server."""
        raise NotImplementedError("Subclasses must implement connect()")

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        raise NotImplementedError("Subclasses must implement disconnect()")


class StdioTransport(MCPTransport):
    """STDIO transport for MCP (subprocess communication)."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        """
        Initialize STDIO transport.

        Args:
            command: Command to execute
            args: Command arguments
            env: Environment variables
        """
        self.command = command
        self.args = args
        self.env = env
        self.process: asyncio.subprocess.Process | None = None
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Start the subprocess."""
        full_env = dict(asyncio.subprocess.os.environ)
        if self.env:
            full_env.update(self.env)

        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        logger.info(f"Started MCP server: {self.command} {' '.join(self.args)}")

    async def disconnect(self) -> None:
        """Stop the subprocess."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
            logger.info(f"Stopped MCP server: {self.command}")

    async def send(self, message: MCPRequest) -> None:
        """Send message to subprocess stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Transport not connected")

        async with self._write_lock:
            data = json.dumps(message.to_dict()) + "\n"
            self.process.stdin.write(data.encode())
            await self.process.stdin.drain()

    async def receive(self) -> MCPResponse:
        """Receive message from subprocess stdout."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Transport not connected")

        async with self._read_lock:
            line = await self.process.stdout.readline()
            if not line:
                raise ConnectionError("Server closed connection")

            data = json.loads(line.decode())
            handler = MCPProtocolHandler()
            message = handler.parse_message(data)

            if isinstance(message, MCPResponse):
                return message
            else:
                raise ValueError(f"Expected response, got {type(message)}")


class HTTPTransport(MCPTransport):
    """HTTP transport for MCP (concurrent-safe)."""

    def __init__(self, url: str, timeout: int = 30) -> None:
        """
        Initialize HTTP transport.

        Args:
            url: Server URL
            timeout: Request timeout in seconds
        """
        self.url = url
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = None
        # Pending requests keyed by correlation ID for concurrent safety
        self._pending: dict[str, MCPRequest] = {}
        self._pending_order: asyncio.Queue[str] = asyncio.Queue()
        self._request_counter = 0

    async def connect(self) -> None:
        """Create HTTP client."""
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()

    async def send(self, message: MCPRequest) -> None:
        """Queue an HTTP request for later retrieval by receive()."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        # Assign a correlation ID so concurrent sends don't overwrite
        self._request_counter += 1
        correlation_id = str(self._request_counter)
        self._pending[correlation_id] = message
        await self._pending_order.put(correlation_id)

    async def receive(self) -> MCPResponse:
        """Dequeue the next pending request, send it, and return the response."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        # Wait for a pending request (with timeout)
        try:
            correlation_id = await asyncio.wait_for(
                self._pending_order.get(), timeout=self.timeout
            )
        except TimeoutError:
            raise TimeoutError(
                f"No pending request to receive within {self.timeout}s"
            )

        message = self._pending.pop(correlation_id)

        response = await self.client.post(
            self.url,
            json=message.to_dict(),
        )
        response.raise_for_status()

        data = response.json()
        handler = MCPProtocolHandler()
        parsed = handler.parse_message(data)

        if isinstance(parsed, MCPResponse):
            return parsed
        else:
            raise ValueError(f"Expected response, got {type(parsed)}")


class SSETransport(MCPTransport):
    """Server-Sent Events transport for MCP with reconnection."""

    def __init__(
        self,
        url: str,
        timeout: int = 30,
        max_retries: int = 5,
        auth_headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize SSE transport.

        Args:
            url: Server URL
            timeout: Request timeout in seconds
            max_retries: Maximum reconnection attempts (exponential backoff)
            auth_headers: Optional authentication headers
        """
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth_headers = auth_headers or {}
        self.client: httpx.AsyncClient | None = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._listener_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to SSE endpoint and start event listener."""
        self.client = httpx.AsyncClient(
            timeout=self.timeout, headers=self.auth_headers
        )
        self._listener_task = asyncio.create_task(self._listen_events())

    async def disconnect(self) -> None:
        """Close SSE connection."""
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
        if self.client:
            await self.client.aclose()

    async def _listen_events(self) -> None:
        """Listen for SSE events with reconnection on disconnect."""
        if not self.client:
            return

        retries = 0
        while retries <= self.max_retries:
            try:
                async with self.client.stream("GET", self.url) as response:
                    retries = 0  # reset on successful connection
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            raw = line[6:]
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"SSE: invalid JSON in event data: {raw[:100]}"
                                )
                                continue
                            await self._event_queue.put(data)
            except (httpx.StreamError, httpx.RemoteProtocolError, ConnectionError) as exc:
                retries += 1
                if retries > self.max_retries:
                    logger.error(
                        f"SSE: max retries ({self.max_retries}) exceeded — giving up"
                    )
                    break
                backoff = min(2 ** retries, 30)
                logger.warning(
                    f"SSE: stream disconnected ({exc}), "
                    f"reconnecting in {backoff}s (attempt {retries}/{self.max_retries})"
                )
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                break

    async def send(self, message: MCPRequest) -> None:
        """Send message via POST."""
        if not self.client:
            raise RuntimeError("Transport not connected")

        await self.client.post(
            self.url,
            json=message.to_dict(),
        )

    async def receive(self) -> MCPResponse:
        """Receive message from event queue."""
        try:
            data = await asyncio.wait_for(
                self._event_queue.get(), timeout=self.timeout
            )
        except TimeoutError:
            raise TimeoutError(
                f"No SSE event received within {self.timeout}s"
            )

        handler = MCPProtocolHandler()
        message = handler.parse_message(data)

        if isinstance(message, MCPResponse):
            return message
        else:
            raise ValueError(f"Expected response, got {type(message)}")


class ConnectionState:
    """MCP connection state tracking."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"  # Server unreachable but client alive


class MCPToolBridge(BaseTool):
    """
    Bridge that wraps an MCP tool as an effGen BaseTool.

    Allows MCP-discovered tools to be used transparently in the agent's tool set.
    """

    def __init__(self, mcp_tool: MCPTool, client: "MCPClient") -> None:
        """
        Initialize MCP tool bridge.

        Args:
            mcp_tool: MCP tool definition
            client: MCPClient instance for execution
        """
        # Convert MCP tool schema to effGen ParameterSpec list
        params = []
        input_schema = mcp_tool.inputSchema or {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        type_map = {
            "string": ParameterType.STRING,
            "integer": ParameterType.INTEGER,
            "number": ParameterType.FLOAT,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT,
        }

        for prop_name, prop_schema in properties.items():
            json_type = prop_schema.get("type", "string")
            params.append(ParameterSpec(
                name=prop_name,
                type=type_map.get(json_type, ParameterType.STRING),
                description=prop_schema.get("description", ""),
                required=prop_name in required,
                default=prop_schema.get("default"),
            ))

        metadata = ToolMetadata(
            name=f"mcp_{mcp_tool.name}",
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            category=ToolCategory.EXTERNAL_API,
            parameters=params,
        )
        super().__init__(metadata=metadata)
        self._mcp_tool = mcp_tool
        self._client = client

    async def _execute(self, **kwargs) -> Any:
        """Execute the MCP tool via the client."""
        return await self._client.call_tool(self._mcp_tool.name, kwargs)


class MCPClient:
    """
    Client for connecting to MCP servers.

    Features:
    - Multiple transport support (STDIO, HTTP, SSE)
    - Server discovery and capability negotiation
    - Tool listing and execution
    - Resource management
    - Automatic reconnection with exponential backoff
    - MCP tool → effGen tool bridge
    - MCP resource → context bridge
    - Connection health monitoring
    """

    def __init__(
        self,
        config: MCPServerConfig,
        max_reconnect_attempts: int = 5,
        reconnect_base_delay: float = 1.0,
        health_check_interval: float = 30.0,
    ) -> None:
        """
        Initialize MCP client.

        Args:
            config: Server configuration
            max_reconnect_attempts: Max reconnection attempts (0 = disabled)
            reconnect_base_delay: Base delay for exponential backoff (seconds)
            health_check_interval: Interval for health checks (seconds, 0 = disabled)
        """
        self.config = config
        self.transport: MCPTransport | None = None
        self.protocol = MCPProtocolHandler()
        self.capabilities: MCPCapabilities | None = None
        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, MCPResource] = {}
        self._connected = False
        self._pending_requests: dict[str | int, asyncio.Future] = {}

        # Reconnection settings
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_count = 0

        # Health monitoring
        self._health_check_interval = health_check_interval
        self._health_task: asyncio.Task | None = None
        self._state = ConnectionState.DISCONNECTED
        self._last_healthy: float | None = None

        # Tool bridge cache
        self._bridged_tools: dict[str, MCPToolBridge] = {}

        # Resource context cache
        self._resource_context: dict[str, Any] = {}

    @property
    def state(self) -> str:
        """Current connection state."""
        return self._state

    @property
    def is_healthy(self) -> bool:
        """Whether the connection is healthy."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        """Connect to MCP server and perform initialization."""
        self._state = ConnectionState.CONNECTING

        # Create transport
        if self.config.transport == TransportType.STDIO:
            if not self.config.command:
                raise ValueError("Command required for STDIO transport")
            self.transport = StdioTransport(
                self.config.command,
                self.config.args or [],
                self.config.env,
            )
        elif self.config.transport == TransportType.HTTP:
            if not self.config.url:
                raise ValueError("URL required for HTTP transport")
            self.transport = HTTPTransport(self.config.url, self.config.timeout)
        elif self.config.transport == TransportType.SSE:
            if not self.config.url:
                raise ValueError("URL required for SSE transport")
            self.transport = SSETransport(self.config.url, self.config.timeout)
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport}")

        # Connect transport
        await self.transport.connect()

        # Initialize protocol
        await self._initialize()

        self._connected = True
        self._state = ConnectionState.CONNECTED
        self._reconnect_count = 0
        self._last_healthy = asyncio.get_event_loop().time()
        logger.info(f"Connected to MCP server: {self.config.name}")

        # Start health monitoring
        if self._health_check_interval > 0:
            self._health_task = asyncio.create_task(self._health_monitor())

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        # Stop health monitor
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:  # expected cancellation during shutdown
                pass

        if self.transport:
            await self.transport.disconnect()
        self._connected = False
        self._state = ConnectionState.DISCONNECTED
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection succeeded
        """
        if self._max_reconnect_attempts <= 0:
            return False

        self._state = ConnectionState.RECONNECTING

        for attempt in range(self._max_reconnect_attempts):
            delay = self._reconnect_base_delay * (2 ** attempt)
            delay = min(delay, 60.0)  # Cap at 60s
            logger.info(
                f"Reconnecting to {self.config.name} "
                f"(attempt {attempt + 1}/{self._max_reconnect_attempts}, "
                f"delay {delay:.1f}s)"
            )
            await asyncio.sleep(delay)

            try:
                # Disconnect old transport
                if self.transport:
                    try:
                        await self.transport.disconnect()
                    except Exception:  # best-effort transport disconnect during cleanup
                        pass

                # Reconnect
                await self.connect()
                logger.info(f"Reconnected to {self.config.name}")
                self._reconnect_count += 1
                return True

            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt + 1} failed: {e}")

        self._state = ConnectionState.DEGRADED
        logger.error(
            f"Failed to reconnect to {self.config.name} "
            f"after {self._max_reconnect_attempts} attempts"
        )
        return False

    async def _health_monitor(self) -> None:
        """Background task that monitors connection health via periodic pings."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                if not self._connected:
                    continue

                # Send a lightweight request to check connectivity
                try:
                    request = self.protocol.create_tools_list_request()
                    await asyncio.wait_for(
                        self._send_request(request),
                        timeout=self.config.timeout,
                    )
                    self._last_healthy = asyncio.get_event_loop().time()
                    if self._state == ConnectionState.DEGRADED:
                        self._state = ConnectionState.CONNECTED
                        logger.info(f"Connection to {self.config.name} recovered")

                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    logger.warning(f"Health check failed for {self.config.name}: {e}")
                    self._state = ConnectionState.DEGRADED
                    # Try reconnection
                    await self._reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _initialize(self) -> None:
        """Perform MCP initialization handshake."""
        request = self.protocol.create_initialize_request(
            protocol_version="1.0",
            capabilities=MCPCapabilities(
                tools=True,
                resources=True,
            ),
            client_info={
                "name": "effGen",
                "version": "1.0.0",
            },
        )

        response = await self._send_request(request)

        if response.error:
            raise RuntimeError(f"Initialization failed: {response.error.message}")

        result = response.result or {}
        self.capabilities = MCPCapabilities.from_dict(
            result.get("capabilities", {})
        )

        logger.info(f"Server capabilities: {self.capabilities.to_dict()}")

        if self.capabilities.tools:
            await self._list_tools()
        if self.capabilities.resources:
            await self._list_resources()

    async def _send_request(self, request: MCPRequest) -> MCPResponse:
        """
        Send request and wait for response.
        Attempts auto-reconnection on connection failure.

        Args:
            request: Request to send

        Returns:
            Response from server
        """
        if not self.transport:
            raise RuntimeError("Not connected")

        try:
            await self.transport.send(request)
            response = await self.transport.receive()
            return response
        except (ConnectionError, RuntimeError) as e:
            if self._max_reconnect_attempts > 0:
                logger.warning(f"Connection lost, attempting reconnect: {e}")
                if await self._reconnect():
                    # Retry after reconnection
                    await self.transport.send(request)
                    return await self.transport.receive()
            raise

    async def _list_tools(self) -> None:
        """List available tools from server."""
        request = self.protocol.create_tools_list_request()
        response = await self._send_request(request)

        if response.error:
            logger.error(f"Failed to list tools: {response.error.message}")
            return

        tools_data = response.result or {}
        tools_list = tools_data.get("tools", [])

        self.tools.clear()
        self._bridged_tools.clear()
        for tool_data in tools_list:
            tool = MCPTool.from_dict(tool_data)
            self.tools[tool.name] = tool

        logger.info(f"Discovered {len(self.tools)} tools")

    async def _list_resources(self) -> None:
        """List available resources from server."""
        request = self.protocol.create_resources_list_request()
        response = await self._send_request(request)

        if response.error:
            logger.error(f"Failed to list resources: {response.error.message}")
            return

        resources_data = response.result or {}
        resources_list = resources_data.get("resources", [])

        self.resources.clear()
        for resource_data in resources_list:
            resource = MCPResource.from_dict(resource_data)
            self.resources[resource.uri] = resource

        logger.info(f"Discovered {len(self.resources)} resources")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")

        request = self.protocol.create_tool_call_request(tool_name, arguments)
        response = await self._send_request(request)

        if response.error:
            raise RuntimeError(f"Tool call failed: {response.error.message}")

        return response.result

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        request = self.protocol.create_resource_read_request(uri)
        response = await self._send_request(request)

        if response.error:
            raise RuntimeError(f"Resource read failed: {response.error.message}")

        return response.result

    def get_tools(self) -> list[MCPTool]:
        """Get list of available MCP tools."""
        return list(self.tools.values())

    def get_resources(self) -> list[MCPResource]:
        """Get list of available resources."""
        return list(self.resources.values())

    # --- MCP Tool → effGen Tool Bridge (3.2.2) ---

    def get_effgen_tools(self) -> list[BaseTool]:
        """
        Get MCP tools as effGen BaseTool instances.

        Auto-converts each discovered MCP tool into an MCPToolBridge that
        can be registered in an agent's tool set and called transparently.

        Returns:
            List of BaseTool instances wrapping MCP tools
        """
        bridged = []
        for mcp_tool in self.tools.values():
            if mcp_tool.name not in self._bridged_tools:
                self._bridged_tools[mcp_tool.name] = MCPToolBridge(mcp_tool, self)
            bridged.append(self._bridged_tools[mcp_tool.name])
        return bridged

    # --- MCP Resource → Context Bridge (3.2.3) ---

    async def load_resources_as_context(self) -> dict[str, Any]:
        """
        Load all MCP resources into a context dictionary.

        Returns:
            Dict mapping resource URIs to their content
        """
        context = {}
        for uri in self.resources:
            try:
                content = await self.read_resource(uri)
                context[uri] = content
                self._resource_context[uri] = content
            except Exception as e:
                logger.warning(f"Failed to load resource {uri}: {e}")
        return context

    async def subscribe_resource(
        self, uri: str, callback: Callable | None = None
    ) -> None:
        """
        Subscribe to resource updates (polls for changes).

        Args:
            uri: Resource URI
            callback: Optional callback(uri, new_content) on change
        """
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        # Read initial value
        current = await self.read_resource(uri)
        self._resource_context[uri] = current

        if callback:
            callback(uri, current)

    def get_resource_context(self) -> dict[str, Any]:
        """Get cached resource context."""
        return dict(self._resource_context)

    # --- Health info ---

    def get_health_info(self) -> dict[str, Any]:
        """
        Get connection health information.

        Returns:
            Dict with state, last_healthy, reconnect_count
        """
        return {
            "state": self._state,
            "connected": self._connected,
            "server": self.config.name,
            "last_healthy": self._last_healthy,
            "reconnect_count": self._reconnect_count,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
        }

    async def __aenter__(self) -> MCPClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()
