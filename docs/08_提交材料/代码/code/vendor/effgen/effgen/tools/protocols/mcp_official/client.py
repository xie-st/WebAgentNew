"""
Official MCP Client implementation using the mcp library.

This module provides a standards-compliant MCP client for connecting to
MCP servers and using their tools and resources.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    Prompt,
    Resource,
    Tool,
)
from pydantic import AnyUrl

from ...base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

# The streamable-HTTP transport factory was renamed in mcp 1.28; the old
# ``streamablehttp_client`` spelling is deprecated there and absent from 2.x.
try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover - older SDKs carry only the old name
    from mcp.client.streamable_http import (  # type: ignore[attr-defined]
        streamablehttp_client as streamable_http_client,
    )

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""
    name: str
    transport: str = "stdio"  # "stdio", "http", or "sse"

    # For STDIO transport
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None

    # For HTTP/SSE transports
    url: str | None = None

    # Common settings
    timeout: int = 30


class EffGenMCPClient:
    """
    Official MCP client implementation for effGen.

    This client uses the official MCP Python SDK to connect to MCP servers,
    discover tools and resources, and execute operations.

    Features:
    - Standards-compliant MCP protocol
    - Multiple transport support (STDIO, HTTP, SSE)
    - Tool discovery and execution
    - Resource access
    - Prompt management
    - Async context manager support

    Example:
        ```python
        from effgen.tools.protocols.mcp_official import EffGenMCPClient, MCPServerConfig

        # Configure server
        config = MCPServerConfig(
            name="my-server",
            transport="stdio",
            command="python",
            args=["server.py"]
        )

        # Use client
        async with EffGenMCPClient(config) as client:
            # List tools
            tools = client.get_tools()

            # Call a tool
            result = await client.call_tool("my_tool", {"arg": "value"})
        ```
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self.session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._connected = False
        # The event loop the session is bound to. MCP transport streams are
        # loop-affine, so tool calls issued from another loop/thread (e.g. an
        # effGen Agent running its tools in a worker loop) must be marshaled
        # back here via run_coroutine_threadsafe.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Tool and resource caches
        self._tools: dict[str, Tool] = {}
        self._resources: dict[str, Resource] = {}
        self._prompts: dict[str, Prompt] = {}

        logger.info(f"Initialized MCP client for server: {config.name}")

    async def connect(self) -> None:
        """Connect to the MCP server.

        ``connect()``, any tool/resource/prompt calls, and the matching
        ``disconnect()`` must run inside the **same** ``asyncio.run()`` (or
        event loop) — the transport is loop-affine and cannot be closed from a
        different loop/task. Prefer ``async with EffGenMCPClient(...)``, which
        guarantees this by construction; calling ``connect()`` in one
        ``asyncio.run()`` and ``disconnect()`` in a separate later one leaves
        the transport's resources stranded on the now-closed first loop.
        """
        if self._connected:
            return

        stack = AsyncExitStack()
        try:
            # Create appropriate transport context
            if self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError("Command required for STDIO transport")

                server_params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args or [],
                    env=self.config.env,
                )

                transport_context = stdio_client(server_params)

            elif self.config.transport in ("http", "streamable-http"):
                if not self.config.url:
                    raise ValueError("URL required for HTTP transport")

                transport_context = streamable_http_client(self.config.url)

            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            # Enter transport context via the exit stack
            transport = await stack.enter_async_context(transport_context)

            # stdio yields (read, write); HTTP yields (read, write, get_session_id)
            read_stream, write_stream = transport[0], transport[1]

            # Create AND enter the session context manager — entering it is what
            # starts the background receive loop; without it initialize() hangs.
            self.session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize session
            await self.session.initialize()

            # Discover capabilities
            await self._discover()

            self._exit_stack = stack
            self._loop = asyncio.get_running_loop()
            self._connected = True
            logger.info(f"Connected to MCP server: {self.config.name}")

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}", exc_info=True)
            await stack.aclose()
            self.session = None
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server.

        Must be awaited from the same event loop ``connect()`` ran on — the
        underlying transport's cancel scopes are bound to that loop/task and
        cannot be torn down from another one (this is what ``async with
        EffGenMCPClient(...)`` guarantees by construction). Calling it from a
        different loop skips the (unrecoverable) transport teardown and logs a
        warning instead of letting the anyio cancel-scope error surface.
        """
        if not self._connected:
            return

        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._loop is not None and current_loop is not self._loop:
            logger.warning(
                "disconnect() for MCP server '%s' was called from a different "
                "event loop than connect() used; the transport is bound to that "
                "loop and cannot be closed from here. Call connect()/disconnect() "
                "within the same asyncio.run() (or event loop), or use "
                "`async with EffGenMCPClient(...)`.",
                self.config.name,
            )
        elif self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}", exc_info=True)

        self._exit_stack = None
        self._connected = False
        self.session = None
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _discover(self) -> None:
        """Discover server capabilities, tools, and resources."""
        if not self.session:
            raise RuntimeError("Not connected")

        try:
            # List tools
            tools_result = await self.session.list_tools()
            for tool in tools_result.tools:
                self._tools[tool.name] = tool

            logger.info(f"Discovered {len(self._tools)} tools")

            # List resources
            try:
                resources_result = await self.session.list_resources()
                for resource in resources_result.resources:
                    self._resources[str(resource.uri)] = resource

                logger.info(f"Discovered {len(self._resources)} resources")
            except Exception as e:
                logger.debug(f"No resources available: {e}")

            # List prompts
            try:
                prompts_result = await self.session.list_prompts()
                for prompt in prompts_result.prompts:
                    self._prompts[prompt.name] = prompt

                logger.info(f"Discovered {len(self._prompts)} prompts")
            except Exception as e:
                logger.debug(f"No prompts available: {e}")

        except Exception as e:
            logger.error(f"Error discovering capabilities: {e}", exc_info=True)
            raise

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
            RuntimeError: If not connected or call fails
        """
        if not self._connected or not self.session:
            raise RuntimeError("Not connected")

        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")

        try:
            result = await self.session.call_tool(
                tool_name,
                arguments=arguments or {}
            )

            logger.debug(f"Tool {tool_name} executed successfully")
            return result

        except Exception as e:
            logger.error(f"Tool call failed: {e}", exc_info=True)
            raise RuntimeError(f"Tool call failed: {e}")

    async def read_resource(self, uri: str) -> Any:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content

        Raises:
            ValueError: If resource not found
            RuntimeError: If not connected or read fails
        """
        if not self._connected or not self.session:
            raise RuntimeError("Not connected")

        try:
            result = await self.session.read_resource(AnyUrl(uri))
            logger.debug(f"Resource {uri} read successfully")
            return result

        except Exception as e:
            logger.error(f"Resource read failed: {e}", exc_info=True)
            raise RuntimeError(f"Resource read failed: {e}")

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None
    ) -> Any:
        """
        Get a prompt from the MCP server.

        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments

        Returns:
            Prompt result

        Raises:
            ValueError: If prompt not found
            RuntimeError: If not connected or get fails
        """
        if not self._connected or not self.session:
            raise RuntimeError("Not connected")

        if prompt_name not in self._prompts:
            raise ValueError(f"Prompt not found: {prompt_name}")

        try:
            result = await self.session.get_prompt(
                prompt_name,
                arguments=arguments or {}
            )

            logger.debug(f"Prompt {prompt_name} retrieved successfully")
            return result

        except Exception as e:
            logger.error(f"Prompt get failed: {e}", exc_info=True)
            raise RuntimeError(f"Prompt get failed: {e}")

    def get_tools(self) -> list[Tool]:
        """
        Get list of available tools.

        This is a **synchronous** accessor — it returns the tool list cached at
        :meth:`connect` time, so call it directly (``client.get_tools()``); do
        **not** ``await`` it. Only :meth:`connect`, :meth:`call_tool`, and the
        other I/O methods are coroutines.

        Returns:
            List of tool definitions
        """
        return list(self._tools.values())

    def get_effgen_tools(self, prefix: str = "mcp_") -> list[Any]:
        """
        Wrap the discovered MCP tools as effGen ``BaseTool`` instances.

        The returned tools can be passed straight into an effGen ``Agent`` so it
        can call tools served by any MCP server. The client must stay connected
        for the lifetime of the agent run.

        This is a **synchronous** accessor (it reads the catalog cached at
        :meth:`connect`); call it directly, do not ``await`` it.

        Each returned tool is named ``{prefix}{server_tool_name}`` — with the
        default ``prefix="mcp_"``, a server tool ``calculator`` is exposed to the
        agent as ``mcp_calculator``. Look tools up by that namespaced name; to
        reach the original server name pass ``prefix=""`` here, or use
        :meth:`get_tool` (which is keyed by the unprefixed name).

        .. note::
            An ``Agent`` that holds these tools must be driven with
            ``await agent.run_async(...)``, not the synchronous ``Agent.run()``:
            each tool call goes back over the MCP session bound to the calling
            event loop, which a synchronous ``run()`` under that loop cannot
            drive.

        Args:
            prefix: Name prefix for the wrapped tools (avoids collisions with
                effGen's builtin tool names). Defaults to ``"mcp_"``; pass
                ``""`` to keep the original server tool names.

        Returns:
            List of effGen ``BaseTool`` instances, each named
            ``{prefix}{server_tool_name}``.
        """
        return [
            _MCPOfficialToolBridge(tool, self, prefix=prefix)
            for tool in self._tools.values()
        ]

    def get_tool(self, name: str) -> Tool | None:
        """
        Get a specific tool by name.

        Args:
            name: Tool name

        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(name)

    def get_resources(self) -> list[Resource]:
        """
        Get list of available resources.

        Returns:
            List of resource definitions
        """
        return list(self._resources.values())

    def get_resource(self, uri: str) -> Resource | None:
        """
        Get a specific resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource definition or None if not found
        """
        return self._resources.get(uri)

    def get_prompts(self) -> list[Prompt]:
        """
        Get list of available prompts.

        Returns:
            List of prompt definitions
        """
        return list(self._prompts.values())

    def get_prompt_info(self, name: str) -> Prompt | None:
        """
        Get a specific prompt info by name.

        Args:
            name: Prompt name

        Returns:
            Prompt definition or None if not found
        """
        return self._prompts.get(name)

    async def __aenter__(self) -> EffGenMCPClient:
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


class _MCPOfficialToolBridge(BaseTool):
    """Wrap an MCP tool (discovered via the official client) as an effGen tool."""

    _JSON_TYPE_MAP = {
        "string": ParameterType.STRING,
        "integer": ParameterType.INTEGER,
        "number": ParameterType.FLOAT,
        "boolean": ParameterType.BOOLEAN,
        "array": ParameterType.ARRAY,
        "object": ParameterType.OBJECT,
    }

    def __init__(self, mcp_tool: Tool, client: EffGenMCPClient, prefix: str = "mcp_"):
        input_schema = mcp_tool.inputSchema or {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        params = []
        for prop_name, prop_schema in properties.items():
            json_type = prop_schema.get("type", "string")
            params.append(
                ParameterSpec(
                    name=prop_name,
                    type=self._JSON_TYPE_MAP.get(json_type, ParameterType.STRING),
                    description=prop_schema.get("description", ""),
                    required=prop_name in required,
                    default=prop_schema.get("default"),
                )
            )

        metadata = ToolMetadata(
            name=f"{prefix}{mcp_tool.name}",
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            category=ToolCategory.EXTERNAL_API,
            parameters=params,
        )
        super().__init__(metadata=metadata)
        self._mcp_name = mcp_tool.name
        self._client = client

    async def _execute(self, **kwargs) -> Any:
        session_loop = self._client._loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if session_loop is not None and current_loop is not session_loop:
            # Called from a different loop/thread (e.g. an Agent's worker loop):
            # marshal the call back to the loop that owns the MCP session. If
            # that loop's own thread is the one synchronously blocked waiting
            # on this call (Agent.run() invoked from inside an `async with
            # EffGenMCPClient(...)` block), the marshal can never be serviced —
            # fail immediately with an actionable message instead of waiting
            # out the bridge's timeout (and without creating a coroutine that
            # would then go unawaited).
            from effgen.utils.async_bridge import is_loop_blocked

            if is_loop_blocked(session_loop):
                raise TimeoutError(
                    f"Cannot call MCP tool '{self._mcp_name}': its session is "
                    "bound to an event loop that is synchronously blocked by "
                    "this same call (Agent.run() was invoked from inside the "
                    "event loop holding the MCP client session). Use "
                    "`await agent.run_async(...)` instead so the tool call "
                    "runs on the loop that owns the MCP session."
                )
            future = asyncio.run_coroutine_threadsafe(
                self._client.call_tool(self._mcp_name, kwargs), session_loop
            )
            result = await asyncio.wrap_future(future)
        else:
            result = await self._client.call_tool(self._mcp_name, kwargs)
        texts = [c.text for c in result.content if getattr(c, "text", None)]
        text = "\n".join(texts)
        if result.isError:
            return {"success": False, "error": text or "MCP tool returned an error"}
        if getattr(result, "structuredContent", None):
            return result.structuredContent
        return text


def create_client(config: MCPServerConfig) -> EffGenMCPClient:
    """
    Create an MCP client instance.

    Args:
        config: Server configuration

    Returns:
        MCP client instance
    """
    return EffGenMCPClient(config)


async def create_stdio_client(
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> EffGenMCPClient:
    """
    Create and connect to a STDIO MCP client.

    Args:
        name: Server name
        command: Command to execute
        args: Command arguments
        env: Environment variables

    Returns:
        Connected MCP client
    """
    config = MCPServerConfig(
        name=name,
        transport="stdio",
        command=command,
        args=args,
        env=env,
    )

    client = create_client(config)
    await client.connect()
    return client


async def create_http_client(
    name: str,
    url: str,
) -> EffGenMCPClient:
    """
    Create and connect to an HTTP MCP client.

    Args:
        name: Server name
        url: Server URL

    Returns:
        Connected MCP client
    """
    config = MCPServerConfig(
        name=name,
        transport="http",
        url=url,
    )

    client = create_client(config)
    await client.connect()
    return client

