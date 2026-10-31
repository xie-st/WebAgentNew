"""
Official MCP Server implementation using the mcp library.

This module provides a standards-compliant MCP server that exposes effGen
tools using the official Model Context Protocol Python SDK.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from ...registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)


# Tools that execute code, reach the shell, mutate the host filesystem, or
# expose host/system internals. They are NOT exposed over MCP unless the
# operator explicitly opts in (``expose_unsafe_tools=True`` or by naming them in
# ``allowed_tools``). Fail-closed by default: an MCP client should never get a
# remote shell or arbitrary code execution just by connecting.
UNSAFE_TOOLS: frozenset[str] = frozenset(
    {
        "bash",
        "python_repl",
        "code_executor",
        "code_execution",
        "docker",
        "git",
        "file_operations",
        "system_info",
        "anthropic_bash",
        "anthropic_computer",
        "anthropic_text_editor",
        "openai_code_interpreter",
    }
)

# Map effGen parameter type names to Python types so FastMCP can build a JSON
# schema from the generated function signature.
_PARAM_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class EffGenMCPServerConfig:
    """Configuration for effGen MCP server."""
    name: str = "effgen-tools"
    version: str = "1.0.0"
    instructions: str = "MCP server exposing effGen tools for AI agents"
    tools_registry: ToolRegistry | None = None
    # Security: fail-closed tool exposure. By default, code-execution / shell /
    # filesystem tools are hidden. Set ``expose_unsafe_tools=True`` to expose
    # them, or list specific tools in ``allowed_tools`` (an explicit opt-in).
    expose_unsafe_tools: bool = False
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] = field(default_factory=list)


class EffGenMCPServer:
    """
    Official MCP server implementation for effGen.

    This server uses the official MCP Python SDK (FastMCP) to expose
    effGen tools in a standards-compliant way.

    Features:
    - Standards-compliant MCP protocol
    - Automatic tool registration and discovery
    - Structured output support
    - Progress reporting
    - Error handling
    - STDIO and HTTP transports

    Example:
        ```python
        from effgen.tools.registry import ToolRegistry
        from effgen.tools.protocols.mcp_official import EffGenMCPServer

        # Create registry and register tools
        registry = ToolRegistry()
        registry.discover_builtin_tools()

        # Create and run server
        server = EffGenMCPServer(tools_registry=registry)
        await server.run_stdio()
        ```
    """

    def __init__(
        self,
        name: str = "effgen-tools",
        version: str = "1.0.0",
        instructions: str | None = None,
        tools_registry: ToolRegistry | None = None,
        expose_unsafe_tools: bool = False,
        allowed_tools: list[str] | None = None,
        blocked_tools: list[str] | None = None,
        config: EffGenMCPServerConfig | None = None,
    ) -> None:
        """
        Initialize EffGen MCP server.

        Takes either the fields directly (``name=``, ``version=``, ...) or a
        single ``config=EffGenMCPServerConfig(...)``. When both are given,
        ``config`` wins for every field it sets.

        Args:
            name: Server name
            version: Server version
            instructions: Server instructions for clients
            tools_registry: Tool registry to expose. Defaults to the shared
                process-wide registry (``effgen.tools.registry.get_registry()``)
                so a tool registered elsewhere in the same process — via
                ``register_tool()``, an ``Agent``, or ``effgen tools list`` —
                is visible here too. Pass ``ToolRegistry()`` explicitly for an
                isolated registry.
            expose_unsafe_tools: Expose code-execution / shell / filesystem
                tools over MCP. Off by default (fail-closed).
            allowed_tools: If set, expose ONLY these tools (an explicit opt-in
                that overrides the unsafe-tool block for the named tools).
            blocked_tools: Additional tool names to hide regardless of the above.
            config: An :class:`EffGenMCPServerConfig` bundling the above fields.
        """
        if config is not None:
            name = config.name
            version = config.version
            instructions = config.instructions
            tools_registry = config.tools_registry
            expose_unsafe_tools = config.expose_unsafe_tools
            allowed_tools = config.allowed_tools
            blocked_tools = config.blocked_tools

        self.name = name
        self.version = version
        self.instructions = instructions or (
            "MCP server exposing effGen tools for AI agents. "
            "Provides access to various tools for "
            "web search, data processing, and more."
        )
        self.tools_registry = tools_registry or get_registry()
        self.expose_unsafe_tools = expose_unsafe_tools
        self.allowed_tools = allowed_tools
        self.blocked_tools = list(blocked_tools or [])
        self._initialized = False
        self._exposed_tools: list[str] = []

        # Create FastMCP instance
        self.mcp = FastMCP(
            name=self.name,
            instructions=self.instructions,
        )

        logger.info(f"Initialized EffGen MCP Server: {self.name} v{self.version}")

    def _select_tools(self) -> list[str]:
        """Apply the fail-closed allowlist/blocklist to the discovered tools."""
        all_tools = sorted(self.tools_registry.list_tools())
        if self.allowed_tools is not None:
            allow = set(self.allowed_tools)
            selected = [t for t in all_tools if t in allow]
        else:
            selected = list(all_tools)
            if not self.expose_unsafe_tools:
                selected = [t for t in selected if t not in UNSAFE_TOOLS]
        if self.blocked_tools:
            blocked = set(self.blocked_tools)
            selected = [t for t in selected if t not in blocked]
        return selected

    async def initialize(self) -> None:
        """Initialize the server and register tools."""
        if self._initialized:
            return

        # Discover builtin tools (metadata only — individual tools are
        # initialized lazily on first use so the fail-closed allowlist also
        # avoids spinning up heavy/unsafe subsystems like the code sandbox).
        self.tools_registry.discover_builtin_tools()

        # Apply the fail-closed allowlist BEFORE registering anything.
        self._exposed_tools = self._select_tools()

        # Register selected tools with MCP
        for tool_name in self._exposed_tools:
            self._register_tool(tool_name)

        self._initialized = True
        hidden = len(self.tools_registry.list_tools()) - len(self._exposed_tools)
        logger.info(
            "MCP server initialized: exposing %d tools (%d hidden by policy)",
            len(self._exposed_tools),
            hidden,
        )

    def _build_signature(self, metadata: Any) -> inspect.Signature:
        """Build a real call signature from effGen tool metadata.

        FastMCP introspects the function signature (``inspect.signature``) to
        generate the MCP input schema and to validate call arguments, so the
        wrapper must advertise the tool's actual parameters — a bare
        ``**kwargs`` would produce an empty/invalid schema.
        """
        params = [
            inspect.Parameter(
                "ctx", inspect.Parameter.KEYWORD_ONLY, annotation=Context
            )
        ]
        # Only advertise model-facing parameters: developer-only safety toggles
        # (model_facing=False) must never enter the signature FastMCP turns into
        # the schema/validation an MCP client/model sees.
        for param in metadata.model_facing_parameters:
            pytype = _PARAM_TYPE_MAP.get(param.type.value, Any)
            if param.required:
                default = inspect.Parameter.empty
            else:
                # Optional parameters are nullable with their declared default.
                pytype = pytype | None if pytype is not Any else Any
                default = param.default
            params.append(
                inspect.Parameter(
                    param.name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=pytype,
                )
            )
        return inspect.Signature(params, return_annotation=dict)

    def _register_tool(self, tool_name: str) -> None:
        """
        Register a single tool with MCP.

        Args:
            tool_name: Name of the tool to register
        """
        try:
            metadata = self.tools_registry.get_metadata(tool_name)
            registry = self.tools_registry
            signature = self._build_signature(metadata)

            # Bind ``tool_name`` as a default argument so each wrapper closes
            # over its OWN name (a plain closure over the loop variable would
            # make every tool call the last-registered tool).
            async def tool_wrapper(*, ctx: Context[ServerSession, None] = None, _tool_name=tool_name, **kwargs):
                """Wrapper function for tool execution."""
                if ctx is not None:
                    with contextlib.suppress(Exception):
                        await ctx.debug(f"Executing tool: {_tool_name}")

                # Drop unset optional arguments so tool defaults apply.
                call_args = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    tool = await registry.get_tool(_tool_name)
                    result = await tool.execute(**call_args)
                except Exception as e:
                    error_msg = f"Error executing tool {_tool_name}: {e}"
                    logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg, "metadata": {}}

                if result.success:
                    return {
                        "success": True,
                        "output": result.output,
                        "metadata": result.metadata or {},
                    }
                return {
                    "success": False,
                    "error": result.error or "Tool execution failed",
                    "metadata": result.metadata or {},
                }

            tool_wrapper.__name__ = tool_name
            tool_wrapper.__doc__ = metadata.description or f"Tool: {tool_name}"
            tool_wrapper.__signature__ = signature
            tool_wrapper.__annotations__ = {
                p.name: p.annotation for p in signature.parameters.values()
            }
            tool_wrapper.__annotations__["return"] = dict

            # Register with FastMCP. structured_output=False keeps the tool's
            # dict result as JSON text content (predictable for any MCP client).
            self.mcp.add_tool(
                tool_wrapper,
                name=tool_name,
                description=metadata.description or f"Tool: {tool_name}",
                structured_output=False,
            )

            logger.debug(f"Registered tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to register tool {tool_name}: {e}", exc_info=True)

    @staticmethod
    def _route_logs_to_stderr() -> None:
        """Ensure no logging/console handler writes to stdout.

        The stdio transport owns stdout for JSON-RPC; any stray byte written
        there (a log line, a rich panel) corrupts the protocol. Redirect every
        stdout-bound handler to stderr, which MCP clients ignore.
        """
        for name in (None, "effgen", "mcp"):
            lg = logging.getLogger(name) if name else logging.getLogger()
            for handler in lg.handlers:
                stream = getattr(handler, "stream", None)
                if stream is sys.stdout:
                    with contextlib.suppress(Exception):
                        handler.setStream(sys.stderr)
                console = getattr(handler, "console", None)  # RichHandler
                if console is not None and getattr(console, "file", None) is sys.stdout:
                    with contextlib.suppress(Exception):
                        from rich.console import Console

                        handler.console = Console(file=sys.stderr)

    async def run_stdio(self) -> None:
        """
        Run server with STDIO transport.

        This is the main entry point for Claude Desktop and other
        STDIO-based MCP clients.
        """
        # Keep stdout carrying nothing but the JSON-RPC stream.
        self._route_logs_to_stderr()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if not self._initialized:
                    await self.initialize()
                logger.info(f"Starting MCP server (STDIO): {self.name}")

            # FastMCP's async entry point — no nested event loop.
            await self.mcp.run_stdio_async()

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            with contextlib.redirect_stdout(sys.stderr):
                await self.cleanup()

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        transport: str = "streamable-http"
    ) -> None:
        """
        Run server with HTTP transport.

        Binds to 127.0.0.1 by default; binding to a non-loopback address
        exposes the server to the network and emits a warning.

        Args:
            host: Host to bind to
            port: Port to bind to
            transport: Transport type ("streamable-http" or "sse")
        """
        self._route_logs_to_stderr()
        try:
            if not self._initialized:
                await self.initialize()

            if host not in ("127.0.0.1", "localhost", "::1"):
                logger.warning(
                    "MCP HTTP server binding to non-loopback address %s — it will "
                    "be reachable from the network. Ensure auth/origin protection "
                    "is in place.",
                    host,
                )

            # Configure bind address on the FastMCP settings (run() doesn't take
            # host/port kwargs).
            self.mcp.settings.host = host
            self.mcp.settings.port = port

            logger.info(f"Starting MCP server ({transport}): {host}:{port}")

            if transport == "sse":
                await self.mcp.run_sse_async()
            else:
                await self.mcp.run_streamable_http_async()

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.tools_registry.cleanup_all()
            logger.info("Server cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    def get_mcp_instance(self) -> FastMCP:
        """
        Get the underlying FastMCP instance.

        This allows for advanced customization and direct access
        to the MCP server.

        Returns:
            FastMCP instance
        """
        return self.mcp


def create_server(
    name: str = "effgen-tools",
    version: str = "1.0.0",
    tools_registry: ToolRegistry | None = None,
    expose_unsafe_tools: bool = False,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
) -> EffGenMCPServer:
    """
    Create an EffGen MCP server instance.

    Args:
        name: Server name
        version: Server version
        tools_registry: Tool registry to expose
        expose_unsafe_tools: Expose code-execution/shell/filesystem tools
            (off by default — fail-closed).
        allowed_tools: If set, expose ONLY these tools.
        blocked_tools: Additional tool names to hide.

    Returns:
        EffGen MCP server instance
    """
    return EffGenMCPServer(
        name=name,
        version=version,
        tools_registry=tools_registry,
        expose_unsafe_tools=expose_unsafe_tools,
        allowed_tools=allowed_tools,
        blocked_tools=blocked_tools,
    )


async def main_stdio() -> None:
    """Main entry point for STDIO server."""
    server = create_server()
    await server.run_stdio()


async def main_http(
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "streamable-http"
) -> None:
    """Main entry point for HTTP server.

    Args:
        host: Address to bind.
        port: Port to bind.
        transport: The MCP HTTP transport to serve.
    """
    server = create_server()
    await server.run_http(host, port, transport)


def run_cli(argv: list[str] | None = None) -> int:
    """Launch the MCP server from the command line.

    Dispatches ``[stdio|http|sse] [host] [port]`` (default: stdio). Shared by
    the package entry point (``python -m effgen.tools.protocols.mcp_official``)
    and this module's ``__main__`` guard so both behave identically.

    Returns a process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        mode = argv[0]
        if mode in ("http", "sse"):
            host = argv[1] if len(argv) > 1 else "127.0.0.1"
            port = int(argv[2]) if len(argv) > 2 else 8000
            transport = "sse" if mode == "sse" else "streamable-http"
            asyncio.run(main_http(host, port, transport=transport))
        elif mode == "stdio":
            asyncio.run(main_stdio())
        else:
            print(
                "Usage: python -m effgen.tools.protocols.mcp_official "
                "[stdio|http|sse] [host] [port]"
            )
            return 1
    else:
        # Default to STDIO
        asyncio.run(main_stdio())
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())

