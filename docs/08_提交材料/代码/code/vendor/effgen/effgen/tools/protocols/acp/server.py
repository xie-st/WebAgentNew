"""
ACP server implementation for exposing effGen via ACP protocol.

This module provides server functionality to expose effGen capabilities
through IBM's Agent Communication Protocol (ACP), including BeeAI platform
integration and manifest generation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from typing import Any

from .protocol import (
    ACPProtocolHandler,
    ACPRequest,
    ACPResponse,
    AgentManifest,
    CapabilityDefinition,
    CapabilityToken,
    ErrorSeverity,
    RequestType,
    SchemaDefinition,
    TaskInfo,
    TaskStatus,
)

# Optional, only needed when running the HTTP server. Imported at module level
# (not just inside create_app) so the string annotations produced by
# ``from __future__ import annotations`` on the route handlers resolve to the
# real FastAPI types — otherwise FastAPI mistakes ``request: Request`` for a
# query parameter and rejects every /execute call with HTTP 422.
try:  # pragma: no cover - import guard
    from fastapi import HTTPException, Request
except Exception:  # pragma: no cover - fastapi is an optional extra
    HTTPException = None  # type: ignore[assignment, misc]
    Request = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


# Type alias for capability handlers
CapabilityHandler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ACPServerConfig:
    """
    Configuration for ACP server.

    Attributes:
        host: Server host address
        port: Server port
        enable_telemetry: Enable OpenTelemetry instrumentation
        enable_cors: Enable CORS support
        cors_origins: Allowed CORS origins
        max_concurrent_tasks: Maximum concurrent tasks
        task_timeout: Default task timeout in seconds
        enable_streaming: Enable streaming responses
        require_auth: Require authentication
    """
    # Bind to loopback by default. Exposing the server to the network requires
    # an explicit non-loopback host (and emits a warning on run()).
    host: str = "127.0.0.1"
    port: int = 8080
    enable_telemetry: bool = True
    enable_cors: bool = True
    # No cross-origin access by default. Set explicit origins to opt in; a
    # wildcard ("*") is honored but warned about when binding non-loopback.
    cors_origins: list[str] = None
    max_concurrent_tasks: int = 100
    task_timeout: int = 300
    enable_streaming: bool = True
    require_auth: bool = False

    def __post_init__(self) -> None:
        if self.cors_origins is None:
            self.cors_origins = []


class ACPCapabilityRegistry:
    """
    Registry for ACP capabilities and their handlers.

    Manages capability definitions and execution handlers for the ACP server.
    """

    def __init__(self) -> None:
        """Initialize capability registry."""
        self._capabilities: dict[str, CapabilityDefinition] = {}
        self._handlers: dict[str, CapabilityHandler] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        handler: CapabilityHandler | None = None,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a capability.

        Args:
            name: Capability name
            description: Capability description
            input_schema: JSON schema for input
            output_schema: JSON schema for output
            handler: Handler function for the capability
            version: Capability version
            metadata: Additional metadata
        """
        # Create schema definitions. Follow JSON Schema semantics for
        # additionalProperties (default: allowed) instead of silently locking
        # the schema closed — a capability declared as {"type": "object"} with
        # no properties must still accept input, not reject every call.
        input_def = SchemaDefinition(
            type=input_schema.get("type", "object"),
            properties=input_schema.get("properties", {}),
            required=input_schema.get("required", []),
            description=input_schema.get("description"),
            additionalProperties=input_schema.get("additionalProperties", True),
        )

        output_def = None
        if output_schema:
            output_def = SchemaDefinition(
                type=output_schema.get("type", "object"),
                properties=output_schema.get("properties", {}),
                required=output_schema.get("required", []),
                description=output_schema.get("description"),
                additionalProperties=output_schema.get("additionalProperties", True),
            )

        # Create capability definition
        capability = CapabilityDefinition(
            name=name,
            description=description,
            inputSchema=input_def,
            outputSchema=output_def,
            version=version,
            metadata=metadata or {},
        )

        self._capabilities[name] = capability
        if handler:
            self._handlers[name] = handler

        logger.info(f"Registered capability: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a capability.

        Args:
            name: Capability name

        Returns:
            True if capability was removed
        """
        if name in self._capabilities:
            del self._capabilities[name]
            if name in self._handlers:
                del self._handlers[name]
            logger.info(f"Unregistered capability: {name}")
            return True
        return False

    def get_capability(self, name: str) -> CapabilityDefinition | None:
        """
        Get a capability definition.

        Args:
            name: Capability name

        Returns:
            Capability definition if found
        """
        return self._capabilities.get(name)

    def get_handler(self, name: str) -> CapabilityHandler | None:
        """
        Get a capability handler.

        Args:
            name: Capability name

        Returns:
            Handler function if found
        """
        return self._handlers.get(name)

    def list_capabilities(self) -> list[CapabilityDefinition]:
        """
        List all registered capabilities.

        Returns:
            List of capability definitions
        """
        return list(self._capabilities.values())

    def has_capability(self, name: str) -> bool:
        """
        Check if a capability exists.

        Args:
            name: Capability name

        Returns:
            True if capability exists
        """
        return name in self._capabilities


def capability(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
    version: str = "1.0.0",
    metadata: dict[str, Any] | None = None,
) -> Callable[[CapabilityHandler], CapabilityHandler]:
    """
    Decorator for registering capability handlers.

    Args:
        name: Capability name
        description: Capability description
        input_schema: Input JSON schema
        output_schema: Output JSON schema
        version: Capability version
        metadata: Additional metadata

    Returns:
        A decorator that registers the handler it wraps and returns it
        unchanged, so the decorated function stays directly callable.

    Example:
        @capability(
            name="summarize_text",
            description="Summarize text content",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_length": {"type": "integer"}
                },
                "required": ["text"]
            }
        )
        async def summarize(input_data, context):
            text = input_data["text"]
            # ... summarization logic ...
            return {"summary": result}
    """
    def decorator(func: CapabilityHandler):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Store capability metadata on function
        wrapper._acp_capability = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "version": version,
            "metadata": metadata,
        }
        return wrapper

    return decorator


class ACPServer:
    """
    ACP server for exposing effGen capabilities.

    Features:
    - Synchronous and asynchronous request handling
    - Streaming response support
    - Task management and tracking
    - Capability token validation
    - BeeAI platform integration
    - OpenTelemetry instrumentation
    - Automatic manifest generation
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        version: str,
        description: str,
        config: ACPServerConfig | None = None,
    ) -> None:
        """
        Initialize ACP server.

        Args:
            agent_id: Unique agent identifier
            name: Agent name
            version: Agent version
            description: Agent description
            config: Server configuration
        """
        self.agent_id = agent_id
        self.name = name
        self.version = version
        self.description = description
        self.config = config or ACPServerConfig()

        # Initialize components
        self.registry = ACPCapabilityRegistry()
        self.manifest = self._create_manifest()
        self.protocol = ACPProtocolHandler(self.manifest)

        # Task management
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._task_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        # Telemetry
        self._tracer = None
        if self.config.enable_telemetry:
            try:
                from opentelemetry import trace
                self._tracer = trace.get_tracer(__name__)
            except ImportError:
                logger.warning("OpenTelemetry not available, telemetry disabled")

        # Auth tokens
        self._valid_tokens: dict[str, CapabilityToken] = {}

    def _create_manifest(self) -> AgentManifest:
        """Create agent manifest."""
        return AgentManifest(
            agentId=self.agent_id,
            name=self.name,
            version=self.version,
            description=self.description,
        )

    def _update_manifest(self) -> None:
        """Update manifest with current capabilities."""
        self.manifest.capabilities = self.registry.list_capabilities()
        self.manifest.updated = datetime.now(UTC).isoformat()

    def register_capability(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        handler: CapabilityHandler | None = None,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a capability with the server.

        Args:
            name: Capability name
            description: Capability description
            input_schema: Input JSON schema
            output_schema: Output JSON schema
            handler: Handler function
            version: Capability version
            metadata: Additional metadata
        """
        self.registry.register(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            handler=handler,
            version=version,
            metadata=metadata,
        )
        self._update_manifest()

    def register_handler(self, func: CapabilityHandler) -> None:
        """
        Register a decorated capability handler.

        Args:
            func: Decorated handler function
        """
        if not hasattr(func, "_acp_capability"):
            raise ValueError("Function must be decorated with @capability")

        meta = func._acp_capability
        self.register_capability(
            name=meta["name"],
            description=meta["description"],
            input_schema=meta["input_schema"],
            output_schema=meta["output_schema"],
            handler=func,
            version=meta["version"],
            metadata=meta["metadata"],
        )

    def add_capability_token(self, token: CapabilityToken) -> None:
        """
        Add a valid capability token.

        Args:
            token: Capability token
        """
        self._valid_tokens[token.tokenId] = token
        logger.info(f"Added capability token {token.tokenId}")

    def validate_token(self, token_id: str, capability: str) -> bool:
        """
        Validate a capability token.

        Args:
            token_id: Token identifier
            capability: Required capability

        Returns:
            True if token is valid for the capability
        """
        token = self._valid_tokens.get(token_id)
        if not token:
            return False
        if not token.is_valid():
            return False
        if not token.has_capability(capability):
            return False
        return True

    async def handle_request(
        self,
        request: ACPRequest,
        token_id: str | None = None,
    ) -> ACPResponse:
        """
        Handle an ACP request.

        Args:
            request: ACP request
            token_id: Optional capability token ID

        Returns:
            ACP response

        Raises:
            ValueError: If request is invalid
            RuntimeError: If execution fails
        """
        # Start telemetry span
        span_context = None
        if self._tracer:
            span_context = self._tracer.start_span(
                "acp.server.handle_request",
                attributes={
                    "acp.request_id": request.requestId,
                    "acp.capability": request.capability,
                    "acp.request_type": request.requestType.value,
                }
            )

        try:
            # Validate token if auth required
            if self.config.require_auth:
                if not token_id or not self.validate_token(token_id, request.capability):
                    error = self.protocol.create_error(
                        code="UNAUTHORIZED",
                        message="Invalid or missing capability token",
                        severity=ErrorSeverity.ERROR,
                    )
                    return self.protocol.create_response(
                        request_id=request.requestId,
                        status=TaskStatus.FAILED,
                        error=error,
                    )

            # Validate request
            is_valid, error_msg = self.protocol.validate_request(request)
            if not is_valid:
                error = self.protocol.create_error(
                    code="INVALID_REQUEST",
                    message=error_msg,
                    severity=ErrorSeverity.ERROR,
                )
                return self.protocol.create_response(
                    request_id=request.requestId,
                    status=TaskStatus.FAILED,
                    error=error,
                )

            # Handle based on request type
            if request.requestType == RequestType.SYNCHRONOUS:
                result = await self._execute_sync(request)
                return self.protocol.create_response(
                    request_id=request.requestId,
                    status=TaskStatus.COMPLETED,
                    output=result,
                )

            elif request.requestType == RequestType.ASYNCHRONOUS:
                task = await self._execute_async(request)
                return self.protocol.create_response(
                    request_id=request.requestId,
                    status=TaskStatus.PENDING,
                    output={"taskId": task.taskId},
                )

            else:
                error = self.protocol.create_error(
                    code="UNSUPPORTED_REQUEST_TYPE",
                    message=f"Request type not supported: {request.requestType.value}",
                    severity=ErrorSeverity.ERROR,
                )
                return self.protocol.create_response(
                    request_id=request.requestId,
                    status=TaskStatus.FAILED,
                    error=error,
                )

        except Exception as e:
            logger.error(f"Error handling request {request.requestId}: {e}", exc_info=True)
            if span_context:
                span_context.record_exception(e)

            error = self.protocol.create_error(
                code="INTERNAL_ERROR",
                message=str(e),
                severity=ErrorSeverity.ERROR,
            )
            return self.protocol.create_response(
                request_id=request.requestId,
                status=TaskStatus.FAILED,
                error=error,
            )

        finally:
            if span_context:
                span_context.end()

    async def _execute_sync(self, request: ACPRequest) -> dict[str, Any]:
        """
        Execute a synchronous request.

        Args:
            request: ACP request

        Returns:
            Execution result

        Raises:
            RuntimeError: If execution fails
        """
        handler = self.registry.get_handler(request.capability)
        if not handler:
            raise RuntimeError(f"No handler for capability: {request.capability}")

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                handler(request.input, request.context),
                timeout=self.config.task_timeout,
            )
            return result

        except TimeoutError:
            raise RuntimeError(f"Execution timed out after {self.config.task_timeout}s")

    async def _execute_async(self, request: ACPRequest) -> TaskInfo:
        """
        Execute an asynchronous request.

        Args:
            request: ACP request

        Returns:
            Task information

        Raises:
            RuntimeError: If execution fails
        """
        # Create task
        task = self.protocol.create_task(request)

        # Start background execution
        async def execute():
            async with self._task_semaphore:
                try:
                    # Update status to running
                    self.protocol.update_task(
                        task.taskId,
                        status=TaskStatus.RUNNING,
                    )

                    # Execute handler
                    handler = self.registry.get_handler(request.capability)
                    if not handler:
                        raise RuntimeError(f"No handler for capability: {request.capability}")

                    result = await asyncio.wait_for(
                        handler(request.input, request.context),
                        timeout=self.config.task_timeout,
                    )

                    # Update task with result
                    self.protocol.update_task(
                        task.taskId,
                        status=TaskStatus.COMPLETED,
                        result=result,
                    )

                except TimeoutError:
                    error = self.protocol.create_error(
                        code="TIMEOUT",
                        message=f"Task timed out after {self.config.task_timeout}s",
                        severity=ErrorSeverity.ERROR,
                    )
                    self.protocol.update_task(
                        task.taskId,
                        error=error,
                    )

                except Exception as e:
                    logger.error(f"Task {task.taskId} failed: {e}", exc_info=True)
                    error = self.protocol.create_error(
                        code="EXECUTION_ERROR",
                        message=str(e),
                        severity=ErrorSeverity.ERROR,
                    )
                    self.protocol.update_task(
                        task.taskId,
                        error=error,
                    )

                finally:
                    # Remove from active tasks
                    if task.taskId in self._active_tasks:
                        del self._active_tasks[task.taskId]

        # Start task
        bg_task = asyncio.create_task(execute())
        self._active_tasks[task.taskId] = bg_task

        return task

    def get_task_status(self, task_id: str) -> TaskInfo | None:
        """
        Get task status.

        Args:
            task_id: Task identifier

        Returns:
            Task information if found
        """
        return self.protocol.get_task(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled
        """
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.cancel()

            # Update task status
            self.protocol.update_task(
                task_id,
                status=TaskStatus.CANCELLED,
            )

            logger.info(f"Cancelled task {task_id}")
            return True

        return False

    def get_manifest(self) -> AgentManifest:
        """
        Get agent manifest.

        Returns:
            Agent manifest
        """
        self._update_manifest()
        return self.manifest

    def export_manifest_json(self, indent: int | None = 2) -> str:
        """
        Export manifest as JSON.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return self.get_manifest().to_json(indent=indent)

    async def register_with_beeai(
        self,
        beeai_url: str,
        api_key: str,
        endpoint: str,
    ) -> bool:
        """
        Register agent with BeeAI platform.

        Args:
            beeai_url: BeeAI platform URL
            api_key: BeeAI API key
            endpoint: This agent's endpoint URL

        Returns:
            True if registration successful

        Raises:
            RuntimeError: If registration fails
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                manifest = self.get_manifest()

                # Prepare registration data
                data = {
                    "manifest": manifest.to_dict(),
                    "endpoint": endpoint,
                    "protocol": "acp",
                    "version": "1.0",
                }

                # Send registration request
                response = await client.post(
                    f"{beeai_url}/api/agents/register",
                    json=data,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

                logger.info(f"Registered agent {self.name} with BeeAI platform")
                return True

        except Exception as e:
            raise RuntimeError(f"BeeAI registration failed: {e}")

    async def unregister_from_beeai(
        self,
        beeai_url: str,
        api_key: str,
    ) -> bool:
        """
        Unregister agent from BeeAI platform.

        Args:
            beeai_url: BeeAI platform URL
            api_key: BeeAI API key

        Returns:
            True if unregistration successful

        Raises:
            RuntimeError: If unregistration fails
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{beeai_url}/api/agents/{self.agent_id}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                response.raise_for_status()

                logger.info(f"Unregistered agent {self.name} from BeeAI platform")
                return True

        except Exception as e:
            raise RuntimeError(f"BeeAI unregistration failed: {e}")

    def create_app(self) -> Any:
        """
        Create a FastAPI application exposing ACP endpoints.

        Endpoints:
            GET  /manifest        — Agent manifest
            GET  /health          — Health check
            POST /execute         — Execute capability (sync or async)
            GET  /tasks/{task_id} — Get task status
            DELETE /tasks/{task_id} — Cancel task

        Returns:
            FastAPI application instance

        Raises:
            ImportError: If fastapi/uvicorn not installed
        """
        try:
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError(
                "FastAPI is required for ACP server mode. "
                "Install with: pip install fastapi uvicorn\n"
                "⚠️  FastAPI server mode requires fastapi (free, open source)."
            )

        app = FastAPI(
            title=f"ACP Agent: {self.name}",
            version=self.version,
            description=self.description,
        )

        # CORS (only when explicit origins are configured — no wildcard default)
        if self.config.enable_cors and self.config.cors_origins:
            if "*" in self.config.cors_origins:
                logger.warning(
                    "ACP server CORS configured with a wildcard origin '*' — any "
                    "website can call this server. Prefer explicit origins."
                )
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        server = self  # capture for closures

        @app.get("/manifest")
        async def get_manifest():
            return server.get_manifest().to_dict()

        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "agent": server.name,
                "version": server.version,
                "capabilities": len(server.registry.list_capabilities()),
            }

        @app.post("/execute")
        async def execute(request: Request):
            body = await request.json()
            try:
                acp_request = ACPRequest.from_dict(body)
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

            token_id = request.headers.get("X-Capability-Token")
            response = await server.handle_request(acp_request, token_id=token_id)
            return response.to_dict()

        @app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            task = server.get_task_status(task_id)
            if not task:
                raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
            return task.to_dict()

        @app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            cancelled = await server.cancel_task(task_id)
            if not cancelled:
                raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
            return {"status": "cancelled", "taskId": task_id}

        return app

    def run(self, **kwargs: Any) -> None:
        """
        Run the ACP server using uvicorn.

        Args:
            **kwargs: Additional uvicorn.run() arguments
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run the ACP server. "
                "Install with: pip install uvicorn"
            )

        host = kwargs.pop("host", self.config.host)
        port = kwargs.pop("port", self.config.port)
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "ACP server binding to non-loopback address %s — it will be "
                "reachable from the network. Enable require_auth and restrict "
                "cors_origins before exposing it.",
                host,
            )
            if not self.config.require_auth:
                logger.warning(
                    "ACP server is exposed to the network with require_auth=False "
                    "— requests are unauthenticated."
                )

        app = self.create_app()
        uvicorn.run(app, host=host, port=port, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return f"<ACPServer(name='{self.name}', version='{self.version}', capabilities={len(self.registry.list_capabilities())})>"
