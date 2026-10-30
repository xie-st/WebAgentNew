"""
Agent Communication Protocol (ACP) handler.

This module implements IBM's ACP protocol specification for agent-to-agent
communication, including agent manifests, synchronous/asynchronous requests,
task tracking, and OpenTelemetry instrumentation.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


logger = logging.getLogger(__name__)


class ACPVersion(Enum):
    """Supported ACP protocol versions."""
    V1_0 = "1.0"


class RequestType(Enum):
    """ACP request types."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"


class TaskStatus(Enum):
    """ACP task status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ACPError:
    """
    ACP error representation.

    Attributes:
        code: Error code
        message: Error message
        severity: Error severity
        details: Additional error details
        timestamp: Error timestamp
    """
    code: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    details: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
        }
        if self.details:
            result["details"] = self.details
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ACPError":
        """Create from dictionary."""
        return cls(
            code=data["code"],
            message=data["message"],
            severity=ErrorSeverity(data.get("severity", "error")),
            details=data.get("details"),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
        )


@dataclass
class SchemaDefinition:
    """
    JSON Schema definition for ACP.

    Attributes:
        type: Schema type
        properties: Schema properties
        required: Required properties
        description: Schema description
        additionalProperties: Whether additional properties are allowed
    """
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    description: str | None = None
    additionalProperties: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type,
            "properties": self.properties,
            "additionalProperties": self.additionalProperties,
        }
        if self.required:
            result["required"] = self.required
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaDefinition":
        """Create from dictionary."""
        return cls(
            type=data["type"],
            properties=data.get("properties", {}),
            required=data.get("required", []),
            description=data.get("description"),
            additionalProperties=data.get("additionalProperties", False),
        )


@dataclass
class CapabilityDefinition:
    """
    ACP capability definition.

    Attributes:
        name: Capability name
        description: Capability description
        inputSchema: Input schema definition
        outputSchema: Output schema definition
        metadata: Additional metadata
        version: Capability version
    """
    name: str
    description: str
    inputSchema: SchemaDefinition
    outputSchema: SchemaDefinition | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema.to_dict(),
            "version": self.version,
        }
        if self.outputSchema:
            result["outputSchema"] = self.outputSchema.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityDefinition":
        """Create from dictionary."""
        output_schema = None
        if "outputSchema" in data:
            output_schema = SchemaDefinition.from_dict(data["outputSchema"])

        return cls(
            name=data["name"],
            description=data["description"],
            inputSchema=SchemaDefinition.from_dict(data["inputSchema"]),
            outputSchema=output_schema,
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0"),
        )


@dataclass
class AgentManifest:
    """
    ACP Agent Manifest for capability advertisement.

    Attributes:
        agentId: Unique agent identifier
        name: Agent name
        version: Agent version
        description: Agent description
        capabilities: List of capabilities
        metadata: Additional metadata
        created: Creation timestamp
        updated: Last update timestamp
    """
    agentId: str
    name: str
    version: str
    description: str
    capabilities: list[CapabilityDefinition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agentId": self.agentId,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "metadata": self.metadata,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentManifest":
        """Create from dictionary."""
        return cls(
            agentId=data["agentId"],
            name=data["name"],
            version=data["version"],
            description=data["description"],
            capabilities=[
                CapabilityDefinition.from_dict(cap)
                for cap in data.get("capabilities", [])
            ],
            metadata=data.get("metadata", {}),
            created=data.get("created", datetime.now(UTC).isoformat()),
            updated=data.get("updated", datetime.now(UTC).isoformat()),
        )

    def to_json(self, indent: int | None = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentManifest":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_capability(self, name: str) -> CapabilityDefinition | None:
        """
        Get a capability by name.

        Args:
            name: Capability name

        Returns:
            Capability if found, None otherwise
        """
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    def add_capability(self, capability: CapabilityDefinition) -> None:
        """
        Add a capability to the manifest.

        Args:
            capability: Capability to add
        """
        if not self.get_capability(capability.name):
            self.capabilities.append(capability)
            self.updated = datetime.now(UTC).isoformat()


@dataclass
class CapabilityToken:
    """
    Capability token for fine-grained permissions.

    Attributes:
        tokenId: Token identifier
        agentId: Agent identifier
        capabilities: Allowed capabilities
        permissions: Permission levels per capability
        expires: Expiration timestamp
        metadata: Additional metadata
    """
    tokenId: str
    agentId: str
    capabilities: list[str]
    permissions: dict[str, list[str]] = field(default_factory=dict)
    expires: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "tokenId": self.tokenId,
            "agentId": self.agentId,
            "capabilities": self.capabilities,
            "permissions": self.permissions,
        }
        if self.expires:
            result["expires"] = self.expires
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityToken":
        """Create from dictionary."""
        return cls(
            tokenId=data["tokenId"],
            agentId=data["agentId"],
            capabilities=data.get("capabilities", []),
            permissions=data.get("permissions", {}),
            expires=data.get("expires"),
            metadata=data.get("metadata", {}),
        )

    def is_valid(self) -> bool:
        """Check if token is still valid."""
        if not self.expires:
            return True
        expiry = datetime.fromisoformat(self.expires)
        # Normalise: if expiry is naive, treat as UTC
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return datetime.now(UTC) < expiry

    def has_capability(self, capability: str) -> bool:
        """
        Check if token grants access to a capability.

        Args:
            capability: Capability name

        Returns:
            True if capability is allowed
        """
        return capability in self.capabilities


@dataclass
class ACPRequest:
    """
    ACP request message.

    Attributes:
        requestId: Unique request identifier
        agentId: Target agent identifier
        capability: Requested capability
        input: Request input data
        requestType: Type of request (sync/async/streaming)
        context: Request context
        metadata: Additional metadata
        timestamp: Request timestamp
    """
    capability: str
    input: dict[str, Any]
    requestId: str = field(default_factory=lambda: str(uuid.uuid4()))
    agentId: str | None = None
    requestType: RequestType = RequestType.SYNCHRONOUS
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "requestId": self.requestId,
            "capability": self.capability,
            "input": self.input,
            "requestType": self.requestType.value,
            "timestamp": self.timestamp,
        }
        if self.agentId:
            result["agentId"] = self.agentId
        if self.context:
            result["context"] = self.context
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ACPRequest":
        """Create from dictionary."""
        return cls(
            requestId=data.get("requestId", str(uuid.uuid4())),
            agentId=data.get("agentId"),
            capability=data["capability"],
            input=data["input"],
            requestType=RequestType(data.get("requestType", "synchronous")),
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
        )


@dataclass
class ACPResponse:
    """
    ACP response message.

    Attributes:
        requestId: Request identifier
        output: Response output data
        status: Task status
        error: Error information if failed
        metadata: Additional metadata
        timestamp: Response timestamp
    """
    requestId: str
    status: TaskStatus
    output: dict[str, Any] | None = None
    error: ACPError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "requestId": self.requestId,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }
        if self.output:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ACPResponse":
        """Create from dictionary."""
        error = None
        if "error" in data:
            error = ACPError.from_dict(data["error"])

        return cls(
            requestId=data["requestId"],
            status=TaskStatus(data["status"]),
            output=data.get("output"),
            error=error,
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
        )


@dataclass
class TaskInfo:
    """
    Task tracking information for asynchronous requests.

    Attributes:
        taskId: Task identifier
        requestId: Associated request ID
        status: Task status
        progress: Task progress (0.0 to 1.0)
        result: Task result when completed
        error: Error information if failed
        created: Creation timestamp
        updated: Last update timestamp
        completed: Completion timestamp
    """
    taskId: str
    requestId: str
    status: TaskStatus
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: ACPError | None = None
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "taskId": self.taskId,
            "requestId": self.requestId,
            "status": self.status.value,
            "progress": self.progress,
            "created": self.created,
            "updated": self.updated,
        }
        if self.result:
            result["result"] = self.result
        if self.error:
            result["error"] = self.error.to_dict()
        if self.completed:
            result["completed"] = self.completed
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInfo":
        """Create from dictionary."""
        error = None
        if "error" in data:
            error = ACPError.from_dict(data["error"])

        return cls(
            taskId=data["taskId"],
            requestId=data["requestId"],
            status=TaskStatus(data["status"]),
            progress=data.get("progress", 0.0),
            result=data.get("result"),
            error=error,
            created=data.get("created", datetime.now(UTC).isoformat()),
            updated=data.get("updated", datetime.now(UTC).isoformat()),
            completed=data.get("completed"),
        )

    def update_progress(self, progress: float) -> None:
        """Update task progress."""
        self.progress = max(0.0, min(1.0, progress))
        self.updated = datetime.now(UTC).isoformat()

    def complete(self, result: dict[str, Any]) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.progress = 1.0
        self.completed = datetime.now(UTC).isoformat()
        self.updated = self.completed

    def fail(self, error: ACPError) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed = datetime.now(UTC).isoformat()
        self.updated = self.completed


class ACPProtocolHandler:
    """
    Handler for ACP protocol operations.

    Implements IBM's Agent Communication Protocol for request handling,
    task tracking, and manifest management.
    """

    def __init__(self, manifest: AgentManifest) -> None:
        """
        Initialize ACP protocol handler.

        Args:
            manifest: Agent manifest
        """
        self.manifest = manifest
        self._tasks: dict[str, TaskInfo] = {}

    def create_request(
        self,
        capability: str,
        input_data: dict[str, Any],
        agent_id: str | None = None,
        request_type: RequestType = RequestType.SYNCHRONOUS,
        context: dict[str, Any] | None = None,
    ) -> ACPRequest:
        """
        Create an ACP request.

        Args:
            capability: Requested capability
            input_data: Request input
            agent_id: Target agent ID
            request_type: Type of request
            context: Request context

        Returns:
            ACP request
        """
        return ACPRequest(
            capability=capability,
            input=input_data,
            agentId=agent_id,
            requestType=request_type,
            context=context or {},
        )

    def create_response(
        self,
        request_id: str,
        status: TaskStatus,
        output: dict[str, Any] | None = None,
        error: ACPError | None = None,
    ) -> ACPResponse:
        """
        Create an ACP response.

        Args:
            request_id: Request identifier
            status: Task status
            output: Response output
            error: Error information

        Returns:
            ACP response
        """
        return ACPResponse(
            requestId=request_id,
            status=status,
            output=output,
            error=error,
        )

    def create_task(self, request: ACPRequest) -> TaskInfo:
        """
        Create a task for asynchronous request.

        Args:
            request: ACP request

        Returns:
            Task info
        """
        task = TaskInfo(
            taskId=str(uuid.uuid4()),
            requestId=request.requestId,
            status=TaskStatus.PENDING,
        )
        self._tasks[task.taskId] = task
        logger.info(f"Created task {task.taskId} for request {request.requestId}")
        return task

    def get_task(self, task_id: str) -> TaskInfo | None:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task info if found
        """
        return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: TaskStatus | None = None,
        progress: float | None = None,
        result: dict[str, Any] | None = None,
        error: ACPError | None = None,
    ) -> bool:
        """
        Update task status.

        Args:
            task_id: Task identifier
            status: New status
            progress: Progress update
            result: Result if completed
            error: Error if failed

        Returns:
            True if task was updated
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if status:
            task.status = status
            task.updated = datetime.now(UTC).isoformat()
        if progress is not None:
            task.update_progress(progress)
        if result:
            task.complete(result)
        if error:
            task.fail(error)

        logger.info(f"Updated task {task_id}: status={task.status.value}")
        return True

    def validate_request(self, request: ACPRequest) -> tuple[bool, str | None]:
        """
        Validate an ACP request against capability schema.

        Uses jsonschema for full JSON Schema validation including types,
        patterns, enums, min/max values. Falls back to basic required-field
        checks if jsonschema is not available.

        Args:
            request: Request to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if capability exists
        capability = self.manifest.get_capability(request.capability)
        if not capability:
            return False, f"Capability not found: {request.capability}"

        # Build JSON Schema dict from SchemaDefinition
        schema = capability.inputSchema.to_dict()

        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=request.input, schema=schema)
                return True, None
            except jsonschema.ValidationError as e:
                # Build a human-readable error with path info
                path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "(root)"
                return False, f"Validation error at {path}: {e.message}"
            except jsonschema.SchemaError as e:
                return False, f"Invalid schema: {e.message}"
        else:
            # Fallback: basic required-field checks
            logger.warning(
                "jsonschema not installed — using basic validation only. "
                "Install with: pip install jsonschema"
            )
            required = capability.inputSchema.required
            for req_field in required:
                if req_field not in request.input:
                    return False, f"Missing required field: {req_field}"

            # Basic type checking from schema properties
            properties = capability.inputSchema.properties
            for prop_name, prop_schema in properties.items():
                if prop_name in request.input:
                    expected_type = prop_schema.get("type")
                    value = request.input[prop_name]
                    if expected_type and not self._check_basic_type(value, expected_type):
                        return False, (
                            f"Field '{prop_name}' expected type '{expected_type}', "
                            f"got '{type(value).__name__}'"
                        )

            return True, None

    @staticmethod
    def _check_basic_type(value: Any, expected_type: str) -> bool:
        """Basic JSON Schema type check (fallback when jsonschema unavailable)."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, accept
        return isinstance(value, expected)

    def create_error(
        self,
        code: str,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: dict[str, Any] | None = None,
    ) -> ACPError:
        """
        Create an ACP error.

        Args:
            code: Error code
            message: Error message
            severity: Error severity
            details: Additional details

        Returns:
            ACP error
        """
        return ACPError(
            code=code,
            message=message,
            severity=severity,
            details=details,
        )
