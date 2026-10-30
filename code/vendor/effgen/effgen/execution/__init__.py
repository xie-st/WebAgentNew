"""
Execution and sandbox systems for effGen.

This package runs code with multi-language support, resource limits, and static
code validation. Isolation depends on the backend: ``DockerSandbox`` confines
the filesystem and network in a container; ``LocalSandbox`` runs code in a
subprocess screened by an AST/regex validator, which is a static check rather
than an operating-system boundary and does not isolate the host. To run code an
untrusted model produced with the project's hardened isolation, use the
registered ``code_executor`` tool (``effgen.security.sandbox``); see
``effgen/execution/sandbox.py`` for the details.
"""

from .docker_sandbox import DOCKER_AVAILABLE, DockerManager, DockerSandbox
from .sandbox import (
    BaseSandbox,
    CodeExecutor,
    CodeExecutorWithHistory,
    ExecutionHistory,
    ExecutionPool,
    ExecutionResult,
    ExecutionStatus,
    LocalSandbox,
    SandboxConfig,
)
from .validators import (
    BashValidator,
    CodeValidator,
    JavaScriptValidator,
    PythonValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

__all__ = [
    # Sandbox
    "CodeExecutor",
    "BaseSandbox",
    "LocalSandbox",
    "SandboxConfig",
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionPool",
    "ExecutionHistory",
    "CodeExecutorWithHistory",

    # Docker sandbox
    "DockerSandbox",
    "DockerManager",
    "DOCKER_AVAILABLE",

    # Validators
    "CodeValidator",
    "PythonValidator",
    "JavaScriptValidator",
    "BashValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
]
