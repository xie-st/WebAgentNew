"""
Tools module for the effGen framework.

This module provides the tool integration system including base classes,
registry, built-in tools, and protocol implementations.
"""

from typing import Any

from .base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)
from .function_tool import FunctionTool, Tool, tool
from .registry import (
    ToolDependencyError,
    ToolRegistrationError,
    ToolRegistry,
    async_reset_registry,
    get_registry,
    reset_registry,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolMetadata",
    "ToolCategory",
    "ToolResult",
    "ParameterSpec",
    "ParameterType",
    # Low-boilerplate authoring (ergonomic wrappers over BaseTool)
    "tool",
    "Tool",
    "FunctionTool",
    # Registry
    "ToolRegistry",
    "ToolDependencyError",
    "ToolRegistrationError",
    "get_registry",
    "reset_registry",
    "async_reset_registry",
    # Protocols
    "protocols",
]


def __getattr__(name: str) -> Any:
    # The protocol subpackage (MCP/A2A/ACP interop) pulls heavy optional SDKs
    # (the official `mcp` SDK, jsonschema). Most agent runs never touch it, so
    # it is imported lazily on first access — `effgen.tools.protocols` and
    # `from effgen.tools import protocols` keep working unchanged.
    if name == "protocols":
        from importlib import import_module
        module = import_module("effgen.tools.protocols")
        globals()["protocols"] = module
        return module
    raise AttributeError(f"module 'effgen.tools' has no attribute {name!r}")
