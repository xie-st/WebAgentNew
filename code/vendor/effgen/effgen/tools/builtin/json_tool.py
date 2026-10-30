"""
JSON processing tool for parsing, querying, and transforming JSON data.

Provides JSONPath-like querying, validation, and transformation
without any external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class JSONTool(BaseTool):
    """
    JSON processing tool.

    Features:
    - Parse JSON strings
    - Query with JSONPath-like expressions ($.key.subkey, $.array[0])
    - Get keys, values, length
    - Validate JSON structure
    - Pretty-print / format JSON
    - No external dependencies
    """

    # Natural-name aliases for the ``operation`` selector.
    operation_aliases = {
        "parse": "query",
        "load": "query",
        "get": "query",
        "extract": "query",
        "stringify": "format",
        "dumps": "format",
        "pretty": "format",
        "prettify": "format",
        "minify": "format",
        "check": "validate",
        "is_valid": "validate",
        "count": "length",
        "size": "length",
        "fields": "keys",
    }

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="json_tool",
                description=(
                    "Process JSON data: parse strings, query with path expressions "
                    "(e.g., $.users[0].name), validate, format, and extract information."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="data",
                        type=ParameterType.STRING,
                        description="JSON string to process",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform",
                        required=False,
                        default="query",
                        enum=["query", "validate", "format", "keys", "length"],
                    ),
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="JSONPath-like query (e.g., $.users[0].name, $.items[*].id)",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "result": {"type": "any"},
                        "type": {"type": "string"},
                    },
                },
                timeout_seconds=5,
                tags=["json", "data", "processing", "query"],
                examples=[
                    {
                        "data": '{"users": [{"name": "Alice"}, {"name": "Bob"}]}',
                        "query": "$.users[0].name",
                        "output": {"result": "Alice"},
                    },
                    {
                        "data": '{"count": 42}',
                        "operation": "keys",
                        "output": {"result": ["count"]},
                    },
                ],
            )
        )

    def _resolve_path(self, data: Any, path: str) -> Any:
        """
        Resolve a JSONPath-like expression against parsed data.

        Supports:
        - $.key — object key access
        - $.key.subkey — nested access
        - $.array[0] — array index
        - $.array[*] — all array elements
        - $.array[-1] — negative indexing
        """
        # Strip leading $. or $
        path = path.strip()
        if path == "$":
            return data
        if path.startswith("$."):
            path = path[2:]
        elif path.startswith("$"):
            path = path[1:]

        current = data
        # Tokenize: split on dots but respect bracket notation
        tokens = re.findall(r'[^.\[\]]+|\[\d+\]|\[\-\d+\]|\[\*\]', path)

        for token in tokens:
            if current is None:
                return None

            # Array index: [0], [-1]
            idx_match = re.match(r'^\[(-?\d+)\]$', token)
            if idx_match:
                idx = int(idx_match.group(1))
                if isinstance(current, list | tuple):
                    if abs(idx) <= len(current):
                        current = current[idx]
                    else:
                        return None
                else:
                    return None
                continue

            # Wildcard: [*]
            if token == "[*]":
                if isinstance(current, list | tuple):
                    return list(current)
                return None

            # Bare number (from path like .0)
            if token.isdigit():
                if isinstance(current, list | tuple):
                    idx = int(token)
                    if idx < len(current):
                        current = current[idx]
                    else:
                        return None
                elif isinstance(current, dict) and token in current:
                    current = current[token]
                else:
                    return None
                continue

            # Key access
            if isinstance(current, dict):
                if token in current:
                    current = current[token]
                else:
                    return None
            else:
                return None

        return current

    async def _execute(
        self,
        data: str,
        operation: str = "query",
        query: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute JSON operation."""
        # Parse JSON
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            if operation == "validate":
                return {"valid": False, "error": str(e)}
            raise ValueError(f"Invalid JSON: {e}")

        if operation == "validate":
            return {"valid": True, "type": type(parsed).__name__}

        if operation == "format":
            return {"result": json.dumps(parsed, indent=2), "type": type(parsed).__name__}

        if operation == "keys":
            if isinstance(parsed, dict):
                return {"result": list(parsed.keys()), "type": "object"}
            elif isinstance(parsed, list):
                return {"result": list(range(len(parsed))), "type": "array"}
            return {"result": [], "type": type(parsed).__name__}

        if operation == "length":
            if isinstance(parsed, dict | list | str):
                return {"result": len(parsed), "type": type(parsed).__name__}
            return {"result": 1, "type": type(parsed).__name__}

        # Default: query
        if query:
            result = self._resolve_path(parsed, query)
            return {"result": result, "query": query, "type": type(result).__name__ if result is not None else "null"}

        return {"result": parsed, "type": type(parsed).__name__}
