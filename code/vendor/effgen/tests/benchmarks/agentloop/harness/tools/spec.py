"""A framework-neutral description of a tool.

Each framework binds these to its own tool type, so every framework gets exactly
the same tool set with the same name, description and schema. That matters: the
paper's finding is partly about tool schema quality, so the schema must not drift
between frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    # JSON Schema for the arguments object.
    parameters: dict[str, Any]
    func: Callable[..., str]

    @property
    def arg_names(self) -> list[str]:
        return list(self.parameters.get("properties", {}).keys())

    def call(self, **kwargs: Any) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as exc:  # a tool error is data, not a crash
            return f"Error: {exc}"

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def one_string_arg(name: str, description: str) -> dict[str, Any]:
    """Shorthand for the common single-string-argument schema."""
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
    }
