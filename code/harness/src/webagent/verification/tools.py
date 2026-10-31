"""An opt-in ``request_finish`` tool for the same live EffGen loop.

Emitting a final answer cannot be resumed after the generator has stopped. This
tool lets a model ask the completion checker while the loop is still alive:
the checker either confirms every requirement (and the model may then answer) or
returns the remaining goals and conditions as a failed tool result, so the same
generator can continue working.

The tool is registered through ``extra_tools``. It is not added to the closed
``TOOL_SPECS`` web schema and is never visible in an official step profile unless
the caller explicitly registers it.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from effgen.tools.base_tool import (
    BaseTool,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)

from .completion import CompletionChecker
from .types import VerificationRequest

COMPLETION_TOOL_NAMES = frozenset({"request_finish"})


class CompletionToolRuntime:
    """The live checker and a factory for the current observation's requests."""

    def __init__(
        self,
        checker: CompletionChecker,
        request_factory: Callable[[], Sequence[VerificationRequest]],
        *,
        commit: bool = True,
    ) -> None:
        self.checker = checker
        self.request_factory = request_factory
        self.commit = commit

    def request_finish(self) -> tuple[bool, str]:
        decision = self.checker.check(self.request_factory(), commit=self.commit)
        text = json.dumps(decision.to_dict(), ensure_ascii=False, separators=(",", ":"))
        if decision.accepted:
            return True, text
        return False, text


class _CompletionTool(BaseTool):
    async def _execute(self, **kwargs: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError("completion tools execute synchronously")

    def execute(self, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        ok, error = self.validate_parameters(**kwargs)
        if not ok:
            return ToolResult(
                success=False, output=None, error=f"Parameter validation failed: {error}"
            )
        runtime: CompletionToolRuntime = self._runtime
        try:
            accepted, text = runtime.request_finish()
        except Exception as exc:  # noqa: BLE001 - report the refusal as tool text
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool execution failed: {type(exc).__name__}: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        if accepted:
            return ToolResult(success=True, output=text)
        return ToolResult(
            success=False,
            output=text,
            error=(
                "Completion is not confirmed. The task has remaining requirements; "
                "continue in this same loop using the returned remaining_goal_ids "
                "and remaining_conditions, or finish honestly as incomplete."
            ),
            metadata={"error_type": "CompletionRefused"},
        )


class RequestFinishTool(_CompletionTool):
    def __init__(self, runtime: CompletionToolRuntime) -> None:
        super().__init__(
            ToolMetadata(
                name="request_finish",
                description=(
                    "Ask the completion checker whether every requirement is "
                    "verified. It returns remaining goals/conditions instead of "
                    "confirming a partial result. Use it before a final answer."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[],
            )
        )
        self._runtime = runtime


def build_completion_tools(runtime: CompletionToolRuntime) -> list[BaseTool]:
    return [RequestFinishTool(runtime)]


__all__ = [
    "COMPLETION_TOOL_NAMES",
    "CompletionToolRuntime",
    "RequestFinishTool",
    "build_completion_tools",
]
