"""Hooks around an agent run, its model calls and its tool calls.

effGen ships guardrails, observability, reliability and cost tracking as
subsystems. Middleware is the general form of the same idea: a place to put
behaviour effGen does not ship — an approval gate, a cache, a redaction pass, a
per-run spend cap, a custom trace exporter — without patching the loop.

There are three points, and each has a "before" and an "after":

- **the run** — once per :meth:`Agent.run`;
- **each model call** — every generation the run makes, including retries;
- **each tool call** — every dispatch the loop makes.

A ``before_`` hook may modify the request in place through its context, or
return a value to **short-circuit**: the real call is skipped and the returned
value is used instead. An ``after_`` hook receives the result and returns it,
possibly changed. ``before_`` hooks run in the order the middleware were given;
``after_`` hooks run in reverse, so a middleware wraps the ones after it the way
nested context managers do.

Subclass :class:`AgentMiddleware` and override only what you need::

    class BlockExpensiveTools(AgentMiddleware):
        def before_tool_call(self, ctx):
            if ctx.tool_name == "web_search" and ctx.budget_spent > 1.0:
                return "Skipped: this run has spent its search budget."

    agent = Agent(AgentConfig(model="gpt-5-nano", middleware=[BlockExpensiveTools()]))

Middleware can also be given per call, and are appended to the configured ones
for that run only::

    agent.run(task, middleware=[Timing()])

A hook that raises is not caught here — the exception reaches the caller like
any other error in the run, which is what makes an approval gate that refuses
able to stop the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_response import AgentResponse

logger = logging.getLogger(__name__)

__all__ = [
    "AgentMiddleware",
    "MiddlewareChain",
    "ModelCallContext",
    "RunContext",
    "ToolCallContext",
    "LoggingMiddleware",
    "ToolApprovalMiddleware",
]


@dataclass
class RunContext:
    """What a run was asked to do.

    Attributes:
        task: The task text. A ``before_run`` hook may rewrite it, and the run
            uses the rewritten text.
        agent_name: The agent's configured name.
        mode: The execution mode the run was started in.
        metadata: Free space middleware can use to pass values along the chain
            and into ``after_run``. effGen never reads it.
    """

    task: str
    agent_name: str = ""
    mode: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCallContext:
    """One generation the run is about to make, or has just made.

    Attributes:
        prompt: What the model is being asked. A ``before_model_call`` hook may
            replace it, and the model is called with the replacement.
        model_name: The model id serving this call.
        kwargs: Generation options for this call, editable in place — a hook can
            pin a temperature or a token cap for its own reasons.
        attempt: 1 for the first try, higher for a retry.
        run: The :class:`RunContext` this call belongs to.
    """

    prompt: Any
    model_name: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    run: RunContext | None = None


@dataclass
class ToolCallContext:
    """One tool dispatch the run is about to make, or has just made.

    Attributes:
        tool_name: The tool's registered name.
        tool_input: The input as the model supplied it. A ``before_tool_call``
            hook may replace it, and the tool is called with the replacement.
        run: The :class:`RunContext` this dispatch belongs to.
    """

    tool_name: str
    tool_input: str = ""
    run: RunContext | None = None


class AgentMiddleware:
    """Base class for a run/model/tool hook. Override only what you need.

    Every hook has a default that does nothing, so a subclass implementing one
    of them behaves exactly as before at the other five points.
    """

    #: Shown in logs and errors. Defaults to the class name.
    name: str = ""

    def __repr__(self) -> str:
        return f"<{self.name or type(self).__name__}>"

    # -- the run --------------------------------------------------------
    def before_run(self, ctx: RunContext) -> "AgentResponse | None":
        """Called once before the run starts.

        Returns:
            None to continue, or an :class:`AgentResponse` to answer the run
            without calling the model at all — a cache hit, or a refusal.
        """
        return None

    def after_run(self, ctx: RunContext, response: "AgentResponse") -> "AgentResponse":
        """Called once after the run finishes. Return the response to report."""
        return response

    # -- each model call ------------------------------------------------
    def before_model_call(self, ctx: ModelCallContext) -> Any:
        """Called before each generation.

        Returns:
            None to continue, or a generation result to use instead of calling
            the model.
        """
        return None

    def after_model_call(self, ctx: ModelCallContext, result: Any) -> Any:
        """Called after each generation. Return the result to use."""
        return result

    # -- each tool call -------------------------------------------------
    def before_tool_call(self, ctx: ToolCallContext) -> str | None:
        """Called before each tool dispatch.

        Returns:
            None to continue, or a string to use as the tool's output without
            running it — an approval gate's refusal, or a cached result.
        """
        return None

    def after_tool_call(self, ctx: ToolCallContext, result: str) -> str:
        """Called after each tool dispatch. Return the output to use."""
        return result


class MiddlewareChain:
    """Runs a list of middleware around one of the three points.

    Holds no state of its own beyond the list, so one chain can serve every
    call in a run. An empty chain costs a single boolean test at each point,
    which is why the hooks can sit on the hot path.
    """

    __slots__ = ("middleware",)

    def __init__(self, middleware: "list[AgentMiddleware] | None" = None) -> None:
        self.middleware: list[AgentMiddleware] = list(middleware or [])

    def __bool__(self) -> bool:
        return bool(self.middleware)

    def __len__(self) -> int:
        return len(self.middleware)

    def __repr__(self) -> str:
        return f"MiddlewareChain({self.middleware!r})"

    def extended_with(
        self, extra: "list[AgentMiddleware] | None"
    ) -> "MiddlewareChain":
        """Return a chain with *extra* appended, leaving this one unchanged.

        Used for ``run(..., middleware=[...])``, which adds to the configured
        middleware for one call rather than replacing them.
        """
        if not extra:
            return self
        return MiddlewareChain(self.middleware + list(extra))

    # -- the run --------------------------------------------------------
    def before_run(self, ctx: RunContext) -> "AgentResponse | None":
        """Run each ``before_run``; return the first short-circuit, if any."""
        for mw in self.middleware:
            short_circuit = mw.before_run(ctx)
            if short_circuit is not None:
                logger.debug("Middleware %r answered the run without the model", mw)
                return short_circuit
        return None

    def after_run(self, ctx: RunContext, response: "AgentResponse") -> "AgentResponse":
        """Run each ``after_run`` in reverse order, threading the response."""
        for mw in reversed(self.middleware):
            response = mw.after_run(ctx, response)
        return response

    # -- each model call ------------------------------------------------
    def before_model_call(self, ctx: ModelCallContext) -> Any:
        """Run each ``before_model_call``; return the first short-circuit."""
        for mw in self.middleware:
            short_circuit = mw.before_model_call(ctx)
            if short_circuit is not None:
                logger.debug("Middleware %r answered a model call directly", mw)
                return short_circuit
        return None

    def after_model_call(self, ctx: ModelCallContext, result: Any) -> Any:
        """Run each ``after_model_call`` in reverse order."""
        for mw in reversed(self.middleware):
            result = mw.after_model_call(ctx, result)
        return result

    # -- each tool call -------------------------------------------------
    def before_tool_call(self, ctx: ToolCallContext) -> str | None:
        """Run each ``before_tool_call``; return the first short-circuit."""
        for mw in self.middleware:
            short_circuit = mw.before_tool_call(ctx)
            if short_circuit is not None:
                logger.debug(
                    "Middleware %r answered the %r call without running it",
                    mw, ctx.tool_name,
                )
                return short_circuit
        return None

    def after_tool_call(self, ctx: ToolCallContext, result: str) -> str:
        """Run each ``after_tool_call`` in reverse order."""
        for mw in reversed(self.middleware):
            result = mw.after_tool_call(ctx, result)
        return result


# ---------------------------------------------------------------------------
# Middleware worth having in the box
# ---------------------------------------------------------------------------


class LoggingMiddleware(AgentMiddleware):
    """Log every model call and tool call the run makes.

    Useful on its own for seeing what an agent did, and as a worked example of
    the interface — it touches four of the six hooks and changes nothing.

    Args:
        level: Level the lines are emitted at. Defaults to INFO.
        logger_name: Logger to emit on. Defaults to this module's.
    """

    def __init__(self, level: int = logging.INFO, logger_name: str | None = None) -> None:
        self.level = level
        self.log = logging.getLogger(logger_name) if logger_name else logger

    def before_run(self, ctx: RunContext) -> None:
        """Log the task the run was given."""
        self.log.log(self.level, "run start: %s", ctx.task[:200])
        return None

    def after_run(self, ctx: RunContext, response: "AgentResponse") -> "AgentResponse":
        """Log how the run ended, and return it unchanged."""
        self.log.log(
            self.level, "run end: success=%s tool_calls=%s",
            response.success, response.tool_calls.total,
        )
        return response

    def before_model_call(self, ctx: ModelCallContext) -> None:
        """Log the model and attempt number of each generation."""
        self.log.log(self.level, "model call: %s (attempt %d)", ctx.model_name, ctx.attempt)
        return None

    def before_tool_call(self, ctx: ToolCallContext) -> None:
        """Log each tool dispatch and its input."""
        self.log.log(self.level, "tool call: %s(%s)", ctx.tool_name, ctx.tool_input[:200])
        return None


class ToolApprovalMiddleware(AgentMiddleware):
    """Ask before letting named tools run.

    Args:
        approve: Called with the tool name and its input; return True to allow
            the call. Anything falsey refuses it, and the model is told so.
        tools: Tool names that need approval. None means every tool.
        refusal: What the model is told when a call is refused.
    """

    def __init__(
        self,
        approve: Any,
        tools: "list[str] | None" = None,
        refusal: str = "This call was not approved, so the tool did not run.",
    ) -> None:
        self.approve = approve
        self.tools = set(tools) if tools is not None else None
        self.refusal = refusal

    def before_tool_call(self, ctx: ToolCallContext) -> str | None:
        """Ask for approval, and refuse the call when it is not given."""
        if self.tools is not None and ctx.tool_name not in self.tools:
            return None
        if self.approve(ctx.tool_name, ctx.tool_input):
            return None
        return self.refusal
