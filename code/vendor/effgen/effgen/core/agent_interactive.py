"""A run that hands some of its tool calls to its caller and waits for them.

The blocking loop executes every call it reads. A page action cannot work that
way: it is executed by an environment outside the process, and the observation
that comes back is what the next turn must read. This module adds one entry
point, :meth:`AgentInteractiveMixin.run_interactive`, that drives the *same*
:func:`~effgen.core.agent_loop.drive` generator as :meth:`Agent.run` and, where
the loop would dispatch a call to an externally executed tool, yields an
:class:`InteractiveToolRequest` to the caller instead. Whatever the caller sends
back with :meth:`~collections.abc.Generator.send` is recorded as that call's
result and the loop continues from the same conversation, guards and counters.

What is deliberately *not* here:

* a second planning loop — every turn, guard, parse and result record is the one
  ``run()`` uses;
* a thread, a worker pool or an event loop standing in for the pause — the
  generator simply suspends, and the caller resumes it;
* state kept in a context variable across a pause. The run's scope is installed
  around each advance and removed before the request reaches the caller, so a
  caller that advances the generator from another callback (an SDK's
  ``step()``) sees no binding left over from the previous one.

Only single-agent, ReAct-text, non-streaming runs are accepted. A native-tool
or decomposed frame answers with request shapes this entry does not implement,
and refusing them is better than pretending to drive them.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from . import ledger as _ledger
from .agent_config import AgentMode, validate_max_turns_without_progress
from .agent_loop import _Collecting, _LoopPolicy, drive
from .agent_response import AgentResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .agent_config import AgentConfig

__all__ = [
    "AgentInteractiveMixin",
    "InteractiveProtocolError",
    "InteractiveRunContext",
    "InteractiveToolRequest",
    "InteractiveToolResult",
    "NO_ITERATION_CAP",
]

#: What the interactive entry uses in place of an iteration cap. The run is
#: bounded by the caller's deadline and cancellation instead: a page task has as
#: many steps as the page needs, and the framework's shipped ten would end a
#: normal walk early. A caller that wants a hard cap passes ``max_iterations``.
NO_ITERATION_CAP = 2**31 - 1

#: ``run()`` arguments this entry does not implement. They change the shape of
#: the run (a second conversation, a decomposed mode, a content-part request, a
#: structured answer) rather than its sampling, and silently accepting one would
#: drive a run this entry cannot pause correctly.
_UNSUPPORTED_RUN_ARGS = frozenset({
    "context", "debug", "inputs", "middleware", "mode", "output_format",
    "output_model", "output_schema", "session", "stream",
})


class InteractiveProtocolError(RuntimeError):
    """The caller resumed a run with something that is not a tool result."""


@dataclass(frozen=True)
class InteractiveToolResult:
    """What an environment returned for one externally executed call.

    A plain string is accepted too; this type exists so a caller can be explicit
    about the value it sends without the loop having to guess.
    """

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InteractiveProtocolError(
                "an externally executed tool's result must be text"
            )


@dataclass(frozen=True)
class InteractiveToolRequest:
    """One tool call the run is waiting for its caller to execute.

    The request is a *waiting* action, not a result: the conversation step for
    the call is written only after the caller sends something back. Closing the
    generator at that point therefore leaves the call unanswered rather than
    recording a success.
    """

    tool: str
    arguments: Mapping[str, Any]
    raw_input: str
    request_id: str
    iteration: int
    call_id: str | None = None
    reasoning: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise InteractiveProtocolError("an interactive request needs a tool name")
        if not isinstance(self.arguments, Mapping):
            raise InteractiveProtocolError("interactive arguments must be a mapping")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType({str(key): value for key, value in self.arguments.items()}),
        )
        if not isinstance(self.request_id, str) or not self.request_id:
            raise InteractiveProtocolError("an interactive request needs an id")
        if not isinstance(self.raw_input, str):
            raise InteractiveProtocolError("interactive raw input must be text")


class InteractiveRunContext:
    """The channel between one interactive run and its caller.

    One instance belongs to one run: it owns that run's external-tool set, mints
    its request ids and holds the caller's hooks. It never holds a conversation
    or a context variable — those belong to the run's scope, installed per step.
    """

    def __init__(
        self,
        external_tools: Iterable[str],
        *,
        on_before_step: Callable[[int], None] | None = None,
        on_before_model_call: Callable[[Any], None] | None = None,
        on_batch_correction: Callable[[tuple[tuple[str, Mapping[str, Any]], ...]], str] | None = None,
    ) -> None:
        names = frozenset(str(name) for name in external_tools)
        if not names:
            raise InteractiveProtocolError(
                "an interactive run needs at least one externally executed tool"
            )
        self.external_tools = names
        self._on_before_step = on_before_step
        self._on_before_model_call = on_before_model_call
        self._on_batch_correction = on_batch_correction
        self._requests: list[InteractiveToolRequest] = []

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> tuple[InteractiveToolRequest, ...]:
        return tuple(self._requests)

    def is_external(self, tool: str) -> bool:
        return tool in self.external_tools

    def before_step(self, iteration: int) -> None:
        """Called once before each turn is built; the caller checks its budget."""
        if self._on_before_step is not None:
            self._on_before_step(int(iteration))

    def before_model_call(self, prompt: Any) -> None:
        """Called once before each model turn; the caller checks its budget."""
        if self._on_before_model_call is not None:
            self._on_before_model_call(prompt)

    def request(
        self,
        *,
        tool: str,
        arguments: Mapping[str, Any],
        raw_input: str,
        iteration: int,
        call_id: str | None = None,
        reasoning: str = "",
    ) -> InteractiveToolRequest:
        request = InteractiveToolRequest(
            tool=tool,
            arguments=arguments,
            raw_input=raw_input,
            request_id=f"ireq-{len(self._requests) + 1}",
            iteration=int(iteration),
            call_id=call_id,
            reasoning=str(reasoning or ""),
        )
        self._requests.append(request)
        return request

    def batch_correction(
        self, calls: tuple[tuple[str, Mapping[str, Any]], ...] | list[Any]
    ) -> str:
        """What the model is told when one output asked for several page actions.

        The caller supplies the words when it can, because it is the party that
        knows which observation is current and which element numbers are valid.
        The loop only knows that a batch asked for these calls and that none of
        them was dispatched.
        """
        normalized = tuple((str(name), dict(args or {})) for name, args in calls)
        if self._on_batch_correction is not None:
            return str(self._on_batch_correction(normalized))
        names = ", ".join(sorted({name for name, _ in normalized}))
        return (
            "This model output asked for more than one page action: "
            f"{names}. No page action from this output was executed or queued. "
            "Emit exactly one page action per model output and wait for its "
            "observation before deciding the next one."
        )

    @staticmethod
    def result_text(value: Any) -> str:
        """The text a caller's ``send()`` value contributes as the tool result."""
        if isinstance(value, InteractiveToolResult):
            return value.text
        if isinstance(value, str):
            return value
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
        raise InteractiveProtocolError(
            "an externally executed tool was resumed with "
            f"{type(value).__name__}; send a string or InteractiveToolResult. "
            "Only the environment's own result may be recorded for that call."
        )


class _InteractiveEmitter(_Collecting):
    """The blocking emitter, plus the caller's check before each model turn.

    The answer still arrives as the run's return value and nothing is streamed,
    exactly as on ``run()``; the only difference is that the caller is asked
    whether the turn can still be afforded before it is sent.
    """

    def __init__(self, agent: Any, context: InteractiveRunContext) -> None:
        super().__init__(agent)
        self.context = context

    def model_turn(
        self, agent: Any, prompt: Any, policy: _LoopPolicy,
        gen_kwargs: dict[str, Any], *, streamable: bool,
    ) -> Generator[Any, None, dict[str, Any]]:
        self.context.before_model_call(prompt)
        return (yield from super().model_turn(
            agent, prompt, policy, gen_kwargs, streamable=streamable,
        ))


class AgentInteractiveMixin:
    """The interactive run entry point contributed to :class:`Agent`."""

    if TYPE_CHECKING:  # pragma: no cover - typing only
        config: "AgentConfig"
        tools: dict[str, Any]
        model: Any
        name: str

        def _new_run_scope(self, **kwargs: Any) -> Any: ...
        def _open_stream_scope(self, scope: Any) -> Any: ...

    def run_interactive(
        self,
        task: str,
        *,
        external_tools: Iterable[str] | None = None,
        before_step: Callable[[int], None] | None = None,
        before_model_call: Callable[[Any], None] | None = None,
        batch_correction: Callable[[tuple[tuple[str, Mapping[str, Any]], ...]], str] | None = None,
        max_iterations: int | None = None,
        max_turns_without_progress: int | None = None,
        **kwargs: Any,
    ) -> Generator[InteractiveToolRequest, Any, AgentResponse]:
        """Drive *task* as a generator that yields the calls the caller executes.

        The caller starts it with ``next()`` and resumes it with
        ``generator.send(result)``, where ``result`` is the environment's own
        result for the request just yielded. The generator's return value is the
        run's :class:`~effgen.core.agent_response.AgentResponse`, the same object
        ``run()`` returns.

        Args:
            task: The task text.
            external_tools: Names of the tools the caller executes. Required, and
                every name must be a tool this agent holds. A native batch that
                asks for one of them is refused as a batch and answered with the
                caller's correction; a single call to one of them is yielded.
            before_step: Called once before each turn is built. Raise from it to
                stop the run (a deadline or a cancellation check).
            before_model_call: Called once before each model turn.
            batch_correction: Words for the model when one output asked for
                several page actions; defaults to a generic refusal.
            max_iterations: A hard turn cap, or ``None`` to bound the run by the
                caller's deadline instead. ``None`` does **not** fall back to the
                framework's default ten turns: a page task decides its own
                length, and the caller is the party that can end it.
            max_turns_without_progress: A stall limit, or ``None`` (the default)
                never to ask for an answer because turns repeated themselves.
            **kwargs: Sampling settings and the run's own recognised arguments,
                forwarded exactly as ``run()`` forwards them.

        Yields:
            InteractiveToolRequest: one call the caller must execute before the
            generator is resumed.

        Returns:
            AgentResponse: the run's terminal response.

        Raises:
            ValueError: If the configuration is not a supported interactive run,
                or the external-tool set is empty or not held by this agent.
            InteractiveProtocolError: If the generated run is resumed with
                something that is not the environment's result.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("run_interactive needs a non-empty task string")
        unsupported = sorted(set(kwargs) & _UNSUPPORTED_RUN_ARGS)
        if unsupported:
            raise ValueError(
                "run_interactive does not accept: " + ", ".join(unsupported)
                + "; it drives one single-agent, text, non-streaming run"
            )
        if not self.tools:
            raise ValueError("run_interactive needs the agent to hold tools")
        config = self.config
        if config.mode is not AgentMode.SINGLE:
            raise ValueError(
                "run_interactive supports AgentMode.SINGLE only; "
                f"this agent is configured for {config.mode.value}"
            )
        if str(config.tool_calling_mode) != "react":
            raise ValueError(
                "run_interactive supports tool_calling_mode='react' only; "
                f"this agent is configured for {config.tool_calling_mode!r}"
            )
        if config.system_prompt_template:
            raise ValueError(
                "run_interactive does not drive a custom prompt template; "
                "the ReAct text frame is the supported one"
            )
        if config.enable_streaming:
            raise ValueError("run_interactive drives a non-streaming run")
        names = frozenset(str(name) for name in (external_tools or ()))
        if not names:
            raise ValueError(
                "run_interactive needs external_tools naming the calls the "
                "caller executes"
            )
        missing = sorted(names - set(self.tools))
        if missing:
            raise ValueError(
                "run_interactive names tools this agent does not hold: "
                + ", ".join(missing)
            )
        if max_iterations is not None:
            max_iterations = int(max_iterations)
            if max_iterations < 1:
                raise ValueError("max_iterations must be a positive integer or None")
        max_turns_without_progress = validate_max_turns_without_progress(
            max_turns_without_progress
        )
        context = InteractiveRunContext(
            names,
            on_before_step=before_step,
            on_before_model_call=before_model_call,
            on_batch_correction=batch_correction,
        )
        loop_kwargs = dict(kwargs)
        loop_kwargs["max_iterations"] = (
            NO_ITERATION_CAP if max_iterations is None else max_iterations
        )
        loop_kwargs["max_turns_without_progress"] = max_turns_without_progress
        policy = _LoopPolicy.for_run(
            self, loop_kwargs, emit_deltas=False, interactive=context,
        )
        emitter = _InteractiveEmitter(self, context)
        return (yield from self._drive_interactive(task, policy, emitter, context))

    def _drive_interactive(
        self,
        task: str,
        policy: _LoopPolicy,
        emitter: _InteractiveEmitter,
        context: InteractiveRunContext,
    ) -> Generator[InteractiveToolRequest, Any, AgentResponse]:
        """Advance the loop generator once per caller callback.

        The run's scope and the run's ledger are installed around each advance
        and removed before the request reaches the caller: a generator cannot
        hold context variables at a yield, and an SDK that advances it from a
        later callback must not read a binding from the earlier one.
        """
        scope = self._open_stream_scope(self._new_run_scope())
        ledger_rec = _ledger.open_run("interactive", self.name)
        ledger_closed = False
        items = drive(self, task, policy, emitter)
        send_value: Any = None
        started = False
        try:
            while True:
                ledger_token = _ledger.enter(ledger_rec)
                scope_tokens = scope.enter()
                try:
                    if started:
                        item = items.send(send_value)
                    else:
                        started = True
                        item = next(items)
                finally:
                    scope.leave(scope_tokens)
                    _ledger.leave(ledger_token)
                send_value = yield item
        except StopIteration as stop:
            response: AgentResponse = stop.value
            if isinstance(getattr(response, "metadata", None), dict):
                response.metadata["interactive_requests"] = context.request_count
                if ledger_rec is not None:
                    run_ledger = ledger_rec.close(
                        int(getattr(response, "iterations", 0) or 0)
                    )
                    ledger_closed = True
                    response.metadata["ledger"] = run_ledger.to_dict()
            return response
        finally:
            # A run abandoned at a request has a call nothing answered; closing
            # the loop generator leaves it that way rather than inventing a
            # result. Both closes are idempotent.
            items.close()
            if ledger_rec is not None and not ledger_closed:
                ledger_rec.close()
            scope.close()
