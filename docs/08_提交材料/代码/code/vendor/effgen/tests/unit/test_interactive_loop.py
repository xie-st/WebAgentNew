"""The interactive entry point: pause on an external call, resume with its result.

``run()`` executes every call it reads. ``run_interactive()`` drives the same
loop but hands the calls to a named set of externally executed tools to its
caller, records the result the caller sends back, and continues from the same
conversation. These tests pin the parts that must not drift:

* one request per call, one call per request; a result is recorded only after
  the caller sends it;
* an externally executed tool is never answered from the run's record and never
  read as a repeat, so two identical page calls are two requests;
* a batch that asks for one of them is refused as a batch, never dispatched;
* the run's scope is installed for each advance and removed before the caller
  holds a request, so two callbacks can advance one generator without leaking;
* the default is no iteration cap (the caller's deadline ends the run), and an
  explicit cap is still honoured;
* ordinary ``run()`` is unchanged.
"""

from __future__ import annotations

import contextvars
import json
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig, AgentMode
from effgen.core.agent_interactive import (
    InteractiveProtocolError,
    NO_ITERATION_CAP,
    InteractiveToolRequest,
    InteractiveToolResult,
)
from effgen.core.agent_tool_loop import NativeToolLoop
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.prompts.tool_contract import ToolUsePolicy
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)


class _PageClick(BaseTool):
    """A page action: executed by an environment outside the process."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="web_click",
            description="Click one element of the current observation.",
            category=ToolCategory.EXTERNAL_API,
            parameters=[ParameterSpec(
                name="target", type=ParameterType.STRING,
                description="element number", required=True,
            )],
        ))
        self.calls = 0

    async def _execute(self, **kwargs: Any) -> str:  # pragma: no cover - not used
        raise AssertionError("a page action must not run in-process")

    def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(
            success=False, output=None,
            error="page actions execute in the environment",
        )


class _Note(BaseTool):
    """An internal tool: it runs inside the ordinary loop path."""

    def __init__(self, notes: list[str]) -> None:
        super().__init__(metadata=ToolMetadata(
            name="task_note",
            description="Save a task-local fact.",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[ParameterSpec(
                name="note", type=ParameterType.STRING,
                description="the fact", required=True,
            )],
        ))
        self.notes = notes

    async def _execute(self, **kwargs: Any) -> str:  # pragma: no cover - not used
        raise AssertionError("harness-free test tools execute synchronously")

    def execute(self, **kwargs: Any) -> ToolResult:
        self.notes.append(str(kwargs.get("note")))
        return ToolResult(success=True, output=f"saved: {kwargs.get('note')}")


class _Script(BaseModel):
    """A scripted model: text turns, or a turn carrying native tool calls."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        super().__init__(model_name="interactive-model", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.script = script
        self.prompts: list[Any] = []
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return False

    def generate(self, prompt, config=None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        turn = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        calls = turn.get("calls")
        return GenerationResult(
            text=str(turn.get("text", "")),
            tokens_used=7,
            finish_reason="tool_calls" if calls else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": calls} if calls else {},
        )

    def generate_stream(self, prompt, config=None, **kwargs: Any):
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _action(tool: str, arguments: dict[str, Any]) -> str:
    return (
        f"Thought: next step.\nAction: {tool}\n"
        f"Action Input: {json.dumps(arguments)}"
    )


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"c-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _agent(model: BaseModel, clicks: _PageClick | None = None, notes: list[str] | None = None) -> Agent:
    tools: list[BaseTool] = [clicks or _PageClick()]
    if notes is not None:
        tools.append(_Note(notes))
    return Agent(AgentConfig(
        model=model,
        tools=tools,
        mode=AgentMode.SINGLE,
        tool_calling_mode="react",
        tool_use=ToolUsePolicy.AUTO,
        max_turns_without_progress=None,
        enable_memory=False,
        enable_sub_agents=False,
        capability_probe=False,
        enable_fallback=False,
    ))


def _drain(generator: Any, results: list[str]) -> tuple[list[InteractiveToolRequest], Any]:
    requests: list[InteractiveToolRequest] = []
    generator = iter(generator)
    request = next(generator)
    requests.append(request)
    for result in results:
        try:
            request = generator.send(result)
        except StopIteration as stop:
            return requests, stop.value
        requests.append(request)
    try:
        next(generator)
    except StopIteration as stop:
        return requests, stop.value
    raise AssertionError("script kept yielding past its results")


def test_external_call_yields_and_the_sent_result_is_recorded() -> None:
    notes: list[str] = []
    clicks = _PageClick()
    model = _Script([
        {"text": _action("task_note", {"note": "want the red one"})},
        {"text": _action("web_click", {"target": "2"})},
        {"text": _action("web_click", {"target": "1"})},
        {"text": "Thought: done.\nFinal Answer: red one opened"},
    ])
    agent = _agent(model, clicks, notes)
    requests, response = _drain(
        agent.run_interactive("open the red one", external_tools=["web_click"]),
        ["PAGE-A result marker", "PAGE-B result marker"],
    )

    assert [request.tool for request in requests] == ["web_click", "web_click"]
    assert requests[0].request_id != requests[1].request_id
    assert requests[0].arguments == {"target": "2"}
    # The internal call ran inside the same advance, before the page request.
    assert notes == ["want the red one"]
    assert clicks.calls == 0, "a page action never runs in-process"
    assert response.success is True
    assert response.output == "red one opened"
    assert response.metadata["interactive_requests"] == 2
    # Three calls the run made: the internal note and the two page actions.
    assert response.metadata["tool_results"]["dispatched"] == 3

    # The observation the caller sent is what the next turn read.
    assert "PAGE-A result marker" in model.prompts[2]
    assert "PAGE-B result marker" in model.prompts[3]
    # ...and the note's observation survived in the same conversation.
    assert "saved: want the red one" in model.prompts[3]

    thread = response.metadata["thread"]
    observations = [
        step.text for step in thread.steps if type(step).__name__ == "ObservationStep"
    ]
    assert "PAGE-A result marker" in observations
    assert "PAGE-B result marker" in observations


def test_a_request_is_a_wait_not_a_result() -> None:
    clicks = _PageClick()
    model = _Script([
        {"text": _action("web_click", {"target": "1"})},
        {"text": "Final Answer: done"},
    ])
    agent = _agent(model, clicks)
    generator = agent.run_interactive("click", external_tools=["web_click"])
    request = next(generator)
    assert isinstance(request, InteractiveToolRequest)
    assert clicks.calls == 0
    assert len(model.prompts) == 1
    # Closing at the request leaves the call unanswered: no further model turn,
    # and no result is invented.
    generator.close()
    assert len(model.prompts) == 1
    assert clicks.calls == 0


def test_two_identical_page_calls_are_two_requests() -> None:
    model = _Script([
        {"text": _action("web_click", {"target": "3"})},
        {"text": _action("web_click", {"target": "3"})},
        {"text": "Final Answer: clicked twice"},
    ])
    agent = _agent(model)
    requests, response = _drain(
        agent.run_interactive("click it twice", external_tools=["web_click"]),
        ["FIRST page", "SECOND page"],
    )
    assert len(requests) == 2
    assert requests[0].arguments == requests[1].arguments == {"target": "3"}
    assert "FIRST page" in model.prompts[1]
    assert "SECOND page" in model.prompts[2]
    assert response.metadata["tool_results"]["dispatched"] == 2
    assert not response.metadata["tool_results"]["tool_side_failures"]


def test_more_than_ten_page_calls_are_not_cut_by_the_default_cap() -> None:
    script: list[dict[str, Any]] = [
        {"text": _action("web_click", {"target": "1"})} for _ in range(12)
    ]
    script.append({"text": "Final Answer: walked twelve steps"})
    model = _Script(script)
    agent = _agent(model)
    requests, response = _drain(
        agent.run_interactive("walk", external_tools=["web_click"]),
        [f"PAGE-{index}" for index in range(12)],
    )
    assert len(requests) == 12
    assert response.success is True
    assert response.metadata["tool_results"]["dispatched"] == 12
    assert NO_ITERATION_CAP > 10


def test_an_explicit_iteration_cap_is_still_honoured() -> None:
    model = _Script([
        {"text": _action("web_click", {"target": "1"})},
        {"text": _action("web_click", {"target": "2"})},
    ])
    agent = _agent(model)
    generator = agent.run_interactive(
        "walk", external_tools=["web_click"], max_iterations=1
    )
    request = next(generator)
    assert request.tool == "web_click"
    with pytest.raises(StopIteration) as stop:
        generator.send("PAGE")
    response = stop.value.value
    assert response.success is False
    assert len(model.prompts) == 1


def test_batch_of_page_actions_is_refused_not_dispatched() -> None:
    model = _Script([
        {"text": "", "calls": [
            _call("web_click", {"target": "1"}),
            _call("web_click", {"target": "2"}),
        ]},
        {"text": "Final Answer: refused the batch"},
    ])
    clicks = _PageClick()
    agent = _agent(model, clicks)
    marker = "ONE PAGE ACTION PER OUTPUT"

    def correction(calls: Any) -> str:
        assert len(calls) == 2
        return marker

    requests, response = _drain_batch(
        agent.run_interactive(
            "click", external_tools=["web_click"], batch_correction=correction,
        )
    )
    assert requests == []
    assert clicks.calls == 0
    assert marker in model.prompts[1]
    assert response.success is True


def _drain_batch(generator: Any) -> tuple[list[InteractiveToolRequest], Any]:
    requests: list[InteractiveToolRequest] = []
    try:
        while True:
            requests.append(next(generator))
    except StopIteration as stop:
        return requests, stop.value


def test_the_run_scope_is_installed_per_advance_and_removed() -> None:
    """Two callbacks advance one generator; neither leaves a binding behind."""

    from effgen.core import agent as agent_module

    model = _Script([
        {"text": _action("web_click", {"target": "1"})},
        {"text": "Final Answer: done"},
    ])
    agent = _agent(model)
    generator = agent.run_interactive("click", external_tools=["web_click"])
    seen: dict[str, bool] = {}

    def inside(callback):
        def run() -> Any:
            outcome = callback()
            # The scope is removed once the advance returns to the caller.
            seen["scope_after"] = agent_module._run_scope_var.get() is None
            seen["call_state_after"] = agent_module._call_state_var.get() is None
            seen["agent_session_after"] = agent.session is None
            return outcome

        return run

    ctx_a = contextvars.Context()
    request = ctx_a.run(inside(lambda: next(generator)))
    assert isinstance(request, InteractiveToolRequest)
    assert seen == {
        "scope_after": True,
        "call_state_after": True,
        "agent_session_after": True,
    }
    assert agent_module._run_scope_var.get() is None

    ctx_b = contextvars.Context()
    try:
        ctx_b.run(inside(lambda: generator.send(InteractiveToolResult("PAGE-X"))))
        response: Any = None
    except StopIteration as stop:
        response = stop.value
    assert response is not None and response.success is True
    assert seen == {
        "scope_after": True,
        "call_state_after": True,
        "agent_session_after": True,
    }
    assert "PAGE-X" in model.prompts[1]
    assert agent_module._run_scope_var.get() is None
    assert agent_module._call_state_var.get() is None


def test_ordinary_run_still_executes_internal_tools() -> None:
    notes: list[str] = []
    model = _Script([
        {"text": _action("task_note", {"note": "ordinary path"})},
        {"text": "Final Answer: noted"},
    ])
    agent = _agent(model, _PageClick(), notes)
    response = agent.run("write a note")
    assert response.success is True
    assert notes == ["ordinary path"]
    assert "saved: ordinary path" in model.prompts[1]


def test_interactive_entry_refuses_unsupported_configuration() -> None:
    model = _Script([{"text": "Final Answer: x"}])
    agent = _agent(model)
    with pytest.raises(ValueError):
        list(agent.run_interactive("t", external_tools=["web_click"], session="x"))
    with pytest.raises(ValueError, match="does not hold"):
        list(agent.run_interactive("t", external_tools=["web_goto"]))
    with pytest.raises(ValueError, match="external_tools"):
        list(agent.run_interactive("t", external_tools=[]))

    native = Agent(AgentConfig(
        model=model, tools=[_PageClick()], mode=AgentMode.SINGLE,
        tool_calling_mode="native", enable_memory=False,
    ))
    with pytest.raises(ValueError, match="react"):
        list(native.run_interactive("t", external_tools=["web_click"]))

    decomposed = Agent(AgentConfig(
        model=model, tools=[_PageClick()], mode=AgentMode.AUTO,
        tool_calling_mode="react", enable_memory=False,
    ))
    with pytest.raises(ValueError, match="SINGLE"):
        list(decomposed.run_interactive("t", external_tools=["web_click"]))


def test_a_resume_must_carry_the_environment_result() -> None:
    model = _Script([
        {"text": _action("web_click", {"target": "1"})},
        {"text": "Final Answer: done"},
    ])
    agent = _agent(model)
    generator = agent.run_interactive("click", external_tools=["web_click"])
    next(generator)
    with pytest.raises(InteractiveProtocolError):
        generator.send(None)


def test_external_tools_bypass_the_repeat_guards() -> None:
    loop = NativeToolLoop(
        {"web_click": _PageClick(), "note": _Note([])},
        external_tools=frozenset({"web_click"}),
    )
    loop.record_action(loop.check_action("web_click", '{"target": "1"}'))
    loop.record_execution("web_click", arguments='{"target": "1"}', result="same page")
    loop.record_pair_result(loop.check_action("web_click", '{"target": "1"}'), "same page")
    loop.record_result("web_click", "same page")
    loop.observe_result("web_click", "same page")
    check = loop.check_action("web_click", '{"target": "1"}')
    assert check.is_exact_loop is False
    assert check.is_fuzzy_loop is False
    assert loop.cached_result(check) is None
    assert loop.result_is_repeat("web_click", "same page") is False
    assert loop.post_tool_nudge(5, 9, "same page", tool="web_click") is None
    assert loop.execution_tools() == []
    assert loop.observe_result("web_click", "same page") is True
