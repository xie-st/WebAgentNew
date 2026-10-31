"""Hooks around the run, each model call and each tool call.

Middleware is how behaviour effGen does not ship — an approval gate, a cache, a
redaction pass, a spend cap — attaches to the loop without patching it.
"""

from __future__ import annotations

import pytest

from effgen import Agent, AgentConfig
from effgen.core.agent_response import AgentResponse
from effgen.core.middleware import (
    AgentMiddleware,
    LoggingMiddleware,
    MiddlewareChain,
    ModelCallContext,
    RunContext,
    ToolApprovalMiddleware,
    ToolCallContext,
)
from effgen.tools import get_registry
from tests.fixtures.mock_models import MockModel

TOOL_TURNS = [
    "Thought: compute.\nAction: calculator\nAction Input: 6*7",
    "Thought: done.\nFinal Answer: 42",
]


class Recorder(AgentMiddleware):
    """Notes every hook it is given, in order."""

    def __init__(self, label: str = "r") -> None:
        self.label = label
        self.seen: list[str] = []

    def before_run(self, ctx):
        self.seen.append(f"{self.label}:before_run")
        return None

    def after_run(self, ctx, response):
        self.seen.append(f"{self.label}:after_run")
        return response

    def before_model_call(self, ctx):
        self.seen.append(f"{self.label}:before_model")
        return None

    def after_model_call(self, ctx, result):
        self.seen.append(f"{self.label}:after_model")
        return result

    def before_tool_call(self, ctx):
        self.seen.append(f"{self.label}:before_tool")
        return None

    def after_tool_call(self, ctx, result):
        self.seen.append(f"{self.label}:after_tool")
        return result


def _agent(*middleware, tools=True, **overrides):
    config = AgentConfig(
        model=MockModel(list(TOOL_TURNS)),
        tools=[get_registry().get_tool_sync("calculator")] if tools else [],
        middleware=list(middleware),
        max_iterations=3,
        raise_on_error=False,
        **overrides,
    )
    return Agent(config)


class TestTheChain:
    """Ordering and short-circuiting, without an agent in the way."""

    def test_before_hooks_run_in_the_order_given(self):
        first, second = Recorder("a"), Recorder("b")
        chain = MiddlewareChain([first, second])
        chain.before_run(RunContext(task="t"))
        assert first.seen + second.seen == ["a:before_run", "b:before_run"]

    def test_after_hooks_run_in_reverse_so_middleware_nest(self):
        order: list[str] = []

        class Note(AgentMiddleware):
            def __init__(self, label):
                self.label = label

            def after_run(self, ctx, response):
                order.append(self.label)
                return response

        MiddlewareChain([Note("outer"), Note("inner")]).after_run(
            RunContext(task="t"), AgentResponse(output="x")
        )
        assert order == ["inner", "outer"]

    def test_an_after_hook_can_change_the_result(self):
        class Shout(AgentMiddleware):
            def after_tool_call(self, ctx, result):
                return result.upper()

        assert MiddlewareChain([Shout()]).after_tool_call(
            ToolCallContext("calculator"), "forty two"
        ) == "FORTY TWO"

    def test_the_first_short_circuit_wins_and_later_hooks_do_not_run(self):
        class Answer(AgentMiddleware):
            def before_tool_call(self, ctx):
                return "answered"

        later = Recorder("later")
        chain = MiddlewareChain([Answer(), later])
        assert chain.before_tool_call(ToolCallContext("calculator")) == "answered"
        assert later.seen == []

    def test_an_empty_chain_is_falsey_so_the_hot_path_can_skip_it(self):
        assert not MiddlewareChain()
        assert not MiddlewareChain([])

    def test_per_call_middleware_are_appended_not_substituted(self):
        base, extra = Recorder("base"), Recorder("extra")
        extended = MiddlewareChain([base]).extended_with([extra])
        assert extended.middleware == [base, extra]

    def test_extending_leaves_the_configured_chain_alone(self):
        base = MiddlewareChain([Recorder("base")])
        base.extended_with([Recorder("extra")])
        assert len(base) == 1


class TestAroundARealRun:
    """The hooks a run actually fires."""

    def test_every_point_is_reached(self):
        recorder = Recorder()
        agent = _agent(recorder)
        try:
            agent.run("What is 6*7?")
        finally:
            agent.close()
        assert "r:before_run" in recorder.seen
        assert "r:after_run" in recorder.seen
        assert "r:before_model" in recorder.seen
        assert "r:after_model" in recorder.seen
        assert "r:before_tool" in recorder.seen
        assert "r:after_tool" in recorder.seen

    def test_the_run_starts_and_ends_around_everything_else(self):
        recorder = Recorder()
        agent = _agent(recorder)
        try:
            agent.run("What is 6*7?")
        finally:
            agent.close()
        assert recorder.seen[0] == "r:before_run"
        assert recorder.seen[-1] == "r:after_run"

    def test_a_hook_can_rewrite_the_task(self):
        class Rewrite(AgentMiddleware):
            def before_run(self, ctx):
                ctx.task = "REWRITTEN"
                return None

        seen: list[str] = []

        class Capture(AgentMiddleware):
            def before_model_call(self, ctx):
                seen.append(str(ctx.prompt))
                return None

        agent = _agent(Rewrite(), Capture(), tools=False)
        try:
            agent.run("original task")
        finally:
            agent.close()
        assert any("REWRITTEN" in prompt for prompt in seen)

    def test_a_hook_can_answer_the_run_without_the_model(self):
        class Cached(AgentMiddleware):
            def before_run(self, ctx):
                return AgentResponse(output="from cache", success=True)

        model = MockModel(list(TOOL_TURNS))
        agent = Agent(AgentConfig(model=model, middleware=[Cached()],
                                  raise_on_error=False))
        try:
            response = agent.run("anything")
        finally:
            agent.close()
        assert response.output == "from cache"
        assert model.call_count == 0, "the model was called despite the cache hit"

    def test_a_short_circuited_run_still_passes_through_after_run(self):
        recorder = Recorder()

        class Cached(AgentMiddleware):
            def before_run(self, ctx):
                return AgentResponse(output="from cache")

        agent = Agent(AgentConfig(model=MockModel(list(TOOL_TURNS)),
                                  middleware=[recorder, Cached()],
                                  raise_on_error=False))
        try:
            agent.run("anything")
        finally:
            agent.close()
        assert "r:after_run" in recorder.seen

    def test_a_hook_can_answer_a_tool_call_without_running_the_tool(self):
        class Refuse(AgentMiddleware):
            def before_tool_call(self, ctx):
                return "Refused: this tool is not allowed here."

        agent = _agent(Refuse())
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert "Refused" in (response.tool_calls[0].result or "")

    def test_a_hook_can_rewrite_a_tool_s_input(self):
        class Redirect(AgentMiddleware):
            def before_tool_call(self, ctx):
                ctx.tool_input = "2*2"
                return None

        agent = _agent(Redirect())
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert "4" in (response.tool_calls[0].result or "")

    def test_a_hook_can_change_the_answer(self):
        class Append(AgentMiddleware):
            def after_run(self, ctx, response):
                response.output = response.output + " [checked]"
                return response

        agent = _agent(Append())
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert response.output.endswith("[checked]")

    def test_a_run_with_no_middleware_behaves_as_before(self):
        agent = _agent()
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert response.success


class TestPerCallMiddleware:
    """``run(..., middleware=[...])``."""

    def test_a_per_call_hook_runs_for_that_call(self):
        recorder = Recorder("call")
        agent = _agent()
        try:
            agent.run("What is 6*7?", middleware=[recorder])
        finally:
            agent.close()
        assert "call:before_run" in recorder.seen

    def test_a_per_call_hook_does_not_apply_to_the_next_call(self):
        recorder = Recorder("call")
        agent = _agent()
        try:
            agent.run("What is 6*7?", middleware=[recorder])
            before = len(recorder.seen)
            agent.model = MockModel(list(TOOL_TURNS))
            agent.run("What is 6*7?")
        finally:
            agent.close()
        assert len(recorder.seen) == before

    def test_configured_and_per_call_hooks_both_run(self):
        configured, per_call = Recorder("cfg"), Recorder("call")
        agent = _agent(configured)
        try:
            agent.run("What is 6*7?", middleware=[per_call])
        finally:
            agent.close()
        assert configured.seen and per_call.seen


class TestShippedMiddleware:
    """The ones in the box."""

    def test_logging_middleware_reports_the_run_and_leaves_it_alone(self, caplog):
        agent = _agent(LoggingMiddleware())
        try:
            with caplog.at_level("INFO", logger="effgen.core.middleware"):
                response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert response.success
        lines = [record.getMessage() for record in caplog.records]
        assert any(line.startswith("run start:") for line in lines)
        assert any(line.startswith("tool call: calculator") for line in lines)
        assert any(line.startswith("run end:") for line in lines)

    def test_an_approval_gate_lets_an_approved_call_through(self):
        agent = _agent(ToolApprovalMiddleware(approve=lambda name, arg: True))
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert "42" in (response.tool_calls[0].result or "")

    def test_an_approval_gate_stops_a_refused_call(self):
        agent = _agent(ToolApprovalMiddleware(approve=lambda name, arg: False))
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert "not approved" in (response.tool_calls[0].result or "")

    def test_an_approval_gate_only_asks_about_the_tools_it_names(self):
        asked: list[str] = []

        def approve(name, arg):
            asked.append(name)
            return True

        agent = _agent(ToolApprovalMiddleware(approve=approve, tools=["web_search"]))
        try:
            agent.run("What is 6*7?")
        finally:
            agent.close()
        assert asked == []


class TestContexts:
    """What a hook is told."""

    def test_a_tool_context_names_the_tool_and_its_input(self):
        seen: list[ToolCallContext] = []

        class Capture(AgentMiddleware):
            def before_tool_call(self, ctx):
                seen.append(ctx)
                return None

        agent = _agent(Capture())
        try:
            agent.run("What is 6*7?")
        finally:
            agent.close()
        assert seen[0].tool_name == "calculator"
        assert "6*7" in seen[0].tool_input

    def test_a_model_context_names_the_model(self):
        seen: list[ModelCallContext] = []

        class Capture(AgentMiddleware):
            def before_model_call(self, ctx):
                seen.append(ctx)
                return None

        agent = _agent(Capture(), tools=False)
        try:
            agent.run("hello")
        finally:
            agent.close()
        assert seen[0].model_name

    def test_a_hook_can_pass_values_to_its_own_after_hook(self):
        class Timed(AgentMiddleware):
            def before_run(self, ctx):
                ctx.metadata["marker"] = "set-in-before"
                return None

            def after_run(self, ctx, response):
                response.metadata["marker"] = ctx.metadata.get("marker")
                return response

        agent = _agent(Timed())
        try:
            response = agent.run("What is 6*7?")
        finally:
            agent.close()
        assert response.metadata["marker"] == "set-in-before"


class TestFailure:
    """What happens when a hook raises."""

    def test_a_raising_hook_reaches_the_caller(self):
        class Refuse(AgentMiddleware):
            def before_run(self, ctx):
                raise PermissionError("this agent may not run here")

        agent = _agent(Refuse())
        try:
            with pytest.raises(PermissionError, match="may not run here"):
                agent.run("What is 6*7?")
        finally:
            agent.close()
