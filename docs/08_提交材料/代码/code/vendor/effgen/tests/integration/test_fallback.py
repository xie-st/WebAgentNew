"""Integration tests for tool fallback chains."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator


@pytest.mark.gpu
class TestFallback:
    """Test fallback chain functionality."""

    def test_agent_with_fallback_enabled(self, real_model):
        agent = Agent(config=AgentConfig(
            name="fallback_test",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
            enable_fallback=True,
        ))
        result = agent.run("What is 10 + 20?")
        assert result.success
        # The model is non-deterministic: sometimes it emits a clean
        # "30" in the final output, sometimes it runs out of iterations
        # with a dangling ReAct step. Either is fine as long as the
        # calculator tool actually computed 30 during the trace.
        found = "30" in (result.output or "")
        if not found:
            for ev in result.execution_trace or []:
                data = ev.get("data") or {}
                if "30" in str(data.get("result", "")):
                    found = True
                    break
        assert found, f"expected '30' in output or trace, got: {result.output!r}"

    def test_agent_with_fallback_disabled(self, real_model):
        agent = Agent(config=AgentConfig(
            name="no_fallback_test",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
            enable_fallback=False,
        ))
        result = agent.run("What is 10 + 20?")
        assert result.success
        # The model is non-deterministic: sometimes it emits a clean
        # "30" in the final output, sometimes it runs out of iterations
        # with a dangling ReAct step. Either is fine as long as the
        # calculator tool actually computed 30 during the trace.
        found = "30" in (result.output or "")
        if not found:
            for ev in result.execution_trace or []:
                data = ev.get("data") or {}
                if "30" in str(data.get("result", "")):
                    found = True
                    break
        assert found, f"expected '30' in output or trace, got: {result.output!r}"
