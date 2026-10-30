"""End-to-end tests for math workflow."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator


@pytest.mark.gpu
class TestMathWorkflow:
    """Test complete math problem-solving workflow."""

    def test_simple_arithmetic(self, real_model):
        agent = Agent(config=AgentConfig(
            name="math_e2e",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is 15 * 23?")
        assert result.success
        trace_str = str(result.execution_trace)
        assert "345" in result.output or "345" in trace_str

    def test_multi_step_calculation(self, real_model):
        agent = Agent(config=AgentConfig(
            name="math_multi",
            model=real_model,
            tools=[Calculator()],
            max_iterations=8,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is (100 + 50) / 3?")
        assert result.success
        trace_str = str(result.execution_trace)
        assert "50" in result.output or "50" in trace_str
