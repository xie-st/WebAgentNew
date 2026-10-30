"""Integration tests for calculator agent with real model."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator


@pytest.mark.gpu
class TestCalculatorAgent:
    """Test calculator tool with real Qwen model."""

    def test_simple_multiplication(self, real_model):
        agent = Agent(config=AgentConfig(
            name="calc_test",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is 15 * 23?", max_tokens=256)
        assert result.success
        assert "345" in result.output

    def test_addition(self, real_model):
        agent = Agent(config=AgentConfig(
            name="calc_test",
            model=real_model,
            tools=[Calculator()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is 127 + 389?", max_tokens=256)
        assert result.success
        assert "516" in result.output

    def test_direct_answer(self, real_model):
        """Test that the model can answer without tools when appropriate."""
        agent = Agent(config=AgentConfig(
            name="direct_test",
            model=real_model,
            tools=[],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run("What is the capital of France?", max_tokens=128)
        assert result.success
        assert "paris" in result.output.lower()
