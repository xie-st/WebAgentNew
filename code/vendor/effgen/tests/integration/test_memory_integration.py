"""Integration tests for memory with real model."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig


@pytest.mark.gpu
class TestMemoryIntegration:
    """Test memory across multi-turn conversations."""

    def test_multi_turn_memory(self, real_model):
        agent = Agent(config=AgentConfig(
            name="memory_test",
            model=real_model,
            tools=[],
            max_iterations=5,
            enable_memory=True,
            enable_sub_agents=False,
        ))
        # Turn 1: Introduce information
        result1 = agent.run("Remember this: My name is Alice and I am 30 years old. Please confirm.")
        assert result1.success

        # Turn 2: Recall information — check both output and trace
        result2 = agent.run("Based on what I told you, what is my name?")
        assert result2.success
        combined = (result2.output + str(result2.execution_trace)).lower()
        assert "alice" in combined
