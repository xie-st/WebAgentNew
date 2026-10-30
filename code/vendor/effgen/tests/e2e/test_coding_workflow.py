"""End-to-end tests for coding workflow."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import JSONTool


@pytest.mark.gpu
class TestCodingWorkflow:
    """Test coding-related agent workflows."""

    def test_json_processing(self, real_model):
        agent = Agent(config=AgentConfig(
            name="coding_e2e",
            model=real_model,
            tools=[JSONTool()],
            max_iterations=5,
            enable_memory=False,
            enable_sub_agents=False,
        ))
        result = agent.run('How many keys are in this JSON: {"name": "Alice", "age": 30, "city": "NYC"}')
        assert result.success
        assert "3" in result.output or "3" in str(result.execution_trace)
