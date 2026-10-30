"""Integration tests for streaming with real model."""

import pytest

from effgen import Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator


@pytest.mark.gpu
class TestStreaming:
    """Test real token streaming."""

    def test_stream_produces_chunks(self, streaming_model):
        agent = Agent(config=AgentConfig(
            name="stream_test",
            model=streaming_model,
            tools=[Calculator()],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
            enable_streaming=True,
        ))
        chunks = list(agent.stream("What is 2 + 2?"))
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    def test_stream_with_callbacks(self, streaming_model):
        agent = Agent(config=AgentConfig(
            name="callback_test",
            model=streaming_model,
            tools=[Calculator()],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
            enable_streaming=True,
        ))
        thoughts = []
        answers = []
        chunks = list(agent.stream(
            "What is 5 + 3?",
            on_thought=lambda t: thoughts.append(t),
            on_answer=lambda a: answers.append(a),
        ))
        assert len(chunks) > 0
