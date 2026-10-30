"""Unit tests for the Agent class."""

import pytest

from effgen.core.agent import Agent, AgentConfig, AgentMode, AgentResponse
from tests.fixtures.mock_models import MockToolCallingModel


@pytest.fixture(autouse=True)
def _reset_reasoning_budget_warned():
    """The reasoning-budget heads-up fires once per (model, kind) per process.
    Clear that record around each test so a test asserting the warning is not
    silenced by a warning an earlier test already emitted for the same model."""
    from effgen.core import agent_generation
    agent_generation._reasoning_budget_warned.clear()
    yield
    agent_generation._reasoning_budget_warned.clear()


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_config(self, mock_model):
        config = AgentConfig(name="test", model=mock_model)
        assert config.name == "test"
        assert config.max_iterations == 10
        assert config.temperature == 0.7
        assert config.enable_sub_agents is True
        assert config.enable_memory is True
        assert config.enable_streaming is False
        assert config.tools == []
        assert config.enable_fallback is True

    def test_custom_config(self, mock_model):
        config = AgentConfig(
            name="custom",
            model=mock_model,
            max_iterations=3,
            temperature=0.1,
            enable_sub_agents=False,
            enable_memory=False,
        )
        assert config.max_iterations == 3
        assert config.temperature == 0.1
        assert config.enable_sub_agents is False

    def test_memory_config_defaults(self, mock_model):
        config = AgentConfig(name="test", model=mock_model)
        assert "short_term_max_tokens" in config.memory_config
        assert "long_term_backend" in config.memory_config


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_basic_init(self, mock_model):
        config = AgentConfig(name="test", model=mock_model, enable_memory=False, enable_sub_agents=False)
        agent = Agent(config=config)
        assert agent.name == "test"
        assert agent.model is mock_model

    def test_init_with_tools(self, mock_model, calculator, datetime_tool):
        config = AgentConfig(
            name="test",
            model=mock_model,
            tools=[calculator, datetime_tool],
            enable_memory=False,
            enable_sub_agents=False,
        )
        agent = Agent(config=config)
        assert "calculator" in agent.tools
        assert "datetime" in agent.tools

    def test_init_with_unresolvable_string_model_defers_the_error(self):
        config = AgentConfig(
            name="test",
            model="nonexistent/model",
            require_model=False,
            enable_memory=False,
            enable_sub_agents=False,
        )
        agent = Agent(config=config)
        assert agent.model is None


class TestAgentRun:
    """Tests for Agent.run() method."""

    def test_simple_run(self, basic_agent):
        result = basic_agent.run("What is the answer?")
        assert isinstance(result, AgentResponse)
        assert result.success is True
        assert result.output is not None
        assert len(result.output) > 0

    def test_run_with_tool(self, tool_agent):
        result = tool_agent.run("What is 2 + 2?")
        assert isinstance(result, AgentResponse)
        assert result.success is True

    def test_simple_calculator_task_returns_tool_result_directly(self, calculator):
        model = MockToolCallingModel([
            {
                "thought": "I need to calculate this.",
                "action": "calculator",
                "action_input": '{"expression": "15 * 23"}',
            },
            {
                "thought": "I should do another calculation.",
                "action": "calculator",
                "action_input": '{"expression": "345 + 50"}',
            },
        ])
        agent = Agent(config=AgentConfig(
            name="direct-calc",
            model=model,
            tools=[calculator],
            enable_memory=False,
            enable_sub_agents=False,
        ))

        result = agent.run("What is 15 * 23?")

        assert result.success is True
        assert result.output == "345"
        assert result.tool_calls == 1
        assert model.call_count == 1

    def test_response_has_metadata(self, basic_agent):
        result = basic_agent.run("test")
        assert hasattr(result, "iterations")
        assert hasattr(result, "tool_calls")
        assert hasattr(result, "execution_time")
        assert result.execution_time >= 0

    def test_response_to_dict(self, basic_agent):
        result = basic_agent.run("test")
        d = result.to_dict()
        assert "output" in d
        assert "success" in d
        assert "mode" in d
        assert "iterations" in d

    def test_run_forwards_sampling_kwargs_to_generation_config(self, mock_model):
        """seed/top_k/penalties passed to run() must reach the model's config,
        matching the temperature/top_p/stop_sequences that already worked."""
        config = AgentConfig(
            name="sampling-test", model=mock_model, enable_memory=False, enable_sub_agents=False,
        )
        agent = Agent(config=config)
        agent.run(
            "test",
            seed=99,
            top_k=7,
            presence_penalty=0.3,
            frequency_penalty=1.2,
            repetition_penalty=1.1,
        )
        gen_config = mock_model._generate_calls[-1]["config"]
        assert gen_config.seed == 99
        assert gen_config.top_k == 7
        assert gen_config.presence_penalty == 0.3
        assert gen_config.frequency_penalty == 1.2
        assert gen_config.repetition_penalty == 1.1

    def test_agent_config_pins_sampling_defaults_for_every_run(self, mock_model):
        """Pinning seed/penalties on AgentConfig applies to every run(), not just
        a single call — matching how `temperature` already behaves."""
        config = AgentConfig(
            name="pinned-sampling", model=mock_model, enable_memory=False, enable_sub_agents=False,
            seed=7, frequency_penalty=0.8,
        )
        agent = Agent(config=config)
        agent.run("test")
        gen_config = mock_model._generate_calls[-1]["config"]
        assert gen_config.seed == 7
        assert gen_config.frequency_penalty == 0.8

    def test_run_rejects_unknown_kwarg(self, basic_agent):
        """A mistyped/unknown run() kwarg must raise, not be silently ignored."""
        import pytest as _pytest

        with _pytest.raises(TypeError, match="totally_made_up_param"):
            basic_agent.run("test", totally_made_up_param=123)

    def test_run_rejects_unknown_kwarg_with_close_match_hint(self, basic_agent):
        import pytest as _pytest

        with _pytest.raises(TypeError, match="Did you mean 'seed'"):
            basic_agent.run("test", seeed=1)

    def test_warn_reasoning_budget_fires_for_tight_pinned_budget(self, mock_model, caplog):
        import logging

        mock_model.model_name = "gpt-5-nano"
        config = AgentConfig(
            name="reasoning-budget", model=mock_model, enable_memory=False, enable_sub_agents=False,
        )
        agent = Agent(config=config)
        with caplog.at_level(logging.WARNING, logger="effgen.core.agent_generation"):
            agent.run("test", max_tokens=250)
        assert any("reasoning model" in r.message for r in caplog.records)
        # The hint also sets expectations about temperature=0 on a tight
        # budget: it does not make reasoning-budget exhaustion deterministic.
        assert any("temperature=0" in r.message for r in caplog.records)

    def test_warn_reasoning_budget_silent_for_non_reasoning_model(self, mock_model, caplog):
        import logging

        config = AgentConfig(
            name="non-reasoning-budget", model=mock_model, enable_memory=False, enable_sub_agents=False,
        )
        agent = Agent(config=config)
        with caplog.at_level(logging.WARNING, logger="effgen.core.agent_generation"):
            agent.run("test", max_tokens=250)
        assert not any("reasoning model" in r.message for r in caplog.records)


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_default_response(self):
        resp = AgentResponse(output="hello")
        assert resp.output == "hello"
        assert resp.success is True
        assert resp.mode == AgentMode.SINGLE
        assert resp.iterations == 0
        assert resp.tool_calls == 0

    def test_response_to_dict(self):
        resp = AgentResponse(output="test", iterations=3, tool_calls=1)
        d = resp.to_dict()
        assert d["output"] == "test"
        assert d["iterations"] == 3
        assert d["tool_calls"] == 1
