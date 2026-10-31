"""Unit tests for prompt engineering modules."""

from effgen.prompts.agent_system_prompt import AgentSystemPromptBuilder
from effgen.prompts.tool_prompt_generator import ToolPromptGenerator


class TestToolPromptGenerator:
    """Tests for ToolPromptGenerator."""

    def test_create_with_no_tools(self):
        gen = ToolPromptGenerator(tools=[])
        section = gen.generate_tools_section()
        assert isinstance(section, str)

    def test_create_with_tools(self, calculator, datetime_tool):
        gen = ToolPromptGenerator(tools=[calculator, datetime_tool])
        section = gen.generate_tools_section()
        assert "calculator" in section.lower()
        assert "datetime" in section.lower()

    def test_model_family_detection_qwen(self):
        gen = ToolPromptGenerator(tools=[], model_name="Qwen/Qwen2.5-3B-Instruct")
        assert gen._model_family == "qwen"

    def test_model_family_detection_llama(self):
        gen = ToolPromptGenerator(tools=[], model_name="meta-llama/Llama-3-8B")
        assert gen._model_family == "llama"

    def test_model_family_detection_unknown(self):
        gen = ToolPromptGenerator(tools=[], model_name="unknown-model")
        assert gen._model_family == "generic"

    def test_verbose_vs_compact(self, calculator):
        gen = ToolPromptGenerator(tools=[calculator])
        verbose = gen.generate_tools_section(verbose=True)
        compact = gen.generate_tools_section(verbose=False)
        assert len(verbose) >= len(compact)


class TestAgentSystemPromptBuilder:
    """Tests for AgentSystemPromptBuilder."""

    def test_instantiation(self):
        builder = AgentSystemPromptBuilder()
        assert builder is not None

    def test_build_with_no_tools(self):
        builder = AgentSystemPromptBuilder()
        prompt = builder.build(tools=[], agent_name="test")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_with_tools(self, calculator):
        builder = AgentSystemPromptBuilder()
        prompt = builder.build(tools=[calculator], agent_name="test")
        assert isinstance(prompt, str)
        # Should contain category instructions for computation tools
        assert "calculator" in prompt.lower() or "computation" in prompt.lower() or "tool" in prompt.lower()
