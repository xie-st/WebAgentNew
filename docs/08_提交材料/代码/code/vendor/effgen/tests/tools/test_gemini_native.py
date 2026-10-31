"""
Unit tests for Gemini native tools (GoogleSearchTool, GeminiUrlContextTool,
GeminiCodeExecutionTool) and their integration with GeminiAdapter / Agent.

All tests are mock-based (no live API calls).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.errors import ToolIncompatibleError
from effgen.tools.builtin.gemini_native import (
    GeminiCodeExecutionTool,
    GeminiNativeTool,
    GeminiUrlContextTool,
    GoogleSearchTool,
)

# ---------------------------------------------------------------------------
# Instantiation + metadata
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_google_search_tool_name(self):
        t = GoogleSearchTool()
        assert t.name == "google_search"

    def test_url_context_tool_name(self):
        t = GeminiUrlContextTool()
        assert t.name == "url_context"

    def test_code_execution_tool_name(self):
        t = GeminiCodeExecutionTool()
        assert t.name == "code_execution"

    def test_is_gemini_native_sentinel(self):
        for cls in (GoogleSearchTool, GeminiUrlContextTool, GeminiCodeExecutionTool):
            t = cls()
            assert t.IS_GEMINI_NATIVE is True
            assert isinstance(t, GeminiNativeTool)

    def test_local_execute_raises(self):
        for cls in (GoogleSearchTool, GeminiUrlContextTool, GeminiCodeExecutionTool):
            t = cls()
            import asyncio
            with pytest.raises(RuntimeError, match="server-side"):
                asyncio.run(t._execute())


# ---------------------------------------------------------------------------
# to_gemini_tool() spec shapes
# ---------------------------------------------------------------------------

class TestGeminiToolSpecs:
    def test_google_search_spec_has_google_search(self):
        from google.genai import types
        spec = GoogleSearchTool().to_gemini_tool()
        assert isinstance(spec, types.Tool)
        assert spec.google_search is not None

    def test_url_context_spec_has_url_context(self):
        from google.genai import types
        spec = GeminiUrlContextTool().to_gemini_tool()
        assert isinstance(spec, types.Tool)
        assert spec.url_context is not None

    def test_code_execution_spec_has_code_execution(self):
        from google.genai import types
        spec = GeminiCodeExecutionTool().to_gemini_tool()
        assert isinstance(spec, types.Tool)
        assert spec.code_execution is not None


# ---------------------------------------------------------------------------
# _convert_tools_to_genai — native tool routing
# ---------------------------------------------------------------------------

class TestConvertToolsToGenAI:
    def test_native_tool_is_converted_to_sdk_tool(self):
        from google.genai import types

        from effgen.models.gemini_adapter import GeminiAdapter
        result = GeminiAdapter._convert_tools_to_genai([GoogleSearchTool()])
        assert len(result) == 1
        assert isinstance(result[0], types.Tool)
        assert result[0].google_search is not None

    def test_url_context_converted(self):
        from effgen.models.gemini_adapter import GeminiAdapter
        result = GeminiAdapter._convert_tools_to_genai([GeminiUrlContextTool()])
        assert result[0].url_context is not None

    def test_code_execution_converted(self):
        from effgen.models.gemini_adapter import GeminiAdapter
        result = GeminiAdapter._convert_tools_to_genai([GeminiCodeExecutionTool()])
        assert result[0].code_execution is not None

    def test_mixed_native_and_function(self):
        from google.genai import types

        from effgen.models.gemini_adapter import GeminiAdapter
        func_tool = {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Do math",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        result = GeminiAdapter._convert_tools_to_genai([GoogleSearchTool(), func_tool])
        # One passthrough Tool for google_search, one Tool with function declarations
        sdk_tools = [t for t in result if isinstance(t, types.Tool)]
        assert len(sdk_tools) == 2
        has_search = any(t.google_search is not None for t in sdk_tools)
        has_fn = any(t.function_declarations for t in sdk_tools)
        assert has_search
        assert has_fn


# ---------------------------------------------------------------------------
# Parallel tool-call parsing
# ---------------------------------------------------------------------------

class TestParallelToolCallParsing:
    """Test that generate() encodes ALL parallel tool calls as <tool_call> blocks."""

    def _make_part(self, name=None, args=None, text=None, thought=False):
        """Create a mock SDK response Part."""
        part = MagicMock()
        part.thought = thought
        part.text = text

        if name:
            fc = MagicMock()
            fc.name = name
            fc.args = args or {}
            part.function_call = fc
        else:
            part.function_call = None

        part.code_execution_result = None
        return part

    def _make_response(self, parts):
        candidate = MagicMock()
        candidate.content.parts = parts
        candidate.finish_reason = "STOP"
        candidate.grounding_metadata = None

        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 20
        response.usage_metadata.total_token_count = 30
        response.grounding_metadata = None
        return response

    def _make_adapter(self):
        from effgen.models.base import ModelType, TokenCount
        from effgen.models.gemini_adapter import GeminiAdapter
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        adapter._is_loaded = True
        adapter.client = MagicMock()
        adapter.model_name = "gemini-2.5-flash"
        adapter.model_type = ModelType.GEMINI
        adapter.total_cost = 0.0
        adapter.total_tokens = 0
        adapter.safety_settings = None
        adapter._context_length = 1_000_000
        # Stub count_tokens to return a real TokenCount
        adapter.count_tokens = MagicMock(return_value=TokenCount(count=5, model_name="gemini-2.5-flash"))
        return adapter

    @patch("effgen.models.gemini_adapter.GeminiAdapter._generate_with_retry")
    def test_single_tool_call_encoded(self, mock_gen):
        adapter = self._make_adapter()

        mock_gen.return_value = self._make_response([
            self._make_part(name="search", args={"query": "hello"}),
        ])

        result = adapter.generate("test prompt")
        assert "<tool_call>" in result.text
        assert '"name": "search"' in result.text
        call = result.metadata["tool_calls"][0]
        # The shape every adapter reports...
        assert call["function"]["name"] == "search"
        assert json.loads(call["function"]["arguments"]) == {"query": "hello"}
        # ...beside the flat keys a caller written against the earlier Gemini
        # shape reads.
        assert call["name"] == "search"
        assert call["arguments"] == {"query": "hello"}

    @patch("effgen.models.gemini_adapter.GeminiAdapter._generate_with_retry")
    def test_parallel_two_tool_calls_both_encoded(self, mock_gen):
        adapter = self._make_adapter()

        mock_gen.return_value = self._make_response([
            self._make_part(name="add", args={"a": 2, "b": 3}),
            self._make_part(name="mul", args={"a": 7, "b": 8}),
        ])

        result = adapter.generate("What is 2+3 AND 7*8?")
        # Both calls must be in the text
        assert result.text.count("<tool_call>") == 2
        assert '"name": "add"' in result.text
        assert '"name": "mul"' in result.text
        assert len(result.metadata["tool_calls"]) == 2

    @patch("effgen.models.gemini_adapter.GeminiAdapter._generate_with_retry")
    def test_code_execution_result_in_text(self, mock_gen):
        adapter = self._make_adapter()

        cer = MagicMock()
        cer.output = "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000\n"

        part = MagicMock()
        part.thought = False
        part.text = "Here is the result:"
        part.function_call = None
        part.code_execution_result = cer

        mock_gen.return_value = self._make_response([part])

        result = adapter.generate("compute 100!")
        assert "93326215" in result.text


# ---------------------------------------------------------------------------
# ToolIncompatibleError — Gemini tools with non-Gemini model
# ---------------------------------------------------------------------------

class TestToolIncompatibleError:
    def _make_openai_mock(self):
        from effgen.models.openai_adapter import OpenAIAdapter
        m = MagicMock(spec=OpenAIAdapter)
        m.model_name = "gpt-4o"
        m.model_type = MagicMock()
        m._context_length = 128_000
        m.get_context_length.return_value = 128_000
        m.supports_tool_calling.return_value = True
        m.supports_function_calling.return_value = True
        m._is_loaded = True
        return m

    def _make_cerebras_mock(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter
        m = MagicMock(spec=CerebrasAdapter)
        m.model_name = "llama3.1-8b"
        m.model_type = MagicMock()
        m._context_length = 128_000
        m.get_context_length.return_value = 128_000
        m.supports_tool_calling.return_value = True
        m.supports_function_calling.return_value = True
        m._is_loaded = True
        return m

    def _make_gemini_mock(self):
        from effgen.models.gemini_adapter import GeminiAdapter
        m = MagicMock(spec=GeminiAdapter)
        m.model_name = "gemini-2.5-flash"
        m.model_type = MagicMock()
        m._context_length = 1_000_000
        m.get_context_length.return_value = 1_000_000
        m.supports_tool_calling.return_value = True
        m.supports_function_calling.return_value = True
        m._is_loaded = True
        return m

    def test_google_search_with_non_gemini_raises(self):
        """GoogleSearchTool must raise ToolIncompatibleError with OpenAI model."""
        from effgen.core.agent import Agent, AgentConfig

        with pytest.raises(ToolIncompatibleError) as exc_info:
            Agent(AgentConfig(
                name="test-agent",
                model=self._make_openai_mock(),
                tools=[GoogleSearchTool()],
            ))
        assert exc_info.value.tool_name == "google_search"

    def test_google_search_with_cerebras_raises(self):
        from effgen.core.agent import Agent, AgentConfig

        with pytest.raises(ToolIncompatibleError):
            Agent(AgentConfig(
                name="test-agent",
                model=self._make_cerebras_mock(),
                tools=[GoogleSearchTool()],
            ))

    def test_all_three_gemini_native_tools_raise_with_non_gemini(self):
        from effgen.core.agent import Agent, AgentConfig

        for tool_cls in (GoogleSearchTool, GeminiUrlContextTool, GeminiCodeExecutionTool):
            with pytest.raises(ToolIncompatibleError):
                Agent(AgentConfig(
                    name="test-agent",
                    model=self._make_openai_mock(),
                    tools=[tool_cls()],
                ))

    def test_google_search_with_gemini_adapter_does_not_raise(self):
        """GoogleSearchTool must NOT raise when used with a GeminiAdapter."""
        from effgen.core.agent import Agent, AgentConfig

        agent = Agent(AgentConfig(
            name="test-agent",
            model=self._make_gemini_mock(),
            tools=[GoogleSearchTool()],
        ))
        assert "google_search" in agent.tools


# ---------------------------------------------------------------------------
# A turn that only called tools still gets answered
# ---------------------------------------------------------------------------

class TestNativeToolFollowUp:
    """A Gemini turn whose whole text is a tool-call block.

    The adapter encodes every call it reports as a ``<tool_call>`` block in the
    generated text as well, so such a turn has non-empty text but no answer in
    it. The agent runs the local tool, then has to re-prompt with the result:
    without that, the tool output is discarded and the run reports a
    placeholder while claiming to have succeeded.

    The payloads below are what ``gemini-3.1-flash-lite`` actually returns for
    a function call.
    """

    def _agent_with_stubbed_turn(self, first_text, follow_up_text="The code is ZQ7."):
        from effgen.core.agent import Agent, AgentConfig
        from effgen.models.base import GenerationResult
        from effgen.models.gemini_adapter import GeminiAdapter
        from effgen.tools import tool

        calls: list[str] = []

        @tool(name="vault_lookup", description="Look up a vault code.")
        def vault_lookup(slot: str) -> str:
            calls.append(slot)
            return "ZQ7"

        model = MagicMock(spec=GeminiAdapter)
        model.model_name = "gemini-3.1-flash-lite"
        model.model_type = MagicMock()
        model._context_length = 1_000_000
        model.get_context_length.return_value = 1_000_000
        model.supports_tool_calling.return_value = True
        model.supports_function_calling.return_value = True
        model._is_loaded = True

        tool_call = {
            "id": "8ysinXqc",
            "type": "function",
            "function": {"name": "vault_lookup",
                         "arguments": '{"slot": "alpha"}'},
            "name": "vault_lookup",
            "arguments": {"slot": "alpha"},
        }
        model.generate.side_effect = [
            GenerationResult(text=first_text, tokens_used=10, finish_reason="stop",
                             model_name="gemini-3.1-flash-lite",
                             metadata={"tool_calls": [tool_call]}),
            GenerationResult(text=follow_up_text, tokens_used=8, finish_reason="stop",
                             model_name="gemini-3.1-flash-lite",
                             metadata={"tool_calls": []}),
        ]

        agent = Agent(AgentConfig(
            name="test-agent", model=model,
            tools=[GeminiCodeExecutionTool(), vault_lookup],
            tool_calling_mode="native",
            # One case here asserts the *failure response* rather than the
            # raise. ``raise_on_error`` defaults to True as of 1.0.0, so a run
            # that fails raises before the assertion can read it.
            raise_on_error=False,
        ))
        return agent, calls

    def test_a_tool_only_turn_is_answered_from_the_tool_result(self):
        agent, calls = self._agent_with_stubbed_turn(
            '<tool_call>{"name": "vault_lookup", "arguments": {"slot": "alpha"}}</tool_call>'
        )

        response = agent.run("Look up the vault code for slot alpha.")

        assert calls == ["alpha"], "the local tool was never executed"
        assert response.success is True
        assert "ZQ7" in response.text
        assert "no output" not in response.text
        assert response.metadata["tool_calling_strategy"] == "gemini_native"

    def test_a_turn_that_answers_directly_is_not_re_prompted(self):
        agent, _calls = self._agent_with_stubbed_turn("The code is ZQ7 already.")

        response = agent.run("Look up the vault code for slot alpha.")

        assert response.success is True
        assert response.text == "The code is ZQ7 already."

    def test_an_unanswerable_tool_only_turn_is_not_reported_as_success(self):
        """The follow-up produced nothing either — that is a failed turn."""
        agent, _calls = self._agent_with_stubbed_turn(
            '<tool_call>{"name": "vault_lookup", "arguments": {"slot": "alpha"}}</tool_call>',
            follow_up_text="",
        )

        response = agent.run("Look up the vault code for slot alpha.")

        assert response.success is False
        assert "no output" in response.text


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_top_level_exports(self):
        from effgen import (
            GeminiCodeExecutionTool as GCET,
        )
        from effgen import (
            GeminiNativeTool as GNT,
        )
        from effgen import (
            GeminiUrlContextTool as GUCT,
        )
        from effgen import (
            GoogleSearchTool as GST,
        )
        assert GST is GoogleSearchTool
        assert GUCT is GeminiUrlContextTool
        assert GCET is GeminiCodeExecutionTool
        assert GNT is GeminiNativeTool

    def test_builtin_init_exports(self):
        from effgen.tools.builtin import (
            GeminiCodeExecutionTool as GCET,
        )
        from effgen.tools.builtin import (
            GeminiUrlContextTool as GUCT,
        )
        from effgen.tools.builtin import (
            GoogleSearchTool as GST,
        )
        assert GST is GoogleSearchTool
        assert GUCT is GeminiUrlContextTool
        assert GCET is GeminiCodeExecutionTool
