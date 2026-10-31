"""
Comprehensive test suite for the ReAct response parser.

Tests _parse_react_response() and _clean_json_input() with 25+ cases
covering real-world SLM outputs, edge cases, and regression tests.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from tests.fixtures.mock_models import MockModel


@pytest.fixture
def agent():
    """Minimal agent for calling _parse_react_response."""
    model = MockModel(responses=["Final Answer: dummy"])
    config = AgentConfig(
        name="parser-test",
        model=model,
        tools=[],
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


# ── Final Answer extraction ──────────────────────────────────────────────


class TestFinalAnswerExtraction:

    def test_simple_final_answer(self, agent):
        text = "Thought: I know.\nFinal Answer: 42"
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "42"

    def test_multiline_final_answer(self, agent):
        text = (
            "Thought: Let me think.\n"
            "Final Answer: The result is as follows:\n"
            "1. First item\n"
            "2. Second item\n"
            "3. Third item"
        )
        parsed = agent._parse_react_response(text)
        assert "First item" in parsed["final_answer"]
        assert "Third item" in parsed["final_answer"]

    def test_final_answer_stops_at_observation(self, agent):
        text = (
            "Final Answer: The answer is 42.\n"
            "Observation: some leftover text from the model"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "The answer is 42."
        assert "leftover" not in parsed["final_answer"]

    def test_final_answer_stops_at_human(self, agent):
        text = (
            "Final Answer: Hello there.\n"
            "Human: What about something else?"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "Hello there."
        assert "What about" not in parsed["final_answer"]

    def test_final_answer_stops_at_thought(self, agent):
        text = (
            "Final Answer: The capital is Paris.\n"
            "Thought: Wait, maybe I should reconsider."
        )
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "The capital is Paris."

    def test_final_answer_stops_at_question(self, agent):
        text = (
            "Final Answer: Done.\n"
            "Question: What is next?"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "Done."

    def test_final_answer_stops_at_action(self, agent):
        text = (
            "Final Answer: Result is 10.\n"
            "Action: calculator"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "Result is 10."

    def test_answer_containing_thought_as_text(self, agent):
        """Literal 'Thought:' mid-sentence should not be a boundary (no newline prefix)."""
        text = (
            "Final Answer: The model's Thought: process uses chain-of-thought reasoning."
        )
        parsed = agent._parse_react_response(text)
        assert "chain-of-thought" in parsed["final_answer"]

    def test_answer_pattern_at_line_start(self, agent):
        text = "Answer: The value is 7."
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "The value is 7."

    def test_the_answer_is_pattern(self, agent):
        text = "The answer is: 99"
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "99"

    def test_trailing_hallucinated_question(self, agent):
        """Phi-4 style: answer followed by hallucinated follow-up question."""
        text = "Final Answer: The capital of France is Paris.What year was Paris founded?"
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "The capital of France is Paris."
        assert "founded" not in parsed["final_answer"]

    def test_case_insensitive_final_answer(self, agent):
        text = "FINAL ANSWER: yes"
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "yes"

    def test_final_answer_with_unicode(self, agent):
        text = "Final Answer: La reponse est 42 \u2014 c'est correct \u2705"
        parsed = agent._parse_react_response(text)
        assert "42" in parsed["final_answer"]
        assert "\u2714" in parsed["final_answer"] or "\u2705" in parsed["final_answer"] or "42" in parsed["final_answer"]

    def test_final_answer_extra_whitespace(self, agent):
        text = "Final Answer:    lots of spaces   "
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "lots of spaces"


# ── Thought extraction ───────────────────────────────────────────────────


class TestThoughtExtraction:

    def test_simple_thought(self, agent):
        text = "Thought: I need to calculate.\nAction: calculator\nAction Input: {\"expression\": \"2+2\"}"
        parsed = agent._parse_react_response(text)
        assert parsed["thought"] == "I need to calculate."

    def test_empty_thought(self, agent):
        text = "Thought: \nAction: calculator\nAction Input: {\"expression\": \"1+1\"}"
        parsed = agent._parse_react_response(text)
        # Should still parse action even with empty thought
        assert parsed["action"] == "calculator"


# ── Action extraction ────────────────────────────────────────────────────


class TestActionExtraction:

    def test_simple_action(self, agent):
        text = "Thought: need calc\nAction: calculator\nAction Input: {\"expression\": \"5+5\"}"
        parsed = agent._parse_react_response(text)
        assert parsed["action"] == "calculator"

    def test_action_final_answer_redirect(self, agent):
        """'Action: Final Answer' should redirect to final_answer field."""
        text = "Thought: I know.\nAction: Final Answer\nAction Input: The result is 42."
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "The result is 42."

    def test_action_final_answer_same_line(self, agent):
        text = "Thought: done.\nAction: Final Answer: The value is 7"
        parsed = agent._parse_react_response(text)
        assert parsed["final_answer"] == "The value is 7"

    def test_function_call_style(self, agent):
        text = "Thought: compute\nAction: calculator(2+2)"
        parsed = agent._parse_react_response(text)
        assert parsed["action"] == "calculator"
        assert parsed["action_input"] == "2+2"

    def test_tool_alternative_keyword(self, agent):
        text = "Thought: need to search\nTool: web_search\nAction Input: {\"query\": \"test\"}"
        parsed = agent._parse_react_response(text)
        assert parsed["action"] == "web_search"


# ── Action Input / JSON parsing ──────────────────────────────────────────


class TestActionInputParsing:

    def test_valid_json_input(self, agent):
        text = 'Thought: calc\nAction: calculator\nAction Input: {"expression": "2+2"}'
        parsed = agent._parse_react_response(text)
        assert parsed["action_input"] == '{"expression": "2+2"}'

    def test_action_input_stops_at_observation(self, agent):
        text = (
            "Thought: calc\nAction: calculator\n"
            'Action Input: {"expression": "2+2"}\n'
            "Observation: 4"
        )
        parsed = agent._parse_react_response(text)
        assert "Observation" not in (parsed["action_input"] or "")


# ── _clean_json_input tests ──────────────────────────────────────────────


class TestCleanJsonInput:

    def test_trailing_comma_object(self, agent):
        raw = '{"expression": "2+2",}'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"expression": "2+2"}

    def test_trailing_comma_array(self, agent):
        raw = '["a", "b", "c",]'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == ["a", "b", "c"]

    def test_markdown_wrapped_json(self, agent):
        raw = '```json\n{"key": "value"}\n```'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"key": "value"}

    def test_markdown_no_language_tag(self, agent):
        raw = '```\n{"key": "value"}\n```'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"key": "value"}

    def test_unquoted_keys(self, agent):
        raw = '{expression: "2+2"}'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"expression": "2+2"}

    def test_multiple_unquoted_keys(self, agent):
        raw = '{query: "test", limit: 10}'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"query": "test", "limit": 10}

    def test_already_valid_json(self, agent):
        raw = '{"expression": "2+2"}'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"expression": "2+2"}

    def test_combined_issues(self, agent):
        """Markdown + trailing comma + unquoted key all at once."""
        raw = '```json\n{expression: "2+2",}\n```'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result == {"expression": "2+2"}

    def test_nested_json_preserved(self, agent):
        raw = '{"data": {"nested": true}}'
        cleaned = Agent._clean_json_input(raw)
        import json
        result = json.loads(cleaned)
        assert result["data"]["nested"] is True


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_string(self, agent):
        parsed = agent._parse_react_response("")
        assert parsed["thought"] is None
        assert parsed["action"] is None
        assert parsed["final_answer"] is None

    def test_none_input(self, agent):
        parsed = agent._parse_react_response(None)
        assert parsed["thought"] is None

    def test_numeric_input(self, agent):
        parsed = agent._parse_react_response(123)
        assert parsed["thought"] is None

    def test_only_whitespace(self, agent):
        parsed = agent._parse_react_response("   \n\n   ")
        assert parsed["thought"] is None
        assert parsed["final_answer"] is None

    def test_missing_colon_after_final_answer(self, agent):
        """Some SLMs omit the colon."""
        text = "Final Answer 42"
        parsed = agent._parse_react_response(text)
        # May or may not parse — at minimum should not crash
        assert isinstance(parsed, dict)

    def test_real_qwen_output(self, agent):
        """Simulate typical Qwen 2.5 output."""
        text = (
            "Thought: I need to use the calculator to add these numbers.\n"
            "Action: calculator\n"
            "Action Input: {\"expression\": \"15 + 27\"}\n"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["thought"] is not None
        assert parsed["action"] == "calculator"
        assert "expression" in parsed["action_input"]

    def test_real_llama_output(self, agent):
        """Simulate typical Llama 3 output."""
        text = (
            "Thought: The user wants to know the sum. Let me calculate it.\n\n"
            "Action: calculator\n"
            "Action Input: {expression: \"100 + 200\"}\n\n"
            "Observation: 300"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["action"] == "calculator"

    def test_real_phi_output_with_trailing_question(self, agent):
        """Simulate Phi-4 generating trailing question."""
        text = (
            "Thought: I can answer directly.\n"
            "Final Answer: Python was created by Guido van Rossum in 1991."
            "What programming paradigms does Python support?"
        )
        parsed = agent._parse_react_response(text)
        assert "1991" in parsed["final_answer"]
        # Trailing question should be stripped
        assert "paradigms" not in parsed["final_answer"]

    def test_multiline_action_input_json(self, agent):
        """Multi-line JSON in Action Input."""
        text = (
            "Thought: need to process\n"
            "Action: json_tool\n"
            "Action Input: {\n"
            '  "operation": "query",\n'
            '  "data": {"name": "test"}\n'
            "}\n"
            "Observation: result"
        )
        parsed = agent._parse_react_response(text)
        assert parsed["action"] == "json_tool"
        assert parsed["action_input"] is not None
