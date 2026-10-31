"""Unit tests for tool-calling strategy edge cases."""

from __future__ import annotations

from effgen.core.tool_calling import HybridStrategy, NativeFunctionCallingStrategy


def test_hybrid_does_not_treat_thought_only_text_as_final_answer():
    strategy = HybridStrategy()
    result = strategy.parse_response(
        "Thought: The previous calculation is correct and we don't need to perform it again."
    )

    assert result.final_answer is None
    assert result.is_tool_call is False
    assert result.thought


# --- Gemma 4 channel / tool_call format -------------------------------------

def test_gemma_tool_call_parsed_and_name_resolved():
    """<|tool_call>call:NAME{...} fires, and an invented name maps to a real tool."""
    strategy = NativeFunctionCallingStrategy()
    tools = {t: object() for t in ["web_search", "arxiv", "wikipedia"]}
    text = (
        "reasoning...<channel|><|tool_call>"
        'call:search_arxiv{query: "LLM agents"}<tool_call|>'
    )
    result = strategy.parse_response(text, tools)

    assert result.is_tool_call is True
    assert result.tool_name == "arxiv"  # search_arxiv → arxiv
    assert result.arguments == {"query": "LLM agents"}


def test_gemma_final_answer_after_channel_close():
    """Reasoning in <|channel>...<channel|> is stripped; the answer survives."""
    strategy = NativeFunctionCallingStrategy()
    text = "<|channel>thought\nenglish planning\n<channel|>富士山の標高は3,776mです。"
    result = strategy.parse_response(text)

    assert result.is_tool_call is False
    assert result.final_answer == "富士山の標高は3,776mです。"
    assert result.thought  # reasoning captured, not leaked into the answer


def test_gemma_truncated_reasoning_yields_no_leaked_answer():
    """An unclosed channel (max_tokens cut) must not surface raw reasoning."""
    strategy = NativeFunctionCallingStrategy()
    result = strategy.parse_response("<|channel>thought\ncut off before finishing")

    assert result.is_tool_call is False
    assert result.final_answer is None


def test_gemma_exact_tool_name_preserved():
    strategy = NativeFunctionCallingStrategy()
    tools = {"arxiv": object()}
    result = strategy.parse_response(
        '<|tool_call>call:arxiv{"query": "x"}<tool_call|>', tools
    )
    assert result.tool_name == "arxiv"
    assert result.arguments == {"query": "x"}
