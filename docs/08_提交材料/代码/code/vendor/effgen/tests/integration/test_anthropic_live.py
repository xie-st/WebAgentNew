"""
Live integration tests for AnthropicAdapter.

These tests require ANTHROPIC_API_KEY in ~/.effgen/.env.
If the key is absent they skip cleanly — no test failures.

Run manually when a key is available:
    ANTHROPIC_API_KEY=<key> pytest tests/integration/test_anthropic_live.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.base import GenerationConfig

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
skip_no_key = pytest.mark.skipif(
    not ANTHROPIC_API_KEY,
    reason="SKIPPED: ANTHROPIC_API_KEY not in ~/.effgen/.env",
)


@skip_no_key
def test_live_basic_generate():
    """Verify basic text generation against the real API."""
    with AnthropicAdapter(model_name="claude-haiku-4-5-20251001") as adapter:
        result = adapter.generate("Say hello in one word.")
    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert result.tokens_used > 0


@skip_no_key
def test_live_thinking_generate():
    """Verify extended thinking returns a thinking trace."""
    cfg = GenerationConfig(
        thinking={"type": "enabled", "budget_tokens": 2000},
        max_tokens=4096,
    )
    with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
        result = adapter.generate("What is 17 * 23?", config=cfg)
    assert "thinking" in result.metadata
    assert len(result.metadata["thinking"]) > 0


@skip_no_key
def test_live_redacted_thinking_multi_turn():
    """
    Verify that redacted_thinking blocks preserved in assistant message
    do not cause a 400 on the second turn.
    """
    cfg = GenerationConfig(
        thinking={"type": "enabled", "budget_tokens": 1000},
        max_tokens=2048,
    )
    adapter = AnthropicAdapter(model_name="claude-sonnet-4-6")
    adapter.load()

    r1 = adapter.generate("What is the square root of 144?", config=cfg)
    asst_msg = AnthropicAdapter.build_assistant_message(r1)

    history = [
        {"role": "user", "content": "What is the square root of 144?"},
        asst_msg,
        {"role": "user", "content": "And the cube root of 27?"},
    ]
    r2 = adapter.generate_with_history(history, config=cfg)
    assert isinstance(r2.text, str)
    assert len(r2.text) > 0
    adapter.unload()


@skip_no_key
def test_live_tool_use():
    """Verify tool use returns tool_uses in metadata."""
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }]
    with AnthropicAdapter(model_name="claude-haiku-4-5-20251001") as adapter:
        result = adapter.generate_with_tools("What is the weather in Paris?", tools)
    assert "tool_uses" in result.metadata


@skip_no_key
def test_live_streaming():
    """Verify streaming yields non-empty chunks."""
    with AnthropicAdapter(model_name="claude-haiku-4-5-20251001") as adapter:
        chunks = list(adapter.generate_stream("Count from 1 to 3."))
    assert len(chunks) > 0
    full = "".join(chunks)
    assert len(full) > 0
