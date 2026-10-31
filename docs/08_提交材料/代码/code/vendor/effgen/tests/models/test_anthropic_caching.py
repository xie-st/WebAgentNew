"""
Unit tests for Anthropic prompt caching (cache_control) support.

Covers:
- mark_cached() helper
- apply_cache_to_system / apply_cache_to_last_tool helpers
- count_cache_breakpoints / validate_breakpoint_count
- 4-breakpoint limit raises ValueError
- Adapter: cache_control passes through in system prompt (list form)
- Adapter: cached_input_tokens / cache_creation_tokens surfaced in metadata
- AgentConfig: cache_system_prompt + cache_tools fields present and default True
- Agent._get_anthropic_system / _get_anthropic_tools helpers
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.anthropic_cache import (
    MAX_CACHE_BREAKPOINTS,
    apply_cache_to_last_tool,
    apply_cache_to_system,
    count_cache_breakpoints,
    get_min_cache_tokens,
    mark_cached,
    validate_breakpoint_count,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _block(**kwargs):
    return SimpleNamespace(**kwargs)


def _response(blocks, stop_reason="end_turn", input_tokens=100, output_tokens=40,
              cached_input=0, cache_creation=0):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cached_input,
        cache_creation_input_tokens=cache_creation,
    )
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=usage)


def _adapter(model="claude-sonnet-4-6"):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        a = AnthropicAdapter(model_name=model)
    a._is_loaded = True
    a.client = MagicMock()
    a.client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=5)
    return a


# ── mark_cached ────────────────────────────────────────────────────────────────

class TestMarkCached:
    def test_dict_block_gets_cache_control(self):
        block = {"type": "text", "text": "hello"}
        result = mark_cached(block)
        assert "cache_control" in result
        assert result["cache_control"]["type"] == "ephemeral"

    def test_default_ttl_is_5m(self):
        result = mark_cached({"type": "text", "text": "x"})
        assert result["cache_control"]["ttl"] == "5m"

    def test_ttl_1h_accepted(self):
        result = mark_cached({"type": "text", "text": "x"}, ttl="1h")
        assert result["cache_control"]["ttl"] == "1h"

    def test_string_converted_to_text_block(self):
        result = mark_cached("plain string")
        assert result["type"] == "text"
        assert result["text"] == "plain string"
        assert "cache_control" in result

    def test_original_dict_not_mutated(self):
        original = {"type": "text", "text": "original"}
        mark_cached(original)
        assert "cache_control" not in original

    def test_existing_fields_preserved(self):
        block = {"type": "document", "source": {"type": "base64", "data": "abc"}}
        result = mark_cached(block)
        assert result["source"] == block["source"]
        assert result["type"] == "document"


# ── apply_cache_to_system ──────────────────────────────────────────────────────

class TestApplyCacheToSystem:
    def test_string_becomes_single_cached_block(self):
        result = apply_cache_to_system("You are helpful.")
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are helpful."
        assert "cache_control" in result[0]

    def test_last_block_of_list_gets_cached(self):
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "last"},
        ]
        result = apply_cache_to_system(blocks)
        assert "cache_control" not in result[0]
        assert "cache_control" in result[1]

    def test_original_list_not_mutated(self):
        blocks = [{"type": "text", "text": "only"}]
        apply_cache_to_system(blocks)
        assert "cache_control" not in blocks[0]

    def test_empty_list_returns_empty(self):
        assert apply_cache_to_system([]) == []

    def test_ttl_passed_through(self):
        result = apply_cache_to_system("sys", ttl="1h")
        assert result[0]["cache_control"]["ttl"] == "1h"


# ── apply_cache_to_last_tool ───────────────────────────────────────────────────

class TestApplyCacheToLastTool:
    def test_last_tool_gets_cache_control(self):
        tools = [{"name": "a"}, {"name": "b"}]
        result = apply_cache_to_last_tool(tools)
        assert "cache_control" not in result[0]
        assert "cache_control" in result[1]

    def test_original_list_not_mutated(self):
        tools = [{"name": "only"}]
        apply_cache_to_last_tool(tools)
        assert "cache_control" not in tools[0]

    def test_empty_list_returns_empty(self):
        assert apply_cache_to_last_tool([]) == []

    def test_ttl_passed_through(self):
        tools = [{"name": "t"}]
        result = apply_cache_to_last_tool(tools, ttl="1h")
        assert result[0]["cache_control"]["ttl"] == "1h"


# ── count_cache_breakpoints / validate_breakpoint_count ───────────────────────

class TestBreakpointCounting:
    def test_zero_when_no_markers(self):
        assert count_cache_breakpoints("plain string", [], []) == 0

    def test_counts_system_list_markers(self):
        system = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b"},
        ]
        assert count_cache_breakpoints(system, [], []) == 1

    def test_counts_message_markers(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}},
            ]},
        ]
        assert count_cache_breakpoints(None, messages, []) == 1

    def test_counts_tool_markers(self):
        tools = [
            {"name": "a", "cache_control": {"type": "ephemeral"}},
        ]
        assert count_cache_breakpoints(None, [], tools) == 1

    def test_counts_across_all_parts(self):
        system = [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "m", "cache_control": {"type": "ephemeral"}},
        ]}]
        tools = [{"name": "t", "cache_control": {"type": "ephemeral"}}]
        assert count_cache_breakpoints(system, messages, tools) == 3

    def test_validate_4_breakpoints_ok(self):
        system = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        ]
        tools = [
            {"name": "t1", "cache_control": {"type": "ephemeral"}},
            {"name": "t2", "cache_control": {"type": "ephemeral"}},
        ]
        validate_breakpoint_count(system, [], tools)  # should not raise

    def test_validate_5_breakpoints_raises_value_error(self):
        system = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        ]
        tools = [
            {"name": "t1", "cache_control": {"type": "ephemeral"}},
            {"name": "t2", "cache_control": {"type": "ephemeral"}},
            {"name": "t3", "cache_control": {"type": "ephemeral"}},
        ]
        with pytest.raises(ValueError, match="Too many cache_control breakpoints"):
            validate_breakpoint_count(system, [], tools)

    def test_validate_error_message_mentions_limit(self):
        tools = [{"name": f"t{i}", "cache_control": {"type": "ephemeral"}} for i in range(5)]
        with pytest.raises(ValueError, match=str(MAX_CACHE_BREAKPOINTS)):
            validate_breakpoint_count(None, [], tools)


# ── Adapter: list-form system prompt with cache_control ───────────────────────

class TestAdapterCachedSystemPrompt:
    def test_list_system_prompt_passed_verbatim(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")]
        )
        cached_system = apply_cache_to_system("You are a helpful assistant.")
        a.generate("hello", system_prompt=cached_system)

        call_kwargs = a.client.messages.create.call_args[1]
        assert call_kwargs["system"] == cached_system
        assert "cache_control" in call_kwargs["system"][-1]

    def test_string_system_prompt_still_works(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")]
        )
        a.generate("hello", system_prompt="plain system")
        call_kwargs = a.client.messages.create.call_args[1]
        assert call_kwargs["system"] == "plain system"

    def test_exceeding_4_breakpoints_raises_before_api_call(self):
        a = _adapter()
        # 5 cached blocks in system prompt → should raise before making API call
        system = [
            {"type": "text", "text": f"block {i}",
             "cache_control": {"type": "ephemeral"}}
            for i in range(5)
        ]
        with pytest.raises((ValueError, RuntimeError)):
            a.generate("hi", system_prompt=system)
        # API should NOT have been called
        a.client.messages.create.assert_not_called()

    def test_4_breakpoints_does_not_raise(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")]
        )
        system = [
            {"type": "text", "text": f"block {i}",
             "cache_control": {"type": "ephemeral"}}
            for i in range(4)
        ]
        # Should not raise
        result = a.generate("hi", system_prompt=system)
        assert result.text == "ok"


# ── Adapter: usage fields surfaced in metadata ─────────────────────────────────

class TestAdapterCacheUsage:
    def test_cached_input_tokens_in_metadata(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")],
            cached_input=800,
            cache_creation=0,
        )
        result = a.generate("hi")
        assert result.metadata["cached_input_tokens"] == 800

    def test_cache_creation_tokens_in_metadata(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")],
            cached_input=0,
            cache_creation=2500,
        )
        result = a.generate("hi")
        assert result.metadata["cache_creation_tokens"] == 2500

    def test_both_fields_zero_when_no_caching(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")]
        )
        result = a.generate("hi")
        assert result.metadata["cached_input_tokens"] == 0
        assert result.metadata["cache_creation_tokens"] == 0

    def test_generate_with_tools_surfaces_cache_usage(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")],
            cached_input=1500,
            cache_creation=100,
        )
        tool = {"name": "calc", "description": "...", "input_schema": {"type": "object"}}
        result = a.generate_with_tools("hi", tools=[tool])
        assert result.metadata["cached_input_tokens"] == 1500
        assert result.metadata["cache_creation_tokens"] == 100

    def test_generate_with_history_surfaces_cache_usage(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="text", text="ok")],
            cached_input=0,
            cache_creation=4096,
        )
        messages = [{"role": "user", "content": "hi"}]
        result = a.generate_with_history(messages)
        assert result.metadata["cache_creation_tokens"] == 4096

    def test_usage_missing_cache_fields_defaults_to_zero(self):
        """Older SDK versions may not include cache fields; default to 0."""
        a = _adapter()
        # Response without cache_*_input_tokens attributes
        usage = SimpleNamespace(input_tokens=50, output_tokens=20)
        response = SimpleNamespace(
            content=[_block(type="text", text="hi")],
            stop_reason="end_turn",
            usage=usage,
        )
        a.client.messages.create.return_value = response
        result = a.generate("q")
        assert result.metadata["cached_input_tokens"] == 0
        assert result.metadata["cache_creation_tokens"] == 0


# ── AgentConfig fields ─────────────────────────────────────────────────────────

class TestAgentConfigCacheFields:
    def test_cache_system_prompt_defaults_true(self):
        from effgen.core.agent import AgentConfig
        cfg = AgentConfig(name="a", model="dummy")
        assert cfg.cache_system_prompt is True

    def test_cache_tools_defaults_true(self):
        from effgen.core.agent import AgentConfig
        cfg = AgentConfig(name="a", model="dummy")
        assert cfg.cache_tools is True

    def test_cache_fields_can_be_disabled(self):
        from effgen.core.agent import AgentConfig
        cfg = AgentConfig(name="a", model="dummy",
                          cache_system_prompt=False, cache_tools=False)
        assert cfg.cache_system_prompt is False
        assert cfg.cache_tools is False


# ── Agent helpers: _get_anthropic_system / _get_anthropic_tools ───────────────

class TestAgentAnthropicCacheHelpers:
    def _agent(self, cache_system=True, cache_tools=True):
        from effgen.core.agent import Agent, AgentConfig
        adapter = _adapter()
        cfg = AgentConfig(
            name="test",
            model=adapter,
            cache_system_prompt=cache_system,
            cache_tools=cache_tools,
            system_prompt="You are a helpful AI.",
        )
        agent = Agent.__new__(Agent)
        agent.config = cfg
        agent.model = adapter
        agent.tools = {}
        return agent

    def test_get_anthropic_system_returns_cached_list(self):
        agent = self._agent(cache_system=True)
        result = agent._get_anthropic_system()
        assert isinstance(result, list)
        assert "cache_control" in result[-1]

    def test_get_anthropic_system_returns_string_when_disabled(self):
        agent = self._agent(cache_system=False)
        result = agent._get_anthropic_system()
        assert isinstance(result, str)

    def test_get_anthropic_tools_marks_last_tool(self):
        agent = self._agent(cache_tools=True)
        tools = [{"name": "a"}, {"name": "b"}]
        result = agent._get_anthropic_tools(tools)
        assert "cache_control" not in result[0]
        assert "cache_control" in result[1]

    def test_get_anthropic_tools_unchanged_when_disabled(self):
        agent = self._agent(cache_tools=False)
        tools = [{"name": "a"}, {"name": "b"}]
        result = agent._get_anthropic_tools(tools)
        assert "cache_control" not in result[-1]

    def test_get_anthropic_tools_empty_returns_empty(self):
        agent = self._agent()
        assert agent._get_anthropic_tools([]) == []

    def test_non_anthropic_model_system_prompt_unchanged(self):
        """Non-Anthropic model → system prompt returned as plain string."""
        from effgen.core.agent import Agent, AgentConfig
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "GeminiAdapter"
        cfg = AgentConfig(
            name="test",
            model=mock_model,
            cache_system_prompt=True,
            system_prompt="You are an assistant.",
        )
        agent = Agent.__new__(Agent)
        agent.config = cfg
        agent.model = mock_model
        result = agent._get_anthropic_system()
        assert isinstance(result, str)


# ── get_min_cache_tokens ──────────────────────────────────────────────────────

class TestMinCacheTokens:
    def test_sonnet_4_6_is_2048(self):
        assert get_min_cache_tokens("claude-sonnet-4-6") == 2048

    def test_opus_4_7_is_4096(self):
        assert get_min_cache_tokens("claude-opus-4-7") == 4096

    def test_haiku_4_5_is_4096(self):
        assert get_min_cache_tokens("claude-haiku-4-5") == 4096

    def test_unknown_model_returns_default(self):
        assert get_min_cache_tokens("claude-unknown-model") == 1024
