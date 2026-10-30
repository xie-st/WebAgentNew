"""
Unit tests for AnthropicAdapter — fully mocked, no live API calls.

Covers:
- Model registry lookups (context length, supports_thinking, costs)
- Adapter construction and capability checks
- Request building: temperature, thinking config, top_k omission
- Response parsing: text, thinking, redacted_thinking, tool_use blocks
- Multi-turn: build_assistant_message / generate_with_history
- Edge cases: unknown model, missing API key
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.anthropic_adapter import AnthropicAdapter, _block_to_dict
from effgen.models.anthropic_models import (
    ANTHROPIC_MODELS,
    get_context_length,
    get_cost_per_million,
    get_model_info,
    supports_thinking,
)
from effgen.models.base import GenerationConfig, GenerationResult

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_block(**kwargs) -> SimpleNamespace:
    """Build a fake Anthropic content block as a SimpleNamespace."""
    return SimpleNamespace(**kwargs)


def _make_response(
    content_blocks: list[SimpleNamespace],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=content_blocks, stop_reason=stop_reason, usage=usage)


def _make_adapter(model_name: str = "claude-opus-4-7") -> AnthropicAdapter:
    """Build a pre-loaded adapter with a mocked Anthropic client."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        adapter = AnthropicAdapter(model_name=model_name)
    adapter._is_loaded = True
    adapter.client = MagicMock()
    # Ensure count_tokens returns a sensible int so validate_prompt passes
    adapter.client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=5)
    return adapter


# ── Registry tests ────────────────────────────────────────────────────────────


class TestAnthropicRegistry:
    def test_all_required_fields_present(self):
        required = {"context", "max_output", "family", "supports_thinking", "supports_native_tools"}
        for name, info in ANTHROPIC_MODELS.items():
            missing = required - info.keys()
            assert not missing, f"{name} missing fields: {missing}"

    def test_claude_4x_supports_thinking(self):
        # Opus 4.7 only has adaptive thinking; explicit extended thinking is
        # not exposed via the `thinking={"type":"enabled"}` API parameter.
        assert not supports_thinking("claude-opus-4-7")
        assert supports_thinking("claude-sonnet-4-6")
        assert supports_thinking("claude-haiku-4-5-20251001")
        assert supports_thinking("claude-3-7-sonnet-20250219")

    def test_legacy_models_no_thinking(self):
        assert not supports_thinking("claude-3-5-sonnet-20241022")
        assert not supports_thinking("claude-3-opus-20240229")
        assert not supports_thinking("claude-3-haiku-20240307")

    def test_context_length_200k(self):
        for name in ["claude-haiku-4-5-20251001", "claude-3-7-sonnet-20250219", "claude-3-opus-20240229"]:
            assert get_context_length(name) == 200_000

    def test_context_length_1m_for_4x_flagships(self):
        # Claude Opus 4.7 and Sonnet 4.6 ship with a 1M-token context window.
        assert get_context_length("claude-opus-4-7") == 1_000_000
        assert get_context_length("claude-sonnet-4-6") == 1_000_000

    def test_unknown_model_defaults(self):
        info = get_model_info("claude-future-unknown")
        assert info["context"] == 200_000
        assert info["supports_thinking"] is False

    def test_cost_per_million_opus(self):
        inp, out = get_cost_per_million("claude-opus-4-7")
        assert inp == 5.0
        assert out == 25.0

    def test_cost_per_million_sonnet(self):
        inp, out = get_cost_per_million("claude-sonnet-4-6")
        assert inp == 3.0
        assert out == 15.0

    def test_cost_per_million_haiku(self):
        inp, out = get_cost_per_million("claude-haiku-4-5-20251001")
        assert inp == 1.0
        assert out == 5.0

    def test_cost_unknown_model_defaults_to_sonnet_pricing(self):
        inp, out = get_cost_per_million("claude-does-not-exist")
        assert inp == 3.0
        assert out == 15.0


# ── Adapter construction ──────────────────────────────────────────────────────


class TestAdapterConstruction:
    def test_raises_without_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="API key"):
                AnthropicAdapter()

    def test_accepts_explicit_api_key(self):
        adapter = AnthropicAdapter(model_name="claude-opus-4-7", api_key="explicit-key")
        assert adapter.api_key == "explicit-key"

    def test_model_name_stored(self):
        adapter = AnthropicAdapter(model_name="claude-sonnet-4-6", api_key="k")
        assert adapter.model_name == "claude-sonnet-4-6"

    def test_context_length_from_registry(self):
        adapter = AnthropicAdapter(model_name="claude-opus-4-7", api_key="k")
        assert adapter.get_context_length() == 1_000_000

    def test_unknown_model_context_length(self):
        adapter = AnthropicAdapter(model_name="claude-mystery-2099", api_key="k")
        assert adapter.get_context_length() == 200_000

    def test_supports_function_calling_all_modern(self):
        for name in ["claude-opus-4-7", "claude-sonnet-4-6", "claude-3-7-sonnet-20250219"]:
            adapter = AnthropicAdapter(model_name=name, api_key="k")
            assert adapter.supports_function_calling()
            assert adapter.supports_tool_calling()


# ── _block_to_dict ────────────────────────────────────────────────────────────


class TestBlockToDict:
    def test_text_block(self):
        block = _make_block(type="text", text="hello")
        d = _block_to_dict(block)
        assert d == {"type": "text", "text": "hello"}

    def test_thinking_block(self):
        block = _make_block(type="thinking", thinking="inner thoughts")
        d = _block_to_dict(block)
        assert d == {"type": "thinking", "thinking": "inner thoughts"}

    def test_redacted_thinking_block(self):
        block = _make_block(type="redacted_thinking", data="REDACTED_DATA")
        d = _block_to_dict(block)
        assert d == {"type": "redacted_thinking", "data": "REDACTED_DATA"}

    def test_tool_use_block(self):
        block = _make_block(type="tool_use", id="tu_1", name="calculator", input={"x": 1})
        d = _block_to_dict(block)
        assert d == {"type": "tool_use", "id": "tu_1", "name": "calculator", "input": {"x": 1}}

    def test_dict_passthrough(self):
        d = {"type": "text", "text": "already a dict"}
        assert _block_to_dict(d) is d


# ── Request building ──────────────────────────────────────────────────────────


class TestRequestBuilding:
    def test_default_config_no_temperature_top_p(self):
        adapter = _make_adapter()
        cfg = GenerationConfig()  # default: temp=0.7, top_p=0.9
        req = adapter._build_request("hi", cfg, None, None, {})
        # Non-default are skipped — adapter only includes when != default
        assert "temperature" not in req
        assert "top_p" not in req
        assert "top_k" not in req

    def test_custom_temperature_included(self):
        adapter = _make_adapter()
        cfg = GenerationConfig(temperature=0.5)
        req = adapter._build_request("hi", cfg, None, None, {})
        assert req["temperature"] == 0.5

    def test_max_tokens_default(self):
        adapter = _make_adapter()
        req = adapter._build_request("hi", GenerationConfig(), None, None, {})
        assert req["max_tokens"] == 4096

    def test_max_tokens_custom(self):
        adapter = _make_adapter()
        req = adapter._build_request("hi", GenerationConfig(max_tokens=1024), None, None, {})
        assert req["max_tokens"] == 1024

    def test_system_prompt_included(self):
        adapter = _make_adapter()
        req = adapter._build_request("hi", GenerationConfig(), "Be helpful", None, {})
        assert req["system"] == "Be helpful"

    def test_thinking_sent_when_model_supports(self):
        adapter = _make_adapter("claude-sonnet-4-6")
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 8000})
        req = adapter._build_request("hi", cfg, None, None, {})
        assert req["thinking"] == {"type": "enabled", "budget_tokens": 8000}
        assert req["temperature"] == 1.0  # forced by Anthropic requirement

    def test_thinking_dropped_when_model_unsupported(self, caplog):
        adapter = _make_adapter("claude-3-5-sonnet-20241022")
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 4000})
        import logging
        with caplog.at_level(logging.DEBUG):
            req = adapter._build_request("hi", cfg, None, None, {})
        assert "thinking" not in req
        assert any("does not support thinking" in r.message for r in caplog.records)

    def test_thinking_not_sent_when_none(self):
        adapter = _make_adapter("claude-opus-4-7")
        cfg = GenerationConfig(thinking=None)
        req = adapter._build_request("hi", cfg, None, None, {})
        assert "thinking" not in req

    def test_tools_included(self):
        adapter = _make_adapter()
        tools = [{"name": "calc", "description": "...", "input_schema": {}}]
        req = adapter._build_request("hi", GenerationConfig(), None, tools, {})
        assert req["tools"] == tools

    def test_stop_sequences_included(self):
        adapter = _make_adapter()
        cfg = GenerationConfig(stop_sequences=["STOP", "END"])
        req = adapter._build_request("hi", cfg, None, None, {})
        assert req["stop_sequences"] == ["STOP", "END"]


# ── Response parsing ──────────────────────────────────────────────────────────


class TestResponseParsing:
    def test_simple_text_response(self):
        adapter = _make_adapter()
        response = _make_response([_make_block(type="text", text="Hello!")])
        text, thinking, redacted, raw = adapter._parse_response(response)
        assert text == "Hello!"
        assert thinking == []
        assert redacted == []
        assert len(raw) == 1
        assert raw[0]["type"] == "text"

    def test_thinking_block_extracted(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="thinking", thinking="deep thoughts"),
            _make_block(type="text", text="Answer here"),
        ]
        text, thinking, redacted, raw = adapter._parse_response(_make_response(blocks))
        assert thinking == ["deep thoughts"]
        assert text == "Answer here"
        assert len(raw) == 2

    def test_redacted_thinking_preserved(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="redacted_thinking", data="REDACTED_BLOB"),
            _make_block(type="text", text="Safe answer"),
        ]
        text, thinking, redacted, raw = adapter._parse_response(_make_response(blocks))
        assert len(redacted) == 1
        assert redacted[0]["data"] == "REDACTED_BLOB"
        assert redacted[0] in raw, "redacted_thinking must appear in raw_content_blocks"

    def test_mixed_blocks(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="thinking", thinking="thought 1"),
            _make_block(type="redacted_thinking", data="BLOB"),
            _make_block(type="text", text="final answer"),
        ]
        text, thinking, redacted, raw = adapter._parse_response(_make_response(blocks))
        assert text == "final answer"
        assert len(thinking) == 1
        assert len(redacted) == 1
        assert len(raw) == 3

    def test_multiple_text_blocks_concatenated(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="text", text="part1 "),
            _make_block(type="text", text="part2"),
        ]
        text, _, _, _ = adapter._parse_response(_make_response(blocks))
        assert text == "part1 part2"


# ── Generate (mocked) ─────────────────────────────────────────────────────────


class TestGenerateMocked:
    def _setup(self, model: str = "claude-opus-4-7"):
        adapter = _make_adapter(model)
        text_block = _make_block(type="text", text="42 is the answer")
        response = _make_response([text_block], input_tokens=100, output_tokens=30)
        adapter.client.messages.create.return_value = response
        return adapter, response

    def test_generate_returns_result(self):
        adapter, _ = self._setup()
        result = adapter.generate("What is 6*7?")
        assert result.text == "42 is the answer"
        assert result.model_name == "claude-opus-4-7"
        assert result.tokens_used == 30

    def test_generate_metadata_has_tokens(self):
        adapter, _ = self._setup()
        result = adapter.generate("hi")
        assert result.metadata["prompt_tokens"] == 100
        assert result.metadata["completion_tokens"] == 30
        assert result.metadata["total_tokens"] == 130

    def test_generate_has_raw_content_blocks(self):
        adapter, _ = self._setup()
        result = adapter.generate("hi")
        assert "raw_content_blocks" in result.metadata
        assert len(result.metadata["raw_content_blocks"]) == 1

    def test_generate_thinking_in_metadata(self):
        adapter = _make_adapter("claude-opus-4-7")
        blocks = [
            _make_block(type="thinking", thinking="my reasoning"),
            _make_block(type="text", text="final"),
        ]
        adapter.client.messages.create.return_value = _make_response(blocks)
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 5000})
        result = adapter.generate("think hard", config=cfg)
        assert result.metadata["thinking"] == ["my reasoning"]

    def test_generate_cost_tracked(self):
        adapter, _ = self._setup()
        adapter.generate("hi")
        assert adapter.total_cost > 0
        assert adapter.total_tokens > 0

    def test_not_loaded_raises(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            adapter = AnthropicAdapter()
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            adapter.generate("hi")


# ── Redacted thinking multi-turn ──────────────────────────────────────────────


class TestRedactedThinkingMultiTurn:
    def test_build_assistant_message_uses_raw_blocks(self):
        raw_blocks = [
            {"type": "redacted_thinking", "data": "BLOB"},
            {"type": "text", "text": "Answer"},
        ]
        result = GenerationResult(
            text="Answer",
            tokens_used=10,
            finish_reason="end_turn",
            model_name="claude-opus-4-7",
            metadata={"raw_content_blocks": raw_blocks},
        )
        msg = AnthropicAdapter.build_assistant_message(result)
        assert msg["role"] == "assistant"
        assert msg["content"] == raw_blocks
        # redacted_thinking block must be present — do NOT strip it
        types = [b["type"] for b in msg["content"]]
        assert "redacted_thinking" in types

    def test_build_assistant_message_fallback_without_raw_blocks(self):
        result = GenerationResult(
            text="plain answer",
            tokens_used=5,
            finish_reason="end_turn",
            model_name="claude-opus-4-7",
            metadata={},
        )
        msg = AnthropicAdapter.build_assistant_message(result)
        assert msg["role"] == "assistant"
        assert msg["content"] == "plain answer"

    def test_redacted_thinking_in_metadata(self):
        adapter = _make_adapter("claude-opus-4-7")
        blocks = [
            _make_block(type="redacted_thinking", data="REDACTED"),
            _make_block(type="text", text="safe"),
        ]
        adapter.client.messages.create.return_value = _make_response(blocks)
        result = adapter.generate("sensitive question")
        assert "redacted_thinking" in result.metadata
        assert result.metadata["redacted_thinking"][0]["data"] == "REDACTED"

    def test_generate_with_history_passes_messages(self):
        adapter = _make_adapter("claude-opus-4-7")
        text_block = _make_block(type="text", text="Follow-up answer")
        adapter.client.messages.create.return_value = _make_response([text_block])

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "BLOB"},
                {"type": "text", "text": "First answer"},
            ]},
            {"role": "user", "content": "Follow-up"},
        ]
        result = adapter.generate_with_history(messages)
        assert result.text == "Follow-up answer"

        call_kwargs = adapter.client.messages.create.call_args[1]
        assert call_kwargs["messages"] == messages

    def test_generate_with_history_thinking_config(self):
        adapter = _make_adapter("claude-sonnet-4-6")
        adapter.client.messages.create.return_value = _make_response(
            [_make_block(type="text", text="ok")]
        )
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 2000})
        messages = [{"role": "user", "content": "think"}]
        adapter.generate_with_history(messages, config=cfg)
        call_kwargs = adapter.client.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2000}
        assert call_kwargs["temperature"] == 1.0


# ── Tool use ──────────────────────────────────────────────────────────────────


class TestToolUseMocked:
    def test_tool_use_block_in_metadata(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="tool_use", id="tu_1", name="calculator", input={"expr": "2+2"}),
        ]
        adapter.client.messages.create.return_value = _make_response(blocks)
        tools = [{"name": "calculator", "description": "...", "input_schema": {}}]
        result = adapter.generate_with_tools("2+2", tools)
        assert result.metadata["tool_uses"] == [
            {"id": "tu_1", "name": "calculator", "input": {"expr": "2+2"}}
        ]

    def test_parallel_tool_uses(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="tool_use", id="tu_1", name="add", input={"a": 2, "b": 3}),
            _make_block(type="tool_use", id="tu_2", name="multiply", input={"a": 7, "b": 8}),
        ]
        adapter.client.messages.create.return_value = _make_response(blocks)
        result = adapter.generate_with_tools("compute", [])
        assert len(result.metadata["tool_uses"]) == 2
        names = [t["name"] for t in result.metadata["tool_uses"]]
        assert "add" in names
        assert "multiply" in names

    def test_tool_use_raw_blocks_preserved(self):
        adapter = _make_adapter()
        blocks = [
            _make_block(type="tool_use", id="tu_1", name="calc", input={}),
            _make_block(type="text", text="calling tool"),
        ]
        adapter.client.messages.create.return_value = _make_response(blocks)
        result = adapter.generate_with_tools("hi", [])
        raw = result.metadata["raw_content_blocks"]
        assert any(b["type"] == "tool_use" for b in raw)


# ── Streaming (mocked) ────────────────────────────────────────────────────────


class TestStreamingMocked:
    def test_stream_yields_text(self):
        from types import SimpleNamespace
        adapter = _make_adapter()

        # Build raw events that generate_stream_full consumes.
        def _text_ev(text):
            return SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=text),
            )

        events = [
            SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text")),
            _text_ev("Hello"),
            _text_ev(" "),
            _text_ev("world"),
            SimpleNamespace(type="content_block_stop", index=0),
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(return_value=iter(events))
        adapter.client.messages.stream.return_value = mock_stream

        chunks = list(adapter.generate_stream("hi"))
        assert chunks == ["Hello", " ", "world"]

    def test_stream_not_loaded_raises(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            adapter = AnthropicAdapter()
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            list(adapter.generate_stream("hi"))


# ── GenerationConfig thinking field ──────────────────────────────────────────


class TestGenerationConfigThinking:
    def test_thinking_defaults_to_none(self):
        cfg = GenerationConfig()
        assert cfg.thinking is None

    def test_thinking_can_be_set(self):
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 10000})
        assert cfg.thinking["budget_tokens"] == 10000

    def test_other_fields_unaffected(self):
        cfg = GenerationConfig(temperature=0.5, thinking={"type": "enabled", "budget_tokens": 1000})
        assert cfg.temperature == 0.5
        assert cfg.max_tokens is None
