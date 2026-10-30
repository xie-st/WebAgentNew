"""
Focused unit tests for Anthropic extended thinking and redacted_thinking handling.

These tests specifically validate the extended thinking feature including:
- thinking config sent correctly
- thinking trace extracted into metadata["thinking"]
- redacted_thinking blocks preserved in raw_content_blocks (multi-turn safety)
- temperature forced to 1.0 when thinking enabled
- thinking disabled for non-supporting models
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.anthropic_models import supports_thinking
from effgen.models.base import GenerationConfig, GenerationResult


def _block(**kwargs):
    return SimpleNamespace(**kwargs)


def _response(blocks, stop_reason="end_turn", input_tokens=50, output_tokens=40):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _adapter(model="claude-sonnet-4-6"):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        a = AnthropicAdapter(model_name=model)
    a._is_loaded = True
    a.client = MagicMock()
    # Ensure count_tokens returns a sensible int so validate_prompt passes
    a.client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=5)
    return a


# ── Extended thinking request ──────────────────────────────────────────────────


class TestThinkingRequest:
    def test_thinking_config_forwarded_to_api(self):
        a = _adapter()
        a.client.messages.create.return_value = _response(
            [_block(type="thinking", thinking="thoughts"), _block(type="text", text="answer")]
        )
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 8000})
        a.generate("reason about this", config=cfg)

        call_kwargs = a.client.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8000}

    def test_temperature_forced_to_1_when_thinking(self):
        a = _adapter()
        a.client.messages.create.return_value = _response([_block(type="text", text="ok")])
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 4000}, temperature=0.3)
        a.generate("hi", config=cfg)

        call_kwargs = a.client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0

    def test_thinking_none_omits_field(self):
        a = _adapter()
        a.client.messages.create.return_value = _response([_block(type="text", text="hi")])
        cfg = GenerationConfig(thinking=None)
        a.generate("no thinking", config=cfg)

        call_kwargs = a.client.messages.create.call_args[1]
        assert "thinking" not in call_kwargs

    def test_thinking_dropped_for_non_supporting_model(self, caplog):
        a = _adapter("claude-3-5-sonnet-20241022")
        a.client.messages.create.return_value = _response([_block(type="text", text="ok")])
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 3000})

        import logging
        with caplog.at_level(logging.DEBUG):
            a.generate("hi", config=cfg)

        call_kwargs = a.client.messages.create.call_args[1]
        assert "thinking" not in call_kwargs

    def test_thinking_with_budget_zero_sent(self):
        """budget_tokens=0 is a valid value — thinking disabled explicitly."""
        a = _adapter("claude-sonnet-4-6")
        a.client.messages.create.return_value = _response([_block(type="text", text="ok")])
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 0})
        a.generate("hi", config=cfg)
        call_kwargs = a.client.messages.create.call_args[1]
        assert call_kwargs["thinking"]["budget_tokens"] == 0


# ── Thinking in response ────────────────────────────────────────────────────────


class TestThinkingResponse:
    def test_thinking_trace_in_metadata(self):
        a = _adapter()
        blocks = [
            _block(type="thinking", thinking="Step 1: reason.\nStep 2: conclude."),
            _block(type="text", text="The answer is 42."),
        ]
        a.client.messages.create.return_value = _response(blocks)
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 5000})
        result = a.generate("solve", config=cfg)

        assert "thinking" in result.metadata
        assert len(result.metadata["thinking"]) == 1
        assert "Step 1" in result.metadata["thinking"][0]

    def test_text_separated_from_thinking(self):
        a = _adapter()
        blocks = [
            _block(type="thinking", thinking="internal monologue"),
            _block(type="text", text="public answer"),
        ]
        a.client.messages.create.return_value = _response(blocks)
        result = a.generate("q")
        assert result.text == "public answer"
        assert result.metadata["thinking"] == ["internal monologue"]

    def test_no_thinking_metadata_when_absent(self):
        a = _adapter()
        a.client.messages.create.return_value = _response([_block(type="text", text="plain")])
        result = a.generate("q")
        assert "thinking" not in result.metadata

    def test_multiple_thinking_blocks(self):
        a = _adapter()
        blocks = [
            _block(type="thinking", thinking="first thought"),
            _block(type="thinking", thinking="second thought"),
            _block(type="text", text="answer"),
        ]
        a.client.messages.create.return_value = _response(blocks)
        result = a.generate("q")
        assert result.metadata["thinking"] == ["first thought", "second thought"]


# ── Redacted thinking ────────────────────────────────────────────────────────────


class TestRedactedThinking:
    def test_redacted_thinking_in_metadata(self):
        a = _adapter()
        blocks = [
            _block(type="redacted_thinking", data="REDACTED_BLOB_DATA"),
            _block(type="text", text="censored answer"),
        ]
        a.client.messages.create.return_value = _response(blocks)
        result = a.generate("sensitive")
        assert "redacted_thinking" in result.metadata
        assert result.metadata["redacted_thinking"][0]["data"] == "REDACTED_BLOB_DATA"

    def test_redacted_thinking_in_raw_content_blocks(self):
        a = _adapter()
        blocks = [
            _block(type="redacted_thinking", data="SECRET"),
            _block(type="text", text="ok"),
        ]
        a.client.messages.create.return_value = _response(blocks)
        result = a.generate("q")
        raw = result.metadata["raw_content_blocks"]
        redacted_blocks = [b for b in raw if b["type"] == "redacted_thinking"]
        assert len(redacted_blocks) == 1
        assert redacted_blocks[0]["data"] == "SECRET"

    def test_redacted_thinking_preserved_in_assistant_message(self):
        """
        Anthropic docs: redacted_thinking MUST be included verbatim in the
        assistant message on re-submit; stripping it causes a 400 error.
        """
        raw_blocks = [
            {"type": "redacted_thinking", "data": "BLOB"},
            {"type": "text", "text": "response"},
        ]
        result = GenerationResult(
            text="response",
            tokens_used=20,
            finish_reason="end_turn",
            model_name="claude-opus-4-7",
            metadata={"raw_content_blocks": raw_blocks},
        )
        msg = AnthropicAdapter.build_assistant_message(result)
        assert msg["role"] == "assistant"
        # redacted_thinking must be in the content list
        types = [b["type"] for b in msg["content"]]
        assert "redacted_thinking" in types

    def test_mixed_thinking_and_redacted(self):
        a = _adapter()
        blocks = [
            _block(type="thinking", thinking="visible thought"),
            _block(type="redacted_thinking", data="HIDDEN"),
            _block(type="text", text="answer"),
        ]
        a.client.messages.create.return_value = _response(blocks)
        result = a.generate("q")
        assert result.metadata["thinking"] == ["visible thought"]
        assert len(result.metadata["redacted_thinking"]) == 1
        assert len(result.metadata["raw_content_blocks"]) == 3

    def test_second_turn_receives_raw_blocks(self):
        """
        Simulate a two-turn conversation: first result has redacted_thinking;
        the second call must receive those blocks verbatim in the assistant message.
        """
        a = _adapter()

        # Turn 1 response
        turn1_blocks = [
            _block(type="redacted_thinking", data="PRIVATE"),
            _block(type="text", text="First response"),
        ]
        turn2_response = _response([_block(type="text", text="Second response")])

        a.client.messages.create.side_effect = [
            _response(turn1_blocks),
            turn2_response,
        ]

        # Turn 1
        cfg = GenerationConfig(thinking={"type": "enabled", "budget_tokens": 2000})
        r1 = a.generate("first question", config=cfg)

        # Build assistant message and compose second turn
        asst_msg = AnthropicAdapter.build_assistant_message(r1)
        history = [
            {"role": "user", "content": "first question"},
            asst_msg,
            {"role": "user", "content": "follow up"},
        ]

        # Turn 2 via generate_with_history
        r2 = a.generate_with_history(history, config=cfg)

        # Assert that the messages passed to the API contain the redacted block
        call2_kwargs = a.client.messages.create.call_args_list[1][1]
        all_contents = [
            b for msg in call2_kwargs["messages"] if isinstance(msg["content"], list)
            for b in msg["content"]
        ]
        assert any(b.get("type") == "redacted_thinking" for b in all_contents)
        assert r2.text == "Second response"


# ── Supports thinking registry check ──────────────────────────────────────────


class TestSupportsThinkingRegistry:
    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5",
        "claude-3-7-sonnet-20250219",
    ])
    def test_thinking_supported(self, model):
        assert supports_thinking(model) is True

    @pytest.mark.parametrize("model", [
        # Opus 4.7 only exposes adaptive thinking; explicit extended thinking
        # via the `thinking` API parameter is not supported.
        "claude-opus-4-7",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ])
    def test_thinking_not_supported(self, model):
        assert supports_thinking(model) is False
