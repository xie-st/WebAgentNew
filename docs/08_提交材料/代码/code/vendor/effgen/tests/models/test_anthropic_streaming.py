"""
Unit tests for Anthropic streaming: thinking blocks, parallel tool_use,
redacted_thinking detection.

All tests are fully mocked — no live API key required.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.anthropic_adapter import AnthropicAdapter, StreamChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter(model: str = "claude-sonnet-4-6") -> AnthropicAdapter:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        a = AnthropicAdapter(model_name=model)
    a._is_loaded = True
    a.client = MagicMock()
    a.client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=5)
    return a


def _ev(type_: str, **kw) -> SimpleNamespace:
    """Build a mock stream event."""
    return SimpleNamespace(type=type_, **kw)


def _block_start(index: int, block_type: str, **extra) -> SimpleNamespace:
    block = SimpleNamespace(type=block_type, **extra)
    return _ev("content_block_start", index=index, content_block=block)


def _text_delta(index: int, text: str) -> SimpleNamespace:
    return _ev("content_block_delta", index=index, delta=SimpleNamespace(type="text_delta", text=text))


def _thinking_delta(index: int, thinking: str) -> SimpleNamespace:
    return _ev("content_block_delta", index=index, delta=SimpleNamespace(type="thinking_delta", thinking=thinking))


def _signature_delta(index: int, signature: str) -> SimpleNamespace:
    return _ev("content_block_delta", index=index, delta=SimpleNamespace(type="signature_delta", signature=signature))


def _input_json_delta(index: int, partial_json: str) -> SimpleNamespace:
    return _ev("content_block_delta", index=index, delta=SimpleNamespace(type="input_json_delta", partial_json=partial_json))


def _block_stop(index: int) -> SimpleNamespace:
    return _ev("content_block_stop", index=index)


def _make_stream(*events):
    """Build a mock stream context manager that iterates the given events."""
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    stream_ctx.__iter__ = MagicMock(return_value=iter(events))
    stream_ctx.text_stream = iter(
        e.delta.text for e in events
        if getattr(e, "type", None) == "content_block_delta"
        and getattr(getattr(e, "delta", None), "type", None) == "text_delta"
    )
    return stream_ctx


# ---------------------------------------------------------------------------
# generate_stream_full: basic text
# ---------------------------------------------------------------------------

class TestStreamFullText:
    def test_plain_text_chunks_yielded(self):
        a = _adapter()
        events = [
            _block_start(0, "text"),
            _text_delta(0, "Hello "),
            _text_delta(0, "world"),
            _block_stop(0),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("hi"))
        texts = [c.text for c in chunks if c.type == "text"]
        assert "".join(texts) == "Hello world"

    def test_only_text_type_in_plain_response(self):
        a = _adapter()
        events = [
            _block_start(0, "text"),
            _text_delta(0, "answer"),
            _block_stop(0),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("q"))
        types = {c.type for c in chunks}
        assert types == {"text"}


# ---------------------------------------------------------------------------
# generate_stream_full: thinking blocks
# ---------------------------------------------------------------------------

class TestStreamFullThinking:
    def test_thinking_chunks_yielded(self):
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _thinking_delta(0, "Step 1: "),
            _thinking_delta(0, "reason"),
            _signature_delta(0, "sig123"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "answer"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("q"))
        thinking_chunks = [c for c in chunks if c.type == "thinking"]
        text_chunks = [c for c in chunks if c.type == "text"]

        assert "".join(c.text for c in thinking_chunks) == "Step 1: reason"
        assert "".join(c.text for c in text_chunks) == "answer"

    def test_thinking_before_text(self):
        """Thinking deltas must arrive before text deltas in Claude streaming."""
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _thinking_delta(0, "think"),
            _signature_delta(0, "s"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "reply"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunk_types = [c.type for c in a.generate_stream_full("q")]
        # All thinking should appear before any text
        first_text_idx = next(i for i, t in enumerate(chunk_types) if t == "text")
        last_thinking_idx = max((i for i, t in enumerate(chunk_types) if t == "thinking"), default=-1)
        assert last_thinking_idx < first_text_idx

    def test_signature_delta_not_yielded(self):
        """signature_delta events are internal integrity data and must not be yielded."""
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _thinking_delta(0, "thought"),
            _signature_delta(0, "encrypted_sig"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "reply"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("q"))
        assert all(c.type != "signature" for c in chunks)


# ---------------------------------------------------------------------------
# generate_stream_full: redacted_thinking detection
# ---------------------------------------------------------------------------

class TestStreamFullRedactedThinking:
    def test_redacted_thinking_detected_from_signature_only_block(self):
        """A thinking block with only signature_delta (no thinking_delta) is redacted."""
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            # No thinking_delta — only signature (redacted)
            _signature_delta(0, "ENCRYPTED_BLOB"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "censored answer"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("sensitive"))
        redacted = [c for c in chunks if c.type == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0].data["type"] == "redacted_thinking"
        assert redacted[0].data["data"] == "ENCRYPTED_BLOB"

    def test_non_redacted_thinking_not_emitted_as_redacted(self):
        """A thinking block that has thinking_delta text is NOT redacted."""
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _thinking_delta(0, "visible thoughts"),
            _signature_delta(0, "sig"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "reply"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("q"))
        assert not any(c.type == "redacted_thinking" for c in chunks)

    def test_redacted_thinking_data_field_populated(self):
        """redacted_thinking chunk's data dict contains 'data' key with signature."""
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _signature_delta(0, "SIG_DATA"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "reply"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("q"))
        redacted = [c for c in chunks if c.type == "redacted_thinking"]
        assert redacted[0].data.get("data") == "SIG_DATA"


# ---------------------------------------------------------------------------
# generate_stream_full: parallel tool_use
# ---------------------------------------------------------------------------

class TestStreamFullParallelToolUse:
    def test_single_tool_use_accumulated(self):
        a = _adapter()
        events = [
            _block_start(0, "tool_use", id="tool1", name="calculator"),
            _input_json_delta(0, '{"expr'),
            _input_json_delta(0, 'ession": "2+3"}'),
            _block_stop(0),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("calc"))
        tool_chunks = [c for c in chunks if c.type == "tool_use"]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].data["name"] == "calculator"
        assert tool_chunks[0].data["input"] == {"expression": "2+3"}
        assert tool_chunks[0].data["id"] == "tool1"

    def test_parallel_tool_use_both_accumulated(self):
        """Two tool_use blocks streaming in sequence (parallel function calling)."""
        a = _adapter()
        events = [
            _block_start(0, "tool_use", id="t1", name="add"),
            _input_json_delta(0, '{"a": 2, "b": 3}'),
            _block_stop(0),
            _block_start(1, "tool_use", id="t2", name="multiply"),
            _input_json_delta(1, '{"a": 7, "b": 8}'),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("compute"))
        tool_chunks = [c for c in chunks if c.type == "tool_use"]
        assert len(tool_chunks) == 2

        names = {c.data["name"] for c in tool_chunks}
        assert names == {"add", "multiply"}

        add_chunk = next(c for c in tool_chunks if c.data["name"] == "add")
        mul_chunk = next(c for c in tool_chunks if c.data["name"] == "multiply")
        assert add_chunk.data["input"] == {"a": 2, "b": 3}
        assert mul_chunk.data["input"] == {"a": 7, "b": 8}

    def test_tool_use_fragmented_json(self):
        """JSON fragments are concatenated correctly before parsing."""
        a = _adapter()
        events = [
            _block_start(0, "tool_use", id="t1", name="search"),
            _input_json_delta(0, '{"qu'),
            _input_json_delta(0, 'ery": "An'),
            _input_json_delta(0, 'thropic"}'),
            _block_stop(0),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("search"))
        tool_chunks = [c for c in chunks if c.type == "tool_use"]
        assert tool_chunks[0].data["input"] == {"query": "Anthropic"}

    def test_tool_use_empty_input_json(self):
        """Empty JSON input (e.g., no-arg tool) yields empty dict."""
        a = _adapter()
        events = [
            _block_start(0, "tool_use", id="t1", name="ping"),
            _block_stop(0),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("ping"))
        tool_chunks = [c for c in chunks if c.type == "tool_use"]
        assert tool_chunks[0].data["input"] == {}


# ---------------------------------------------------------------------------
# generate_stream_full: mixed thinking + tools
# ---------------------------------------------------------------------------

class TestStreamFullMixed:
    def test_thinking_then_tool_use(self):
        """Claude 4.x can emit thinking blocks followed by tool_use blocks."""
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _thinking_delta(0, "I need to call a tool"),
            _signature_delta(0, "sig"),
            _block_stop(0),
            _block_start(1, "tool_use", id="t1", name="search"),
            _input_json_delta(1, '{"q": "test"}'),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        chunks = list(a.generate_stream_full("q"))
        thinking_chunks = [c for c in chunks if c.type == "thinking"]
        tool_chunks = [c for c in chunks if c.type == "tool_use"]

        assert len(thinking_chunks) >= 1
        assert len(tool_chunks) == 1
        assert tool_chunks[0].data["name"] == "search"


# ---------------------------------------------------------------------------
# generate_stream (text-only wrapper)
# ---------------------------------------------------------------------------

class TestGenerateStreamTextOnly:
    def test_generate_stream_yields_only_text(self):
        a = _adapter()
        events = [
            _block_start(0, "thinking"),
            _thinking_delta(0, "internal"),
            _signature_delta(0, "sig"),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "public"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        # generate_stream must only yield text chunks, not thinking
        result = "".join(a.generate_stream("q"))
        assert result == "public"

    def test_generate_stream_skips_tool_use_events(self):
        a = _adapter()
        events = [
            _block_start(0, "tool_use", id="t1", name="calc"),
            _input_json_delta(0, '{"a": 1}'),
            _block_stop(0),
            _block_start(1, "text"),
            _text_delta(1, "done"),
            _block_stop(1),
        ]
        a.client.messages.stream = MagicMock(return_value=_make_stream(*events))

        result = "".join(a.generate_stream("q"))
        assert result == "done"


# ---------------------------------------------------------------------------
# StreamChunk dataclass
# ---------------------------------------------------------------------------

class TestStreamChunk:
    def test_defaults(self):
        c = StreamChunk(type="text")
        assert c.text == ""
        assert c.data == {}

    def test_text_chunk(self):
        c = StreamChunk(type="text", text="hello")
        assert c.type == "text"
        assert c.text == "hello"

    def test_tool_use_chunk(self):
        data = {"type": "tool_use", "id": "x", "name": "calc", "input": {"a": 1}}
        c = StreamChunk(type="tool_use", data=data)
        assert c.type == "tool_use"
        assert c.data["name"] == "calc"

    def test_redacted_thinking_chunk(self):
        c = StreamChunk(type="redacted_thinking", data={"type": "redacted_thinking", "data": "BLOB"})
        assert c.type == "redacted_thinking"
        assert c.data["data"] == "BLOB"


# ---------------------------------------------------------------------------
# Integration skip guard
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="SKIPPED: ANTHROPIC_API_KEY not in ~/.effgen/.env",
)
class TestStreamingIntegration:
    def test_live_stream_basic(self):
        """Live streaming test — requires ANTHROPIC_API_KEY."""
        from pathlib import Path

        from dotenv import load_dotenv
        load_dotenv(Path.home() / ".effgen" / ".env", override=False)

        a = AnthropicAdapter(model_name="claude-sonnet-4-6")
        a.load()
        try:
            result = "".join(a.generate_stream("Say exactly: hello world"))
            assert "hello" in result.lower()
        finally:
            a.unload()
