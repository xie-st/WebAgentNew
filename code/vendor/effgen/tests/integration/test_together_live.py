"""Integration tests for TogetherAdapter — skipped if TOGETHER_API_KEY absent, real calls otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("TOGETHER_API_KEY"))


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: TOGETHER_API_KEY not set")
class TestTogetherLive:
    def test_generate_default_serverless(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter(
            "Qwen/Qwen3.5-9B",
            max_retries=1,
            timeout=30,
        )
        adapter.load()
        try:
            result = adapter.generate("Respond with exactly: TOGETHER_OK")
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("rate", "429", "timeout", "timed out", "503", "service unavailable", "quota")):
                pytest.skip(f"Together transient: {exc}")
            raise
        else:
            assert result.text, "Expected non-empty response"
            assert result.tokens_used > 0
            assert result.metadata is not None
            assert result.metadata["provider"] == "together"
        finally:
            adapter.unload()

    def test_generate_qwen(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("Qwen/Qwen3.5-9B")
        adapter.load()
        try:
            result = adapter.generate("What is 2 + 2? Answer with just the number.")
            assert "4" in result.text
        finally:
            adapter.unload()

    def test_generate_deepseek(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("deepseek-ai/DeepSeek-V4-Pro")
        adapter.load()
        try:
            result = adapter.generate("Say hello in one word.")
        except Exception as exc:
            msg = str(exc).lower()
            # A dedicated-endpoint model whose endpoint is stopped is an account
            # state, not a defect. The adapter signals this with InvalidRequestError,
            # so match on the message rather than the exception type.
            if "dedicated endpoint" in msg:
                pytest.skip(f"Together dedicated endpoint not running: {exc}")
            if any(k in msg for k in ("rate", "429", "timeout", "timed out", "503", "service unavailable", "quota")):
                pytest.skip(f"Together transient: {exc}")
            raise
        else:
            assert result.text
            assert result.tokens_used > 0
        finally:
            adapter.unload()

    def test_generate_openai_oss(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("openai/gpt-oss-20b")
        adapter.load()
        try:
            result = adapter.generate("What is the capital of Japan?")
            assert "tokyo" in result.text.lower() or "japan" in result.text.lower()
        finally:
            adapter.unload()

    def test_load_model_via_provider(self):
        from effgen.models import load_model

        model = load_model("Qwen/Qwen3.5-9B", provider="together")
        try:
            result = model.generate("Say hello in one word")
            assert result.text
        finally:
            model.unload()

    def test_generate_stream_yields_chunks(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("Qwen/Qwen3.5-9B")
        adapter.load()
        try:
            chunks = list(adapter.generate_stream("Count from 1 to 3, one per line."))
            assert len(chunks) >= 3, f"Expected >=3 chunks, got {len(chunks)}"
            full_text = "".join(chunks)
            assert "1" in full_text and "2" in full_text and "3" in full_text
        finally:
            adapter.unload()

    def test_tool_calling_llama70b(self):

        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        adapter.load()
        try:
            tool = {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"],
                    },
                },
            }
            result = adapter.generate_with_tools(
                "What's the weather in Paris?",
                tools=[tool],
            )
            assert result.metadata is not None
            tool_calls = result.metadata.get("tool_calls", [])
            assert len(tool_calls) >= 1, f"Expected at least 1 tool call, got {tool_calls}"
            assert tool_calls[0]["function"]["name"] == "get_weather"
            assert isinstance(tool_calls[0]["function"]["arguments"], str)
        finally:
            adapter.unload()

    def test_usage_metadata_populated(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("Qwen/Qwen3.5-9B")
        adapter.load()
        try:
            result = adapter.generate("Hello")
            assert result.metadata is not None
            assert result.metadata["prompt_tokens"] > 0
            assert result.metadata["completion_tokens"] > 0
            assert result.metadata["total_tokens"] > 0
            assert "cost_usd" in result.metadata
        finally:
            adapter.unload()

    def test_serverless_property(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("Qwen/Qwen3.5-9B")
        assert adapter.is_serverless is True

    def test_context_length(self):
        from effgen.models.together_adapter import TogetherAdapter

        adapter = TogetherAdapter("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert adapter.get_context_length() == 131_072
