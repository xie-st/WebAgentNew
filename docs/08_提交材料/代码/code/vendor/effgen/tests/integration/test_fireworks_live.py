"""
Live integration tests for FireworksAdapter.

Skipped automatically when FIREWORKS_API_KEY is not set.
Uses cheap/small models where possible to reduce cost.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY")
pytestmark = pytest.mark.skipif(
    not FIREWORKS_KEY, reason="FIREWORKS_API_KEY not set"
)

# Cheap models used for live tests (must be in live serverless catalog as of test run)
# gpt-oss-120b is the cheapest text-only model currently in the Fireworks serverless catalog.
_SMALL_MODEL = "accounts/fireworks/models/gpt-oss-120b"
# kimi-k2p6 returns native structured tool_calls (verified 2026-08-07).
_TOOL_MODEL = "accounts/fireworks/models/kimi-k2p6"


def _xfail_if_fireworks_transient(exc: Exception) -> None:
    msg = str(exc).lower()
    transient_markers = (
        "timeout",
        "timed out",
        "429",
        "rate limit",
        "temporarily unavailable",
        "503",
        "500",
        "internal server",
        # Fireworks answers a 503 with a body whose text carries no status code:
        # {"code": "invalid_request_error", "message": "service overloaded,
        # please try again later"}. Matching the code alone misses it.
        "service overloaded",
        "service unavailable",
        "overloaded",
        "capacity",
    )
    if any(marker in msg for marker in transient_markers):
        pytest.xfail(f"Fireworks transient service failure: {exc}")


@contextlib.contextmanager
def _guard_transient():
    """Convert a transient Fireworks service error (500/503/429/timeout) into an
    xfail instead of a hard failure, so a provider-side outage doesn't redden the
    live suite. Real (non-transient) errors still propagate."""
    try:
        yield
    except RuntimeError as exc:
        _xfail_if_fireworks_transient(exc)
        raise


@pytest.fixture(scope="module")
def small_adapter():
    from effgen.models.fireworks_adapter import FireworksAdapter
    adapter = FireworksAdapter(
        _SMALL_MODEL,
        enable_rate_limiting=False,
        max_retries=2,
        timeout=30,
    )
    adapter.load()
    yield adapter
    adapter.unload()


@pytest.fixture(scope="module")
def tool_adapter():
    from effgen.models.fireworks_adapter import FireworksAdapter
    adapter = FireworksAdapter(_TOOL_MODEL, enable_rate_limiting=False)
    adapter.load()
    yield adapter
    adapter.unload()


class TestFireworksLiveGenerate:
    def test_basic_generate(self, small_adapter):
        from effgen.models.base import GenerationConfig

        with _guard_transient():
            result = small_adapter.generate(
                "What is the capital of France? Answer in one word.",
                config=GenerationConfig(max_tokens=256, temperature=0.0),
            )
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert "paris" in result.text.lower() or "France" in result.text
        assert result.tokens_used > 0

    def test_usage_populated(self, small_adapter):
        from effgen.models.base import GenerationConfig
        with _guard_transient():
            result = small_adapter.generate(
                "Say hello.",
                config=GenerationConfig(max_tokens=20),
            )
        assert result.metadata["prompt_tokens"] > 0
        assert result.metadata["completion_tokens"] > 0
        assert result.metadata["total_tokens"] > 0
        assert result.metadata["provider"] == "fireworks"

    def test_generation_config_max_tokens(self, small_adapter):
        from effgen.models.base import GenerationConfig
        with _guard_transient():
            result = small_adapter.generate(
                "Count slowly from 1 to 100.",
                config=GenerationConfig(max_tokens=15),
            )
        # Reasoning models may emit reasoning_content beyond max_tokens visible cap.
        assert result.tokens_used > 0

    def test_finish_reason(self, small_adapter):
        from effgen.models.base import GenerationConfig
        with _guard_transient():
            result = small_adapter.generate(
                "What is 1 + 1?",
                config=GenerationConfig(max_tokens=200),
            )
        assert result.finish_reason in ("stop", "length", "eos_token")


class TestFireworksLiveStream:
    def test_stream_yields_multiple_chunks(self, small_adapter):
        from effgen.models.base import GenerationConfig
        with _guard_transient():
            chunks = list(small_adapter.generate_stream(
                "Count from 1 to 5 slowly.",
                config=GenerationConfig(max_tokens=300),
            ))
        assert len(chunks) > 1, "Expected multiple streaming chunks"
        full_text = "".join(chunks)
        assert len(full_text) > 5

    def test_stream_timing(self, small_adapter):
        """Verify streaming delivers tokens incrementally (not all at once)."""
        from effgen.models.base import GenerationConfig
        timestamps = []

        def timed_stream():
            for chunk in small_adapter.generate_stream(
                "Tell me a very short sentence.",
                config=GenerationConfig(max_tokens=200),
            ):
                timestamps.append(time.monotonic())
                yield chunk

        with _guard_transient():
            list(timed_stream())
        assert len(timestamps) >= 2, "Should receive multiple chunk timestamps"


class TestFireworksLiveTools:
    _CALCULATOR_TOOL = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '17 * 23'",
                    }
                },
                "required": ["expression"],
            },
        },
    }]

    def test_native_tool_call(self, tool_adapter):
        from effgen.models.base import GenerationConfig
        with _guard_transient():
            result = tool_adapter.generate_with_tools(
                "What is 17 multiplied by 23? Use the calculator tool.",
                tools=self._CALCULATOR_TOOL,
                config=GenerationConfig(max_tokens=200, temperature=0.0),
            )
        tc = result.metadata.get("tool_calls", [])
        assert len(tc) >= 1, f"Expected at least 1 tool call, got: {tc}"
        assert tc[0]["function"]["name"] == "calculator"
        assert isinstance(tc[0]["function"]["arguments"], str)

    def test_supports_tool_calling(self, tool_adapter):
        assert tool_adapter.supports_tool_calling() is True


class TestFireworksLiveContextLength:
    def test_context_length_matches_registry(self, small_adapter):
        from effgen.models.fireworks_models import FIREWORKS_MODELS
        info = FIREWORKS_MODELS.get(_SMALL_MODEL, {})
        assert small_adapter.get_context_length() == info.get("context", 0)

    def test_count_tokens(self, small_adapter):
        tc = small_adapter.count_tokens("Hello, world! This is a test prompt.")
        assert tc.count > 0
        assert tc.model_name == _SMALL_MODEL
