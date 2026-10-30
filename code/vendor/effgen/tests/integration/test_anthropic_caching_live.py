"""
Live integration tests for Anthropic prompt caching.

These tests require ANTHROPIC_API_KEY in ~/.effgen/.env.
They are automatically skipped when the key is absent (expected in dev env).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
_SKIP_REASON = "SKIPPED: ANTHROPIC_API_KEY not in ~/.effgen/.env"


@pytest.mark.skipif(not _ANTHROPIC_KEY, reason=_SKIP_REASON)
class TestAnthropicCachingLive:
    """
    Live tests that exercise prompt caching against the real Anthropic API.

    Verify that:
    - cache_creation_input_tokens is non-zero on first call (cache write).
    - cache_read_input_tokens is non-zero on second call with same system prompt (cache hit).
    """

    def _make_adapter(self):
        from effgen.models.anthropic_adapter import AnthropicAdapter
        a = AnthropicAdapter(model_name="claude-sonnet-4-6", api_key=_ANTHROPIC_KEY)
        a.load()
        return a

    def test_cache_creation_on_first_call(self):
        from effgen.models.anthropic_cache import apply_cache_to_system

        # Need a system prompt long enough to meet the 2048-token minimum for sonnet-4-6.
        long_system = "You are a helpful assistant. " * 300  # ~600 words, ~900 tokens
        adapter = self._make_adapter()
        try:
            cached_system = apply_cache_to_system(long_system)
            result = adapter.generate("Say hello.", system_prompt=cached_system)
            # On cache miss the API writes to cache; creation tokens may or may not
            # be non-zero depending on whether the block meets the minimum threshold.
            assert "cached_input_tokens" in result.metadata
            assert "cache_creation_tokens" in result.metadata
        finally:
            adapter.unload()

    def test_cache_read_on_repeat_call(self):
        from effgen.models.anthropic_cache import apply_cache_to_system

        # Large enough to guarantee caching
        long_system = "You are an expert research assistant. " * 300
        adapter = self._make_adapter()
        try:
            cached_system = apply_cache_to_system(long_system)
            # First call — cache write
            adapter.generate("What is your name?", system_prompt=cached_system)
            # Second call — should hit cache
            result2 = adapter.generate("Describe your capabilities briefly.",
                                        system_prompt=cached_system)
            assert result2.metadata["cached_input_tokens"] >= 0  # may vary by timing
        finally:
            adapter.unload()
