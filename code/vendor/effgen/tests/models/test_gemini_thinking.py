"""Unit tests for Gemini thinking_budget / include_thoughts (no live API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effgen.models.base import GenerationConfig
from effgen.models.gemini_adapter import GeminiAdapter

# ---------------------------------------------------------------------------
# GenerationConfig field presence
# ---------------------------------------------------------------------------

def test_generation_config_has_thinking_budget():
    cfg = GenerationConfig()
    assert cfg.thinking_budget is None
    assert cfg.include_thoughts is False


def test_generation_config_thinking_budget_set():
    cfg = GenerationConfig(thinking_budget=8192, include_thoughts=True)
    assert cfg.thinking_budget == 8192
    assert cfg.include_thoughts is True


def test_generation_config_thinking_budget_zero():
    cfg = GenerationConfig(thinking_budget=0)
    assert cfg.thinking_budget == 0


# ---------------------------------------------------------------------------
# _build_config: thinking_budget=None → no thinking_config sent
# ---------------------------------------------------------------------------

def _make_adapter(model_name: str) -> GeminiAdapter:
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_name = model_name
    adapter.safety_settings = None
    return adapter


def test_build_config_no_thinking_when_none():
    pytest.importorskip("google.genai")
    adapter = _make_adapter("gemini-3.1-flash-lite-preview")
    cfg = GenerationConfig(thinking_budget=None)
    result = adapter._build_config(cfg)
    assert result.thinking_config is None


# ---------------------------------------------------------------------------
# _build_config: thinking_budget=0 → field sent as 0 (disabled)
# ---------------------------------------------------------------------------

def test_build_config_thinking_budget_zero_is_sent():
    pytest.importorskip("google.genai")
    adapter = _make_adapter("gemini-3.1-flash-lite-preview")
    cfg = GenerationConfig(thinking_budget=0, include_thoughts=False)
    result = adapter._build_config(cfg)
    assert result.thinking_config is not None
    assert result.thinking_config.thinking_budget == 0


# ---------------------------------------------------------------------------
# _build_config: thinking_budget=8192 → field sent with budget
# ---------------------------------------------------------------------------

def test_build_config_thinking_budget_positive():
    pytest.importorskip("google.genai")
    adapter = _make_adapter("gemini-3.1-flash-lite-preview")
    cfg = GenerationConfig(thinking_budget=8192, include_thoughts=True)
    result = adapter._build_config(cfg)
    assert result.thinking_config is not None
    assert result.thinking_config.thinking_budget == 8192
    assert result.thinking_config.include_thoughts is True


# ---------------------------------------------------------------------------
# _build_config: unsupported model + thinking_budget → dropped silently
# ---------------------------------------------------------------------------

def test_build_config_thinking_dropped_for_unsupported_model(caplog):
    pytest.importorskip("google.genai")
    import logging
    # gemma-4-31b-it does not support thinking
    adapter = _make_adapter("gemma-4-31b-it")
    cfg = GenerationConfig(thinking_budget=4096)
    with caplog.at_level(logging.DEBUG, logger="effgen.models.gemini_adapter"):
        result = adapter._build_config(cfg)

    assert result.thinking_config is None
    assert any("does not support thinking_budget" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _model_supports_thinking
# ---------------------------------------------------------------------------

def test_model_supports_thinking_true_for_flash_lite():
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_name = "gemini-3.1-flash-lite-preview"
    assert adapter._model_supports_thinking() is True


def test_model_supports_thinking_via_alias():
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_name = "gemini-3.1-flash-lite"
    assert adapter._model_supports_thinking() is True


def test_model_supports_thinking_false_for_flash_lite_25():
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_name = "gemini-2.5-flash-lite"
    assert adapter._model_supports_thinking() is False


def test_model_supports_thinking_false_for_unknown():
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_name = "some-unknown-model"
    assert adapter._model_supports_thinking() is False


# ---------------------------------------------------------------------------
# generate(): thinking trace surfaced in metadata["thinking"]
# ---------------------------------------------------------------------------

def _make_mock_response(text_parts: list[tuple[str, bool]], *, thoughts_tokens: int = 0) -> MagicMock:
    """Build a fake google.genai GenerateContentResponse.

    Args:
        text_parts: list of (text, is_thought) tuples.
        thoughts_tokens: value for usage_metadata.thoughts_token_count.
    """
    parts = []
    for text, is_thought in text_parts:
        p = MagicMock()
        p.text = text
        p.thought = is_thought
        p.function_call = None
        parts.append(p)

    candidate = MagicMock()
    candidate.content.parts = parts
    candidate.finish_reason = "STOP"

    um = MagicMock()
    um.prompt_token_count = 10
    um.candidates_token_count = 20
    um.total_token_count = 30
    um.thoughts_token_count = thoughts_tokens

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = um
    response.safety_ratings = None
    return response


def _make_full_adapter(model_name: str) -> GeminiAdapter:
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.model_name = model_name
    adapter.safety_settings = None
    adapter._is_loaded = True
    adapter._context_length = 1_000_000
    adapter.total_cost = 0.0
    adapter.total_tokens = 0
    return adapter


def test_generate_surfaces_thinking_in_metadata():
    pytest.importorskip("google.genai")
    adapter = _make_full_adapter("gemini-3.1-flash-lite-preview")

    mock_response = _make_mock_response(
        [("I think about it.", True), ("The answer is 42.", False)],
        thoughts_tokens=15,
    )

    with patch.object(adapter, "_generate_with_retry", return_value=mock_response):
        with patch.object(adapter, "validate_prompt"):
            result = adapter.generate(
                "What is 6×7?",
                config=GenerationConfig(thinking_budget=8192, include_thoughts=True),
            )

    assert "thinking" in result.metadata
    assert "I think about it." in result.metadata["thinking"]
    assert result.text == "The answer is 42."
    assert result.metadata["thoughts_token_count"] == 15


def test_generate_no_thinking_when_not_requested():
    pytest.importorskip("google.genai")
    adapter = _make_full_adapter("gemini-3.1-flash-lite-preview")

    mock_response = _make_mock_response(
        [("The answer is 42.", False)],
    )

    with patch.object(adapter, "_generate_with_retry", return_value=mock_response):
        with patch.object(adapter, "validate_prompt"):
            result = adapter.generate("What is 6×7?")

    assert "thinking" not in result.metadata
    assert result.text == "The answer is 42."
