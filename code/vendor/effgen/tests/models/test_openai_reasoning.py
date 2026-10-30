"""
Unit tests for OpenAI adapter reasoning_effort plumbing and model registry.

These tests are all unit-level (no real API calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effgen.models.base import GenerationConfig
from effgen.models.openai_models import (
    OPENAI_MODELS,
    available_models,
    chat_models,
    get_context_length,
    get_max_output,
    model_info,
    reasoning_models,
    supports_reasoning,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestOpenAIModelRegistry:
    def test_registry_not_empty(self):
        assert len(OPENAI_MODELS) > 10

    def test_all_models_have_required_keys(self):
        required = {"family", "context", "max_output", "supports_reasoning",
                    "supports_native_tools", "supports_prompt_caching"}
        for model_id, info in OPENAI_MODELS.items():
            missing = required - set(info.keys())
            assert not missing, f"{model_id} missing keys: {missing}"

    def test_reasoning_models_present(self):
        rm = reasoning_models()
        assert "o3-mini" in rm
        assert "o4-mini" in rm
        assert "o1" in rm

    def test_chat_models_present(self):
        cm = chat_models()
        assert "gpt-4o-mini" in cm
        assert "gpt-5.4-nano" in cm

    def test_reasoning_models_have_large_max_output(self):
        for mid in reasoning_models():
            assert OPENAI_MODELS[mid]["max_output"] >= 32_768, (
                f"{mid} max_output too small for a reasoning model"
            )

    def test_context_lengths_positive(self):
        for mid, info in OPENAI_MODELS.items():
            assert info["context"] > 0, f"{mid} context must be > 0"

    def test_model_info_raises_for_unknown(self):
        with pytest.raises(KeyError, match="Unknown OpenAI model"):
            model_info("gpt-99-mega-turbo")

    def test_supports_reasoning_false_for_chat(self):
        assert not supports_reasoning("gpt-4o-mini")
        assert not supports_reasoning("gpt-5.4-nano")

    def test_supports_reasoning_true_for_o_series(self):
        assert supports_reasoning("o3-mini")
        assert supports_reasoning("o4-mini")
        assert supports_reasoning("o1")

    def test_get_context_length_known(self):
        assert get_context_length("gpt-4o-mini") == 128_000

    def test_get_context_length_unknown_defaults(self):
        assert get_context_length("gpt-unknown-xyz") == 128_000

    def test_get_max_output_known(self):
        assert get_max_output("o3-mini") == 100_000

    def test_available_models_includes_all_families(self):
        models = available_models()
        assert "gpt-4o-mini" in models
        assert "o3-mini" in models
        assert "gpt-5.4-nano" in models


# ---------------------------------------------------------------------------
# GenerationConfig extension tests
# ---------------------------------------------------------------------------

class TestGenerationConfigReasoningFields:
    def test_defaults_none(self):
        cfg = GenerationConfig()
        assert cfg.reasoning_effort is None
        assert cfg.max_reasoning_tokens is None

    def test_set_reasoning_effort(self):
        cfg = GenerationConfig(reasoning_effort="high")
        assert cfg.reasoning_effort == "high"

    def test_set_max_reasoning_tokens(self):
        cfg = GenerationConfig(max_reasoning_tokens=5000)
        assert cfg.max_reasoning_tokens == 5000

    def test_combined(self):
        cfg = GenerationConfig(reasoning_effort="minimal", max_reasoning_tokens=1000)
        assert cfg.reasoning_effort == "minimal"
        assert cfg.max_reasoning_tokens == 1000


# ---------------------------------------------------------------------------
# Adapter unit tests (no real API)
# ---------------------------------------------------------------------------

def _make_mock_response(text: str = "OPENAI_OK", model: str = "gpt-4o-mini"):
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.prompt_tokens_details = None

    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    return response


class TestOpenAIAdapterReasoningPayload:
    """Verify _build_request_params produces correct payloads."""

    def _make_adapter(self, model_name: str):
        from effgen.models.openai_adapter import OpenAIAdapter
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            adapter = OpenAIAdapter(model_name=model_name)
        adapter._is_loaded = True  # skip real load()
        return adapter

    def test_reasoning_effort_included_for_reasoning_model(self):
        adapter = self._make_adapter("o3-mini")
        cfg = GenerationConfig(reasoning_effort="minimal")
        messages = [{"role": "user", "content": "hi"}]
        params = adapter._build_request_params(messages, cfg)
        # SDK expects reasoning_effort as a top-level parameter
        assert "reasoning_effort" in params
        assert params["reasoning_effort"] == "minimal"

    def test_max_reasoning_tokens_overrides_max_completion_tokens(self):
        adapter = self._make_adapter("o3-mini")
        cfg = GenerationConfig(reasoning_effort="high", max_reasoning_tokens=4000)
        messages = [{"role": "user", "content": "hi"}]
        params = adapter._build_request_params(messages, cfg)
        assert params["max_completion_tokens"] == 4000

    def test_reasoning_payload_absent_when_effort_none(self):
        adapter = self._make_adapter("o3-mini")
        cfg = GenerationConfig()
        messages = [{"role": "user", "content": "hi"}]
        params = adapter._build_request_params(messages, cfg)
        assert "reasoning_effort" not in params

    def test_reasoning_effort_silently_dropped_for_chat_model(self):
        """reasoning_effort on a chat model: no error, no reasoning= payload."""
        adapter = self._make_adapter("gpt-4o-mini")
        cfg = GenerationConfig(reasoning_effort="high")
        messages = [{"role": "user", "content": "hi"}]
        params = adapter._build_request_params(messages, cfg)
        assert "reasoning" not in params
        # temperature must still be present for chat models
        assert "temperature" in params

    def test_invalid_reasoning_effort_raises_valueerror(self):
        adapter = self._make_adapter("o3-mini")
        cfg = GenerationConfig(reasoning_effort="absurd")
        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="Invalid reasoning_effort"):
            adapter._build_request_params(messages, cfg)

    def test_valueerror_lists_valid_values(self):
        adapter = self._make_adapter("o3-mini")
        cfg = GenerationConfig(reasoning_effort="absurd")
        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError) as exc_info:
            adapter._build_request_params(messages, cfg)
        msg = str(exc_info.value)
        # At least the most common values should be listed
        assert "minimal" in msg
        assert "high" in msg

    def test_reasoning_model_uses_max_completion_tokens(self):
        adapter = self._make_adapter("o4-mini")
        cfg = GenerationConfig(reasoning_effort="medium", max_tokens=2048)
        messages = [{"role": "user", "content": "hi"}]
        params = adapter._build_request_params(messages, cfg)
        # All models now use max_completion_tokens (max_tokens is deprecated)
        assert "max_completion_tokens" in params
        assert "max_tokens" not in params

    def test_stop_dropped_for_gpt5_family(self):
        """GPT-5 models reject 'stop' — adapter must drop it silently."""
        adapter = self._make_adapter("gpt-5.4-nano")
        cfg = GenerationConfig(stop_sequences=["\nFinal Answer:"])
        params = adapter._build_request_params([{"role": "user", "content": "hi"}], cfg)
        assert "stop" not in params

    def test_sampling_params_dropped_for_gpt5_family(self):
        """GPT-5 chat models reject non-default sampling params."""
        adapter = self._make_adapter("gpt-5.4-nano")
        cfg = GenerationConfig(
            temperature=0.0,
            top_p=0.5,
            presence_penalty=0.2,
            frequency_penalty=0.2,
        )
        params = adapter._build_request_params([{"role": "user", "content": "hi"}], cfg)
        assert "temperature" not in params
        assert "top_p" not in params
        assert "presence_penalty" not in params
        assert "frequency_penalty" not in params

    def test_stop_dropped_for_reasoning_models(self):
        """Reasoning models reject 'stop' — adapter must drop it silently."""
        adapter = self._make_adapter("o3-mini")
        cfg = GenerationConfig(stop_sequences=["END"])
        params = adapter._build_request_params([{"role": "user", "content": "hi"}], cfg)
        assert "stop" not in params

    def test_stop_kept_for_gpt4o(self):
        """Legacy chat models still accept 'stop'."""
        adapter = self._make_adapter("gpt-4o-mini")
        cfg = GenerationConfig(stop_sequences=["END"])
        params = adapter._build_request_params([{"role": "user", "content": "hi"}], cfg)
        assert params.get("stop") == ["END"]

    def test_chat_model_uses_max_completion_tokens(self):
        adapter = self._make_adapter("gpt-4o-mini")
        cfg = GenerationConfig(max_tokens=512)
        messages = [{"role": "user", "content": "hi"}]
        params = adapter._build_request_params(messages, cfg)
        # All models use max_completion_tokens (deprecated max_tokens removed)
        assert "max_completion_tokens" in params
        assert "max_tokens" not in params

    def test_is_reasoning_model_flag(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            from effgen.models.openai_adapter import OpenAIAdapter
            chat = OpenAIAdapter("gpt-4o-mini")
            reasoning = OpenAIAdapter("o3-mini")
        assert not chat.is_reasoning_model()
        assert reasoning.is_reasoning_model()

    def test_list_models_class_method(self):
        from effgen.models.openai_adapter import OpenAIAdapter
        models = OpenAIAdapter.list_models()
        assert "gpt-4o-mini" in models
        assert "o3-mini" in models

    def test_list_reasoning_models_class_method(self):
        from effgen.models.openai_adapter import OpenAIAdapter
        rm = OpenAIAdapter.list_reasoning_models()
        assert "o4-mini" in rm
        assert "gpt-4o-mini" not in rm

    def test_generate_calls_api_with_correct_params(self):
        adapter = self._make_adapter("gpt-4o-mini")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response()
        adapter.client = mock_client

        with patch.object(adapter, "validate_prompt", return_value=True):
            result = adapter.generate("Hello", config=GenerationConfig())

        assert result.text == "OPENAI_OK"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert "temperature" in call_kwargs

    def test_generate_with_reasoning_model_sends_reasoning_payload(self):
        adapter = self._make_adapter("o3-mini")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(model="o3-mini")
        adapter.client = mock_client

        with patch.object(adapter, "validate_prompt", return_value=True):
            result = adapter.generate(
                "Hello",
                config=GenerationConfig(reasoning_effort="minimal"),
            )

        assert result.text == "OPENAI_OK"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "minimal"
        assert "temperature" not in call_kwargs

    def test_generate_invalid_effort_raises(self):
        adapter = self._make_adapter("o3-mini")
        adapter.client = MagicMock()
        with patch.object(adapter, "validate_prompt", return_value=True):
            with pytest.raises(ValueError, match="Invalid reasoning_effort"):
                adapter.generate("Hello", config=GenerationConfig(reasoning_effort="absurd"))
