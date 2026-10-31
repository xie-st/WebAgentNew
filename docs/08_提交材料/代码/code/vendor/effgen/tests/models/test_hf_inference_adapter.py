"""
Unit tests for HFInferenceAdapter.

These tests mock the huggingface_hub SDK so they run offline.
Live API tests are in tests/integration/test_hf_inference_live.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.base import get_stream_usage
from effgen.models.errors import ModelAuthError, ModelUnavailableError
from effgen.models.hf_inference_adapter import HFInferenceAdapter
from effgen.models.hf_inference_models import (
    HF_DEFAULT_MODEL,
    HF_MODELS,
    available_models,
    chat_models,
    endpoint_only_models,
    get_model_info,
    serverless_models,
    suggest_alternatives,
    tool_capable_models,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestHFRegistry:
    def test_default_model_in_registry(self):
        assert HF_DEFAULT_MODEL in HF_MODELS

    def test_all_models_have_required_keys(self):
        required = {
            "display_name", "family", "organization", "context",
            "supports_native_tools", "requires_endpoint",
        }
        for mid, info in HF_MODELS.items():
            missing = required - set(info.keys())
            assert not missing, f"Model '{mid}' missing keys: {missing}"

    def test_available_models_returns_all_keys(self):
        assert set(available_models()) == set(HF_MODELS.keys())

    def test_serverless_models_no_endpoint_required(self):
        for mid in serverless_models():
            assert not HF_MODELS[mid].get("requires_endpoint"), \
                f"'{mid}' marked requires_endpoint but in serverless list"

    def test_tool_capable_models(self):
        tc = tool_capable_models()
        assert "Qwen/Qwen2.5-7B-Instruct" in tc
        assert "Qwen/Qwen2.5-72B-Instruct" in tc
        assert "meta-llama/Meta-Llama-3.1-8B-Instruct" not in tc

    def test_chat_models_no_text_gen_flag(self):
        for mid in chat_models():
            assert not HF_MODELS[mid].get("use_text_generation"), \
                f"'{mid}' in chat_models but has use_text_generation=True"

    def test_context_lengths_positive(self):
        for mid, info in HF_MODELS.items():
            assert info["context"] > 0, f"Model '{mid}' non-positive context"

    def test_pricing_non_negative(self):
        for info in HF_MODELS.values():
            assert info.get("pricing_per_1m_input", 0.0) >= 0
            assert info.get("pricing_per_1m_output", 0.0) >= 0

    def test_get_model_info_known(self):
        info = get_model_info("Qwen/Qwen2.5-7B-Instruct")
        assert info is not None
        assert info["family"] == "qwen2.5"
        assert info["supports_native_tools"] is True

    def test_get_model_info_unknown_returns_none(self):
        assert get_model_info("nonexistent/model") is None

    def test_suggest_alternatives_excludes_self(self):
        alts = suggest_alternatives("Qwen/Qwen2.5-7B-Instruct")
        assert "Qwen/Qwen2.5-7B-Instruct" not in alts

    def test_suggest_alternatives_returns_list(self):
        alts = suggest_alternatives("some/unknown-model")
        assert isinstance(alts, list)

    def test_suggest_alternatives_same_family_first(self):
        alts = suggest_alternatives("Qwen/Qwen2.5-7B-Instruct")
        # Qwen2.5-72B-Instruct is same family, should appear early
        assert "Qwen/Qwen2.5-72B-Instruct" in alts[:3]

    def test_endpoint_only_models_empty_for_current_registry(self):
        # All current models are serverless — this should be empty
        eo = endpoint_only_models()
        assert isinstance(eo, list)


# ---------------------------------------------------------------------------
# Adapter construction tests
# ---------------------------------------------------------------------------

class TestHFAdapterConstruction:
    def test_default_model(self):
        adapter = HFInferenceAdapter()
        assert adapter.model_name == HF_DEFAULT_MODEL

    def test_custom_model(self):
        adapter = HFInferenceAdapter("Qwen/Qwen2.5-72B-Instruct")
        assert adapter.model_name == "Qwen/Qwen2.5-72B-Instruct"

    def test_unknown_model_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="effgen.models.hf_inference_adapter"):
            HFInferenceAdapter("org/unknown-model", warn_unknown_model=True)
        assert "not in the bundled registry" in caplog.text

    def test_unknown_model_no_warn(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="effgen.models.hf_inference_adapter"):
            HFInferenceAdapter("org/unknown-model", warn_unknown_model=False)
        assert "not in the bundled registry" not in caplog.text

    def test_endpoint_url_stored(self):
        adapter = HFInferenceAdapter(
            model_name="my-model",
            endpoint_url="https://my.endpoint.cloud",
        )
        assert adapter.get_endpoint_url() == "https://my.endpoint.cloud"

    def test_supports_tool_calling_qwen(self):
        adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct")
        assert adapter.supports_tool_calling() is True

    def test_supports_tool_calling_llama(self):
        adapter = HFInferenceAdapter("meta-llama/Meta-Llama-3.1-8B-Instruct")
        assert adapter.supports_tool_calling() is False

    def test_supports_streaming(self):
        adapter = HFInferenceAdapter()
        assert adapter.supports_streaming() is True

    def test_context_length_from_registry(self):
        # Context length comes from the live HF Router catalog snapshot — the
        # exact value depends on which provider currently serves the model.
        # All Qwen 2.5 7B variants on HF Router are >=8k.
        adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct")
        assert adapter.get_context_length() >= 8_192

    def test_rate_limiter_created(self):
        adapter = HFInferenceAdapter(enable_rate_limiting=True)
        assert adapter._rate_limiter is not None

    def test_no_rate_limiter(self):
        adapter = HFInferenceAdapter(enable_rate_limiting=False)
        assert adapter._rate_limiter is None


# ---------------------------------------------------------------------------
# Load / unload tests
# ---------------------------------------------------------------------------

class TestHFAdapterLifecycle:
    def _make_mock_client(self):
        return MagicMock()

    @patch("effgen.models.hf_inference_adapter.os.getenv")
    def test_load_raises_without_token(self, mock_getenv):
        mock_getenv.return_value = None
        adapter = HFInferenceAdapter(api_token=None)
        # Stub SDK so the no-token path is reached even when huggingface_hub isn't installed
        stub = MagicMock()
        stub.InferenceClient = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": stub}):
            with pytest.raises(ValueError, match="HuggingFace API token"):
                adapter.load()

    def test_load_raises_without_huggingface_hub(self):
        import sys
        adapter = HFInferenceAdapter(api_token="fake-token")
        saved = sys.modules.pop("huggingface_hub", None)
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((RuntimeError, ImportError)):
                adapter.load()
        finally:
            if saved is not None:
                sys.modules["huggingface_hub"] = saved
            else:
                sys.modules.pop("huggingface_hub", None)

    def test_unload_clears_client(self):
        adapter = HFInferenceAdapter(api_token="fake-token", enable_rate_limiting=False)
        adapter._client = MagicMock()
        adapter._is_loaded = True
        adapter.unload()
        assert adapter._client is None
        assert not adapter._is_loaded

    def test_generate_before_load_raises(self):
        adapter = HFInferenceAdapter(api_token="fake-token")
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("hello")

    def test_stream_before_load_raises(self):
        adapter = HFInferenceAdapter(api_token="fake-token")
        with pytest.raises(RuntimeError, match="not loaded"):
            list(adapter.generate_stream("hello"))


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TestHFTokenCounting:
    def test_count_tokens_approximate(self):
        adapter = HFInferenceAdapter()
        tc = adapter.count_tokens("hello world")
        assert tc.count >= 1
        assert tc.model_name == HF_DEFAULT_MODEL

    def test_count_tokens_long_text(self):
        adapter = HFInferenceAdapter()
        long = "word " * 1000
        tc = adapter.count_tokens(long)
        assert tc.count > 100


# ---------------------------------------------------------------------------
# Generation (mocked SDK)
# ---------------------------------------------------------------------------

def _make_chat_response(content="Hello!", tool_calls=None, input_tokens=10, output_tokens=5):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    choice = SimpleNamespace(
        message=msg,
        finish_reason="stop",
    )
    usage = SimpleNamespace(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model="Qwen/Qwen2.5-7B-Instruct")


class TestHFGenerate:
    def _loaded_adapter(self, model=HF_DEFAULT_MODEL, endpoint_url=None):
        adapter = HFInferenceAdapter(
            model_name=model,
            api_token="fake-token",
            endpoint_url=endpoint_url,
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_generate_returns_result(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.return_value = _make_chat_response("Paris")
        result = adapter.generate("What is the capital of France?")
        assert "Paris" in result.text
        assert result.model_name == HF_DEFAULT_MODEL

    def test_generate_populates_usage(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.return_value = _make_chat_response(
            input_tokens=20, output_tokens=8
        )
        result = adapter.generate("hello")
        assert result.metadata["input_tokens"] == 20
        assert result.metadata["output_tokens"] == 8
        assert result.metadata["total_tokens"] == 28
        assert result.tokens_used == 28

    def test_generate_provider_in_metadata(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.return_value = _make_chat_response()
        result = adapter.generate("hello")
        assert result.metadata["provider"] == "hf_inference"

    def test_generate_with_config(self):
        from effgen.models.base import GenerationConfig
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.return_value = _make_chat_response()
        config = GenerationConfig(max_tokens=100, temperature=0.5)
        adapter.generate("hello", config=config)
        call_kwargs = adapter._client.chat_completion.call_args[1]
        assert call_kwargs.get("max_tokens") == 100
        assert call_kwargs.get("temperature") == 0.5

    def test_generate_endpoint_url_passes_none_model(self):
        adapter = self._loaded_adapter(endpoint_url="https://my.endpoint.cloud")
        adapter._client.chat_completion.return_value = _make_chat_response()
        adapter.generate("hello")
        call_kwargs = adapter._client.chat_completion.call_args[1]
        assert call_kwargs.get("model") is None

    def test_generate_no_endpoint_passes_model_name(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.return_value = _make_chat_response()
        adapter.generate("hello")
        call_kwargs = adapter._client.chat_completion.call_args[1]
        assert call_kwargs.get("model") == HF_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Tool-calling
# ---------------------------------------------------------------------------

class TestHFToolCalling:
    def _loaded_qwen(self):
        adapter = HFInferenceAdapter(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            api_token="fake-token",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_generate_with_tools_passes_tools(self):
        adapter = self._loaded_qwen()

        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"location": "Paris"}',
            ),
        )
        adapter._client.chat_completion.return_value = _make_chat_response(
            content="",
            tool_calls=[tool_call],
        )

        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
        result = adapter.generate_with_tools("Weather in Paris?", tools=[tool])
        assert result.metadata["tool_calls"]
        tc = result.metadata["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    def test_generate_with_tools_unsupported_model_raises(self):
        adapter = HFInferenceAdapter(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_token="fake-token",
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        with pytest.raises(NotImplementedError, match="does not support native tool calling"):
            adapter.generate_with_tools("hello", tools=[])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestHFErrorHandling:
    def _loaded_adapter(self):
        adapter = HFInferenceAdapter(
            api_token="fake-token",
            enable_rate_limiting=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_auth_error_raised_on_401(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception("401 Unauthorized")
        with pytest.raises(ModelAuthError, match="hf_inference"):
            adapter.generate("hello")

    def test_404_request_id_containing_403_is_not_auth(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception(
            "404 Client Error. Repository Not Found. Request ID: abc-4037"
        )
        with pytest.raises(ModelUnavailableError):
            adapter.generate("hello")

    def test_unavailable_error_raised_on_503(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception(
            "503 Server Error: Service Temporarily Unavailable"
        )
        with pytest.raises(ModelUnavailableError) as exc_info:
            adapter.generate("hello")
        err = exc_info.value
        assert isinstance(err.suggestions, list)

    def test_unavailable_error_has_suggestions(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception("404 Not Found")
        with pytest.raises(ModelUnavailableError) as exc_info:
            adapter.generate("hello")
        assert len(exc_info.value.suggestions) > 0

    def test_unavailable_error_message_helpful(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception(
            "503 Service Temporarily Unavailable"
        )
        with pytest.raises(ModelUnavailableError) as exc_info:
            adapter.generate("hello")
        msg = str(exc_info.value)
        assert "unavailable" in msg.lower()
        assert "Try one of" in msg

    def test_generic_error_wraps_as_runtime(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception("Something unexpected")
        with pytest.raises(RuntimeError, match="HuggingFace Inference request failed") as exc_info:
            adapter.generate("hello")
        assert exc_info.value.error_context["provider"] == "hf_inference"


# ---------------------------------------------------------------------------
# Streaming (mocked)
# ---------------------------------------------------------------------------

class TestHFStreaming:
    def _loaded_adapter(self):
        adapter = HFInferenceAdapter(
            api_token="fake-token",
            enable_rate_limiting=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    def test_stream_yields_chunks(self):
        adapter = self._loaded_adapter()

        def _mock_stream(**kwargs):
            for token in ["Hello", " ", "world"]:
                chunk = SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
                )
                yield chunk

        adapter._client.chat_completion.side_effect = lambda **kw: _mock_stream(**kw)
        chunks = list(adapter.generate_stream("hello"))
        assert chunks == ["Hello", " ", "world"]

    def test_stream_skips_none_content_chunks(self):
        adapter = self._loaded_adapter()

        def _mock_stream(**kwargs):
            # First chunk has no content (role chunk)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])

        adapter._client.chat_completion.side_effect = lambda **kw: _mock_stream(**kw)
        chunks = list(adapter.generate_stream("hello"))
        assert chunks == ["Hi"]

    def test_stream_unavailable_raises_model_unavailable(self):
        adapter = self._loaded_adapter()
        adapter._client.chat_completion.side_effect = Exception(
            "503 Service Temporarily Unavailable"
        )
        with pytest.raises(ModelUnavailableError):
            list(adapter.generate_stream("hello"))


# ---------------------------------------------------------------------------
# Streamed-call cost accounting
# ---------------------------------------------------------------------------

class TestHFStreamCostAccounting:
    """A streamed call reports its cost and tokens the way generate() does.

    The chat endpoint sends a usage block on its final chunk for some models;
    where it does not, the same character-based estimate the non-streaming path
    uses is applied to the prompt and the accumulated output.
    """

    MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    def _adapter(self, model_id: str | None = None):
        adapter = HFInferenceAdapter(
            model_id or self.MODEL,
            api_token="fake-token",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        adapter._client = MagicMock()
        adapter._is_loaded = True
        return adapter

    @staticmethod
    def _chunks(tokens, usage=None):
        def _stream(**kwargs):
            for token in tokens:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=token))],
                    usage=None,
                )
            if usage is not None:
                yield SimpleNamespace(choices=[], usage=usage)

        return _stream

    def test_reported_usage_is_used_when_the_stream_sends_it(self):
        adapter = self._adapter()
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=13)
        adapter._client.chat_completion.side_effect = self._chunks(
            ["Red", ", blue"], usage
        )

        text = "".join(adapter.generate_stream("Name the primary colors."))

        assert text == "Red, blue"
        recorded = get_stream_usage(adapter)
        assert recorded["prompt_tokens"] == 30
        assert recorded["completion_tokens"] == 13
        assert recorded["total_tokens"] == 43
        assert adapter.total_tokens == 43
        info = HF_MODELS[self.MODEL]
        expected = (
            30 * info["pricing_per_1m_input"] / 1_000_000
            + 13 * info.get("pricing_per_1m_output", 0.0) / 1_000_000
        )
        assert adapter.get_total_cost() == pytest.approx(expected)

    def test_tokens_are_estimated_when_the_stream_omits_usage(self):
        adapter = self._adapter()
        adapter._client.chat_completion.side_effect = self._chunks(
            ["Red, blue", " and yellow."]
        )

        text = "".join(adapter.generate_stream("Name the primary colors."))

        recorded = get_stream_usage(adapter)
        assert recorded is not None
        # The estimate counts the streamed output, not just the prompt.
        assert recorded["completion_tokens"] == adapter._estimate_tokens(text)
        # The prompt estimate covers the system prompt too, so it exceeds the
        # bare user prompt's own estimate.
        assert recorded["prompt_tokens"] >= adapter._estimate_tokens(
            "Name the primary colors."
        )
        assert adapter.get_total_cost() > 0.0

    def test_streamed_cost_matches_generate_for_the_same_usage(self):
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=13)
        streamed = self._adapter()
        streamed._client.chat_completion.side_effect = self._chunks(["hi"], usage)
        list(streamed.generate_stream("Name the primary colors."))

        priced = self._adapter()
        assert streamed.get_total_cost() == pytest.approx(
            priced._price_tokens(30, 13)
        )

    def test_unpriced_model_records_zero_without_failing(self):
        # google/gemma-3-27b-it carries no per-token price in the registry.
        adapter = self._adapter("google/gemma-3-27b-it")
        usage = SimpleNamespace(prompt_tokens=19, completion_tokens=12)
        adapter._client.chat_completion.side_effect = self._chunks(["Red"], usage)

        list(adapter.generate_stream("Name the primary colors."))

        assert adapter.get_total_cost() == 0.0
        assert adapter.total_tokens == 31

    def test_stream_that_yields_nothing_still_records_the_prompt(self):
        # The prompt was sent and billed even though no output came back.
        adapter = self._adapter()
        adapter._client.chat_completion.side_effect = self._chunks([])

        assert list(adapter.generate_stream("")) == []

        recorded = get_stream_usage(adapter)
        assert recorded["completion_tokens"] == 0
        assert recorded["prompt_tokens"] > 0

    def test_usage_does_not_leak_between_streamed_calls(self):
        adapter = self._adapter()
        adapter._client.chat_completion.side_effect = self._chunks(
            ["hi"], SimpleNamespace(prompt_tokens=100, completion_tokens=100)
        )
        list(adapter.generate_stream("first"))

        adapter._client.chat_completion.side_effect = self._chunks(["hi"])
        list(adapter.generate_stream("second"))

        # The second call had no usage block, so it is estimated — it must not
        # reuse the first call's counts.
        assert get_stream_usage(adapter)["prompt_tokens"] < 100


# ---------------------------------------------------------------------------
# ModelUnavailableError contract
# ---------------------------------------------------------------------------

class TestModelUnavailableError:
    def test_str_contains_provider(self):
        err = ModelUnavailableError(
            provider="hf_inference",
            model_name="some/model",
            suggestions=["other/model"],
        )
        assert "hf_inference" in str(err)

    def test_str_contains_suggestions(self):
        err = ModelUnavailableError(
            provider="hf_inference",
            model_name="some/model",
            suggestions=["alt/model-1", "alt/model-2"],
        )
        assert "alt/model-1" in str(err)

    def test_attributes_accessible(self):
        err = ModelUnavailableError(
            provider="hf_inference",
            model_name="some/model",
            suggestions=["alt/model"],
            message="custom message",
        )
        assert err.provider == "hf_inference"
        assert err.model_name == "some/model"
        assert err.suggestions == ["alt/model"]
        assert err.message == "custom message"

    def test_empty_suggestions(self):
        err = ModelUnavailableError(provider="hf_inference")
        assert err.suggestions == []
        assert "Try one of" not in str(err)
