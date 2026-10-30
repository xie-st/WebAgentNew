"""
Unit tests for TogetherAdapter and together_models registry.
Mocks are OK for adapter plumbing tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from effgen.models import together_models
from effgen.models.together_adapter import TogetherAdapter
from effgen.models.together_models import (
    REGISTRY_FETCH_DATE,
    TOGETHER_DEFAULT_MODEL,
    TOGETHER_EMBEDDING_MODELS,
    TOGETHER_LANGUAGE_MODELS,
    TOGETHER_MODELS,
    available_models,
    chat_models,
    model_info,
    pricing_table,
    serverless_models,
    tool_capable_models,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestTogetherModelsRegistry:
    def test_available_models_nonempty(self):
        assert len(available_models()) > 0

    def test_chat_models_count(self):
        assert len(chat_models()) == 168

    def test_chat_models_match_bundled_snapshot(self):
        """The in-package catalog and the shipped snapshot list the same ids."""
        snapshot = json.loads(
            (Path(together_models.__file__).parent / "_data" / "together.json").read_text()
        )
        assert {m["id"] for m in snapshot["models"]} == set(chat_models())
        assert snapshot["count"] == len(chat_models())

    def test_serverless_models_nonempty(self):
        assert len(serverless_models()) > 0

    def test_serverless_subset_of_chat(self):
        assert set(serverless_models()).issubset(set(available_models()))

    def test_tool_capable_nonempty(self):
        assert len(tool_capable_models()) > 0

    def test_tool_capable_subset_of_chat(self):
        assert set(tool_capable_models()).issubset(set(chat_models()))

    def test_default_model_in_registry(self):
        assert TOGETHER_DEFAULT_MODEL in TOGETHER_MODELS

    def test_default_model_is_serverless(self):
        assert TOGETHER_MODELS[TOGETHER_DEFAULT_MODEL].get("serverless", False) is True

    def test_model_info_known(self):
        info = model_info("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        assert info["context"] == 131_072
        assert info["family"] == "llama"
        assert info["serverless"] is True

    def test_model_info_language_model(self):
        # Language models are in TOGETHER_LANGUAGE_MODELS
        info = model_info("meta-llama/Meta-Llama-3.1-8B")
        assert info["modality"] == "language"

    def test_model_info_unknown_raises(self):
        with pytest.raises(KeyError):
            model_info("nonexistent-model-xyz-999")

    def test_all_chat_models_have_required_fields(self):
        required = {"context", "max_output", "supports_native_tools", "supports_streaming", "serverless"}
        for mid, info in TOGETHER_MODELS.items():
            for field in required:
                assert field in info, f"Model {mid!r} missing field {field!r}"

    def test_pricing_table_sorted_by_input(self):
        rows = pricing_table()
        prices = [r["input_per_1m_usd"] for r in rows]
        assert prices == sorted(prices)

    def test_pricing_table_contains_all_models(self):
        assert len(pricing_table()) == len(TOGETHER_MODELS)

    def test_language_models_nonempty(self):
        assert len(TOGETHER_LANGUAGE_MODELS) > 0

    def test_embedding_models_nonempty(self):
        assert len(TOGETHER_EMBEDDING_MODELS) > 0

    def test_registry_fetch_date_format(self):
        # Should be YYYY-MM-DD
        parts = REGISTRY_FETCH_DATE.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year

    def test_serverless_models_have_serverless_true(self):
        for mid in serverless_models():
            assert TOGETHER_MODELS[mid]["serverless"] is True

    def test_known_serverless_models_present(self):
        expected = [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen3.5-9B",
            "openai/gpt-oss-20b",
        ]
        sm = set(serverless_models())
        for mid in expected:
            assert mid in sm, f"{mid!r} should be in serverless_models()"

    def test_dedicated_endpoint_models_are_not_serverless(self):
        """Models Together serves only from a dedicated endpoint must not be
        advertised as serverless — `is_serverless` and the "use a serverless
        model" hint on InvalidRequestError both read this flag."""
        dedicated = [
            "deepseek-ai/DeepSeek-V3.1",
            "deepseek-ai/DeepSeek-R1-0528",
            "zai-org/GLM-5",
            "zai-org/GLM-5.1",
            "Qwen/Qwen3-Coder-Next-FP8",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "Qwen/Qwen3.5-397B-A17B",
        ]
        sm = set(serverless_models())
        for mid in dedicated:
            assert mid in TOGETHER_MODELS, f"{mid!r} missing from the catalog"
            assert mid not in sm, f"{mid!r} needs a dedicated endpoint, not serverless"


# ---------------------------------------------------------------------------
# Adapter construction tests
# ---------------------------------------------------------------------------

class TestTogetherAdapterInit:
    def test_init_default_model(self):
        adapter = TogetherAdapter(enable_rate_limiting=False, enable_cost_tracking=False)
        assert adapter.model_name == TOGETHER_DEFAULT_MODEL

    def test_init_known_model(self):
        adapter = TogetherAdapter(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        assert adapter.model_name == "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def test_init_unknown_model_raises(self):
        from effgen.models.errors import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="Unknown Together model"):
            TogetherAdapter("nonexistent-model-xyz-999")

    def test_init_context_length(self):
        adapter = TogetherAdapter(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        assert adapter.get_context_length() == 131_072

    def test_supports_tools_property(self):
        adapter = TogetherAdapter(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        assert adapter.supports_native_tools is True

    def test_is_serverless_property(self):
        adapter = TogetherAdapter(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        assert adapter.is_serverless is True

    def test_pricing_method(self):
        adapter = TogetherAdapter(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        p = adapter.pricing()
        assert "input_per_1m_usd" in p
        assert "output_per_1m_usd" in p
        assert p["input_per_1m_usd"] == 1.04

    def test_rate_limiting_enabled_by_default(self):
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            enable_rate_limiting=True,
            enable_cost_tracking=False,
        )
        status = adapter.rate_limit_status()
        assert status["enabled"] is True

    def test_rate_limiting_disabled(self):
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        status = adapter.rate_limit_status()
        assert status["enabled"] is False

    def test_not_loaded_raises_on_generate(self):
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("hello")

    def test_not_loaded_raises_on_stream(self):
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            list(adapter.generate_stream("hello"))


# ---------------------------------------------------------------------------
# Adapter load/unload with mocked SDK
# ---------------------------------------------------------------------------

class TestTogetherAdapterLoad:
    def _patch_together(self):
        """Context manager: mock the 'together' SDK import inside load()."""
        import together as _together_mod
        mock_client = MagicMock()
        mock_together_cls = MagicMock(return_value=mock_client)
        return patch.object(_together_mod, "Together", mock_together_cls), mock_client

    def test_load_success(self):
        import together as _together_mod
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            api_key="test-key",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        mock_client = MagicMock()
        with patch.object(_together_mod, "Together", return_value=mock_client) as mock_together:
            adapter.load()
        assert adapter._is_loaded is True
        assert adapter._client is mock_client
        mock_together.assert_called_once_with(
            api_key="test-key",
            timeout=adapter.timeout,
            # The SDK's own retry is off: the adapter runs the backoff, and a
            # second layer underneath multiplies the upstream requests one
            # client request becomes rather than sharing the budget.
            max_retries=0,
        )

    def test_load_missing_key_raises(self):
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            api_key=None,
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        import os
        saved = os.environ.pop("TOGETHER_API_KEY", None)
        # Stub SDK so the no-key path is reached even when together isn't installed
        stub = MagicMock()
        stub.Together = MagicMock()
        try:
            with patch.dict("sys.modules", {"together": stub}):
                with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                    adapter.load()
        finally:
            if saved is not None:
                os.environ["TOGETHER_API_KEY"] = saved

    def test_load_missing_sdk_raises(self):
        import builtins
        import sys
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "together":
                raise ImportError("No module named 'together'")
            return real_import(name, *args, **kwargs)

        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            api_key="test-key",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        saved = sys.modules.pop("together", None)
        try:
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises((RuntimeError, ImportError)):
                    adapter.load()
        finally:
            if saved is not None:
                sys.modules["together"] = saved

    def test_unload_clears_client(self):
        import together as _together_mod
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            api_key="test-key",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        mock_client = MagicMock()
        with patch.object(_together_mod, "Together", return_value=mock_client):
            adapter.load()
        adapter.unload()
        assert adapter._client is None
        assert adapter._is_loaded is False

    def test_metadata_populated_after_load(self):
        import together as _together_mod
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            api_key="test-key",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        mock_client = MagicMock()
        with patch.object(_together_mod, "Together", return_value=mock_client):
            adapter.load()
        assert adapter._metadata["provider"] == "together"
        assert adapter._metadata["model_name"] == TOGETHER_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Adapter generate with mocked SDK
# ---------------------------------------------------------------------------

def _make_mock_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 20):
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    choice.message = message
    choice.finish_reason = "stop"
    response.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response.usage = usage
    return response


class TestTogetherAdapterGenerate:
    def _loaded_adapter(self, model_name=TOGETHER_DEFAULT_MODEL):
        import together as _together_mod
        adapter = TogetherAdapter(
            model_name,
            api_key="test-key",
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        mock_client = MagicMock()
        with patch.object(_together_mod, "Together", return_value=mock_client):
            adapter.load()
        return adapter, mock_client

    def test_generate_returns_text(self):
        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.return_value = _make_mock_response("Paris")
        result = adapter.generate("Capital of France?")
        assert result.text == "Paris"
        assert result.model_name == TOGETHER_DEFAULT_MODEL

    def test_generate_populates_usage(self):
        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Paris", prompt_tokens=5, completion_tokens=3
        )
        result = adapter.generate("test")
        assert result.metadata["prompt_tokens"] == 5
        assert result.metadata["completion_tokens"] == 3
        assert result.metadata["total_tokens"] == 8
        assert result.metadata["provider"] == "together"

    def test_generate_forwards_penalties(self):
        """A pinned penalty must reach the request, matching seed/top_p."""
        from effgen.models.base import GenerationConfig
        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.return_value = _make_mock_response("hi")
        adapter.generate(
            "Hi",
            config=GenerationConfig(frequency_penalty=1.5, presence_penalty=0.5),
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["frequency_penalty"] == 1.5
        assert call_kwargs["presence_penalty"] == 0.5

    def test_generate_omits_default_penalties(self):
        """The neutral 0.0 default is not sent."""
        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.return_value = _make_mock_response("hi")
        adapter.generate("Hi")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs

    def test_generate_with_tools_passes_tools(self):
        adapter, mock_client = self._loaded_adapter(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        mock_response = _make_mock_response("")
        tc = MagicMock()
        tc.id = "call_abc"
        tc.type = "function"
        tc.function.name = "calculator"
        tc.function.arguments = '{"expression": "2+2"}'
        mock_response.choices[0].message.tool_calls = [tc]
        mock_client.chat.completions.create.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "calculator", "description": "calc", "parameters": {}}}]
        result = adapter.generate_with_tools("2+2?", tools=tools)
        assert len(result.metadata["tool_calls"]) == 1
        assert result.metadata["tool_calls"][0]["function"]["name"] == "calculator"

    def test_generate_stream_yields_chunks(self):
        adapter, mock_client = self._loaded_adapter()

        chunks = []
        for text in ["Hello", " world", "!"]:
            chunk = MagicMock()
            choice = MagicMock()
            delta = MagicMock()
            delta.content = text
            delta.tool_calls = None
            choice.delta = delta
            choice.finish_reason = None
            chunk.choices = [choice]
            chunk.usage = None
            chunks.append(chunk)

        mock_client.chat.completions.create.return_value = iter(chunks)
        result = list(adapter.generate_stream("Hi"))
        assert result == ["Hello", " world", "!"]

    def test_auth_error_raises_model_auth_error(self):
        from effgen.models.errors import ModelAuthError
        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized invalid_api_key")
        with pytest.raises(ModelAuthError):
            adapter.generate("test")

    def test_endpoint_error_raises_invalid_request_error(self):
        from effgen.models.errors import InvalidRequestError

        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.side_effect = Exception(
            "400 Unable to access non-serverless model"
        )
        with pytest.raises(InvalidRequestError, match="dedicated endpoint"):
            adapter.generate("test")

    def test_endpoint_error_is_non_retryable_configuration_error(self):
        """A dedicated-endpoint-not-running condition is permanent, not
        transient — it must not be retried or burn a fallback_chain attempt.
        """
        from effgen.models.errors import classify_provider_error

        adapter, mock_client = self._loaded_adapter()
        mock_client.chat.completions.create.side_effect = Exception(
            "400 dedicated_endpoint_not_running"
        )
        try:
            adapter.generate("test")
        except Exception as exc:
            ec = classify_provider_error(exc)
            assert ec.category == "invalid_request"
            assert ec.should_retry is False
        else:
            raise AssertionError("expected the endpoint error to raise")

    def test_count_tokens(self):
        adapter = TogetherAdapter(
            TOGETHER_DEFAULT_MODEL,
            enable_rate_limiting=False,
            enable_cost_tracking=False,
        )
        result = adapter.count_tokens("Hello world")
        assert result.count > 0
        assert result.model_name == TOGETHER_DEFAULT_MODEL
