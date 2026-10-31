"""
Unit tests for FireworksAdapter.

Uses mocks for the fireworks-ai SDK calls so no API key is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.fireworks_adapter import _FIREWORKS_PREFIX, FireworksAdapter
from effgen.models.fireworks_models import (
    FIREWORKS_DEFAULT_MODEL,
    FIREWORKS_MODELS,
    available_models,
    chat_models,
    pricing_table,
    refresh_models,
    tool_capable_models,
)

# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestFireworksRegistry:
    def test_models_not_empty(self):
        assert len(FIREWORKS_MODELS) > 0

    def test_all_ids_have_prefix(self):
        for mid in FIREWORKS_MODELS:
            assert mid.startswith(_FIREWORKS_PREFIX), f"{mid} missing prefix"

    def test_required_fields(self):
        required = {"context", "supports_native_tools", "supports_streaming"}
        for mid, info in FIREWORKS_MODELS.items():
            for field in required:
                assert field in info, f"{mid} missing field '{field}'"

    def test_pricing_keys_come_as_a_pair(self):
        """An entry carries both prices or neither; one alone reports as free."""
        for mid, info in FIREWORKS_MODELS.items():
            has_in = "pricing_per_1m_input" in info
            has_out = "pricing_per_1m_output" in info
            assert has_in == has_out, f"{mid} has only one pricing key"

    def test_context_positive(self):
        for mid, info in FIREWORKS_MODELS.items():
            # A model Fireworks publishes no context length for is recorded as 0
            # and flagged, so an accidental 0 still fails here.
            if info.get("context_unpublished"):
                assert info["context"] == 0, f"{mid} flagged unpublished but has a context"
                continue
            assert info["context"] > 0, f"{mid} has context={info['context']}"

    def test_default_model_in_registry(self):
        assert FIREWORKS_DEFAULT_MODEL in FIREWORKS_MODELS

    def test_available_models_returns_list(self):
        models = available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(m.startswith(_FIREWORKS_PREFIX) for m in models)

    def test_chat_models_subset(self):
        chat = chat_models()
        assert set(chat).issubset(set(FIREWORKS_MODELS.keys()))

    def test_tool_capable_models(self):
        tool_mods = tool_capable_models()
        assert len(tool_mods) > 0
        for mid in tool_mods:
            assert FIREWORKS_MODELS[mid]["supports_native_tools"]

    def test_pricing_table_structure(self):
        rows = pricing_table()
        assert len(rows) == len(FIREWORKS_MODELS)
        for row in rows:
            assert "model" in row
            assert "input_per_1m_usd" in row
            assert "output_per_1m_usd" in row
            assert "context" in row

    def test_pricing_sorted_ascending(self):
        rows = pricing_table()
        prices = [r["input_per_1m_usd"] for r in rows]
        assert prices == sorted(prices)

    def test_families_diverse(self):
        families = {info.get("family") for info in FIREWORKS_MODELS.values()}
        # Should have at least llama, deepseek, qwen3, mistral, kimi families
        assert len(families) >= 5


# ---------------------------------------------------------------------------
# Adapter instantiation tests
# ---------------------------------------------------------------------------

class TestFireworksAdapterInit:
    def test_default_model(self):
        adapter = FireworksAdapter(enable_rate_limiting=False)
        assert adapter.model_name == FIREWORKS_DEFAULT_MODEL

    def test_short_id_expansion(self):
        adapter = FireworksAdapter("kimi-k2p6", enable_rate_limiting=False)
        assert adapter.model_name == f"{_FIREWORKS_PREFIX}kimi-k2p6"

    def test_full_id_unchanged(self):
        full = f"{_FIREWORKS_PREFIX}kimi-k2p6"
        adapter = FireworksAdapter(full, enable_rate_limiting=False)
        assert adapter.model_name == full

    def test_unknown_model_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="effgen.models.fireworks_adapter"):
            FireworksAdapter("accounts/fireworks/models/unknown-xyz", enable_rate_limiting=False)
        assert any("not in the bundled registry" in r.message for r in caplog.records)

    def test_unknown_model_no_warn_when_disabled(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="effgen.models.fireworks_adapter"):
            FireworksAdapter(
                "accounts/fireworks/models/unknown-xyz",
                warn_unknown_model=False,
                enable_rate_limiting=False,
            )
        assert not any("not in the bundled registry" in r.message for r in caplog.records)

    def test_rate_limiter_created_for_known_model(self):
        adapter = FireworksAdapter(
            f"{_FIREWORKS_PREFIX}kimi-k2p6",
            enable_rate_limiting=True,
        )
        assert adapter._rate_limiter is not None

    def test_rate_limiter_disabled(self):
        adapter = FireworksAdapter(
            f"{_FIREWORKS_PREFIX}kimi-k2p6",
            enable_rate_limiting=False,
        )
        assert adapter._rate_limiter is None


# ---------------------------------------------------------------------------
# Load / unload tests
# ---------------------------------------------------------------------------

class TestFireworksAdapterLoadUnload:
    def _make_adapter(self, model: str | None = None):
        return FireworksAdapter(
            model or FIREWORKS_DEFAULT_MODEL,
            api_key="fw_test_key",
            enable_rate_limiting=False,
        )

    def test_load_no_sdk_raises(self):
        adapter = self._make_adapter()
        with patch.dict("sys.modules", {"fireworks": None, "fireworks.client": None}):
            with pytest.raises(RuntimeError, match="fireworks-ai SDK is not installed"):
                adapter.load()

    def test_load_no_key_raises(self):
        adapter = FireworksAdapter(
            FIREWORKS_DEFAULT_MODEL,
            api_key=None,
            enable_rate_limiting=False,
        )
        # Stub SDK so the no-key path is reached even when fireworks-ai isn't installed
        client_stub = MagicMock()
        client_stub.Fireworks = MagicMock()
        fw_root = MagicMock()
        fw_root.client = client_stub
        with patch.dict("sys.modules", {"fireworks": fw_root, "fireworks.client": client_stub}):
            with patch("os.getenv", return_value=None):
                with pytest.raises(ValueError, match="Fireworks API key not found"):
                    adapter.load()

    def test_load_sets_is_loaded(self):
        adapter = self._make_adapter()
        mock_fireworks = MagicMock()
        with patch("effgen.models.fireworks_adapter.FireworksAdapter.load"):
            adapter._client = mock_fireworks
            adapter._is_loaded = True
        assert adapter._is_loaded

    def test_load_passes_timeout_to_sdk(self):
        adapter = FireworksAdapter(
            FIREWORKS_DEFAULT_MODEL,
            api_key="fw_test_key",
            timeout=17,
            enable_rate_limiting=False,
        )
        client_stub = MagicMock()
        client_stub.Fireworks = MagicMock()
        fw_root = MagicMock()
        fw_root.client = client_stub

        with patch.dict("sys.modules", {"fireworks": fw_root, "fireworks.client": client_stub}):
            adapter.load()

        client_stub.Fireworks.assert_called_once_with(
            api_key="fw_test_key",
            timeout=17,
        )
        assert adapter.is_loaded()

    def test_unload_clears_client(self):
        adapter = self._make_adapter()
        adapter._client = MagicMock()
        adapter._is_loaded = True
        adapter.unload()
        assert adapter._client is None
        assert not adapter._is_loaded

    def test_metadata_populated_after_load(self):
        adapter = self._make_adapter()
        mock_client_cls = MagicMock()
        with patch("effgen.models.fireworks_adapter.FireworksAdapter.load"):
            adapter._client = mock_client_cls
            adapter._is_loaded = True
            # Manually trigger metadata (would be set in real load)
            info = FIREWORKS_MODELS.get(adapter.model_name, {})
            adapter._metadata = {
                "model_name": adapter.model_name,
                "provider": "fireworks",
                "supports_native_tools": info.get("supports_native_tools", False),
            }
        assert adapter._metadata["provider"] == "fireworks"


# ---------------------------------------------------------------------------
# Generate tests (mocked SDK)
# ---------------------------------------------------------------------------

def _make_mock_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a mock chat completion response object."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = None
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _loaded_adapter(model: str | None = None) -> FireworksAdapter:
    adapter = FireworksAdapter(
        model or FIREWORKS_DEFAULT_MODEL,
        api_key="fw_test_key",
        enable_rate_limiting=False,
        enable_cost_tracking=False,
    )
    adapter._client = MagicMock()
    adapter._is_loaded = True
    return adapter


class TestFireworksAdapterGenerate:
    def test_generate_returns_text(self):
        adapter = _loaded_adapter()
        mock_resp = _make_mock_response("Paris is the capital of France.")
        adapter._client.chat.completions.create.return_value = mock_resp

        result = adapter.generate("What is the capital of France?")
        assert result.text == "Paris is the capital of France."

    def test_generate_populates_usage(self):
        adapter = _loaded_adapter()
        mock_resp = _make_mock_response("Hello", prompt_tokens=20, completion_tokens=3)
        adapter._client.chat.completions.create.return_value = mock_resp

        result = adapter.generate("Say hello")
        assert result.tokens_used == 3
        assert result.metadata is not None
        assert result.metadata["prompt_tokens"] == 20
        assert result.metadata["completion_tokens"] == 3

    def test_generate_not_loaded_raises(self):
        adapter = FireworksAdapter(
            FIREWORKS_DEFAULT_MODEL,
            api_key="fw_test_key",
            enable_rate_limiting=False,
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("Hello")

    def test_generate_auth_error(self):
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "401 Unauthorized: Invalid API key"
        )
        from effgen.models.errors import ModelAuthError
        with pytest.raises(ModelAuthError):
            adapter.generate("Hello")

    def test_generate_model_not_found_error(self):
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "NOT_FOUND: Model not found, inaccessible, and/or not deployed"
        )
        from effgen.models.errors import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="was not found") as exc_info:
            adapter.generate("Hello")
        # Fireworks answers a key-visibility failure the same way, so the
        # message names both causes.
        assert "cannot access" in str(exc_info.value)

    def test_generate_retries_on_rate_limit(self):
        adapter = _loaded_adapter()
        mock_resp = _make_mock_response("ok")
        adapter._client.chat.completions.create.side_effect = [
            Exception("429 rate limit exceeded"),
            mock_resp,
        ]
        with patch("time.sleep"):
            result = adapter.generate("Hello")
        assert result.text == "ok"
        assert adapter._client.chat.completions.create.call_count == 2

    def test_generate_with_config(self):
        from effgen.models.base import GenerationConfig
        adapter = _loaded_adapter()
        mock_resp = _make_mock_response("hello")
        adapter._client.chat.completions.create.return_value = mock_resp

        config = GenerationConfig(max_tokens=100, temperature=0.5, seed=42)
        result = adapter.generate("Hi", config=config)
        assert result.text == "hello"

        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["seed"] == 42

    def test_generate_forwards_penalties(self):
        """A pinned penalty must reach the request, matching seed/top_p."""
        from effgen.models.base import GenerationConfig
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.return_value = _make_mock_response("hi")
        adapter.generate(
            "Hi",
            config=GenerationConfig(frequency_penalty=1.5, presence_penalty=0.5),
        )
        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert call_kwargs["frequency_penalty"] == 1.5
        assert call_kwargs["presence_penalty"] == 0.5

    def test_generate_omits_default_penalties(self):
        """The neutral 0.0 default is not sent."""
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.return_value = _make_mock_response("hi")
        adapter.generate("Hi")
        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs


# ---------------------------------------------------------------------------
# Tool-calling tests (mocked)
# ---------------------------------------------------------------------------

def _make_tool_response(tool_name: str, tool_args: dict):
    """Build a mock response with a tool call."""
    tc = MagicMock()
    tc.id = "call_abc123"
    tc.type = "function"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_args)

    choice = MagicMock()
    choice.message.content = ""
    choice.message.tool_calls = [tc]
    choice.finish_reason = "tool_calls"

    usage = MagicMock()
    usage.prompt_tokens = 50
    usage.completion_tokens = 20
    usage.total_tokens = 70

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestFireworksAdapterTools:
    def test_generate_with_tools_returns_tool_calls(self):
        model = f"{_FIREWORKS_PREFIX}kimi-k2p6"
        adapter = _loaded_adapter(model)
        mock_resp = _make_tool_response("calculator", {"expression": "17*23"})
        adapter._client.chat.completions.create.return_value = mock_resp

        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }]
        result = adapter.generate_with_tools("What is 17*23?", tools=tools)
        assert result.metadata is not None
        tc = result.metadata["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "calculator"
        # The reported element shape is pinned once, for every adapter, in
        # tests/unit/test_adapter_consistency.py.

    def test_tools_not_passed_for_non_tool_model(self):
        non_tool_model = f"{_FIREWORKS_PREFIX}qwen3-embedding-8b"
        adapter = _loaded_adapter(non_tool_model)
        mock_resp = _make_mock_response("result")
        adapter._client.chat.completions.create.return_value = mock_resp

        tools = [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}]
        adapter.generate_with_tools("Calculate 2+2", tools=tools)
        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_supports_tool_calling_property(self):
        # kimi-k2p6 returns native tool calls
        tool_model = f"{_FIREWORKS_PREFIX}kimi-k2p6"
        adapter = _loaded_adapter(tool_model)
        assert adapter.supports_tool_calling() is True
        assert adapter.supports_native_tools is True

        # Embedding-modality models do not support tools.
        embedding_model = f"{_FIREWORKS_PREFIX}qwen3-embedding-8b"
        adapter2 = _loaded_adapter(embedding_model)
        assert adapter2.supports_tool_calling() is False


# ---------------------------------------------------------------------------
# Streaming tests (mocked)
# ---------------------------------------------------------------------------

def _make_stream_chunks(text: str):
    """Generate mock SSE chunks."""
    chunks = []
    for char in text:
        delta = MagicMock()
        delta.content = char
        delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = None
        chunks.append(chunk)

    # Final chunk with finish_reason and usage
    delta_final = MagicMock()
    delta_final.content = None
    delta_final.tool_calls = None
    choice_final = MagicMock()
    choice_final.delta = delta_final
    choice_final.finish_reason = "stop"
    chunk_final = MagicMock()
    chunk_final.choices = [choice_final]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = len(text)
    chunk_final.usage = usage
    chunks.append(chunk_final)
    return chunks


class TestFireworksAdapterStream:
    def test_stream_yields_chunks(self):
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.return_value = iter(
            _make_stream_chunks("Hello world")
        )
        chunks = list(adapter.generate_stream("Say hello"))
        assert "".join(chunks) == "Hello world"

    def test_stream_not_loaded_raises(self):
        adapter = FireworksAdapter(
            FIREWORKS_DEFAULT_MODEL,
            api_key="fw_test_key",
            enable_rate_limiting=False,
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            list(adapter.generate_stream("Hello"))

    def test_stream_auth_error(self):
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.side_effect = Exception(
            "401 Unauthorized: Invalid API key"
        )
        from effgen.models.errors import ModelAuthError
        with pytest.raises(ModelAuthError):
            list(adapter.generate_stream("Hello"))

    def test_stream_uses_stream_param(self):
        adapter = _loaded_adapter()
        adapter._client.chat.completions.create.return_value = iter([])
        list(adapter.generate_stream("Hello"))
        call_kwargs = adapter._client.chat.completions.create.call_args[1]
        assert call_kwargs.get("stream") is True


# ---------------------------------------------------------------------------
# Token count / context length tests
# ---------------------------------------------------------------------------

class TestFireworksAdapterTokens:
    def test_count_tokens_returns_count(self):
        adapter = _loaded_adapter()
        result = adapter.count_tokens("Hello world")
        assert result.count > 0
        assert result.model_name == FIREWORKS_DEFAULT_MODEL

    def test_get_context_length(self):
        model = f"{_FIREWORKS_PREFIX}kimi-k2p6"
        adapter = _loaded_adapter(model)
        ctx = adapter.get_context_length()
        assert ctx == FIREWORKS_MODELS[model]["context"]

    def test_context_length_unknown_model(self):
        adapter = FireworksAdapter(
            "accounts/fireworks/models/unknown-xyz",
            enable_rate_limiting=False,
            warn_unknown_model=False,
        )
        ctx = adapter.get_context_length()
        assert ctx == 131_072  # default fallback

    def test_unpublished_context_does_not_reject_every_prompt(self):
        """An unknown context window is not a zero-token window.

        Fireworks publishes ``contextLength: 0`` for some custom deployments, and
        the catalog records that faithfully alongside ``context_unpublished``.
        Validating a prompt against 0 would refuse every prompt for such a model.
        """
        unpublished = [m for m, i in FIREWORKS_MODELS.items() if i.get("context_unpublished")]
        if not unpublished:
            pytest.skip("no catalog entry currently has an unpublished context window")
        adapter = _loaded_adapter(unpublished[0])
        assert adapter.get_context_length() == 0
        assert adapter.validate_prompt("What is the capital of France?") is True

    def test_a_known_context_window_is_still_enforced(self):
        """The skip above must not disable the check for models that publish one."""
        model = f"{_FIREWORKS_PREFIX}kimi-k2p6"
        adapter = _loaded_adapter(model)
        window = FIREWORKS_MODELS[model]["context"]
        assert window > 0
        with pytest.raises(ValueError, match="exceeds"):
            adapter.validate_prompt("word " * (window + 1000))


# ---------------------------------------------------------------------------
# Helpers / properties
# ---------------------------------------------------------------------------

class TestFireworksAdapterHelpers:
    def test_pricing(self):
        adapter = _loaded_adapter(f"{_FIREWORKS_PREFIX}kimi-k2p6")
        p = adapter.pricing()
        assert "input_per_1m_usd" in p
        assert "output_per_1m_usd" in p

    def test_supports_streaming_property(self):
        adapter = _loaded_adapter()
        assert adapter.supports_streaming is True

    def test_rate_limit_status_disabled(self):
        adapter = _loaded_adapter()
        status = adapter.rate_limit_status()
        assert status["enabled"] is False


# ---------------------------------------------------------------------------
# Registry refresh tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFireworksRefreshModels:
    def test_refresh_no_key_raises(self):
        with patch("os.getenv", return_value=None):
            with pytest.raises(ValueError, match="FIREWORKS_API_KEY not set"):
                refresh_models(api_key=None)

    def test_refresh_no_requests_raises(self):
        with patch.dict("sys.modules", {"requests": None}):
            with pytest.raises(RuntimeError, match="requests is required"):
                refresh_models(api_key="fw_test")

    def test_refresh_returns_summary(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "accounts/fireworks/models/kimi-k2p6",
                    "contextLength": 131072,
                    "displayName": "Llama 3.3 70B Instruct",
                    "baseModelDetails": {"modelType": "llama"},
                    "deprecationDate": None,
                }
            ],
            "nextPageToken": None,
        }

        import requests
        with patch.object(requests, "get", return_value=mock_response):
            result = refresh_models(api_key="fw_test", warn_on_drift=False)
        assert "live_total" in result
        assert "bundled_total" in result
        assert "new_models" in result
        assert "removed_models" in result

    def test_refresh_detects_new_models(self, caplog):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "accounts/fireworks/models/brand-new-model-9000",
                    "contextLength": 131072,
                    "displayName": "Brand New Model",
                    "baseModelDetails": {},
                    "deprecationDate": None,
                }
            ],
            "nextPageToken": None,
        }

        import logging

        import requests
        with caplog.at_level(logging.WARNING, logger="effgen.models.fireworks_models"):
            with patch.object(requests, "get", return_value=mock_response):
                result = refresh_models(api_key="fw_test", warn_on_drift=True)
        assert "accounts/fireworks/models/brand-new-model-9000" in result["new_models"]
        assert any("drift" in r.message for r in caplog.records)
