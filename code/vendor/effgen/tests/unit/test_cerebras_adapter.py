"""Unit tests for CerebrasAdapter — mocks allowed here (unit test boundary)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.base import GenerationConfig, GenerationResult, TokenCount
from effgen.models.cerebras_adapter import CerebrasAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(
    text: str = "hello",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _mock_cerebras_sdk() -> MagicMock:
    """Return a mock Cerebras SDK module that can be injected via sys.modules."""
    mock_sdk = MagicMock()
    mock_class = MagicMock()
    mock_sdk.Cerebras = mock_class
    # Build the module tree
    cerebras_mod = MagicMock()
    cerebras_cloud_mod = MagicMock()
    cerebras_cloud_mod.sdk = mock_sdk
    cerebras_mod.cloud = cerebras_cloud_mod
    return cerebras_mod, cerebras_cloud_mod, mock_sdk, mock_class


def _inject_sdk_into_adapter(mock_client: MagicMock, api_key: str = "test-key") -> CerebrasAdapter:
    """Create and load a CerebrasAdapter with a fake client injected directly."""
    adapter = CerebrasAdapter(enable_rate_limiting=False)
    adapter._client = mock_client
    adapter._is_loaded = True
    adapter._metadata = {
        "model_name": adapter.model_name,
        "context_length": adapter.get_context_length(),
        "provider": "cerebras",
        "free_tier": True,
    }
    return adapter


# ---------------------------------------------------------------------------
# 1. Constructor defaults
# ---------------------------------------------------------------------------

class TestCerebrasAdapterInit:
    def test_default_model(self):
        adapter = CerebrasAdapter()
        assert adapter.model_name == "gpt-oss-120b"

    def test_custom_model(self):
        adapter = CerebrasAdapter(model_name="gpt-oss-120b")
        assert adapter.model_name == "gpt-oss-120b"

    def test_not_loaded_on_init(self):
        adapter = CerebrasAdapter()
        assert not adapter.is_loaded()
        assert adapter._client is None

    def test_unknown_model_raises_value_error(self):
        from effgen.models.errors import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="Unknown Cerebras model"):
            CerebrasAdapter(model_name="does-not-exist")


# ---------------------------------------------------------------------------
# 2. load() reads env key
# ---------------------------------------------------------------------------

class TestCerebrasAdapterLoad:
    def _sdk_patch(self, api_key_to_expect: str) -> tuple:
        """Return a mock_class that tracks instantiation."""
        mock_client = MagicMock()
        mock_class = MagicMock(return_value=mock_client)

        # Build minimal module tree
        mock_sdk_module = MagicMock()
        mock_sdk_module.Cerebras = mock_class

        modules = {
            "cerebras": MagicMock(cloud=MagicMock(sdk=mock_sdk_module)),
            "cerebras.cloud": MagicMock(sdk=mock_sdk_module),
            "cerebras.cloud.sdk": mock_sdk_module,
        }
        return modules, mock_class, mock_client

    def test_load_reads_env_key(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key-abc")
        modules, mock_class, mock_client = self._sdk_patch("test-key-abc")
        with patch.dict(sys.modules, modules):
            adapter = CerebrasAdapter(enable_rate_limiting=False)
            adapter.load()
            mock_class.assert_called_once_with(
                api_key="test-key-abc",
                timeout=adapter.timeout,
                max_retries=0,
            )
            assert adapter.is_loaded()

    def test_load_uses_explicit_key(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        modules, mock_class, mock_client = self._sdk_patch("explicit-key")
        with patch.dict(sys.modules, modules):
            adapter = CerebrasAdapter(api_key="explicit-key", enable_rate_limiting=False)
            adapter.load()
            mock_class.assert_called_once_with(
                api_key="explicit-key",
                timeout=adapter.timeout,
                max_retries=0,
            )

    def test_load_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        modules, _, _ = self._sdk_patch("unused")
        with patch.dict(sys.modules, modules):
            adapter = CerebrasAdapter(enable_rate_limiting=False)
            with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
                adapter.load()

    def test_load_raises_if_sdk_missing(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "k")
        with patch.dict(sys.modules, {"cerebras": None, "cerebras.cloud": None, "cerebras.cloud.sdk": None}):
            adapter = CerebrasAdapter(enable_rate_limiting=False)
            with pytest.raises((RuntimeError, ImportError)):
                adapter.load()


# ---------------------------------------------------------------------------
# 3. generate() formats messages correctly
# ---------------------------------------------------------------------------

class TestCerebrasAdapterGenerate:
    def test_generate_formats_messages(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("hi")
        adapter = _inject_sdk_into_adapter(mock_client)

        adapter.generate("Say hello")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]
        assert call_kwargs["model"] == adapter.model_name

    def test_generate_returns_generation_result(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("world", 10, 4)
        adapter = _inject_sdk_into_adapter(mock_client)

        result = adapter.generate("prompt")
        assert isinstance(result, GenerationResult)
        assert result.text == "world"
        assert result.tokens_used == 4
        assert result.finish_reason == "stop"
        assert result.model_name == adapter.model_name
        assert result.metadata is not None
        assert result.metadata["prompt_tokens"] == 10

    def test_generate_propagates_api_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        adapter = _inject_sdk_into_adapter(mock_client)

        with pytest.raises(RuntimeError, match="Cerebras generation failed"):
            adapter.generate("prompt")

    def test_generate_raises_if_not_loaded(self):
        adapter = CerebrasAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.generate("hello")

    def test_generate_passes_config_temperature(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response()
        adapter = _inject_sdk_into_adapter(mock_client)

        config = GenerationConfig(temperature=0.1, max_tokens=50)
        adapter.generate("test", config=config)
        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_completion_tokens"] == 50


# ---------------------------------------------------------------------------
# 4. generate_stream() yields chunks via mock SDK
# ---------------------------------------------------------------------------

def _make_stream_chunks(texts: list[str]) -> list[MagicMock]:
    """Build a list of fake SSE chunks for streaming."""
    chunks = []
    for t in texts:
        delta = MagicMock()
        delta.content = t
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = None
        chunks.append(chunk)
    # Terminal chunk with usage
    delta_end = MagicMock()
    delta_end.content = None
    choice_end = MagicMock()
    choice_end.delta = delta_end
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    end_chunk = MagicMock()
    end_chunk.choices = [choice_end]
    end_chunk.usage = usage
    chunks.append(end_chunk)
    return chunks


class TestCerebrasAdapterStream:
    def test_generate_stream_yields_chunks(self):
        mock_client = MagicMock()
        chunks = _make_stream_chunks(["Hello", " world", "!"])
        mock_client.chat.completions.create.return_value = iter(chunks)
        adapter = _inject_sdk_into_adapter(mock_client)

        result = list(adapter.generate_stream("test"))
        assert result == ["Hello", " world", "!"]

    def test_generate_stream_raises_if_not_loaded(self):
        adapter = CerebrasAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(adapter.generate_stream("test"))

    def test_generate_stream_passes_stream_true(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        adapter = _inject_sdk_into_adapter(mock_client)
        list(adapter.generate_stream("test"))
        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["stream"] is True

    def test_generate_stream_wraps_api_errors(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("network error")
        adapter = _inject_sdk_into_adapter(mock_client)
        with pytest.raises(RuntimeError, match="streaming failed"):
            list(adapter.generate_stream("test"))


# ---------------------------------------------------------------------------
# 5. count_tokens and get_context_length
# ---------------------------------------------------------------------------

class TestCerebrasAdapterTokens:
    def test_count_tokens_returns_token_count(self):
        adapter = CerebrasAdapter()
        tc = adapter.count_tokens("Hello world")
        assert isinstance(tc, TokenCount)
        assert tc.count > 0
        assert tc.model_name == adapter.model_name

    def test_get_context_length_gpt_oss(self):
        adapter = CerebrasAdapter(model_name="gpt-oss-120b")
        # Free-tier context for gpt-oss-120b is 65k (paid: 131k)
        assert adapter.get_context_length() == 65_536

    def test_get_max_output_zai_glm(self):
        adapter = CerebrasAdapter(model_name="zai-glm-4.7")
        assert adapter.get_max_output() == 40_960


# ---------------------------------------------------------------------------
# 6. unload()
# ---------------------------------------------------------------------------

class TestCerebrasAdapterUnload:
    def test_unload_clears_client(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "k")
        mock_client = MagicMock()
        adapter = _inject_sdk_into_adapter(mock_client)
        assert adapter.is_loaded()
        adapter.unload()
        assert not adapter.is_loaded()
        assert adapter._client is None


# ---------------------------------------------------------------------------
# 7. Model registry helpers
# ---------------------------------------------------------------------------

class TestCerebrasModelRegistry:
    def test_available_models_returns_live_two(self):
        from effgen.models.cerebras_models import available_models
        models = available_models()
        # Only the two models the live Cerebras API serves remain.
        assert set(models) == {"gpt-oss-120b", "zai-glm-4.7"}

    def test_free_tier_models_are_the_live_models(self):
        from effgen.models.cerebras_models import free_tier_models
        free = free_tier_models()
        assert "gpt-oss-120b" in free
        assert "zai-glm-4.7" in free

    def test_model_info_returns_limits(self):
        from effgen.models.cerebras_models import model_info
        info = model_info("gpt-oss-120b")
        assert info["rpm"] == 30
        assert info["tpm"] == 64_000
        assert info["context"] == 65_536

    def test_model_info_unknown_raises(self):
        from effgen.models.cerebras_models import model_info
        with pytest.raises(KeyError):
            model_info("nonexistent-model")

    def test_all_models_have_rate_limits(self):
        from effgen.models.cerebras_models import CEREBRAS_MODELS
        for mid, info in CEREBRAS_MODELS.items():
            for field in ("rpm", "rph", "rpd", "tpm", "tph", "tpd"):
                assert field in info, f"Model {mid!r} missing {field!r}"
                assert info[field] > 0, f"Model {mid!r}: {field} must be positive"

    def test_adapter_list_models(self):
        assert len(CerebrasAdapter.list_models()) == 2

    def test_adapter_get_model_info(self):
        info = CerebrasAdapter.get_model_info("gpt-oss-120b")
        # Free-tier context is 65k, paid-tier is 131k
        assert info["context"] == 65_536
        assert info["context_paid"] == 131_072

    def test_adapter_list_free_tier(self):
        free = CerebrasAdapter.list_free_tier_models()
        assert "gpt-oss-120b" in free
        assert "zai-glm-4.7" in free


# ---------------------------------------------------------------------------
# 8. Rate-limit coordinator wired into adapter
# ---------------------------------------------------------------------------

class TestCerebrasRateLimiter:
    def test_rate_limiter_created_by_default(self):
        adapter = CerebrasAdapter()
        assert adapter._rate_limiter is not None

    def test_rate_limiter_disabled(self):
        adapter = CerebrasAdapter(enable_rate_limiting=False)
        assert adapter._rate_limiter is None

    def test_rate_limiter_uses_model_limits(self):
        adapter = CerebrasAdapter(model_name="zai-glm-4.7")
        rl = adapter._rate_limiter
        assert rl is not None
        assert rl._req_minute.limit == 10   # zai-glm-4.7 has rpm=10
        assert rl._req_day.limit == 100      # rpd=100

    def test_rate_limit_status_returns_dict(self):
        adapter = CerebrasAdapter()
        status = adapter.rate_limit_status()
        assert "req_minute_limit" in status
        assert status["req_minute_limit"] == 30

    def test_rate_limit_status_empty_when_disabled(self):
        adapter = CerebrasAdapter(enable_rate_limiting=False)
        assert adapter.rate_limit_status() == {}


# ---------------------------------------------------------------------------
# 9. Native tool-calling unit tests
# ---------------------------------------------------------------------------

def _make_mock_response_with_tools(
    text: str = "",
    tool_name: str = "calculator",
    tool_args: dict | None = None,
    prompt_tokens: int = 20,
    completion_tokens: int = 10,
) -> MagicMock:
    """Build a mock response that contains a tool call."""
    import json

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    tc = MagicMock()
    tc.id = "call_abc123"
    tc.type = "function"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(tool_args or {"expression": "2+2"})

    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = [tc]
    choice.finish_reason = "tool_calls"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestCerebrasNativeTools:
    def test_supports_tool_calling_gpt_oss(self):
        adapter = CerebrasAdapter(model_name="gpt-oss-120b")
        assert adapter.supports_tool_calling() is True

    def test_supports_tool_calling_zai_glm_false(self):
        adapter = CerebrasAdapter(model_name="zai-glm-4.7")
        assert adapter.supports_tool_calling() is False

    def test_generate_with_tools_parses_tool_calls(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response_with_tools(
            tool_name="calculator", tool_args={"expression": "17*23"}
        )
        # Use gpt-oss-120b which has supports_native_tools=True
        adapter = CerebrasAdapter(model_name="gpt-oss-120b", enable_rate_limiting=False,
                                  enable_cost_tracking=False)
        adapter._client = mock_client
        adapter._is_loaded = True

        tools = [{"type": "function", "function": {"name": "calculator",
                  "description": "Eval math", "parameters": {}}}]
        result = adapter.generate_with_tools("17*23", tools=tools)
        assert result.metadata is not None
        tc_list = result.metadata.get("tool_calls", [])
        assert len(tc_list) == 1
        assert tc_list[0]["function"]["name"] == "calculator"
        # The reported element shape is pinned once, for every adapter, in
        # tests/unit/test_adapter_consistency.py.

    def test_generate_with_tools_raises_on_unsupported_model(self):
        adapter = CerebrasAdapter(model_name="zai-glm-4.7", enable_rate_limiting=False)
        adapter._client = MagicMock()
        adapter._is_loaded = True
        with pytest.raises(NotImplementedError, match="does not support native tool-calling"):
            adapter.generate_with_tools("hello", tools=[])


# ---------------------------------------------------------------------------
# 10. CostTracker unit tests
# ---------------------------------------------------------------------------

class TestCostTracker:
    def setup_method(self):
        from effgen.models._cost import CostTracker
        CostTracker.reset()

    def test_cerebras_cost_is_zero(self):
        from effgen.models._cost import CostTracker
        tracker = CostTracker.get()
        cost = tracker.record("cerebras", "gpt-oss-120b", 100, 50)
        assert cost == 0.0

    def test_accumulates_tokens(self):
        from effgen.models._cost import CostTracker
        tracker = CostTracker.get()
        tracker.record("cerebras", "gpt-oss-120b", 50, 20)
        tracker.record("cerebras", "gpt-oss-120b", 30, 10)
        totals = tracker.total_tokens("cerebras", "gpt-oss-120b")
        assert totals["prompt"] == 80
        assert totals["completion"] == 30
        assert totals["total"] == 110

    def test_total_cost_across_providers(self):
        from effgen.models._cost import CostTracker
        tracker = CostTracker.get()
        tracker.record("cerebras", "gpt-oss-120b", 100, 50)
        tracker.record("openai", "gpt-4o-mini", 1000, 500)
        cerebras_cost = tracker.total_cost("cerebras")
        openai_cost = tracker.total_cost("openai")
        assert cerebras_cost == 0.0
        assert openai_cost > 0.0

    def test_summary_returns_rows(self):
        from effgen.models._cost import CostTracker
        tracker = CostTracker.get()
        tracker.record("cerebras", "zai-glm-4.7", 100, 40)
        summary = tracker.summary()
        assert len(summary) == 1
        row = summary[0]
        assert row["provider"] == "cerebras"
        assert row["model"] == "zai-glm-4.7"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 40
        assert row["cost_usd"] == 0.0

    def test_cost_tracker_wired_into_adapter(self):
        from effgen.models._cost import CostTracker
        CostTracker.reset()
        tracker = CostTracker.get()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "hi", prompt_tokens=15, completion_tokens=8
        )
        adapter = CerebrasAdapter(enable_rate_limiting=False, enable_cost_tracking=True)
        adapter._client = mock_client
        adapter._is_loaded = True

        adapter.generate("test")
        totals = tracker.total_tokens("cerebras", "gpt-oss-120b")
        assert totals["prompt"] == 15
        assert totals["completion"] == 8
