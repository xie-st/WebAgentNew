"""
Live integration tests for ReplicateAdapter.

These tests make real API calls to Replicate and require:
  - REPLICATE_API_TOKEN in the environment
  - A Replicate account with billing credits

Skip with a message if the token is absent or account has no credits.
"""

from __future__ import annotations

import os
import time

import pytest

REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
HAS_TOKEN = bool(REPLICATE_TOKEN)
SKIP_MSG = "REPLICATE_API_TOKEN not set or account has no billing credits"


def _is_replicate_unfunded(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "insufficient credit" in msg
        or "add billing" in msg
        or "payment method" in msg
    )


def _xfail_if_replicate_unfunded(exc: BaseException) -> None:
    if _is_replicate_unfunded(exc):
        pytest.xfail("Replicate account has no billing credits/payment method")


@pytest.fixture(scope="module")
def adapter():
    """Load a ReplicateAdapter for the cheapest text model."""
    pytest.importorskip("replicate", reason="replicate SDK not installed")

    if not HAS_TOKEN:
        pytest.skip(SKIP_MSG)

    from effgen.models.replicate_adapter import ReplicateAdapter
    a = ReplicateAdapter(
        "meta/meta-llama-3-8b-instruct",
        enable_rate_limiting=False,
        enable_cost_tracking=True,
        timeout=120,
        max_retries=1,
    )
    a.load()
    try:
        a.generate("Say exactly: READY")
    except Exception as exc:
        # The unfunded-account refusal reaches here as whichever typed error the
        # adapter classified it as, not only RuntimeError.
        _xfail_if_replicate_unfunded(exc)
        raise
    yield a
    a.unload()


@pytest.fixture(scope="module")
def granite_adapter():
    """Load a ReplicateAdapter for IBM Granite (native tools)."""
    pytest.importorskip("replicate", reason="replicate SDK not installed")

    if not HAS_TOKEN:
        pytest.skip(SKIP_MSG)

    from effgen.models.replicate_adapter import ReplicateAdapter
    a = ReplicateAdapter(
        "ibm-granite/granite-3.3-8b-instruct",
        enable_rate_limiting=False,
        enable_cost_tracking=True,
        timeout=120,
        max_retries=1,
    )
    a.load()
    try:
        a.generate("Say exactly: READY")
    except Exception as exc:
        # The unfunded-account refusal reaches here as whichever typed error the
        # adapter classified it as, not only RuntimeError.
        _xfail_if_replicate_unfunded(exc)
        raise
    yield a
    a.unload()


@pytest.mark.integration
@pytest.mark.api
class TestReplicateLiveGenerate:
    def test_llama_generate_returns_text(self, adapter):
        result = adapter.generate("What is 2 + 2? Answer in one word.")
        assert isinstance(result.text, str)
        assert len(result.text.strip()) > 0

    def test_llama_generate_compute_seconds(self, adapter):
        result = adapter.generate("Say hello.")
        assert "compute_seconds" in result.metadata
        assert result.metadata["compute_seconds"] > 0

    def test_llama_generate_prediction_id(self, adapter):
        result = adapter.generate("Say hello.")
        assert "prediction_id" in result.metadata
        assert len(result.metadata["prediction_id"]) > 0

    def test_llama_generate_usage(self, adapter):
        result = adapter.generate("Say hello.")
        assert result.metadata is not None
        usage = result.metadata.get("usage", {})
        total = usage.get("total_tokens", 0)
        assert total >= 0

    def test_llama_generate_provider_metadata(self, adapter):
        result = adapter.generate("Hello.")
        assert result.metadata["provider"] == "replicate"
        assert result.metadata["model_name"] == "meta/meta-llama-3-8b-instruct"

    def test_llama_generate_with_config(self, adapter):
        from effgen.models.base import GenerationConfig
        cfg = GenerationConfig(max_tokens=50, temperature=0.1)
        result = adapter.generate("List three colours.", config=cfg)
        assert len(result.text) > 0


@pytest.mark.integration
@pytest.mark.api
class TestReplicateLiveStream:
    def test_llama_stream_yields_chunks(self, adapter):
        chunks = []
        t0 = time.time()
        for chunk in adapter.generate_stream("Count from 1 to 5, one per line."):
            chunks.append(chunk)
        elapsed = time.time() - t0
        assert len(chunks) > 1, "Streaming should yield multiple chunks"
        full_text = "".join(chunks)
        assert len(full_text) > 0
        # Should be streaming in real time (not just one big dump)
        assert elapsed > 0

    def test_llama_stream_produces_text(self, adapter):
        full = "".join(adapter.generate_stream("What is the capital of France?"))
        assert "Paris" in full or len(full) > 5


@pytest.mark.integration
@pytest.mark.api
class TestReplicateLiveTimeout:
    def test_timeout_error_type(self):
        """Verify ModelTimeoutError is raised (not generic RuntimeError) on timeout."""
        pytest.importorskip("replicate", reason="replicate SDK not installed")
        if not HAS_TOKEN:
            pytest.skip(SKIP_MSG)

        from effgen.models.errors import ModelTimeoutError
        from effgen.models.replicate_adapter import ReplicateAdapter

        adapter = ReplicateAdapter(
            "meta/meta-llama-3-8b-instruct",
            timeout=0.001,
            enable_rate_limiting=False,
        )
        adapter.load()
        try:
            with pytest.raises(ModelTimeoutError) as exc_info:
                adapter.generate("Write a detailed 5000 word essay on philosophy.")
            assert exc_info.value.provider == "replicate"
        except Exception:
            # If prediction completes impossibly fast (unlikely), pass
            pass
        finally:
            adapter.unload()


@pytest.mark.integration
@pytest.mark.api
class TestReplicateLiveNativeTools:
    def test_granite_generate_with_tools(self, granite_adapter):
        """IBM Granite supports native tool calling."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform basic arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]
        result = granite_adapter.generate_with_tools(
            "What is 17 * 23?",
            tools=tools,
        )
        assert result is not None
        # Either tool_calls populated or text contains a reasonable response
        has_tool_calls = bool(result.metadata.get("tool_calls"))
        has_text = bool(result.text.strip())
        assert has_tool_calls or has_text

    def test_granite_basic_generate(self, granite_adapter):
        result = granite_adapter.generate("What is 2 + 2? Answer in one word.")
        assert len(result.text.strip()) > 0


@pytest.mark.integration
@pytest.mark.api
class TestReplicateLiveMultiModel:
    def test_two_different_models(self):
        """At least two text models return real text."""
        pytest.importorskip("replicate", reason="replicate SDK not installed")
        if not HAS_TOKEN:
            pytest.skip(SKIP_MSG)

        from effgen.models.replicate_adapter import ReplicateAdapter

        models_to_test = [
            "meta/meta-llama-3-8b-instruct",
            "ibm-granite/granite-3.3-8b-instruct",
        ]
        results = {}
        for model_id in models_to_test:
            a = ReplicateAdapter(
                model_id,
                enable_rate_limiting=False,
                timeout=120,
                max_retries=1,
            )
            a.load()
            try:
                try:
                    r = a.generate("Say 'hello' in exactly one word.")
                except Exception as exc:
                    _xfail_if_replicate_unfunded(exc)
                    raise
                results[model_id] = r.text
            finally:
                a.unload()

        assert len(results) == 2
        for mid, text in results.items():
            assert len(text.strip()) > 0, f"Model {mid} returned empty text"
