"""Integration tests for HFInferenceAdapter — skipped if HF_TOKEN absent, real calls otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)


# A stalled provider socket must surface as effGen's own typed timeout, not as a
# pytest-timeout kill that dumps every thread stack into the log. The adapter's
# default per-request timeout is 120s and pytest's cap is also 120s, so the two
# fire together and the kill wins; a shorter per-request timeout here lets the
# adapter report first. Observed against a throttled account whose last requests
# hung on an SSL read with no server-side close.
_LIVE_TIMEOUT_S = 45.0


def _has_key() -> bool:
    return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"))


def _skip_if_hf_credits_exhausted(exc: BaseException) -> None:
    msg = str(exc).lower()
    if (
        "402" in msg
        or "payment required" in msg
        or "credits exhausted" in msg
        or "exceeded your monthly included credits" in msg
        or "depleted your monthly included credits" in msg
    ):
        pytest.skip(f"HuggingFace Inference credits exhausted: {exc}")


def _skip_if_hf_model_rotated_out(exc: BaseException) -> None:
    # HF serverless rotates models in/out of the free tier; when the happy-path
    # model is not currently served the adapter raises a typed
    # ModelUnavailableError naming alternatives. That is provider-side state,
    # not an effGen regression, so the happy-path tests skip with the reason.
    # The typed-error tests below call the adapter directly (no helper) and
    # still assert the raise.
    msg = str(exc)
    if "model_not_supported" in msg or "not supported by any provider" in msg:
        pytest.skip(f"HuggingFace serverless does not currently serve this model: {exc}")
    from effgen.models.errors import ModelUnavailableError

    if isinstance(exc, ModelUnavailableError) and "temporarily unavailable" in msg:
        pytest.skip(f"HuggingFace serverless model rotated out: {exc}")


def _call_or_skip_hf_quota(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        _skip_if_hf_credits_exhausted(exc)
        _skip_if_hf_model_rotated_out(exc)
        raise


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: HF_TOKEN not set")
class TestHFInferenceLive:
    def test_generate_qwen25_7b(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct", timeout=_LIVE_TIMEOUT_S)
        adapter.load()
        try:
            result = _call_or_skip_hf_quota(
                adapter.generate,
                "Respond with exactly: HF_OK",
                config=None,
            )
            assert result.text, "Expected non-empty response"
            assert result.tokens_used > 0
            assert result.metadata["provider"] == "hf_inference"
        finally:
            adapter.unload()

    def test_generate_meta_llama_8b(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        # HF Router catalog uses the canonical "meta-llama/Llama-3.1-8B-Instruct"
        # path (not the older "Meta-Llama-3.1-8B-Instruct" alias).
        adapter = HFInferenceAdapter("meta-llama/Llama-3.1-8B-Instruct", timeout=_LIVE_TIMEOUT_S)
        adapter.load()
        try:
            result = _call_or_skip_hf_quota(
                adapter.generate,
                "What is 2 + 2? Answer with just the number.",
            )
            assert "4" in result.text
        finally:
            adapter.unload()

    def test_load_model_via_provider(self):
        from effgen.models import load_model

        model = load_model("Qwen/Qwen2.5-7B-Instruct", provider="hf_inference")
        try:
            result = _call_or_skip_hf_quota(model.generate, "Say hello in one word")
            assert result.text
        finally:
            model.unload()

    def test_generate_stream_yields_multiple_chunks(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct", timeout=_LIVE_TIMEOUT_S)
        adapter.load()
        try:
            chunks = _call_or_skip_hf_quota(
                lambda: list(adapter.generate_stream("Count 1 to 5, one per line."))
            )
            assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
            text = "".join(chunks)
            assert len(text) > 0
        finally:
            adapter.unload()

    def test_generate_qwen25_72b(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter("Qwen/Qwen2.5-72B-Instruct", timeout=_LIVE_TIMEOUT_S)
        adapter.load()
        try:
            result = _call_or_skip_hf_quota(
                adapter.generate,
                "What is the capital of Germany? One word answer.",
            )
            assert result.text
            assert "Berlin" in result.text
        finally:
            adapter.unload()

    def test_native_tools_qwen(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct", timeout=_LIVE_TIMEOUT_S)
        adapter.load()
        try:
            tool = {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                },
            }
            result = _call_or_skip_hf_quota(
                adapter.generate_with_tools,
                "What is the weather in Paris?",
                tools=[tool],
            )
            assert result.metadata["tool_calls"], "Expected tool calls in response"
            tc = result.metadata["tool_calls"][0]
            assert tc["function"]["name"] == "get_weather"
            assert isinstance(tc["function"]["arguments"], str)
        finally:
            adapter.unload()

    def test_unavailable_model_raises_helpful_error(self):
        from effgen.models.errors import (
            ModelNotFoundError,
            ModelUnavailableError,
        )
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        # Use a model ID that has never been on serverless (made up)
        adapter = HFInferenceAdapter(
            "org/this-model-does-not-exist-abc123", warn_unknown_model=False
        )
        adapter.load()
        try:
            with pytest.raises(
                (ModelUnavailableError, ModelNotFoundError, RuntimeError)
            ):
                adapter.generate("hello")
        finally:
            adapter.unload()

    def test_unsupported_model_raises_typed_error(self):
        # Hub model id that has no live HF Router backend.  Adapter should
        # surface this as ModelUnavailableError or ModelNotFoundError with
        # alternatives, not a raw RuntimeError.
        from effgen.models.errors import (
            ModelNotFoundError,
            ModelUnavailableError,
        )
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter(
            "this-org-does-not-exist/totally-fake-model-zzz",
            warn_unknown_model=False,
        )
        adapter.load()
        try:
            with pytest.raises((ModelUnavailableError, ModelNotFoundError)):
                adapter.generate("hi")
        finally:
            adapter.unload()

    def test_generate_llama_3_3_70b(self):
        from effgen.models.hf_inference_adapter import HFInferenceAdapter

        adapter = HFInferenceAdapter("meta-llama/Llama-3.3-70B-Instruct", timeout=_LIVE_TIMEOUT_S)
        adapter.load()
        try:
            try:
                result = adapter.generate("Capital of France? One word.")
            except RuntimeError as exc:
                _skip_if_hf_credits_exhausted(exc)
                raise
            assert "Paris" in result.text
        finally:
            adapter.unload()
