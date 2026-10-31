"""Every engine's raw ``generate``/``generate_batch`` result carries call latency.

``BaseModel.__init_subclass__`` auto-instruments concrete engines so a benchmarker
calling ``model.generate(...)`` directly can read ``latency_ms``/``duration_s`` off
``GenerationResult.metadata`` without a manual timer — mirroring what the agent layer
already surfaces. These tests use a tiny in-process engine (no network/model) to
exercise the wrapping mechanism itself.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from effgen.models.base import (
    BaseModel,
    GenerationConfig,
    GenerationResult,
    ModelType,
    TokenCount,
)


class _TinyEngine(BaseModel):
    """Minimal concrete engine that just echoes — to test latency instrumentation."""

    def __init__(self):
        super().__init__("tiny", ModelType.TRANSFORMERS, context_length=128)
        self._is_loaded = True

    def load(self) -> None:  # pragma: no cover - trivial
        self._is_loaded = True

    def unload(self) -> None:  # pragma: no cover - trivial
        self._is_loaded = False

    def get_context_length(self) -> int:  # pragma: no cover - trivial
        return self._context_length or 128

    def generate(self, prompt: str, config: GenerationConfig | None = None, **kwargs):
        time.sleep(0.005)
        return GenerationResult(
            text=f"echo:{prompt}", tokens_used=1, finish_reason="stop",
            model_name=self.model_name, metadata={"prompt_tokens": 1},
        )

    def generate_stream(self, prompt, config=None, **kwargs) -> Iterator[str]:
        yield prompt

    def generate_batch(self, prompts, config=None, **kwargs):
        time.sleep(0.005)
        return [
            GenerationResult(text=f"echo:{p}", tokens_used=1, finish_reason="stop",
                             model_name=self.model_name, metadata={})
            for p in prompts
        ]

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover - trivial
        return TokenCount(count=len(text.split()), model_name=self.model_name)


class _SelfTimedEngine(_TinyEngine):
    """An engine that already records its own latency — must not be overwritten."""

    def generate(self, prompt, config=None, **kwargs):
        return GenerationResult(
            text="x", tokens_used=1, finish_reason="stop", model_name=self.model_name,
            metadata={"latency_ms": 1.0, "duration_s": 0.001},
        )


class _LengthTruncatedEngine(_TinyEngine):
    """An engine whose output was cut off at the token budget."""

    def generate(self, prompt, config=None, **kwargs):
        return GenerationResult(
            text="", tokens_used=16, finish_reason="length", model_name=self.model_name,
            metadata={},
        )


class _PricedEngine(_TinyEngine):
    """An engine that reports a per-call ``cost_usd`` but no cumulative total —
    mirrors adapters (Groq, Cerebras, Together, Fireworks, Replicate) that price
    each call without tracking a running total themselves."""

    def __init__(self, cost_usd: float):
        super().__init__()
        self._cost_usd = cost_usd

    def generate(self, prompt, config=None, **kwargs):
        return GenerationResult(
            text="x", tokens_used=1, finish_reason="stop", model_name=self.model_name,
            metadata={"cost_usd": self._cost_usd},
        )


class _SelfTrackedCostEngine(_TinyEngine):
    """An engine that already tracks its own cumulative ``total_cost`` (like the
    OpenAI/Gemini/Anthropic adapters) — the generic accumulator must not touch it."""

    def __init__(self):
        super().__init__()
        self.total_cost = 0.0

    def generate(self, prompt, config=None, **kwargs):
        cost = 0.01
        self.total_cost += cost
        return GenerationResult(
            text="x", tokens_used=1, finish_reason="stop", model_name=self.model_name,
            metadata={"cost_usd": cost, "total_cost": self.total_cost},
        )


class _ReasoningEngine(_TinyEngine):
    """An engine named like a reasoning model that can return empty text when
    ``finish_reason == "length"`` — mimics gpt-5*/o-series exhausting the token
    budget on hidden reasoning before any visible token."""

    def __init__(self, *, empty: bool):
        super().__init__()
        self.model_name = "gpt-5-nano"
        self._empty = empty

    def generate(self, prompt, config=None, **kwargs):
        if self._empty:
            return GenerationResult(
                text="", tokens_used=500, finish_reason="length",
                model_name=self.model_name, metadata={},
            )
        return GenerationResult(
            text="hi", tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )


def test_generate_stamps_latency():
    r = _TinyEngine().generate("hi")
    assert "latency_ms" in r.metadata and "duration_s" in r.metadata
    assert r.metadata["latency_ms"] >= 0
    assert r.metadata["duration_s"] >= 0
    # original metadata is preserved alongside the new keys.
    assert r.metadata["prompt_tokens"] == 1


def test_generate_stamps_truncated_flag():
    # A "stop"-finished result is not truncated.
    r = _TinyEngine().generate("hi")
    assert r.metadata["truncated"] is False

    # A "length"-finished result (budget exhausted) is truncated — the raw
    # GenerationResult carries the same signal the Agent path derives from
    # finish_reason, so a low-level caller doesn't have to string-match it.
    r2 = _LengthTruncatedEngine().generate("hi")
    assert r2.metadata["truncated"] is True


def test_generate_batch_stamps_latency_on_every_item():
    results = _TinyEngine().generate_batch(["a", "b", "c"])
    assert len(results) == 3
    for r in results:
        assert "latency_ms" in r.metadata and "duration_s" in r.metadata


def test_engine_self_timing_is_not_overwritten():
    # setdefault semantics: an engine that measures its own latency keeps it.
    r = _SelfTimedEngine().generate("hi")
    assert r.metadata["latency_ms"] == 1.0
    assert r.metadata["duration_s"] == 0.001


def test_wrapping_is_idempotent_across_subclassing():
    # _SelfTimedEngine subclasses _TinyEngine; neither generate should be double
    # wrapped (the marker prevents re-wrapping an already-timed method).
    assert getattr(_TinyEngine.generate, "__effgen_timed__", False) is True
    assert getattr(_SelfTimedEngine.generate, "__effgen_timed__", False) is True


# --------------------------------------------------------------------------- #
# cumulative total_cost, extended to every adapter (not just OpenAI/Gemini/
# Anthropic, which already tracked it themselves)
# --------------------------------------------------------------------------- #


def test_generate_accumulates_total_cost_across_calls():
    engine = _PricedEngine(cost_usd=0.001)
    r1 = engine.generate("a")
    r2 = engine.generate("b")
    assert r1.metadata["total_cost"] == 0.001
    assert r2.metadata["total_cost"] == pytest.approx(0.002)
    assert engine.get_total_cost() == pytest.approx(0.002)


def test_reset_cost_zeroes_the_running_total():
    engine = _PricedEngine(cost_usd=0.001)
    engine.generate("a")
    engine.reset_cost()
    assert engine.get_total_cost() == 0.0
    r = engine.generate("b")
    assert r.metadata["total_cost"] == pytest.approx(0.001)


def test_no_total_cost_when_engine_reports_no_cost():
    # A local/unpriced engine has no cost_usd, so no fabricated total_cost key.
    r = _TinyEngine().generate("hi")
    assert "cost_usd" not in r.metadata
    assert "total_cost" not in r.metadata


def test_self_tracked_total_cost_is_not_double_counted():
    # An adapter that already stamps its own cumulative total_cost (OpenAI,
    # Gemini, Anthropic) keeps exactly its own number — the generic accumulator
    # skips a result that already carries the key.
    engine = _SelfTrackedCostEngine()
    r1 = engine.generate("a")
    r2 = engine.generate("b")
    assert r1.metadata["total_cost"] == pytest.approx(0.01)
    assert r2.metadata["total_cost"] == pytest.approx(0.02)


def test_generate_batch_accumulates_total_cost_per_item():
    class _PricedBatchEngine(_PricedEngine):
        def generate_batch(self, prompts, config=None, **kwargs):
            return [self.generate(p) for p in prompts]

    engine = _PricedBatchEngine(cost_usd=0.001)
    results = engine.generate_batch(["a", "b", "c"])
    assert [r.metadata["total_cost"] for r in results] == [
        pytest.approx(0.001), pytest.approx(0.002), pytest.approx(0.003),
    ]


# --------------------------------------------------------------------------- #
# a silent-empty raw generate() on a reasoning model logs a visible warning
# --------------------------------------------------------------------------- #


def test_reasoning_model_empty_truncated_result_logs_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="effgen.models.base"):
        r = _ReasoningEngine(empty=True).generate("say hi")
    assert r.text == ""
    assert any(
        "exhausting its token budget on internal reasoning" in rec.message
        for rec in caplog.records
    )


def test_reasoning_model_non_empty_result_does_not_warn(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="effgen.models.base"):
        r = _ReasoningEngine(empty=False).generate("say hi")
    assert r.text == "hi"
    assert not any(
        "exhausting its token budget" in rec.message for rec in caplog.records
    )


def test_non_reasoning_model_empty_truncated_result_does_not_warn(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="effgen.models.base"):
        r = _LengthTruncatedEngine().generate("hi")
    assert r.text == ""
    assert not any(
        "exhausting its token budget" in rec.message for rec in caplog.records
    )
