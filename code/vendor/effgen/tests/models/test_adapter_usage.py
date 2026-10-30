"""Tests for the shared adapter usage/cost plumbing in effgen.models._usage
and the model-not-found error builder in effgen.models._adapter_utils."""

from types import SimpleNamespace

import pytest

from effgen.models._adapter_utils import model_not_found_error
from effgen.models._usage import (
    extract_openai_usage,
    record_tracker_cost,
    tool_calls_from_message,
    usage_metadata,
)
from effgen.models.errors import BudgetExceededError, ModelNotFoundError


def _usage(prompt=10, completion=5, total=15, details=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=details,
    )


class TestExtractOpenAIUsage:
    def test_basic_counts(self):
        assert extract_openai_usage(_usage()) == (10, 5, 15, 0)

    def test_cached_tokens_from_details(self):
        u = _usage(details=SimpleNamespace(cached_tokens=7))
        assert extract_openai_usage(u) == (10, 5, 15, 7)

    def test_details_none(self):
        assert extract_openai_usage(_usage(details=None))[3] == 0

    def test_details_absent(self):
        u = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        assert extract_openai_usage(u) == (1, 2, 3, 0)

    def test_cached_tokens_none_coerced_to_zero(self):
        u = _usage(details=SimpleNamespace(cached_tokens=None))
        assert extract_openai_usage(u)[3] == 0


class TestUsageMetadata:
    def test_canonical_keys_and_values(self):
        md = usage_metadata(10, 5, 15, 2, 0.001, 0.003)
        assert md == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cached_input_tokens": 2,
            "cost_usd": 0.001,
            "total_cost": 0.003,
        }


class TestToolCallsFromMessage:
    def test_empty_when_no_tool_calls(self):
        assert tool_calls_from_message(SimpleNamespace(tool_calls=None)) == []
        assert tool_calls_from_message(SimpleNamespace(tool_calls=[])) == []

    def test_converts_to_plain_dicts(self):
        tc = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="calculator", arguments='{"expression": "6*7"}'),
        )
        out = tool_calls_from_message(SimpleNamespace(tool_calls=[tc]))
        assert out == [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'},
        }]


class TestRecordTrackerCost:
    def test_returns_tracker_cost(self, monkeypatch):
        import logging

        from effgen.models import _cost

        class _Tracker:
            def record(self, **kwargs):
                assert kwargs["provider"] == "openai"
                return 0.00042

        monkeypatch.setattr(_cost.CostTracker, "get", classmethod(lambda cls: _Tracker()))
        cost = record_tracker_cost("openai", "gpt-5-nano", 10, 5, log=logging.getLogger(__name__))
        assert cost == 0.00042

    def test_budget_exceeded_propagates(self, monkeypatch):
        import logging

        from effgen.models import _cost

        class _Tracker:
            def record(self, **kwargs):
                raise BudgetExceededError(1.0, 2.5, provider="gemini")

        monkeypatch.setattr(_cost.CostTracker, "get", classmethod(lambda cls: _Tracker()))
        with pytest.raises(BudgetExceededError):
            record_tracker_cost("gemini", "gemini-3.1-flash-lite", 10, 5,
                                log=logging.getLogger(__name__))

    def test_tracker_failure_returns_none(self, monkeypatch):
        import logging

        from effgen.models import _cost

        class _Tracker:
            def record(self, **kwargs):
                raise OSError("store unavailable")

        monkeypatch.setattr(_cost.CostTracker, "get", classmethod(lambda cls: _Tracker()))
        assert record_tracker_cost("openai", "gpt-5-nano", 10, 5,
                                   log=logging.getLogger(__name__)) is None


class TestModelNotFoundErrorBuilder:
    def test_builds_typed_error_with_context(self):
        err = model_not_found_error("openai", "gpt-nonexistent-z9", "404 no such model")
        assert isinstance(err, ModelNotFoundError)
        assert err.provider == "openai"
        assert err.model_name == "gpt-nonexistent-z9"
        assert "404 no such model" in str(err)
        assert err.error_context["category"] == "not_found"
