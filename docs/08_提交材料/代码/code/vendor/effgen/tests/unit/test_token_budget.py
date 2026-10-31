"""Unit tests for effgen.memory.token_budget."""
from __future__ import annotations

import pytest

from effgen.memory.token_budget import (
    DEFAULT_SHARES,
    TokenBudget,
    fit_to_budget,
    smart_truncate,
)


def _word_tokens(s: str) -> int:
    return len(s.split())


class TestTokenBudget:
    def test_default_allocations_sum_to_context(self):
        b = TokenBudget(context_length=10_000)
        allocs = b.allocations()
        assert sum(allocs.values()) <= 10_000
        # default shares sum to 1.0 → integer truncation yields ≤ 10_000
        assert sum(allocs.values()) >= 9_996

    def test_default_shares_match_constant(self):
        b = TokenBudget(context_length=10_000)
        for k, v in DEFAULT_SHARES.items():
            assert b.shares[k] == pytest.approx(v)

    def test_invalid_context_length(self):
        with pytest.raises(ValueError):
            TokenBudget(context_length=0)
        with pytest.raises(ValueError):
            TokenBudget(context_length=-5)

    def test_shares_renormalize_when_not_unity(self):
        b = TokenBudget(context_length=1000, shares={"system": 2.0, "history": 2.0})
        assert sum(b.shares.values()) == pytest.approx(1.0)
        assert b.shares["system"] == pytest.approx(0.5)

    def test_invalid_shares(self):
        with pytest.raises(ValueError):
            TokenBudget(context_length=1000, shares={"x": 0.0, "y": 0.0})

    def test_allocate_unknown_section_returns_zero(self):
        b = TokenBudget(context_length=1000)
        assert b.allocate("does-not-exist") == 0

    def test_reserve_overrides_section(self):
        b = TokenBudget(context_length=1000)
        b.reserve("system", 500)
        assert b.allocate("system") == 500
        assert sum(b.shares.values()) == pytest.approx(1.0)

    def test_reserve_out_of_range(self):
        b = TokenBudget(context_length=1000)
        with pytest.raises(ValueError):
            b.reserve("system", -1)
        with pytest.raises(ValueError):
            b.reserve("system", 1001)


class TestSmartTruncate:
    def test_empty_input(self):
        assert smart_truncate([], 100, _word_tokens) == []

    def test_zero_budget(self):
        assert smart_truncate(["a", "b"], 0, _word_tokens) == []

    def test_within_budget_returns_all(self):
        items = ["one", "two three", "four"]
        out = smart_truncate(items, max_tokens=100, count_tokens=_word_tokens)
        assert out == items

    def test_truncates_middle_inserts_marker(self):
        items = ["sys " * 1, "old turn one", "old turn two", "old turn three", "recent a", "recent b"]
        out = smart_truncate(items, max_tokens=8, count_tokens=_word_tokens, keep_head=1, keep_tail=2)
        # head + marker + some middle (recent-most) + tail
        assert out[0] == items[0]
        assert "[... earlier turns summarized ...]" in out
        assert out[-1] == "recent b"
        assert out[-2] == "recent a"

    def test_marker_alone_when_no_middle_fits(self):
        items = ["s", "a a a a a a a a a a", "b b b b b b b b b b", "tail"]
        out = smart_truncate(items, max_tokens=4, count_tokens=_word_tokens, keep_head=1, keep_tail=1)
        assert out[0] == "s"
        assert out[-1] == "tail"
        assert "[... earlier turns summarized ...]" in out


class TestFitToBudget:
    def test_per_section_truncation(self):
        budget = TokenBudget(context_length=20)
        sections = {
            "system": ["sys word " * 1] * 1,
            "tools": ["tool description " * 5],
            "history": ["turn one", "turn two", "turn three", "turn four"],
            "response": [],
        }
        out = fit_to_budget(sections, budget, _word_tokens)
        # Each section yields a list (possibly empty)
        assert set(out.keys()) == {"system", "tools", "history", "response"}
        for v in out.values():
            assert isinstance(v, list)
