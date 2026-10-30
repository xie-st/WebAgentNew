"""Tests for effgen.server.budget — per-principal daily cost caps."""
from __future__ import annotations

import pytest

from effgen.server.budget import (
    BudgetExceeded,
    charge,
    check_budget,
    get_spend,
    reset,
)


@pytest.fixture(autouse=True)
def _isolated_budget(tmp_path, monkeypatch):
    """Persist to a temp dir and clear the in-memory ledger per test."""
    monkeypatch.setenv("EFFGEN_BUDGET_DIR", str(tmp_path))
    # Force the module to re-resolve the dir.
    from effgen.server import budget as _b

    _b._BUDGET_DIR = None
    reset()
    yield
    reset()
    _b._BUDGET_DIR = None


class TestSpendAccounting:
    def test_initial_spend_is_zero(self):
        assert get_spend("alice") == 0.0

    def test_charge_accumulates(self):
        charge("alice", 0.01)
        charge("alice", 0.02)
        assert get_spend("alice") == pytest.approx(0.03)

    def test_charge_is_per_principal(self):
        charge("alice", 0.10)
        assert get_spend("bob") == 0.0

    def test_negative_amount_is_ignored(self):
        charge("alice", -5.0)
        assert get_spend("alice") == 0.0


class TestCostCap:
    def test_unlimited_cap_never_raises(self):
        for _ in range(100):
            charge("alice", 1.0, cap=0.0)  # 0 == unlimited
        check_budget("alice", 0.0)  # no raise

    def test_first_call_under_cap_succeeds_then_rejects(self):
        """A $0.01/day cap admits the first charge and rejects the next."""
        # First call: spend is 0 < cap → charge succeeds.
        charge("carol", 0.01, cap=0.01)
        assert get_spend("carol") == pytest.approx(0.01)
        # Second call: spend (0.01) >= cap (0.01) → BudgetExceeded.
        with pytest.raises(BudgetExceeded, match="BudgetExceeded"):
            charge("carol", 0.01, cap=0.01)

    def test_check_budget_raises_when_met(self):
        charge("dave", 5.0)
        with pytest.raises(BudgetExceeded):
            check_budget("dave", 5.0)

    def test_check_budget_ok_under_cap(self):
        charge("dave", 1.0)
        check_budget("dave", 5.0)  # no raise

    def test_budget_exceeded_status_is_429(self):
        exc = BudgetExceeded("nope")
        assert exc.status_code == 429


class TestPersistenceAndReset:
    def test_reset_clears_principal(self):
        charge("alice", 1.0)
        reset("alice")
        assert get_spend("alice") == 0.0

    def test_persistence_survives_in_memory_clear(self, monkeypatch):
        from effgen.server import budget as _b

        charge("erin", 0.42)
        # Drop the in-memory ledger but keep the on-disk snapshot.
        _b._ledger = {}
        assert get_spend("erin") == pytest.approx(0.42)

    def test_persist_disabled_does_not_write(self, monkeypatch, tmp_path):
        from effgen.server import budget as _b

        monkeypatch.setenv("EFFGEN_BUDGET_PERSIST", "0")
        charge("frank", 0.5)
        _b._ledger = {}
        # No persisted snapshot → spend resets.
        assert get_spend("frank") == 0.0
