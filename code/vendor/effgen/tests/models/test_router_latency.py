"""Unit tests for LatencyBasedPolicy."""

from __future__ import annotations

import pytest

from effgen.models.capabilities import Capability
from effgen.models.errors import NoCandidateWithinLatencyError
from effgen.models.latency_tracker import LatencyTracker
from effgen.models.router import (
    PolicyBasedRouter,
    ProviderModelPair,
    RoutingContext,
)
from effgen.models.routing.latency import LatencyBasedPolicy


def _ctx(
    budget_ms: int | None = None,
    caps: set | None = None,
) -> RoutingContext:
    return RoutingContext(
        latency_budget_ms=budget_ms,
        required_capabilities=caps or {Capability.chat},
    )


def _pair(provider: str, model_id: str = "m") -> ProviderModelPair:
    return ProviderModelPair(provider=provider, model_id=model_id)


def _tracker_with(*entries: tuple[str, str, float]) -> LatencyTracker:
    """Create a fresh tracker pre-seeded with (provider, model, p50_ms) tuples."""
    t = LatencyTracker()
    for provider, model, ms in entries:
        t.record(provider, model, ms)
    return t


# ---------------------------------------------------------------------------
# Helpers to patch registry
# ---------------------------------------------------------------------------

class _MockRegistry:
    """Minimal registry stub for policy unit tests."""

    def __init__(self, providers: dict) -> None:
        self._providers = providers

    def get_provider_info(self, provider: str) -> dict:
        return self._providers.get(provider, {})

    def get_capabilities(self, provider: str) -> set:
        return self._providers.get(provider, {}).get("caps", set())


def _patch_registry(monkeypatch, providers: dict) -> None:
    """Patch ProviderRegistry module-level functions used by LatencyBasedPolicy."""
    import effgen.models.registry as reg_mod

    mock = _MockRegistry(providers)
    monkeypatch.setattr(reg_mod.ProviderRegistry, "get_provider_info", mock.get_provider_info)
    monkeypatch.setattr(reg_mod.ProviderRegistry, "get_capabilities", mock.get_capabilities)


# ---------------------------------------------------------------------------
# Tests — basic selection
# ---------------------------------------------------------------------------

def test_picks_fastest(monkeypatch):
    providers = {
        "fast": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "slow": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = _tracker_with(("fast", "m", 200.0), ("slow", "m", 800.0))
    policy = LatencyBasedPolicy(tracker=tracker)
    candidates = [_pair("slow"), _pair("fast")]
    decision = policy.select(candidates, _ctx())
    assert decision.chosen.provider == "fast"


def test_budget_eliminates_slow(monkeypatch):
    providers = {
        "fast": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "slow": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = _tracker_with(("fast", "m", 200.0), ("slow", "m", 3000.0))
    policy = LatencyBasedPolicy(tracker=tracker)
    candidates = [_pair("fast"), _pair("slow")]
    decision = policy.select(candidates, _ctx(budget_ms=1000))
    assert decision.chosen.provider == "fast"
    eliminated_providers = [p.provider for p, _ in decision.eliminated]
    assert "slow" in eliminated_providers


def test_no_candidates_raises(monkeypatch):
    providers = {
        "slow": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = _tracker_with(("slow", "m", 5000.0))
    policy = LatencyBasedPolicy(tracker=tracker)
    candidates = [_pair("slow")]
    with pytest.raises(NoCandidateWithinLatencyError):
        policy.select(candidates, _ctx(budget_ms=1000))


# ---------------------------------------------------------------------------
# Tests — seed / no data
# ---------------------------------------------------------------------------

def test_seed_used_for_unknown(monkeypatch):
    """Providers with no data get seed latency; policy still works."""
    providers = {
        "cerebras": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "groq": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = LatencyTracker()  # empty — no observations
    policy = LatencyBasedPolicy(tracker=tracker, enable_warmup=False)
    candidates = [_pair("cerebras"), _pair("groq")]
    # Both have seed latencies; groq seed (350) < cerebras seed (300)?
    # Actually cerebras=300, groq=350 → cerebras wins
    decision = policy.select(candidates, _ctx(budget_ms=5000))
    assert decision.chosen is not None


def test_custom_seed_latency(monkeypatch):
    """seed_latency_ms overrides provider defaults."""
    providers = {
        "a": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "b": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = LatencyTracker()  # no data
    # Force all seeds to 999ms → budget 1000ms should allow both
    policy = LatencyBasedPolicy(tracker=tracker, seed_latency_ms=999.0, enable_warmup=False)
    candidates = [_pair("a"), _pair("b")]
    decision = policy.select(candidates, _ctx(budget_ms=1000))
    assert decision.chosen is not None


def test_seed_too_slow_eliminates(monkeypatch):
    """When seed > budget, candidates with no data are eliminated."""
    providers = {
        "a": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = LatencyTracker()  # no data → seed for "a" = 1500ms (unknown)
    policy = LatencyBasedPolicy(tracker=tracker, seed_latency_ms=2000.0, enable_warmup=False)
    candidates = [_pair("a")]
    with pytest.raises(NoCandidateWithinLatencyError):
        policy.select(candidates, _ctx(budget_ms=500))


def test_observed_latency_beats_seed_guess(monkeypatch):
    providers = {
        "cerebras": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "groq": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = _tracker_with(("groq", "m", 400.0))
    policy = LatencyBasedPolicy(tracker=tracker, enable_warmup=False)
    candidates = [_pair("cerebras"), _pair("groq")]
    decision = policy.select(candidates, _ctx(budget_ms=500))
    assert decision.chosen.provider == "groq"
    assert any(
        pair.provider == "cerebras" and "no observed latency" in reason
        for pair, reason in decision.eliminated
    )


# ---------------------------------------------------------------------------
# Tests — API key and capability filtering
# ---------------------------------------------------------------------------

def test_missing_key_eliminated(monkeypatch):
    providers = {
        "nokey": {"caps": {Capability.chat}, "env_keys": ["NOKEY_API_KEY"], "models": {}},
        "haskey": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    monkeypatch.delenv("NOKEY_API_KEY", raising=False)
    tracker = _tracker_with(("nokey", "m", 100.0), ("haskey", "m", 500.0))
    policy = LatencyBasedPolicy(tracker=tracker)
    candidates = [_pair("nokey"), _pair("haskey")]
    decision = policy.select(candidates, _ctx())
    assert decision.chosen.provider == "haskey"


def test_missing_capability_eliminated(monkeypatch):
    providers = {
        "nocap": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "hascap": {"caps": {Capability.chat, Capability.tools}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = _tracker_with(("nocap", "m", 100.0), ("hascap", "m", 500.0))
    policy = LatencyBasedPolicy(tracker=tracker)
    candidates = [_pair("nocap"), _pair("hascap")]
    decision = policy.select(candidates, _ctx(caps={Capability.chat, Capability.tools}))
    assert decision.chosen.provider == "hascap"


# ---------------------------------------------------------------------------
# Tests — RouterDecision fields
# ---------------------------------------------------------------------------

def test_decision_fields(monkeypatch):
    providers = {
        "fast": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
        "slow": {"caps": {Capability.chat}, "env_keys": [], "models": {}},
    }
    _patch_registry(monkeypatch, providers)
    tracker = _tracker_with(("fast", "m", 150.0), ("slow", "m", 900.0))
    policy = LatencyBasedPolicy(tracker=tracker)
    candidates = [_pair("fast"), _pair("slow")]
    decision = policy.select(candidates, _ctx(budget_ms=2000))
    assert decision.policy_name == "latency_based"
    assert decision.score == 150.0  # p50 of chosen
    assert len(decision.eliminated) >= 1


# ---------------------------------------------------------------------------
# Integration: probe seeds tracker (minimal, no real API call)
# ---------------------------------------------------------------------------

def test_probe_cache_flag():
    from effgen.models.routing._probe import reset_probe_cache
    reset_probe_cache()
    # After reset, the flag is False — warm_up can run again
    import effgen.models.routing._probe as probe_mod
    assert not probe_mod._probe_done


def test_empty_tracker_router_triggers_warmup(monkeypatch):
    providers = {
        "groq": {"caps": {Capability.chat}, "env_keys": [], "models": {"m": {}}},
    }
    _patch_registry(monkeypatch, providers)
    import effgen.models.registry as reg_mod

    monkeypatch.setattr(reg_mod.ProviderRegistry, "list_providers", lambda: ["groq"])
    monkeypatch.setattr(
        reg_mod.ProviderRegistry,
        "list_models",
        lambda provider: [{"model_id": "m"}],
    )

    tracker = LatencyTracker()
    calls = []

    def fake_warm_up(context_caps, tracker, force=False, max_workers=8):
        calls.append((context_caps, force, max_workers))
        tracker.record("groq", "m", 123.0)

    monkeypatch.setattr("effgen.models.routing._probe.warm_up_providers", fake_warm_up)
    router = PolicyBasedRouter(policies=[LatencyBasedPolicy(tracker=tracker)])
    decision = router.route(_ctx(budget_ms=500))

    assert calls
    assert decision.chosen == ProviderModelPair("groq", "m")
    assert decision.score == 123.0


def test_latency_budget_error_does_not_fallback(monkeypatch):
    providers = {
        "slow": {"caps": {Capability.chat}, "env_keys": [], "models": {"m": {}}},
    }
    _patch_registry(monkeypatch, providers)
    import effgen.models.registry as reg_mod

    monkeypatch.setattr(reg_mod.ProviderRegistry, "list_providers", lambda: ["slow"])
    monkeypatch.setattr(
        reg_mod.ProviderRegistry,
        "list_models",
        lambda provider: [{"model_id": "m"}],
    )
    tracker = _tracker_with(("slow", "m", 5000.0))
    router = PolicyBasedRouter(policies=[LatencyBasedPolicy(tracker=tracker)])

    with pytest.raises(NoCandidateWithinLatencyError):
        router.route(_ctx(budget_ms=100))
