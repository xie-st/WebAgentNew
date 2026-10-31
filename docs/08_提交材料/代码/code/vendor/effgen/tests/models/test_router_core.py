"""Unit tests for the policy-based ModelRouter (v0.2.4)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from effgen.models.capabilities import Capability
from effgen.models.registry import ProviderRegistry
from effgen.models.router import (
    ModelRouter,
    NoCandidateError,
    PolicyBasedRouter,
    ProviderModelPair,
    RouterDecision,
    RoutingContext,
)
from effgen.models.routing.first_available import FirstAvailablePolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(*providers: str) -> list[ProviderModelPair]:
    return [ProviderModelPair(p, f"{p}-model-001") for p in providers]


class _AlwaysFailPolicy:
    name = "always_fail"

    def select(self, candidates, context):
        raise NoCandidateError("deliberate failure")


# ---------------------------------------------------------------------------
# ProviderModelPair
# ---------------------------------------------------------------------------

def test_provider_model_pair_is_namedtuple():
    pair = ProviderModelPair("groq", "llama-3.3-70b")
    assert pair.provider == "groq"
    assert pair.model_id == "llama-3.3-70b"


# ---------------------------------------------------------------------------
# FirstAvailablePolicy — key checking
# ---------------------------------------------------------------------------

def test_first_available_skips_missing_key(monkeypatch):
    """Providers without a configured key must be eliminated."""
    ProviderRegistry.clear()
    from effgen.models.capabilities import Capability as Cap

    class _DummyAdapter:
        pass

    ProviderRegistry.register(
        "provA", _DummyAdapter, {"m1": {}},
        env_keys=["PROV_A_KEY"],
        capabilities={Cap.chat},
    )
    ProviderRegistry.register(
        "provB", _DummyAdapter, {"m2": {}},
        env_keys=["PROV_B_KEY"],
        capabilities={Cap.chat},
    )

    # Only provB has a key
    monkeypatch.setenv("PROV_B_KEY", "secret")
    monkeypatch.delenv("PROV_A_KEY", raising=False)

    policy = FirstAvailablePolicy()
    ctx = RoutingContext(required_capabilities={Cap.chat})
    candidates = _make_candidates("provA", "provB")
    decision = policy.select(candidates, ctx)

    assert decision.chosen.provider == "provB"
    # provA must appear in eliminated
    eliminated_providers = [pair.provider for pair, _ in decision.eliminated]
    assert "provA" in eliminated_providers

    # Restore global registry state
    ProviderRegistry.reset()


def test_first_available_all_missing_keys_raises(monkeypatch):
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    ProviderRegistry.register(
        "provX", _DummyAdapter, {"mx": {}},
        env_keys=["PROV_X_KEY"],
        capabilities={Cap.chat},
    )
    monkeypatch.delenv("PROV_X_KEY", raising=False)

    policy = FirstAvailablePolicy()
    ctx = RoutingContext(required_capabilities={Cap.chat})
    candidates = _make_candidates("provX")

    with pytest.raises(NoCandidateError):
        policy.select(candidates, ctx)

    ProviderRegistry.reset()


# ---------------------------------------------------------------------------
# FirstAvailablePolicy — capability filtering
# ---------------------------------------------------------------------------

def test_capability_filtering_eliminates_unsupported(monkeypatch):
    """Provider missing a required capability must be eliminated."""
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    # provA supports only chat, provB supports chat+vision
    ProviderRegistry.register(
        "provA", _DummyAdapter, {"ma": {}},
        env_keys=["PROV_A_KEY"],
        capabilities={Cap.chat},
    )
    ProviderRegistry.register(
        "provB", _DummyAdapter, {"mb": {}},
        env_keys=["PROV_B_KEY"],
        capabilities={Cap.chat, Cap.vision},
    )
    monkeypatch.setenv("PROV_A_KEY", "akey")
    monkeypatch.setenv("PROV_B_KEY", "bkey")

    policy = FirstAvailablePolicy()
    ctx = RoutingContext(required_capabilities={Cap.chat, Cap.vision})
    candidates = _make_candidates("provA", "provB")
    decision = policy.select(candidates, ctx)

    assert decision.chosen.provider == "provB"
    eliminated_providers = [pair.provider for pair, _ in decision.eliminated]
    assert "provA" in eliminated_providers
    # Check reason mentions missing capability
    for pair, reason in decision.eliminated:
        if pair.provider == "provA":
            assert "vision" in reason

    ProviderRegistry.reset()


# ---------------------------------------------------------------------------
# RouterDecision structure
# ---------------------------------------------------------------------------

def test_router_decision_records_eliminations(monkeypatch):
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    ProviderRegistry.register(
        "no_key_prov", _DummyAdapter, {"nk": {}},
        env_keys=["NO_KEY_ENV_VAR"],
        capabilities={Cap.chat},
    )
    ProviderRegistry.register(
        "has_key_prov", _DummyAdapter, {"hk": {}},
        env_keys=["HAS_KEY_ENV_VAR"],
        capabilities={Cap.chat},
    )
    monkeypatch.delenv("NO_KEY_ENV_VAR", raising=False)
    monkeypatch.setenv("HAS_KEY_ENV_VAR", "present")

    policy = FirstAvailablePolicy()
    ctx = RoutingContext(required_capabilities={Cap.chat})
    candidates = _make_candidates("no_key_prov", "has_key_prov")
    decision = policy.select(candidates, ctx)

    assert isinstance(decision, RouterDecision)
    assert decision.policy_name == "first_available"
    assert decision.score == 1.0
    assert len(decision.eliminated) >= 1
    # Elimination reasons should be non-empty strings
    for _, reason in decision.eliminated:
        assert isinstance(reason, str) and reason

    ProviderRegistry.reset()


def test_router_decision_records_every_non_chosen_candidate(monkeypatch):
    """FirstAvailablePolicy should explain every candidate it does not choose."""
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    ProviderRegistry.register(
        "chosen_prov",
        _DummyAdapter,
        {"c1": {}, "c2": {}},
        env_keys=["CHOSEN_KEY"],
        capabilities={Cap.chat, Cap.tools},
    )
    ProviderRegistry.register(
        "later_valid",
        _DummyAdapter,
        {"lv": {}},
        env_keys=["LATER_VALID_KEY"],
        capabilities={Cap.chat, Cap.tools},
    )
    ProviderRegistry.register(
        "later_no_cap",
        _DummyAdapter,
        {"lnc": {}},
        env_keys=["LATER_NO_CAP_KEY"],
        capabilities={Cap.chat},
    )
    monkeypatch.setenv("CHOSEN_KEY", "present")
    monkeypatch.setenv("LATER_VALID_KEY", "present")
    monkeypatch.setenv("LATER_NO_CAP_KEY", "present")

    candidates = [
        ProviderModelPair("chosen_prov", "c1"),
        ProviderModelPair("chosen_prov", "c2"),
        ProviderModelPair("later_valid", "lv"),
        ProviderModelPair("later_no_cap", "lnc"),
    ]
    decision = FirstAvailablePolicy().select(
        candidates,
        RoutingContext(required_capabilities={Cap.chat, Cap.tools}),
    )

    assert decision.chosen == ProviderModelPair("chosen_prov", "c1")
    assert len(decision.eliminated) == len(candidates) - 1
    reasons = dict(decision.eliminated)
    assert "selected chosen_prov/c1" in reasons[ProviderModelPair("chosen_prov", "c2")]
    assert "not selected: first available was chosen_prov/c1" in reasons[
        ProviderModelPair("later_valid", "lv")
    ]
    assert "tools" in reasons[ProviderModelPair("later_no_cap", "lnc")]

    ProviderRegistry.reset()


# ---------------------------------------------------------------------------
# PolicyBasedRouter
# ---------------------------------------------------------------------------

def test_policy_based_router_uses_fallback_when_policies_fail(monkeypatch):
    """When all policies fail, fallback is used."""
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    ProviderRegistry.register(
        "fallback_prov", _DummyAdapter, {"fp": {}},
        env_keys=["FALLBACK_KEY"],
        capabilities={Cap.chat},
    )
    monkeypatch.setenv("FALLBACK_KEY", "fk")

    fallback = FirstAvailablePolicy()
    router = PolicyBasedRouter(policies=[_AlwaysFailPolicy()], fallback=fallback)
    ctx = RoutingContext(required_capabilities={Cap.chat})

    with patch.object(router, "_get_candidates", return_value=_make_candidates("fallback_prov")):
        decision = router.route(ctx)

    assert decision.chosen.provider == "fallback_prov"

    ProviderRegistry.reset()
    import effgen.models.hf_inference_adapter  # noqa: F401
    import effgen.models.replicate_adapter  # noqa: F401


def test_policy_based_router_raises_when_all_fail(monkeypatch):
    """NoCandidateError raised when policies AND fallback both fail."""
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    ProviderRegistry.register(
        "no_key", _DummyAdapter, {"nk": {}},
        env_keys=["MISSING_KEY_XYZ"],
        capabilities={Cap.chat},
    )
    monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)

    router = PolicyBasedRouter(
        policies=[_AlwaysFailPolicy()],
        fallback=FirstAvailablePolicy(),
    )
    ctx = RoutingContext(required_capabilities={Cap.chat})

    with patch.object(router, "_get_candidates", return_value=_make_candidates("no_key")):
        with pytest.raises(NoCandidateError):
            router.route(ctx)

    ProviderRegistry.reset()


def test_model_router_policy_constructor_routes(monkeypatch):
    """ModelRouter must expose the policy-based routing surface."""
    ProviderRegistry.clear()

    class _DummyAdapter:
        pass

    from effgen.models.capabilities import Capability as Cap
    ProviderRegistry.register(
        "policy_prov",
        _DummyAdapter,
        {"pm": {}},
        env_keys=["POLICY_KEY"],
        capabilities={Cap.chat},
    )
    monkeypatch.setenv("POLICY_KEY", "present")

    router = ModelRouter(policies=[FirstAvailablePolicy()])
    decision = router.route(RoutingContext(required_capabilities={Cap.chat}))

    assert decision.chosen == ProviderModelPair("policy_prov", "pm")
    assert decision.policy_name == "first_available"

    ProviderRegistry.reset()


# ---------------------------------------------------------------------------
# Capability enum presence
# ---------------------------------------------------------------------------

def test_every_provider_declares_its_capabilities():
    """Every registered provider must declare at least one capability."""
    ProviderRegistry.register_builtins()

    providers = ProviderRegistry.list_providers()
    assert len(providers) == 10, f"Expected 10 providers, got {len(providers)}: {providers}"
    expected = {
        "anthropic": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.vision,
            Capability.thinking,
        },
        "cerebras": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.json_schema,
        },
        "fireworks": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.json_schema,
        },
        "gemini": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.vision,
            Capability.audio_input,
            Capability.video_input,
            Capability.grounding,
            Capability.thinking,
            Capability.json_schema,
        },
        "groq": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.vision,
            Capability.json_schema,
        },
        "hf": {Capability.chat, Capability.streaming, Capability.audio_input},
        # A protocol rather than a service: what the endpoint behind it can do
        # is the server's business, so only the four every implementation of
        # the protocol supports are claimed.
        "openai_compatible": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.json_schema,
        },
        "openai": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.vision,
            Capability.audio_input,
            Capability.video_input,
            Capability.thinking,
            Capability.json_schema,
        },
        "replicate": {Capability.chat, Capability.streaming, Capability.vision},
        "together": {
            Capability.chat,
            Capability.streaming,
            Capability.tools,
            Capability.vision,
            Capability.json_schema,
        },
    }
    assert set(providers) == set(expected)
    for p, expected_caps in expected.items():
        caps = ProviderRegistry.get_capabilities(p)
        assert len(caps) > 0, f"Provider '{p}' has no capabilities registered"
        assert Capability.chat in caps, f"Provider '{p}' is missing Capability.chat"
        assert caps == expected_caps


def test_capability_enum_values():
    expected = {"chat", "tools", "streaming", "vision", "audio_input", "video_input", "grounding", "thinking", "json_schema"}
    actual = {c.value for c in Capability}
    assert actual == expected
