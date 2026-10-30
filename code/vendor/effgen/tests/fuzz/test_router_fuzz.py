"""
Hypothesis-driven fuzz tests for the effGen router.

Asserts that:
  1. Router always returns a valid RouterDecision or raises NoCandidateError.
  2. No silent null/empty outputs — a result is always one or the other.
  3. RouterDecision.chosen is always a valid ProviderModelPair.
  4. estimate_complexity handles arbitrary query strings without raising.
  5. candidate_unavailable_reason never crashes on arbitrary inputs.

Exit criterion: ≥500 examples per test.
"""

from __future__ import annotations

import string
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.models.capabilities import Capability
from effgen.models.router import (
    NoCandidateError,
    PolicyBasedRouter,
    ProviderModelPair,
    RouterDecision,
    RoutingContext,
    RoutingPolicy,
    candidate_unavailable_reason,
    estimate_complexity,
)

# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    alphabet=string.ascii_letters + string.digits + " +-_./,@()[]{}",
    min_size=0,
    max_size=500,
)

_PROVIDER_NAME = st.text(
    alphabet=string.ascii_lowercase + "_-",
    min_size=1,
    max_size=20,
)

_MODEL_ID = st.text(
    alphabet=string.ascii_letters + string.digits + "-_.",
    min_size=1,
    max_size=50,
)

# Capabilities subset to fuzz with
_ALL_CAPABILITIES = list(Capability)


@st.composite
def provider_model_pair(draw) -> ProviderModelPair:
    provider = draw(_PROVIDER_NAME)
    model_id = draw(_MODEL_ID)
    return ProviderModelPair(provider=provider, model_id=model_id)


@st.composite
def routing_context(draw) -> RoutingContext:
    n_caps = draw(st.integers(min_value=0, max_value=3))
    caps = set(draw(st.lists(
        st.sampled_from(_ALL_CAPABILITIES),
        min_size=n_caps,
        max_size=n_caps,
    )))
    budget = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False)))
    latency = draw(st.one_of(st.none(), st.integers(min_value=100, max_value=60_000)))
    tokens = draw(st.integers(min_value=0, max_value=100_000))
    return RoutingContext(
        prompt_tokens_estimate=tokens,
        user_budget_usd=budget,
        latency_budget_ms=latency,
        required_capabilities=caps,
    )


# ---------------------------------------------------------------------------
# Minimal in-memory routing policy for fuzz
# ---------------------------------------------------------------------------

class _InMemoryPolicy(RoutingPolicy):
    """Policy that selects from a fixed candidate list without any registry/env lookup."""

    def __init__(self, available: list[ProviderModelPair]) -> None:
        self._available = available

    @property
    def name(self) -> str:
        return "in_memory"

    def select(
        self,
        candidates: list[ProviderModelPair],
        context: RoutingContext,
    ) -> RouterDecision:
        # Filter to our known-available set
        eligible = [c for c in candidates if c in self._available]
        if not eligible:
            raise NoCandidateError("no available candidates")
        chosen = eligible[0]
        eliminated = [(c, "not selected") for c in candidates if c != chosen]
        return RouterDecision(
            chosen=chosen,
            eliminated=eliminated,
            policy_name=self.name,
            score=1.0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(query=_SAFE_TEXT)
def test_estimate_complexity_fuzz(query):
    """estimate_complexity never crashes on arbitrary strings."""
    from effgen.models.router import ComplexityEstimate
    result = estimate_complexity(query)
    assert isinstance(result, ComplexityEstimate)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.required_capabilities, list)
    assert isinstance(result.reasoning, str)


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    query=_SAFE_TEXT,
    n_tools=st.integers(min_value=0, max_value=20),
)
def test_estimate_complexity_with_tools_fuzz(query, n_tools):
    """estimate_complexity with tool list never crashes."""
    fake_tools = [object()] * n_tools
    result = estimate_complexity(query, tools=fake_tools)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    candidates=st.lists(provider_model_pair(), min_size=0, max_size=8),
    context=routing_context(),
    n_available=st.integers(min_value=0, max_value=5),
)
def test_router_always_decides_or_raises(candidates, context, n_available):
    """PolicyBasedRouter returns RouterDecision or NoCandidateError — never None or silent fail."""
    # Pick a subset as "available"
    available = candidates[:n_available]
    policy = _InMemoryPolicy(available)
    router = PolicyBasedRouter(policies=[policy], fallback=_InMemoryPolicy(available))

    # Patch _get_candidates to return our fuzz candidates
    with patch.object(router, "_get_candidates", return_value=candidates):
        try:
            decision = router.route(context)
            # Post-condition: must be a valid RouterDecision
            assert isinstance(decision, RouterDecision), (
                f"route() returned {type(decision)}, expected RouterDecision"
            )
            assert decision.chosen is not None, "RouterDecision.chosen must not be None"
            assert isinstance(decision.chosen, ProviderModelPair)
            assert decision.chosen.provider, "chosen.provider must be non-empty"
            assert decision.chosen.model_id, "chosen.model_id must be non-empty"
        except NoCandidateError:
            # Acceptable — when no candidates are available
            pass
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"Unexpected exception from router.route(): {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_providers=st.integers(min_value=0, max_value=5),
    context=routing_context(),
)
def test_router_no_silent_empty(n_providers, context):
    """Router never returns an empty/None decision silently."""
    pairs = [ProviderModelPair(f"prov_{i}", f"model_{i}") for i in range(n_providers)]
    policy = _InMemoryPolicy(pairs)
    router = PolicyBasedRouter(policies=[policy], fallback=_InMemoryPolicy(pairs))

    with patch.object(router, "_get_candidates", return_value=pairs):
        try:
            decision = router.route(context)
            assert decision is not None
            assert decision.chosen is not None
        except NoCandidateError:
            pass  # Valid when no providers
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    pair=provider_model_pair(),
    context=routing_context(),
)
def test_candidate_unavailable_reason_fuzz(pair, context):
    """candidate_unavailable_reason never crashes on arbitrary (provider, context) inputs."""
    # Mock registry to avoid real env key lookups
    mock_provider_info: dict[str, Any] = {
        "env_keys": [],
        "models": {
            pair.model_id: {
                "modality": "text",
                "supports_streaming": True,
                "supports_native_tools": True,
            }
        },
    }
    mock_caps: set[Capability] = set()

    with (
        patch("effgen.models.registry.ProviderRegistry.get_provider_info", return_value=mock_provider_info),
        patch("effgen.models.registry.ProviderRegistry.get_capabilities", return_value=mock_caps),
    ):
        try:
            reason = candidate_unavailable_reason(pair, context)
            assert reason is None or isinstance(reason, str)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    required_caps=st.frozensets(st.sampled_from(_ALL_CAPABILITIES), max_size=5),
    model_info=st.fixed_dictionaries({
        "modality": st.sampled_from(["text", "vision", "multimodal", "audio",
                                      "embedding", "image", "rerank", "tts", "stt", ""]),
        "supports_streaming": st.one_of(st.booleans(), st.just(None)),
        "supports_native_tools": st.one_of(st.booleans(), st.just(None)),
        "supports_vision": st.one_of(st.booleans(), st.just(None)),
        "active": st.one_of(st.booleans(), st.just(None)),
    }),
)
def test_model_capability_reason_fuzz(required_caps, model_info):
    """_model_capability_reason never crashes on arbitrary model_info dicts."""
    from effgen.models.router import _model_capability_reason
    try:
        reason = _model_capability_reason(model_info, required_caps)
        assert reason is None or isinstance(reason, str)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    failover_hops=st.integers(min_value=0, max_value=5),
)
def test_router_construction_fuzz(failover_hops):
    """PolicyBasedRouter construction never crashes on valid failover_hops."""
    policy = _InMemoryPolicy([])
    router = PolicyBasedRouter(
        policies=[policy],
        fallback=policy,
        failover_hops=failover_hops,
    )
    assert router._failover_hops == failover_hops


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    failover_hops=st.integers(min_value=-100, max_value=-1),
)
def test_router_invalid_failover_hops_fuzz(failover_hops):
    """PolicyBasedRouter raises ValueError for negative failover_hops."""
    policy = _InMemoryPolicy([])
    try:
        PolicyBasedRouter(
            policies=[policy],
            fallback=policy,
            failover_hops=failover_hops,
        )
        pytest.fail("Expected ValueError for negative failover_hops")
    except ValueError:
        pass  # Expected


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    candidates=st.lists(provider_model_pair(), min_size=1, max_size=10),
    context=routing_context(),
)
def test_router_decision_fields_fuzz(candidates, context):
    """RouterDecision from fuzz inputs has all required fields populated."""
    policy = _InMemoryPolicy(list(candidates))
    router = PolicyBasedRouter(policies=[policy], fallback=_InMemoryPolicy(list(candidates)))

    with patch.object(router, "_get_candidates", return_value=list(candidates)):
        try:
            decision = router.route(context)
            # Every field must be populated
            assert decision.chosen is not None
            assert isinstance(decision.eliminated, list)
            assert isinstance(decision.policy_name, str)
            assert isinstance(decision.score, int | float)
        except NoCandidateError:
            pass
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    provider=_PROVIDER_NAME,
    model_id=_MODEL_ID,
)
def test_provider_model_pair_fuzz(provider, model_id):
    """ProviderModelPair construction is always stable."""
    pair = ProviderModelPair(provider=provider, model_id=model_id)
    assert pair.provider == provider
    assert pair.model_id == model_id
    # Should be hashable (used in sets/dicts)
    s = {pair}
    assert pair in s


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(context=routing_context())
def test_routing_context_fuzz(context):
    """RoutingContext construction from random inputs is always valid."""
    assert context.prompt_tokens_estimate >= 0
    assert isinstance(context.required_capabilities, set)
    if context.user_budget_usd is not None:
        assert context.user_budget_usd >= 0.0
