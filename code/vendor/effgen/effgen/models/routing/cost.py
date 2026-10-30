"""CostBasedPolicy — cheapest-first routing for the effGen ModelRouter.

Picks the cheapest (provider, model_id) pair that:
  1. Has a configured API key.
  2. Satisfies all required capabilities.
  3. Has an estimated cost ≤ user_budget_usd (if set).

Cost estimate per call:
    cost = (input_per_1m * prompt_tokens / 1_000_000)
         + (output_per_1m * expected_output_tokens / 1_000_000)

Per-model pricing from model registry (``pricing_per_1m_input`` /
``pricing_per_1m_output`` fields) takes precedence over provider-level defaults,
except providers that are routed as zero-cost free tiers.

Free-tier providers (``free_tier=True``) rank ahead of paid providers when
estimated cost is equal. Remaining ties are deterministic: provider priority,
published list cost, provider name, then model id.

Pricing data verified against provider pricing pages on 2026-05-11.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from effgen.models.errors import NoCandidateWithinBudgetError
from effgen.models.router import (
    NoCandidateError,
    ProviderModelPair,
    RouterDecision,
    RoutingContext,
    RoutingPolicy,
    candidate_unavailable_reason,
)

logger = logging.getLogger(__name__)

_DEFAULT_EXPECTED_OUTPUT = 1024  # tokens

# These providers have a free hosted/serverless tier that should route as
# zero out-of-pocket cost while quota remains.  Gemini also has free quota, but
# effGen keeps its paid published token prices for routing because the free tier
# has stricter product/data-use nuances.
_ZERO_COST_FREE_TIER_PROVIDERS = frozenset({"cerebras", "groq", "hf"})

_PROVIDER_TIE_BREAK_ORDER = {
    provider: rank
    for rank, provider in enumerate((
        "groq",
        "cerebras",
        "hf",
        "gemini",
        "together",
        "fireworks",
        "openai",
        "replicate",
        "anthropic",
    ))
}


@dataclass(order=True)
class _ScoredCandidate:
    """Internal sort key for ranking candidates cheapest-first."""
    cost: float
    # free_tier candidates sort before paid ones at equal cost
    is_free_tier: bool = field(compare=False)
    list_cost: float = field(compare=False)
    pair: ProviderModelPair = field(compare=False)
    input_per_1m: float = field(compare=False)
    output_per_1m: float = field(compare=False)

    def sort_key(self) -> tuple:
        """Stable, documented ordering for otherwise equal candidates."""
        provider_rank = _PROVIDER_TIE_BREAK_ORDER.get(
            self.pair.provider,
            len(_PROVIDER_TIE_BREAK_ORDER),
        )
        return (
            self.cost,
            0 if self.is_free_tier else 1,
            provider_rank,
            self.list_cost,
            self.pair.provider,
            self.pair.model_id,
        )


def _cost_for(
    input_per_1m: float,
    output_per_1m: float,
    prompt_tokens: int,
    expected_output_tokens: int,
) -> float:
    """Estimate USD cost from per-million-token prices."""
    return (
        input_per_1m * prompt_tokens / 1_000_000
        + output_per_1m * expected_output_tokens / 1_000_000
    )


def _get_model_pricing(
    provider: str,
    model_id: str,
) -> tuple[float, float, bool]:
    """Return (input_per_1m, output_per_1m, free_tier) for a (provider, model_id).

    Priority:
      1. Per-model ``pricing_per_1m_input`` / ``pricing_per_1m_output`` in the
         model info dict (if present).
      2. Provider-level ``pricing`` dict from the ProviderRegistry.
    """
    from effgen.models.registry import ProviderRegistry

    # Provider-level defaults
    prov_pricing = ProviderRegistry.get_pricing(provider)
    prov_input = float(prov_pricing.get("input_per_1m", 0.0))
    prov_output = float(prov_pricing.get("output_per_1m", 0.0))
    provider_free_tier = bool(prov_pricing.get("free_tier", False))
    free_tier = provider_free_tier

    # Try per-model override — support two naming conventions used across providers:
    # - Standard: pricing_per_1m_input / pricing_per_1m_output
    # - OpenAI-style: input_price_per_1m / output_price_per_1m
    model_info: dict = {}
    try:
        rec = ProviderRegistry.get_provider_info(provider)
        model_info = rec.get("models", {}).get(model_id, {})
        if "free_tier" in model_info and provider_free_tier:
            free_tier = bool(model_info["free_tier"])
        if "pricing_per_1m_input" in model_info:
            prov_input = float(model_info["pricing_per_1m_input"])
        elif "input_price_per_1m" in model_info:
            prov_input = float(model_info["input_price_per_1m"])
        if "pricing_per_1m_output" in model_info:
            prov_output = float(model_info["pricing_per_1m_output"])
        elif "output_price_per_1m" in model_info:
            prov_output = float(model_info["output_price_per_1m"])
    except Exception:
        logger.debug("Failed to read provider pricing from model info", exc_info=True)

    if provider in _ZERO_COST_FREE_TIER_PROVIDERS and free_tier:
        return 0.0, 0.0, True

    return prov_input, prov_output, free_tier


class CostBasedPolicy(RoutingPolicy):
    """Select the cheapest provider/model pair within the user's budget.

    Args:
        expected_output_tokens: Token count to use for output cost estimation
                                when actual output length is unknown.
                                Defaults to 1024.

    Usage::

        from effgen.models.routing.cost import CostBasedPolicy
        from effgen.models.router import PolicyBasedRouter, RoutingContext
        from effgen.models.capabilities import Capability

        router = PolicyBasedRouter(policies=[CostBasedPolicy()])
        ctx = RoutingContext(
            prompt_tokens_estimate=500,
            user_budget_usd=0.001,
            required_capabilities={Capability.chat},
        )
        decision = router.route(ctx)
    """

    def __init__(self, expected_output_tokens: int = _DEFAULT_EXPECTED_OUTPUT) -> None:
        self._expected_output = expected_output_tokens

    @property
    def name(self) -> str:
        """Policy identifier: ``"cost_based"``."""
        return "cost_based"

    def select(
        self,
        candidates: list[ProviderModelPair],
        context: RoutingContext,
    ) -> RouterDecision:
        """Pick the cheapest capable candidate within the user's budget."""
        from effgen.models.registry import ProviderRegistry

        prompt_tokens = max(context.prompt_tokens_estimate, 0)
        budget = context.user_budget_usd  # None = unlimited

        eliminated: list[tuple[ProviderModelPair, str]] = []
        scored: list[_ScoredCandidate] = []

        for pair in candidates:
            provider = pair.provider
            rec = ProviderRegistry.get_provider_info(provider)
            unavailable_reason = candidate_unavailable_reason(pair, context)
            if unavailable_reason:
                eliminated.append((pair, unavailable_reason))
                continue

            model_info = rec.get("models", {}).get(pair.model_id, {})
            inp_price, out_price, free_tier = _get_model_pricing(provider, pair.model_id)
            cost = _cost_for(inp_price, out_price, prompt_tokens, self._expected_output)

            list_inp = inp_price
            list_out = out_price
            list_cost = _cost_for(list_inp, list_out, prompt_tokens, self._expected_output)
            if provider in _ZERO_COST_FREE_TIER_PROVIDERS and free_tier:
                has_list_price = False
                if "pricing_per_1m_input" in model_info:
                    list_inp = float(model_info["pricing_per_1m_input"])
                    has_list_price = True
                elif "input_price_per_1m" in model_info:
                    list_inp = float(model_info["input_price_per_1m"])
                    has_list_price = True
                if "pricing_per_1m_output" in model_info:
                    list_out = float(model_info["pricing_per_1m_output"])
                    has_list_price = True
                elif "output_price_per_1m" in model_info:
                    list_out = float(model_info["output_price_per_1m"])
                    has_list_price = True
                if not has_list_price:
                    list_cost = float("inf")
                else:
                    list_cost = _cost_for(
                        list_inp,
                        list_out,
                        prompt_tokens,
                        self._expected_output,
                    )

            scored.append(_ScoredCandidate(
                cost=cost,
                is_free_tier=free_tier,
                list_cost=list_cost,
                pair=pair,
                input_per_1m=inp_price,
                output_per_1m=out_price,
            ))

        if not scored:
            raise NoCandidateError(
                f"CostBasedPolicy: no candidate has a configured key and "
                f"required capabilities. Eliminated {len(eliminated)}. "
                "Configure a key for one of the candidate providers, or relax "
                "the required capabilities."
            )

        # Sort cheapest-first; free-tier breaks ties
        scored.sort(key=lambda s: s.sort_key())

        # Find cheapest within budget
        within_budget = [
            s for s in scored
            if budget is None or s.cost <= budget
        ]

        if not within_budget:
            cheapest = scored[0]
            raise NoCandidateWithinBudgetError(
                user_budget_usd=budget,
                cheapest_cost_usd=cheapest.cost,
                cheapest_pair=(cheapest.pair.provider, cheapest.pair.model_id),
            )

        chosen_scored = within_budget[0]
        chosen = chosen_scored.pair

        # Mark the rest as eliminated
        for s in scored:
            if s.pair != chosen:
                if budget is not None and s.cost > budget:
                    eliminated.append((s.pair, f"cost ${s.cost:.6f} > budget ${budget:.6f}"))
                else:
                    eliminated.append((
                        s.pair,
                        f"not cheapest (cost=${s.cost:.6f}, chosen=${chosen_scored.cost:.6f})",
                    ))

        logger.info(
            "CostBasedPolicy: chose %s/%s (cost=$%.6f, free_tier=%s, budget=%s)",
            chosen.provider, chosen.model_id,
            chosen_scored.cost, chosen_scored.is_free_tier,
            f"${budget:.6f}" if budget is not None else "unlimited",
        )

        return RouterDecision(
            chosen=chosen,
            eliminated=eliminated,
            policy_name=self.name,
            score=chosen_scored.cost,
        )
