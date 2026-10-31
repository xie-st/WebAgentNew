"""LatencyBasedPolicy — SLA-aware routing for the effGen ModelRouter.

Picks the fastest (provider, model_id) pair (by p50 observed latency) that:
  1. Has a configured API key.
  2. Satisfies all required capabilities.
  3. Has p50 latency ≤ latency_budget_ms (if set).

Pairs with no observed latency data are assigned a tie-breaker seed value so
they can still be selected.  The warm-up probe in ``_probe.py`` seeds the
tracker before routing when history is empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from effgen.models.errors import NoCandidateWithinLatencyError
from effgen.models.latency_tracker import LatencyTracker
from effgen.models.router import (
    NoCandidateError,
    ProviderModelPair,
    RouterDecision,
    RoutingContext,
    RoutingPolicy,
    candidate_unavailable_reason,
)

logger = logging.getLogger(__name__)

# Default seed latency for providers with no data.
# Reflects rough real-world ordering: fast cloud inference first.
_PROVIDER_SEED_MS: dict[str, float] = {
    "cerebras": 300.0,
    "groq": 350.0,
    "hf": 800.0,
    "together": 1000.0,
    "fireworks": 1100.0,
    "openai": 1200.0,
    "gemini": 1300.0,
    "replicate": 2000.0,
    "anthropic": 1400.0,
}
_UNKNOWN_SEED_MS = 1500.0


@dataclass(frozen=True)
class _LatencyScore:
    latency_ms: float
    pair: ProviderModelPair
    has_observed_data: bool


def _seed_latency(provider: str) -> float:
    return _PROVIDER_SEED_MS.get(provider, _UNKNOWN_SEED_MS)


class LatencyBasedPolicy(RoutingPolicy):
    """Select the fastest provider/model pair that satisfies the SLA.

    When ``latency_budget_ms`` is set in the routing context, all candidates
    whose observed p50 latency exceeds the budget are eliminated.  Among the
    survivors the pair with the lowest p50 is chosen.

    Candidates with no observed latency use a conservative seed value derived
    from provider defaults, so they can still participate in routing.

    Args:
        tracker:          LatencyTracker to query.  Defaults to the process-
                          global singleton (``LatencyTracker.get()``).
        seed_latency_ms:  Override the per-provider seed values with a single
                          fallback.  Useful for testing.
        enable_warmup:    If True, an empty tracker triggers the warm-up probe
                          before candidates are scored.

    Usage::

        from effgen.models.routing.latency import LatencyBasedPolicy
        from effgen.models.router import PolicyBasedRouter, RoutingContext
        from effgen.models.capabilities import Capability

        router = PolicyBasedRouter(policies=[LatencyBasedPolicy()])
        ctx = RoutingContext(
            latency_budget_ms=2000,
            required_capabilities={Capability.chat},
        )
        decision = router.route(ctx)
        print(decision.chosen, decision.score)  # score = p50 ms
    """

    def __init__(
        self,
        tracker: LatencyTracker | None = None,
        seed_latency_ms: float | None = None,
        enable_warmup: bool = True,
    ) -> None:
        self._tracker = tracker or LatencyTracker.get()
        self._seed_latency_ms = seed_latency_ms  # None → use per-provider seeds
        self._enable_warmup = enable_warmup

    @property
    def name(self) -> str:
        """Policy identifier: ``"latency_based"``."""
        return "latency_based"

    def _get_latency(self, provider: str, model_id: str) -> float:
        """Return observed p50 or a seed value if no data."""
        p50 = self._tracker.p50(provider, model_id)
        if p50 is not None:
            return p50
        if self._seed_latency_ms is not None:
            return self._seed_latency_ms
        return _seed_latency(provider)

    def _warm_up_if_needed(self, context: RoutingContext) -> None:
        """Seed the tracker before first latency route when it has no data."""
        if not self._enable_warmup or not self._tracker.is_empty():
            return

        from effgen.models.routing._probe import warm_up_providers

        logger.info("LatencyBasedPolicy: empty tracker; running warm-up probe")
        warm_up_providers(
            context_caps=context.required_capabilities,
            tracker=self._tracker,
        )

    def select(
        self,
        candidates: list[ProviderModelPair],
        context: RoutingContext,
    ) -> RouterDecision:
        """Pick the candidate with the lowest observed (or seeded) latency."""
        self._warm_up_if_needed(context)

        budget_ms = context.latency_budget_ms
        eliminated: list[tuple[ProviderModelPair, str]] = []
        available: list[_LatencyScore] = []

        for pair in candidates:
            unavailable_reason = candidate_unavailable_reason(pair, context)
            if unavailable_reason:
                eliminated.append((pair, unavailable_reason))
                continue

            latency = self._get_latency(pair.provider, pair.model_id)
            has_data = self._tracker.has_data(pair.provider, pair.model_id)
            available.append(_LatencyScore(
                latency_ms=latency,
                pair=pair,
                has_observed_data=has_data,
            ))

        if not available:
            raise NoCandidateError(
                "LatencyBasedPolicy: no candidate has a configured key, "
                f"available model, and required capabilities. Eliminated {len(eliminated)}. "
                "Configure a key for one of the candidate providers, or relax "
                "the required capabilities."
            )

        # Once there are real measurements, route only on measured candidates.
        # Seed values are cold-start hints, not evidence that an unmeasured
        # provider is faster than one that has actually been observed.
        observed_available = [
            score for score in available
            if score.has_observed_data
        ]
        candidate_pool = observed_available or available

        within_budget = [
            score for score in candidate_pool
            if budget_ms is None or score.latency_ms <= budget_ms
        ]

        if not within_budget:
            fastest = min(
                candidate_pool,
                key=lambda score: (score.latency_ms, score.pair.provider, score.pair.model_id),
            )
            for score in candidate_pool:
                eliminated.append((
                    score.pair,
                    f"p50={score.latency_ms:.0f}ms > budget={budget_ms}ms"
                    + ("" if score.has_observed_data else " (seed)"),
                ))
            if observed_available:
                for score in available:
                    if not score.has_observed_data:
                        eliminated.append((
                            score.pair,
                            "no observed latency yet; real measurements were available",
                        ))
            raise NoCandidateWithinLatencyError(
                latency_budget_ms=float(budget_ms or 0),
                fastest_latency_ms=fastest.latency_ms,
                fastest_pair=(fastest.pair.provider, fastest.pair.model_id),
            )

        chosen_score = min(
            within_budget,
            key=lambda score: (score.latency_ms, score.pair.provider, score.pair.model_id),
        )
        chosen = chosen_score.pair
        chosen_latency = chosen_score.latency_ms

        for score in available:
            if score.pair == chosen:
                continue
            if budget_ms is not None and score.latency_ms > budget_ms:
                eliminated.append((
                    score.pair,
                    f"p50={score.latency_ms:.0f}ms > budget={budget_ms}ms"
                    + ("" if score.has_observed_data else " (seed)"),
                ))
                continue
            if observed_available and not score.has_observed_data:
                eliminated.append((
                    score.pair,
                    "no observed latency yet; real measurements were available",
                ))
                continue
            eliminated.append((
                score.pair,
                f"p50={score.latency_ms:.0f}ms > chosen={chosen_latency:.0f}ms",
            ))

        logger.info(
            "LatencyBasedPolicy: chose %s/%s (p50=%.0fms%s, budget=%s)",
            chosen.provider, chosen.model_id,
            chosen_latency,
            "" if chosen_score.has_observed_data else " seed",
            f"{budget_ms}ms" if budget_ms is not None else "unlimited",
        )

        return RouterDecision(
            chosen=chosen,
            eliminated=eliminated,
            policy_name=self.name,
            score=chosen_latency,
        )
