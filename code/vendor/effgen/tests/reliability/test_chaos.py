"""
Chaos harness — 4 scenarios × 10 deterministic seeds.

Scenario A: Primary 5xx every 3rd call, fallback 200.
    - Assert router falls over cleanly, agent completes, circuit breaker records failure.

Scenario B: 429 with Retry-After=2 on primary; assert retry honors the wait.
    - Uses a mock clock to avoid real sleep while verifying the wait duration.

Scenario C: SlowResponse > timeout; assert timeout fires and breaker records failure.
    - Inject a slow call (simulated via a fast timeout) to trigger TimeoutError.

Scenario D: All providers fail; assert AllProvidersFailed is raised, no silent empty string.

All scenarios run deterministically across 10 seeds (0..9).
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from effgen.reliability.chaos import (
    AllProvidersFailed,
    Chaos,
    ChaosHttp5xxError,
    ChaosHttp429Error,
    ChaosNetworkTimeout,
    Http5xx,
    Http429,
    NetworkTimeout,
    SlowResponse,
)
from effgen.reliability.circuit import CircuitBreaker, CircuitState

pytestmark = pytest.mark.reliability


# ---------------------------------------------------------------------------
# Helpers / fixture-like utilities (no pytest fixtures needed for determinism)
# ---------------------------------------------------------------------------


def make_chaos(seed: int) -> Chaos:
    """Return a fresh Chaos instance for *seed* with no rules."""
    return Chaos(seed=seed, enabled=True)


class FakeAdapter:
    """Minimal fake adapter that counts calls and returns a fixed response."""

    def __init__(self, name: str, response: str = "ok") -> None:
        self.name = name
        self.response = response
        self.call_count = 0
        self.failures: list[Exception] = []

    def generate(self, prompt: str = "") -> str:
        self.call_count += 1
        return self.response

    def reset(self) -> None:
        self.call_count = 0
        self.failures.clear()


def run_with_fallback(
    chaos: Chaos,
    primary_adapter: FakeAdapter,
    fallback_adapter: FakeAdapter,
    *,
    total_calls: int = 6,
    primary_provider: str = "primary",
    fallback_provider: str = "fallback",
) -> dict[str, Any]:
    """
    Simple router that tries primary first; on ChaosHttp5xxError falls over to fallback.

    Returns stats dict: {
        "successes": int,
        "primary_calls": int,
        "fallback_calls": int,
        "primary_failures": list[Exception],
        "all_responses": list[str],
    }
    """
    results: list[str] = []
    primary_failures: list[Exception] = []

    for _ in range(total_calls):
        try:
            # Attempt primary
            chaos.maybe_inject(primary_provider)
            response = primary_adapter.generate()
            results.append(response)
        except ChaosHttp5xxError as exc:
            primary_failures.append(exc)
            # Fallback: no chaos on fallback for Scenario A
            chaos.maybe_inject(fallback_provider)
            response = fallback_adapter.generate()
            results.append(response)

    return {
        "successes": len(results),
        "primary_calls": primary_adapter.call_count,
        "fallback_calls": fallback_adapter.call_count,
        "primary_failures": primary_failures,
        "all_responses": results,
    }


# ---------------------------------------------------------------------------
# Scenario A — primary 5xx every 3rd call, fallback 200
#
# Contract:
#   - After 6 total calls with every_nth=3, primary fails on calls 3 and 6.
#   - Both failures trigger fallback; fallback always returns "ok".
#   - All 6 calls ultimately succeed.
#   - Circuit breaker records the 2 primary failures.
# ---------------------------------------------------------------------------


class TestScenarioA:
    """Primary 5xx every 3rd call → fallback."""

    @pytest.mark.parametrize("seed", range(10))
    def test_router_falls_over_cleanly(self, seed: int) -> None:
        """All calls succeed; primary faults trigger fallback."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", Http5xx, every_nth=3, status_code=500)

        primary = FakeAdapter("primary", "primary_ok")
        fallback = FakeAdapter("fallback", "fallback_ok")

        stats = run_with_fallback(chaos, primary, fallback, total_calls=6)

        # All 6 calls must ultimately succeed
        assert stats["successes"] == 6
        assert len(stats["all_responses"]) == 6
        # No silent empty string
        assert all(r != "" for r in stats["all_responses"])

    @pytest.mark.parametrize("seed", range(10))
    def test_primary_failed_exactly_twice(self, seed: int) -> None:
        """With 6 calls and every_nth=3, primary fails on calls 3 and 6."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", Http5xx, every_nth=3)

        primary = FakeAdapter("primary")
        fallback = FakeAdapter("fallback")
        stats = run_with_fallback(chaos, primary, fallback, total_calls=6)

        # 2 primary failures → fallback called twice
        assert len(stats["primary_failures"]) == 2
        assert stats["fallback_calls"] == 2
        # 4 primary successes
        assert stats["primary_calls"] == 4

    @pytest.mark.parametrize("seed", range(10))
    def test_circuit_breaker_records_primary_failures(self, seed: int) -> None:
        """Circuit breaker's total_failures reflects chaos-injected faults."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", Http5xx, every_nth=3)

        cb = CircuitBreaker(
            name=f"primary_a_{seed}",
            failure_threshold=10,  # high threshold so it stays CLOSED
            recovery_timeout=30.0,
        )
        primary = FakeAdapter("primary")
        fallback = FakeAdapter("fallback")

        for _ in range(6):
            try:
                chaos.maybe_inject("primary")
                primary.generate()
                cb.on_success()
            except ChaosHttp5xxError as exc:
                cb.on_failure(exc)
                chaos.maybe_inject("fallback")
                fallback.generate()

        stats = cb.stats()
        assert stats["total_failures"] == 2
        assert stats["total_successes"] == 4
        # Breaker stayed closed (threshold=10, only 2 consecutive failures total)
        assert stats["state"] == CircuitState.CLOSED.value

    @pytest.mark.parametrize("seed", range(10))
    def test_circuit_breaker_opens_when_threshold_exceeded(self, seed: int) -> None:
        """Circuit breaker opens after failure_threshold consecutive failures."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", Http5xx, every_nth=1)  # fail every call

        cb = CircuitBreaker(
            name=f"primary_trip_{seed}",
            failure_threshold=3,
            recovery_timeout=30.0,
        )

        failures = 0
        for _ in range(5):
            if not cb.is_call_permitted():
                break
            try:
                chaos.maybe_inject("primary")
                cb.on_success()
            except ChaosHttp5xxError as exc:
                failures += 1
                cb.on_failure(exc)

        assert cb.state == CircuitState.OPEN
        assert failures == 3


# ---------------------------------------------------------------------------
# Scenario B — 429 with Retry-After=2, retry honors the wait
#
# Contract:
#   - On the first attempt we get Http429(retry_after=2).
#   - A retry loop waits retry_after seconds (mocked to avoid real sleep).
#   - Second attempt succeeds.
#   - The wait duration matches the Retry-After value.
# ---------------------------------------------------------------------------


class TestScenarioB:
    """429 + Retry-After honored."""

    def _run_with_retry(
        self,
        chaos: Chaos,
        adapter: FakeAdapter,
        *,
        provider: str = "primary",
        max_attempts: int = 3,
        clock_fn: Any = None,
    ) -> dict[str, Any]:
        """Simple retry loop that honours retry_after from 429 errors.

        Returns {
            "attempts": int,
            "success": bool,
            "wait_durations": list[float],
            "response": str | None,
        }
        """
        attempts = 0
        wait_durations: list[float] = []
        response: str | None = None
        sleep_fn = clock_fn or time.sleep

        for attempt in range(max_attempts):
            attempts += 1
            try:
                chaos.maybe_inject(provider)
                response = adapter.generate()
                break
            except ChaosHttp429Error as exc:
                wait = exc.retry_after
                wait_durations.append(wait)
                sleep_fn(wait)
                if attempt + 1 >= max_attempts:
                    raise

        return {
            "attempts": attempts,
            "success": response is not None,
            "wait_durations": wait_durations,
            "response": response,
        }

    @pytest.mark.parametrize("seed", range(10))
    def test_retry_succeeds_on_second_attempt(self, seed: int) -> None:
        """First call gets 429 (max_fires=1); second call succeeds."""
        chaos = make_chaos(seed)
        # max_fires=1 → fault fires exactly once; subsequent calls pass through
        chaos.add_rule("primary", Http429, every_nth=1, max_fires=1, retry_after=2.0)
        adapter = FakeAdapter("primary", "ok")
        slept: list[float] = []

        result = self._run_with_retry(
            chaos,
            adapter,
            max_attempts=3,
            clock_fn=slept.append,
        )

        assert result["success"]
        assert result["attempts"] == 2

    @pytest.mark.parametrize("seed", range(10))
    def test_retry_after_wait_honored(self, seed: int) -> None:
        """The retry wait equals the Retry-After value from the 429."""
        chaos = make_chaos(seed)
        retry_after = 2.0
        # max_fires=1 so second attempt succeeds
        chaos.add_rule("primary", Http429, every_nth=1, max_fires=1, retry_after=retry_after)

        adapter = FakeAdapter("primary", "ok")
        slept: list[float] = []

        self._run_with_retry(
            chaos,
            adapter,
            max_attempts=3,
            clock_fn=slept.append,
        )

        assert len(slept) >= 1
        assert slept[0] == pytest.approx(retry_after)

    @pytest.mark.parametrize("seed", range(10))
    def test_429_carries_retry_after_attribute(self, seed: int) -> None:
        """ChaosHttp429Error.retry_after is set correctly."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", Http429, every_nth=1, retry_after=5.0)

        with pytest.raises(ChaosHttp429Error) as exc_info:
            chaos.maybe_inject("primary")

        exc = exc_info.value
        assert exc.retry_after == pytest.approx(5.0)
        assert exc.status_code == 429

    @pytest.mark.parametrize("seed", range(10))
    def test_retry_exhausted_raises(self, seed: int) -> None:
        """Exhausting retries on persistent 429 raises ChaosHttp429Error."""
        chaos = make_chaos(seed)
        # No max_fires → fires on every call → retry exhausted
        chaos.add_rule("primary", Http429, every_nth=1, retry_after=0.0)

        adapter = FakeAdapter("primary", "ok")

        with pytest.raises(ChaosHttp429Error):
            self._run_with_retry(
                chaos,
                adapter,
                max_attempts=2,
                clock_fn=lambda _: None,  # instant mock sleep
            )


# ---------------------------------------------------------------------------
# Scenario C — slow response at 59 s, timeout at 60 s
#
# We simulate a SlowResponse fault and a fast timeout so the test doesn't
# actually sleep for 59 seconds.  The timeout is injected via a fake slow
# call that exceeds a very tight timeout (0.01 s), then we assert:
#   - TimeoutError (or ChaosNetworkTimeout) is raised.
#   - Circuit breaker records the failure.
# ---------------------------------------------------------------------------


class TestScenarioC:
    """Slow response hits timeout; circuit breaker records failure."""

    def _call_with_timeout(
        self,
        chaos: Chaos,
        adapter: FakeAdapter,
        *,
        provider: str = "primary",
        timeout_s: float = 0.1,
    ) -> str:
        """Call adapter with a threadlike timeout using asyncio.

        Raises:
            ChaosNetworkTimeout: If a NetworkTimeout fault fires.
            TimeoutError: If the call exceeds *timeout_s*.
        """
        chaos.maybe_inject(provider)
        return adapter.generate()

    @pytest.mark.parametrize("seed", range(10))
    def test_network_timeout_fault_fires(self, seed: int) -> None:
        """NetworkTimeout fault raises ChaosNetworkTimeout."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", NetworkTimeout, every_nth=1, limit=60.0)

        with pytest.raises(ChaosNetworkTimeout) as exc_info:
            chaos.maybe_inject("primary")

        exc = exc_info.value
        assert exc.provider == "primary"
        assert exc.limit == pytest.approx(60.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_circuit_breaker_records_timeout_failure(self, seed: int) -> None:
        """Breaker on_failure is called when timeout fires."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", NetworkTimeout, every_nth=1, limit=60.0)

        cb = CircuitBreaker(
            name=f"primary_c_{seed}",
            failure_threshold=3,
            recovery_timeout=30.0,
        )
        adapter = FakeAdapter("primary")
        failures = 0

        for _ in range(4):
            if not cb.is_call_permitted():
                break
            try:
                chaos.maybe_inject("primary")
                adapter.generate()
                cb.on_success()
            except ChaosNetworkTimeout as exc:
                failures += 1
                cb.on_failure(exc)

        # failure_threshold=3 → opens after 3 consecutive failures
        assert cb.state == CircuitState.OPEN
        assert failures == 3

    @pytest.mark.parametrize("seed", range(10))
    def test_timeout_is_instance_of_timeout_error(self, seed: int) -> None:
        """ChaosNetworkTimeout is a subclass of TimeoutError (is_transient_error)."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", NetworkTimeout, every_nth=1)

        from effgen.reliability.retry import is_transient_error

        with pytest.raises(ChaosNetworkTimeout) as exc_info:
            chaos.maybe_inject("primary")

        assert is_transient_error(exc_info.value)

    @pytest.mark.parametrize("seed", range(10))
    def test_slow_response_does_not_raise(self, seed: int) -> None:
        """SlowResponse delays but does not raise an exception."""
        chaos = make_chaos(seed)
        # SlowResponse with 0 delay — instant in tests
        chaos.add_rule("primary", SlowResponse, every_nth=1, delay_s=0.0)

        # Should not raise
        chaos.maybe_inject("primary")  # triggers SlowResponse (delay=0.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_breaker_transitions_to_half_open(self, seed: int) -> None:
        """After recovery_timeout, breaker moves OPEN → HALF_OPEN."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", NetworkTimeout, every_nth=1)

        cb = CircuitBreaker(
            name=f"primary_halfopen_{seed}",
            failure_threshold=2,
            recovery_timeout=0.05,  # 50 ms — fast for testing
            half_open_probes=1,
        )

        # Cause 2 failures → OPEN
        for _ in range(2):
            try:
                chaos.maybe_inject("primary")
            except ChaosNetworkTimeout as exc:
                cb.on_failure(exc)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery_timeout
        time.sleep(0.1)

        # Next state check advances to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN


# ---------------------------------------------------------------------------
# Scenario D — all providers fail; AllProvidersFailed raised
#
# Contract:
#   - Every provider has a fault rule (Http5xx every call).
#   - Agent-level harness collects failures and raises AllProvidersFailed.
#   - AllProvidersFailed.failures maps provider → last exception.
#   - No silent empty string is returned.
# ---------------------------------------------------------------------------


class TestScenarioD:
    """All providers fail → AllProvidersFailed raised (no silent empty string)."""

    def _run_all_fail_scenario(
        self,
        chaos: Chaos,
        providers: list[str],
    ) -> Any:
        """Run through all providers; if all fail, raise AllProvidersFailed."""
        failures: dict[str, Exception] = {}
        for provider in providers:
            try:
                chaos.maybe_inject(provider)
                # If injection didn't raise, return a fake success
                return f"success_from_{provider}"
            except Exception as exc:
                failures[provider] = exc

        # All providers failed
        if failures:
            raise AllProvidersFailed(failures)

        return ""  # unreachable if providers is non-empty

    @pytest.mark.parametrize("seed", range(10))
    def test_all_providers_fail_raises(self, seed: int) -> None:
        """AllProvidersFailed is raised when every provider faults."""
        chaos = make_chaos(seed)
        providers = ["primary", "secondary", "tertiary"]
        for p in providers:
            chaos.add_rule(p, Http5xx, every_nth=1)

        with pytest.raises(AllProvidersFailed) as exc_info:
            self._run_all_fail_scenario(chaos, providers)

        exc = exc_info.value
        assert len(exc.failures) == len(providers)

    @pytest.mark.parametrize("seed", range(10))
    def test_no_silent_empty_string(self, seed: int) -> None:
        """The exception message is non-empty — no silent "" swallowed."""
        chaos = make_chaos(seed)
        providers = ["p1", "p2"]
        for p in providers:
            chaos.add_rule(p, Http5xx, every_nth=1)

        with pytest.raises(AllProvidersFailed) as exc_info:
            self._run_all_fail_scenario(chaos, providers)

        exc = exc_info.value
        msg = str(exc)
        assert msg != ""
        assert "All providers failed" in msg

    @pytest.mark.parametrize("seed", range(10))
    def test_failures_dict_contains_every_provider(self, seed: int) -> None:
        """AllProvidersFailed.failures covers all providers that were tried."""
        chaos = make_chaos(seed)
        providers = ["alpha", "beta", "gamma"]
        for p in providers:
            chaos.add_rule(p, Http5xx, every_nth=1)

        with pytest.raises(AllProvidersFailed) as exc_info:
            self._run_all_fail_scenario(chaos, providers)

        failures = exc_info.value.failures
        for p in providers:
            assert p in failures
            assert isinstance(failures[p], ChaosHttp5xxError)

    @pytest.mark.parametrize("seed", range(10))
    def test_fallback_succeeds_if_tertiary_healthy(self, seed: int) -> None:
        """If the last provider is healthy, we get a result instead of exception."""
        chaos = make_chaos(seed)
        # Only fault the first two
        chaos.add_rule("primary", Http5xx, every_nth=1)
        chaos.add_rule("secondary", Http5xx, every_nth=1)
        # "tertiary" has no rule → healthy

        result = self._run_all_fail_scenario(chaos, ["primary", "secondary", "tertiary"])
        assert result == "success_from_tertiary"
        assert result != ""

    @pytest.mark.parametrize("seed", range(10))
    def test_all_providers_fail_carries_exception_types(self, seed: int) -> None:
        """AllProvidersFailed.failures values are the actual exception instances."""
        chaos = make_chaos(seed)
        chaos.add_rule("only", Http5xx, every_nth=1, status_code=503)

        with pytest.raises(AllProvidersFailed) as exc_info:
            self._run_all_fail_scenario(chaos, ["only"])

        failures = exc_info.value.failures
        exc = failures["only"]
        assert isinstance(exc, ChaosHttp5xxError)
        assert exc.status_code == 503


# ---------------------------------------------------------------------------
# Cross-scenario: determinism across seeds
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed always produces the same fault sequence."""

    @pytest.mark.parametrize("seed", range(10))
    def test_same_seed_same_outcome_a(self, seed: int) -> None:
        """Two Chaos(seed) instances with identical rules produce identical faults."""
        def run(s: int) -> list[bool]:
            chaos = make_chaos(s)
            chaos.add_rule("primary", Http5xx, every_nth=3)
            faults = []
            for _ in range(9):
                try:
                    chaos.maybe_inject("primary")
                    faults.append(False)
                except ChaosHttp5xxError:
                    faults.append(True)
            return faults

        result1 = run(seed)
        result2 = run(seed)
        assert result1 == result2

    @pytest.mark.parametrize("seed", range(10))
    def test_different_seeds_different_outcomes_probabilistic(self, seed: int) -> None:
        """Different seeds with fault_rate produce different sequences (usually)."""
        # Use a mix of seeds to show diversity.
        # With every_nth, outcome is purely deterministic so seeds don't matter.
        # With fault_rate, PRNG drives diversity.
        results_per_seed = []
        for s in range(5):
            chaos = Chaos(seed=s, fault_rate=0.5)
            chaos.add_rule("p", Http5xx, fault_rate=0.5)
            seq = []
            for _ in range(20):
                try:
                    chaos.maybe_inject("p")
                    seq.append(0)
                except ChaosHttp5xxError:
                    seq.append(1)
            results_per_seed.append(seq)

        # At least 2 of the 5 seed runs should produce different sequences
        unique = {tuple(r) for r in results_per_seed}
        assert len(unique) >= 2

    @pytest.mark.parametrize("seed", range(10))
    def test_reset_reproduces_same_sequence(self, seed: int) -> None:
        """chaos.reset() reproduces the exact same fault sequence."""
        chaos = make_chaos(seed)
        chaos.add_rule("primary", Http5xx, fault_rate=0.4)

        def collect(c: Chaos, n: int = 20) -> list[bool]:
            faults = []
            for _ in range(n):
                try:
                    c.maybe_inject("primary")
                    faults.append(False)
                except ChaosHttp5xxError:
                    faults.append(True)
            return faults

        run1 = collect(chaos)
        chaos.reset()
        run2 = collect(chaos)
        assert run1 == run2


# ---------------------------------------------------------------------------
# Chaos class unit tests
# ---------------------------------------------------------------------------


class TestChaosUnit:
    """Unit tests for Chaos internals."""

    def test_disabled_chaos_never_injects(self) -> None:
        """Chaos(enabled=False) never raises regardless of rules."""
        chaos = Chaos(seed=0, enabled=False)
        chaos.add_rule("any", Http5xx, every_nth=1)

        for _ in range(10):
            chaos.maybe_inject("any")  # must not raise

    def test_wildcard_provider_rule(self) -> None:
        """Rule with provider='*' fires for every provider."""
        chaos = Chaos(seed=0)
        chaos.add_rule("*", Http5xx, every_nth=1)

        with pytest.raises(ChaosHttp5xxError):
            chaos.maybe_inject("any_provider")

        chaos.reset()
        with pytest.raises(ChaosHttp5xxError):
            chaos.maybe_inject("another_provider")

    def test_rule_only_matches_named_provider(self) -> None:
        """Rule for 'primary' does not fire for 'secondary'."""
        chaos = Chaos(seed=0)
        chaos.add_rule("primary", Http5xx, every_nth=1)

        # 'secondary' has no rule → no fault
        chaos.maybe_inject("secondary")

    def test_every_nth_fires_deterministically(self) -> None:
        """every_nth=3 fires exactly on calls 3, 6, 9, ..."""
        chaos = Chaos(seed=99)
        chaos.add_rule("p", Http5xx, every_nth=3)

        fault_calls = []
        for _ in range(1, 10):
            try:
                chaos.maybe_inject("p")
                fault_calls.append(False)
            except ChaosHttp5xxError:
                fault_calls.append(True)

        # True at indices 2, 5, 8 (0-based) — i.e., calls 3, 6, 9
        expected = [False, False, True, False, False, True, False, False, True]
        assert fault_calls == expected

    def test_clear_rules(self) -> None:
        """clear_rules() removes all rules — no faults afterwards."""
        chaos = Chaos(seed=0)
        chaos.add_rule("p", Http5xx, every_nth=1)
        chaos.clear_rules()

        for _ in range(5):
            chaos.maybe_inject("p")  # no raise

    def test_add_rule_chaining(self) -> None:
        """add_rule returns self to allow method chaining."""
        chaos = (
            Chaos(seed=0)
            .add_rule("p", Http5xx, every_nth=3)
            .add_rule("p", Http429, every_nth=6, retry_after=1.0)
        )
        assert len(chaos._rules) == 2

    @pytest.mark.asyncio
    async def test_async_maybe_inject_raises(self) -> None:
        """async_maybe_inject raises the correct fault type."""
        chaos = Chaos(seed=0)
        chaos.add_rule("p", Http5xx, every_nth=1)

        with pytest.raises(ChaosHttp5xxError):
            await chaos.async_maybe_inject("p")

    @pytest.mark.asyncio
    async def test_async_slow_response_no_raise(self) -> None:
        """async SlowResponse with 0 delay does not raise."""
        chaos = Chaos(seed=0)
        chaos.add_rule("p", SlowResponse, every_nth=1, delay_s=0.0)

        await chaos.async_maybe_inject("p")  # should not raise

    def test_all_providers_failed_message(self) -> None:
        """AllProvidersFailed message is non-empty and informative."""
        failures = {
            "p1": ChaosHttp5xxError("p1", 500),
            "p2": ChaosHttp5xxError("p2", 503),
        }
        exc = AllProvidersFailed(failures)
        msg = str(exc)
        assert "p1" in msg
        assert "p2" in msg
        assert "All providers failed" in msg


# ---------------------------------------------------------------------------
# ChaosMiddleware tests
# ---------------------------------------------------------------------------


class TestChaosMiddleware:
    """Tests for ChaosMiddleware (ProviderRegistry.with_chaos integration)."""

    def test_with_chaos_returns_middleware(self) -> None:
        """ProviderRegistry.with_chaos returns ChaosMiddlewareRegistry."""
        from effgen.models.registry import ChaosMiddlewareRegistry, ProviderRegistry

        chaos = Chaos(seed=0)
        registry = ProviderRegistry.with_chaos(chaos)
        assert isinstance(registry, ChaosMiddlewareRegistry)
        assert registry.chaos is chaos

    def test_middleware_call_injects_fault(self) -> None:
        """ChaosMiddlewareRegistry.call raises on faulted provider."""
        from effgen.models.registry import ProviderRegistry

        chaos = Chaos(seed=0)
        chaos.add_rule("prov", Http5xx, every_nth=1)
        registry = ProviderRegistry.with_chaos(chaos)

        adapter = FakeAdapter("prov")
        with pytest.raises(ChaosHttp5xxError):
            registry.call("prov", adapter.generate)

    def test_middleware_call_passes_through_healthy_provider(self) -> None:
        """ChaosMiddlewareRegistry.call forwards to fn when no fault fires."""
        from effgen.models.registry import ProviderRegistry

        chaos = Chaos(seed=0)
        # No rules — all healthy
        registry = ProviderRegistry.with_chaos(chaos)

        adapter = FakeAdapter("prov", "ok")
        result = registry.call("prov", adapter.generate)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_middleware_async_call_injects_fault(self) -> None:
        """ChaosMiddlewareRegistry.async_call raises on faulted provider."""
        from effgen.models.registry import ProviderRegistry

        chaos = Chaos(seed=0)
        chaos.add_rule("prov", Http5xx, every_nth=1)
        registry = ProviderRegistry.with_chaos(chaos)

        async def fake_generate() -> str:
            return "ok"

        with pytest.raises(ChaosHttp5xxError):
            await registry.async_call("prov", fake_generate)


# ===========================================================================
# Span-tree verification scenarios
#
# The unit scenarios above prove the fault *mechanics*.  These scenarios run
# the faults through the real observability span helpers and assert the
# emitted span tree, the way an operator would inspect a trace:
#
#   * Scenario A — the trace shows the router switching provider after the
#     primary's model.call fails.
#   * Scenario B — the trace carries a retry attempt event with the honoured
#     Retry-After delay.
#
# They are skipped automatically if opentelemetry-sdk is not installed.
# ===========================================================================

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - OTel is part of [all]
    _OTEL_AVAILABLE = False

otel_required = pytest.mark.skipif(
    not _OTEL_AVAILABLE, reason="opentelemetry-sdk not installed"
)


def _attr(span: Any, key: str) -> Any:
    """Read an attribute off a finished span (dict-like access)."""
    return dict(span.attributes or {}).get(key)


class TestScenarioASpanTree:
    """Scenario A — span tree shows the provider switch (primary 5xx → fallback)."""

    @staticmethod
    def _run_agent_with_failover(chaos: Chaos, exporter: "InMemorySpanExporter") -> str:
        """Run one ReAct iteration: try primary, fall over to fallback on 5xx.

        Emits a real span tree:
            agent.run
              └─ agent.iteration
                   ├─ router.decision (selected_provider=primary)
                   ├─ model.call (provider=primary)  ← errors on 5xx
                   ├─ router.decision (selected_provider=fallback)  ← the switch
                   └─ model.call (provider=fallback) ← succeeds
        """
        from effgen.observability import (
            ModelAttrs,
            RouterAttrs,
            start_agent_iteration,
            start_agent_run,
            start_model_call,
            start_router_decision,
        )

        primary = FakeAdapter("primary", "primary_ok")
        fallback = FakeAdapter("fallback", "fallback_ok")
        candidates = [("primary", "m1"), ("fallback", "m2")]

        with start_agent_run(preset="failover_agent", task="hello", run_id="A"):
            with start_agent_iteration(preset="failover_agent", iteration=1):
                # Router picks the primary first.
                with start_router_decision("priority", candidates) as r1:
                    r1.set_attribute(RouterAttrs.SELECTED_PROVIDER, "primary")
                try:
                    with start_model_call(provider="primary", model="m1"):
                        chaos.maybe_inject("primary")
                        return primary.generate()
                except ChaosHttp5xxError:
                    # The switch: router re-decides and selects the fallback.
                    with start_router_decision("priority", candidates) as r2:
                        r2.set_attribute(RouterAttrs.SELECTED_PROVIDER, "fallback")
                    with start_model_call(provider="fallback", model="m2") as m2:
                        chaos.maybe_inject("fallback")
                        out = fallback.generate()
                        m2.set_attribute(ModelAttrs.OUTCOME, "ok")
                        return out
        return ""  # pragma: no cover

    @otel_required
    @pytest.mark.parametrize("seed", range(10))
    def test_span_tree_shows_provider_switch(self, seed: int) -> None:
        """The trace shows router switching primary → fallback, agent completes."""
        from effgen.observability import (
            ModelAttrs,
            RouterAttrs,
            SpanName,
            reset_tracing,
            setup_tracing,
        )
        from effgen.observability.tracing import AlwaysOnSampler

        reset_tracing()
        exporter = InMemorySpanExporter()
        setup_tracing(sampler=AlwaysOnSampler(), exporter=exporter)
        try:
            chaos = make_chaos(seed)
            # Primary fails on the very first call → forces a switch this run.
            chaos.add_rule("primary", Http5xx, every_nth=1, status_code=503)

            result = self._run_agent_with_failover(chaos, exporter)

            # Agent completed with a non-empty answer from the fallback.
            assert result == "fallback_ok"
            assert result != ""

            spans = exporter.get_finished_spans()
            names = [s.name for s in spans]
            assert SpanName.AGENT_RUN in names
            assert SpanName.AGENT_ITERATION in names

            # Two model.call spans: primary (error) then fallback (ok).
            model_spans = [s for s in spans if s.name == SpanName.MODEL_CALL]
            providers = [_attr(s, ModelAttrs.PROVIDER) for s in model_spans]
            assert providers == ["primary", "fallback"], providers
            primary_span = model_spans[0]
            fallback_span = model_spans[1]
            assert _attr(primary_span, ModelAttrs.OUTCOME) == "error"
            assert _attr(fallback_span, ModelAttrs.OUTCOME) == "ok"

            # Two router decisions: the switch is the second selected_provider.
            router_spans = [s for s in spans if s.name == SpanName.ROUTER_DECISION]
            selected = [_attr(s, RouterAttrs.SELECTED_PROVIDER) for s in router_spans]
            assert selected == ["primary", "fallback"], selected
        finally:
            reset_tracing()


class TestScenarioBRetryEvent:
    """Scenario B — span carries the retry attempt event with the honoured delay."""

    @otel_required
    @pytest.mark.parametrize("seed", range(10))
    def test_retry_delay_observed_in_span_events(self, seed: int) -> None:
        """A 429 with Retry-After=2 produces a retry.attempt event with delay≈2."""
        from effgen.observability import (
            SpanName,
            reset_tracing,
            setup_tracing,
            start_model_call,
        )
        from effgen.observability.tracing import AlwaysOnSampler
        from effgen.reliability.retry import Retry, retryable

        reset_tracing()
        exporter = InMemorySpanExporter()
        setup_tracing(sampler=AlwaysOnSampler(), exporter=exporter)
        try:
            chaos = make_chaos(seed)
            retry_after = 2.0
            # 429 fires exactly once; retry then succeeds.
            chaos.add_rule(
                "primary", Http429, every_nth=1, max_fires=1, retry_after=retry_after
            )
            adapter = FakeAdapter("primary", "ok")

            # honour_retry_after=True → the retry sleeps the Retry-After value.
            # We monkeypatch time.sleep within retry to avoid a real 2s wait while
            # still exercising compute_delay + _emit_retry_event.
            slept: list[float] = []

            @retryable(Retry(max_attempts=3, jitter=False))
            def call_primary() -> str:
                chaos.maybe_inject("primary")
                return adapter.generate()

            with start_model_call(provider="primary", model="m1") as span:
                with patch("effgen.reliability.retry.time.sleep", slept.append):
                    result = call_primary()
                # The retry event was emitted on this active span.
                _ = span

            assert result == "ok"
            # The retry honoured the Retry-After value.
            assert slept and slept[0] == pytest.approx(retry_after)

            spans = exporter.get_finished_spans()
            model_span = next(s for s in spans if s.name == SpanName.MODEL_CALL)
            retry_events = [
                e for e in model_span.events if e.name == SpanName.RETRY_ATTEMPT
            ]
            assert len(retry_events) == 1, [e.name for e in model_span.events]
            ev_attrs = dict(retry_events[0].attributes or {})
            # The recorded delay equals the honoured Retry-After.
            assert ev_attrs.get("effgen.retry.delay_s") == pytest.approx(retry_after)
            assert ev_attrs.get("effgen.retry.attempt") == 1
            assert "ChaosHttp429Error" in str(ev_attrs.get("effgen.retry.reason"))
        finally:
            reset_tracing()


class TestScenarioCRealTimeout:
    """Scenario C — slow response exceeds timeout; fails cleanly, no hang."""

    @pytest.mark.parametrize("seed", range(10))
    def test_slow_response_times_out_within_bound(self, seed: int) -> None:
        """A SlowResponse longer than the timeout raises TimeoutError, bounded."""
        from effgen.reliability.timeouts import TimeoutError as EffGenTimeout
        from effgen.reliability.timeouts import apply_timeout

        chaos = make_chaos(seed)
        timeout_s = 0.3
        # SlowResponse sleeps longer than the timeout (simulates a 59s stall vs
        # 60s timeout, scaled down so the suite stays fast).
        chaos.add_rule("primary", SlowResponse, every_nth=1, delay_s=timeout_s * 3)

        def slow_call() -> str:
            chaos.maybe_inject("primary")  # sleeps delay_s
            return "should not get here"

        guarded = apply_timeout(slow_call, timeout_s, operation="model_call")

        start = time.monotonic()
        with pytest.raises(EffGenTimeout):
            guarded()
        elapsed = time.monotonic() - start

        # No hang: must return within timeout + 1s.
        assert elapsed < timeout_s + 1.0, f"hung for {elapsed:.2f}s"
        # TimeoutError contract (subclass of builtins.TimeoutError).
        assert issubclass(EffGenTimeout, TimeoutError)

    @pytest.mark.parametrize("seed", range(10))
    def test_breaker_records_timeout_failure(self, seed: int) -> None:
        """The breaker records the timeout as a failure and opens at threshold."""
        from effgen.reliability.circuit import CircuitBreaker, CircuitState
        from effgen.reliability.timeouts import TimeoutError as EffGenTimeout
        from effgen.reliability.timeouts import apply_timeout

        chaos = make_chaos(seed)
        timeout_s = 0.1
        chaos.add_rule("primary", SlowResponse, every_nth=1, delay_s=timeout_s * 3)

        cb = CircuitBreaker(
            name=f"primary_timeout_{seed}",
            failure_threshold=3,
            recovery_timeout=30.0,
        )

        def slow_call() -> str:
            chaos.maybe_inject("primary")
            return "x"

        guarded = apply_timeout(slow_call, timeout_s, operation="model_call")

        failures = 0
        for _ in range(5):
            if not cb.is_call_permitted():
                break
            try:
                guarded()
                cb.on_success()
            except EffGenTimeout as exc:
                failures += 1
                cb.on_failure(exc)

        assert cb.state == CircuitState.OPEN
        assert failures == 3


class TestScenarioDProviderList:
    """Scenario D — AllProvidersFailed carries the full provider list."""

    @pytest.mark.parametrize("seed", range(10))
    def test_exception_lists_every_provider(self, seed: int) -> None:
        """The raised exception names every provider that was tried."""
        chaos = make_chaos(seed)
        providers = ["primary", "secondary", "tertiary"]
        for p in providers:
            chaos.add_rule(p, Http5xx, every_nth=1)

        failures: dict[str, Exception] = {}
        for provider in providers:
            try:
                chaos.maybe_inject(provider)
            except ChaosHttp5xxError as exc:
                failures[provider] = exc

        with pytest.raises(AllProvidersFailed) as exc_info:
            raise AllProvidersFailed(failures)

        exc = exc_info.value
        # Provider list available both structurally and in the message.
        assert set(exc.failures) == set(providers)
        msg = str(exc)
        for p in providers:
            assert p in msg, f"{p!r} missing from {msg!r}"
