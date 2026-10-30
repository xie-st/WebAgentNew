"""
Deterministic chaos / fault-injection harness for effGen.

Attach a :class:`Chaos` instance to a provider call-chain to simulate
realistic failure modes in a reproducible way.  Because the PRNG is
seeded once at construction time, the same seed always produces the same
sequence of faults — making tests deterministic across reruns.

Fault types
-----------
* :class:`NetworkTimeout`   — simulates a stalled TCP connection (raises
  :class:`~effgen.reliability.timeouts.TimeoutError`).
* :class:`Http5xx`          — provider returns an HTTP 5xx error.
* :class:`Http429`          — provider enforces rate-limiting; carries an
  optional ``retry_after`` hint in seconds.
* :class:`SlowResponse`     — provider responds normally but after an
  artificial delay.
* :class:`PartialResponse`  — provider returns a truncated / cut-off response
  body.
* :class:`MalformedJSON`    — provider returns HTTP 200 but the body is not
  valid JSON.

Integration
-----------

.. code-block:: python

    from effgen.reliability.chaos import Chaos, ChaosMiddleware
    from effgen.models.registry import ProviderRegistry

    chaos = Chaos(seed=42, fault_rate=0.33)
    chaos.add_rule("primary_provider", Http5xx, every_nth=3)

    # Wrap any callable with chaos-injected faults:
    result = chaos.wrap_call(adapter.generate, provider="primary", **kwargs)

    # Or attach at the ProviderRegistry level so every call goes through it:
    registry_with_chaos = ProviderRegistry.with_chaos(chaos)

Usage in tests
--------------

.. code-block:: python

    import pytest
    from effgen.reliability.chaos import Chaos, Http5xx, Http429, NetworkTimeout

    @pytest.mark.reliability
    @pytest.mark.parametrize("seed", range(10))
    def test_primary_5xx_fallback(seed):
        chaos = Chaos(seed=seed)
        chaos.add_rule("primary", Http5xx, every_nth=3)
        # ... run scenario ...
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    # Fault types
    "FaultBase",
    "NetworkTimeout",
    "Http5xx",
    "Http429",
    "SlowResponse",
    "PartialResponse",
    "MalformedJSON",
    # Core classes
    "Chaos",
    "ChaosRule",
    "ChaosMiddleware",
    # Exceptions raised by chaos faults
    "ChaosNetworkTimeout",
    "ChaosHttp5xxError",
    "ChaosHttp429Error",
    "ChaosPartialResponseError",
    "ChaosMalformedJSONError",
    # Registry integration
    "AllProvidersFailed",
]


# ---------------------------------------------------------------------------
# Chaos-specific exception hierarchy
# ---------------------------------------------------------------------------


class ChaosNetworkTimeout(TimeoutError):
    """Raised when chaos injects a :class:`NetworkTimeout` fault.

    Attributes:
        provider: Provider name that was targeted.
        operation: Name of the operation that timed out.
        limit: Configured timeout limit in seconds.
    """

    def __init__(
        self,
        provider: str = "",
        operation: str = "model_call",
        limit: float = 60.0,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.limit = limit
        super().__init__(
            f"chaos: network timeout on provider={provider!r} "
            f"operation={operation!r} (>{limit:.1f}s)"
        )


class ChaosHttp5xxError(Exception):
    """Raised when chaos injects an :class:`Http5xx` fault.

    Carries ``status_code`` so that :func:`~effgen.reliability.retry.is_transient_error`
    and the circuit-breaker see it as a retryable 5xx failure.

    Attributes:
        status_code: The injected HTTP status code (default 500).
        provider: Provider name targeted by this fault.
    """

    def __init__(
        self,
        provider: str = "",
        status_code: int = 500,
        message: str = "",
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        msg = message or f"chaos: HTTP {status_code} from provider={provider!r}"
        super().__init__(msg)


class ChaosHttp429Error(Exception):
    """Raised when chaos injects an :class:`Http429` fault.

    Attributes:
        retry_after: How long the caller should wait before retrying (seconds).
        provider: Provider name targeted by this fault.
        status_code: Always 429.
    """

    def __init__(
        self,
        provider: str = "",
        retry_after: float = 2.0,
    ) -> None:
        self.provider = provider
        self.retry_after = retry_after
        self.status_code = 429
        super().__init__(
            f"chaos: HTTP 429 from provider={provider!r} "
            f"retry_after={retry_after:.1f}s"
        )


class ChaosPartialResponseError(Exception):
    """Raised when chaos injects a :class:`PartialResponse` fault.

    Attributes:
        provider: Provider name targeted by this fault.
        partial_content: The truncated content returned so far.
    """

    def __init__(self, provider: str = "", partial_content: str = "") -> None:
        self.provider = provider
        self.partial_content = partial_content
        super().__init__(
            f"chaos: partial/truncated response from provider={provider!r}"
        )


class ChaosMalformedJSONError(ValueError):
    """Raised when chaos injects a :class:`MalformedJSON` fault.

    Attributes:
        provider: Provider name targeted by this fault.
        raw: The malformed content that was returned.
    """

    def __init__(self, provider: str = "", raw: str = "") -> None:
        self.provider = provider
        self.raw = raw
        super().__init__(
            f"chaos: malformed JSON from provider={provider!r}: {raw!r}"
        )


class AllProvidersFailed(Exception):
    """Raised by chaos-aware agent harness when every provider has failed.

    Unlike the router's ``AllCandidatesExhaustedError``, this exception is
    raised explicitly by the chaos scenario to signal a total blackout.  It
    must never result in a silent empty-string response.

    Attributes:
        failures: Mapping of ``provider_name → last exception``.
    """

    def __init__(self, failures: dict[str, Exception]) -> None:
        self.failures = failures
        summary = "; ".join(
            f"{p}: {type(e).__name__}({e})" for p, e in failures.items()
        )
        super().__init__(
            f"All providers failed — no successful response available. "
            f"Failures: [{summary}]"
        )


# ---------------------------------------------------------------------------
# Fault type descriptors  (sentinel classes, not instances)
# ---------------------------------------------------------------------------


class FaultBase:
    """Abstract base for fault-type sentinels."""


class NetworkTimeout(FaultBase):
    """Simulates a stalled TCP connection that hits the timeout."""


class Http5xx(FaultBase):
    """Simulates an HTTP 5xx error from the provider (default: 500)."""


class Http429(FaultBase):
    """Simulates HTTP 429 rate-limiting with a Retry-After header."""


class SlowResponse(FaultBase):
    """Simulates a valid response that arrives after an artificial delay."""


class PartialResponse(FaultBase):
    """Simulates a truncated / cut-off response body."""


class MalformedJSON(FaultBase):
    """Simulates an HTTP 200 with invalid JSON body."""


# ---------------------------------------------------------------------------
# Rule: how often a fault fires for a specific provider
# ---------------------------------------------------------------------------


@dataclass
class ChaosRule:
    """Fault injection rule for one provider.

    Attributes:
        provider:    Provider name this rule applies to.  ``"*"`` matches all.
        fault_type:  The fault class to inject (e.g. :class:`Http5xx`).
        every_nth:   If set, inject the fault on every *nth* call (1-based).
                     ``every_nth=3`` means calls 3, 6, 9, … get the fault.
        fault_rate:  Probability (0.0–1.0) that any given call gets the fault.
                     Used when ``every_nth`` is not set.
        max_fires:   Maximum number of times this rule may fire in total.
                     ``None`` means unlimited.  Use ``max_fires=1`` to inject
                     a fault exactly once (e.g. for retry scenarios).
        params:      Extra keyword arguments forwarded to the fault exception.
                     E.g. ``{"status_code": 503}`` for :class:`Http5xx` or
                     ``{"retry_after": 2.0}`` for :class:`Http429`.
    """

    provider: str
    fault_type: type[FaultBase]
    every_nth: int | None = None
    fault_rate: float = 0.0
    max_fires: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    # Internal call counter per rule instance (reset with Chaos.reset())
    _call_count: int = field(default=0, init=False, repr=False, compare=False)
    _fire_count: int = field(default=0, init=False, repr=False, compare=False)

    def should_fire(self, rng: random.Random) -> bool:
        """Decide whether this rule fires on the current call.

        Increments the internal call counter.  Deterministic when ``every_nth``
        is set; probabilistic when ``fault_rate`` is used.

        Respects ``max_fires``: once the rule has fired that many times, it
        always returns ``False``.
        """
        # Check max_fires cap first
        if self.max_fires is not None and self._fire_count >= self.max_fires:
            self._call_count += 1
            return False

        self._call_count += 1
        if self.every_nth is not None:
            result = self._call_count % self.every_nth == 0
        else:
            result = rng.random() < self.fault_rate

        if result:
            self._fire_count += 1
        return result

    def reset(self) -> None:
        """Reset the internal call counter and fire count."""
        self._call_count = 0
        self._fire_count = 0


# ---------------------------------------------------------------------------
# Core Chaos class
# ---------------------------------------------------------------------------


class Chaos:
    """Deterministic fault injector.

    Construct once per test scenario with a fixed *seed*.  The same seed
    always produces the same sequence of faults, making tests reproducible
    across reruns and CI runs.

    Args:
        seed:       Integer seed for the internal PRNG.
        fault_rate: Default probability (0.0–1.0) that a call is faulted when
                    no ``every_nth`` is specified.  Individual rules can override.
        enabled:    Master on/off switch.  When ``False``, :meth:`maybe_inject`
                    always returns without fault (useful for quick local runs).

    Example::

        chaos = Chaos(seed=42)
        chaos.add_rule("primary", Http5xx, every_nth=3)
        chaos.add_rule("primary", Http429, fault_rate=0.1, params={"retry_after": 2.0})

        # Later, in your test harness:
        chaos.maybe_inject("primary")  # raises ChaosHttp5xxError on 3rd call
    """

    def __init__(
        self,
        seed: int = 0,
        fault_rate: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.seed = seed
        self.fault_rate = fault_rate
        self.enabled = enabled
        self._rng = random.Random(seed)
        self._rules: list[ChaosRule] = []
        # Per-provider call counters used for every_nth without a rule
        self._provider_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        provider: str,
        fault_type: type[FaultBase],
        *,
        every_nth: int | None = None,
        fault_rate: float | None = None,
        max_fires: int | None = None,
        **params: Any,
    ) -> "Chaos":
        """Add a fault-injection rule.

        Args:
            provider:   Target provider name.  Use ``"*"`` for all providers.
            fault_type: Fault class to inject (e.g. :class:`Http5xx`).
            every_nth:  Fire on every *nth* call to this provider.
            fault_rate: Probability to fire (overrides instance default if set).
            max_fires:  Maximum number of times this rule may fire (``None`` = unlimited).
                        Use ``max_fires=1`` to inject exactly one fault (ideal for retry tests).
            **params:   Extra keyword arguments forwarded to the fault constructor.

        Returns:
            ``self`` for method chaining.
        """
        rule = ChaosRule(
            provider=provider,
            fault_type=fault_type,
            every_nth=every_nth,
            fault_rate=fault_rate if fault_rate is not None else self.fault_rate,
            max_fires=max_fires,
            params=dict(params),
        )
        self._rules.append(rule)
        return self

    def clear_rules(self) -> None:
        """Remove all registered rules."""
        self._rules.clear()

    def reset(self) -> None:
        """Reset all rule counters and re-seed the PRNG.

        Call this between test scenarios to ensure a clean slate while
        preserving the original seed.
        """
        self._rng = random.Random(self.seed)
        for rule in self._rules:
            rule.reset()
        self._provider_counters.clear()

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def maybe_inject(self, provider: str, *, operation: str = "model_call") -> None:
        """Check rules and raise a fault exception if one fires.

        This is the primary hook called before / instead of a real provider
        call.  Calling it is a no-op when :attr:`enabled` is ``False`` or no
        rule matches.

        Args:
            provider:  The provider name about to be called.
            operation: Human-readable operation name (e.g. ``"generate"``).

        Raises:
            :class:`ChaosNetworkTimeout`: When a :class:`NetworkTimeout` rule fires.
            :class:`ChaosHttp5xxError`: When an :class:`Http5xx` rule fires.
            :class:`ChaosHttp429Error`: When an :class:`Http429` rule fires.
            :class:`ChaosPartialResponseError`: When a :class:`PartialResponse` rule fires.
            :class:`ChaosMalformedJSONError`: When a :class:`MalformedJSON` rule fires.
            ``RuntimeError``: When a :class:`SlowResponse` rule fires — the delay
                is applied synchronously here; for async code use :meth:`async_maybe_inject`.
        """
        if not self.enabled:
            return

        for rule in self._rules:
            if rule.provider not in (provider, "*"):
                continue
            if not rule.should_fire(self._rng):
                continue
            self._raise_fault(rule.fault_type, provider, rule.params)

    async def async_maybe_inject(
        self, provider: str, *, operation: str = "model_call"
    ) -> None:
        """Async version of :meth:`maybe_inject`.

        Same semantics as :meth:`maybe_inject` but uses ``asyncio.sleep`` for
        :class:`SlowResponse` faults instead of ``time.sleep``.

        Args:
            provider:  The provider name about to be called.
            operation: Human-readable operation name.

        Raises:
            Same exceptions as :meth:`maybe_inject`.
        """
        if not self.enabled:
            return

        for rule in self._rules:
            if rule.provider not in (provider, "*"):
                continue
            if not rule.should_fire(self._rng):
                continue
            await self._async_raise_fault(rule.fault_type, provider, rule.params)

    def _raise_fault(
        self,
        fault_type: type[FaultBase],
        provider: str,
        params: dict[str, Any],
    ) -> None:
        """Materialise a fault exception (sync path)."""
        if issubclass(fault_type, NetworkTimeout):
            limit = float(params.get("limit", 60.0))
            raise ChaosNetworkTimeout(provider=provider, limit=limit)

        if issubclass(fault_type, Http5xx):
            status_code = int(params.get("status_code", 500))
            raise ChaosHttp5xxError(provider=provider, status_code=status_code)

        if issubclass(fault_type, Http429):
            retry_after = float(params.get("retry_after", 2.0))
            raise ChaosHttp429Error(provider=provider, retry_after=retry_after)

        if issubclass(fault_type, SlowResponse):
            delay_s = float(params.get("delay_s", 1.0))
            time.sleep(delay_s)
            return  # not an error — just slow

        if issubclass(fault_type, PartialResponse):
            partial = str(params.get("partial_content", "...truncated..."))
            raise ChaosPartialResponseError(
                provider=provider, partial_content=partial
            )

        if issubclass(fault_type, MalformedJSON):
            raw = str(params.get("raw", "{invalid json}"))
            raise ChaosMalformedJSONError(provider=provider, raw=raw)

    async def _async_raise_fault(
        self,
        fault_type: type[FaultBase],
        provider: str,
        params: dict[str, Any],
    ) -> None:
        """Materialise a fault exception (async path)."""
        if issubclass(fault_type, SlowResponse):
            delay_s = float(params.get("delay_s", 1.0))
            await asyncio.sleep(delay_s)
            return  # not an error — just slow

        # All other faults are sync-compatible — delegate
        self._raise_fault(fault_type, provider, params)

    # ------------------------------------------------------------------
    # Callable wrapper helpers
    # ------------------------------------------------------------------

    def wrap_call(
        self,
        fn: Callable,
        *,
        provider: str,
        operation: str = "model_call",
        **kwargs: Any,
    ) -> Any:
        """Call *fn* after running fault injection for *provider*.

        If a fault fires, the wrapped callable is never called.

        Args:
            fn:        The callable to protect / fault-inject.
            provider:  Provider name used for rule matching.
            operation: Human-readable operation label for diagnostics.
            **kwargs:  Forwarded verbatim to *fn*.

        Returns:
            The return value of *fn* if no fault fires.

        Raises:
            Any chaos exception (see :meth:`maybe_inject`).
        """
        self.maybe_inject(provider, operation=operation)
        return fn(**kwargs)

    async def async_wrap_call(
        self,
        fn: Callable,
        *,
        provider: str,
        operation: str = "model_call",
        **kwargs: Any,
    ) -> Any:
        """Async version of :meth:`wrap_call`.

        Args:
            fn:        An async callable.
            provider:  Provider name used for rule matching.
            operation: Human-readable operation label.
            **kwargs:  Forwarded to *fn*.

        Returns:
            The awaited return value of *fn* if no fault fires.
        """
        await self.async_maybe_inject(provider, operation=operation)
        return await fn(**kwargs)

    def __repr__(self) -> str:
        return (
            f"Chaos(seed={self.seed!r}, enabled={self.enabled!r}, "
            f"rules={len(self._rules)})"
        )


# ---------------------------------------------------------------------------
# ChaosMiddleware — thin wrapper around ProviderRegistry
# ---------------------------------------------------------------------------


class ChaosMiddleware:
    """Wraps a provider's generate call with fault injection.

    This is what :meth:`ProviderRegistry.with_chaos` returns.  It intercepts
    adapter calls and routes them through :class:`Chaos` before forwarding to
    the real adapter.

    Args:
        chaos:    The :class:`Chaos` instance that carries the PRNG state and rules.
        adapters: Mapping of ``provider_name → adapter_instance``.

    Example::

        middleware = ChaosMiddleware(chaos, {"primary": primary_adapter})
        result = middleware.call("primary", prompt="Hello")
    """

    def __init__(
        self,
        chaos: Chaos,
        adapters: dict[str, Any] | None = None,
    ) -> None:
        self.chaos = chaos
        self.adapters: dict[str, Any] = adapters or {}

    def register(self, provider: str, adapter: Any) -> None:
        """Register an adapter under *provider*."""
        self.adapters[provider] = adapter

    def call(self, provider: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run fault injection then forward to *fn* if no fault fires.

        Args:
            provider: Provider being called (used for rule matching).
            fn:       The real adapter callable.
            *args:    Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            Return value of *fn*.
        """
        self.chaos.maybe_inject(provider)
        return fn(*args, **kwargs)

    async def async_call(
        self, provider: str, fn: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Async version of :meth:`call`.

        Args:
            provider: Provider being called.
            fn:       Async callable.
            *args:    Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Awaited return value of *fn*.
        """
        await self.chaos.async_maybe_inject(provider)
        return await fn(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ChaosMiddleware(chaos={self.chaos!r}, "
            f"providers={list(self.adapters.keys())!r})"
        )
