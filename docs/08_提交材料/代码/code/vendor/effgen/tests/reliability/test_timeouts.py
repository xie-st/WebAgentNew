"""
Tests for effGen timeout primitives.

Coverage:
- with_timeout() raises TimeoutError on overrun (sync)
- async_timeout() raises TimeoutError on overrun (async)
- apply_timeout() wraps sync and async callables
- make_httpx_timeout() returns a proper Timeout object
- audit_no_none_timeouts() detects timeout=None in source
- No timeout=None appears in effgen source tree
- TimeoutError carries operation name and limit
- ValueError on bad (None / ≤0) timeout values
"""

from __future__ import annotations

import asyncio
import time

import pytest

from effgen.reliability.timeouts import (
    TimeoutError as EffGenTimeout,
)
from effgen.reliability.timeouts import (
    apply_timeout,
    async_timeout,
    audit_no_none_timeouts,
    make_httpx_timeout,
    with_timeout,
)

pytestmark = pytest.mark.reliability


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sleep_and_return(seconds: float, value: str = "ok") -> str:
    time.sleep(seconds)
    return value


async def _async_sleep_and_return(seconds: float, value: str = "ok") -> str:
    await asyncio.sleep(seconds)
    return value


# ---------------------------------------------------------------------------
# Sync with_timeout
# ---------------------------------------------------------------------------


class TestSyncTimeout:
    def test_completes_within_limit(self):
        """Fast operation should not raise."""
        with with_timeout(5.0, "test_op"):
            result = _sleep_and_return(0.01, "done")
        assert result == "done"

    def test_raises_on_overrun(self):
        """Operation exceeding limit must raise EffGenTimeout."""
        with pytest.raises(EffGenTimeout) as exc_info:
            with with_timeout(0.1, "slow_op"):
                _sleep_and_return(10.0)
        err = exc_info.value
        assert err.operation == "slow_op"
        assert err.limit == pytest.approx(0.1)
        assert "slow_op" in str(err)
        assert "0.1" in str(err)

    def test_error_is_timeout_subclass(self):
        """EffGenTimeout must be a subclass of built-in TimeoutError."""
        assert issubclass(EffGenTimeout, TimeoutError)

    def test_invalid_timeout_zero(self):
        with pytest.raises(ValueError, match="positive float"):
            with with_timeout(0.0, "op"):
                pass  # pragma: no cover

    def test_invalid_timeout_negative(self):
        with pytest.raises(ValueError, match="positive float"):
            with with_timeout(-1.0, "op"):
                pass  # pragma: no cover

    def test_invalid_timeout_none(self):
        with pytest.raises(ValueError):
            with with_timeout(None, "op"):  # type: ignore[arg-type]
                pass  # pragma: no cover

    def test_no_error_on_exception_within_limit(self):
        """Non-timeout exceptions should propagate unchanged."""
        with pytest.raises(ValueError, match="inner"):
            with with_timeout(5.0, "op"):
                raise ValueError("inner")

    def test_default_operation_label(self):
        """Default operation label is 'operation'."""
        with pytest.raises(EffGenTimeout) as exc_info:
            with with_timeout(0.05):
                _sleep_and_return(10.0)
        assert exc_info.value.operation == "operation"

    def test_survives_one_swallowed_interrupt(self):
        """A single absorbed interrupt must not let the call run unbounded.

        Simulates what a cloud SDK's own internal retry loop does: catch the
        first raised TimeoutError (broadly, the way a network-error handler
        would) and keep going. The periodic re-arm must still bound the
        total overrun to about one extra tick, not let the call run to its
        natural (much longer) completion.
        """
        swallowed = {"count": 0}

        def _flaky_long_call():
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    time.sleep(0.05)
                except EffGenTimeout:
                    # Absorb exactly the first interrupt, like an SDK's own
                    # broad except-and-retry around a blocking call.
                    if swallowed["count"] == 0:
                        swallowed["count"] += 1
                        continue
                    raise
            return "ran to completion"  # pragma: no cover - should not happen

        t0 = time.monotonic()
        with pytest.raises(EffGenTimeout):
            with with_timeout(0.3, "flaky_op"):
                _flaky_long_call()
        elapsed = time.monotonic() - t0

        assert swallowed["count"] == 1, "the simulated swallow never happened"
        # Bounded to a small number of retry ticks past the deadline, not
        # anywhere near the call's natural ~30s completion.
        assert elapsed < 2.0


# ---------------------------------------------------------------------------
# Async async_timeout
# ---------------------------------------------------------------------------


class TestAsyncTimeout:
    @pytest.mark.asyncio
    async def test_completes_within_limit(self):
        async with async_timeout(5.0, "async_op"):
            await _async_sleep_and_return(0.01)

    @pytest.mark.asyncio
    async def test_raises_on_overrun(self):
        with pytest.raises(EffGenTimeout) as exc_info:
            async with async_timeout(0.1, "slow_async"):
                await _async_sleep_and_return(10.0)
        assert exc_info.value.operation == "slow_async"

    @pytest.mark.asyncio
    async def test_invalid_timeout_none_async(self):
        with pytest.raises(ValueError):
            async with async_timeout(None, "op"):  # type: ignore[arg-type]
                pass  # pragma: no cover


# ---------------------------------------------------------------------------
# apply_timeout
# ---------------------------------------------------------------------------


class TestApplyTimeout:
    def test_sync_completes(self):
        wrapped = apply_timeout(_sleep_and_return, 5.0)
        assert wrapped(0.01, "hello") == "hello"

    def test_sync_raises_on_overrun(self):
        wrapped = apply_timeout(_sleep_and_return, 0.1, "wrapped_sync")
        with pytest.raises(EffGenTimeout):
            wrapped(10.0)

    def test_sync_preserves_name(self):
        wrapped = apply_timeout(_sleep_and_return, 5.0)
        assert wrapped.__name__ == "_sleep_and_return"

    @pytest.mark.asyncio
    async def test_async_completes(self):
        wrapped = apply_timeout(_async_sleep_and_return, 5.0)
        assert await wrapped(0.01, "async_hello") == "async_hello"

    @pytest.mark.asyncio
    async def test_async_raises_on_overrun(self):
        wrapped = apply_timeout(_async_sleep_and_return, 0.1)
        with pytest.raises(EffGenTimeout):
            await wrapped(10.0)

    def test_invalid_none_raises_value_error(self):
        with pytest.raises(ValueError):
            apply_timeout(_sleep_and_return, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_httpx_timeout
# ---------------------------------------------------------------------------


class TestMakeHttpxTimeout:
    def test_returns_value(self):
        """Should return something non-None."""
        t = make_httpx_timeout(20.0)
        assert t is not None

    def test_connect_defaults_to_min(self):
        """connect timeout should default to min(http, 10)."""
        t = make_httpx_timeout(30.0)
        # If httpx is available, check the Timeout object; otherwise check dict
        try:
            import httpx
            assert isinstance(t, httpx.Timeout)
            assert t.connect == min(30.0, 10.0)
        except ImportError:
            assert t["connect"] == min(30.0, 10.0)

    def test_explicit_connect(self):
        t = make_httpx_timeout(20.0, connect=3.0)
        try:
            import httpx  # noqa: F401

            assert t.connect == pytest.approx(3.0)
        except ImportError:
            assert t["connect"] == pytest.approx(3.0)

    def test_none_http_raises(self):
        with pytest.raises(ValueError):
            make_httpx_timeout(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Source-tree audit — no timeout=None anywhere in effgen/
# ---------------------------------------------------------------------------


class TestNoNoneTimeouts:
    def test_no_none_timeouts_in_source(self):
        """Fail if any adapter/tool uses timeout=None."""
        import pathlib

        effgen_root = pathlib.Path(__file__).parent.parent.parent / "effgen"
        violations = audit_no_none_timeouts(str(effgen_root))

        # Report all violations at once for easy fixing
        if violations:
            msg = (
                "Found timeout=None in effGen source — this is forbidden.\n"
                "Every I/O boundary must have an explicit timeout.\n"
                "Violations:\n" + "\n".join(f"  {v}" for v in violations)
            )
            pytest.fail(msg)
