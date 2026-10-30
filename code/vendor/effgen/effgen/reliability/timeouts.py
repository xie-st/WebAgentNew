"""
Timeout propagation utilities for effGen.

Every I/O boundary in effGen **must** use an explicit timeout — ``timeout=None``
is forbidden and will cause the CI test ``test_timeouts.py::test_no_none_timeouts``
to fail.

This module provides:

* :class:`TimeoutError` — a subclass of Python's built-in ``TimeoutError`` that
  carries the operation name and limit.
* :func:`with_timeout` — ``async``-aware / threaded context manager that raises
  :class:`TimeoutError` if the wrapped code exceeds the given limit.
* :func:`apply_timeout` — wrap a synchronous or async callable so it raises
  :class:`TimeoutError` on overrun.
* :func:`make_httpx_timeout` — build an ``httpx.Timeout`` object from our config
  so adapters don't guess values.

Usage::

    from effgen.reliability.timeouts import with_timeout, EffGenTimeoutError

    async def fetch_model_response():
        async with with_timeout(60.0, "model_call"):
            return await adapter.generate(...)

    # Sync usage
    with with_timeout(30.0, "tool_call"):
        result = tool.run(...)
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import signal
import threading
from collections.abc import Callable, Iterator
from typing import Any

__all__ = [
    "TimeoutError",
    "with_timeout",
    "apply_timeout",
    "make_httpx_timeout",
]


class TimeoutError(builtins.TimeoutError):  # noqa: A001 — intentional shadow of builtin
    """Raised when an effGen operation exceeds its configured time limit.

    Attributes:
        operation:  Human-readable name of the timed-out operation.
        limit:      Configured timeout in seconds.
    """

    def __init__(self, operation: str, limit: float) -> None:
        super().__init__(
            f"effGen operation '{operation}' exceeded {limit:.1f}s timeout. "
            "Raise the timeout for this operation, or reduce what the call "
            "asks for (a smaller prompt, fewer tools, a faster model)."
        )
        self.operation = operation
        self.limit = limit

    def __reduce__(self) -> tuple[type[TimeoutError], tuple[str, float]]:  # pickling support
        return (type(self), (self.operation, self.limit))


# ---------------------------------------------------------------------------
# Context-manager based timeout
# ---------------------------------------------------------------------------


class _SyncTimeoutContext:
    """
    Synchronous timeout context manager using SIGALRM (Unix) or a daemon
    thread (Windows / environments without SIGALRM).

    Re-arms after the deadline until the wrapped call actually returns —  a
    single one-shot interrupt can be absorbed by code inside the call that
    catches a broad exception class and transparently retries (observed with
    cloud provider SDKs' own internal retry loops), which would otherwise
    let the call run to its natural completion instead of the requested
    bound. Re-firing every :attr:`_RETRY_TICK` seconds past the deadline
    bounds how much longer a swallowed interrupt can let the call run.
    """

    #: Re-fire interval once the deadline has passed.
    _RETRY_TICK = 0.25

    def __init__(self, seconds: float, operation: str) -> None:
        if seconds is None or seconds <= 0:
            raise ValueError(
                f"timeout must be a positive float, got {seconds!r}. "
                "effGen forbids timeout=None on every I/O boundary."
            )
        self._seconds = seconds
        self._operation = operation
        self._use_signal = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
        self._timer: threading.Timer | None = None
        self._timed_out = False
        self._active = False

    # -- signal-based (Unix main thread) --

    def _signal_handler(self, signum, frame):  # noqa: ANN001
        self._timed_out = True
        raise TimeoutError(self._operation, self._seconds)

    # -- thread-based --

    def _thread_raise(self, thread_id: int) -> None:
        """Raise TimeoutError in the calling thread, then reschedule.

        Re-fires every :attr:`_RETRY_TICK` seconds until ``__exit__`` marks
        the context inactive, since ``PyThreadState_SetAsyncExc`` delivery is
        best-effort and a single attempt can land at a point the target
        thread swallows or is temporarily uninterruptible.
        """
        import ctypes

        self._timed_out = True
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(TimeoutError(self._operation, self._seconds)),
        )
        if self._active:
            self._timer = threading.Timer(self._RETRY_TICK, self._thread_raise, args=(thread_id,))
            self._timer.daemon = True
            self._timer.start()

    # -- context protocol --

    def __enter__(self) -> "_SyncTimeoutContext":
        self._active = True
        if self._use_signal:
            self._old_handler = signal.signal(signal.SIGALRM, self._signal_handler)
            # Fire once at the deadline, then keep re-firing every
            # _RETRY_TICK seconds past it (see class docstring) until
            # __exit__ disarms the timer.
            signal.setitimer(signal.ITIMER_REAL, self._seconds, self._RETRY_TICK)
        else:
            # Fallback: daemon thread that sends a signal to the calling
            # thread. Best-effort on non-main threads.
            thread_id = threading.current_thread().ident
            self._timer = threading.Timer(self._seconds, self._thread_raise, args=(thread_id,))
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._active = False
        if self._use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old_handler)
        else:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        return False  # do not suppress exceptions


class _AsyncTimeoutContext:
    """Async timeout context manager using ``asyncio.timeout`` (3.11+)."""

    def __init__(self, seconds: float, operation: str) -> None:
        if seconds is None or seconds <= 0:
            raise ValueError(
                f"timeout must be a positive float, got {seconds!r}. "
                "effGen forbids timeout=None on every I/O boundary."
            )
        self._seconds = seconds
        self._operation = operation
        # Created in __aenter__, so the attribute exists before there is a
        # context to put in it.
        self._ctx: asyncio.Timeout | None = None

    async def __aenter__(self) -> "_AsyncTimeoutContext":
        self._ctx = asyncio.timeout(self._seconds)
        await self._ctx.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._ctx is None:  # pragma: no cover - __aenter__ always sets it
            return False
        try:
            await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
        except builtins.TimeoutError as exc:
            # asyncio.TimeoutError is an alias of builtins.TimeoutError from
            # 3.11 on, so a timeout arrives here as the builtin. Re-raise ours
            # unchanged and wrap anything else.
            if isinstance(exc, TimeoutError):
                raise
            raise TimeoutError(self._operation, self._seconds) from exc
        return False


@contextlib.contextmanager
def with_timeout(seconds: float, operation: str = "operation") -> Iterator["_SyncTimeoutContext"]:
    """Synchronous context manager that enforces a wall-clock timeout.

    Args:
        seconds:    Maximum allowed wall-clock time.  Must be > 0; ``None``
                    is explicitly forbidden.
        operation:  Human-readable label used in the :class:`TimeoutError`
                    message.

    Raises:
        :class:`TimeoutError`: If the body takes longer than *seconds*.
        ValueError:            If *seconds* is ``None`` or ≤ 0.

    Example::

        with with_timeout(30.0, "tool_call"):
            result = tool.run(...)
    """
    ctx = _SyncTimeoutContext(seconds, operation)
    with ctx:
        yield ctx


def async_timeout(seconds: float, operation: str = "operation") -> _AsyncTimeoutContext:
    """Async context manager that enforces a wall-clock timeout.

    Args:
        seconds:    Maximum allowed wall-clock time.  Must be > 0.
        operation:  Human-readable label.

    Example::

        async with async_timeout(60.0, "model_call"):
            result = await adapter.generate(...)
    """
    return _AsyncTimeoutContext(seconds, operation)


def apply_timeout(fn: Callable[..., Any], seconds: float, operation: str | None = None) -> Callable[..., Any]:
    """Wrap *fn* so it raises :class:`TimeoutError` after *seconds* seconds.

    Works for both regular functions and coroutines.

    Args:
        fn:         Callable to wrap (sync or async).
        seconds:    Timeout in seconds.
        operation:  Label for the error message (defaults to ``fn.__name__``).

    Returns:
        A wrapped callable with the same signature as *fn*.

    Raises:
        :class:`TimeoutError`: When the limit is exceeded.
    """
    if seconds is None or seconds <= 0:
        raise ValueError(
            f"timeout must be a positive float, got {seconds!r}."
        )
    label = operation or getattr(fn, "__name__", "unknown")

    if asyncio.iscoroutinefunction(fn):
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with _AsyncTimeoutContext(seconds, label):
                return await fn(*args, **kwargs)

        _async_wrapper.__name__ = fn.__name__
        _async_wrapper.__qualname__ = fn.__qualname__
        return _async_wrapper
    else:
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _SyncTimeoutContext(seconds, label)
            with ctx:
                return fn(*args, **kwargs)

        _sync_wrapper.__name__ = fn.__name__
        _sync_wrapper.__qualname__ = fn.__qualname__
        return _sync_wrapper


def make_httpx_timeout(http: float = 20.0, *, connect: float | None = None) -> Any:
    """Build an ``httpx.Timeout`` from our config values.

    Args:
        http:     Default read/write timeout in seconds.
        connect:  Connection timeout (defaults to ``min(http, 10.0)``).

    Returns:
        ``httpx.Timeout`` instance, or a plain dict if httpx is not installed.
    """
    if http is None:
        raise ValueError("http timeout must not be None.")
    connect_t = connect if connect is not None else min(http, 10.0)
    try:
        import httpx
        return httpx.Timeout(timeout=http, connect=connect_t)
    except ImportError:
        return {"timeout": http, "connect": connect_t}


# ---------------------------------------------------------------------------
# Audit helper — used in tests to catch timeout=None in adapter code
# ---------------------------------------------------------------------------

def audit_no_none_timeouts(source_root: str) -> list[str]:
    """Scan Python source files under *source_root* for ``timeout=None``.

    Returns a list of ``"<file>:<line>"`` strings where ``timeout=None``
    appears in non-comment, non-docstring code.  An empty list means clean.

    Used by ``tests/reliability/test_timeouts.py`` to enforce the rule.
    """
    import ast
    import pathlib

    violations: list[str] = []
    root = pathlib.Path(source_root)

    for py_file in root.rglob("*.py"):
        # Skip test files themselves and reliability module
        if "test_" in py_file.name:
            continue
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Look for keyword argument timeout=None in function calls
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "timeout" and isinstance(kw.value, ast.Constant) and kw.value.value is None:
                        violations.append(f"{py_file}:{node.lineno}")

    return violations
