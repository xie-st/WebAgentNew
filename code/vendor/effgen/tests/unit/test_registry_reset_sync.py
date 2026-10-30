"""Tool-registry teardown must be event-loop-safe.

``reset_registry()`` and ``unregister_tool()`` used ``asyncio.create_task(...)``
which raised ``RuntimeError: no running event loop`` from synchronous code (and
leaked "coroutine was never awaited" warnings). They must now clean up tools
synchronously, and an async variant must be available for coroutine callers.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from effgen.tools import (
    async_reset_registry,
    get_registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _restore_global_registry():
    """Keep these tests from leaking global-registry state into the suite.

    Several tests here mutate the process-wide registry (``unregister_tool``,
    instantiating tools). Under selective collection (``-k``), test sharding, or
    randomized order the sibling tests that would happen to restore it may not
    run, leaving downstream tests with a registry that is missing ``calculator``.
    Drop the global registry before and after each test so the next access
    rebuilds a fresh, fully-discovered one.
    """
    reset_registry()
    yield
    reset_registry()


def _fresh_registry_with_instance():
    reset_registry()
    reg = get_registry()
    reg.discover_builtin_tools()
    # Instantiate one tool so the cleanup path actually has work to do.
    asyncio.run(reg.get_tool("calculator"))
    assert len(reg._instances) >= 1
    return reg


def test_reset_registry_no_running_loop_does_not_raise():
    _fresh_registry_with_instance()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reset_registry()  # must not raise RuntimeError("no running event loop")
        never_awaited = [
            w for w in caught if "never awaited" in str(w.message).lower()
        ]
    assert not never_awaited, f"leaked coroutine warnings: {never_awaited}"


def test_unregister_tool_no_running_loop_does_not_raise():
    reg = _fresh_registry_with_instance()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.unregister_tool("calculator")
        never_awaited = [
            w for w in caught if "never awaited" in str(w.message).lower()
        ]
    assert "calculator" not in reg._instances
    assert "calculator" not in reg._tools
    assert not never_awaited


def test_reset_registry_from_within_running_loop():
    """Calling the sync reset from inside a running loop must still work
    (it bridges onto a worker thread rather than crashing)."""

    async def scenario():
        reg = get_registry()
        reg.discover_builtin_tools()
        await reg.get_tool("calculator")
        reset_registry()  # sync API invoked from within a live loop
        return True

    assert asyncio.run(scenario()) is True


@pytest.mark.asyncio
async def test_async_reset_registry_awaits_cleanup():
    reg = get_registry()
    reg.discover_builtin_tools()
    await reg.get_tool("calculator")
    assert len(reg._instances) >= 1
    await async_reset_registry()
    # A brand-new registry is created on next access.
    new_reg = get_registry()
    assert len(new_reg._instances) == 0
