"""AgentPool — pool of pre-initialized effGen agents for the API server.

Rather than constructing a new `Agent` per request (expensive — it loads the
model, initializes tools, etc.), the server borrows a pre-warmed agent from a
pool. The pool auto-scales between ``min_size`` and ``max_size``, performs
periodic health checks, and recycles unhealthy agents.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PooledAgent:
    """Wrapper around an Agent instance tracked by the pool."""

    agent: Any
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    in_use: bool = False
    healthy: bool = True
    request_count: int = 0

    def touch(self) -> None:
        """Record a use: refresh the last-used time and bump the request count."""
        self.last_used_at = time.time()
        self.request_count += 1


class AgentPool:
    """Pool of pre-initialized agents with auto-scaling and health checks.

    Parameters
    ----------
    factory:
        Zero-arg callable that returns a fresh ``Agent`` instance.
    min_size:
        Minimum number of agents kept warm.
    max_size:
        Upper bound on concurrent agents.
    idle_ttl:
        Seconds an idle agent may live before being scaled down (only down
        to ``min_size``).
    health_check_interval:
        Seconds between background health sweeps.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        min_size: int = 1,
        max_size: int = 8,
        idle_ttl: float = 300.0,
        health_check_interval: float = 60.0,
        health_checker: Callable[[Any], bool] | None = None,
    ) -> None:
        if min_size < 0 or max_size < min_size:
            raise ValueError("require 0 <= min_size <= max_size")
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_ttl = idle_ttl
        self.health_check_interval = health_check_interval
        self._health_checker = health_checker or self._default_health_checker
        self._agents: list[PooledAgent] = []
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._closed = False
        self._health_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Warm up to ``min_size`` agents and start background health checks."""
        async with self._lock:
            while len(self._agents) < self.min_size:
                self._agents.append(self._create_locked())
        if self.health_check_interval > 0 and self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        """Close the pool: stop health checks and destroy every agent."""
        async with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except (asyncio.CancelledError, Exception):
                pass
            self._health_task = None
        async with self._lock:
            for pa in self._agents:
                self._destroy(pa)
            self._agents.clear()

    # ------------------------------------------------------------------ checkout

    async def acquire(self, timeout: float | None = None) -> PooledAgent:
        """Borrow a healthy agent. Creates one on-demand up to ``max_size``."""
        end = time.time() + timeout if timeout is not None else None
        async with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("AgentPool is closed")
                for pa in self._agents:
                    if not pa.in_use and pa.healthy:
                        pa.in_use = True
                        pa.touch()
                        return pa
                if len(self._agents) < self.max_size:
                    pa = self._create_locked()
                    self._agents.append(pa)
                    pa.in_use = True
                    pa.touch()
                    return pa
                remaining = None
                if end is not None:
                    remaining = end - time.time()
                    if remaining <= 0:
                        raise TimeoutError("AgentPool.acquire timed out")
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    raise TimeoutError("AgentPool.acquire timed out")

    async def release(self, pa: PooledAgent) -> None:
        """Return *pa* to the pool (an unhealthy surplus agent is destroyed)."""
        async with self._cond:
            pa.in_use = False
            if not pa.healthy and len(self._agents) > self.min_size:
                self._destroy(pa)
                if pa in self._agents:
                    self._agents.remove(pa)
            self._cond.notify()

    # ------------------------------------------------------------------ internals

    def _create_locked(self) -> PooledAgent:
        try:
            agent = self.factory()
        except Exception as e:
            logger.error("AgentPool factory failed: %s", e)
            raise
        return PooledAgent(agent=agent)

    def _destroy(self, pa: PooledAgent) -> None:
        close = getattr(pa.agent, "close", None) or getattr(pa.agent, "shutdown", None)
        if callable(close):
            try:
                close()
            except Exception as e:  # pragma: no cover
                logger.warning("Error closing pooled agent: %s", e)

    def _default_health_checker(self, agent: Any) -> bool:
        # An agent is considered healthy if it exists and has a callable `run`.
        return agent is not None and callable(getattr(agent, "run", None))

    async def _health_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._sweep_once()
            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover
                logger.warning("AgentPool health sweep error: %s", e)

    async def _sweep_once(self) -> None:
        now = time.time()
        async with self._lock:
            survivors: list[PooledAgent] = []
            for pa in self._agents:
                if pa.in_use:
                    survivors.append(pa)
                    continue
                try:
                    pa.healthy = bool(self._health_checker(pa.agent))
                except Exception:
                    pa.healthy = False
                idle_for = now - pa.last_used_at
                scale_down = (
                    len(survivors) >= self.min_size and idle_for > self.idle_ttl
                )
                if not pa.healthy or scale_down:
                    self._destroy(pa)
                    continue
                survivors.append(pa)
            # Maintain minimum warm size.
            while len(survivors) < self.min_size:
                try:
                    survivors.append(self._create_locked())
                except Exception:
                    break
            self._agents = survivors

    # ------------------------------------------------------------------ utilities

    def stats(self) -> dict:
        """Return pool counters: total/in-use/healthy agents and size bounds."""
        return {
            "total": len(self._agents),
            "in_use": sum(1 for a in self._agents if a.in_use),
            "healthy": sum(1 for a in self._agents if a.healthy),
            "min_size": self.min_size,
            "max_size": self.max_size,
        }

    class _CtxManager:
        def __init__(self, pool: "AgentPool", timeout: float | None) -> None:
            self.pool = pool
            self.timeout = timeout
            self.pa: PooledAgent | None = None

        async def __aenter__(self) -> PooledAgent:
            self.pa = await self.pool.acquire(self.timeout)
            return self.pa

        async def __aexit__(self, exc_type, exc, tb) -> None:
            if self.pa is not None:
                await self.pool.release(self.pa)

    def borrow(self, timeout: float | None = None) -> "AgentPool._CtxManager":
        """`async with pool.borrow() as pa:` context manager."""
        return AgentPool._CtxManager(self, timeout)
