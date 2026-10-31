"""Async cleanup symmetry and a clear error when awaiting the sync ``run()``.

``run()`` is sync and ``run_async()`` is async; cleanup should be reachable from
both worlds. These offline tests pin:

* ``await agent.aclose()`` works and is idempotent with ``close()``;
* ``async with agent:`` cleans up via the async path;
* ``await agent.run(...)`` fails with a helpful message (not the opaque
  "object AgentResponse can't be used in 'await' expression").
"""

import asyncio

import pytest

from effgen import create_agent
from effgen.core.agent import AgentResponse


def _agent():
    return create_agent("minimal", "x", require_model=False)


def test_aclose_closes_and_is_idempotent():
    agent = _agent()

    async def go():
        await agent.aclose()
        # Second call is a no-op, never raises.
        await agent.aclose()

    asyncio.run(go())
    assert agent._closed is True


def test_async_with_cleans_up():
    async def go():
        async with _agent() as agent:
            assert agent._closed is False
        assert agent._closed is True

    asyncio.run(go())


def test_await_sync_run_gives_helpful_error():
    resp = AgentResponse(output="hi")

    async def go():
        with pytest.raises(TypeError) as exc:
            await resp
        msg = str(exc.value)
        assert "run_async" in msg
        assert "synchronous" in msg

    asyncio.run(go())
