#!/usr/bin/env python3
"""
Async & Concurrent Agent Example
=================================

Demonstrates:
1. Running Agent.run() from async code
2. Running multiple agents concurrently with asyncio.gather
3. Using Agent.run_async() for non-blocking execution
4. Parallel tool execution within a single agent

Requires: a local model (e.g. Qwen2.5-1.5B-Instruct) or set
OPENAI_API_KEY / ANTHROPIC_API_KEY for a cloud backend.
"""

from __future__ import annotations

import asyncio
import time

from effgen import Agent, AgentConfig
from effgen.models import load_model
from effgen.tools.builtin import Calculator, DateTimeTool

# ---------------------------------------------------------------------------
# 1. Agent.run() inside async code  (sync-in-async)
# ---------------------------------------------------------------------------

async def sync_agent_in_async():
    """
    Agent.run() is synchronous but safe to call from async code.
    effGen's internal _run_coroutine_sync() handles the event-loop
    detection so this will NOT crash with "cannot run nested event loop".
    """
    model = load_model("Qwen/Qwen2.5-1.5B-Instruct")
    config = AgentConfig(
        name="sync-in-async",
        model=model,
        tools=[Calculator()],
        max_iterations=5,
        enable_sub_agents=False,
    )

    with Agent(config=config) as agent:
        result = agent.run("What is 42 * 58?")
        print(f"[sync-in-async] {result.output}")
        return result


# ---------------------------------------------------------------------------
# 2. Multiple agents with asyncio.gather (true concurrency)
# ---------------------------------------------------------------------------

async def concurrent_agents():
    """
    Spin up several agents and run them concurrently using
    Agent.run_async() which delegates to a thread executor.
    """
    model = load_model("Qwen/Qwen2.5-1.5B-Instruct")

    agents = []
    tasks = [
        "What is 2 + 2?",
        "What is the current date and time?",
        "What is 100 factorial divided by 99 factorial?",
    ]

    for i, _task in enumerate(tasks):
        config = AgentConfig(
            name=f"concurrent-{i}",
            model=model,
            tools=[Calculator(), DateTimeTool()],
            max_iterations=5,
            enable_sub_agents=False,
        )
        agents.append(Agent(config=config))

    start = time.perf_counter()
    results = await asyncio.gather(
        *(agent.run_async(task) for agent, task in zip(agents, tasks))
    )
    elapsed = time.perf_counter() - start

    for task, result in zip(tasks, results):
        print(f"  Q: {task}")
        print(f"  A: {result.output}\n")

    print(f"[concurrent] {len(tasks)} tasks completed in {elapsed:.2f}s")

    # Clean up
    for agent in agents:
        agent.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=== 1. Sync agent inside async context ===")
    await sync_agent_in_async()

    print("\n=== 2. Concurrent agents with asyncio.gather ===")
    await concurrent_agents()


if __name__ == "__main__":
    asyncio.run(main())
