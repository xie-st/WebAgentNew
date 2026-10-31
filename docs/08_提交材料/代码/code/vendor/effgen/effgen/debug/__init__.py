"""
effGen debug module — interactive debugging and inspection for agent execution.

Provides:
    - DebugAgent: wrapper that captures full execution trace
    - DebugTrace / DebugIteration: structured trace data
    - run_debug_cli: interactive TUI for step-through debugging

Usage:
    from effgen.debug import DebugAgent

    agent = DebugAgent(config)
    result = agent.run("What is 2+2?")
    trace = result.metadata["debug_trace"]
    for it in trace.iterations:
        print(it.action, it.observation)
"""

from __future__ import annotations

from effgen.debug.inspector import (
    DebugAgent,
    DebugIteration,
    DebugTrace,
    run_debug_cli,
)

__all__ = [
    "DebugAgent",
    "DebugIteration",
    "DebugTrace",
    "run_debug_cli",
]
