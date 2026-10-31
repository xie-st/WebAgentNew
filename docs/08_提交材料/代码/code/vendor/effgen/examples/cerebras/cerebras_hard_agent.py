"""
Cerebras hard agentic task example — multi-tool, multi-step reasoning.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.effgen/.env

What this demonstrates:
  - Complex multi-step agent tasks requiring tool chains
  - Calculator + DateTimeTool + Python REPL
  - Both gpt-oss-120b and zai-glm-4.7
  - Inspecting iteration count and tool call traces
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from effgen.core.agent import Agent, AgentConfig  # noqa: E402
from effgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402
from effgen.tools.builtin import Calculator, DateTimeTool  # noqa: E402
from effgen.tools.builtin.python_repl import PythonREPL  # noqa: E402

HARD_TASKS = [
    {
        "name": "Compound interest calculation",
        "task": (
            "I have $12,500. I want to invest it at 8.5% annual interest compounded monthly "
            "for 7 years. What will be the final amount? Show the formula and the calculation."
        ),
    },
    {
        "name": "Prime factorization + date math",
        "task": (
            "Find the prime factorization of 3,600. "
            "Then tell me today's date and calculate what date it will be "
            "after adding 3,600 days from now."
        ),
    },
    {
        "name": "Fibonacci series",
        "task": (
            "Write a Python function to compute the first 15 Fibonacci numbers, "
            "run it, and report the sum of those numbers."
        ),
    },
]


def run_hard_task(model_id: str, task_name: str, task: str) -> None:
    print(f"\n{'='*60}")
    print(f"Model : {model_id}")
    print(f"Task  : {task_name}")
    print(f"{'='*60}")

    adapter = CerebrasAdapter(model_name=model_id, enable_rate_limiting=False)
    adapter.load()

    config = AgentConfig(
        name=f"cerebras-hard-agent-{model_id}",
        model=adapter,
        tools=[Calculator(), DateTimeTool(), PythonREPL()],
        max_iterations=8,
        system_prompt=(
            "You are a precise assistant. Use your tools to compute exact answers. "
            "Show your work step by step."
        ),
    )

    try:
        with Agent(config) as agent:
            response = agent.run(task)
        print(f"\nAnswer:\n{response.output}")
        if hasattr(response, "iterations"):
            print(f"\n[iterations={response.iterations}]")
    except Exception as exc:
        print(f"\nERROR: {exc}")
    finally:
        adapter.unload()


if __name__ == "__main__":
    models = ["gpt-oss-120b", "zai-glm-4.7"]

    for model_id in models:
        for task_spec in HARD_TASKS:
            run_hard_task(model_id, task_spec["name"], task_spec["task"])

    print("\n\nAll hard agent tasks complete.")
