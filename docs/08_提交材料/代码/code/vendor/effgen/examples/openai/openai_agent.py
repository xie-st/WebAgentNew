"""
Full effGen Agent using OpenAI models with native tools + effGen tools.

Demonstrates:
- Agent with OpenAIAdapter (gpt-4.1-nano for cost efficiency)
- effGen built-in tools (Calculator, DateTime)
- Hard agentic tasks: multi-step math + calendar reasoning + portfolio analysis
- Native OpenAI tool-calling path integrated transparently

Run:
    python examples/openai/openai_agent.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

# Show token/cost info
logging.basicConfig(level=logging.WARNING)
logging.getLogger("effgen.models.openai_adapter.usage").setLevel(logging.INFO)

from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.datetime_tool import DateTimeTool


def run_agent(task: str, model_name: str = "gpt-4.1-nano"):
    print(f"\n{'='*60}")
    print(f"Task: {task[:80]}{'...' if len(task) > 80 else ''}")
    print(f"Model: {model_name}")
    print("=" * 60)

    model = OpenAIAdapter(model_name)
    model.load()

    agent = Agent(
        config=AgentConfig(
            name="openai-agent",
            model=model,
            tools=[Calculator(), DateTimeTool()],
            max_iterations=8,
            tool_calling_mode="auto",
        ),
    )

    try:
        result = agent.run(task)
        print(f"\nFinal answer:\n{result.output}")
    finally:
        agent.close()
        print(f"\nModel cost: ${model.get_total_cost():.6f}")
        model.unload()

    return result


def main():
    tasks = [
        # Multi-step math requiring tool use
        "Calculate (17 * 23) + sqrt(144) + 2^8. Show each step and the final total.",

        # Temporal reasoning
        ("A project started 45 days before today. What was the start date? "
         "How many weeks is 45 days (to 2 decimal places)?"),

        # Hard: portfolio analysis requiring multiple tool calls
        ("I have a portfolio: Asset A = 250 shares @ $42.50 each, "
         "Asset B = 150 shares @ $87.20 each, Asset C = 75 shares @ $310.00 each. "
         "Calculate: (1) total value of each asset, (2) total portfolio value, "
         "(3) what percentage of the total is each asset (to 2 decimal places)?"),
    ]

    for task in tasks:
        run_agent(task)


if __name__ == "__main__":
    main()
