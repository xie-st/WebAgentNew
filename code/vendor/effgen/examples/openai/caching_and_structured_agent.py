"""
Advanced effGen agent: prompt caching + structured outputs + native tools.

This example demonstrates:
  1. Stable system prompt for automatic prefix caching across agent calls.
  2. Structured output extraction from agent reasoning.
  3. effGen tools (Calculator, DateTimeTool) + hard multi-step agentic tasks.
  4. Combining generate_structured() for structured result extraction with
     the Agent loop for complex reasoning.
  5. How AgentConfig.stable_system_prompt enables caching-friendly behaviour.

Run:
    python examples/openai/caching_and_structured_agent.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("effgen.models.openai_adapter.usage").setLevel(logging.INFO)

from pydantic import BaseModel, Field

from effgen.core.agent import Agent, AgentConfig
from effgen.models.errors import ModelRefusalError
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_schema import to_openai_schema
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.datetime_tool import DateTimeTool

MODEL = "gpt-5.4-nano"

# ---------------------------------------------------------------------------
# A large, stable system prompt for prefix caching
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = (
    "You are an advanced quantitative research assistant with expertise in "
    "mathematics, statistics, finance, and data analysis.\n\n"
    "You always use tools when calculations are required. "
    "You show your reasoning step-by-step. "
    "You express numerical results to at least 4 significant figures. "
    "You never guess — if you need a calculation, you use the Calculator tool. "
    "If you need the current date or time, you use the DateTimeTool. "
    "You double-check your final answer before reporting it. " * 80
)


# ---------------------------------------------------------------------------
# Structured result schemas
# ---------------------------------------------------------------------------

class MathResult(BaseModel):
    answer: float
    units: str
    reasoning_steps: list[str]


class PortfolioAllocation(BaseModel):
    asset: str
    weight_pct: float = Field(ge=0, le=100)
    rationale: str


class PortfolioResult(BaseModel):
    total_value: float
    allocations: list[PortfolioAllocation]
    risk_level: Literal["low", "medium", "high"]
    recommendation: str


def run_task(adapter: OpenAIAdapter, agent: Agent, label: str, task: str) -> str:
    print(f"\n{'='*60}")
    print(f"Task: {label}")
    print(f"{'='*60}")
    print(f"Prompt: {task[:100]}...")
    response = agent.run(task)
    result = response.output if hasattr(response, "output") else str(response)
    print(f"Answer:\n{result[:400]}")
    return result


def main():
    print("effGen: Caching + Structured Outputs + Native Tools Demo")
    print(f"Model: {MODEL}")
    print()

    adapter = OpenAIAdapter(model_name=MODEL)
    adapter.load()

    agent = Agent(
        config=AgentConfig(
            name="quant-agent",
            model=adapter,
            tools=[Calculator(), DateTimeTool()],
            system_prompt=AGENT_SYSTEM_PROMPT,
            max_iterations=10,
            tool_calling_mode="auto",
            stable_system_prompt=True,
        )
    )

    # -----------------------------------------------------------------------
    # Hard agentic task 1: multi-step compound calculation
    # -----------------------------------------------------------------------
    run_task(
        adapter, agent,
        label="Compound interest with irregular schedule",
        task=(
            "A $50,000 investment grows at 8.5% compounded quarterly for 7 years, "
            "then at 6.2% compounded monthly for 3 more years. "
            "What is the final value? Show each step."
        ),
    )

    # -----------------------------------------------------------------------
    # Hard agentic task 2: date arithmetic
    # -----------------------------------------------------------------------
    run_task(
        adapter, agent,
        label="Date arithmetic + elapsed time",
        task=(
            "A project started on 2024-03-15 and has a 500-day deadline. "
            "How many days are left from today? Is the project overdue?"
        ),
    )

    # -----------------------------------------------------------------------
    # Hard agentic task 3: portfolio math
    # -----------------------------------------------------------------------
    run_task(
        adapter, agent,
        label="Portfolio rebalancing calculation",
        task=(
            "I have a portfolio: $120,000 in stocks, $45,000 in bonds, $35,000 in cash. "
            "I want to rebalance to 60% stocks / 30% bonds / 10% cash. "
            "How much do I need to buy or sell of each asset?"
        ),
    )

    # -----------------------------------------------------------------------
    # Structured output extraction: parse portfolio result into typed schema
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Structured Output: Portfolio Analysis")
    print("=" * 60)

    portfolio_rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "PortfolioResult",
            "schema": to_openai_schema(PortfolioResult),
            "strict": True,
        },
    }

    try:
        result = adapter.generate_structured(
            prompt=(
                "Analyze this portfolio: $120,000 stocks, $45,000 bonds, $35,000 cash. "
                "Provide target 60/30/10 allocations as percentages, total value, "
                "risk level, and a one-sentence recommendation."
            ),
            response_format=portfolio_rf,
            system_prompt="You are a portfolio analysis assistant. Always respond with exact JSON.",
        )
        portfolio = PortfolioResult.model_validate_json(result.text)
        print(f"Total value: ${portfolio.total_value:,.2f}")
        print(f"Risk level: {portfolio.risk_level}")
        print(f"Recommendation: {portfolio.recommendation}")
        print("Allocations:")
        for alloc in portfolio.allocations:
            print(f"  {alloc.asset:10} {alloc.weight_pct:5.1f}%  —  {alloc.rationale[:60]}")
        print(f"\nCached input tokens: {result.metadata.get('cached_input_tokens', 0)}")
    except ModelRefusalError as e:
        print(f"Model refused: {e}")
    except Exception as e:
        print(f"Structured extraction failed: {e}")

    # -----------------------------------------------------------------------
    # Show caching stats across all calls
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Session Cost Summary")
    print("=" * 60)
    print(f"Total tokens used: {adapter.get_total_tokens()}")
    print(f"Total cost:        ${adapter.get_total_cost():.6f}")
    print()
    print("Note: The stable system prompt is cached after the first call,")
    print("reducing cost and latency for all subsequent requests.")

    adapter.unload()


if __name__ == "__main__":
    main()
