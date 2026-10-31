"""
OpenAI reasoning models example — o4-mini with reasoning_effort control.

Demonstrates:
- Using o4-mini (an o-series reasoning model)
- Setting reasoning_effort via GenerationConfig
- Comparing low vs high effort on the same problem
- Why reasoning models differ from chat models

Run:
    python examples/openai/reasoning_models.py
"""

from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

from effgen.models.base import GenerationConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_models import reasoning_models

PROBLEM = (
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? "
    "Show your reasoning."
)


def ask(adapter: OpenAIAdapter, effort: str) -> dict:
    cfg = GenerationConfig(reasoning_effort=effort, max_tokens=512)
    t0 = time.perf_counter()
    result = adapter.generate(PROBLEM, config=cfg)
    latency = time.perf_counter() - t0
    return {"text": result.text, "tokens": result.tokens_used, "latency": latency, "effort": effort}


def main():
    print("Available reasoning models:", reasoning_models())
    print()

    adapter = OpenAIAdapter("o4-mini")
    adapter.load()
    print(f"Loaded: {adapter.model_name}")
    print(f"Is reasoning model: {adapter.is_reasoning_model()}")
    print(f"Context: {adapter.get_context_length():,} tokens")
    print()

    # --- Low effort ---
    print("--- Low effort (fast, cheap) ---")
    low = ask(adapter, "low")
    print(f"Answer: {low['text']}")
    print(f"Tokens: {low['tokens']}  Latency: {low['latency']:.2f}s")
    print()

    # --- High effort ---
    print("--- High effort (deeper reasoning) ---")
    high = ask(adapter, "high")
    print(f"Answer: {high['text']}")
    print(f"Tokens: {high['tokens']}  Latency: {high['latency']:.2f}s")
    print()

    # --- Comparison ---
    print("--- Comparison ---")
    print("Effort   | Tokens | Latency")
    print(f"low      | {low['tokens']:6d} | {low['latency']:.2f}s")
    print(f"high     | {high['tokens']:6d} | {high['latency']:.2f}s")
    print(f"Delta    | {high['tokens'] - low['tokens']:+6d} | {high['latency'] - low['latency']:+.2f}s")

    print(f"\nTotal cost: ${adapter.get_total_cost():.6f}")
    adapter.unload()


if __name__ == "__main__":
    main()
