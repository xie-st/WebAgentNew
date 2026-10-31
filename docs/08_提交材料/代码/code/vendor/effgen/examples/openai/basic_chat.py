"""
Basic OpenAI chat example using the effGen OpenAIAdapter.

Demonstrates:
- Initializing OpenAIAdapter with gpt-5.4-nano
- Simple text generation
- Streaming generation
- Cost tracking

Run:
    python examples/openai/basic_chat.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

# Show token/cost breakdown from the adapter
logging.basicConfig(level=logging.WARNING)
logging.getLogger("effgen.models.openai_adapter.usage").setLevel(logging.INFO)

from effgen.models.base import GenerationConfig
from effgen.models.openai_adapter import OpenAIAdapter


def main():
    # Use gpt-5.4-nano — cheapest model, great for quick tasks
    adapter = OpenAIAdapter("gpt-5.4-nano")
    adapter.load()
    print(f"Loaded: {adapter.model_name} (context={adapter.get_context_length():,} tokens)")

    # --- Basic generation ---
    print("\n--- Basic generation ---")
    result = adapter.generate(
        "In one sentence, explain what a large language model is.",
        config=GenerationConfig(max_tokens=100),
    )
    print(f"Response: {result.text}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Cost: ${result.metadata['cost']:.6f}")

    # --- Streaming generation ---
    print("\n--- Streaming ---")
    print("Response: ", end="", flush=True)
    for chunk in adapter.generate_stream(
        "List three fun facts about the Eiffel Tower. Be brief.",
        config=GenerationConfig(max_tokens=150),
    ):
        print(chunk, end="", flush=True)
    print()

    # --- Summary ---
    print(f"\nTotal cost so far: ${adapter.get_total_cost():.6f}")
    print(f"Total tokens used: {adapter.get_total_tokens()}")

    adapter.unload()


if __name__ == "__main__":
    main()
