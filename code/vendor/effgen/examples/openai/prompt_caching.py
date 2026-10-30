"""
OpenAI automatic prompt caching example.

OpenAI automatically caches prompt prefixes that are ≥1024 tokens long.
When you send repeated requests that share a large, stable system prompt,
tokens in the cached prefix are billed at the lower cached-input rate and
the latency for those tokens drops to near-zero.

This example shows:
  1. How to use generate_with_system_prompt() to anchor the system prompt
     at the front of the message list (required for prefix caching).
  2. How to read cached_input_tokens from the response metadata.
  3. The cost savings from cache hits over 5 sequential calls.

Run:
    python examples/openai/prompt_caching.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("effgen.models.openai_adapter.usage").setLevel(logging.INFO)

from effgen.models.openai_adapter import OpenAIAdapter

# ---------------------------------------------------------------------------
# Build a large, stable system prompt (≥1024 tokens needed for caching)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert research assistant with deep knowledge in physics, "
    "chemistry, biology, computer science, mathematics, and engineering.\n\n"
    + (
        "You always provide accurate, well-sourced, and concise answers. "
        "When asked a factual question you state the exact value first, "
        "then briefly explain the context. "
        "You do not add unnecessary caveats or hedging language. "
        "You are direct and to the point. " * 100
    )
    + "\n\nAlways cite units when relevant. Prefer SI units."
)

questions = [
    "What is the speed of light in vacuum?",
    "What is the Boltzmann constant?",
    "What is Avogadro's number?",
    "What is the gravitational constant G?",
    "What is Planck's constant?",
]

print("OpenAI Prompt Caching Demo")
print("=" * 60)
print("Model: gpt-5.4-nano")
print(f"System prompt: ~{len(SYSTEM_PROMPT.split())} words")
print()

adapter = OpenAIAdapter(model_name="gpt-5.4-nano")
adapter.load()

total_prompt_tokens = 0
total_cached_tokens = 0
total_cost = 0.0

try:
    for i, question in enumerate(questions, 1):
        result = adapter.generate_with_system_prompt(
            prompt=question,
            system_prompt=SYSTEM_PROMPT,
        )
        meta = result.metadata
        prompt_tok = meta["prompt_tokens"]
        cached_tok = meta["cached_input_tokens"]
        completion_tok = meta["completion_tokens"]
        cost = meta["cost"]

        total_prompt_tokens += prompt_tok
        total_cached_tokens += cached_tok
        total_cost += cost

        cache_pct = (cached_tok / prompt_tok * 100) if prompt_tok > 0 else 0
        status = f"CACHED {cached_tok}/{prompt_tok} ({cache_pct:.0f}%)" if cached_tok > 0 else "COLD (no cache yet)"

        print(f"Call {i}: {question}")
        print(f"  Answer: {result.text[:120].strip()}")
        print(f"  Tokens: input={prompt_tok} cached={cached_tok} output={completion_tok}")
        print(f"  Cache:  {status}")
        print(f"  Cost:   ${cost:.6f}")
        print()

finally:
    adapter.unload()

print("=" * 60)
print("Summary")
print(f"  Total input tokens:  {total_prompt_tokens}")
print(f"  Total cached tokens: {total_cached_tokens}")
if total_prompt_tokens > 0:
    print(f"  Cache hit rate:      {total_cached_tokens / total_prompt_tokens * 100:.1f}%")
print(f"  Total cost:          ${total_cost:.6f}")
print()
print("Tip: Run this script twice — the second run will show higher cache hits")
print("     because OpenAI warms the prefix on the first call.")
