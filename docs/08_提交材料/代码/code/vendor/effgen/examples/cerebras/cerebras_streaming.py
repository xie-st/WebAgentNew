"""
Cerebras streaming example — real token-by-token output.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.effgen/.env

What this demonstrates:
  - Real streaming from Cerebras with generate_stream()
  - Chunk-by-chunk output with live timestamps
  - Both gpt-oss-120b and zai-glm-4.7 models
  - CostTracker showing $0 for free-tier usage
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from effgen.models._cost import CostTracker  # noqa: E402
from effgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402


def stream_demo(model_id: str, prompt: str) -> None:
    print(f"\n{'='*60}")
    print(f"Streaming with {model_id}")
    print(f"Prompt: {prompt!r}")
    print("="*60)

    adapter = CerebrasAdapter(model_id, enable_rate_limiting=False)
    adapter.load()

    chunks = []
    start = time.monotonic()

    try:
        for chunk in adapter.generate_stream(prompt):
            time.monotonic() - start
            chunks.append(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()
    finally:
        adapter.unload()

    total_time = time.monotonic() - start
    print("\n\n--- Stats ---")
    print(f"Chunks received: {len(chunks)}")
    print(f"Total characters: {sum(len(c) for c in chunks)}")
    print(f"Wall time: {total_time:.2f}s")


def main():
    prompt = (
        "Write a short paragraph (3-4 sentences) explaining why streaming is "
        "useful for large language model outputs."
    )

    # Stream from two free-tier models
    for model_id in ["gpt-oss-120b", "zai-glm-4.7"]:
        stream_demo(model_id, prompt)

    # Show cost summary — Cerebras free tier = $0
    print("\n--- Cost Summary (Cerebras free tier = $0) ---")
    for row in CostTracker.get().summary():
        # ``cost_usd`` is None for a model the catalog publishes no rate for.
        cost = row["cost_usd"]
        print(
            f"  {row['provider']}/{row['model']}: "
            f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} "
            f"cost={'unpriced' if cost is None else f'${cost:.6f}'}"
        )


if __name__ == "__main__":
    main()
