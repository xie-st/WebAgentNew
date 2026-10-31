"""
Cerebras multi-model example — compare all free-tier models side-by-side.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.effgen/.env

What this demonstrates:
  - Listing all registered Cerebras models and their metadata
  - Calling accessible free-tier models in parallel via asyncio.gather
  - Comparing responses, latency, and token usage across models
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from effgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402
from effgen.models.cerebras_models import (  # noqa: E402
    available_models,
    free_tier_models,
    model_info,
)

# -----------------------------------------------------------
# 1. Inspect the model registry
# -----------------------------------------------------------
print("=== Cerebras Model Registry ===\n")
for model_id in available_models():
    info = model_info(model_id)
    tier = "free" if info["free_tier"] else "restricted"
    print(
        f"  {model_id:<45}  context={info['context']//1000}k  "
        f"max_out={info['max_output']}  rpm={info['rpm']}  [{tier}]"
    )

print(f"\nFree-tier accessible: {free_tier_models()}\n")

# -----------------------------------------------------------
# 2. Parallel generation across free-tier models
# -----------------------------------------------------------
PROMPT = (
    "In one sentence, explain what makes you unique compared to other language models."
)

async def call_model(model_id: str) -> dict:
    adapter = CerebrasAdapter(model_name=model_id, enable_rate_limiting=False)
    adapter.load()
    t0 = time.monotonic()
    try:
        result = await adapter.async_generate(PROMPT)
        elapsed = time.monotonic() - t0
        return {
            "model": model_id,
            "text": result.text,
            "tokens": result.metadata.get("total_tokens", 0) if result.metadata else 0,
            "elapsed_s": round(elapsed, 2),
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model_id,
            "text": None,
            "tokens": 0,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "error": str(exc),
        }
    finally:
        adapter.unload()


async def main() -> None:
    models = free_tier_models()
    print(f"=== Parallel generation across {len(models)} free-tier models ===")
    print(f"Prompt: {PROMPT}\n")

    results = await asyncio.gather(*[call_model(m) for m in models])

    for r in results:
        if r["error"] is None:
            print(f"[{r['model']}]  ({r['elapsed_s']}s, {r['tokens']} tokens)")
            print(f"  {r['text']}\n")
        else:
            print(f"[{r['model']}]  ERROR: {r['error']}\n")


asyncio.run(main())
