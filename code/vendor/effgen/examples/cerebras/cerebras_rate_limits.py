"""
Cerebras rate-limit coordinator example.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.effgen/.env

What this demonstrates:
  - Per-model rate-limit metadata from the registry
  - The built-in RateLimitCoordinator that throttles requests automatically
  - Observing coordinator status before and after calls
  - Sending a burst of requests and watching the coordinator pace them
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
from effgen.models.cerebras_models import model_info  # noqa: E402

# -----------------------------------------------------------
# 1. Show rate limits for each model
# -----------------------------------------------------------
print("=== Rate limits per model (free tier) ===\n")
for model_id in ["gpt-oss-120b", "zai-glm-4.7", "zai-glm-4.7", "gpt-oss-120b"]:
    info = model_info(model_id)
    print(
        f"  {model_id:<45}  "
        f"RPM={info['rpm']}  RPH={info['rph']}  RPD={info['rpd']}  "
        f"TPM={info['tpm']//1000}k  TPH={info['tph']//1_000_000}M  TPD={info['tpd']//1_000_000}M"
    )

# -----------------------------------------------------------
# 2. Demonstrate the rate-limit coordinator
# -----------------------------------------------------------
print("\n=== Rate-limit coordinator in action ===\n")

MODEL = "gpt-oss-120b"
adapter = CerebrasAdapter(model_name=MODEL)  # rate limiting enabled by default
adapter.load()

print(f"Model: {MODEL}")
print("Initial coordinator status:")
status = adapter.rate_limit_status()
print(
    f"  Requests used (minute): {status['req_minute_used']}/{status['req_minute_limit']}\n"
    f"  Tokens used (minute):   {status['tok_minute_used']}/{status['tok_minute_limit']}\n"
)

# Make 3 sequential calls; the coordinator tracks them
for i in range(3):
    result = adapter.generate(f"Reply with just the number {i + 1}.")
    status = adapter.rate_limit_status()
    print(
        f"  Call {i + 1}: '{result.text.strip()}'  "
        f"(req_min_used={status['req_minute_used']}, "
        f"tokens_min_used={status['tok_minute_used']})"
    )

print(f"\nTotal requests recorded: {adapter.rate_limit_status()['total_requests']}")
print(f"Total tokens recorded:   {adapter.rate_limit_status()['total_tokens']}")

adapter.unload()

# -----------------------------------------------------------
# 3. Async burst with automatic throttling
# -----------------------------------------------------------
print("\n=== Async burst: 5 concurrent calls (coordinator paces them) ===\n")


async def burst_demo() -> None:
    adapter2 = CerebrasAdapter(model_name=MODEL)
    adapter2.load()

    async def one_call(n: int) -> str:
        t0 = time.monotonic()
        result = await adapter2.async_generate(f"Say only: CALL_{n}_OK")
        elapsed = time.monotonic() - t0
        return f"  Call {n}: '{result.text.strip()}'  ({elapsed:.2f}s)"

    outputs = await asyncio.gather(*[one_call(n) for n in range(1, 6)])
    for line in outputs:
        print(line)

    status = adapter2.rate_limit_status()
    print("\nCoordinator stats:")
    print(f"  total_requests:      {status['total_requests']}")
    print(f"  total_throttled:     {status['total_throttled']}")
    print(f"  total_throttle_secs: {status['total_throttle_seconds']:.3f}s")

    adapter2.unload()


asyncio.run(burst_demo())

print("\nDone.")
