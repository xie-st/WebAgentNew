# Cerebras Backend

effGen supports [Cerebras Cloud](https://inference.cerebras.ai/) as a hosted
inference backend via `CerebrasAdapter`.  The models the live API currently
serves are registered, with automatic rate-limit enforcement built in. Run
`effgen models refresh --provider cerebras` to re-sync the catalog if it drifts.

## Setup

```bash
pip install "effgen[cerebras]"
```

Set your API key (get one at <https://cloud.cerebras.ai/>):

```bash
export CEREBRAS_API_KEY="your-key-here"
# or place it in ~/.effgen/.env
```

## Quick start

```python
from effgen import CerebrasAdapter

adapter = CerebrasAdapter(model_name="gpt-oss-120b")
adapter.load()

result = adapter.generate("What is 7 * 8?")
print(result.text)
print("tokens:", result.metadata["prompt_tokens"], "->", result.metadata["completion_tokens"])

adapter.unload()
```

### Via `load_model`

```python
from effgen.models import load_model

model = load_model("gpt-oss-120b", provider="cerebras")
print(model.generate("Hello!").text)
model.unload()
```

### Async generation (recommended in async contexts)

```python
import asyncio
from effgen import CerebrasAdapter

async def main():
    adapter = CerebrasAdapter(model_name="gpt-oss-120b")
    adapter.load()
    result = await adapter.async_generate("Explain quantum entanglement briefly.")
    print(result.text)
    adapter.unload()

asyncio.run(main())
```

## Supported models

The catalog tracks the models the live API currently serves. Use
`effgen models list --provider cerebras` (or `available_models()` in code) for
the authoritative, up-to-date list; the snapshot below is a point-in-time view.

| Model | Max output | RPM | RPH | RPD | TPM | Free-tier callable |
|-------|------------|-----|-----|-----|-----|--------------------|
| `gpt-oss-120b` | 32k | 30 | 900 | 14 400 | 64 000 | ✓ |
| `zai-glm-4.7` | 40k | 10 | 100 | 100 | 60 000 | ✓ |

Cerebras rotates its hosted line-up frequently. The previously-listed
`llama3.1-8b` and `qwen-3-235b-a22b-instruct-2507` have been retired and now
return `404 model_not_found`; effGen suggests the nearest live alternative when
you request a model that is no longer served. If your catalog looks stale, run
`effgen models refresh --provider cerebras`.

### Paid-tier limits

The Pay-as-You-Go tier has no hourly/daily caps and significantly higher
per-minute limits than the free tier.
The effGen `RateLimitCoordinator` is initialised with free-tier limits by
default — construct `CerebrasAdapter(model_name=..., enable_rate_limiting=False)`
or pass a custom `RateLimitCoordinator` if you're on the paid tier.

### Inspect the registry in code

```python
from effgen.models.cerebras_models import available_models, free_tier_models, model_info

print(available_models())          # all 2 models
print(free_tier_models())          # models accessible on the free tier

info = model_info("gpt-oss-120b")
print(info["rpm"], info["tpm"])    # 30, 64000
```

## Rate-limit coordinator

Each `CerebrasAdapter` instance comes with a built-in `RateLimitCoordinator`
that tracks sliding-window RPM / RPH / RPD and TPM / TPH / TPD limits.

When you call `generate()` or `async_generate()`, the coordinator:
1. Estimates the token cost from the prompt.
2. Blocks with `asyncio.sleep` if a per-minute / per-hour limit would be exceeded.
3. Records the actual tokens used after each call.

```python
adapter = CerebrasAdapter(model_name="gpt-oss-120b")
adapter.load()

result = adapter.generate("Hello!")

# Inspect coordinator state
status = adapter.rate_limit_status()
print(status["req_minute_used"], "/", status["req_minute_limit"])  # e.g. 1 / 30
print(status["total_throttled"])   # number of requests that were delayed
```

To disable rate limiting (e.g., in tests or when managing limits externally):

```python
adapter = CerebrasAdapter(model_name="gpt-oss-120b", enable_rate_limiting=False)
```

### RateLimitExceeded

If the **daily** budget is fully consumed, `acquire()` raises
`effgen.models._rate_limit.RateLimitExceeded` instead of sleeping indefinitely.

```python
from effgen import RateLimitExceeded

try:
    result = adapter.generate("hello")
except RateLimitExceeded as exc:
    print(f"Daily budget exhausted: {exc}")
```

## Agent integration

```python
from effgen.core.agent import Agent, AgentConfig
from effgen.models.cerebras_adapter import CerebrasAdapter
from effgen.tools.builtin import Calculator, DateTimeTool

adapter = CerebrasAdapter(model_name="gpt-oss-120b")
adapter.load()

config = AgentConfig(
    name="cerebras-assistant",
    model=adapter,
    tools=[Calculator(), DateTimeTool()],
    max_iterations=5,
)

with Agent(config) as agent:
    response = agent.run(
        "If I invest $5000 at 7% annual interest for 10 years, "
        "what is the final amount? Use compound interest: A = P*(1+r)^t"
    )
    print(response.output)
```

## Streaming

`generate_stream()` yields tokens incrementally as they arrive:

```python
adapter = CerebrasAdapter(model_name="gpt-oss-120b")
adapter.load()
for chunk in adapter.generate_stream("Explain streaming in one sentence."):
    print(chunk, end="", flush=True)
adapter.unload()
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `CEREBRAS_API_KEY` | Cerebras Cloud API key (required) |

## Examples

See `examples/cerebras/` for runnable examples:

| File | Description |
|------|-------------|
| `basic_cerebras.py` | Simple generation, token counting |
| `cerebras_agent.py` | Agent with Calculator and DateTimeTool |
| `cerebras_all_models.py` | Compare all models in parallel |
| `cerebras_rate_limits.py` | Rate-limit coordinator demo |
| `cerebras_hard_agent.py` | Hard multi-step agentic tasks |
| `cerebras_multi_turn.py` | Multi-turn conversation |
