# Together AI Backend

effGen ships a full-featured `TogetherAdapter` for the Together AI inference API. It supports 168 chat models (as of 2026-08-07) including serverless access (no dedicated endpoint required), native tool-calling on 66 models, real token-by-token streaming, and per-request cost tracking with official Together AI pricing.

## Setup

```bash
pip install 'effgen[together]'
```

Set your API key:

```bash
export TOGETHER_API_KEY=your_key_here
# or in .env
TOGETHER_API_KEY=your_key_here
```

## Quick Start

```python
from effgen.models.together_adapter import TogetherAdapter

# Default model: Qwen/Qwen3.5-9B (cheapest serverless model with tool support)
adapter = TogetherAdapter()
adapter.load()

result = adapter.generate("What is the capital of France?")
print(result.text)
adapter.unload()
```

## Model Selection

```python
from effgen import TogetherAdapter, together_serverless_models, together_pricing_table

# See all serverless models (no dedicated endpoint needed)
print(together_serverless_models())

# Full pricing table sorted by input cost
for row in together_pricing_table():
    print(f"{row['model_id']:60s}  ${row['input_per_1m_usd']:.4f}/1M in  ${row['output_per_1m_usd']:.4f}/1M out  serverless={row['serverless']}")
```

### Recommended Serverless Models

| Model | Context | Input $/1M | Output $/1M | Tools |
|-------|---------|-----------|------------|-------|
| `openai/gpt-oss-20b` | 131K | $0.05 | $0.20 | Yes |
| `google/gemma-3n-E4B-it` | 32K | $0.06 | $0.12 | No |
| `Qwen/Qwen3.5-9B` | 262K | $0.10 | $0.15 | Yes |
| `openai/gpt-oss-120b` | 131K | $0.15 | $0.60 | Yes |
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | 32K | $0.30 | $0.30 | Yes |
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 131K | $0.88 | $0.88 | Yes |

`Qwen/Qwen3.5-9B`, the `gpt-oss` models and the DeepSeek-R1, QwQ, GLM-5, MiniMax
and Cogito families emit reasoning tokens before their answer. The catalog marks
them `reasoning`, and effGen gives a reasoning model a much larger default output
budget for that reason. If you pass `max_tokens` yourself, keep it generous: a
budget exhausted mid-thought returns empty `result.text` with
`metadata["truncated"] is True`. Unused budget is not billed.

Together sends the chain as `message.reasoning` beside an empty `content`, so a
turn that produced nothing but reasoning comes back with
`metadata["reasoning_only"] is True` and a `metadata["empty_response_reason"]`
naming the cap and the reasoning budget it spent. See
[API conventions](../api/conventions.md#reasoning-models-that-emit-no-visible-token)
for the full contract, including why the agent applies its stop sequences to a
reasoning model's answer rather than sending them to the provider.

## Streaming

```python
adapter = TogetherAdapter("meta-llama/Llama-3.3-70B-Instruct-Turbo")
adapter.load()

for chunk in adapter.generate_stream("Tell me a short story about a robot."):
    print(chunk, end="", flush=True)
print()
adapter.unload()
```

## Tool Calling (Native)

```python
import json
from effgen.models.together_adapter import TogetherAdapter

adapter = TogetherAdapter("meta-llama/Llama-3.3-70B-Instruct-Turbo")
adapter.load()

calc_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"],
        },
    },
}

result = adapter.generate_with_tools(
    "What is 17 * 23?",
    tools=[calc_tool],
)

tool_calls = result.metadata["tool_calls"]
print(f"Tool calls: {tool_calls}")
# Execute tool and continue conversation...
adapter.unload()
```

The shape of `metadata["tool_calls"]` is the same for every adapter — see
[Tool calls](tool-calls.md).

## Via ModelLoader

```python
from effgen.models import ModelLoader

loader = ModelLoader()
model = loader.load_model(
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    provider="together",
)
result = model.generate("Hello!")
```

## Cost Tracking

Costs use official Together AI pricing (fetched 2026-04-28). Zero for free/dedicated-endpoint models.

```python
from effgen.models._cost import CostTracker

adapter = TogetherAdapter("Qwen/Qwen3.5-9B")
adapter.load()
adapter.generate("Summarize AI in one sentence.")
adapter.unload()

tracker = CostTracker.get()
print(tracker.summary())
```

## Registry Drift Detection

The bundled registry is a snapshot of the Together AI catalog. To check if it's still in sync with the live API:

```python
from effgen.models.together_models import refresh_models

drift = refresh_models()
# Prints a warning if new/removed models or pricing changes are detected.
# Returns dict with: new_models, removed_models, pricing_changes

if drift["new_models"]:
    print("New models available:", drift["new_models"])
```

effGen will **always** use the bundled registry as the offline fallback. `refresh_models()` is purely informational — it never mutates the local registry.

## Model Coverage (2026-08-07 snapshot)

- **Total chat models**: 168
- **Serverless (no endpoint needed)**: 25
- **Tool-capable**: 66
- **Registry date**: 2026-08-07
- **Families**: Llama 3/4, Qwen 2.5/3/3.5, DeepSeek V3/R1, Mistral/Mixtral, Gemma 3/4, OpenAI OSS, GLM, Kimi, MiniMax, Nemotron, Cogito, LFM, and more

## Dedicated Endpoint Models

Some models listed in the registry require a dedicated endpoint started in your Together console. If you try to use one without an active endpoint, effGen raises a clear error:

```
RuntimeError: Together model 'Qwen/Qwen2.5-72B-Instruct-Turbo' requires a dedicated endpoint
that is not running. Start it at https://api.together.ai/models/...
or use a serverless model: [...]
```
