# Replicate Backend

effGen's `ReplicateAdapter` connects to [Replicate](https://replicate.com) — a platform that hosts hundreds of open-source and commercial AI models, billed per second of GPU compute.

## Setup

```bash
pip install 'effgen[replicate]'
```

Set your API token:

```bash
export REPLICATE_API_TOKEN=r8_...
```

Or add it to `.env`:

```
REPLICATE_API_TOKEN=r8_...
```

Add billing at https://replicate.com/account/billing before running models.

---

## Quick Start

```python
from effgen.models.replicate_adapter import ReplicateAdapter

adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct")
adapter.load()

result = adapter.generate("What is the capital of France?")
print(result.text)
print("Compute seconds:", result.metadata["compute_seconds"])

adapter.unload()
```

---

## Streaming

```python
adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct")
adapter.load()

for chunk in adapter.generate_stream("Count from 1 to 10."):
    print(chunk, end="", flush=True)

adapter.unload()
```

Streaming uses Replicate's SSE endpoint for real token-by-token delivery.  
For models that don't support SSE (`supports_streaming=False`), the adapter falls back to a polling-based iterator transparently.

---

## Native Tool Calling

IBM Granite supports native function-calling via the `tools` + `messages` input schema:

```python
from effgen.models.replicate_adapter import ReplicateAdapter

adapter = ReplicateAdapter("ibm-granite/granite-3.3-8b-instruct")
adapter.load()

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate math expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
            },
        },
    }
]

result = adapter.generate_with_tools("What is 17 * 23?", tools=tools)
print(result.text)
print("Tool calls:", result.metadata["tool_calls"])

adapter.unload()
```

The hosted model decides what it emits, so each call is coerced into the
shape every adapter reports — see [Tool calls](tool-calls.md).

For models that don't support native tools, `generate_with_tools()` raises `NotImplementedError` — use an Agent with `strategy="react"` instead.

---

## Polling Abstraction

Replicate's API is async-first: every call creates a prediction that you then poll until it succeeds or fails.  The adapter hides this entirely:

- `generate()` creates the prediction, polls with **exponential backoff** (1s → 1.5s → 2.25s … capped at 30s), and returns a `GenerationResult` when done.
- `generate_stream()` uses SSE streaming (or polling fallback) to yield tokens as they arrive.
- `ModelTimeoutError` is raised (not a generic `RuntimeError`) if polling exceeds `timeout=` seconds.

```python
from effgen.models.errors import ModelTimeoutError

adapter = ReplicateAdapter("meta/meta-llama-3-8b-instruct", timeout=60)
adapter.load()

try:
    result = adapter.generate("...")
except ModelTimeoutError as e:
    print(f"Timed out after {e.timeout_seconds}s (prediction_id={e.prediction_id})")
```

---

## Compute Seconds Billing

Replicate bills per second of GPU compute (`predict_time` from prediction metrics).  effGen surfaces this in every result:

```python
result = adapter.generate("Hello")
print("compute_seconds:", result.metadata["compute_seconds"])
print("cost_usd:", result.metadata["cost_usd"])  # None when the registry has no rate
```

Typical hardware costs:

| Hardware    | Cost/second |
|-------------|-------------|
| T4 GPU      | $0.000225   |
| L40S GPU    | $0.000575   |
| A100 GPU    | $0.001400   |
| H100 GPU    | $0.001974   |

Some hosted models (Anthropic, OpenAI via Replicate) are billed per-token instead — `cost_per_second_usd=0` for those, with per-token pricing tracked separately.

---

## Model Registry

effGen ships a bundled registry of all models in the Replicate `language-models` collection (fetched 2026-04-29):

```python
from effgen.models.replicate_models import (
    REPLICATE_MODELS,
    available_models,
    tool_capable_models,
    streaming_models,
    print_registry_summary,
)

# List all registered models
print(available_models())

# Only models with native tool support
print(tool_capable_models())

# Summary table
print_registry_summary()
```

### Dynamic Drift Detection

Call `refresh_models()` to compare the bundled registry against the live Replicate catalog and find newly added or removed models:

```python
from effgen.models.replicate_models import refresh_models

report = refresh_models()
print("New models not in registry:", report["added"])
print("Registry models removed from Replicate:", report["removed"])
print("Registry count:", report["registry_count"])
print("Live count:", report["live_count"])
```

The adapter also warns at init time when the requested model is not in the registry:

```
WARNING: ReplicateAdapter: model 'org/new-model' is not in the bundled registry
(registry date: 2026-04-29). The model may be new — call
replicate_models.refresh_models() to check for drift.
```

---

## Full Model Table (2026-04-29)

| Model ID | Family | Context | Native Tools | Streaming | Cost/sec |
|----------|--------|---------|--------------|-----------|----------|
| `meta/meta-llama-3-8b-instruct` | llama3 | 8K | No | Yes | $0.000225 |
| `meta/meta-llama-3-70b-instruct` | llama3 | 8K | No | Yes | $0.000575 |
| `meta/meta-llama-3-8b` | llama3 | 8K | No | Yes | $0.000225 |
| `meta/meta-llama-3-70b` | llama3 | 8K | No | Yes | $0.000575 |
| `ibm-granite/granite-3.3-8b-instruct` | granite | 128K | **Yes** | Yes | $0.000225 |
| `deepseek-ai/deepseek-r1` | deepseek | 64K | No | Yes | $0.001400 |
| `deepseek-ai/deepseek-v3` | deepseek | 64K | No | Yes | $0.001400 |
| `deepseek-ai/deepseek-v3.1` | deepseek | 64K | No | Yes | $0.001400 |
| `qwen/qwen3-235b-a22b-instruct-2507` | qwen3 | 32K | No | Yes | $0.001974 |
| `moonshotai/kimi-k2.5` | kimi | 131K | No | Yes | $0.001974 |
| `google-deepmind/gemma-2b-it` | gemma | 8K | No | Yes | $0.000225 |
| `replicate/flan-t5-xl` | flan-t5 | 2K | No | No | $0.000225 |
| `stability-ai/stablelm-tuned-alpha-7b` | stablelm | 4K | No | Yes | $0.000225 |
| `replit/replit-code-v1-3b` | replit-code | 2K | No | Yes | $0.000225 |
| **Anthropic (via Replicate)** | | | | | |
| `anthropic/claude-3.7-sonnet` | claude | 200K | **Yes** | Yes | per-token |
| `anthropic/claude-3.5-haiku` | claude | 200K | **Yes** | Yes | per-token |
| `anthropic/claude-4.5-sonnet` | claude | 200K | **Yes** | Yes | per-token |
| `anthropic/claude-4.5-haiku` | claude | 200K | **Yes** | Yes | per-token |
| `anthropic/claude-4-sonnet` | claude | 200K | **Yes** | Yes | per-token |
| `anthropic/claude-opus-4.6` | claude | 200K | **Yes** | Yes | per-token |
| **OpenAI (via Replicate)** | | | | | |
| `openai/gpt-4o` | gpt4 | 128K | **Yes** | Yes | per-token |
| `openai/gpt-4o-mini` | gpt4 | 128K | **Yes** | Yes | per-token |
| `openai/gpt-4.1` | gpt4 | 1000K | **Yes** | Yes | per-token |
| `openai/gpt-4.1-mini` | gpt4 | 1000K | **Yes** | Yes | per-token |
| `openai/gpt-4.1-nano` | gpt4 | 1000K | **Yes** | Yes | per-token |
| `openai/o1` | o1 | 200K | **Yes** | No | per-token |
| `openai/o1-mini` | o1 | 128K | **Yes** | No | per-token |
| `openai/o4-mini` | o4 | 200K | **Yes** | No | per-token |
| `openai/gpt-5` | gpt5 | 1000K | **Yes** | Yes | per-token |
| `openai/gpt-5-mini` | gpt5 | 1000K | **Yes** | Yes | per-token |
| `openai/gpt-5-nano` | gpt5 | 1000K | **Yes** | Yes | per-token |
| `openai/gpt-5.2` | gpt5 | 1000K | **Yes** | Yes | per-token |
| `openai/gpt-oss-120b` | gpt-oss | 128K | **Yes** | Yes | $0.001400 |
| `openai/gpt-oss-20b` | gpt-oss | 128K | **Yes** | Yes | $0.000575 |
| **Google (via Replicate)** | | | | | |
| `google/gemini-2.5-flash` | gemini | 1000K | **Yes** | Yes | per-token |
| `google/gemini-3-pro` | gemini | 1000K | **Yes** | Yes | per-token |
| `google/gemini-3.1-pro` | gemini | 1000K | **Yes** | Yes | per-token |
| **xAI** | | | | | |

---

## Configuration Reference

```python
ReplicateAdapter(
    model_name="meta/meta-llama-3-8b-instruct",  # owner/name format
    api_token=None,          # falls back to REPLICATE_API_TOKEN env var
    timeout=300.0,           # seconds before ModelTimeoutError
    poll_interval=1.0,       # initial polling interval (grows with backoff)
    max_retries=4,           # retries on transient HTTP errors
    enable_rate_limiting=True,
    enable_cost_tracking=True,
    warn_unknown_model=True, # warns if model not in bundled registry
)
```

## Using with ModelLoader

```python
from effgen.models import load_model

model = load_model(
    "meta/meta-llama-3-8b-instruct",
    provider="replicate",
    timeout=120,
)
result = model.generate("Hello!")
```

## Error Handling

```python
from effgen.models.errors import ModelAuthError, ModelTimeoutError

try:
    result = adapter.generate("Hello")
except ModelAuthError as e:
    print(f"Auth failed for {e.provider}: {e.message}")
except ModelTimeoutError as e:
    print(f"Timed out after {e.timeout_seconds}s, prediction={e.prediction_id}")
except RuntimeError as e:
    if "insufficient credits" in str(e):
        print("Add billing at https://replicate.com/account/billing")
    else:
        raise
```
