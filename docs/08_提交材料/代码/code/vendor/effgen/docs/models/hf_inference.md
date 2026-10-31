# HuggingFace Inference API

`HFInferenceAdapter` connects to the HuggingFace Inference Providers API
(a.k.a. **HF Router**) — a single OpenAI-compatible surface that fronts a
rotating set of third-party providers (Together, Novita, Sambanova,
Fireworks, Cerebras, Groq, Cohere, …). One token (`HF_TOKEN`) gives you
access to all of them.

Two modes are supported:

1. **Router (default)** — any model in the live router catalog.  effGen
   ships a snapshot of every model + its providers + per-provider pricing
   and refreshes on demand.
2. **Dedicated Endpoints** — your own private/fine-tuned model at a custom
   URL (`endpoint_url=`).

## Quick start

```python
from effgen.models.hf_inference_adapter import HFInferenceAdapter

adapter = HFInferenceAdapter("Qwen/Qwen2.5-7B-Instruct")
adapter.load()

result = adapter.generate("What is the capital of France?")
print(result.text)   # "Paris"

adapter.unload()
```

## Setup

```bash
pip install "effgen[hf]"
```

Set your token in `.env` or shell:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Streaming

```python
for chunk in adapter.generate_stream("Count 1 to 10."):
    print(chunk, end="", flush=True)
```

## Native tool calling

Models whose registry entry has `supports_native_tools=True` accept
OpenAI-style function calling:

```python
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

result = adapter.generate_with_tools("Weather in Paris?", tools=[tool])
print(result.metadata["tool_calls"])
```

The shape of `metadata["tool_calls"]` is the same for every adapter — see
[Tool calls](tool-calls.md).

For models without native tools, use an Agent with
`tool_calling_mode="react"`.

## Provider routing

The adapter passes `provider="auto"` to the HF Router by default so the
router picks any live backend that hosts the model. To pin a specific
provider (e.g. for latency, pricing, or compliance reasons), pass
`provider=` explicitly:

```python
adapter = HFInferenceAdapter(
    "Qwen/Qwen2.5-72B-Instruct", provider="together"
)
```

Discover what's available for a model:

```python
from effgen import hf_list_providers_for, hf_cheapest_provider

print(hf_list_providers_for("Qwen/Qwen2.5-72B-Instruct"))
print(hf_cheapest_provider("Qwen/Qwen2.5-72B-Instruct"))
```

## Dedicated Endpoints

Point at a private Inference Endpoint:

```python
adapter = HFInferenceAdapter(
    model_name="my-private-model",
    endpoint_url="https://my-endpoint.endpoints.huggingface.cloud",
)
```

When `endpoint_url` is set, all requests go to that URL and the
`model_name` is not sent in the request payload — the endpoint already
knows its model.

## Via `load_model`

```python
from effgen.models import load_model

model = load_model("Qwen/Qwen2.5-7B-Instruct", provider="hf_inference")
result = model.generate("Tell me about quantum computing.")
```

## Errors

| Exception | When |
|-----------|------|
| `ModelAuthError` | 401/403 — bad / missing `HF_TOKEN` |
| `ModelNotFoundError` | The model ID doesn't exist on the Hub |
| `ModelUnavailableError` | The model exists but no provider currently serves it; `e.suggestions` lists alternatives |
| `RuntimeError` | Anything else (network, timeout) |

```python
from effgen.models.errors import ModelUnavailableError

try:
    adapter.generate("Hello")
except ModelUnavailableError as e:
    print("Try one of:", e.suggestions)
```

## Dynamic model catalog

Availability and pricing on the HF Router rotate frequently. effGen ships
a bundled snapshot but also lets you refresh on demand:

```python
from effgen import hf_refresh_models, hf_check_drift, hf_catalog_summary

# Print the active snapshot stats
print(hf_catalog_summary())
# {'fetched_at': '2026-04-29', 'total_models': 124, 'tool_capable': 83, ...}

# Compare bundled vs live and warn on drift
report = hf_check_drift()
print(report)  # added / removed / pricing_changed

# Pull the live catalog and update the in-process registry
# (also persists to ~/.effgen/cache/hf_inference_catalog.json so the next
#  process start picks it up automatically)
hf_refresh_models()
```

Discover models programmatically:

```python
from effgen import (
    hf_available_models,
    hf_tool_capable_models,
    hf_serverless_models,
    hf_get_model_info,
)

print(len(hf_available_models()))           # ≈124
print(hf_tool_capable_models()[:10])        # tool-capable subset
info = hf_get_model_info("Qwen/Qwen2.5-72B-Instruct")
print(info["context"], info["pricing_per_1m_input"], info["providers"])
```

## Featured models

Snapshot taken 2026-04-29.  Run `hf_refresh_models()` for the live list.

| Model | Family | Tools |
|-------|--------|-------|
| `Qwen/Qwen2.5-7B-Instruct` | Qwen 2.5 | ✓ |
| `Qwen/Qwen2.5-72B-Instruct` | Qwen 2.5 | ✓ |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Qwen 3 | ✓ |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 | ✓ |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 | ✓ |
| `deepseek-ai/DeepSeek-V3.1` | DeepSeek V3 | ✓ |
| `google/gemma-3-27b-it` | Gemma 3 | ✓ |
| `moonshotai/Kimi-K2-Instruct` | Kimi | ✓ |

## Rate limits and known gap

HF enforces limits per user tier (free / PRO / Enterprise) and per backend
provider rather than simple RPM/TPM windows. The `RateLimitCoordinator` is
wired with conservative defaults as a local circuit-breaker — it does NOT
accurately model the server-side limits.

For production workloads, upgrade to HF PRO or use dedicated Endpoints.

## Pricing

The router's per-provider pricing (input / output USD per 1M tokens) is
folded into the bundled registry, and `CostTracker` records it
automatically per call.  Free-tier providers report `0`.
