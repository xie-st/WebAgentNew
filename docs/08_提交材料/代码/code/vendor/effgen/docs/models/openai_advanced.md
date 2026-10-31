# OpenAI Advanced Features: Prompt Caching & Structured Outputs

## Automatic Prompt Caching

OpenAI automatically caches prompt prefixes that are **≥1024 tokens** long.  
When consecutive API calls share the same prefix, cached tokens are billed at a lower rate and served with reduced latency.

### How It Works

1. You send a request with a large, stable system prompt (≥1024 tokens).
2. OpenAI caches the prefix on its backend.
3. Subsequent calls with the same prefix report `cached_input_tokens > 0` in usage.
4. Cached tokens are cheaper (see `openai_models.py` for `cached_input_price_per_1m` per model).

### Using `generate_with_system_prompt()`

This method places the system prompt at position 0 in the message list — a requirement for reliable prefix caching.

```python
from effgen.models.openai_adapter import OpenAIAdapter

SYSTEM_PROMPT = (
    "You are a research assistant specializing in physics and chemistry. " * 100
    # ≥1024 tokens
)

adapter = OpenAIAdapter(model_name="gpt-5.4-nano")
adapter.load()

for question in ["What is the speed of light?", "What is Planck's constant?"]:
    result = adapter.generate_with_system_prompt(
        prompt=question,
        system_prompt=SYSTEM_PROMPT,
    )
    cached = result.metadata["cached_input_tokens"]
    print(f"Answer: {result.text[:80]} | cached={cached} tokens")

adapter.unload()
```

### Reading Cache Stats

Every `GenerationResult` exposes cache info in `metadata`:

```python
result.metadata["cached_input_tokens"]   # tokens served from cache
result.metadata["prompt_tokens"]          # total input tokens
result.metadata["cost"]                   # cost this call (cached tokens cheaper)
```

### `AgentConfig.stable_system_prompt`

Set `stable_system_prompt=True` (default) on `AgentConfig` to signal that the agent's system prompt should remain fixed across all calls — enabling automatic prefix caching.

```python
from effgen.core.agent import Agent, AgentConfig

agent = Agent(config=AgentConfig(
    name="my-agent",
    model=adapter,
    system_prompt=SYSTEM_PROMPT,
    stable_system_prompt=True,   # default; prefix will be cache-eligible
))
```

### Models That Support Caching

All GPT-4o, GPT-4.1, GPT-5, GPT-5.4-nano/mini, and o-series models support caching.  
Check `OPENAI_MODELS[model_id]["supports_prompt_caching"]` in `openai_models.py`.

---

## Structured Outputs v2 (Strict JSON Schema)

OpenAI's structured outputs mode enforces your Pydantic schema at the token level — you always receive valid, parseable JSON that matches your model.

### Quick Start

```python
from typing import Literal
from pydantic import BaseModel
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.openai_schema import to_openai_schema
from effgen.models.errors import ModelRefusalError

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float

adapter = OpenAIAdapter(model_name="gpt-5.4-nano")
adapter.load()

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "SentimentResult",
        "schema": to_openai_schema(SentimentResult),
        "strict": True,
    },
}

try:
    result = adapter.generate_structured(
        prompt="Classify: 'I absolutely love this product!'",
        response_format=response_format,
    )
    parsed = SentimentResult.model_validate_json(result.text)
    print(parsed.sentiment, parsed.confidence)
except ModelRefusalError as e:
    print(f"Refused: {e.refusal_message}")

adapter.unload()
```

### `to_openai_schema(pydantic_model)`

Converts any Pydantic `BaseModel` subclass to an OpenAI-compatible JSON Schema dict.

Transformations applied:
- All `$ref` and `$defs` are **inlined** — no external references remain.
- Every object node gets `"additionalProperties": false`.
- Every object node gets an explicit `"required"` array listing all properties.

```python
from effgen.models.openai_schema import to_openai_schema
# Also available as:
from effgen import to_openai_schema

schema = to_openai_schema(SentimentResult)
# {
#   "type": "object",
#   "properties": {
#     "sentiment": {"enum": ["positive", "negative", "neutral"]},
#     "confidence": {"type": "number"}
#   },
#   "required": ["sentiment", "confidence"],
#   "additionalProperties": false
# }
```

**Nested models are fully supported:**

```python
class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address

schema = to_openai_schema(Person)
# Address is inlined — no $ref left in the output
```

### `ModelRefusalError`

When OpenAI returns a `refusal` field instead of content, `generate_structured()` raises `ModelRefusalError`.

```python
from effgen.models.errors import ModelRefusalError
# Also available as:
from effgen import ModelRefusalError

try:
    result = adapter.generate_structured(...)
except ModelRefusalError as e:
    print(e.refusal_message)   # raw refusal string
    print(e.model_name)        # model that refused
```

### Combining Caching + Structured Outputs

Use `system_prompt` in `generate_structured()` to get both features in one call:

```python
result = adapter.generate_structured(
    prompt="Classify this review: ...",
    response_format=response_format,
    system_prompt=LARGE_STABLE_SYSTEM_PROMPT,  # cached after first call
)
```

### Supported Models

All GPT-4o, GPT-4.1, GPT-5, and GPT-5.4 chat models support structured outputs.  
Reasoning models (o-series) may have limited structured-output support — test with your specific model.

---

## Examples

| File | What it shows |
|------|--------------|
| `examples/openai/prompt_caching.py` | Cache hits across 5 sequential calls, cost savings |
| `examples/openai/structured_outputs.py` | Sentiment, nested schema, refusal handling |
| `examples/openai/caching_and_structured_agent.py` | Full agent: caching + structured + tools |
