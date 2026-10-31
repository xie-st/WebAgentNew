# OpenAI Models in effGen

effGen's `OpenAIAdapter` supports the full OpenAI chat and reasoning model lineup.

## Supported Models

### Chat Models

| Model | Context | Max Output | Prompt Caching | Price (in / out per 1M) |
|-------|---------|------------|----------------|------------------------|
| `gpt-5` | 1,047,576 | 32,768 | ✓ | $10 / $30 |
| `gpt-5.4-mini` | 1,047,576 | 32,768 | ✓ | $0.40 / $1.60 |
| `gpt-5.4-nano` | 1,047,576 | 32,768 | ✓ | $0.10 / $0.40 |
| `gpt-4.1` | 1,047,576 | 32,768 | ✓ | $2 / $8 |
| `gpt-4.1-mini` | 1,047,576 | 32,768 | ✓ | $0.40 / $1.60 |
| `gpt-4.1-nano` | 1,047,576 | 32,768 | ✓ | $0.10 / $0.40 |
| `gpt-4o` | 128,000 | 16,384 | ✓ | $2.50 / $10 |
| `gpt-4o-mini` | 128,000 | 16,384 | ✓ | $0.15 / $0.60 |

### Reasoning Models (o-series)

Reasoning models perform extended internal thinking before producing an answer. They accept a `reasoning_effort` parameter that trades cost/latency for depth of reasoning.

| Model | Context | Max Output | Valid `reasoning_effort` |
|-------|---------|------------|--------------------------|
| `o4-mini` | 200,000 | 100,000 | `low`, `medium`, `high`, `xhigh` |
| `o3` | 200,000 | 100,000 | `low`, `medium`, `high`, `xhigh` |
| `o3-mini` | 200,000 | 100,000 | `low`, `medium`, `high`, `xhigh` |
| `o1` | 200,000 | 100,000 | `low`, `medium`, `high`, `xhigh` |
| `o1-mini` | 128,000 | 65,536 | `low`, `medium`, `high`, `xhigh` |

> **Note:** `minimal` is not supported for o3/o4 models — use `low` as the cheapest option.

## Quick Start

```python
import os
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.base import GenerationConfig

# OPENAI_API_KEY must be set in env or .env
adapter = OpenAIAdapter("gpt-5.4-nano")
adapter.load()
result = adapter.generate("Explain what a transformer is in one sentence.")
print(result.text)
adapter.unload()
```

## Using `reasoning_effort`

Pass `reasoning_effort` through `GenerationConfig` to control how much the model reasons:

```python
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.models.base import GenerationConfig

adapter = OpenAIAdapter("o4-mini")
adapter.load()

# Low effort — fastest, cheapest
result = adapter.generate(
    "What is the optimal strategy in a repeated prisoner's dilemma?",
    config=GenerationConfig(reasoning_effort="low"),
)

# High effort — deeper reasoning, more tokens
result_deep = adapter.generate(
    "What is the optimal strategy in a repeated prisoner's dilemma?",
    config=GenerationConfig(reasoning_effort="high"),
)

adapter.unload()
```

### Rules

- `reasoning_effort` is valid only for o-series models (`o1`, `o3`, `o3-mini`, `o4-mini`, etc.)
- Setting `reasoning_effort` on a chat model (e.g., `gpt-4o-mini`) is silently ignored — no error.
- Setting an invalid value (e.g., `"absurd"`) always raises `ValueError`, listing the valid options.
- Valid values: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`
  - `"minimal"` is only supported on o1-series; o3/o4 models require at least `"low"`.

## Tool Calling

All OpenAI chat and most reasoning models support native function calling:

```python
from effgen.models.openai_adapter import OpenAIAdapter

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

adapter = OpenAIAdapter("gpt-5.4-nano")
adapter.load()
result = adapter.generate_with_tools("What is the weather in Tokyo?", tools=tools)
tool_calls = result.metadata["tool_calls"]
print(tool_calls)
adapter.unload()
```

The shape of `metadata["tool_calls"]` is the same for every adapter — see
[Tool calls](tool-calls.md).

## Using with the Agent

```python
from effgen import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.calculator import Calculator

model = OpenAIAdapter("gpt-5.4-nano")
model.load()

agent = Agent(
    config=AgentConfig(name="math-agent", model=model, tools=[Calculator()]),
)
result = agent.run("What is 17 * 23 + sqrt(144)?")
print(result.output)
model.unload()
```

## Listing Models Programmatically

```python
from effgen import openai_available_models, openai_chat_models, openai_reasoning_models, openai_model_info

print(openai_available_models())     # all models
print(openai_chat_models())          # chat family only
print(openai_reasoning_models())     # o-series only
print(openai_model_info("o4-mini"))  # metadata dict
```

## Migration Note

If you previously used `gpt-4-turbo` or `gpt-3.5-turbo`, those models remain in the registry but are not recommended for new projects. Prefer `gpt-5.4-nano` (cheapest), `gpt-4o-mini` (proven quality), or `gpt-4.1-mini` for the best cost/quality tradeoff.

The `max_tokens` parameter is deprecated by OpenAI. The adapter now always uses `max_completion_tokens` internally — you don't need to change your code, as `GenerationConfig.max_tokens` is automatically mapped.
