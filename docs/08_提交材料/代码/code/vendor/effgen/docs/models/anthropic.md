# Anthropic Claude

effGen supports the full Anthropic Claude family through `AnthropicAdapter`.

> **Note:** Anthropic features in v0.2.2 are implemented and unit-tested but not live-tested
> (no Anthropic API key available in the development environment). Validate the adapter with your
> own key before deploying.

## Supported models

Verified against Anthropic's [models overview](https://docs.anthropic.com/en/docs/about-claude/models/overview).

| Model ID | Family | Context | Max output | Extended thinking | Caching |
|---|---|---|---|---|---|
| `claude-opus-4-7` | Opus 4.7 | 1 M | 128 K | adaptive only* | ✓ |
| `claude-sonnet-4-6` | Sonnet 4.6 | 1 M | 64 K | ✓ | ✓ |
| `claude-haiku-4-5-20251001` | Haiku 4.5 | 200 K | 64 K | ✓ | ✓ |
| `claude-opus-4-6` | Opus 4.6 | 1 M | 128 K | ✓ | ✓ |
| `claude-sonnet-4-5-20250929` | Sonnet 4.5 | 200 K | 64 K | ✓ | ✓ |
| `claude-opus-4-5-20251101` | Opus 4.5 | 200 K | 64 K | ✓ | ✓ |
| `claude-opus-4-1-20250805` | Opus 4.1 | 200 K | 32 K | ✓ | ✓ |
| `claude-3-7-sonnet-20250219` | Sonnet 3.7 | 200 K | 64 K | ✓ | ✓ |
| `claude-3-5-sonnet-20241022` | Sonnet 3.5 | 200 K | 8 K | — | ✓ |
| `claude-3-opus-20240229` | Opus 3 | 200 K | 4 K | — | ✓ |
| `claude-3-haiku-20240307` | Haiku 3 | 200 K | 4 K | — | ✓ |

\* Opus 4.7 uses adaptive thinking (the model decides when to reason internally).
The explicit extended-thinking API parameter (`thinking={"type":"enabled", ...}`)
is **not** supported on Opus 4.7 — choose Sonnet 4.6 or Haiku 4.5 if you need a
verifiable thinking trace under your own budget control.

The registry is in `effgen/models/anthropic_models.py`.

## Basic usage

```python
from effgen.models import AnthropicAdapter

with AnthropicAdapter(model_name="claude-opus-4-7") as adapter:
    result = adapter.generate("What is the capital of France?")
    print(result.text)
```

## Extended thinking

Claude Sonnet 4.6, Haiku 4.5, and Claude 3.7 Sonnet support explicit extended
thinking — an internal reasoning phase under a controllable token budget that
improves accuracy on complex tasks.

```python
from effgen.models import AnthropicAdapter
from effgen.models.base import GenerationConfig

cfg = GenerationConfig(
    thinking={"type": "enabled", "budget_tokens": 8000},
    max_tokens=16000,  # must be > budget_tokens
)

with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    result = adapter.generate("Prove that sqrt(2) is irrational.", config=cfg)

print(result.text)
print("Thinking trace:", result.metadata.get("thinking"))
```

**Constraints:**
- `max_tokens` must be greater than `budget_tokens`.
- Temperature is automatically forced to `1.0` when thinking is enabled (Anthropic requirement).
- `thinking` is silently ignored for models that do not expose extended thinking
  (Opus 4.7, Claude 3.5 and earlier). A debug log notes when this happens.
- Thinking is billed as output, so a `max_tokens` spent entirely on it returns
  an empty answer you paid for. That turn comes back with
  `metadata["reasoning_only"] is True` and a `metadata["empty_response_reason"]`
  naming the cap and what it spent. See
  [API conventions](../api/conventions.md#reasoning-models-that-emit-no-visible-token).

## Redacted thinking (multi-turn)

When Anthropic's safety filters redact part of the thinking trace, the response contains
`redacted_thinking` blocks. These **must be preserved verbatim** when you submit the next
turn — stripping them causes an HTTP 400 error.

effGen handles this automatically via `raw_content_blocks`:

```python
from effgen.models import AnthropicAdapter
from effgen.models.base import GenerationConfig

cfg = GenerationConfig(
    thinking={"type": "enabled", "budget_tokens": 4000},
    max_tokens=8192,
)

adapter = AnthropicAdapter(model_name="claude-sonnet-4-6")
adapter.load()

# Turn 1
r1 = adapter.generate("What is the square root of 144?", config=cfg)
print(r1.text)

# Build the assistant message — includes redacted_thinking if present
asst_msg = AnthropicAdapter.build_assistant_message(r1)

# Turn 2 — pass the full history including the preserved blocks
history = [
    {"role": "user", "content": "What is the square root of 144?"},
    asst_msg,  # contains raw_content_blocks with redacted_thinking preserved
    {"role": "user", "content": "And the cube root of 27?"},
]
r2 = adapter.generate_with_history(history, config=cfg)
print(r2.text)

adapter.unload()
```

`build_assistant_message(result)` returns:
```python
{"role": "assistant", "content": <raw_content_blocks>}
```
where `raw_content_blocks` is the exact list Anthropic returned, including any
`redacted_thinking` entries that must not be stripped.

## Tool use

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

with AnthropicAdapter(model_name="claude-opus-4-7") as adapter:
    result = adapter.generate_with_tools("What's the weather in Paris?", tools)

for tool_call in result.metadata["tool_uses"]:
    print(tool_call["name"], tool_call["input"])
```

## Streaming

### Basic streaming (text only)

```python
with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    for chunk in adapter.generate_stream("Tell me a story"):
        print(chunk, end="", flush=True)
```

`generate_stream()` yields plain text strings. Thinking deltas, tool_use blocks,
and redacted_thinking events are consumed internally and not yielded.

### Full-fidelity streaming with typed chunks

Use `generate_stream_full()` to receive all block types as typed `StreamChunk` objects:

```python
from effgen.models import AnthropicAdapter, StreamChunk
from effgen.models.base import GenerationConfig

cfg = GenerationConfig(
    thinking={"type": "enabled", "budget_tokens": 8000},
    max_tokens=16000,
)

with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    for chunk in adapter.generate_stream_full("Prove that sqrt(2) is irrational.", config=cfg):
        if chunk.type == "thinking":
            print(f"[thinking] {chunk.text}", end="", flush=True)
        elif chunk.type == "text":
            print(chunk.text, end="", flush=True)
        elif chunk.type == "tool_use":
            print(f"\n[tool] {chunk.data['name']}({chunk.data['input']})")
        elif chunk.type == "redacted_thinking":
            print("\n[redacted thinking block — preserved for multi-turn]")
```

**`StreamChunk` fields:**

| `type` | `text` | `data` | Notes |
|---|---|---|---|
| `"text"` | answer delta | `{}` | Incremental answer text |
| `"thinking"` | thinking delta | `{}` | Visible reasoning trace (budget-controlled) |
| `"redacted_thinking"` | `""` | `{"type": "redacted_thinking", "data": "<sig>"}` | Safety-filtered thinking block; preserve verbatim on re-submit |
| `"tool_use"` | `""` | `{"type": "tool_use", "id": ..., "name": ..., "input": {...}}` | Fully accumulated parallel tool call |

### Streaming with parallel tool calls

Claude 4.x can emit multiple `tool_use` blocks in a single response (parallel function
calling). `generate_stream_full()` accumulates each block's `input_json_delta` fragments,
parses them on `content_block_stop`, and yields one `StreamChunk(type="tool_use")` per call:

```python
with AnthropicAdapter(model_name="claude-sonnet-4-6") as adapter:
    tool_calls = []
    for chunk in adapter.generate_stream_full("What is 2+3 AND 7*8?"):
        if chunk.type == "tool_use":
            tool_calls.append(chunk.data)

# tool_calls may contain multiple entries for parallel calls
for call in tool_calls:
    print(call["name"], call["input"])
```

### Streaming and redacted_thinking in multi-turn

When `generate_stream_full()` emits a `"redacted_thinking"` chunk, the corresponding
block must be preserved when building the assistant message for the next turn.
Use `generate()` + `build_assistant_message()` for multi-turn with thinking — it
handles preservation automatically via `raw_content_blocks`.

## Environment setup

Set `ANTHROPIC_API_KEY` in `~/.effgen/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

effGen loads this automatically via `python-dotenv`.

## Cost tracking

```python
adapter = AnthropicAdapter(model_name="claude-opus-4-7")
adapter.load()
adapter.generate("Hello")
print(f"Total cost: ${adapter.get_total_cost():.4f}")
print(f"Total tokens: {adapter.get_total_tokens()}")
adapter.reset_usage_stats()
adapter.unload()
```
