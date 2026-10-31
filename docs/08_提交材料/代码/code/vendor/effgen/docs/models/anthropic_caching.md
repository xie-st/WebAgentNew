# Anthropic Prompt Caching

Anthropic's Claude models support explicit prompt caching via `cache_control`
markers. Cached prefixes are reused across requests, reducing latency and cost
for repeated long context (system prompts, tool specs, documents).

> **Dev note:** Anthropic live tests are skipped in this environment (no API key).
> All caching functionality is unit-tested with mocked responses.

---

## How it works

You mark specific content blocks with `cache_control`. The first request that
hits a marked block writes the prefix to cache. Subsequent requests that share
the same prefix up to a breakpoint read from cache at ~10× cheaper cost.

### Cache TTL

| TTL | Write cost | Read cost | Recommended for |
|-----|------------|-----------|-----------------|
| `"5m"` (default) | 1.25× input | 0.1× input | Most use cases |
| `"1h"` | 2× input | 0.1× input | Long-lived contexts, heavy tools |

### Minimum block size (tokens)

Blocks below the threshold are silently billed at normal rates — no error.

| Model | Min tokens |
|-------|-----------|
| claude-opus-4-7 | 4 096 |
| claude-sonnet-4-6 | 2 048 |
| claude-haiku-4-5 | 4 096 |
| claude-sonnet-4-5, claude-opus-4-1, claude-3-7-sonnet | 1 024 |
| claude-3-5-sonnet | 2 048 |
| claude-3-haiku | 1 024 |

---

## API

### `mark_cached(block, ttl="5m")`

Attaches `cache_control` to a single content block dict (or converts a plain
string to a text block first).

```python
from effgen.models.anthropic_cache import mark_cached

block = mark_cached({"type": "text", "text": long_document})
block = mark_cached(long_document)           # string shorthand
block = mark_cached(long_document, ttl="1h") # 1-hour cache
```

### `apply_cache_to_system(system_prompt, ttl="5m")`

Converts a system prompt (string or list) to a list of content blocks with
`cache_control` on the **last** block — the recommended pattern.

```python
from effgen.models.anthropic_cache import apply_cache_to_system

cached_system = apply_cache_to_system("You are a helpful assistant.")

adapter.generate("Hello", system_prompt=cached_system)
```

### `apply_cache_to_last_tool(tools, ttl="5m")`

Marks the last entry in a tool list with `cache_control`.

```python
from effgen.models.anthropic_cache import apply_cache_to_last_tool

tools = adapter.build_tool_specs(my_tools)
cached_tools = apply_cache_to_last_tool(tools)
```

---

## 4-breakpoint limit

Anthropic allows a maximum of **4** `cache_control` markers per request across
tools + system + messages combined. Exceeding this raises `ValueError` before
any API call is made.

```
Priority if you need to reduce breakpoints:
  1. System prompt last block  (highest value — present every turn)
  2. Last tool spec            (tool list seldom changes)
  3. Message / document blocks (situational)
```

The adapter validates the count automatically:

```python
# This would raise ValueError: "Too many cache_control breakpoints: 5 (max 4)"
system_with_5_markers = [
    {"type": "text", "text": f"block {i}",
     "cache_control": {"type": "ephemeral"}}
    for i in range(5)
]
adapter.generate("hi", system_prompt=system_with_5_markers)
```

---

## Reading cache usage from the response

The adapter surfaces Anthropic's cache token counts in `result.metadata`:

```python
result = adapter.generate("What is the capital of France?", system_prompt=cached_system)

print(result.metadata["cached_input_tokens"])   # tokens read from cache (hit)
print(result.metadata["cache_creation_tokens"]) # tokens written to cache (miss)
```

Both fields are always present and default to `0` when prompt caching was not
used (e.g., block too short, or no `cache_control` markers in the request).

---

## AgentConfig integration

Set these flags on `AgentConfig` when using an `AnthropicAdapter` with an
`Agent`:

```python
from effgen.core.agent import Agent, AgentConfig
from effgen.models.anthropic_adapter import AnthropicAdapter

adapter = AnthropicAdapter(model_name="claude-sonnet-4-6")
adapter.load()

config = AgentConfig(
    name="my-agent",
    model=adapter,
    system_prompt="You are a research assistant. " * 200,
    cache_system_prompt=True,  # default True — marks system prompt last block
    cache_tools=True,          # default True — marks last tool spec
)

agent = Agent(config)
```

When `cache_system_prompt=True`, calling `agent._get_anthropic_system()` returns
the system prompt as a `cache_control`-marked list of content blocks.  When
`cache_tools=True`, `agent._get_anthropic_tools(tool_specs)` marks the last spec.

These helpers are used internally by agent code paths that build Anthropic
requests.  For direct adapter calls, use `apply_cache_to_system()` and
`apply_cache_to_last_tool()` explicitly.

---

## Cache evaluation order

Anthropic evaluates cache markers in this order:

```
tools → system → messages
```

A change at any level invalidates that level and everything after it. Put
your most stable content (tools) first for best cache utilization.

---

## Full example

```python
from effgen.models.anthropic_adapter import AnthropicAdapter
from effgen.models.anthropic_cache import apply_cache_to_system, apply_cache_to_last_tool

adapter = AnthropicAdapter(model_name="claude-sonnet-4-6")
adapter.load()

# Long, stable system prompt — mark for caching
system = apply_cache_to_system(
    "You are an expert Python engineer with deep knowledge of "
    "distributed systems, databases, and API design. " * 200
)

# Tool specs — mark last one
tools = [
    {"name": "search", "description": "Search the web", ...},
    {"name": "calculate", "description": "Run calculations", ...},
]
cached_tools = apply_cache_to_last_tool(tools)

# First call — cache miss; writes to cache
result1 = adapter.generate_with_tools("Explain async/await", cached_tools, system_prompt=system)
print(result1.metadata["cache_creation_tokens"])  # > 0 on first call

# Second call — cache hit (within 5 minutes)
result2 = adapter.generate_with_tools("How does GIL work?", cached_tools, system_prompt=system)
print(result2.metadata["cached_input_tokens"])    # > 0 on cache hit

adapter.unload()
```
