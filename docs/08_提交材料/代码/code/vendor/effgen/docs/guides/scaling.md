# Scaling Guide

This guide covers scaling effGen for production workloads — from request throughput to multi-GPU model management.

## Request Queue Tuning

The production API server uses a priority queue with backpressure:

```python
from effgen.api.queue import RequestQueue, RequestPriority

queue = RequestQueue(
    max_size=1000,          # Max pending requests before backpressure (503)
    default_priority=RequestPriority.NORMAL,
)
```

Priority levels: `HIGH` > `NORMAL` > `LOW`. High-priority requests (e.g., from premium tenants) are always dequeued first.

When the queue is full, new requests receive a `QueueFullError` (HTTP 503). Clients should implement exponential backoff.

## Agent Pool Sizing

Pre-warm agents to avoid cold-start latency:

```python
from effgen.api.pool import AgentPool

pool = AgentPool(
    factory=lambda: create_agent("general", model),
    min_size=2,     # Always keep 2 agents warm
    max_size=10,    # Scale up to 10 under load
    idle_ttl=300,   # Reclaim idle agents after 5 minutes
)
```

**Sizing guidelines:**
- `min_size` = expected baseline concurrency
- `max_size` = peak concurrency (limited by GPU memory)
- `idle_ttl` = balance between memory and cold-start latency

## Model Pool & LRU Eviction

Manage multiple models across GPUs:

```python
from effgen.models.pool import ModelPool, PoolConfig

pool = ModelPool(config=PoolConfig(
    max_loaded_models=4,        # Keep at most 4 models in GPU memory
    gpu_memory_limit_gb=40,     # Total GPU memory budget
))

# Pre-warm critical models
pool.prewarm("Qwen/Qwen2.5-3B-Instruct")

# Models are loaded on demand and evicted LRU when limits are reached
model = pool.get_or_load("Qwen/Qwen2.5-7B-Instruct")
```

## Lazy Model Loading

Defer model loading until first use:

```python
from effgen.models import LazyModel

model = LazyModel(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    idle_timeout=600,   # Unload after 10 minutes of inactivity
)
# Model is NOT loaded yet
result = model.generate(prompt)  # NOW it loads, then generates
```

## Continuous Batching

Coalesce concurrent requests for higher GPU throughput:

```python
from effgen.models import ContinuousBatcher

batcher = ContinuousBatcher(
    model=model,
    max_batch_size=8,     # Flush after 8 requests
    max_wait_ms=50,       # Or after 50ms, whichever comes first
)

with batcher:
    # Multiple concurrent submit() calls are batched automatically
    future = batcher.submit(prompt, generation_config)
    result = future.result()
```

## Multi-GPU Distribution

effGen uses `CUDA_VISIBLE_DEVICES` for GPU targeting:

```bash
# Run on specific GPUs
CUDA_VISIBLE_DEVICES=0,1 effgen serve --port 8000

# Model loader respects device_map
model = load_model("Qwen/Qwen2.5-7B-Instruct", device_map="auto")
```

For vLLM, tensor parallelism is auto-detected based on model size.

## Batch Execution

Process large query sets efficiently:

```python
results = agent.run_batch(
    queries=["query1", "query2", ...],
    concurrency=5,
    timeout=60,
    retries=2,
)
```

Or via CLI:

```bash
effgen batch --input queries.jsonl --output results.jsonl \
  --concurrency 10 --batch-size 100 --timeout 60 --retries 2
```

### Structured output per row

Pass a JSON Schema file with `--schema`, or a Pydantic model with
`--output-model module:ClassName`, to validate every row. Each output row then
carries a `parsed` object; a row that cannot be coerced to the schema is written
as a failed row with a reason rather than an off-schema string.

```bash
effgen batch --input tickets.jsonl --output results.jsonl \
  --preset minimal --schema ticket_schema.json --max-tokens 4096
```

For pure extraction or generation, prefer `--preset minimal`: it runs the model
directly without the default tool loop, which small models can otherwise spend
iterations on.

### Output columns, per-job cost, and reruns

Each output row carries token counts (`prompt_tokens`/`completion_tokens`/
`total_tokens`), `cost_usd` for priced models, the validated `parsed` object
(when a schema is set), and an `error` reason for failed rows. A local or
unpriced model reports tokens without a `cost_usd`. The final line reports a
per-job token total and, for priced models, a cost total:

```
Batch complete: 1000/1000 succeeded in 42.11s · 1,204,882 tokens · $0.48
```

A `.jsonl` `--output` is written row-by-row as each finishes, so an interrupted
job keeps the rows already done. Rerun the same command with `--resume` to skip
the indices already present and run only the remaining rows:

```bash
effgen batch --input queries.jsonl --output results.jsonl --resume
```

A malformed input line is skipped with a message naming the file and line
number; pass `--strict` to hard-fail on the first bad line instead. A line that
parses but is not a question — a JSON scalar or array, or a row whose fields
carry no query text — is reported the same way and is not sent to the model, so
an input file with nothing usable fails instead of billing empty prompts. Use
`--temperature 0` for deterministic reruns where the provider supports it.

## Domain Keyword Expansion

Scale keyword coverage per domain:

A single `expand()` call combines the enabled strategies; toggle them with the
constructor flags (`use_templates`, `use_wordnet`, `use_llm`).

```python
from effgen.domains import KeywordExpander

seeds = ["machine learning", "data science"]

# Template-based (on by default): 2 seeds → ~20 search-query variants
expander = KeywordExpander()
expanded = expander.expand(seeds, factor=10)

# WordNet synonyms (opt-in; requires nltk): 2 seeds → ~300 terms
expander = KeywordExpander(use_templates=False, use_wordnet=True)
expanded = expander.expand(seeds)

# LLM-based (uses a loaded model): 2 seeds → ~40+ related terms
expander = KeywordExpander(use_templates=False, use_llm=True, model=model)
expanded = expander.expand(seeds, factor=20)
```

A `Domain` exposes the same expansion via `expand_keywords(...)`. For non-tech
domains (legal, finance, health, science), the LLM strategy gives the highest-
quality results:

```python
from effgen.domains import LegalDomain

terms = LegalDomain().expand_keywords(use_llm=True, model=model)
```

### From a domain to a runnable agent

A domain is more than keywords: it bundles a system prompt, recommended tools,
and guardrails. `Domain.to_agent(model)` (or `create_agent(domain=...)`) wires
all three into an agent you can run — the one obvious on-ramp from a domain to
something that answers questions:

```python
from effgen.domains import LegalDomain

# Wires the domain's prompt + tools + guardrails into an agent.
agent = LegalDomain().to_agent("gpt-5-nano")          # or a local model id
response = agent.run("Summarize the obligations in a standard NDA.")
print(response)

# Equivalent, and any create_agent option works (extra_tools, temperature, ...):
from effgen.presets import create_agent
agent = create_agent(domain=LegalDomain(), model="gpt-5-nano", temperature=0.2)
```

## Caching

### Prompt Cache

Avoid re-computing prompts for identical or similar queries:

```python
from effgen.cache import PromptCache

cache = PromptCache(max_size=10000, ttl=3600)
cache.put(prompt, result)
cached = cache.get(prompt)  # O(1) lookup via sha256 fingerprint
print(cache.stats)  # {"hits": 42, "misses": 10, "hit_rate": 0.81}
```

### Result Cache

Cache tool results with per-tool TTL:

```python
from effgen.cache import ResultCache

cache = ResultCache(max_size=5000)
cache.set_tool_ttl("web_search", 300)   # 5 minutes for web search
cache.set_tool_ttl("calculator", 86400)  # 24 hours for math (deterministic)
```

## Token Budget Management

Optimize context window usage:

```python
from effgen.memory.token_budget import TokenBudget

budget = TokenBudget(
    total_tokens=4096,
    system_share=0.20,   # 20% for system prompt
    tools_share=0.30,    # 30% for tool descriptions
    history_share=0.40,  # 40% for conversation history
    response_share=0.10, # 10% reserved for response
)

truncated = budget.fit_to_budget(system=system_prompt, tools=tools_text, history=history)
```

## Monitoring at Scale

- **Prometheus metrics** — `GET /metrics` exposes latency histograms (p50/p95/p99), throughput, error rates, GPU memory
- **Grafana dashboard** — import `configs/grafana/effgen-dashboard.json`
- **OpenTelemetry** — trace propagation across agents, exporters for OTLP/Jaeger/Zipkin
- **Structured logging** — JSON format with run_id correlation for distributed tracing
