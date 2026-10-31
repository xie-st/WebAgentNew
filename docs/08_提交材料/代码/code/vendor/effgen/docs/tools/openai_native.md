# OpenAI Native Tools

effGen exposes OpenAI's first-party server-side tools — **web_search**, **code_interpreter**, and **file_search** — as `BaseTool` subclasses.  Unlike effGen's local tools, these run inside OpenAI's infrastructure and are never executed on your machine.

---

## Overview

| Tool | Class | OpenAI type | What it does |
|------|-------|-------------|--------------|
| Web Search | `OpenAIWebSearchTool` | `web_search_preview` | Live web queries with citations |
| Code Interpreter | `OpenAICodeInterpreterTool` | `code_interpreter` | Sandboxed Python runtime on OpenAI servers |
| File Search | `OpenAIFileSearchTool` | `file_search` | Vector search over uploaded documents (RAG) |

**Key constraints:**
- All three require an **OpenAI model**.  Using them with Cerebras, Gemini, or any other provider raises `ToolIncompatibleError` at `Agent` initialization — before any API call is made.
- They use the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) (`client.responses.create`), not Chat Completions.

---

## Cost Warning — web_search

> **`OpenAIWebSearchTool` adds a per-call surcharge on top of normal token costs.**
>
> As of 2026-04-24, `web_search_preview` costs approximately **$30 per 1,000 calls** for GPT-4o-class models (lower tiers may differ).  Budget accordingly and monitor usage on the [OpenAI usage dashboard](https://platform.openai.com/usage).

`code_interpreter` and `file_search` are billed at token rates only (no search surcharge), but running code may generate additional compute charges on the OpenAI side.

---

## Quick Start

```python
from dotenv import load_dotenv
load_dotenv()  # OPENAI_API_KEY must be set

from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

adapter = OpenAIAdapter(model_name="gpt-5.4-nano")
adapter.load()

agent = Agent(AgentConfig(
    name="search-agent",
    model=adapter,
    tools=[OpenAIWebSearchTool()],
    tool_calling_mode="native",
))

result = agent.run("What did Anthropic announce this week?")
print(result.output)
agent.close()
adapter.unload()
```

---

## Tools Reference

### `OpenAIWebSearchTool`

```python
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool

tool = OpenAIWebSearchTool(
    search_context_size="medium",   # "low" | "medium" | "high"
    user_location=None,             # optional geographic context
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_context_size` | `str` | `"medium"` | Amount of context per result. Higher = more tokens used |
| `user_location` | `dict \| None` | `None` | Geo context, e.g. `{"type": "approximate", "country": "US"}` |

**OpenAI spec:** `{"type": "web_search_preview", "search_context_size": "medium"}`

---

### `OpenAICodeInterpreterTool`

```python
from effgen.tools.builtin.openai_native import OpenAICodeInterpreterTool

tool = OpenAICodeInterpreterTool(
    container={"type": "auto"},     # OpenAI chooses the runtime
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `container` | `dict` | `{"type": "auto"}` | Container configuration for the sandbox |

The model writes and executes Python code inside OpenAI's secure runtime.  Results (stdout, stderr, files) come back as part of the response.

**OpenAI spec:** `{"type": "code_interpreter", "container": {"type": "auto"}}`

---

### `OpenAIFileSearchTool`

```python
from effgen.tools.builtin.openai_native import OpenAIFileSearchTool

tool = OpenAIFileSearchTool(
    vector_store_ids=["vs_abc123"],  # created via Files API
    max_num_results=10,              # 1–50
    ranking_options=None,
    filters=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_store_ids` | `list[str]` | `[]` | OpenAI vector store IDs to search |
| `max_num_results` | `int` | `10` | Max chunks retrieved (1–50) |
| `ranking_options` | `dict \| None` | `None` | Ranker + score threshold override |
| `filters` | `dict \| None` | `None` | Metadata filter for scoped retrieval |

**Mutating the store list at runtime:**
```python
tool.add_vector_store("vs_new")
tool.remove_vector_store("vs_old")
```

**How to create a vector store:**
```python
from openai import OpenAI
client = OpenAI()

# 1. Upload the file
with open("my_doc.txt", "rb") as f:
    file_obj = client.files.create(file=f, purpose="assistants")

# 2. Create a vector store
vs = client.vector_stores.create(name="my-docs")

# 3. Add the file
client.vector_stores.files.create(vector_store_id=vs.id, file_id=file_obj.id)

# 4. Pass the ID to the tool
tool = OpenAIFileSearchTool(vector_store_ids=[vs.id])
```

---

## Incompatibility Error

Mixing an OpenAI native tool with a non-OpenAI model raises `ToolIncompatibleError` **at Agent init** — you get a clear message immediately rather than a cryptic API error at run time.

```python
from effgen.models.errors import ToolIncompatibleError
from effgen.models.cerebras_adapter import CerebrasAdapter
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
from effgen.core.agent import Agent, AgentConfig

adapter = CerebrasAdapter(model_name="gpt-oss-120b")

try:
    agent = Agent(AgentConfig(
        name="bad-agent",
        model=adapter,
        tools=[OpenAIWebSearchTool()],
    ))
except ToolIncompatibleError as e:
    print(e)
    # Tool 'openai_web_search' is incompatible with model 'gpt-oss-120b'.
    # OpenAI native tools ... require an OpenAIAdapter.
```

---

## Combining Native Tools with effGen Local Tools

You can mix OpenAI native tools and effGen local tools in the same agent.  The adapter routes native tools through the Responses API while local tools are dispatched locally as usual.

```python
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
from effgen.tools.builtin.calculator import Calculator
from effgen.core.agent import Agent, AgentConfig
from effgen.models.openai_adapter import OpenAIAdapter

adapter = OpenAIAdapter(model_name="gpt-5.4-nano")
adapter.load()

agent = Agent(AgentConfig(
    name="hybrid-agent",
    model=adapter,
    tools=[
        OpenAIWebSearchTool(),   # server-side, via Responses API
        Calculator(),             # runs locally
    ],
    tool_calling_mode="auto",
))

result = agent.run("What is the current BTC price multiplied by 1.05?")
print(result.output)
agent.close()
adapter.unload()
```

---

## When to Use Native Tools vs effGen Built-ins

| Scenario | Recommendation |
|----------|---------------|
| Need the latest live web data with citations | `OpenAIWebSearchTool` |
| Controlled web scraping / custom sources | `effgen.tools.builtin.web_search.WebSearch` |
| Sandboxed code execution on OpenAI's infra | `OpenAICodeInterpreterTool` |
| Local code execution (trusted environment) | `effgen.tools.builtin.code_executor.CodeExecutor` |
| RAG over files in OpenAI's vector store | `OpenAIFileSearchTool` |
| RAG over local docs with custom embeddings | `effgen.rag` + `Retrieval` tool |

Native tools offload computation to OpenAI — simpler setup but they incur additional API costs and require an OpenAI model.  Local tools run on your machine — more control, zero extra cost per call, and work with any model.
