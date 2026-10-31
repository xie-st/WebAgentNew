# Gemini Native Tools

effGen exposes Gemini's three built-in server-side tools as
`BaseTool` subclasses:

| Class | Tool name | What it does |
|---|---|---|
| `GoogleSearchTool` | `google_search` | Live web search with citations |
| `GeminiUrlContextTool` | `url_context` | Fetches URL content server-side |
| `GeminiCodeExecutionTool` | `code_execution` | Runs Python in a secure Google sandbox |

These tools run **entirely inside Google's infrastructure** — effGen passes the
tool spec to the API, and the model decides when (and whether) to use them.

---

## Prerequisites

- A `GeminiAdapter` model.  Passing a Gemini native tool to a non-Gemini model
  raises `ToolIncompatibleError` at `Agent` init time (before any API calls).
- A valid `GOOGLE_API_KEY`.
- Free-tier note: `GoogleSearchTool` requires a model whose free tier supports
  grounding (e.g. `gemini-2.5-flash-lite`).  The `gemini-3.1-flash-lite`
  free tier does not include Google Search grounding.

---

## Quick Start

```python
from dotenv import load_dotenv
load_dotenv()

from effgen.models.gemini_adapter import GeminiAdapter
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.gemini_native import (
    GoogleSearchTool,
    GeminiUrlContextTool,
    GeminiCodeExecutionTool,
)

model = GeminiAdapter(model_name="gemini-2.5-flash-lite")
model.load()

# --- Google Search ---
agent = Agent(AgentConfig(
    name="search-agent",
    model=model,
    tools=[GoogleSearchTool()],
))
response = agent.run("What AI announcements did Google make this week?")
print(response.output)

# --- URL Context ---
agent = Agent(AgentConfig(
    name="url-agent",
    model=model,
    tools=[GeminiUrlContextTool()],
))
response = agent.run(
    "Summarise this page: https://en.wikipedia.org/wiki/Python_(programming_language)"
)
print(response.output)

# --- Code Execution ---
model2 = GeminiAdapter(model_name="gemini-3.1-flash-lite")
model2.load()
agent = Agent(AgentConfig(
    name="code-agent",
    model=model2,
    tools=[GeminiCodeExecutionTool()],
))
response = agent.run("Compute 100! exactly using Python.")
print(response.output)
```

---

## GoogleSearchTool

Activates Gemini's built-in Google Search grounding.  The model searches the
live web when it determines a search would help answer the question, and the
response includes citations.

```python
from effgen.tools.builtin.gemini_native import GoogleSearchTool

tool = GoogleSearchTool()
```

**Compatible models:** Any Gemini model that supports the `google_search` tool
and whose free-tier quota allows grounding (e.g. `gemini-2.5-flash-lite`).

---

## GeminiUrlContextTool

Fetches the content of one or more URLs server-side.  The retrieved page text
is given to the model as additional context.

```python
from effgen.tools.builtin.gemini_native import GeminiUrlContextTool

tool = GeminiUrlContextTool()
```

**Compatible models:** `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite`,
and other models that support `url_context`.

---

## GeminiCodeExecutionTool

Runs Python code in a sandboxed Google-hosted environment.  The model decides
when to write and execute code, and the output (stdout, errors) appears in its
response.

```python
from effgen.tools.builtin.gemini_native import GeminiCodeExecutionTool

tool = GeminiCodeExecutionTool()
```

**Compatible models:** `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite`,
and most Gemini models.

---

## Mixing Native and Local Tools

You can combine Gemini native tools with regular effGen local tools.
The adapter routes native tools server-side and local tools through
the standard function-calling loop.

```python
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.gemini_native import GeminiCodeExecutionTool

agent = Agent(AgentConfig(
    name="mixed-agent",
    model=gemini_model,
    tools=[GeminiCodeExecutionTool(), Calculator()],
))
```

---

## ToolIncompatibleError

Using a Gemini native tool with a non-Gemini model raises
`ToolIncompatibleError` at Agent init time:

```python
from effgen.models.errors import ToolIncompatibleError
from effgen.models.openai_adapter import OpenAIAdapter
from effgen.tools.builtin.gemini_native import GoogleSearchTool

openai_model = OpenAIAdapter(model_name="gpt-4o")
try:
    Agent(AgentConfig(name="bad", model=openai_model, tools=[GoogleSearchTool()]))
except ToolIncompatibleError as e:
    print(e)  # Tool 'google_search' is incompatible with model 'gpt-4o'.
```

---

## Parallel Function Calls

Gemini can return multiple `functionCall` parts in a single response turn.
effGen's adapter encodes **all** of them as `<tool_call>` blocks and surfaces
them in `GenerationResult.metadata["tool_calls"]`, one element per call, in
the shape every adapter reports (see [Tool calls](../models/tool-calls.md)).
Gemini additionally carries top-level `name` and `arguments` keys on each
element for callers written against its earlier flat shape.  The Agent loop
processes all parallel calls in one iteration.

```python
# The model may call Calculator twice in one turn
response = agent.run("What is 2+3? Also what is 7×8?")
# response.tool_calls == 2
```
