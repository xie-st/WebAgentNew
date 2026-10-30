# API Reference

## Core Classes

### Agent

```python
from effgen import Agent, AgentConfig

config = AgentConfig(
    name="my-agent",
    model=model,            # BaseModel instance or model ID string
    tools=[Calculator()],   # List of BaseTool instances
    system_prompt="...",    # Custom system prompt
    max_iterations=10,      # Max ReAct loop iterations
    temperature=0.7,        # Generation temperature
    enable_sub_agents=True, # Enable task decomposition
    enable_memory=True,     # Enable conversation memory
    enable_streaming=False, # Enable streaming output
)

agent = Agent(config)
```

#### `agent.run(task, mode=AgentMode.AUTO) -> AgentResponse`

Run a task synchronously. Returns `AgentResponse` with:
- `output: str` — Final response text
- `success: bool` — Whether execution succeeded
- `outcome: str` — `"answered"`, `"stopped"` or `"failed"`
- `stop_reason: str` — What ended the run; present on every response
- `partial: PartialResult | None` — What a stopped run had reached
- `iterations: int` — ReAct iterations performed
- `tool_calls: ToolCallList` — The tool calls the run made
- `tool_call_count: int` — How many calls that was
- `tokens_used: int` — Total tokens consumed
- `execution_time: float` — Wall-clock seconds
- `execution_trace: List[Dict]` — Full ReAct trace

`tool_calls` carries a `ToolCall` record per call — `name`, `arguments`,
`result`, `duration`, `error` and the `iteration` it was made on — so you can
see *which* call went wrong, not only that one did:

```python
for call in result.tool_calls:
    print(call.name, call.arguments, "->", call.error or call.result)

failed = result.tool_calls.failed          # only the calls that errored
searches = result.tool_calls.by_name("web_search")
```

It also compares and casts as the count, so `result.tool_calls == 2` and
`result.tool_calls > 0` read as they did before 1.0.0, and
`result.tool_calls.total` states the number outright — spelled `total` rather
than `count` so the inherited `list.count(value)` keeps working. `to_dict()`
keeps the number under `tool_calls` and puts the records in
`tool_call_details`.

The same calls are also reported by the model adapter in
`GenerationResult.metadata["tool_calls"]`, in one shape for every provider —
see [Tool calls](../models/tool-calls.md).

#### `agent.stream(task, mode=AgentMode.AUTO) -> Iterator[str]`

Stream response tokens. Yields strings as they are generated.

#### `agent.reset_memory()`

Clear conversation history.

---

### load_model

```python
from effgen import load_model

model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit",          # None, "4bit", "8bit"
    device_map="auto",            # "auto", "cpu", or GPU index
    max_memory=None,              # Dict of device -> max memory
)
```

Supported backends:
- **Transformers** (default): HuggingFace models
- **vLLM**: High-throughput serving (`pip install effgen[vllm]`)
- **OpenAI**: `load_model("openai:gpt-4o")`
- **Anthropic**: `load_model("anthropic:claude-3-haiku")`
- **Gemini**: `load_model("gemini:gemini-pro")`

---

### Presets

```python
from effgen.presets import create_agent, list_presets

# See available presets
print(list_presets())
# {'math': '...', 'research': '...', 'coding': '...', 'general': '...', 'minimal': '...'}

# Create a preset agent
agent = create_agent("math", model)
agent = create_agent("coding", model, extra_tools=[MyTool()])
```

---

## Tools

### BaseTool

All tools extend `BaseTool` from `effgen.tools.base_tool`:

```python
from effgen.tools.base_tool import BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="my_tool",
                description="What this tool does",
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(name="input", type=ParameterType.STRING,
                                  description="Input text", required=True),
                ],
                returns={"type": "object"},
            )
        )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        return {"result": kwargs["input"]}
```

### Built-in Tools

| Tool | Import | Description |
|------|--------|-------------|
| Calculator | `from effgen.tools.builtin import Calculator` | Arithmetic expressions |
| PythonREPL | `from effgen.tools.builtin import PythonREPL` | Execute Python code |
| CodeExecutor | `from effgen.tools.builtin import CodeExecutor` | Multi-language code execution |
| WebSearch | `from effgen.tools.builtin import WebSearch` | Web search (DuckDuckGo free) |
| FileOperations | `from effgen.tools.builtin import FileOperations` | Read/write/list files |
| BashTool | `from effgen.tools.builtin import BashTool` | Shell commands |
| JSONTool | `from effgen.tools.builtin import JSONTool` | JSON manipulation |
| DateTimeTool | `from effgen.tools.builtin import DateTimeTool` | Date/time operations |
| TextProcessingTool | `from effgen.tools.builtin import TextProcessingTool` | Text manipulation |
| URLFetchTool | `from effgen.tools.builtin import URLFetchTool` | Fetch URL content |
| WikipediaTool | `from effgen.tools.builtin import WikipediaTool` | Wikipedia lookup |
| AgenticSearch | `from effgen.tools.builtin import AgenticSearch` | Local file search |
| Retrieval | `from effgen.tools.builtin import Retrieval` | RAG retrieval |
| StockPriceTool | `from effgen.tools.builtin import StockPriceTool` | Stock prices (yfinance/Yahoo) |
| CurrencyConverterTool | `from effgen.tools.builtin import CurrencyConverterTool` | Currency conversion (ECB) |
| CryptoTool | `from effgen.tools.builtin import CryptoTool` | Crypto prices (CoinGecko) |
| DataFrameTool | `from effgen.tools.builtin import DataFrameTool` | Pandas DataFrame ops |
| PlotTool | `from effgen.tools.builtin import PlotTool` | Matplotlib charts |
| StatsTool | `from effgen.tools.builtin import StatsTool` | Statistical analysis |
| GitTool | `from effgen.tools.builtin import GitTool` | Git operations (read-only) |
| DockerTool | `from effgen.tools.builtin import DockerTool` | Docker operations (read-only) |
| SystemInfoTool | `from effgen.tools.builtin import SystemInfoTool` | System monitoring |
| HTTPTool | `from effgen.tools.builtin import HTTPTool` | HTTP requests |
| ArXivTool | `from effgen.tools.builtin import ArXivTool` | arXiv search, fetch, and PDF download |
| ArxivTool | `from effgen.tools.builtin import ArxivTool` | Legacy alias for `ArXivTool` |
| PubMedTool | `from effgen.tools.builtin import PubMedTool` | PubMed search, metadata fetch, and abstracts |
| SemanticScholarTool | `from effgen.tools.builtin import SemanticScholarTool` | Semantic Scholar paper search, citations, and references |
| StackOverflowTool | `from effgen.tools.builtin import StackOverflowTool` | StackOverflow search |
| GitHubTool | `from effgen.tools.builtin import GitHubTool` | GitHub search |
| WolframAlphaTool | `from effgen.tools.builtin import WolframAlphaTool` | Wolfram Alpha (API key) |
| EmailDraftTool | `from effgen.tools.builtin import EmailDraftTool` | Email drafting |
| SlackDraftTool | `from effgen.tools.builtin import SlackDraftTool` | Slack message drafting |
| NotificationTool | `from effgen.tools.builtin import NotificationTool` | Desktop notifications |

---

## Guardrails

```python
from effgen.guardrails import (
    GuardrailChain, PIIGuardrail, PromptInjectionGuardrail,
    ToxicityGuardrail, LengthGuardrail, TopicGuardrail,
    ToolPermissionGuardrail, get_guardrail_preset,
)

# Use a preset
chain = get_guardrail_preset("strict")

# Or build custom
chain = GuardrailChain([
    PromptInjectionGuardrail(sensitivity="high"),
    PIIGuardrail(),
    LengthGuardrail(max_length=5000),
])

# Attach to agent
config = AgentConfig(name="safe_agent", model=model, guardrails=chain)
```

---

## RAG Pipeline

```python
from effgen.rag import DocumentIngester, HybridSearchEngine, ContextBuilder

# Ingest documents
ingester = DocumentIngester()
chunks = ingester.ingest("./docs/")

# Search
engine = HybridSearchEngine(chunks)
results = engine.search("scaling architecture", top_k=5)

# Build context with citations
builder = ContextBuilder(max_tokens=2048)
context, citations = builder.build(results)

# Or use the preset
from effgen.presets import create_agent
agent = create_agent("rag", model, knowledge_base="./docs/")
result = agent.run("What does the architecture doc say about scaling?")
print(result.citations)
```

---

## Evaluation

```python
from effgen.eval import AgentEvaluator, MathSuite, RegressionTracker
from effgen.eval.evaluator import ScoringMode

evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)
results = evaluator.run_suite(MathSuite())
print(results.summary())

# Regression tracking
tracker = RegressionTracker()
tracker.save_baseline("math", results, version="0.2.0")
report = tracker.compare("math", new_results, version="0.2.1")
```

---

## Model Router

```python
from effgen.models.router import ModelRouter, estimate_complexity
from effgen.models.capabilities import MODEL_CAPABILITIES

# Auto-routing
config = AgentConfig(
    name="smart_agent",
    model=small_model,
    models=[small_model, large_model],  # Router auto-created
)

# Complexity estimation
estimate = estimate_complexity("Write a recursive merge sort in Python")
print(estimate.level)  # → ComplexityLevel.COMPLEX
```

---

## Checkpointing & Sessions

```python
# Checkpointing
agent.run("Long research task...", checkpoint_interval=3)
# If interrupted:
agent.resume(checkpoint_id="latest")

# Sessions
agent = Agent(config=config, session_id="user-123")
agent.run("My name is Alice")
# Later, even in new process:
agent = Agent(config=config, session_id="user-123")
result = agent.run("What's my name?")  # Recalls "Alice"
```

---

## Human-in-the-Loop

```python
from effgen.core.human_loop import HumanApproval

config = AgentConfig(
    name="careful_agent",
    model=model,
    tools=[BashTool()],
    approval_mode="dangerous_only",
    approval_callback=HumanApproval.cli_callback,
)
```

---

## Observability

```python
# Debug mode
result = agent.run("What is 2+2?", debug=True)
trace = result.metadata["debug_trace"]
trace.print_rich()  # Rich TUI output

# Structured logging
from effgen.utils.structured_logging import StructuredLogger
logger = StructuredLogger("my_app")

# Prometheus metrics
from effgen.utils.prometheus_metrics import metrics
print(metrics.export())
```

---

## Client SDK

```python
from effgen.client import EffGenClient

client = EffGenClient(base_url="http://localhost:8000", api_key="...")

# Chat
response = client.chat("What is 2+2?")

# Streaming
for chunk in client.chat_stream_sync("Tell me a story"):
    print(chunk, end="")

# Embeddings
vectors = client.embed(["hello", "world"])
```

---

## Plugin System

```python
from effgen.tools.plugin import ToolPlugin, PluginManager, discover_plugins

# Auto-discover all plugins
discover_plugins()

# Manual plugin loading
mgr = PluginManager()
mgr.load_plugin(my_plugin_instance)
```

See [Plugin Development Guide](../guides/plugin-development.md) for details.

---

## Configuration

```python
from effgen import ConfigLoader, Config

loader = ConfigLoader()
config = loader.load_config("config.yaml")
```

See [Configuration Reference](configuration.md) for all options.

---

## Memory

```python
from effgen import ShortTermMemory, LongTermMemory, VectorMemoryStore

# Short-term (conversation context)
stm = ShortTermMemory(max_tokens=4096)

# Long-term (persistent facts)
ltm = LongTermMemory(backend="sqlite", persist_path="~/.effgen/memory/")

# Vector store (semantic search)
vs = VectorMemoryStore()
```
