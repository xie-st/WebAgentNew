# effGen Examples

Ready-to-run examples demonstrating effGen's agentic AI capabilities with Small Language Models.

## Requirements

- effGen installed (`pip install -e .`)
- GPU recommended (all examples use local model inference)
- Default model: `Qwen/Qwen2.5-3B-Instruct` (downloaded automatically on first run)

## Quick Start

```bash
# Set GPU (default: GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Basic examples
python examples/basic/basic_agent.py            # Basic agent with calculator
python examples/basic/qa_agent.py               # Q&A agent (no tools)
python examples/basic/calculator_agent.py       # Math agent with Calculator + PythonREPL

# Tool-use examples
python examples/tools/multi_tool_agent.py       # Multi-tool agent (simple)
python examples/tools/advanced_multi_tool_agent.py  # Multi-tool with 5 tools
python examples/tools/file_operations_agent.py  # File read/write/search
python examples/tools/coding_agent.py           # Code execution + iteration

# Advanced examples
python examples/advanced/conversational_agent.py   # Multi-turn memory
python examples/advanced/advanced_streaming_agent.py   # Real-time token streaming
python examples/advanced/data_processing_agent.py  # JSON & data pipelines
python examples/advanced/multi_agent_pipeline.py   # Multi-agent orchestration
python examples/advanced/error_recovery_agent.py   # Error handling patterns

# Preset and plugin examples
python examples/plugins_presets/preset_agents.py          # Ready-to-use agent presets
python examples/plugins_presets/plugin_example.py         # Custom tool plugins

# Web and retrieval examples
python examples/web_retrieval/web_agent.py              # Web search agent
python examples/web_retrieval/weather_agent.py          # Weather via Open-Meteo (free)
python examples/web_retrieval/streaming_agent.py        # Simple streaming demo
python examples/web_retrieval/memory_agent.py           # Simple memory demo
python examples/web_retrieval/retrieval_agent.py        # RAG-based retrieval
python examples/web_retrieval/agentic_search_agent.py   # Grep-based agentic search

# Cerebras (cloud, free-tier API — set CEREBRAS_API_KEY in ~/.effgen/.env)
python examples/cerebras/basic_cerebras.py              # Hello-world generation
python examples/cerebras/cerebras_load_model.py         # load_model(provider="cerebras")
python examples/cerebras/cerebras_streaming.py          # Token-by-token streaming
python examples/cerebras/cerebras_multi_turn.py         # Multi-turn chat
python examples/cerebras/cerebras_all_models.py         # Smoke all 4 models in parallel
python examples/cerebras/cerebras_rate_limits.py        # Rate-limit coordinator demo
python examples/cerebras/cerebras_cost_tracker.py       # CostTracker integration
python examples/cerebras/cerebras_tool_calling.py       # Native function-calling
python examples/cerebras/cerebras_agent.py              # Full Agent + tools
python examples/cerebras/cerebras_hard_agent.py         # Hard agentic tasks (math + code + dates)

# OpenAI (cloud — set OPENAI_API_KEY in ~/.effgen/.env)
python examples/openai/basic_chat.py                    # Hello-world chat
python examples/openai/multi_turn_chat.py               # Multi-turn chat
python examples/openai/tool_calling.py                  # Native function-calling
python examples/openai/structured_outputs.py            # JSON-schema strict outputs
python examples/openai/prompt_caching.py                # Cached_input_tokens demo
python examples/openai/reasoning_models.py              # reasoning_effort on o-series
python examples/openai/openai_agent.py                  # Full Agent + effGen tools
python examples/openai/caching_and_structured_agent.py  # Caching + structured outputs combined
python examples/openai/native_tools_web_search.py       # OpenAIWebSearchTool (Responses API)
python examples/openai/native_tools_code_interpreter.py # OpenAICodeInterpreterTool
python examples/openai/native_tools_file_search.py      # OpenAIFileSearchTool
python examples/openai/native_tools_hybrid_agent.py     # OpenAI native tools + effGen tools together
```

## Example Agents

### Core Examples

| Example | Description | Tools | Recommended Model |
|---------|-------------|-------|-------------------|
| `basic/qa_agent.py` | Q&A agent with no tools — direct inference | None | Qwen2.5-1.5B+ |
| `basic/calculator_agent.py` | Math agent with step-by-step calculation | Calculator, PythonREPL | Qwen2.5-1.5B+ |
| `tools/advanced_multi_tool_agent.py` | Agent with 5 tools, fallback chains, circuit breaker | Calculator, PythonREPL, DateTimeTool, BashTool, TextProcessingTool | Qwen2.5-3B+ |
| `tools/file_operations_agent.py` | File read/write/list/search with sandbox | FileOperations, BashTool, TextProcessingTool | Qwen2.5-3B+ |
| `tools/coding_agent.py` | Write, run, and iterate on code | CodeExecutor, PythonREPL, FileOperations, BashTool | Qwen2.5-3B+ |
| `advanced/conversational_agent.py` | Multi-turn conversation with memory persistence | Calculator, DateTimeTool + ShortTermMemory + LongTermMemory | Qwen2.5-3B+ |
| `advanced/error_recovery_agent.py` | Intentional failures to test framework resilience | Custom broken/slow tools | Qwen2.5-3B+ |
| `advanced/data_processing_agent.py` | JSON querying, validation, text analysis, data pipelines | JSONTool, TextProcessingTool, PythonREPL, FileOperations | Qwen2.5-1.5B+ |
| `advanced/advanced_streaming_agent.py` | Token streaming with thought/tool/answer callbacks | Calculator, DateTimeTool | Qwen2.5-1.5B+ |
| `advanced/multi_agent_pipeline.py` | Multi-agent orchestration: manual pipeline + sub-agents | Calculator, PythonREPL + SubAgentRouter | Qwen2.5-3B+ |

### Simple Examples (from v0.1.1)

| Example | Description |
|---------|-------------|
| `basic/basic_agent.py` | Basic calculator agent |
| `basic/basic_agent_vllm.py` | Basic agent with vLLM backend (5-10x faster) |
| `plugins_presets/preset_agents.py` | Create agents from built-in presets |
| `web_retrieval/streaming_agent.py` | Simple streaming demo |
| `web_retrieval/memory_agent.py` | Simple multi-turn memory demo |
| `tools/multi_tool_agent.py` | Simple multi-tool agent |
| `web_retrieval/weather_agent.py` | Weather queries (free, no API key) |
| `plugins_presets/plugin_example.py` | Custom tool plugin creation |
| `web_retrieval/web_agent.py` | Web search agent |
| `web_retrieval/retrieval_agent.py` | RAG-based retrieval |
| `web_retrieval/agentic_search_agent.py` | Grep-based search |

### Cerebras (cloud, free-tier API)

Requires `CEREBRAS_API_KEY` (free tier; place in `~/.effgen/.env`). All 4 free-tier
models supported: `gpt-oss-120b`, `zai-glm-4.7`. Each example prints raw API output.

| Example | Demonstrates |
|---------|--------------|
| `cerebras/basic_cerebras.py`        | Adapter + `generate()` + token counting |
| `cerebras/cerebras_load_model.py`   | `load_model(..., provider="cerebras")` smoke |
| `cerebras/cerebras_streaming.py`    | `generate_stream()` token-by-token |
| `cerebras/cerebras_multi_turn.py`   | Multi-turn chat with message history |
| `cerebras/cerebras_all_models.py`   | Parallel smoke across all 4 models |
| `cerebras/cerebras_rate_limits.py`  | RateLimitCoordinator throttling |
| `cerebras/cerebras_cost_tracker.py` | CostTracker per-call accounting (free-tier = $0) |
| `cerebras/cerebras_tool_calling.py` | Native function-calling (`generate_with_tools`) |
| `cerebras/cerebras_agent.py`        | Full Agent + Calculator + DateTimeTool |
| `cerebras/cerebras_hard_agent.py`   | Hard tasks: compound interest, prime factorization, Fibonacci |

### OpenAI (cloud)

Requires `OPENAI_API_KEY`. Examples default to `gpt-5.4-nano` / `gpt-4.1-nano` (cheapest tier). Reasoning examples use `o4-mini` with `reasoning_effort`.

| Example | Demonstrates |
|---------|--------------|
| `openai/basic_chat.py`                       | One-shot generation |
| `openai/multi_turn_chat.py`                  | Multi-turn chat with system prompt |
| `openai/tool_calling.py`                     | Native function-calling on chat models |
| `openai/structured_outputs.py`               | Strict JSON Schema → Pydantic round-trip |
| `openai/prompt_caching.py`                   | `cached_input_tokens` surfaced on repeated calls |
| `openai/reasoning_models.py`                 | `reasoning_effort` on `o4-mini` (low vs high) |
| `openai/openai_agent.py`                     | Full Agent + effGen Calculator/DateTime tools |
| `openai/caching_and_structured_agent.py`     | Caching + strict outputs in one agent |
| `openai/native_tools_web_search.py`          | `OpenAIWebSearchTool` (Responses API) |
| `openai/native_tools_code_interpreter.py`    | `OpenAICodeInterpreterTool` (server-side Python) |
| `openai/native_tools_file_search.py`         | `OpenAIFileSearchTool` (RAG over uploaded files) |
| `openai/native_tools_hybrid_agent.py`        | OpenAI native tools + effGen local tools in one agent |

### Utilities

| File | Description |
|------|-------------|
| `utils/sweep_model.py` | Cross-model compatibility sweep runner |
| `utils/compatibility_matrix.md` | Full model compatibility results (11 models x 10 agents) |

## Model Recommendations

Based on testing across 11 models (see [utils/compatibility_matrix.md](utils/compatibility_matrix.md)):

| Model | Size | Score | Best For |
|-------|------|-------|----------|
| **Qwen2.5-1.5B-Instruct** | 1.5B | 10/10 | Best value — perfect score at small size |
| **Qwen2.5-3B-Instruct** | 3B | 10/10 | Recommended default — fast, reliable |
| **Phi-4-mini-instruct** | 3.8B | 10/10 | Best cross-family validation |
| Qwen3-1.7B | 1.7B | 9.5/10 | Strong alternative at 1.7B |
| Qwen2.5-7B-Instruct | 7B | 9.0/10 | When 7B quality is needed |
| Llama-3.2-3B-Instruct | 3B | 8.5/10 | Good for single-turn tasks |

## Configuration

All examples support:
- `CUDA_VISIBLE_DEVICES` environment variable for GPU selection
- `--model` argument to specify the model
- Most support `--interactive` for chat mode
