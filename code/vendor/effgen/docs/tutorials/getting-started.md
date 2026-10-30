# Getting Started with effGen

Get up and running with effGen in 5 minutes.

## Installation

```bash
pip install effgen
```

For GPU support with vLLM:
```bash
pip install effgen[vllm]
```

## Hello, agent (copy-paste)

The shortest path is a preset plus a model id. Pick **one** model line:

```python
from effgen.presets import create_agent

# (a) cheap cloud model — needs OPENAI_API_KEY in your environment
agent = create_agent("math", "gpt-5-nano")

# (b) or a local small model — downloads once, runs on CPU/GPU, no key needed
# agent = create_agent("math", "Qwen/Qwen2.5-1.5B-Instruct")

result = agent.run("What is 17% of 250?")
print(result)          # 42.5   (printing the result shows the answer)
print(result.text)     # same as result.output / result.content
```

`print(result)` shows the answer; `result.output` (aliased as `result.text` /
`result.content`) is the string, and `result.to_dict()` has the full detail
(tokens, cost, trace).

## Your First Agent (explicit)

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator

# Load a small language model (runs on a single GPU)
model = load_model("Qwen/Qwen2.5-1.5B-Instruct")

# Create an agent with a calculator tool
agent = Agent(AgentConfig(
    name="my-agent",
    model=model,
    tools=[Calculator()],
))

# Run a task
result = agent.run("What is 42 * 58?")
print(result.output)  # "2436"
```

## Using Presets

Skip configuration boilerplate with presets:

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# Math agent (Calculator + PythonREPL)
math_agent = create_agent("math", model)
result = math_agent.run("What is the square root of 144?")

# Coding agent (CodeExecutor + PythonREPL + FileOps + Bash)
code_agent = create_agent("coding", model)
result = code_agent.run("Write a Python script that lists prime numbers under 100")
```

Available presets: `math`, `minimal`, `coding`, `research`, `rag`, `media`,
`multimodal`, `notify`, `general`. New to effGen? Start with **`math`** or
**`minimal`** (small and fast); **`general`** is the "kitchen sink" preset that
loads every tool — powerful but heavier for a small model to reason over. Run
`list_presets()` (or `effgen presets`) for the full descriptions.

## CLI Usage

```bash
# Start a project: config, .env template, runnable example, spend cap
effgen quickstart --init my-agent

# Run a task directly
effgen run "What is 2+2?" --model Qwen/Qwen2.5-3B-Instruct

# Use a preset
effgen run --preset math "What is the square root of 144?"

# Interactive chat
effgen chat --model Qwen/Qwen2.5-3B-Instruct

# List presets
effgen presets

# Verbose output with execution trace
effgen run "Calculate 10!" --preset math --verbose
```

## Next Steps

- [Starting a new project](../cli/new-project.md) — `effgen quickstart --init`
- [Building a Math Agent](building-math-agent.md)
- [Building a Research Agent](building-research-agent.md)
- [Custom Tools Guide](custom-tools.md)
- [API Conventions](../api/conventions.md) — naming, results, streaming, errors
- [API Reference](../api/reference.md)
