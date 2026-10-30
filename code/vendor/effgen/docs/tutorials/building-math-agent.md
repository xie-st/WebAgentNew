# Building a Math Agent

This tutorial walks through creating an agent that solves mathematical problems using the Calculator and PythonREPL tools.

## Quick Way — Using Presets

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("math", model)

result = agent.run("If I invest $1000 at 5% annual interest compounded monthly, how much will I have after 10 years?")
print(result.output)
```

## Custom Configuration

For more control, build the agent manually:

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import Calculator, PythonREPL

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(AgentConfig(
    name="math-agent",
    model=model,
    tools=[Calculator(), PythonREPL()],
    system_prompt=(
        "You are a precise math tutor. Show your work step by step. "
        "Use the calculator for simple arithmetic and python_repl for "
        "complex computations like statistics or plotting."
    ),
    max_iterations=8,
    temperature=0.3,  # Lower temperature for deterministic math
))

result = agent.run("What is the standard deviation of [4, 8, 15, 16, 23, 42]?")
print(f"Answer: {result.output}")
print(f"Tools used: {result.tool_calls}")
print(f"Iterations: {result.iterations}")
```

## How It Works

The agent follows the ReAct (Reasoning + Acting) loop:

1. **Thought**: "I need to calculate the standard deviation. I'll use python_repl."
2. **Action**: Calls `python_repl` with statistics code
3. **Observation**: Gets the computation result
4. **Thought**: "The result is X. I can now answer."
5. **Final Answer**: Presents the formatted result

## Tips

- Use `temperature=0.3` or lower for math — reduces hallucination
- `PythonREPL` handles complex math better than `Calculator` alone
- Set `max_iterations=8` — math rarely needs many iterations
- Add `--explain` flag in CLI to see the agent's tool selection reasoning
