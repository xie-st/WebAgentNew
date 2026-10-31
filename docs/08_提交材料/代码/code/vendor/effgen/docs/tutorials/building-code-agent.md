# Building a Code Agent

Create an agent that writes, executes, and debugs code.

## Quick Way

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("coding", model)

result = agent.run("Write a Python function that checks if a string is a palindrome, then test it")
print(result.output)
```

## Custom Configuration

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import CodeExecutor, PythonREPL, FileOperations, BashTool

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(AgentConfig(
    name="code-agent",
    model=model,
    tools=[CodeExecutor(), PythonREPL(), FileOperations(), BashTool()],
    system_prompt=(
        "You are an expert programmer. Write clean, tested code. "
        "Always run your code to verify it works before presenting the result."
    ),
    max_iterations=12,
    temperature=0.4,
))
```

## Available Code Tools

| Tool | Use Case |
|------|----------|
| `PythonREPL` | Quick Python snippets and calculations |
| `CodeExecutor` | Multi-language code execution with sandboxing |
| `FileOperations` | Read/write files on disk |
| `BashTool` | Shell commands (git, pip, system tools) |

## Tips

- The coding preset includes all four tools above
- Use `temperature=0.4` — balanced between creativity and correctness
- `max_iterations=12` gives the agent room to write, test, and fix code
- The agent will iteratively debug if the first attempt fails
