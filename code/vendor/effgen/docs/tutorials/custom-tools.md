# Custom Tools Guide

Learn how to create your own tools for effGen agents.

## The quick way: the `@tool` decorator (recommended)

For most tools you don't need any boilerplate — just write a typed function and
decorate it. effGen derives the tool's name, description and JSON schema from
the function's name, type hints and docstring:

```python
from effgen import tool, Agent, AgentConfig

@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count words in.
    """
    return len(text.split())

agent = Agent(AgentConfig(
    name="my-agent",
    model="gpt-5-nano",            # or a local id like "Qwen/Qwen2.5-1.5B-Instruct"
    tools=[word_count],
))
result = agent.run("How many words are in 'the quick brown fox'?")
print(result)
```

The decorated `word_count` is a real tool object, so it works anywhere a tool
instance is expected — `AgentConfig(tools=[...])` and provider-native
function-calling. Async functions are supported too. You can override any field:

```python
@tool(name="multiply", category="computation")
def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

Prefer an explicit call? `Tool.from_function` does the same thing without the
decorator:

```python
from effgen import Tool

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

add_tool = Tool.from_function(add)
```

`@tool`/`Tool.from_function` return a tool instance, not a class. To make it show
up in `effgen tools list`, `effgen run -t <name>`, or an MCP server, register that
instance directly:

```python
from effgen.tools import get_registry

get_registry().register_tool(word_count)
```

`register_tool` accepts a `BaseTool` instance exactly this way, alongside the
class-based form used by `BaseTool` subclasses below — a `ToolPlugin.tools` list
can also mix instances and subclasses.

The sections below show the full `BaseTool` interface — reach for it when you
need rich parameter validation, custom initialization, or lifecycle hooks.

## Basic Tool Structure

Every tool extends `BaseTool` and implements two things:
1. An `__init__` that passes a `ToolMetadata` to `super().__init__()`
2. An `_execute` async method that does the work

```python
from effgen.tools.base_tool import (
    BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType,
)

class UpperCaseTool(BaseTool):
    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="uppercase",
                description="Convert text to uppercase",
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="text",
                        type=ParameterType.STRING,
                        description="Text to convert",
                        required=True,
                    ),
                ],
                returns={"type": "object", "properties": {"result": {"type": "string"}}},
            )
        )

    async def _execute(self, **kwargs):
        return {"result": kwargs["text"].upper()}
```

`UpperCaseTool()` takes no arguments, so the metadata must be supplied from
`__init__` via `super().__init__(metadata=...)` — this is how `Agent(tools=[...])`,
`register_tool`, and the plugin loader all construct a tool.

## Using Your Tool

```python
from effgen import Agent, AgentConfig, load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = Agent(AgentConfig(
    name="my-agent",
    model=model,
    tools=[UpperCaseTool()],
))

result = agent.run("Convert 'hello world' to uppercase")
```

## Parameter Validation

`ParameterSpec` supports rich validation:

```python
ParameterSpec(
    name="count",
    type=ParameterType.INTEGER,
    description="Number of items",
    required=True,
    min_value=1,
    max_value=100,
)

ParameterSpec(
    name="format",
    type=ParameterType.STRING,
    description="Output format",
    enum=["json", "csv", "text"],
    default="json",
)
```

## Distributing as a Plugin

See the [Plugin Development Guide](../guides/plugin-development.md) to package your tools as an installable plugin.
