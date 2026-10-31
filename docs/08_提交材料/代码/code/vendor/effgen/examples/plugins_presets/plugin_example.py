"""Plugin example — create a custom tool plugin and use it in an agent.

Demonstrates how to:
1. Define a custom tool by subclassing BaseTool
2. Package it as a ToolPlugin
3. Register it with PluginManager
4. Use it in an agent

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/plugin_example.py
"""

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from effgen import Agent, AgentConfig, load_model
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)
from effgen.tools.plugin import PluginManager, ToolPlugin


# Step 1: Define a custom tool
class GreetingTool(BaseTool):
    """A simple tool that generates greetings."""

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="greeting",
                description="Generate a friendly greeting for a given name.",
                category=ToolCategory.COMMUNICATION,
                parameters=[
                    ParameterSpec(
                        name="name",
                        type=ParameterType.STRING,
                        description="The name to greet",
                        required=True,
                    ),
                ],
            )
        )

    async def _execute(self, name: str, **kwargs) -> ToolResult:
        greeting = f"Hello, {name}! Welcome to effGen!"
        return ToolResult(success=True, output=greeting)


# Step 2: Create a plugin containing the tool
class GreetingPlugin(ToolPlugin):
    name = "greeting_plugin"
    version = "1.0.0"
    description = "A demo plugin that adds a greeting tool."
    tools = [GreetingTool]


# Step 3: Register the plugin
manager = PluginManager()
plugin = GreetingPlugin()
registered = plugin.register(manager.registry)
print(f"Registered tools: {registered}")

# Step 4: Use the custom tool in an agent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
print(f"\nLoading model: {MODEL_ID}")
model = load_model(MODEL_ID, quantization="4bit")

config = AgentConfig(
    name="plugin_agent",
    model=model,
    tools=[GreetingTool()],
    system_prompt="You are a friendly assistant. Use the greeting tool to greet people.",
    max_iterations=5,
)

agent = Agent(config=config)

print("\n" + "=" * 60)
print("Query: 'Please greet Alice'")
print("=" * 60)

result = agent.run("Please greet Alice.")
print(f"\nAnswer: {result.output}")
print(f"Success: {result.success}")

print("\n" + "=" * 60)
print("Plugin example complete!")
print("=" * 60)
