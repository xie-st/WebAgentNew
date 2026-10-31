"""Multi-tool agent example — agent with Calculator, BashTool, and DateTimeTool.

Demonstrates an agent that can use multiple tools to solve a task.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/multi_tool_agent.py
"""

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import BashTool, Calculator, DateTimeTool

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("Loading model:", MODEL_ID)
model = load_model(MODEL_ID, quantization="4bit")

config = AgentConfig(
    name="multi_tool_agent",
    model=model,
    tools=[Calculator(), BashTool(), DateTimeTool()],
    system_prompt=(
        "You are a helpful assistant with access to a calculator, "
        "bash shell, and datetime tool. Use the appropriate tool for each task."
    ),
    max_iterations=8,
)

agent = Agent(config=config)

print("\n" + "=" * 60)
print("Task: 'What is today's date, and what is 365 * 24?'")
print("=" * 60)

result = agent.run("What is today's date, and what is 365 * 24?")
print(f"\nAnswer: {result.output}")
print(f"Success: {result.success}")
steps = len(result.execution_history) if hasattr(result, "execution_history") else "N/A"
print(f"Steps: {steps}")

print("\n" + "=" * 60)
print("Multi-tool example complete!")
print("=" * 60)
