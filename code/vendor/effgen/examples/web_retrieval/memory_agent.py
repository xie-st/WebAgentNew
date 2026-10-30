"""Memory agent example — multi-turn conversation with context recall.

Demonstrates enable_memory=True in AgentConfig so the agent remembers
previous interactions within the same session.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/memory_agent.py
"""

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from effgen import Agent, AgentConfig, load_model

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("Loading model:", MODEL_ID)
model = load_model(MODEL_ID, quantization="4bit")

config = AgentConfig(
    name="memory_agent",
    model=model,
    tools=[],
    system_prompt="You are a helpful assistant with memory. Remember what the user tells you.",
    enable_memory=True,
    max_iterations=3,
)

agent = Agent(config=config)

# Turn 1: Provide information
print("\n" + "=" * 60)
print("Turn 1: Providing information")
print("=" * 60)
result1 = agent.run("My name is Alice and I'm studying quantum computing.")
print(f"Agent: {result1.output}")

# Turn 2: Ask the agent to recall
print("\n" + "=" * 60)
print("Turn 2: Testing recall")
print("=" * 60)
result2 = agent.run("What's my name and what am I studying?")
print(f"Agent: {result2.output}")

# Check if the agent recalled correctly
output_lower = result2.output.lower()
recalled_name = "alice" in output_lower
recalled_topic = "quantum" in output_lower
print(f"\nRecalled name ('Alice'): {recalled_name}")
print(f"Recalled topic ('quantum'): {recalled_topic}")

print("\n" + "=" * 60)
if recalled_name and recalled_topic:
    print("Memory test PASSED!")
else:
    print("Memory test: agent responded but may not have recalled all details.")
print("=" * 60)
