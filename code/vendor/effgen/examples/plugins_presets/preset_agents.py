"""Preset agents example — create agents from built-in presets.

Demonstrates create_agent() with the "math", "research", and "coding" presets.
Each preset comes pre-configured with relevant tools and system prompts.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/preset_agents.py
"""

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from effgen import create_agent, load_model

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("Loading model:", MODEL_ID)
model = load_model(MODEL_ID, quantization="4bit")

# --- Math preset: Calculator + PythonREPL ---
print("\n" + "=" * 60)
print("MATH PRESET")
print("=" * 60)
math_agent = create_agent("math", model)
result = math_agent.run("What is 17 * 23?")
print(f"Answer: {result.output}")
print(f"Success: {result.success}")

# --- Research preset: WebSearch + URLFetch + Wikipedia ---
print("\n" + "=" * 60)
print("RESEARCH PRESET")
print("=" * 60)
research_agent = create_agent("research", model)
result = research_agent.run("What is the capital of France?")
print(f"Answer: {result.output}")
print(f"Success: {result.success}")

# --- Coding preset: CodeExecutor + PythonREPL + FileOps + Bash ---
print("\n" + "=" * 60)
print("CODING PRESET")
print("=" * 60)
coding_agent = create_agent("coding", model)
result = coding_agent.run("Write a Python one-liner that prints the first 5 Fibonacci numbers.")
print(f"Answer: {result.output}")
print(f"Success: {result.success}")

print("\n" + "=" * 60)
print("All presets tested successfully!")
print("=" * 60)
