"""Weather agent example — query weather using the free Open-Meteo API.

No API key required! The WeatherTool uses Open-Meteo as its primary backend.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/weather_agent.py
"""

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import WeatherTool

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("Loading model:", MODEL_ID)
model = load_model(MODEL_ID, quantization="4bit")

config = AgentConfig(
    name="weather_agent",
    model=model,
    tools=[WeatherTool()],
    system_prompt=(
        "You are a weather assistant. Use the weather tool to get current "
        "weather information. Report temperature, conditions, and wind speed."
    ),
    max_iterations=5,
)

agent = Agent(config=config)

print("\n" + "=" * 60)
print("Query: 'What is the weather in London?'")
print("=" * 60)

result = agent.run("What is the weather in London?")
print(f"\nAnswer: {result.output}")
print(f"Success: {result.success}")

print("\n" + "=" * 60)
print("Weather example complete!")
print("=" * 60)
