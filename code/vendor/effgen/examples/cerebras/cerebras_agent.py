"""
Cerebras-powered Agent example with built-in tools.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.effgen/.env

What this demonstrates:
  - Building an Agent backed by CerebrasAdapter
  - Using Calculator and DateTimeTool
  - Multi-step task with tool invocations
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from effgen.core.agent import Agent, AgentConfig  # noqa: E402
from effgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402
from effgen.tools.builtin import Calculator, DateTimeTool  # noqa: E402

# -----------------------------------------------------------
# Build the model and agent
# -----------------------------------------------------------
adapter = CerebrasAdapter(model_name="gpt-oss-120b")
adapter.load()

config = AgentConfig(
    name="cerebras-assistant",
    model=adapter,
    tools=[Calculator(), DateTimeTool()],
    system_prompt=(
        "You are a helpful assistant. "
        "Use tools when they help you give a precise answer."
    ),
    max_iterations=5,
)

with Agent(config) as agent:
    tasks = [
        "What is 1337 * 42?",
        "What day of the week is it today?",
        "If I invest $5000 at 7% annual interest for 10 years, what is the final amount? (Use compound interest: A = P*(1+r)^t)",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        response = agent.run(task)
        print(f"Answer: {response.output}")
        print(f"  iterations={response.iterations}  tools_used={response.tool_calls}  tokens={response.tokens_used}")

adapter.unload()
print("\nAll tasks complete.")
