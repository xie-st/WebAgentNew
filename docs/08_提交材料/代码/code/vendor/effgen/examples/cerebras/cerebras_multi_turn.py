"""
Multi-turn conversation with Cerebras using CerebrasAdapter directly.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"

What this demonstrates:
  - Maintaining conversation history manually
  - Passing a list of messages for multi-turn context
  - Controlling temperature for creative vs deterministic output
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from cerebras.cloud.sdk import Cerebras  # noqa: E402

from effgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402

adapter = CerebrasAdapter(model_name="gpt-oss-120b")
adapter.load()

# Access the underlying SDK client for multi-turn message passing
client: Cerebras = adapter._client

history = [
    {"role": "system", "content": "You are a concise travel guide. Keep replies under 3 sentences."},
]

turns = [
    "I want to visit Japan. What's the best time of year to go?",
    "What's one thing I absolutely must eat there?",
    "Any tips for getting around Tokyo?",
]

print("=== Multi-turn conversation with Cerebras ===\n")
for user_msg in turns:
    history.append({"role": "user", "content": user_msg})
    print(f"User  : {user_msg}")

    resp = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=history,
        max_completion_tokens=128,
        temperature=0.7,
    )
    assistant_reply = resp.choices[0].message.content or ""
    history.append({"role": "assistant", "content": assistant_reply})
    print(f"Cerebras: {assistant_reply}\n")

adapter.unload()
