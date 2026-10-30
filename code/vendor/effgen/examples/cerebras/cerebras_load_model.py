"""
Using Cerebras via the load_model() convenience function.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"

What this demonstrates:
  - The load_model(model_id, provider="cerebras") shortcut
  - Checking available Cerebras models
  - Adapter metadata inspection
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from effgen.models import load_model  # noqa: E402
from effgen.models.cerebras_models import CEREBRAS_MODELS  # noqa: E402

# -----------------------------------------------------------
# Show registered models
# -----------------------------------------------------------
print("Registered Cerebras models:")
for name, meta in CEREBRAS_MODELS.items():
    print(f"  {name:40s}  context={meta['context']:>8,}  max_output={meta['max_output']:>6,}")

# -----------------------------------------------------------
# Load via shorthand
# -----------------------------------------------------------
print("\nLoading gpt-oss-120b via load_model(provider='cerebras') ...")
model = load_model("gpt-oss-120b", provider="cerebras")
print(f"Model type     : {type(model).__name__}")
print(f"Context length : {model.get_context_length():,}")
print(f"Is loaded      : {model.is_loaded()}")

result = model.generate("Explain what Cerebras hardware is in one sentence.")
print(f"\nResponse: {result.text}")

model.unload()
print("Done.")
