"""Live DX verification: drive the Jupyter magics through a real IPython shell.

Uses the configured Cerebras model. Not a unit test; run manually.

Usage:  python tools/_live_jupyter_verify.py
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv("/data/wang/gks/Development/effgen_dev/effGen/.env")

# Cheap model for chat; agent uses the general preset (has a calculator tool).
os.environ.setdefault("EFFGEN_JUPYTER_MODEL", "cerebras:llama3.1-8b")

from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from IPython.utils.capture import capture_output  # noqa: E402

shell = InteractiveShell.instance()
shell.run_line_magic("load_ext", "effgen.jupyter")

failures: list[str] = []

# --- Step 2: %effgen_chat "hi" -> non-empty response ---------------------
with capture_output() as cap:
    shell.run_line_magic("effgen_chat", "Say exactly: JUPYTER_LIVE_OK")
chat_out = "\n".join(o.data.get("text/markdown", "") if hasattr(o, "data") else str(o)
                     for o in cap.outputs)
print("=== %effgen_chat output ===")
print(chat_out)
if "JUPYTER_LIVE_OK" not in chat_out and len(chat_out.strip()) < 20:
    failures.append("effgen_chat produced no usable response")

# --- Step 3: %%effgen_agent general "Compute 17*23." ---------------------
# The answer includes 391 when the configured free model fits the preset, or
# skips cleanly with a message when it does not. The default Jupyter model
# (llama3.1-8b, 8k context) cannot fit the large `general` preset, so accept the
# clean context-overflow hint there and separately prove the tool path computes
# 391 using the small-footprint `math` preset.
with capture_output() as cap:
    shell.run_cell_magic("effgen_agent", "general", "Compute 17*23.")
agent_out = "\n".join(o.data.get("text/markdown", "") if hasattr(o, "data") else str(o)
                      for o in cap.outputs)
print("\n=== %%effgen_agent general output ===")
print(agent_out)
clean_skip = "don't fit this model's context window" in agent_out
rate_limited = "rate limit" in agent_out.lower()
if "391" not in agent_out and not clean_skip and not rate_limited:
    failures.append("effgen_agent general neither produced 391 nor skipped cleanly")

# The math preset fits the free model. Retry around transient free-tier 429s.
math_out = ""
for _attempt in range(4):
    time.sleep(20)  # respect free-tier requests-per-minute quota
    with capture_output() as cap:
        shell.run_cell_magic("effgen_agent", "math", "Compute 17*23.")
    math_out = "\n".join(o.data.get("text/markdown", "") if hasattr(o, "data") else str(o)
                         for o in cap.outputs)
    if "391" in math_out or "rate limit" not in math_out.lower():
        break
print("\n=== %%effgen_agent math output ===")
print(math_out)
if "391" not in math_out:
    failures.append("effgen_agent math did not produce 391 for 17*23")

# --- %effgen_metrics renders ---------------------------------------------
with capture_output() as cap:
    shell.run_line_magic("effgen_metrics", "")
metrics_out = "\n".join(o.data.get("text/html", "") if hasattr(o, "data") else str(o)
                        for o in cap.outputs)
print("\n=== %effgen_metrics rendered (len) ===", len(metrics_out))

if failures:
    print("\nLIVE VERIFY FAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("\nLIVE VERIFY: ALL PASS")
