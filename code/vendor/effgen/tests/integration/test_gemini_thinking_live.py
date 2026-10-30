"""
Live integration test for Gemini thinking_budget.

Skipped automatically when GOOGLE_API_KEY is absent from ~/.effgen/.env.
Run with:
    pytest tests/integration/test_gemini_thinking_live.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).parents[2] / ".env", override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

skip_no_key = pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="GOOGLE_API_KEY not in ~/.effgen/.env — live Gemini tests skipped",
)

TEST_MODEL = "gemini-3.1-flash-lite"
MATH_PROMPT = (
    "A store sells apples for $1.50 each and oranges for $2.00 each. "
    "If Alice buys 4 apples and 3 oranges, how much does she spend in total? "
    "Show your work step by step."
)


def _skip_on_503(exc: Exception) -> None:
    """Skip on transient upstream unavailability (overload or free-tier quota).

    503/UNAVAILABLE is a transient server overload; 429/RESOURCE_EXHAUSTED is a
    free-tier rate/quota limit. Neither indicates an effGen bug, so we skip
    rather than fail the live test (mirrors the Cerebras/Replicate live tests).
    """
    msg = str(exc)
    if "503" in msg or "UNAVAILABLE" in msg:
        pytest.skip(f"Model unavailable (503 server overload) — {exc}")
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        pytest.skip(f"Free-tier quota exhausted (429) — {exc}")
    raise


@skip_no_key
def test_thinking_budget_disabled():
    """thinking_budget=0 disables reasoning; response should arrive without a thinking trace."""
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    try:
        with GeminiAdapter(model_name=TEST_MODEL) as model:
            result = model.generate(
                MATH_PROMPT,
                config=GenerationConfig(thinking_budget=0, max_tokens=512),
            )
    except RuntimeError as exc:
        _skip_on_503(exc)

    assert result.text.strip()
    assert "thinking" not in result.metadata or not result.metadata["thinking"]
    assert result.metadata["thoughts_token_count"] == 0


@skip_no_key
def test_thinking_budget_enabled():
    """thinking_budget=8192 with include_thoughts=True should surface a thinking trace."""
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    try:
        with GeminiAdapter(model_name=TEST_MODEL) as model:
            result = model.generate(
                MATH_PROMPT,
                config=GenerationConfig(
                    thinking_budget=8192,
                    include_thoughts=True,
                    max_tokens=1024,
                ),
            )
    except RuntimeError as exc:
        _skip_on_503(exc)

    assert result.text.strip()
    assert "thinking" in result.metadata and result.metadata["thinking"]
    assert result.metadata["thoughts_token_count"] >= 0


@skip_no_key
def test_thinking_budget_affects_answer_quality():
    """Both budget=0 and budget=4096 should yield the correct answer."""
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter

    try:
        with GeminiAdapter(model_name=TEST_MODEL) as model:
            result_no_think = model.generate(
                MATH_PROMPT,
                config=GenerationConfig(thinking_budget=0, max_tokens=256),
            )
            result_think = model.generate(
                MATH_PROMPT,
                config=GenerationConfig(thinking_budget=4096, max_tokens=512),
            )
    except RuntimeError as exc:
        _skip_on_503(exc)

    # Both should mention the correct answer ($12.00)
    assert "12" in result_no_think.text
    assert "12" in result_think.text
