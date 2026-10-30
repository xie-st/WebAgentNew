"""
Live integration test: the Gemini adapter reports a real per-call cost.

The Gemini adapter previously emitted only a ``cost`` metadata key and never the
canonical ``cost_usd`` that every other adapter sets, so cost-aware callers
(``effgen cost``, dashboards) saw ``None`` for a clearly-priced model. This
asserts ``cost_usd`` is a float for both a flash-lite and a paid 2.5 model, and
that the process-global cost tracker aggregates Gemini spend.

Skipped automatically when GOOGLE_API_KEY is absent.
Run with:
    pytest tests/integration/test_gemini_cost_live.py -v -s
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
    reason="GOOGLE_API_KEY not set — live Gemini cost test skipped",
)


def _skip_on_transient(exc: Exception) -> None:
    msg = str(exc).lower()
    if any(k in msg for k in ("503", "unavailable", "overload", "429", "resource_exhausted", "quota")):
        pytest.skip(f"transient upstream condition: {exc}")


@skip_no_key
@pytest.mark.parametrize("model_id", ["gemini-3.1-flash-lite", "gemini-2.5-flash"])
def test_gemini_reports_float_cost_usd(model_id):
    from effgen import load_model
    from effgen.models.base import GenerationConfig

    model = load_model(f"gemini:{model_id}")
    try:
        result = model.generate("Say hi.", config=GenerationConfig(max_tokens=8))
    except Exception as exc:  # noqa: BLE001
        _skip_on_transient(exc)
        raise

    cost = result.metadata.get("cost_usd")
    assert cost is not None, f"{model_id}: cost_usd is None (silent missing cost)"
    assert isinstance(cost, float), f"{model_id}: cost_usd is {type(cost).__name__}, not float"
    assert cost >= 0.0


@skip_no_key
def test_cost_tracker_aggregates_gemini_spend():
    from effgen import load_model
    from effgen.models._cost import CostTracker
    from effgen.models.base import GenerationConfig

    model = load_model("gemini:gemini-3.1-flash-lite")
    try:
        model.generate("Hi.", config=GenerationConfig(max_tokens=8))
    except Exception as exc:  # noqa: BLE001
        _skip_on_transient(exc)
        raise

    rows = [r for r in CostTracker.get().summary() if r.get("provider") == "gemini"]
    assert rows, "CostTracker recorded no gemini spend"
    assert all(isinstance(r["cost_usd"], float) for r in rows)
