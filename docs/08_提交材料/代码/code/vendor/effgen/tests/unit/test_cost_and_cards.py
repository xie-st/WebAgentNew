"""Cost display never hides a real charge, and the model layer renders a card.

The cheap small models effGen targets cost a tiny fraction of a cent per turn,
so a fixed ``${:.4f}`` collapsed almost every real cost to ``$0.0000``. These
tests pin the adaptive formatter and that ``GenerationResult`` renders a tidy
notebook card (not a dataclass repr).
"""

from __future__ import annotations

from effgen.models.base import GenerationResult
from effgen.ui.render import _summary_parts, format_cost


def test_format_cost_keeps_subcent_costs_visible():
    # The exact case from the field report: a real $0.0000486 must not show $0.0000.
    assert format_cost(0.0000486) == "$0.000049"
    assert format_cost(9.9e-5) == "$0.000099"
    # Familiar 4-dp form for >= $0.0001.
    assert format_cost(0.0006) == "$0.0006"
    assert format_cost(0.0123) == "$0.0123"
    # Vanishingly small -> scientific, still not $0.0000.
    assert format_cost(4.9e-7) == "$4.9e-07"


def test_format_cost_distinguishes_zero_from_unpriced():
    assert format_cost(0.0) == "$0.00"
    assert format_cost(None) is None
    assert format_cost("not a number") is None


def test_summary_strip_uses_adaptive_cost():
    class _R:
        execution_time = 1.0
        tool_calls = 0
        tokens_used = 94
        metadata = {"cost_usd": 0.0000486}

    parts = _summary_parts(_R())
    assert "$0.000049" in parts
    assert "$0.0000" not in parts


def test_generation_result_has_notebook_card():
    r = GenerationResult(
        text="**Bonjour**", tokens_used=12, finish_reason="stop", model_name="gpt-5-nano",
        metadata={"cost_usd": 0.0000486, "duration_s": 0.8, "total_tokens": 40},
    )
    html = r._repr_html_()
    assert "effGen model output" in html
    assert "gpt-5-nano" in html
    assert "$0.000049" in html  # adaptive cost on the card too
    md = r._repr_markdown_()
    assert md.startswith("**Bonjour**")
    assert "gpt-5-nano" in md


def test_generation_result_card_handles_missing_metadata():
    r = GenerationResult(text="hi", tokens_used=0, finish_reason="stop", model_name="m", metadata=None)
    # Must not raise even with no cost/latency present.
    assert "effGen model output" in r._repr_html_()
    assert r._repr_markdown_().startswith("hi")


def test_generation_result_card_is_theme_aware():
    # The card must follow the notebook's own light/dark theme instead of
    # hardcoding light-theme colors that are illegible on a dark background.
    r = GenerationResult(text="hi", tokens_used=1, finish_reason="stop", model_name="m")
    html = r._repr_html_()
    assert "prefers-color-scheme: dark" in html
    assert "color:#1a1a1a" not in html
    assert "color:#666" not in html
