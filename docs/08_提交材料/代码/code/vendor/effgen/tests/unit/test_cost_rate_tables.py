"""The fallback rate table and the catalogs must not price the same id twice.

``effgen/models/_cost.py`` carries a small table of per-token rates for ids no
catalog knows. It is a fallback *and* a price history: a stored run record whose
model has since left its provider's listing is still priced from it, which is
why it is not pruned to the current roster.

The property that has to hold is agreement. Where the table and a catalog both
price an id, the number a run reports would otherwise depend on which lookup
answered first.
"""
from __future__ import annotations

import pytest

from effgen.models._cost import _RATES


def _catalogs():
    from effgen.models.cerebras_models import CEREBRAS_MODELS
    from effgen.models.fireworks_models import FIREWORKS_MODELS
    from effgen.models.groq_models import GROQ_MODELS
    from effgen.models.together_models_data import TOGETHER_MODELS

    return {
        "groq": GROQ_MODELS,
        "together": TOGETHER_MODELS,
        "fireworks": FIREWORKS_MODELS,
        "cerebras": CEREBRAS_MODELS,
    }


@pytest.mark.parametrize("provider", sorted(_catalogs()))
def test_a_shared_id_is_priced_the_same_in_both_tables(provider):
    catalog = _catalogs()[provider]
    rates = _RATES.get(provider, {})
    disagreements = []
    for model_id, (price_in, price_out) in rates.items():
        if model_id == "*" or model_id not in catalog:
            continue
        record = catalog[model_id]
        catalog_in = record.get("pricing_per_1m_input")
        catalog_out = record.get("pricing_per_1m_output")
        if catalog_in is None or catalog_out is None:
            continue
        # Compared with a tolerance: several catalog prices are computed rather
        # than written, so they carry the usual binary-float tail.
        if (
            abs(catalog_in - price_in) > 1e-9
            or abs(catalog_out - price_out) > 1e-9
        ):
            disagreements.append(
                f"{provider}:{model_id} — table {(price_in, price_out)} "
                f"vs catalog {(catalog_in, catalog_out)}"
            )
    assert not disagreements, "\n".join(disagreements)


def test_the_wildcard_row_never_makes_an_unknown_id_look_priced():
    """A ``"*"`` placeholder is an estimate, not a published price.

    It stays available to callers that want a rough figure, and nothing bills
    off it: an id the catalog has not seen reports no cost rather than a
    fabricated one.
    """
    from effgen.models._cost import call_cost, pricing_status

    assert pricing_status("openai", "ft:gpt-4o-2024-08-06:acme::abc123") == "unpriced"
    assert call_cost("openai", "ft:gpt-4o-2024-08-06:acme::abc123", 1000, 1000) is None


def test_an_id_the_catalog_dropped_is_still_priced_from_the_table():
    """The reason the table is not pruned: it prices stored history.

    Eight Together ids sit in the table and not in the catalog. They cannot be
    called — the adapter refuses an id its catalog does not carry — but a run
    record that names one still reports what it cost.
    """
    from effgen.models._cost import _RATES as rates

    historical = [
        model_id
        for model_id in rates.get("together", {})
        if model_id != "*" and model_id not in _catalogs()["together"]
    ]
    assert historical, "the table no longer carries any history to price"

    from effgen.models._cost import call_cost

    for model_id in historical:
        cost = call_cost("together", model_id, 1_000_000, 0)
        assert cost is not None and cost > 0, model_id
