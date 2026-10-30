"""The ``reasoning`` catalog flag must reach the adapter that needs it.

A model that spends output budget on a hidden chain before writing anything
visible needs a larger first budget, or it returns an empty — but billed —
result. The budget comes from ``default_max_output_tokens()``, which reads
``_is_reasoning_model`` off the adapter, which in turn reads ``reasoning`` off
the catalog record. A catalog that flags a model whose adapter never reads the
flag looks correct and changes nothing, which is how groq, fireworks and
cerebras came to carry the metadata without the behaviour.

These tests need no network: the adapters resolve their catalog record in
``__init__`` and defer the client to first use.
"""
from __future__ import annotations

import pytest

from effgen.models._adapter_utils import default_max_output_tokens, needs_reasoning_headroom

_FAKE_KEY = "test-key-not-used"


def _providers():
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


def _adapter(provider: str, model_id: str):
    if provider == "groq":
        from effgen.models.groq_adapter import GroqAdapter

        return GroqAdapter(model_name=model_id, api_key=_FAKE_KEY, enable_rate_limiting=False)
    if provider == "together":
        from effgen.models.together_adapter import TogetherAdapter

        return TogetherAdapter(model_name=model_id, api_key=_FAKE_KEY)
    if provider == "fireworks":
        from effgen.models.fireworks_adapter import FireworksAdapter

        return FireworksAdapter(
            model_name=model_id, api_key=_FAKE_KEY, enable_rate_limiting=False
        )
    from effgen.models.cerebras_adapter import CerebrasAdapter

    return CerebrasAdapter(model_name=model_id, api_key=_FAKE_KEY, enable_rate_limiting=False)


@pytest.mark.parametrize("provider", sorted(_providers()))
def test_a_flagged_catalog_entry_earns_the_larger_budget(provider):
    """Every ``reasoning: True`` record produces an adapter that says so."""
    catalog = _providers()[provider]
    flagged = [mid for mid, rec in catalog.items() if rec.get("reasoning")]
    assert flagged, f"{provider} flags no model as a reasoning model"
    for model_id in flagged:
        adapter = _adapter(provider, model_id)
        assert needs_reasoning_headroom(adapter), (
            f"{provider}:{model_id} is flagged 'reasoning' in the catalog but its "
            "adapter does not report it — the flag changes nothing"
        )
        assert default_max_output_tokens(adapter) == 4096


@pytest.mark.parametrize("provider", sorted(_providers()))
def test_an_unflagged_entry_keeps_the_ordinary_budget(provider):
    """The flag is read, not assumed: an unflagged model keeps 1024.

    Guards the mirror-image mistake — an adapter that hardcodes the larger
    budget for the whole provider would pass the test above and silently
    quadruple the first budget for every ordinary chat model.
    """
    catalog = _providers()[provider]
    unflagged = [
        mid
        for mid, rec in catalog.items()
        if not rec.get("reasoning") and rec.get("modality", "chat") in ("chat", "vision")
        and not mid.split("/")[-1].startswith(("gpt-5", "o1", "o3", "o4"))
    ]
    if not unflagged:
        pytest.skip(f"every chat entry in the {provider} catalog is a reasoning model")
    adapter = _adapter(provider, unflagged[0])
    assert not needs_reasoning_headroom(adapter)
    assert default_max_output_tokens(adapter) == 1024


def test_every_callable_fireworks_and_cerebras_model_is_flagged():
    """Both providers serve only reasoning models today.

    Measured on 2026-08-13 with one real call per id: every Fireworks chat and
    vision entry and both Cerebras entries returned a reasoning chain. Recorded
    as a test so a future catalog refresh that adds a model has to make the same
    decision deliberately rather than inheriting an unflagged default.
    """
    from effgen.models.cerebras_models import CEREBRAS_MODELS
    from effgen.models.fireworks_models import FIREWORKS_MODELS

    unflagged_fw = [
        mid
        for mid, rec in FIREWORKS_MODELS.items()
        if rec.get("modality") in ("chat", "vision") and not rec.get("reasoning")
    ]
    assert unflagged_fw == []
    assert [mid for mid, rec in CEREBRAS_MODELS.items() if not rec.get("reasoning")] == []
    # The embedding entries are not chat models and carry no flag.
    assert [
        mid for mid, rec in FIREWORKS_MODELS.items()
        if rec.get("modality") == "embedding" and rec.get("reasoning")
    ] == []
