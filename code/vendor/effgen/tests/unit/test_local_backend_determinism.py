"""Local-backend deterministic-generation and tokenizer-hygiene tests.

These exercise real (non-mocked) small-model behavior of the Transformers
backend:

* ``temperature <= 0`` must be treated as greedy decoding (``do_sample=False``)
  instead of being passed through to Transformers, which (5.x) raises
  ``temperature (=0.0) has to be a strictly positive float``.
* Greedy generation must be deterministic and must not emit the "generation
  flags are not valid" warning, nor the destructive BPE
  ``clean_up_tokenization_spaces`` warning.

The model used is the tiny, ungated ``gpt2`` (already a repo test anchor); these
run on CPU and are skipped offline when it is not cached.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from effgen.models.base import GenerationConfig
from effgen.models.transformers_engine import TransformersEngine

TINY_MODEL = "gpt2"


def _skip_if_not_cached(model_id: str) -> None:
    """Skip when ``model_id`` isn't already in the local HF cache.

    A *cold* download can stall a network-restricted CI runner well past the
    per-test timeout (a hang reported as an error, not a clean skip). When the
    weights aren't cached we skip up front rather than attempt the fetch; a warm
    cache (the normal local/CI-with-prefetch case) runs as before. ``HF_HUB_OFFLINE``
    is respected by letting ``load()`` fail fast into the fixture's skip path.
    """
    import os

    if os.environ.get("HF_HUB_OFFLINE") not in (None, "", "0"):
        return
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # huggingface_hub layout changed → let load() decide
        return
    # The "previously 404'd" sentinel moved from .constants to the package root
    # across huggingface_hub versions; try both, else fall back to a never-equal
    # object so the plain ``is None`` check below still works.
    try:
        from huggingface_hub import _CACHED_NO_EXIST
    except Exception:
        try:
            from huggingface_hub.constants import _CACHED_NO_EXIST
        except Exception:
            _CACHED_NO_EXIST = object()
    cached = try_to_load_from_cache(model_id, "config.json")
    if cached is None or cached is _CACHED_NO_EXIST:
        pytest.skip(
            f"'{model_id}' not in local HF cache; skipping to avoid a cold "
            "download that can hang CI"
        )


@pytest.fixture(scope="module")
def gpt2_engine():
    """Load the tiny gpt2 model once for the module; skip cleanly if unavailable
    (offline / not cached). Runs on GPU when one is visible, else CPU."""
    _skip_if_not_cached(TINY_MODEL)
    engine = TransformersEngine(TINY_MODEL, use_flash_attention=False)
    try:
        engine.load()
    except Exception as exc:  # offline / not cached / no usable device
        pytest.skip(f"Could not load '{TINY_MODEL}': {exc}")
    yield engine
    engine.unload()


def test_create_generation_config_greedy_for_nonpositive_temperature(gpt2_engine):
    """temperature <= 0 -> do_sample=False; sampling stays on for temperature > 0."""
    hf_cfg, _ = gpt2_engine._create_generation_config(
        GenerationConfig(max_tokens=8, temperature=0.0)
    )
    assert hf_cfg.do_sample is False

    hf_neg, _ = gpt2_engine._create_generation_config(
        GenerationConfig(max_tokens=8, temperature=-0.5)
    )
    assert hf_neg.do_sample is False

    hf_sample, _ = gpt2_engine._create_generation_config(
        GenerationConfig(max_tokens=8, temperature=0.7)
    )
    assert hf_sample.do_sample is True
    assert hf_sample.temperature == 0.7


def test_greedy_config_uses_neutral_sampling_values(gpt2_engine):
    """Greedy config must carry no-op sampling values so Transformers does not
    warn about "generation flags not valid for do_sample=False"."""
    hf_cfg, _ = gpt2_engine._create_generation_config(
        GenerationConfig(max_tokens=8, temperature=0.0)
    )
    assert hf_cfg.temperature == 1.0
    assert hf_cfg.top_p == 1.0
    assert hf_cfg.top_k == 50


def test_temperature_zero_generates_without_crashing(gpt2_engine):
    """The reproduced bug: temperature=0.0 raised "strictly positive float"."""
    result = gpt2_engine.generate(
        "The capital of France is",
        GenerationConfig(max_tokens=8, temperature=0.0),
    )
    assert isinstance(result.text, str)


def test_negative_temperature_generates_without_crashing(gpt2_engine):
    result = gpt2_engine.generate(
        "The capital of France is",
        GenerationConfig(max_tokens=8, temperature=-1.0),
    )
    assert isinstance(result.text, str)


def test_greedy_generation_is_deterministic(gpt2_engine):
    cfg = GenerationConfig(max_tokens=12, temperature=0.0)
    first = gpt2_engine.generate("Hello, my name is", cfg).text
    second = gpt2_engine.generate("Hello, my name is", cfg).text
    assert first == second


def test_per_call_zero_temperature_kwarg_is_greedy(gpt2_engine):
    """An explicit generate(..., temperature=0) override must also be greedy,
    not crash and not leave sampling flags set."""
    result = gpt2_engine.generate(
        "The capital of France is",
        GenerationConfig(max_tokens=8, temperature=0.7),
        temperature=0.0,
    )
    assert isinstance(result.text, str)


def test_per_call_greedy_override_folds_to_conflict_free_config(gpt2_engine):
    """Folding a per-call temperature<=0 override into a config that defaulted to
    sampling must overwrite the sampling fields with no-op values — not merely
    flip do_sample — or Transformers warns "generation flags are not valid for
    do_sample=False" on every such call."""
    base, _ = gpt2_engine._create_generation_config(
        GenerationConfig(max_tokens=8, temperature=0.7)
    )
    assert base.do_sample is True
    folded, extra = gpt2_engine._fold_into_generation_config(
        base,
        gpt2_engine._sanitize_generation_kwargs({"temperature": 0, "max_tokens": 8}),
    )
    assert folded.do_sample is False
    assert folded.temperature == 1.0
    assert folded.top_p == 1.0
    assert folded.top_k == 50
    assert extra == {}


def test_per_call_positive_temperature_enables_sampling(gpt2_engine):
    """A positive per-call temperature folded into a greedy base config must
    enable sampling so the override is honoured, not silently ignored."""
    base, _ = gpt2_engine._create_generation_config(
        GenerationConfig(max_tokens=8, temperature=0.0)
    )
    assert base.do_sample is False
    folded, _ = gpt2_engine._fold_into_generation_config(
        base, gpt2_engine._sanitize_generation_kwargs({"temperature": 0.8})
    )
    assert folded.do_sample is True
    assert folded.temperature == 0.8


def test_clean_up_tokenization_spaces_disabled_on_decode(gpt2_engine):
    """gpt2 uses a BPE tokenizer; decoding must not strip spaces before
    punctuation (clean_up_tokenization_spaces must be False)."""
    decoded = gpt2_engine.tokenizer.decode(
        gpt2_engine.tokenizer.encode("Hello, world! Done."),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # The destructive cleanup would turn "world! Done." into "world!Done." etc.
    assert "Hello, world! Done." in decoded


def test_small_max_tokens_reports_length_not_stop(gpt2_engine):
    """Cutting generation off at a tiny token budget (unlikely to land on EOS)
    must surface finish_reason="length" and metadata["truncated"]=True, not
    the "stop" HuggingFace's raw output otherwise defaults to."""
    result = gpt2_engine.generate(
        "The quick brown fox jumps over the lazy dog and then",
        GenerationConfig(max_tokens=2, temperature=0.0),
    )
    assert result.finish_reason == "length"
    assert result.metadata["truncated"] is True


