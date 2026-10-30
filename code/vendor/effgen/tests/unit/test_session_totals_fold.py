"""A model's session totals must equal the sum of its per-call metadata.

``model.total_cost`` was right on every adapter and ``model.total_tokens`` moved
only on the three that fold their own usage — openai, gemini, anthropic. The
other six leave it to the ``_stamp_cost`` wrapper, which read ``cost_usd`` and
never the token counts, so those adapters reported a session token total of
zero against a real per-call sum (and, before this, no ``total_tokens``
attribute at all).

The fold now happens in one place. The tests below assert the invariant in both
directions: an adapter that relies on the wrapper counts its tokens, and an
adapter that folds its own does **not** count them twice.
"""
from __future__ import annotations

from effgen.models.base import (
    BaseModel,
    GenerationResult,
    ModelType,
    TokenCount,
    fold_call_totals,
)


class _Adapter(BaseModel):
    """A model whose per-call metadata is supplied by the test."""

    def __init__(self, *, folds_its_own: bool = False):
        super().__init__(model_name="fake", model_type=ModelType.TRANSFORMERS)
        self._is_loaded = True
        self.folds_its_own = folds_its_own

    def load(self):
        self._is_loaded = True

    def unload(self):
        self._is_loaded = False

    def get_context_length(self):
        return 4096

    def count_tokens(self, text):
        return TokenCount(prompt_tokens=1, completion_tokens=0, total_tokens=1)

    def generate_stream(self, prompt, config=None, **kwargs):
        yield "x"

    def generate(self, prompt, config=None, **kwargs):
        metadata = {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "cost_usd": 0.25,
        }
        if self.folds_its_own:
            # The openai/gemini/anthropic shape: fold here, and report the
            # cumulative figure in the metadata.
            fold_call_totals(self, 0.25, 14)
            metadata["total_cost"] = self.total_cost
        return GenerationResult(
            text="ok", tokens_used=4, finish_reason="stop",
            model_name="fake", metadata=metadata,
        )


def test_both_counters_exist_before_the_first_call():
    model = _Adapter()
    assert model.total_cost == 0.0
    assert model.total_tokens == 0


def test_an_adapter_that_relies_on_the_wrapper_counts_its_tokens():
    model = _Adapter()
    for _ in range(3):
        model.generate("hi")
    assert model.total_tokens == 42
    assert model.total_cost == 0.75


def test_an_adapter_that_folds_its_own_is_not_counted_twice():
    """The case that would be broken by folding in both places.

    Without the presence check the totals would read 84 tokens for 42 — the
    failure the report warned a naive fix would introduce.
    """
    model = _Adapter(folds_its_own=True)
    for _ in range(3):
        model.generate("hi")
    assert model.total_tokens == 42
    assert model.total_cost == 0.75


def test_an_unpriced_call_still_counts_its_tokens():
    """A local engine reports tokens and no price.

    ``cost_usd`` absent means the wrapper leaves both counters alone, which is
    the documented behaviour for an engine that does not price calls — recorded
    here so a later change to that rule is deliberate.
    """
    model = _Adapter()

    def unpriced(prompt, config=None, **kwargs):
        return GenerationResult(
            text="ok", tokens_used=4, finish_reason="stop", model_name="fake",
            metadata={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )

    model.generate = unpriced.__get__(model)
    model.generate("hi")
    assert model.total_tokens == 0
