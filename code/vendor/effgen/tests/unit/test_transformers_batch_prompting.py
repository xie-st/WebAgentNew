"""Batched local generation prepares its prompts the way single generation does.

``generate_batch`` fans prompts at one local model. Three things have to hold
for the answers to be answers at all, and none of them is visible in the return
type — a batch that gets them wrong still returns strings, they are simply not
replies to the prompts that were sent:

* every prompt goes through the chat template, so an instruct model sees its
  role tags rather than raw text to continue;
* padding goes on the **left**, because a decoder-only model continues from the
  last position of each row and the decode slices every row at the shared
  padded width;
* the prompt-token count reported per result is the prompt's own length, not
  the width it was padded to.

Driven with a stand-in tokenizer and model, so no weights are loaded.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from effgen.models.base import GenerationConfig  # noqa: E402
from effgen.models.transformers_engine import TransformersEngine  # noqa: E402


class _Encoding(dict):
    """Mapping the engine indexes like a tokenizer batch encoding."""


class _Tokenizer:
    """Records how it was called and left-pads to a common width."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.padding_side = "right"
        self.padding_side_when_called: str | None = None
        self.templated: list[str] = []
        self.encoded: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kw):
        text = messages[-1]["content"]
        self.templated.append(text)
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, prompts, **_kw):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.padding_side_when_called = self.padding_side
        self.encoded = list(prompts)
        lengths = [max(1, len(p.split())) for p in prompts]
        width = max(lengths)
        ids, mask = [], []
        for length in lengths:
            pad = width - length
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + [5] * length)
                mask.append([0] * pad + [1] * length)
            else:
                ids.append([5] * length + [self.pad_token_id] * pad)
                mask.append([1] * length + [0] * pad)
        return _Encoding(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)
        )

    def encode(self, text, **_kw):
        return [5] * max(1, len(text.split()))

    def decode(self, ids, **_kw):
        return "answer"


class _Model:
    """Returns the inputs with two generated tokens appended to each row."""

    device = "cpu"

    def generate(self, **kwargs):
        ids = kwargs["input_ids"]
        extra = torch.full((ids.shape[0], 2), 7, dtype=ids.dtype)
        return torch.cat([ids, extra], dim=1)


def _engine():
    engine = TransformersEngine.__new__(TransformersEngine)
    engine.model_name = "stand-in/model"
    engine.device = "cpu"
    engine.device_map = None
    engine.model = _Model()
    engine.tokenizer = _Tokenizer()
    engine._is_loaded = True
    engine._context_length = 2048
    engine._pin_device_map_for_cuda = False
    engine._retrying_after_cuda_assert = False
    engine._device_map_probe_passed = True
    engine._tool_template_probe = None
    import threading

    engine._tokenizer_lock = threading.RLock()
    return engine


PROMPTS = ["Say hi", "Reply with one word: hello", "What is two plus two exactly"]


def test_every_batched_prompt_goes_through_the_chat_template():
    """Without this an instruct model continues the prompt instead of answering."""
    engine = _engine()

    engine.generate_batch(PROMPTS, GenerationConfig(max_tokens=8))

    assert engine.tokenizer.templated == PROMPTS, (
        f"the template saw {engine.tokenizer.templated} instead of every prompt"
    )
    assert all("<|im_start|>" in encoded for encoded in engine.tokenizer.encoded), (
        "raw prompts reached the tokenizer, so the role tags never got there"
    )


def test_batched_prompts_are_padded_on_the_left():
    """A decoder-only model continues from the last position of each row."""
    engine = _engine()

    engine.generate_batch(PROMPTS, GenerationConfig(max_tokens=8))

    assert engine.tokenizer.padding_side_when_called == "left"


def test_the_tokenizer_keeps_the_padding_side_it_had():
    """The batch borrows the setting; it does not keep it."""
    engine = _engine()
    engine.tokenizer.padding_side = "right"

    engine.generate_batch(PROMPTS, GenerationConfig(max_tokens=8))

    assert engine.tokenizer.padding_side == "right"


def test_padding_is_not_reported_as_prompt_tokens():
    """The shortest prompt must not be billed for the longest one's width."""
    engine = _engine()

    results = engine.generate_batch(PROMPTS, GenerationConfig(max_tokens=8))

    counts = [r.metadata["prompt_tokens"] for r in results]
    assert len(set(counts)) > 1, (
        f"every prompt reported the same length {counts}, which is the padded width"
    )
    assert counts == sorted(counts), "prompt lengths do not track the prompts sent"
    for result in results:
        meta = result.metadata
        assert meta["total_tokens"] == meta["prompt_tokens"] + meta["completion_tokens"]
