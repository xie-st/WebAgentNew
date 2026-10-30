"""Regression tests for special-token stripping on the local tool-call path.

When a local model is given tools, its output is decoded with
``skip_special_tokens=False`` so native tool-call delimiters survive for the
parser. The chat-template turn/end markers must still be removed before the
text reaches the user — a fixed strip-list silently let Gemma-family models
leak ``<end_of_turn><eos>`` into the answer. The strip set is therefore derived
from the tokenizer itself; these tests lock that behavior in offline (no model
download / GPU required) via a stub tokenizer.
"""

import pytest

pytest.importorskip("torch")

from effgen.models.transformers_engine import (  # noqa: E402
    _strip_special_tokens_keep_tool_calls as strip,
)


class _Tok:
    def __init__(self, specials):
        self.all_special_tokens = list(specials)


GEMMA = _Tok(["<bos>", "<eos>", "<end_of_turn>", "<start_of_turn>", "<pad>", "<unk>"])
QWEN = _Tok(["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
MISTRAL = _Tok(["<s>", "</s>", "<unk>"])
LLAMA = _Tok(["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>"])


@pytest.mark.parametrize(
    "tok, text, expected",
    [
        # Gemma: the exact leak that motivated the fix.
        (GEMMA, "8123 * 9512 = 779,285,968 \n<end_of_turn><eos>",
         "8123 * 9512 = 779,285,968"),
        (GEMMA, "<start_of_turn>Hello<end_of_turn><eos>", "Hello"),
        # Qwen turn markers stripped.
        (QWEN, "<|im_start|>Paris<|im_end|>", "Paris"),
        # Llama end markers stripped.
        (LLAMA, "Paris<|eot_id|><|end_of_text|>", "Paris"),
        # Mistral end marker stripped.
        (MISTRAL, "Paris</s>", "Paris"),
    ],
)
def test_strips_chat_template_markers(tok, text, expected):
    assert strip(text, tok) == expected


@pytest.mark.parametrize(
    "tok, text, expected",
    [
        # Native tool-call delimiters MUST survive for the downstream parser,
        # even when the tokenizer also lists end markers as special.
        (QWEN, '<|im_start|><tool_call>{"name":"calc"}</tool_call><|im_end|>',
         '<tool_call>{"name":"calc"}</tool_call>'),
        (MISTRAL, '[TOOL_CALLS][{"name":"calc"}]</s>',
         '[TOOL_CALLS][{"name":"calc"}]'),
        (LLAMA, '<|python_tag|>{"name":"calc"}<|eot_id|>',
         '<|python_tag|>{"name":"calc"}'),
    ],
)
def test_preserves_tool_call_delimiters(tok, text, expected):
    assert strip(text, tok) == expected


def test_tokenizer_without_special_tokens_attr_is_safe():
    """Belt-and-suspenders: common end markers stripped even with no specials."""
    class _Bare:
        pass
    assert strip("Paris</s><|im_end|>", _Bare()) == "Paris"
