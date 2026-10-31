"""Stop-token selection for the local Transformers engine.

A model may declare several terminators in its own ``generation_config`` while
the tokenizer reports only one. Llama 3.x is the case that matters: the
tokenizer reports ``<|eot_id|>``, but a tool call ends with ``<|eom_id|>``.
Passing only the tokenizer's id leaves ``<|eom_id|>`` an ordinary token, so the
model continues past its tool call and writes the assistant turn that should
have followed the tool result — inventing an observation it never received.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from effgen.models.transformers_engine import TransformersEngine  # noqa: E402


def _engine(declared, tokenizer_eos):
    """An engine stub carrying only what stop-token selection reads."""
    engine = TransformersEngine.__new__(TransformersEngine)
    engine.model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=declared)
    )
    engine.tokenizer = SimpleNamespace(eos_token_id=tokenizer_eos)
    return engine


class TestEosTokenIds:
    def test_llama3_end_of_message_is_a_stop_token(self):
        # <|end_of_text|>, <|eom_id|>, <|eot_id|> as Llama 3.2 declares them.
        ids = _engine([128001, 128008, 128009], 128009)._eos_token_ids()
        assert 128008 in ids, "end-of-message must stop generation"
        assert set(ids) == {128001, 128008, 128009}

    def test_qwen_terminators_are_all_kept(self):
        ids = _engine([151645, 151643], 151645)._eos_token_ids()
        assert set(ids) == {151645, 151643}

    def test_single_terminator_model_is_unchanged(self):
        """A model with one terminator still passes a bare int, as before."""
        assert _engine(2, 2)._eos_token_ids() == 2

    def test_tokenizer_id_is_kept_when_the_model_declares_none(self):
        assert _engine(None, 50256)._eos_token_ids() == 50256

    def test_tokenizer_id_is_added_when_the_model_omits_it(self):
        ids = _engine([128001], 128009)._eos_token_ids()
        assert set(ids) == {128001, 128009}

    def test_no_terminator_anywhere_returns_none(self):
        assert _engine(None, None)._eos_token_ids() is None

    def test_ids_are_not_duplicated(self):
        """A repeated id collapses to the single-terminator form."""
        assert _engine([128009, 128009], 128009)._eos_token_ids() == 128009

    def test_booleans_are_not_read_as_token_ids(self):
        assert _engine([True, 128009], False)._eos_token_ids() == 128009

    def test_non_integer_entries_are_ignored(self):
        ids = _engine([128001, None, "x", 128008], 128009)._eos_token_ids()
        assert set(ids) == {128001, 128008, 128009}
