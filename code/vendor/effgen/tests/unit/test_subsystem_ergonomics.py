"""Ergonomics of public subsystem constructors and method names.

These cover the "obvious call works" contract for memory / eval / config /
rag / prompts: bare constructors are populated (not traps) and the expected
method/argument names exist as additive aliases.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def test_template_manager_populated_by_default():
    from effgen.prompts import TemplateManager, create_default_template_manager

    tm = TemplateManager()
    assert len(tm.templates) > 0, "TemplateManager() must load bundled templates"
    # Equivalent to the explicit factory, with no spurious version history.
    default = create_default_template_manager()
    assert len(default.templates) == len(tm.templates)
    assert all(len(v) == 0 for v in default.template_versions.values())


def test_template_manager_opt_out():
    from effgen.prompts import TemplateManager

    empty = TemplateManager(load_defaults=False)
    assert len(empty.templates) == 0


def test_template_manager_get_alias():
    from effgen.prompts import TemplateManager

    tm = TemplateManager()
    name = tm.list_templates()[0]
    assert tm.get(name) is tm.get_template(name)


def test_prompt_optimizer_front_door():
    from effgen.prompts import PromptOptimizer

    res = PromptOptimizer().optimize("Could you please summarize this text for me?")
    assert hasattr(res, "optimized_prompt")
    assert res.optimized_prompt  # non-empty


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_loader_load_alias(tmp_path):
    from effgen.config import ConfigLoader

    cfg = tmp_path / "c.yaml"
    # A model entry needs a `type`; load_config validates by default, and this
    # test is about load() forwarding, not about a document being rejected.
    cfg.write_text("models:\n  demo:\n    type: openai\n    model_name: demo\n")
    loader = ConfigLoader()
    assert hasattr(loader, "load")
    # load() forwards to load_config()
    out = loader.load(str(cfg))
    assert out is loader.config


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_short_term_memory_get_messages():
    from effgen.memory import MessageRole, ShortTermMemory

    stm = ShortTermMemory()
    stm.add_message(MessageRole.USER, "one")
    stm.add_message(MessageRole.ASSISTANT, "two")
    assert len(stm.get_messages()) == 2
    assert len(stm.get_messages(1)) == 1
    assert stm.get_messages() == stm.get_recent_messages()


# ---------------------------------------------------------------------------
# RAG chunking
# ---------------------------------------------------------------------------

def test_rag_text_chunker():
    import effgen.rag as rag
    from effgen.rag import TextChunker

    assert "TextChunker" in rag.__all__
    chunker = TextChunker(chunk_size=80, overlap=10)
    text = "Sentence one is here. Sentence two follows next. " * 8
    pieces = chunker.split_text(text)
    assert len(pieces) > 1
    assert all(isinstance(p, str) for p in pieces)
    docs = chunker.chunk(text, "doc1")
    assert len(docs) > 1


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def test_testsuite_accepts_test_cases():
    from effgen.eval import TestCase, TestSuite

    cases = [TestCase(query="2+2?", expected_output="4")]
    suite = TestSuite(test_cases=cases)
    assert len(suite) == 1
    assert list(suite)[0].query == "2+2?"


def test_bundled_suite_still_loads_from_file():
    from effgen.eval.suites import MathSuite

    assert len(MathSuite()) > 0


def test_testcase_input_alias():
    from effgen.eval import TestCase

    tc = TestCase(input="What is the capital of France?", expected_output="Paris")
    assert tc.query == "What is the capital of France?"
    # The alias is not retained as a persistent field.
    assert "input" not in repr(tc)


def test_testcase_difficulty_string_coerced():
    from effgen.eval import TestCase
    from effgen.eval.evaluator import Difficulty

    tc = TestCase(query="q", difficulty="easy")
    assert tc.difficulty is Difficulty.EASY


def test_testcase_requires_query():
    from effgen.eval import TestCase

    with pytest.raises(ValueError):
        TestCase()
