"""Tests for LanguageDetectTool (fully offline — no network required)."""

from __future__ import annotations

import asyncio

from effgen.tools.builtin import LanguageDetectTool


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Metadata / static tests
# ---------------------------------------------------------------------------

def test_langdetect_metadata():
    tool = LanguageDetectTool()
    assert tool.metadata.name == "language_detect"
    assert "detect" in tool.metadata.tags


def test_langdetect_openai_schema_has_array_items():
    schema = LanguageDetectTool().metadata.to_json_schema()
    texts_schema = schema["parameters"]["properties"]["texts"]
    assert texts_schema["type"] == "array"
    assert texts_schema["items"] == {"type": "string"}


def test_langdetect_missing_text_fails():
    r = _run(LanguageDetectTool().execute(operation="detect"))
    assert not r.success
    assert r.metadata["error_type"] == "InvalidInputError"
    assert "non-empty" in r.error


def test_langdetect_missing_texts_fails():
    r = _run(LanguageDetectTool().execute(operation="detect_batch"))
    assert not r.success


def test_langdetect_unknown_operation_fails():
    r = _run(LanguageDetectTool().execute(operation="bogus", text="hello"))
    assert not r.success


def test_langdetect_empty_text_fails():
    r = _run(LanguageDetectTool().execute(operation="detect", text="   "))
    assert not r.success
    assert r.output is None
    assert r.metadata["error_type"] == "InvalidInputError"
    assert "non-empty" in r.error


# ---------------------------------------------------------------------------
# Functional tests (uses real langdetect — fully offline)
# ---------------------------------------------------------------------------

def test_langdetect_english():
    r = _run(LanguageDetectTool().execute(operation="detect", text="The quick brown fox jumps over the lazy dog"))
    assert r.success
    out = r.output
    assert out["success"] is True
    assert out["language"] == "en"
    assert out["confidence"] > 0.5


def test_langdetect_french():
    r = _run(LanguageDetectTool().execute(operation="detect", text="Bonjour, comment allez-vous aujourd'hui?"))
    assert r.success
    out = r.output
    assert out["success"] is True
    assert out["language"] == "fr"


def test_langdetect_spanish():
    r = _run(LanguageDetectTool().execute(operation="detect", text="Buenos días, ¿cómo estás hoy?"))
    assert r.success
    out = r.output
    assert out["success"] is True
    assert out["language"] == "es"


def test_langdetect_german():
    r = _run(LanguageDetectTool().execute(operation="detect", text="Guten Morgen, wie geht es Ihnen heute?"))
    assert r.success
    out = r.output
    assert out["success"] is True
    assert out["language"] == "de"


def test_langdetect_japanese():
    r = _run(LanguageDetectTool().execute(operation="detect", text="こんにちは、今日はどうですか？"))
    assert r.success
    out = r.output
    assert out["success"] is True
    assert out["language"] == "ja"


def test_langdetect_output_structure():
    r = _run(LanguageDetectTool().execute(operation="detect", text="Hello world, this is a test sentence."))
    assert r.success
    out = r.output
    assert "language" in out
    assert "confidence" in out
    assert "language_name" in out
    assert "all_candidates" in out["data"]
    assert isinstance(out["data"]["all_candidates"], list)
    assert len(out["data"]["all_candidates"]) >= 1


def test_langdetect_batch():
    texts = [
        "Hello, how are you?",
        "Bonjour le monde",
        "Hola, ¿cómo estás?",
        "こんにちは",
    ]
    r = _run(LanguageDetectTool().execute(operation="detect_batch", texts=texts))
    assert r.success
    out = r.output
    assert out["success"] is True
    assert out["count"] == 4
    results = out["results"]
    assert len(results) == 4
    # Check structure
    for item in results:
        assert "index" in item
        assert "language" in item
        assert "confidence" in item
        assert "success" in item
    # English should be detected correctly
    assert results[0]["language"] == "en"


def test_langdetect_batch_includes_indices():
    texts = ["Hello", "Bonjour", "Hola"]
    r = _run(LanguageDetectTool().execute(operation="detect_batch", texts=texts))
    assert r.success
    out = r.output
    indices = [item["index"] for item in out["results"]]
    assert indices == [0, 1, 2]


def test_langdetect_language_name_populated():
    r = _run(LanguageDetectTool().execute(operation="detect", text="This is an English sentence."))
    assert r.success
    out = r.output
    assert out["language_name"] == "English"
