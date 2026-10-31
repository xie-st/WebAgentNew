"""OpenAIAdapter registry-fallback log level.

An unknown model id is a normal, recoverable situation (a new or hot-swapped
id not yet in the bundled catalog) — not something requiring the user's
attention on every successful run. Construction never makes a network call,
so this is a pure offline unit test.
"""

from __future__ import annotations

import logging

from effgen.models.openai_adapter import OpenAIAdapter


def test_unknown_model_registry_fallback_logs_at_info_not_warning(caplog):
    with caplog.at_level(logging.INFO, logger="effgen.models.openai_adapter"):
        OpenAIAdapter(model_name="gpt-5-nano-totally-unknown-variant", api_key="sk-test-construction-only")
    fallback_records = [
        r for r in caplog.records if "is not in the OpenAI registry" in r.message
    ]
    assert fallback_records, "expected the registry-fallback note to be logged"
    assert all(r.levelno == logging.INFO for r in fallback_records)
    assert not any(r.levelno >= logging.WARNING for r in fallback_records)


def test_known_model_logs_no_registry_fallback_note(caplog):
    with caplog.at_level(logging.INFO, logger="effgen.models.openai_adapter"):
        OpenAIAdapter(model_name="gpt-5-nano", api_key="sk-test-construction-only")
    assert not any("is not in the OpenAI registry" in r.message for r in caplog.records)
