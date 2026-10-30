"""Integration test for CerebrasAdapter — skipped if key absent, real call otherwise."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("CEREBRAS_API_KEY"))


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: CEREBRAS_API_KEY not in ~/.effgen/.env")
class TestCerebrasLive:
    def test_generate_returns_text(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter
        from effgen.models.cerebras_models import CEREBRAS_DEFAULT_MODEL

        adapter = CerebrasAdapter(model_name=CEREBRAS_DEFAULT_MODEL)
        adapter.load()
        try:
            result = adapter.generate("Respond with exactly: CEREBRAS_OK")
            assert result.text, "Expected non-empty response from Cerebras"
            assert result.tokens_used > 0
            assert result.model_name == CEREBRAS_DEFAULT_MODEL
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "queue_exceeded" in msg or "high traffic" in msg.lower() or "rate limit" in msg.lower():
                pytest.skip(f"Cerebras transiently overloaded: {exc}")
            raise
        finally:
            adapter.unload()

    def test_load_model_via_provider(self):
        from effgen.models import load_model
        from effgen.models.cerebras_models import CEREBRAS_DEFAULT_MODEL

        model = load_model(CEREBRAS_DEFAULT_MODEL, provider="cerebras")
        try:
            result = model.generate("Say hello")
            assert result.text
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "queue_exceeded" in msg or "high traffic" in msg.lower() or "rate limit" in msg.lower():
                pytest.skip(f"Cerebras transiently overloaded: {exc}")
            raise
        finally:
            model.unload()

    def test_generate_stream_yields_chunks(self):
        from effgen.models.cerebras_adapter import CerebrasAdapter

        adapter = CerebrasAdapter()
        adapter.load()
        try:
            chunks = list(adapter.generate_stream("Count from 1 to 3 briefly."))
            assert len(chunks) >= 1, "Expected at least one chunk from streaming"
            full_text = "".join(chunks)
            assert len(full_text) > 0, "Expected non-empty streamed text"
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "queue_exceeded" in msg or "high traffic" in msg.lower() or "rate limit" in msg.lower():
                pytest.skip(f"Cerebras transiently overloaded: {exc}")
            raise
        finally:
            adapter.unload()
