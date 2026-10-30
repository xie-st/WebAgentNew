"""Integration tests for Gemini Files API (live API calls, skipped without key)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
pytestmark = pytest.mark.skipif(
    not GOOGLE_API_KEY,
    reason="GOOGLE_API_KEY not in ~/.effgen/.env",
)

MODEL = "gemini-3.1-flash-lite"
UNIQUE_FACT = "The secret ingredient of Elara's potion is powdered moonstone."


def test_file_upload_and_qa():
    """Upload a text file with a unique fact; model must mention it in answer."""
    from effgen.models.base import GenerationConfig
    from effgen.models.gemini_adapter import GeminiAdapter
    from effgen.models.gemini_files import upload_file

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, prefix="effgen_test_"
    ) as fh:
        fh.write(f"Elara's Secret Potion Recipe\n\n{UNIQUE_FACT}\n")
        tmp_path = Path(fh.name)

    try:
        ref = upload_file(tmp_path, api_key=GOOGLE_API_KEY)
        assert ref.uri, "upload_file should return a non-empty URI"

        adapter = GeminiAdapter(model_name=MODEL, api_key=GOOGLE_API_KEY)
        with adapter:
            result = adapter.generate(
                "What is the secret ingredient of Elara's potion? Answer in one sentence.",
                config=GenerationConfig(max_tokens=128),
                files=[ref],
            )

        assert result.text, "Expected a non-empty answer"
        lower = result.text.lower()
        assert "moonstone" in lower or "powdered" in lower, (
            f"Expected answer to mention 'moonstone' or 'powdered', got: {result.text!r}"
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def test_upload_file_ref_fields():
    """FileRef returned by upload_file has required fields."""
    from effgen.models.gemini_files import FileRef, upload_file

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, prefix="effgen_fref_"
    ) as fh:
        fh.write("Hello from effGen test.\n")
        tmp_path = Path(fh.name)

    try:
        ref = upload_file(tmp_path, api_key=GOOGLE_API_KEY)
        assert isinstance(ref, FileRef)
        assert ref.uri
        assert ref.mime_type == "text/plain"
        assert ref.display_name == tmp_path.name
        assert ref.raw is not None
    finally:
        tmp_path.unlink(missing_ok=True)
