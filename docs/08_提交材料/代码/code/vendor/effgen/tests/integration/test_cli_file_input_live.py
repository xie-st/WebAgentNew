"""``effgen run --file`` — attaching a document or image from the CLI.

The offline tests exercise the help text and the fail-closed missing-file
path. The live test drives the real CLI subprocess against a cheap cloud
model to prove a spreadsheet handed to ``--file`` actually reaches the model
as task context — no mocking of model behavior. Skipped when no OpenAI key
is configured.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _run_cli(*args, timeout=60):
    cmd = [sys.executable, "-m", "effgen.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class TestRunFileHelp:
    def test_help_documents_file_option(self):
        result = _run_cli("run", "--help")
        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "--file" in output
        assert "--input" in output


class TestRunFileFailClosed:
    def test_missing_file_fails_before_any_model_call(self, tmp_path):
        result = _run_cli(
            "run", "summarize this",
            "--file", str(tmp_path / "does-not-exist.pdf"),
            "-m", "openai:gpt-5-nano", "-q",
        )
        assert result.returncode == 1
        assert "file not found" in (result.stdout + result.stderr).lower()

    def test_image_with_stream_rejected(self, tmp_path):
        png = tmp_path / "x.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        result = _run_cli(
            "run", "describe this",
            "--file", str(png), "--stream",
            "-m", "openai:gpt-5-nano", "-q",
        )
        assert result.returncode == 1
        assert "--stream" in (result.stdout + result.stderr)


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: OPENAI_API_KEY not set")
class TestRunFileLive:
    def test_xlsx_content_reaches_model(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["item", "amount"])
        ws.append(["widgets", 42])
        xlsx = tmp_path / "data.xlsx"
        wb.save(xlsx)

        result = _run_cli(
            "run", "What amount is listed for widgets? Answer with just the number.",
            "--file", str(xlsx), "-m", "openai:gpt-5-nano", "-q", "--json",
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["success"] is True
        assert "42" in payload["output"]

    def test_txt_content_reaches_model(self, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("The secret code word is banjo-77.")
        result = _run_cli(
            "run", "What is the secret code word in the attached document?",
            "--file", str(doc), "-m", "openai:gpt-5-nano", "-q", "--json",
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["success"] is True
        assert "banjo-77" in payload["output"]
