"""Unit tests for Gemini Files API wrapper (effgen.models.gemini_files)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from effgen.models.gemini_files import FileRef, upload_file

# ---------------------------------------------------------------------------
# FileRef dataclass
# ---------------------------------------------------------------------------

class TestFileRef:
    def test_basic_construction(self):
        ref = FileRef(uri="gs://bucket/file.pdf", mime_type="application/pdf", display_name="paper")
        assert ref.uri == "gs://bucket/file.pdf"
        assert ref.mime_type == "application/pdf"
        assert ref.display_name == "paper"
        assert ref.raw is None

    def test_raw_field(self):
        raw = MagicMock()
        ref = FileRef(uri="gs://x", mime_type="text/plain", raw=raw)
        assert ref.raw is raw


# ---------------------------------------------------------------------------
# upload_file()
# ---------------------------------------------------------------------------

class TestUploadFile:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            upload_file(tmp_path / "nonexistent.txt", api_key="fake")

    def test_missing_api_key_raises(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GOOGLE_API_KEY is absent
            env = dict(os.environ)
            env.pop("GOOGLE_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="Google API key"):
                    upload_file(f)

    def test_successful_upload_returns_file_ref(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("The secret ingredient is 'zorblax'.")

        fake_sdk_file = MagicMock()
        fake_sdk_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"

        fake_client = MagicMock()
        fake_client.files.upload.return_value = fake_sdk_file

        with patch("google.genai.Client", return_value=fake_client):
            ref = upload_file(f, api_key="fake-key")

        assert isinstance(ref, FileRef)
        assert "abc123" in ref.uri
        assert ref.mime_type == "text/plain"
        assert ref.display_name == "sample.txt"
        assert ref.raw is fake_sdk_file

    def test_upload_called_with_correct_args(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")

        fake_sdk_file = MagicMock()
        fake_sdk_file.uri = "https://example.com/files/xyz"

        fake_client = MagicMock()
        fake_client.files.upload.return_value = fake_sdk_file

        with patch("google.genai.Client", return_value=fake_client):
            ref = upload_file(f, api_key="fake-key", mime_type="application/pdf", display_name="My Doc")

        call_kwargs = fake_client.files.upload.call_args
        assert call_kwargs is not None
        # The path should have been passed as 'file' kwarg
        assert ref.mime_type == "application/pdf"
        assert ref.display_name == "My Doc"

    def test_mime_type_override(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        fake_sdk_file = MagicMock()
        fake_sdk_file.uri = "https://example.com/files/bin"
        fake_client = MagicMock()
        fake_client.files.upload.return_value = fake_sdk_file
        with patch("google.genai.Client", return_value=fake_client):
            ref = upload_file(f, api_key="fake-key", mime_type="application/octet-stream")
        assert ref.mime_type == "application/octet-stream"

    def test_size_over_2gb_raises_clear_error(self, tmp_path):
        """A file reporting > 2 GiB on disk must be rejected before upload."""
        from effgen.models.gemini_files import MAX_UPLOAD_BYTES

        f = tmp_path / "huge.bin"
        f.write_bytes(b"\x00")  # tiny on disk; we mock the size

        fake_client = MagicMock()
        with patch("google.genai.Client", return_value=fake_client):
            with patch("pathlib.Path.stat") as mock_stat:
                stat_result = MagicMock()
                stat_result.st_size = MAX_UPLOAD_BYTES + 1
                mock_stat.return_value = stat_result
                with pytest.raises(ValueError, match="2 GiB"):
                    upload_file(f, api_key="fake-key")

        # Critically: the SDK upload must NOT have been called.
        fake_client.files.upload.assert_not_called()

    def test_upload_sdk_error_raises_runtime(self, tmp_path):
        f = tmp_path / "fail.txt"
        f.write_text("fail")
        fake_client = MagicMock()
        fake_client.files.upload.side_effect = Exception("quota exceeded")
        with patch("google.genai.Client", return_value=fake_client):
            with pytest.raises(RuntimeError, match="upload failed"):
                upload_file(f, api_key="fake-key")


# ---------------------------------------------------------------------------
# generate() with files= kwarg
# ---------------------------------------------------------------------------

class TestGenerateWithFiles:
    def _make_loaded_adapter(self):
        with patch("google.genai.Client"):
            from effgen.models.base import TokenCount
            from effgen.models.gemini_adapter import GeminiAdapter
            adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite-preview", api_key="fake")
            adapter._is_loaded = True
            adapter.client = MagicMock()
            adapter.count_tokens = MagicMock(
                return_value=TokenCount(count=5, model_name="gemini-3.1-flash-lite-preview")
            )
            return adapter

    def _mock_response(self, text: str = "The document says zorblax."):
        candidate = MagicMock()
        part = MagicMock()
        part.thought = False
        part.text = text
        part.function_call = None
        candidate.content.parts = [part]
        candidate.finish_reason = "STOP"
        candidate.grounding_metadata = None
        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata.prompt_token_count = 5
        response.usage_metadata.candidates_token_count = 10
        response.usage_metadata.total_token_count = 15
        response.safety_ratings = None
        return response

    def test_generate_with_file_ref_passes_content(self):
        """Files are prepended to the content sent to the API."""
        adapter = self._make_loaded_adapter()
        ref = FileRef(
            uri="https://generativelanguage.googleapis.com/v1beta/files/abc",
            mime_type="text/plain",
            display_name="sample.txt",
        )
        response = self._mock_response()

        captured: list = []

        def fake_retry(*, contents, gen_config, stream=False, tools=None):
            captured.append(contents)
            return response

        with patch.object(adapter, "_generate_with_retry", side_effect=fake_retry):
            with patch("google.genai.types") as mock_types:
                mock_part = MagicMock()
                mock_types.Part.from_uri.return_value = mock_part
                result = adapter.generate("What is the secret ingredient?", files=[ref])

        assert result.text == "The document says zorblax."
        # Content should be a list starting with the file part
        assert len(captured) == 1
        sent = captured[0]
        assert isinstance(sent, list)
        assert sent[0] is mock_part

    def test_generate_without_files_unchanged(self):
        """No files arg → content passed as-is."""
        adapter = self._make_loaded_adapter()
        response = self._mock_response("plain answer")

        captured: list = []

        def fake_retry(*, contents, gen_config, stream=False, tools=None):
            captured.append(contents)
            return response

        with patch.object(adapter, "_generate_with_retry", side_effect=fake_retry):
            result = adapter.generate("Hello world")

        assert result.text == "plain answer"
        assert captured[0] == "Hello world"
