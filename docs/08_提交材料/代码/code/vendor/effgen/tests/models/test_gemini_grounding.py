"""Unit tests for Gemini grounding (GenerationConfig.grounding)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from effgen.models.base import GenerationConfig

# ---------------------------------------------------------------------------
# Helper: build a minimal mock response
# ---------------------------------------------------------------------------

def _mock_response(text: str = "test answer", grounding_chunks: list | None = None):
    candidate = MagicMock()
    part = MagicMock()
    part.thought = False
    part.text = text
    part.function_call = None
    candidate.content.parts = [part]
    candidate.finish_reason = "STOP"

    # Grounding metadata
    if grounding_chunks:
        gmd = MagicMock()
        chunks = []
        for url, title in grounding_chunks:
            web = MagicMock()
            web.uri = url
            web.title = title
            chunk = MagicMock()
            chunk.web = web
            chunks.append(chunk)
        gmd.grounding_chunks = chunks
        candidate.grounding_metadata = gmd
    else:
        candidate.grounding_metadata = None

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata.prompt_token_count = 10
    response.usage_metadata.candidates_token_count = 20
    response.usage_metadata.total_token_count = 30
    response.safety_ratings = None
    return response


# ---------------------------------------------------------------------------
# GenerationConfig tests
# ---------------------------------------------------------------------------

class TestGenerationConfigGrounding:
    def test_default_grounding_is_false(self):
        cfg = GenerationConfig()
        assert cfg.grounding is False

    def test_grounding_can_be_set_true(self):
        cfg = GenerationConfig(grounding=True)
        assert cfg.grounding is True

    def test_grounding_does_not_affect_other_fields(self):
        cfg = GenerationConfig(grounding=True, temperature=0.5)
        assert cfg.temperature == 0.5
        assert cfg.thinking_budget is None


# ---------------------------------------------------------------------------
# Adapter _build_config tests
# ---------------------------------------------------------------------------

class TestBuildConfigGrounding:
    """_build_config injects GoogleSearch tool when grounding=True and model supports it."""

    def _make_adapter(self, model_name: str = "gemini-3.1-flash-lite-preview"):
        with patch("google.genai.Client"):
            from effgen.models.gemini_adapter import GeminiAdapter
            adapter = GeminiAdapter(model_name=model_name, api_key="fake")
            adapter._is_loaded = True
            adapter.client = MagicMock()
            return adapter

    def test_grounding_false_no_tools_injected(self):
        with patch("google.genai.Client"):
            adapter = self._make_adapter()
        cfg = GenerationConfig(grounding=False)
        with patch("google.genai.types") as mock_types:
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.ThinkingConfig = MagicMock()
            mock_types.GoogleSearch = MagicMock()
            mock_types.Tool = MagicMock()
            adapter._build_config(cfg)
            # Tool should NOT have been constructed
            mock_types.Tool.assert_not_called()

    def test_grounding_true_supported_model_injects_tool(self):
        with patch("google.genai.Client"):
            # Use gemini-2.5-flash which has supports_grounding=True
            adapter = self._make_adapter("gemini-2.5-flash")
        cfg = GenerationConfig(grounding=True)
        with patch("google.genai.types") as mock_types:
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.ThinkingConfig = MagicMock()
            mock_types.GoogleSearch = MagicMock(return_value=MagicMock())
            mock_types.Tool = MagicMock(return_value=MagicMock())
            adapter._build_config(cfg)
            mock_types.Tool.assert_called_once()

    def test_grounding_true_unsupported_model_no_tool(self):
        with patch("google.genai.Client"):
            # gemma-4-26b-a4b-it and gemini-3.1-flash-lite-preview do NOT support grounding
            adapter = self._make_adapter("gemma-4-26b-a4b-it")
        cfg = GenerationConfig(grounding=True)
        with patch("google.genai.types") as mock_types:
            mock_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
            mock_types.ThinkingConfig = MagicMock()
            mock_types.GoogleSearch = MagicMock()
            mock_types.Tool = MagicMock()
            adapter._build_config(cfg)
            mock_types.Tool.assert_not_called()


# ---------------------------------------------------------------------------
# Grounding metadata extraction tests
# ---------------------------------------------------------------------------

class TestGroundingChunksExtraction:
    """generate() surfaces grounding_chunks in metadata."""

    def _make_loaded_adapter(self):
        with patch("google.genai.Client"):
            from effgen.models.base import TokenCount
            from effgen.models.gemini_adapter import GeminiAdapter
            adapter = GeminiAdapter(model_name="gemini-2.5-flash", api_key="fake")
            adapter._is_loaded = True
            adapter.client = MagicMock()
            # Make count_tokens return a real TokenCount so validate_prompt works.
            adapter.count_tokens = MagicMock(
                return_value=TokenCount(count=5, model_name="gemini-2.5-flash")
            )
            return adapter

    def test_grounding_chunks_populated(self):
        adapter = self._make_loaded_adapter()
        response = _mock_response(
            text="Some news...",
            grounding_chunks=[
                ("https://example.com/story1", "Story One"),
                ("https://example.com/story2", "Story Two"),
            ],
        )
        with patch.object(adapter, "_generate_with_retry", return_value=response):
            result = adapter.generate("What happened today?")
        chunks = result.metadata.get("grounding_chunks", [])
        assert len(chunks) == 2
        urls = [c["url"] for c in chunks]
        assert "https://example.com/story1" in urls
        assert "https://example.com/story2" in urls

    def test_no_grounding_chunks_when_not_grounded(self):
        adapter = self._make_loaded_adapter()
        response = _mock_response(text="Plain answer", grounding_chunks=None)
        with patch.object(adapter, "_generate_with_retry", return_value=response):
            result = adapter.generate("Tell me something.")
        chunks = result.metadata.get("grounding_chunks", [])
        assert chunks == []

    def test_grounding_metadata_without_content_parts_does_not_crash(self):
        adapter = self._make_loaded_adapter()
        web = SimpleNamespace(uri="https://example.com/story", title="Story")
        chunk = SimpleNamespace(web=web)
        gmd = SimpleNamespace(grounding_chunks=[chunk])
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=None),
            finish_reason="STOP",
            grounding_metadata=gmd,
        )
        response = SimpleNamespace(
            candidates=[candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=0,
                total_token_count=10,
            ),
            safety_ratings=None,
            text=None,
        )

        with patch.object(adapter, "_generate_with_retry", return_value=response):
            result = adapter.generate("What happened today?")

        assert result.text == ""
        assert result.metadata["grounding_chunks"] == [
            {"url": "https://example.com/story", "title": "Story"}
        ]
