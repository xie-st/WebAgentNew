"""Unit tests for GeminiAdapter helpers (no live API calls)."""

from __future__ import annotations

import time

import pytest

from effgen.models import gemini_models
from effgen.models.gemini_adapter import GeminiAdapter

# ---------------------------------------------------------------------------
# Schema sanitization
# ---------------------------------------------------------------------------

def test_sanitize_schema_strips_unsupported_fields():
    raw = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "expr",
                "minLength": 1,
                "maxLength": 1024,
            },
            "precision": {
                "type": "integer",
                "minimum": 0,
                "maximum": 15,
                "default": 2,
            },
        },
        "required": ["expression"],
        "additionalProperties": False,  # also unsupported by Gemini
    }
    cleaned = GeminiAdapter._sanitize_schema(raw)
    assert cleaned == {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "expr"},
            "precision": {"type": "integer"},
        },
        "required": ["expression"],
    }


def test_sanitize_schema_defaults_missing_array_items():
    """Gemini rejects an ``array`` property without ``items``; the sanitizer
    defaults the element schema so tools with a plain ``type: array`` param are
    accepted (this is what the general preset's maps/email tools declare)."""
    raw = {
        "type": "object",
        "properties": {
            "markers": {"type": "array", "description": "map markers"},
            "attachments": {"type": "array"},
            "typed": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["markers"],
    }
    cleaned = GeminiAdapter._sanitize_schema(raw)
    assert cleaned["properties"]["markers"]["items"] == {"type": "string"}
    assert cleaned["properties"]["attachments"]["items"] == {"type": "string"}
    # An explicit items schema is preserved, not overwritten.
    assert cleaned["properties"]["typed"]["items"] == {"type": "integer"}


def test_sanitize_schema_top_level_array_gets_items():
    cleaned = GeminiAdapter._sanitize_schema({"type": "array"})
    assert cleaned == {"type": "array", "items": {"type": "string"}}


def test_sanitize_schema_preserves_property_names():
    """`properties` keys are user-defined names, not schema fields — keep them."""
    raw = {
        "type": "object",
        "properties": {
            "minLength": {"type": "string"},  # property literally named "minLength"
            "type": {"type": "string"},
        },
        "required": ["minLength"],
    }
    cleaned = GeminiAdapter._sanitize_schema(raw)
    assert "minLength" in cleaned["properties"]
    assert "type" in cleaned["properties"]
    assert cleaned["required"] == ["minLength"]


def test_build_config_sanitizes_native_response_schema():
    """A native JSON-mode schema with JSON-Schema extras (e.g.
    ``additionalProperties``) must be stripped before reaching the SDK — Gemini
    rejects those fields at request time, which previously forced a slow reprompt
    fallback instead of the one-shot native path."""
    pytest.importorskip("google.genai")
    from effgen.models.base import GenerationConfig

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.safety_settings = None
    adapter.model_name = "gemini-3.1-flash-lite"

    cfg = GenerationConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
    gc = adapter._build_config(cfg)
    assert gc.response_mime_type == "application/json"
    assert "additionalProperties" not in gc.response_schema
    assert gc.response_schema["properties"] == {"city": {"type": "string"}}


def test_build_config_forwards_seed_and_penalties():
    """A pinned seed/penalty must reach the Gemini request config, matching top_k."""
    pytest.importorskip("google.genai")
    from effgen.models.base import GenerationConfig

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.safety_settings = None
    adapter.model_name = "gemini-3.1-flash-lite"

    cfg = GenerationConfig(seed=42, presence_penalty=0.5, frequency_penalty=1.5)
    gc = adapter._build_config(cfg)
    assert gc.seed == 42
    assert gc.presence_penalty == 0.5
    assert gc.frequency_penalty == 1.5


def test_build_config_omits_default_seed_and_penalties():
    """The neutral defaults (seed=None, penalty=0.0) are not sent."""
    pytest.importorskip("google.genai")
    from effgen.models.base import GenerationConfig

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.safety_settings = None
    adapter.model_name = "gemini-3.1-flash-lite"

    gc = adapter._build_config(GenerationConfig())
    assert gc.seed is None
    assert gc.presence_penalty is None
    assert gc.frequency_penalty is None


def test_build_config_native_schema_with_no_real_properties_keeps_json_mode_only():
    """If sanitization leaves no usable properties, keep JSON mode but drop the
    schema rather than send Gemini an empty/invalid one."""
    pytest.importorskip("google.genai")
    from effgen.models.base import GenerationConfig

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter.safety_settings = None
    adapter.model_name = "gemini-3.1-flash-lite"

    cfg = GenerationConfig(
        response_mime_type="application/json",
        response_schema={"type": "object", "additionalProperties": False},
    )
    gc = adapter._build_config(cfg)
    assert gc.response_mime_type == "application/json"
    assert gc.response_schema is None


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

def test_convert_tools_to_genai_openai_format():
    pytest.importorskip("google.genai")
    tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "math",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "minLength": 1}},
                "required": ["expression"],
            },
        },
    }]
    converted = GeminiAdapter._convert_tools_to_genai(tools)
    assert len(converted) == 1
    fds = converted[0].function_declarations
    assert len(fds) == 1
    assert fds[0].name == "calculator"


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def test_suggested_retry_delay_parses_message_tail():
    class FakeExc(Exception):
        pass

    exc = FakeExc("429 quota exceeded. Please retry in 21.5s")
    assert GeminiAdapter._suggested_retry_delay(exc) == pytest.approx(21.5)


def test_suggested_retry_delay_returns_none_when_absent():
    assert GeminiAdapter._suggested_retry_delay(RuntimeError("boom")) is None


def test_generate_with_retry_honors_retry_delay(monkeypatch):
    """The retry helper should sleep the suggested delay and then succeed."""
    pytest.importorskip("google.api_core.exceptions")
    from google.api_core import exceptions as gax_exc

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    # Two retries after the first request: three attempts in all.
    adapter.max_retries = 2
    adapter.model_name = "gemini-3.1-flash-lite"

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                exc = gax_exc.ResourceExhausted("Please retry in 7s")
                raise exc
            return "ok"

    class FakeClient:
        models = FakeModels()

    adapter.client = FakeClient()
    out = adapter._generate_with_retry(contents="hi", gen_config=None)
    assert out == "ok"
    assert calls["n"] == 2
    assert sleeps and sleeps[0] >= 7.0  # honored suggested delay


def test_generate_with_retry_gives_up_after_max(monkeypatch):
    pytest.importorskip("google.api_core.exceptions")
    from google.api_core import exceptions as gax_exc

    adapter = GeminiAdapter.__new__(GeminiAdapter)
    # One retry after the first request: two attempts in all.
    adapter.max_retries = 1
    adapter.model_name = "gemini-3.1-flash-lite"
    monkeypatch.setattr(time, "sleep", lambda s: None)

    class FakeModels:
        def generate_content(self, **kwargs):
            raise gax_exc.ResourceExhausted("Please retry in 1s")

    class FakeClient:
        models = FakeModels()

    adapter.client = FakeClient()
    with pytest.raises(gax_exc.ResourceExhausted):
        adapter._generate_with_retry(contents="hi", gen_config=None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_3_1_flash_lite():
    info = gemini_models.model_info("gemini-3.1-flash-lite")
    assert info["canonical_id"] == "gemini-3.1-flash-lite"
    assert info["free_tier"] is True
    assert info["rpm"] == 15
    assert info["supports_native_tools"] is True


def test_registry_free_tier_excludes_pro_preview():
    free = set(gemini_models.free_tier_models())
    assert "gemini-3.1-flash-lite" in free
    # A paid id the registry still lists, so the exclusion is a real one and
    # not vacuously true against an id that was removed from the catalog.
    assert "gemini-3.1-pro-preview" in gemini_models.GEMINI_MODELS
    assert "gemini-3.1-pro-preview" not in free
    assert "gemma-4-31b-it" in free


def test_registry_recommended_models_filters_by_tier():
    free = gemini_models.recommended_models(tier="free")
    assert all(m["tier"] == "free" for m in free)
    assert any(m["id"] == "gemini-3.1-flash-lite" for m in free)


def test_registry_unknown_model_raises():
    with pytest.raises(KeyError):
        gemini_models.model_info("gemini-does-not-exist")


# ---------------------------------------------------------------------------
# generate_stream — cost/token accounting: generate_stream() must read
# usage_metadata from the terminal chunk and fold it into the running totals
# the same way generate() does, so streamed turns are costed and counted too.
# ---------------------------------------------------------------------------

class _FakeStreamChunk:
    def __init__(self, text=None, usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class TestGeminiAdapterStream:
    def _loaded_adapter(self, model="gemini-3.1-flash-lite"):
        from unittest.mock import MagicMock, patch

        from effgen.models.base import TokenCount

        with patch("google.genai.Client"):
            adapter = GeminiAdapter(model_name=model, api_key="fake")
        adapter._is_loaded = True
        adapter.client = MagicMock()
        adapter.count_tokens = MagicMock(
            return_value=TokenCount(count=5, model_name=model)
        )
        return adapter

    def test_stream_yields_text_and_accumulates_total_cost(self):
        from unittest.mock import MagicMock, patch

        usage = MagicMock()
        usage.prompt_token_count = 10
        usage.candidates_token_count = 20
        usage.total_token_count = 30
        chunks = [
            _FakeStreamChunk(text="Hel"),
            _FakeStreamChunk(text="lo"),
            # The terminal chunk carries the cumulative usage.
            _FakeStreamChunk(text=None, usage_metadata=usage),
        ]
        adapter = self._loaded_adapter()
        assert getattr(adapter, "total_cost", 0.0) == 0.0
        with patch.object(adapter, "_generate_with_retry", return_value=iter(chunks)):
            out = "".join(adapter.generate_stream("hi"))
        assert out == "Hello"
        assert adapter.total_cost > 0.0
        assert adapter.total_tokens == 30

    def test_stream_with_no_usage_metadata_leaves_total_cost_unset(self):
        from unittest.mock import patch

        chunks = [_FakeStreamChunk(text="Hi")]
        adapter = self._loaded_adapter()
        with patch.object(adapter, "_generate_with_retry", return_value=iter(chunks)):
            list(adapter.generate_stream("hi"))
        assert getattr(adapter, "total_cost", 0.0) == 0.0


class _FakePart:
    """One content part: text, a function call, or both absent."""

    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _FakeFunctionCall:
    def __init__(self, name, args, call_id=""):
        self.name = name
        self.args = args
        self.id = call_id


class _FakeCandidate:
    def __init__(self, parts, finish_reason=None):
        self.content = type("_C", (), {"parts": parts})()
        self.finish_reason = finish_reason


class _FakePartsChunk:
    """A chunk shaped the way the SDK sends one: candidates carrying parts."""

    def __init__(self, parts, usage_metadata=None, finish_reason=None):
        self.candidates = [_FakeCandidate(parts, finish_reason)]
        self.usage_metadata = usage_metadata

    @property
    def text(self):  # the accessor that drops non-text parts
        joined = "".join(p.text for p in self.candidates[0].content.parts if p.text)
        if not joined:
            raise ValueError("no text parts in this chunk")
        return joined


class TestGeminiAdapterStreamToolCalls:
    """A streamed function call must reach the caller, not be dropped."""

    def _adapter(self):
        from unittest.mock import MagicMock, patch

        from effgen.models.base import TokenCount

        with patch("google.genai.Client"):
            adapter = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key="fake")
        adapter._is_loaded = True
        adapter.client = MagicMock()
        adapter.count_tokens = MagicMock(
            return_value=TokenCount(count=5, model_name=adapter.model_name)
        )
        return adapter

    def test_a_streamed_function_call_is_recorded(self):
        from unittest.mock import patch

        from effgen.models.base import get_stream_tool_calls

        chunks = [
            _FakePartsChunk([_FakePart(text="Let me check. ")]),
            _FakePartsChunk([
                _FakePart(function_call=_FakeFunctionCall(
                    "calculator", {"expression": "6*7"}, "call_1"
                ))
            ]),
        ]
        adapter = self._adapter()
        with patch.object(adapter, "_generate_with_retry", return_value=iter(chunks)):
            text = "".join(adapter.generate_stream("What is 6*7?"))

        assert text == "Let me check. "
        calls = get_stream_tool_calls(adapter)
        assert [c["function"]["name"] for c in calls] == ["calculator"]
        assert calls[0]["arguments"] == {"expression": "6*7"}

    def test_the_call_is_readable_while_the_turn_still_streams(self):
        from unittest.mock import patch

        from effgen.models.base import get_stream_tool_calls

        chunks = [
            _FakePartsChunk([
                _FakePart(function_call=_FakeFunctionCall(
                    "calculator", {"expression": "6*7"}
                ))
            ]),
            _FakePartsChunk([_FakePart(text="working")]),
        ]
        adapter = self._adapter()
        seen = []
        with patch.object(adapter, "_generate_with_retry", return_value=iter(chunks)):
            for _text in adapter.generate_stream("What is 6*7?"):
                seen.append(len(get_stream_tool_calls(adapter)))
        assert seen and all(count == 1 for count in seen)

    def test_a_turn_with_no_call_records_none(self):
        from unittest.mock import patch

        from effgen.models.base import get_stream_tool_calls

        chunks = [_FakePartsChunk([_FakePart(text="just an answer")])]
        adapter = self._adapter()
        with patch.object(adapter, "_generate_with_retry", return_value=iter(chunks)):
            assert "".join(adapter.generate_stream("hi")) == "just an answer"
        assert get_stream_tool_calls(adapter) == []
