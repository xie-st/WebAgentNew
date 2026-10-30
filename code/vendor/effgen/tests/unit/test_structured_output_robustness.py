"""Structured-output robustness & speed (native-first, attempt surfacing, clear failure).

These tests exercise the *orchestration* of structured generation with lightweight
stub models. They do not mock any live provider HTTP — they verify that
``structured_generate`` prefers native JSON modes, counts attempts, time-boxes
repairs, and fails explicitly when no schema-valid object can be produced.
"""
from __future__ import annotations

import pytest

from effgen.core.structured_output import (
    StructuredOutcome,
    StructuredOutputConfig,
    _openai_strict_response_format,
    _supports_native_json,
    constrain_output,
    extract_json_from_text,
    structured_generate,
    validate_json_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
VALID = '{"city": "Tokyo"}'


class _Result:
    def __init__(self, text: str) -> None:
        self.text = text


class GroqAdapter:
    """Stub named like a real OpenAI-compatible adapter (drives the native path)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[dict] = []

    def generate(self, prompt, config=None, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        return _Result(self._texts.pop(0) if self._texts else "still not json")


class _PlainModel:
    """Stub with no native JSON mode (forces the reprompt fallback)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def generate(self, prompt, config=None, **kwargs):  # noqa: ANN001
        self.calls += 1
        return _Result(self._texts.pop(0) if self._texts else "no json here")


class TransformersEngine:
    """Stub named like the real local engine (drives the transformers branch:
    grammar-decoding eligibility + the install-outlines remediation hint)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    def generate(self, prompt, config=None, **kwargs):  # noqa: ANN001
        return _Result(self._texts.pop(0) if self._texts else "no json here")


# --------------------------------------------------------------------------- #
# Native-first behaviour & attempt accounting
# --------------------------------------------------------------------------- #
def test_native_mode_succeeds_in_one_call():
    model = GroqAdapter([VALID])
    outcome = structured_generate(model, "capital of Japan?", SCHEMA)
    assert isinstance(outcome, StructuredOutcome)
    assert outcome.success is True
    assert outcome.attempts == 1
    assert outcome.method == "native"
    assert outcome.parsed == {"city": "Tokyo"}
    # The native call must request JSON mode.
    assert model.calls and model.calls[0].get("response_format") == {"type": "json_object"}


def test_native_failure_falls_back_to_reprompt_and_counts_attempts():
    # First (native) call returns junk; the reprompt then returns valid JSON.
    model = GroqAdapter(["I cannot do that", VALID])
    outcome = structured_generate(model, "capital of Japan?", SCHEMA)
    assert outcome.success is True
    assert outcome.method == "reprompt"
    assert outcome.attempts == 2  # native (1) + one reprompt (2)


def test_fast_path_value_already_valid_via_extract():
    # A plain model that immediately returns valid JSON on the first reprompt.
    model = _PlainModel([VALID])
    outcome = structured_generate(model, "capital?", SCHEMA)
    assert outcome.success is True
    assert outcome.method == "reprompt"
    assert outcome.attempts == 1


# --------------------------------------------------------------------------- #
# Reported failure
# --------------------------------------------------------------------------- #
def test_failure_preserves_raw_and_reports_error():
    model = _PlainModel(["definitely not json", "still prose", "nope"])
    cfg = StructuredOutputConfig(schema=SCHEMA, max_retries=2)
    outcome = structured_generate(model, "capital?", SCHEMA, cfg)
    assert outcome.success is False
    assert outcome.attempts == 2  # bounded by max_retries
    assert outcome.error  # a human-readable reason is present
    assert outcome.raw_text is not None  # raw model text preserved for callers
    assert outcome.json_str is None


def test_constrain_output_wrapper_raises_on_failure():
    model = _PlainModel(["nope", "nope", "nope"])
    with pytest.raises(ValueError, match="structured output"):
        constrain_output(model, "capital?", SCHEMA, StructuredOutputConfig(schema=SCHEMA, max_retries=1))


def test_constrain_output_wrapper_returns_tuple_on_success():
    model = GroqAdapter([VALID])
    json_str, parsed = constrain_output(model, "capital?", SCHEMA)
    assert parsed == {"city": "Tokyo"}
    assert "Tokyo" in json_str


# --------------------------------------------------------------------------- #
# Time-boxing
# --------------------------------------------------------------------------- #
def test_time_budget_zero_stops_immediately():
    model = _PlainModel(["nope", "nope", "nope"])
    cfg = StructuredOutputConfig(schema=SCHEMA, max_retries=5, time_budget_s=0.0)
    outcome = structured_generate(model, "capital?", SCHEMA, cfg)
    assert outcome.success is False
    assert outcome.attempts == 0  # the deadline tripped before any reprompt call
    assert "time budget" in (outcome.error or "")


# --------------------------------------------------------------------------- #
# local grammar path: install-outlines remediation hint + token budget
# --------------------------------------------------------------------------- #
def test_local_failure_hints_install_outlines_when_absent(monkeypatch):
    import effgen.core.structured_output as so

    # Simulate outlines NOT installed: the grammar path is skipped and a local
    # model that won't follow the schema by prompting should be told the one-line
    # fix instead of hitting a dead end.
    monkeypatch.setattr(so, "_outlines_available", lambda: False)
    model = TransformersEngine(["definitely not json", "still prose"])
    cfg = StructuredOutputConfig(schema=SCHEMA, max_retries=2)
    outcome = structured_generate(model, "capital?", SCHEMA, cfg)
    assert outcome.success is False
    assert "effgen[grammar]" in (outcome.error or "")
    assert "instruction-following model" in (outcome.error or "")


def test_no_outlines_hint_when_grammar_is_available(monkeypatch):
    import effgen.core.structured_output as so

    # When outlines IS available, a fallback failure must NOT tell the user to
    # install it (it already is) — the hint would be misleading.
    monkeypatch.setattr(so, "_outlines_available", lambda: True)
    monkeypatch.setattr(so, "_try_grammar_constrained", lambda *a, **k: None)
    model = TransformersEngine(["nope", "nope", "nope"])
    cfg = StructuredOutputConfig(schema=SCHEMA, max_retries=2)
    outcome = structured_generate(model, "capital?", SCHEMA, cfg)
    assert outcome.success is False
    assert "effgen[grammar]" not in (outcome.error or "")


def test_cloud_failure_has_no_local_grammar_hint(monkeypatch):
    import effgen.core.structured_output as so

    # A non-local (API) model that fails must not get the local-only outlines hint.
    monkeypatch.setattr(so, "_outlines_available", lambda: False)
    model = _PlainModel(["nope", "nope", "nope"])
    cfg = StructuredOutputConfig(schema=SCHEMA, max_retries=2)
    outcome = structured_generate(model, "capital?", SCHEMA, cfg)
    assert outcome.success is False
    assert "effgen[grammar]" not in (outcome.error or "")


def test_grammar_max_new_tokens_honours_caller_budget():
    from effgen.core.structured_output import _grammar_max_new_tokens

    assert _grammar_max_new_tokens({}) == 512  # sensible default
    assert _grammar_max_new_tokens({"max_new_tokens": 1024}) == 1024
    assert _grammar_max_new_tokens({"max_tokens": 256}) == 256
    assert _grammar_max_new_tokens({"max_tokens": 0}) == 512  # ignores nonsense


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_supports_native_json_classification():
    assert _supports_native_json(GroqAdapter([])) is True
    assert _supports_native_json(_PlainModel([])) is False


def test_openai_strict_response_format_envelope():
    rf = _openai_strict_response_format(SCHEMA)
    assert rf["type"] == "json_schema"
    js = rf["json_schema"]
    assert js["strict"] is True
    assert js["schema"]["additionalProperties"] is False
    assert js["schema"]["required"] == ["city"]


def test_extract_and_validate_roundtrip():
    txt = "Here you go:\n```json\n{\"city\": \"Tokyo\"}\n```"
    js = extract_json_from_text(txt)
    import json as _json
    valid, err = validate_json_schema(_json.loads(js), SCHEMA)
    assert valid is True and err is None


@pytest.mark.parametrize(
    "text, expected",
    [
        # A ``` inside a JSON string value must not truncate fence extraction.
        ("```json\n{\"```\": null}\n```", {"```": None}),
        ("{\"note\": \"see ```code``` block\"}", {"note": "see ```code``` block"}),
        # The fence is a stronger "JSON is here" signal than a stray prose brace.
        ("intro { not json } ```json\n{\"real\": 1}\n```", {"real": 1}),
        # Plain fenced / prose / array still work.
        ("```json\n{\"a\": 1, \"b\": [1, 2]}\n```", {"a": 1, "b": [1, 2]}),
        ("Here is {\"x\": 5} thanks", {"x": 5}),
        ("[{\"id\": 1}, {\"id\": 2}]", [{"id": 1}, {"id": 2}]),
    ],
)
def test_extract_json_handles_backticks_and_fences(text, expected):
    import json as _json

    out = extract_json_from_text(text)
    assert out is not None
    assert _json.loads(out) == expected


# --------------------------------------------------------------------------- #
# Agent-level metadata mapping (attempt surfacing + clear failure)
# --------------------------------------------------------------------------- #
def _agent_with(model):
    from effgen.core.agent import Agent, AgentConfig
    # Build a model-less agent then inject the stub directly — _apply_structured_output
    # only touches ``self.model``, and the stubs intentionally aren't full BaseModels.
    agent = Agent(config=AgentConfig(name="p14-test", model=None, tools=[], require_model=False))
    agent.model = model
    return agent


def _response(output: str):
    from effgen.core.agent import AgentResponse
    return AgentResponse(output=output, success=True, metadata={})


def test_agent_surfaces_attempts_on_success():
    agent = _agent_with(GroqAdapter([VALID]))
    resp = agent._apply_structured_output(_response("The capital is Tokyo."), SCHEMA, None, "capital?")
    assert resp.success is True
    assert resp.metadata["structured_output"] is True
    assert resp.metadata["structured_output_attempts"] == 1
    assert resp.metadata["structured_output_method"] == "native"
    # A single native-JSON-mode call is not a reprompt: attempts > 0 alone
    # would mislabel this as a repair loop that never ran.
    assert resp.metadata["structured_output_reprompted"] is False


def test_agent_reprompted_flag_true_only_for_the_reprompt_fallback():
    # No native JSON support (_PlainModel) forces the reprompt fallback; the
    # first reprompt attempt succeeds, so method == "reprompt" and the flag
    # accurately reports that a repair loop ran.
    agent = _agent_with(_PlainModel([VALID]))
    resp = agent._apply_structured_output(_response("some prose"), SCHEMA, None, "capital?")
    assert resp.success is True
    assert resp.metadata["structured_output_method"] == "reprompt"
    assert resp.metadata["structured_output_attempts"] == 1
    assert resp.metadata["structured_output_reprompted"] is True


def test_agent_fast_path_zero_attempts_when_answer_already_valid():
    agent = _agent_with(GroqAdapter([VALID]))
    resp = agent._apply_structured_output(_response(VALID), SCHEMA, None, "capital?")
    assert resp.success is True
    assert resp.metadata["structured_output_attempts"] == 0
    assert resp.metadata["structured_output_method"] == "agent_output"


def test_agent_failure_marks_unsuccessful_and_preserves_raw():
    raw = "I think the capital is some city but I won't say."
    agent = _agent_with(_PlainModel(["still prose", "more prose", "nope"]))
    resp = agent._apply_structured_output(_response(raw), SCHEMA, None, "capital?")
    assert resp.success is False
    assert resp.metadata["structured_output"] is False
    assert resp.metadata["structured_output_attempts"] == 2
    assert resp.metadata["structured_output_error"]
    assert resp.metadata["raw_output"] == raw
    assert resp.metadata["reason"] == "structured_output_failed"
    assert resp.output == raw  # raw answer preserved, not blanked
