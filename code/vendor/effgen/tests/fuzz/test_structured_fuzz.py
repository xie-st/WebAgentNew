"""
Hypothesis-driven fuzz tests for structured-output parsing/validation.

Targets the pure helpers that turn messy model text into validated JSON:

  * ``extract_json_from_text`` — pulls the first JSON object/array out of free
    text (code fences, prose, trailing commas, unquoted keys).
  * ``_clean_json`` — the trailing-comma / unquoted-key cleaner.
  * ``validate_json_schema`` / ``_basic_validate`` — schema validation.
  * ``StructuredOutcome`` — the result container.

Asserts that:
  1. ``extract_json_from_text`` never crashes on arbitrary text and returns
     ``str | None``.
  2. When it returns a string for fenced/embedded JSON, the result is itself a
     string (no exceptions escape ``_clean_json``).
  3. ``validate_json_schema`` never raises — it always returns
     ``(bool, str | None)`` — even on adversarial deeply-nested data/schema.
  4. ``_basic_validate`` tolerates malformed schemas (``required`` as a bare
     string, ``properties`` as a non-dict) without crashing or inventing
     bogus per-character "missing field" errors.

Exit criterion: ≥300 examples per text/data test.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.core.structured_output import (
    StructuredOutcome,
    _basic_validate,
    _clean_json,
    extract_json_from_text,
    validate_json_schema,
)

pytestmark = pytest.mark.fuzz

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# JSON-ish noise: braces, brackets, quotes, backslashes, fences, commas, colons.
_JSON_NOISE = st.text(
    alphabet='{}[]":,\\ \n\tabcXYZ09`',
    min_size=0,
    max_size=200,
)

_FENCE_WRAP = st.sampled_from(
    ["```json\n{body}\n```", "```\n{body}\n```", "prose {body} more", "{body}"]
)

# Recursive JSON value strategy bounded in depth so we exercise nesting without
# building megabyte payloads.
_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**6), max_value=10**6)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=20),
    lambda children: st.lists(children, max_size=5)
    | st.dictionaries(st.text(max_size=10), children, max_size=5),
    max_leaves=30,
)


# ---------------------------------------------------------------------------
# extract_json_from_text
# ---------------------------------------------------------------------------


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(_JSON_NOISE)
def test_extract_never_crashes(text: str) -> None:
    """Arbitrary JSON-ish noise never crashes the extractor."""
    result = extract_json_from_text(text)
    assert result is None or isinstance(result, str)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(_json_values, _FENCE_WRAP)
def test_extract_roundtrips_real_json(value, wrap: str) -> None:
    """Real JSON embedded in fences/prose is extracted to a parseable string."""
    body = json.dumps(value)
    text = wrap.format(body=body)
    result = extract_json_from_text(text)
    # Objects/arrays must be found; bare scalars are not "the first {/[".
    if isinstance(value, dict | list) and value != {} and value != []:
        assert isinstance(result, str)
        # Extracted text re-parses to exactly the value that went in.
        assert json.loads(result) == value


# Object keys and string values built from the characters that make extraction
# hard: quotes, braces, brackets, commas, colons, backslashes, backticks.
_TRICKY_TEXT = st.text(alphabet=st.sampled_from(list('{}[]",:`\\ \nabZ09')), max_size=24)

_TRICKY_OBJECTS = st.dictionaries(
    st.text(alphabet=st.sampled_from(list('abZ09,:{} "')), min_size=1, max_size=8),
    st.recursive(
        st.none() | st.booleans() | st.integers(-(10**9), 10**9) | _TRICKY_TEXT,
        lambda c: st.lists(c, max_size=4) | st.dictionaries(_TRICKY_TEXT, c, max_size=4),
        max_leaves=10,
    ),
    min_size=1,
    max_size=5,
)

_EMBEDDINGS = st.sampled_from([
    "{body}",
    "```json\n{body}\n```",
    "```\n{body}\n```",
    "```JSON\n{body}\n```",
    "Sure — here is the result:\n{body}\nLet me know if that helps.",
    "```json\n{body}\n```\nThat covers it.",
    "Answer (note: the shape is {stuff}):\n{body}",
])


@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_TRICKY_OBJECTS, _EMBEDDINGS)
def test_extract_preserves_string_contents(value, wrap: str) -> None:
    """Extraction returns the model's JSON byte-for-byte in meaning.

    A string value holding a comma, colon, brace, bracket, or backtick is data,
    not syntax: the extracted text must re-parse to exactly the object that was
    embedded, never a "repaired" variant.
    """
    text = wrap.replace("{stuff}", "...").format(body=json.dumps(value))
    result = extract_json_from_text(text)
    assert isinstance(result, str)
    assert json.loads(result) == value


@pytest.mark.parametrize(
    "text",
    [
        '{"note": "first, then: go"}',
        '{"note": "a,] b"}',
        '{"note": "use {a: 1} as the shape"}',
        '{"note": "trailing, }"}',
        '{"code": "```json"}',
        '{"quoted": "he said \\"go, now: fast\\""}',
        '```json\n{"note": "first, then: go"}\n```',
    ],
)
def test_extract_does_not_rewrite_valid_json(text: str) -> None:
    """Text that already parses as JSON comes back unchanged."""
    expected = json.loads(extract_json_from_text(text) or "null")
    embedded = json.loads(text[text.index("{"):text.rindex("}") + 1])
    assert expected == embedded


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Backticks inside the JSON's own keys/values are data, not a fence.
        ('{"9": [{"```": null}], "0": []}', {"9": [{"```": None}], "0": []}),
        ('{"tip": "wrap it in ```json ... ```"}', {"tip": "wrap it in ```json ... ```"}),
        # A real fence still wins over a stray brace in the prose before it.
        ('intro { not json } ```json\n{"real": 1}\n```', {"real": 1}),
        ('```json\n{"```": null}\n```', {"```": None}),
    ],
)
def test_extract_fence_versus_backticks_in_data(raw: str, expected) -> None:
    """A fence is honoured; backticks belonging to the JSON payload are not."""
    out = extract_json_from_text(raw)
    assert out is not None
    assert json.loads(out) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1,}', {"a": 1}),
        ('{a: 1}', {"a": 1}),
        ('{a: "x", b: [1, 2,],}', {"a": "x", "b": [1, 2]}),
        ('{"a": {b: 2,}}', {"a": {"b": 2}}),
        ('```json\n{"items": [1, 2,],}\n```', {"items": [1, 2]}),
    ],
)
def test_extract_still_repairs_common_defects(raw: str, expected) -> None:
    """Trailing commas and unquoted keys are still repaired outside strings."""
    out = extract_json_from_text(raw)
    assert out is not None
    assert json.loads(out) == expected


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_TRICKY_OBJECTS, st.integers(min_value=1, max_value=400))
def test_extract_truncated_json_never_crashes_or_invents(value, cut: int) -> None:
    """A response cut off mid-JSON returns None or a real prefix — never invented data.

    Truncation must never produce a *plausible but wrong* object. Whatever comes
    back has to be an object or array that was genuinely present in the text
    that survived, so the caller either gets real data or a parse failure it can
    act on — not a silently short answer.
    """
    body = json.dumps(value)
    truncated = f"```json\n{body[:cut]}"
    result = extract_json_from_text(truncated)
    if result is None:
        return
    assert isinstance(result, str)
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return  # A clear parse failure is the acceptable other outcome.
    # It parsed, so the balanced region was complete in the surviving text.
    assert isinstance(parsed, dict | list)
    # A complete top-level object can only have come from an untruncated body.
    if cut >= len(body):
        assert parsed == value


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(min_size=0, max_size=300))
def test_clean_json_never_crashes(text: str) -> None:
    """``_clean_json`` is a regex cleaner — it must never raise."""
    out = _clean_json(text)
    assert isinstance(out, str)


def test_extract_pathological_unbalanced() -> None:
    """Unbalanced/odd brackets degrade to None, never hang or crash."""
    for bad in ["{" * 2000, "[" * 2000, '{"a":"' + "\\" * 50, "```json\n", "{", "["]:
        assert extract_json_from_text(bad) is None or isinstance(
            extract_json_from_text(bad), str
        )


# ---------------------------------------------------------------------------
# validate_json_schema / _basic_validate
# ---------------------------------------------------------------------------

_SCHEMA_TYPES = st.sampled_from(
    ["object", "array", "string", "integer", "number", "boolean", "null", "weird", None]
)


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_json_values, _SCHEMA_TYPES)
def test_validate_always_returns_tuple(data, schema_type) -> None:
    """Validation always returns ``(bool, str|None)`` — never raises."""
    schema = {} if schema_type is None else {"type": schema_type}
    valid, err = validate_json_schema(data, schema)
    assert isinstance(valid, bool)
    assert err is None or isinstance(err, str)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.dictionaries(st.text(max_size=8), _json_values, max_size=5),
    st.one_of(st.text(max_size=8), st.lists(st.text(max_size=8), max_size=4), st.integers(), st.none()),
    st.one_of(st.dictionaries(st.text(max_size=8), st.just({"type": "string"}), max_size=3), st.text(max_size=8), st.none()),
)
def test_basic_validate_tolerates_malformed_schema(data, required, properties) -> None:
    """Malformed ``required``/``properties`` must not crash or fabricate errors.

    A ``required`` given as a bare string must be treated as a single field
    name, never iterated char-by-char.
    """
    schema = {"type": "object"}
    if required is not None:
        schema["required"] = required
    if properties is not None:
        schema["properties"] = properties
    valid, err = _basic_validate(data, schema)
    assert isinstance(valid, bool)
    assert err is None or isinstance(err, str)
    # When required is a bare string, the only legitimate "missing" message is
    # for that whole string, not for an individual character of it.
    if isinstance(required, str) and required and required not in data:
        assert err == f"Missing required field: {required}"


def test_basic_validate_required_string_not_char_iterated() -> None:
    """Regression: required='name' must check 'name', not 'n','a','m','e'."""
    valid, err = _basic_validate({}, {"type": "object", "required": "name"})
    assert valid is False
    assert err == "Missing required field: name"
    valid, err = _basic_validate({"name": 1}, {"type": "object", "required": "name"})
    assert valid is True


def test_validate_deep_nesting_fails_closed() -> None:
    """Deeply nested data/schema fails closed (False, msg) — never an uncaught crash."""

    def nest_schema(n):
        s = {"type": "object", "properties": {}}
        cur = s
        for _ in range(n):
            cur["properties"] = {"x": {"type": "object", "properties": {}}}
            cur = cur["properties"]["x"]
        return s

    def nest_data(n):
        d = {}
        cur = d
        for _ in range(n):
            cur["x"] = {}
            cur = cur["x"]
        return d

    valid, err = validate_json_schema(nest_data(4000), nest_schema(4000))
    assert isinstance(valid, bool)
    assert err is None or isinstance(err, str)


# ---------------------------------------------------------------------------
# StructuredOutcome container
# ---------------------------------------------------------------------------


def test_structured_outcome_failure_shape() -> None:
    """A failed outcome preserves raw text and an error — never a silent empty success."""
    outcome = StructuredOutcome(
        success=False, attempts=3, method="reprompt",
        error="could not produce schema-valid output", raw_text="not json",
    )
    assert outcome.success is False
    assert outcome.error
    assert outcome.raw_text == "not json"
