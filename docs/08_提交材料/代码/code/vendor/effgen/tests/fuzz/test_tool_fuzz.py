"""
Hypothesis-driven fuzz tests for BaseTool subclasses.

For every tool class discoverable without requiring external API keys /
system binaries, we generate random valid-schema inputs and assert:

  1. No *unhandled* exception escapes (every exception is caught or
     returned as ToolResult(success=False)).
  2. If the result has ``success=False``, ``error`` is a non-empty string.
  3. No secret pattern appears in the error message.

Tools that require heavy network/system resources (OCR, audio, image) are
tested with schema-shape inputs only; the actual _execute path is patched
to avoid live calls while still exercising validation + coercion logic.

Exit criterion: ≥500 examples per parametrised test (controlled via
``settings(max_examples=500)``).
"""

from __future__ import annotations

import asyncio
import re
import string
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Secret patterns — same as Redactor built-ins
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"csk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"gsk_[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"Bearer [^\s]{6,}"),
]


def _has_secret(text: str) -> bool:
    return any(pat.search(text) for pat in _SECRET_PATTERNS)


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_SAFE_TEXT = st.text(
    alphabet=string.ascii_letters + string.digits + " +-_./,@()[]{}",
    min_size=0,
    max_size=200,
)

_SAFE_NONEMPTY = st.text(
    alphabet=string.ascii_letters + string.digits + " +-_./",
    min_size=1,
    max_size=200,
)


def _strategy_for_spec(spec: ParameterSpec) -> st.SearchStrategy:
    """Return a Hypothesis strategy matching the declared parameter type."""
    base: st.SearchStrategy
    if spec.type == ParameterType.STRING:
        lo = spec.min_length or 0
        hi = min(spec.max_length or 200, 200)
        if hi < lo:
            hi = lo
        base = st.text(
            alphabet=string.ascii_letters + string.digits + " +-_./,@()[]{}",
            min_size=lo,
            max_size=hi,
        )
        if spec.enum:
            base = st.sampled_from(spec.enum)
    elif spec.type == ParameterType.INTEGER:
        lo_int = int(spec.min_value) if spec.min_value is not None else -10_000
        hi_int = int(spec.max_value) if spec.max_value is not None else 10_000
        if spec.enum:
            base = st.sampled_from([int(v) for v in spec.enum])
        else:
            base = st.integers(min_value=lo_int, max_value=hi_int)
    elif spec.type == ParameterType.FLOAT:
        lo_f = float(spec.min_value) if spec.min_value is not None else -1e6
        hi_f = float(spec.max_value) if spec.max_value is not None else 1e6
        if spec.enum:
            base = st.sampled_from([float(v) for v in spec.enum])
        else:
            base = st.floats(
                min_value=lo_f,
                max_value=hi_f,
                allow_nan=False,
                allow_infinity=False,
            )
    elif spec.type == ParameterType.BOOLEAN:
        base = st.booleans()
    elif spec.type == ParameterType.ARRAY:
        base = st.lists(_SAFE_TEXT, min_size=0, max_size=5)
    elif spec.type == ParameterType.OBJECT:
        base = st.dictionaries(_SAFE_TEXT, _SAFE_TEXT, max_size=5)
    else:
        # ANY — just send a string
        base = _SAFE_TEXT

    if not spec.required:
        # Optional params may be absent
        base = st.one_of(st.none(), base)

    return base


def _kwargs_strategy(params: list[ParameterSpec]) -> st.SearchStrategy:
    """Build a fixed_dictionaries strategy for the given parameter list."""
    if not params:
        return st.just({})
    return st.fixed_dictionaries({p.name: _strategy_for_spec(p) for p in params})


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_result_ok(result: ToolResult, tool_name: str) -> None:
    """Core post-condition: result must be a valid ToolResult with no leaked secrets."""
    assert isinstance(result, ToolResult), (
        f"{tool_name}: execute() must return ToolResult, got {type(result)}"
    )
    if not result.success:
        err = result.error or ""
        assert err, (
            f"{tool_name}: success=False but error is empty/None"
        )
        assert not _has_secret(err), (
            f"{tool_name}: secret pattern found in error message"
        )


# ---------------------------------------------------------------------------
# Lightweight tool registry (no external deps)
# ---------------------------------------------------------------------------

def _make_simple_tool(
    name: str,
    params: list[ParameterSpec],
    execute_return: Any = "ok",
) -> BaseTool:
    """Factory for lightweight in-process tools used for pure schema fuzz."""

    class _FuzzTool(BaseTool):
        async def _execute(self, **kwargs: Any) -> Any:
            return execute_return

    return _FuzzTool(
        metadata=ToolMetadata(
            name=name,
            description=f"Fuzz target: {name}",
            category=ToolCategory.COMPUTATION,
            parameters=params,
        )
    )


# ---------------------------------------------------------------------------
# Tests: built-in lightweight tools
# ---------------------------------------------------------------------------

# --- Calculator ---

from effgen.tools.builtin.calculator import Calculator  # noqa: E402


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    expression=_SAFE_TEXT,
    operation=st.one_of(
        st.none(),
        st.sampled_from(["evaluate", "convert", "statistics"]),
    ),
    precision=st.one_of(st.none(), st.integers(min_value=0, max_value=10)),
)
def test_calculator_fuzz(expression, operation, precision):
    """Calculator tolerates arbitrary fuzzed inputs without unhandled exceptions."""
    tool = Calculator()
    kwargs: dict[str, Any] = {"expression": expression}
    if operation is not None:
        kwargs["operation"] = operation
    if precision is not None:
        kwargs["precision"] = precision

    result = asyncio.run(tool.execute(**kwargs))
    _assert_result_ok(result, "Calculator")


# --- DateTimeTool ---

from effgen.tools.builtin.datetime_tool import DateTimeTool  # noqa: E402


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    operation=st.one_of(
        st.just("current_time"),
        st.sampled_from(["current_time", "format_date", "add_time", "difference", "timezone"]),
        _SAFE_TEXT,
    ),
    timezone=st.one_of(st.none(), _SAFE_TEXT),
    date_string=st.one_of(st.none(), _SAFE_TEXT),
)
def test_datetime_fuzz(operation, timezone, date_string):
    """DateTimeTool handles random operation/timezone/date inputs cleanly."""
    tool = DateTimeTool()
    kwargs: dict[str, Any] = {"operation": operation}
    if timezone is not None:
        kwargs["timezone"] = timezone
    if date_string is not None:
        kwargs["date_string"] = date_string

    result = asyncio.run(tool.execute(**kwargs))
    _assert_result_ok(result, "DateTimeTool")


# --- TextProcessingTool ---

from effgen.tools.builtin.text_processing import TextProcessingTool  # noqa: E402


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    text=_SAFE_TEXT,
    operation=st.one_of(
        st.sampled_from(["uppercase", "lowercase", "word_count", "char_count",
                          "reverse", "strip", "title_case"]),
        _SAFE_TEXT,
    ),
)
def test_text_processing_fuzz(text, operation):
    """TextProcessingTool handles arbitrary text + operation without crashing."""
    tool = TextProcessingTool()
    result = asyncio.run(
        tool.execute(text=text, operation=operation)
    )
    _assert_result_ok(result, "TextProcessingTool")


# --- JSONTool ---

from effgen.tools.builtin.json_tool import JSONTool  # noqa: E402


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    operation=st.one_of(
        st.sampled_from(["parse", "format", "validate", "extract"]),
        _SAFE_TEXT,
    ),
    data=st.one_of(
        st.just("{}"),
        st.just("null"),
        st.just("[1,2,3]"),
        _SAFE_TEXT,
        st.just('{"key": "value"}'),
    ),
    path=st.one_of(st.none(), _SAFE_TEXT),
)
def test_json_tool_fuzz(operation, data, path):
    """JSONTool handles random JSON strings and operations without crashing."""
    tool = JSONTool()
    kwargs: dict[str, Any] = {"operation": operation, "data": data}
    if path is not None:
        kwargs["path"] = path

    result = asyncio.run(tool.execute(**kwargs))
    _assert_result_ok(result, "JSONTool")


# --- WikipediaTool (mocked network) ---

from effgen.tools.builtin.wikipedia_tool import WikipediaTool  # noqa: E402


@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(
    query=_SAFE_NONEMPTY,
    sentences=st.integers(min_value=1, max_value=10),
)
def test_wikipedia_fuzz(query, sentences):
    """WikipediaTool validation layer handles random queries; network is mocked."""
    tool = WikipediaTool()
    # Mock the actual network call; we're testing the validation + error-handling path
    with patch.object(tool, "_execute", new=AsyncMock(return_value="mocked summary")):
        result = asyncio.run(
            tool.execute(query=query, sentences=sentences)
        )
    _assert_result_ok(result, "WikipediaTool")


# --- Generic schema-only tool fuzz ---

@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    name=st.text(alphabet=string.ascii_lowercase + "_", min_size=1, max_size=30),
    params=st.lists(
        st.fixed_dictionaries({
            # "self" is excluded: it is the bound-method receiver of execute(),
            # so splatting a param of that name into tool.execute(**kwargs) would
            # collide with the receiver — a name no real tool parameter can take.
            "pname": st.text(
                alphabet=string.ascii_lowercase + "_", min_size=1, max_size=20
            ).filter(lambda s: s != "self"),
            "ptype": st.sampled_from(list(ParameterType)),
            "required": st.booleans(),
        }),
        min_size=0,
        max_size=6,
        unique_by=lambda x: x["pname"],
    ),
)
def test_generic_tool_schema_fuzz(name, params):
    """ToolMetadata + BaseTool construction/validation never crashes on random schemas."""
    specs = [
        ParameterSpec(
            name=p["pname"],
            type=p["ptype"],
            description="fuzz",
            required=p["required"],
        )
        for p in params
    ]
    tool = _make_simple_tool(name, specs)
    # Build kwargs from the schema
    kwargs: dict[str, Any] = {}
    for spec in specs:
        if spec.required:
            if spec.type == ParameterType.STRING:
                kwargs[spec.name] = "test"
            elif spec.type == ParameterType.INTEGER:
                kwargs[spec.name] = 0
            elif spec.type == ParameterType.FLOAT:
                kwargs[spec.name] = 0.0
            elif spec.type == ParameterType.BOOLEAN:
                kwargs[spec.name] = False
            elif spec.type == ParameterType.ARRAY:
                kwargs[spec.name] = []
            elif spec.type == ParameterType.OBJECT:
                kwargs[spec.name] = {}

    result = asyncio.run(tool.execute(**kwargs))
    _assert_result_ok(result, name)


# --- validate_parameters coercion fuzz ---

@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    int_as_str=st.one_of(
        st.integers(min_value=-1000, max_value=1000).map(str),
        st.just("abc"),
        st.just(""),
        st.just("3.14"),
    ),
    float_as_str=st.one_of(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False).map(str),
        st.just("not_a_float"),
        st.just(""),
    ),
    bool_as_str=st.one_of(
        st.sampled_from(["true", "false", "True", "False", "1", "0", "yes", "no"]),
        _SAFE_TEXT,
    ),
)
def test_parameter_coercion_fuzz(int_as_str, float_as_str, bool_as_str):
    """_coerce_parameters never crashes on arbitrary string inputs."""
    specs = [
        ParameterSpec("n", ParameterType.INTEGER, "int param", required=False),
        ParameterSpec("f", ParameterType.FLOAT, "float param", required=False),
        ParameterSpec("b", ParameterType.BOOLEAN, "bool param", required=False),
    ]
    tool = _make_simple_tool("coerce_test", specs)
    kwargs = {"n": int_as_str, "f": float_as_str, "b": bool_as_str}
    # Should not raise — coercion failures are silent
    coerced = tool._coerce_parameters(kwargs)
    assert isinstance(coerced, dict)


# --- ParameterSpec.validate fuzz ---

@pytest.mark.fuzz
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    ptype=st.sampled_from(list(ParameterType)),
    value=st.one_of(
        st.none(),
        st.integers(-10_000, 10_000),
        st.floats(-1e6, 1e6, allow_nan=False),
        st.booleans(),
        _SAFE_TEXT,
        st.lists(_SAFE_TEXT, max_size=5),
        st.dictionaries(_SAFE_TEXT, _SAFE_TEXT, max_size=3),
    ),
)
def test_param_spec_validate_fuzz(ptype, value):
    """ParameterSpec.validate never raises; always returns (bool, str|None)."""
    spec = ParameterSpec(
        name="x",
        type=ptype,
        description="fuzz",
        required=False,
    )
    is_valid, error = spec.validate(value)
    assert isinstance(is_valid, bool)
    if not is_valid:
        assert error is None or isinstance(error, str)


# ---------------------------------------------------------------------------
# Every BaseTool subclass — registry-driven coverage
# ---------------------------------------------------------------------------
#
# The lightweight tests above hand-pick a handful of tools. To honour the
# "for every BaseTool subclass" contract we discover *all* built-in tools via
# the registry and fuzz each one's public ``execute()`` entry point. Tools
# that hit the network or a system binary have their ``_execute`` patched, so
# we exercise the validation + coercion + error-handling envelope (which is
# where unhandled exceptions and secret leaks would surface) without any live
# calls.

from effgen.tools.registry import ToolRegistry  # noqa: E402


def _discover_all_tools() -> dict[str, type[BaseTool]]:
    """Return every built-in BaseTool subclass keyed by registered name."""
    registry = ToolRegistry()
    registry.discover_builtin_tools()
    return {name: registry._tools[name] for name in registry.list_tools()}


_ALL_TOOL_CLASSES = _discover_all_tools()
_ALL_TOOL_NAMES = sorted(_ALL_TOOL_CLASSES)


def _hostile_value(spec: ParameterSpec) -> st.SearchStrategy:
    """Strategy mixing schema-valid and deliberately hostile inputs.

    Type-confused inputs (wrong type, None, oversized, control chars) probe the
    validation/coercion path. The contract holds regardless: ``execute()`` must
    still return a ``ToolResult`` and never leak a secret in the error string.
    """
    return st.one_of(
        _strategy_for_spec(spec),
        st.none(),
        st.integers(-10_000, 10_000),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.text(alphabet=string.printable, min_size=0, max_size=64),
        st.lists(_SAFE_TEXT, max_size=4),
        st.dictionaries(_SAFE_TEXT, _SAFE_TEXT, max_size=3),
    )


@pytest.mark.fuzz
@pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
@settings(
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_every_tool_execute_fuzz(tool_name, data):
    """Every registered BaseTool subclass survives fuzzed execute() inputs.

    Post-conditions for *all* tools:
      * execute() returns a ToolResult (never raises, never returns None).
      * On failure, error is a non-empty string with no leaked secret.
    """
    tool = _ALL_TOOL_CLASSES[tool_name]()

    kwargs: dict[str, Any] = {}
    for spec in tool.metadata.parameters:
        # Optional params are sometimes omitted entirely.
        if not spec.required and data.draw(st.booleans()):
            continue
        kwargs[spec.name] = data.draw(_hostile_value(spec))

    # Patch the network/system layer so no live calls are made. ``initialize``
    # is also stubbed in case a tool lazily connects to a service there.
    with (
        patch.object(tool, "_execute", new=AsyncMock(return_value="mocked-output")),
        patch.object(tool, "initialize", new=AsyncMock(return_value=None)),
    ):
        result = asyncio.run(tool.execute(**kwargs))

    _assert_result_ok(result, tool_name)


# Synthetic secrets — shaped to match every built-in redactor pattern. None of
# these are real credentials; they are random-looking tokens used purely to
# prove they never survive into a surfaced error message.
_SYNTHETIC_SECRETS = [
    "sk-ant-" + "A1b2C3d4E5f6G7h8I9j0",          # anthropic
    "csk-" + "Z9y8X7w6V5u4T3s2R1q0aA",            # cerebras
    "AIza" + "B" * 35,                            # google
    "hf_" + "h" * 24,                             # huggingface
    "gsk_" + "g" * 24,                            # groq
    "sk-" + "k" * 24,                             # openai
    "Bearer abcdef0123456789",                    # bearer token
    "https://hooks.slack.com/services/T00/B00/XXXXXXXX",  # slack webhook
]


@pytest.mark.fuzz
@pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
@settings(
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_secret_injection_never_leaks(tool_name, data):
    """Inject synthetic secrets into tool inputs; assert no error message echoes them.

    The underlying ``_execute`` is forced to raise an exception that echoes its
    arguments — the worst case for a leak. The surfaced ``ToolResult.error``
    must never contain the raw secret (it is scrubbed by the redactor).
    """
    secret = data.draw(st.sampled_from(_SYNTHETIC_SECRETS))
    tool = _ALL_TOOL_CLASSES[tool_name]()

    kwargs: dict[str, Any] = {}
    for spec in tool.metadata.parameters:
        if spec.type in (ParameterType.STRING, ParameterType.ANY):
            kwargs[spec.name] = secret if spec.enum is None else spec.enum[0]
        elif spec.type == ParameterType.ARRAY:
            kwargs[spec.name] = [secret]
        elif spec.type == ParameterType.OBJECT:
            kwargs[spec.name] = {"token": secret}
        elif spec.required:
            # Fill required non-string params with a benign valid value.
            kwargs[spec.name] = data.draw(_strategy_for_spec(spec))

    async def _boom(**kw: Any) -> Any:
        raise RuntimeError(f"upstream failure with args={kw}")

    with patch.object(tool, "_execute", new=_boom):
        result = asyncio.run(tool.execute(**kwargs))

    assert isinstance(result, ToolResult)
    if not result.success:
        assert secret not in (result.error or ""), (
            f"{tool_name}: raw secret leaked into error message"
        )
        assert not _has_secret(result.error or ""), (
            f"{tool_name}: secret pattern survived redaction"
        )


@pytest.mark.fuzz
def test_tool_discovery_nonempty():
    """Sanity guard: discovery must find the full built-in tool set.

    If a refactor drops tools from discovery, the parametrised fuzz above would
    silently shrink. This guards against that by asserting a healthy lower bound
    and that every discovered class is a BaseTool subclass.
    """
    assert len(_ALL_TOOL_NAMES) >= 30, (
        f"expected ≥30 built-in tools, discovered {len(_ALL_TOOL_NAMES)}"
    )
    for name in _ALL_TOOL_NAMES:
        assert issubclass(_ALL_TOOL_CLASSES[name], BaseTool)


# ---------------------------------------------------------------------------
# Tool-argument JSON cleaning
# ---------------------------------------------------------------------------

# The ReAct paths repair the JSON a small model writes for a tool call. The
# repair must fix syntax only — an argument value is data and has to reach the
# tool unchanged, commas, colons, braces and all.

_ARG_VALUES = st.text(
    alphabet=st.sampled_from(list('{}[]",:` \nabZ09')), min_size=0, max_size=30
)
_ARG_OBJECTS = st.dictionaries(
    st.sampled_from(["query", "expression", "command", "text", "path", "url"]),
    _ARG_VALUES | st.integers(-1000, 1000) | st.booleans() | st.none(),
    min_size=1,
    max_size=4,
)


@pytest.mark.fuzz
@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ARG_OBJECTS, st.sampled_from(["{body}", "```json\n{body}\n```", "```\n{body}\n```"]))
def test_clean_json_input_preserves_argument_values(args, wrap):
    """Valid tool-call JSON survives cleaning with every value intact."""
    import json

    from effgen.core.agent_runtime import AgentRuntimeMixin
    from effgen.core.tool_calling import ReActStrategy

    raw = wrap.format(body=json.dumps(args))
    for clean in (ReActStrategy.clean_json_input, AgentRuntimeMixin._clean_json_input):
        assert json.loads(clean(raw)) == args


@pytest.mark.fuzz
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"query": "Paris, France: population"}', {"query": "Paris, France: population"}),
        ('{"command": "cd build, then: make all"}', {"command": "cd build, then: make all"}),
        ('{"text": "Trailing, ] bracket"}', {"text": "Trailing, ] bracket"}),
        # Syntax defects are still repaired.
        ('{expression: "2+2"}', {"expression": "2+2"}),
        ('{"expression": "2+2",}', {"expression": "2+2"}),
        ('```json\n{"query": "Rome, Italy: history"}\n```', {"query": "Rome, Italy: history"}),
    ],
)
def test_clean_json_input_repairs_syntax_not_data(raw, expected):
    """Trailing commas and unquoted keys are repaired; string values are not."""
    import json

    from effgen.core.agent_runtime import AgentRuntimeMixin
    from effgen.core.tool_calling import ReActStrategy

    for clean in (ReActStrategy.clean_json_input, AgentRuntimeMixin._clean_json_input):
        assert json.loads(clean(raw)) == expected


@pytest.mark.fuzz
def test_react_parse_keeps_comma_colon_argument():
    """A ReAct tool call with a comma+colon argument parses into real arguments.

    When the JSON cannot be parsed the parser falls back to handing the whole
    raw string to the tool as ``__raw_input__``; a value containing a comma and
    a colon must not push it onto that path.
    """
    from effgen.core.tool_calling import get_strategy

    text = (
        "Thought: I need the population.\n"
        "Action: lookup\n"
        'Action Input: {"query": "Paris, France: population", "exact": true}\n'
    )
    result = get_strategy(mode="react").parse_response(text, tools={"lookup": object()})
    assert result.tool_name == "lookup"
    assert result.arguments == {"query": "Paris, France: population", "exact": True}
