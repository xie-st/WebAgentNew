"""Layout of the ``effgen.observability.tracing`` split.

Tracing is one facade plus five concerns:

* ``tracing_otel`` — the one optional-SDK import attempt and the SDK names the
  rest of the split needs;
* ``tracing_samplers`` — the five samplers, in both the SDK and the no-SDK form;
* ``tracing_provider`` — the ``TracerProvider`` lifecycle and the no-op tracer;
* ``tracing_buffer`` — the span ring buffer the dashboard reads, and the context
  variables that group records into runs and executions;
* ``tracing_spans`` — the ``start_*`` context managers and the helpers that
  stamp an outcome onto a span;
* ``tracing`` — the module docstring, ``__all__``, the re-exports, the legacy
  shims and the lookup that forwards the two provider globals.

Six invariants keep that split safe:

* every name has exactly one owning module;
* the facade stays the single import path: the set of names it exposes is
  unchanged, and each is the *same object* as on its owner. Sixteen modules bind
  these at import time, so a re-export that produced a copy would silently
  detach them;
* imports go one way only, so the graph stays acyclic;
* ``_provider`` and ``_initialized`` are rebound at run time by
  ``setup_tracing`` / ``shutdown_tracing`` / ``reset_tracing``. The facade must
  resolve both through ``__getattr__`` rather than binding a copy, which would
  freeze at ``None``/``False`` while the real provider moved on;
* the span buffer, its lock and the three context variables are shared by
  identity. Four test modules and the dashboard reach for them through the
  facade; a second buffer would leave one of the two readers permanently empty;
* every module logs under the same logger name, so a log line does not move to a
  name no operator has configured.

One consequence worth stating: ``_OTEL_AVAILABLE`` is imported by value, so
assigning ``tracing._OTEL_AVAILABLE = False`` no longer changes what
``get_tracer`` does. Nothing does that, and a test that needs the no-SDK path
should exercise ``tracing_otel`` instead.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import effgen.observability.tracing as facade

try:  # The SDK is optional; only the lifecycle check below needs it installed.
    import opentelemetry.sdk.trace  # noqa: F401

    _OTEL_SDK_INSTALLED = True
except ImportError:  # pragma: no cover - depends on the installed extras
    _OTEL_SDK_INSTALLED = False

PACKAGE = "effgen.observability"
FACADE = "tracing"

#: Every module the split is made of, the facade last.
SPLIT_MODULES = (
    "tracing_otel",
    "tracing_samplers",
    "tracing_provider",
    "tracing_buffer",
    "tracing_spans",
    FACADE,
)

#: Which module owns each name the facade exposes.
OWNERS = {
    # the optional SDK import
    "_OTEL_AVAILABLE": "tracing_otel",
    "Context": "tracing_otel",
    "Decision": "tracing_otel",
    "ParentBased": "tracing_otel",
    "Resource": "tracing_otel",
    "Sampler": "tracing_otel",
    "SamplingResult": "tracing_otel",
    "SimpleSpanProcessor": "tracing_otel",
    "SpanKind": "tracing_otel",
    "StatusCode": "tracing_otel",
    "TracerProvider": "tracing_otel",
    "_get_current_span": "tracing_otel",
    "_otel_trace": "tracing_otel",
    # samplers
    "AlwaysOnSampler": "tracing_samplers",
    "AlwaysOffSampler": "tracing_samplers",
    "TraceIdRatioSampler": "tracing_samplers",
    "RateLimitedSampler": "tracing_samplers",
    "ParentBasedSampler": "tracing_samplers",
    # provider lifecycle
    "setup_tracing": "tracing_provider",
    "get_tracer": "tracing_provider",
    "shutdown_tracing": "tracing_provider",
    "reset_tracing": "tracing_provider",
    "_provider_lock": "tracing_provider",
    "_NoOpSpan": "tracing_provider",
    "_NoOpTracer": "tracing_provider",
    # the span stream
    "_RUN_CONTEXT": "tracing_buffer",
    "_EXECUTION_CONTEXT": "tracing_buffer",
    "_SPAN_OUTCOMES": "tracing_buffer",
    "_SPAN_BUFFER": "tracing_buffer",
    "_SPAN_BUFFER_LOCK": "tracing_buffer",
    "new_execution_id": "tracing_buffer",
    "current_execution": "tracing_buffer",
    "execution_scope": "tracing_buffer",
    "_outcome_scope": "tracing_buffer",
    "mark_span_error": "tracing_buffer",
    "mark_run_error": "tracing_buffer",
    "record_skipped_step": "tracing_buffer",
    "_buffer_span": "tracing_buffer",
    "get_recent_spans": "tracing_buffer",
    # span construction
    "start_agent_run": "tracing_spans",
    "start_agent_iteration": "tracing_spans",
    "start_model_call": "tracing_spans",
    "start_tool_call": "tracing_spans",
    "start_router_decision": "tracing_spans",
    "record_retry_attempt": "tracing_spans",
    "set_span_ok": "tracing_spans",
    "set_span_error": "tracing_spans",
    "set_span_attribute": "tracing_spans",
    "_set_safe": "tracing_spans",
    "_mark_ok": "tracing_spans",
    "_mark_error_message": "tracing_spans",
    "_mark_error": "tracing_spans",
    "_has_outcome": "tracing_spans",
    "_has_tool_status": "tracing_spans",
    # the facade's own
    "trace_agent_run": FACADE,
    "trace_agent_iterate": FACADE,
    "trace_model_generate": FACADE,
    "trace_tool_execute": FACADE,
    "_infer_provider": FACADE,
}

#: What the facade itself defines. Everything else it exposes is a re-export.
FACADE_DEFINITIONS = {
    "__all__",
    "__dir__",
    "__getattr__",
    "_infer_provider",
    "logger",
    "trace_agent_iterate",
    "trace_agent_run",
    "trace_model_generate",
    "trace_tool_execute",
}

#: Rebound at run time, so the facade forwards them instead of binding a copy.
FORWARDED = ("_provider", "_initialized")

#: The names the facade exposed as a single file, pinned. Anything added or
#: dropped here is a change to what a caller can import from it.
EXPOSED_NAMES = {
    "AgentAttrs", "AlwaysOffSampler", "AlwaysOnSampler", "Any", "Context", "Decision",
    "ExecutionAttrs", "Generator", "ModelAttrs", "ParentBased", "ParentBasedSampler",
    "RateLimitedSampler", "Resource", "RetryAttrs", "RouterAttrs", "Sampler", "SamplingResult",
    "SimpleSpanProcessor", "SpanKind", "SpanName", "StatusCode", "ToolAttrs",
    "TraceIdRatioSampler", "TracerProvider", "_EXECUTION_CONTEXT", "_NoOpSpan", "_NoOpTracer",
    "_OTEL_AVAILABLE", "_RUN_CONTEXT", "_SPAN_BUFFER", "_SPAN_BUFFER_LOCK", "_SPAN_OUTCOMES",
    "_buffer_span", "_get_current_span", "_has_outcome", "_has_tool_status", "_infer_provider",
    "_initialized", "_mark_error", "_mark_error_message", "_mark_ok", "_otel_trace",
    "_outcome_scope", "_provider", "_provider_lock", "_set_safe", "annotations", "contextmanager",
    "contextvars", "current_execution", "deque", "execution_scope", "get_recent_spans",
    "get_tracer", "logger", "logging", "mark_run_error", "mark_span_error", "new_execution_id",
    "os", "record_retry_attempt", "record_skipped_step", "reset_tracing", "set_span_attribute",
    "set_span_error", "set_span_ok", "setup_tracing", "shutdown_tracing", "stamp_call_cost",
    "start_agent_iteration",
    "start_agent_run", "start_model_call", "start_router_decision", "start_tool_call",
    "threading", "time", "trace_agent_iterate", "trace_agent_run", "trace_model_generate",
    "trace_tool_execute", "uuid",
}

#: ``tracing.__all__``, pinned.
PUBLIC_API = [
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "RateLimitedSampler",
    "ParentBasedSampler",
    "setup_tracing",
    "get_tracer",
    "shutdown_tracing",
    "start_agent_run",
    "start_agent_iteration",
    "start_model_call",
    "start_tool_call",
    "start_router_decision",
    "record_retry_attempt",
    "set_span_ok",
    "set_span_error",
    "set_span_attribute",
    "stamp_call_cost",
    "mark_span_error",
    "mark_run_error",
    "record_skipped_step",
    "execution_scope",
    "current_execution",
    "new_execution_id",
    "trace_agent_run",
    "trace_agent_iterate",
    "trace_model_generate",
    "trace_tool_execute",
]

#: What ``effgen.observability`` re-exports from the facade.
PACKAGE_RE_EXPORTS = (
    "AlwaysOffSampler",
    "AlwaysOnSampler",
    "ParentBasedSampler",
    "RateLimitedSampler",
    "TraceIdRatioSampler",
    "get_tracer",
    "record_retry_attempt",
    "reset_tracing",
    "set_span_attribute",
    "set_span_error",
    "setup_tracing",
    "shutdown_tracing",
    "start_agent_iteration",
    "start_agent_run",
    "start_model_call",
    "start_router_decision",
    "start_tool_call",
)

#: What each module may import from within the split.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "tracing_otel": set(),
    "tracing_samplers": {"tracing_otel"},
    "tracing_provider": {"tracing_otel", "tracing_samplers"},
    "tracing_buffer": {"tracing_otel"},
    "tracing_spans": {"tracing_otel", "tracing_provider", "tracing_buffer"},
    FACADE: set(SPLIT_MODULES) - {FACADE},
}

#: Every module in the split logs under this name, whichever one the call sits in.
LOGGER_NAME = "effgen.observability.tracing"

#: The buffer bound, pinned: the dashboard reads a fixed-size window.
BUFFER_MAXLEN = 500

#: State shared by identity. A duplicate leaves one reader permanently empty.
SHARED_STATE = ("_SPAN_BUFFER", "_SPAN_BUFFER_LOCK", "_RUN_CONTEXT", "_EXECUTION_CONTEXT",
                "_SPAN_OUTCOMES")

#: No module in the split may grow past this.
# Raised from 500 to 520 on 2026-08-13, for one function: ``stamp_call_cost``
# belongs beside ``set_span_attribute`` (both write model-call attributes) and
# is what puts the cost on a span independently of context propagation. The cap
# exists so the split modules do not regrow into the file they came from, not to
# refuse a function that belongs; it moves only with a reason recorded here.
MAX_LINES = 520


def module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


def source_path(name: str) -> Path:
    return Path(inspect.getsourcefile(module(name)))


def tree_of(name: str) -> ast.Module:
    return ast.parse(source_path(name).read_text(encoding="utf-8"))


def _module_scope_body(node: ast.Module) -> list[ast.stmt]:
    """Module-scope statements, stepping into ``if`` and ``try`` blocks.

    The samplers are declared inside ``if _OTEL_AVAILABLE:`` / ``else:`` and the
    SDK names inside a ``try``/``except ImportError``, so a walk over
    ``Module.body`` alone would report both groups as undefined.
    """
    out: list[ast.stmt] = []
    for stmt in node.body:
        if isinstance(stmt, ast.If):
            out.extend(_module_scope_body(ast.Module(body=stmt.body, type_ignores=[])))
            out.extend(_module_scope_body(ast.Module(body=stmt.orelse, type_ignores=[])))
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                out.extend(_module_scope_body(ast.Module(body=block, type_ignores=[])))
            for handler in stmt.handlers:
                out.extend(_module_scope_body(ast.Module(body=handler.body, type_ignores=[])))
        else:
            out.append(stmt)
    return out


def declared_names(name: str) -> set[str]:
    """Names the module declares in its own source — every import excluded."""
    names: set[str] = set()
    for node in _module_scope_body(tree_of(name)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def own_definitions(name: str) -> set[str]:
    """Names the module binds itself — re-exports from a sibling excluded.

    An SDK name pulled in from ``opentelemetry`` counts as a definition: that
    import is how ``tracing_otel`` owns it. A name imported from another module
    of this split does not.
    """
    names: set[str] = set()
    for node in _module_scope_body(tree_of(name)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            tail = (node.module or "").split(".")[-1]
            if tail in SPLIT_MODULES:
                continue
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def _import_module_argument(node: ast.Call) -> str | None:
    """The literal module name a dynamic import call names, if it is one."""
    func = node.func
    dynamic = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
        isinstance(func, ast.Name) and func.id in ("import_module", "__import__")
    )
    if not dynamic or not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def imported_split_modules(name: str) -> set[str]:
    """The modules of this split that *name* imports, however it spells them.

    All four spellings count: ``from .tracing_x import y``,
    ``from effgen.observability import tracing_x`` (which names the module in the
    alias rather than the module path), ``import effgen.observability.tracing_x``
    and a dynamic ``importlib.import_module("effgen.observability.tracing_x")``.
    """
    found: set[str] = set()

    def note(dotted: str) -> None:
        tail = dotted.split(".")[-1]
        if tail in SPLIT_MODULES:
            found.add(tail)

    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                note(node.module)
            for alias in node.names:
                if alias.name in SPLIT_MODULES:
                    found.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.Call):
            dotted = _import_module_argument(node)
            if dotted:
                note(dotted)
    return found


# ---------------------------------------------------------------------------
# One owner per name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("member", "owner"), sorted(OWNERS.items()))
def test_each_name_is_defined_by_exactly_one_module(member, owner):
    assert member in own_definitions(owner), f"{owner} does not define {member}"
    others = [m for m in SPLIT_MODULES if m != owner and member in own_definitions(m)]
    assert not others, f"{member} is also defined in {others}"


def test_facade_defines_only_its_shims_and_the_forwarding_lookup():
    assert declared_names(FACADE) == FACADE_DEFINITIONS


# ---------------------------------------------------------------------------
# The facade stays the single import path
# ---------------------------------------------------------------------------


def test_the_facade_exposes_exactly_the_names_it_always_exposed():
    assert {n for n in dir(facade) if not n.startswith("__")} == EXPOSED_NAMES


@pytest.mark.parametrize(("member", "owner"), sorted(OWNERS.items()))
def test_facade_re_exports_every_moved_name_as_the_same_object(member, owner):
    assert hasattr(facade, member), f"{member} no longer resolves on the facade"
    assert getattr(facade, member) is getattr(module(owner), member), (
        f"{member} on the facade is a different object from {owner}.{member}"
    )


def test_public_api_is_unchanged():
    assert facade.__all__ == PUBLIC_API


@pytest.mark.parametrize("name", PACKAGE_RE_EXPORTS)
def test_the_package_still_re_exports_the_same_object(name):
    package = importlib.import_module(PACKAGE)
    assert name in package.__all__, f"{name} dropped out of {PACKAGE}.__all__"
    assert getattr(package, name) is getattr(facade, name)


# ---------------------------------------------------------------------------
# Imports go one way only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_split_modules_only_import_downwards(name):
    allowed = ALLOWED_IMPORTS[name]
    assert imported_split_modules(name) <= allowed, (
        f"{name} imports {sorted(imported_split_modules(name) - allowed)}"
    )


def test_no_sibling_imports_the_facade():
    for name in SPLIT_MODULES:
        if name == FACADE:
            continue
        assert FACADE not in imported_split_modules(name), f"{name} imports the facade"


def test_the_buffer_does_not_reach_for_the_provider_or_the_span_helpers():
    reached = imported_split_modules("tracing_buffer")
    assert not reached & {"tracing_provider", "tracing_spans"}, (
        f"tracing_buffer imports {sorted(reached)}"
    )


# ---------------------------------------------------------------------------
# The two rebound globals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FORWARDED)
def test_the_rebound_globals_are_not_bound_on_the_facade(name):
    """A copy taken at import time would stay ``None``/``False`` forever."""
    assert name not in vars(facade), (
        f"{name} is bound on the facade; module __getattr__ never runs for it"
    )
    assert name in vars(module("tracing_provider"))


@pytest.mark.skipif(
    not _OTEL_SDK_INSTALLED, reason="opentelemetry-sdk not installed"
)
def test_the_facade_follows_the_provider_through_its_lifecycle():
    """The rest of this module reads the split's shape and needs no SDK.

    This one is the exception: it asserts a provider is actually built, which
    only happens when the OpenTelemetry SDK is installed. Without the guard it
    fails on a minimal install, where ``setup_tracing`` returns early by design
    and leaves the provider unset.
    """
    provider_module = module("tracing_provider")
    facade.reset_tracing()
    assert facade._provider is None
    assert facade._initialized is False

    facade.setup_tracing(service_name="layout-gate")
    try:
        assert facade._provider is not None
        assert facade._provider is provider_module._provider
        assert facade._initialized is provider_module._initialized is True
    finally:
        facade.reset_tracing()

    assert facade._provider is None
    assert facade._initialized is False


def test_an_unknown_attribute_still_raises():
    """Without the raise, a typo would resolve to ``None`` instead of failing."""
    missing = "no_such_name"
    with pytest.raises(AttributeError):
        getattr(facade, missing)


def test_dir_lists_the_forwarded_globals():
    for name in FORWARDED:
        assert name in dir(facade)


# ---------------------------------------------------------------------------
# Shared state and the logger name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SHARED_STATE)
def test_the_span_stream_state_is_one_object_not_two(name):
    assert getattr(facade, name) is getattr(module("tracing_buffer"), name)


def test_the_buffer_bound_is_unchanged():
    assert facade._SPAN_BUFFER.maxlen == BUFFER_MAXLEN


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_every_module_logs_under_the_one_tracing_logger(name):
    assert module(name).logger.name == LOGGER_NAME


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_no_module_in_the_split_grows_past_the_cap(name):
    lines = len(source_path(name).read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_LINES, f"{name} is {lines} lines"
