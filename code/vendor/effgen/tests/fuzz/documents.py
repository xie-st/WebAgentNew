"""Shared corpus and surface table for the document-loading fuzz harnesses.

effGen reads documents a user wrote by hand, documents a previous build wrote, and
documents a provider refresh wrote — configs, workflow YAML, batch input files,
session and checkpoint files, catalog snapshots — and it writes an HTML report a
browser renders. Every one of those is a place where an arbitrary document reaches
parsing code.

This module holds the three things every such harness needs, so covering a new
surface is one row rather than a new file:

* :data:`HOSTILE_DOCUMENTS` — a curated corpus of JSON-representable documents that
  are hostile to a loader: the wrong top-level type, the wrong nested type, an
  absent required key, non-string keys, control characters, lone surrogates, markup,
  template syntax, degenerate numbers, deep nesting, duplicate ids, and a cycle.
* :func:`json_documents` — a recursive Hypothesis strategy over the same value space.
* :data:`SURFACES` — the surfaces as data. Each :class:`Surface` names the entry
  point, a callable that writes a document where that entry point reads it and calls
  it, and the exception types that entry point's own contract permits.

A surface's ``contract`` is the set of errors it documents. Anything else escaping —
``AttributeError``, ``TypeError``, ``KeyError``, ``IndexError`` — is a defect: the
caller gets a crash from inside a parser instead of a message naming the file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Marker payloads. A surface that renders or echoes a document must not emit
# these in an executable position, and must not lose them silently.
# ---------------------------------------------------------------------------

MARKUP_PAYLOAD = '<script>alert("x")</script>'
ATTR_PAYLOAD = '" onmouseover="alert(1)'
CLOSE_PAYLOAD = "</textarea></style></title>"
JS_URL = "javascript:alert(1)"
DATA_URL = "data:text/html,<script>alert(1)</script>"
TEMPLATE_PAYLOAD = "${HOME}{{7*7}}#{x}"
ENV_PAYLOAD = "${EFFGEN_FUZZ_UNSET_VAR}"

#: Payloads that must never reach a rendered page as live markup or a live URL.
INJECTION_PAYLOADS: tuple[str, ...] = (
    MARKUP_PAYLOAD,
    ATTR_PAYLOAD,
    CLOSE_PAYLOAD,
    JS_URL,
    DATA_URL,
)


# ---------------------------------------------------------------------------
# The curated corpus
# ---------------------------------------------------------------------------

#: Values that are legal JSON but hostile in some position.
HOSTILE_SCALARS: tuple[Any, ...] = (
    None,
    True,
    False,
    0,
    -1,
    10**18,
    -(10**18),
    0.0,
    -0.0,
    1e308,
    1e-308,
    "",
    " ",
    "\t\n\r",
    "\x00\x01\x1f",
    "​‮",
    "퟿",
    "0" * 4096,
    "не-ascii — ünïcödé 🙂",
    MARKUP_PAYLOAD,
    ATTR_PAYLOAD,
    CLOSE_PAYLOAD,
    JS_URL,
    DATA_URL,
    TEMPLATE_PAYLOAD,
    ENV_PAYLOAD,
    "../../etc/passwd",
    "..\\..\\windows",
    "NaN",
    "Infinity",
    "1970-01-01",
    "not-a-date",
    "9999-99-99",
)


def _deep(depth: int) -> Any:
    node: Any = "leaf"
    for _ in range(depth):
        node = {"nested": node}
    return node


#: Documents that are hostile to *any* loader, regardless of the shape it wants.
GENERIC_DOCUMENTS: tuple[Any, ...] = (
    None,
    [],
    {},
    0,
    -1,
    3.5,
    True,
    False,
    "a bare string",
    "",
    [1, 2, 3],
    ["a", "b"],
    [None, None],
    [[]],
    [{}],
    [{"a": 1}, "b", 3, None],
    {"": ""},
    {"a": None},
    {"a": []},
    {"a": {}},
    {"a": [1, {"b": None}]},
    {"0": {"1": {"2": None}}},
    _deep(12),
    {"models": "not-a-mapping"},
    {"models": [1, 2]},
    {"tools": 5},
    {"prompts": None},
    {"system_prompts": ["x"]},
    {MARKUP_PAYLOAD: MARKUP_PAYLOAD},
    {"a": MARKUP_PAYLOAD, "b": [ATTR_PAYLOAD, JS_URL]},
    {"a": TEMPLATE_PAYLOAD},
    {"a": ENV_PAYLOAD},
    {"a": "0" * 4096},
    {"a": 10**18, "b": 1e308, "c": -0.0},
    {"a": "\x00\x01", "b": "‮"},
    {"a": {"b": {"c": [{"d": None}]}}},
)

#: Documents shaped like a workflow file but wrong in one specific way.
WORKFLOW_DOCUMENTS: tuple[Any, ...] = (
    {"workflow": None},
    {"workflow": "a string"},
    {"workflow": []},
    {"workflow": 7},
    {"workflow": {"name": None}},
    {"workflow": {"name": ["x"]}},
    {"nodes": None},
    {"nodes": "abc"},
    {"nodes": 5},
    {"nodes": {"a": 1}},
    {"nodes": [None]},
    {"nodes": ["a string node"]},
    {"nodes": [5]},
    {"nodes": [{}]},
    {"nodes": [{"agent": "x"}]},
    {"nodes": [{"id": None}]},
    {"nodes": [{"id": []}]},
    {"nodes": [{"id": {"a": 1}}]},
    {"nodes": [{"id": 1}, {"id": 1}]},
    {"nodes": [{"id": "a"}, {"id": "a"}]},
    {"nodes": [{"id": "a", "tools": "web"}]},
    {"nodes": [{"id": "a", "depends_on": "b"}]},
    {"nodes": [{"id": "a", "depends_on": ["missing"]}]},
    {"nodes": [{"id": "a"}, {"id": "b", "depends_on": [None]}]},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": None},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": "a,b"},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": 3},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [["a"]]},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a"}]},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"from": "a", "to": "b"}]},
    {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [["a", "b"], ["b", "a"]]},
    {"nodes": [{"id": "a"}], "edges": [["a", "a"]]},
    {"nodes": [{"id": "a"}], "edge": [["a", "a"]]},
    {"nodes": [{"id": "a", "output_key": None}]},
    {"nodes": [{"id": MARKUP_PAYLOAD}]},
)

#: Documents shaped like a session or checkpoint file but wrong in one way.
STATE_DOCUMENTS: tuple[Any, ...] = (
    {"session_id": None},
    {"session_id": 5},
    {"session_id": ["a"]},
    {"messages": None},
    {"messages": "abc"},
    {"messages": 3},
    {"messages": [None]},
    {"messages": ["plain"]},
    {"messages": [{"role": "assistant", "metadata": None}]},
    {"messages": [{"role": "assistant", "metadata": "x"}]},
    {"messages": [{"role": "assistant", "metadata": {"cost_usd": "free"}}]},
    {"messages": [{"role": "assistant", "metadata": {"cost_usd": True}}]},
    {"messages": [{"role": "assistant", "metadata": {"cost_usd": float("inf")}}]},
    {"messages": [{"role": "assistant", "metadata": {"run_id": ["a"], "cost_usd": 1}}]},
    {"metadata": None},
    {"metadata": []},
    {"memory": "not a dict"},
    {"created_at": 5},
    {"updated_at": 5},
    {"updated_at": None},
    {"updated_at": {"a": 1}},
    {"updated_at": "not-a-timestamp"},
    {"agent_name": 12},
    {"iteration": "many"},
    {"tool_calls": None},
    {"tokens_used": "lots"},
    {"scratchpad": ["a"]},
    {"tool_states": []},
    {"partial_output": 5},
    {"unknown_field_from_a_future_build": 1},
    {"agent_id": None},
    {"agent_id": 5},
    {"agent_id": ["a"]},
    {"agent_id": "a", "conversation_history": None},
    {"agent_id": "a", "conversation_history": "abc"},
    {"agent_id": "a", "conversation_history": ["plain", 5, None]},
    {"agent_id": "a", "tool_history": {"a": 1}},
    {"agent_id": "a", "tool_history": ["plain"]},
    {"agent_id": "a", "created_at": 5},
    {"agent_id": "a", "created_at": "not-a-timestamp"},
    {"agent_id": "a", "updated_at": []},
)

#: Documents shaped like a catalog snapshot but wrong in one way.
SNAPSHOT_DOCUMENTS: tuple[Any, ...] = (
    {"models": None},
    {"models": "abc"},
    {"models": 5},
    {"models": {"a": 1}},
    {"models": [None]},
    {"models": ["a string"]},
    {"models": [5]},
    {"models": [{}]},
    {"models": [{"id": None}]},
    {"models": [{"id": "m", "context_window": "big"}]},
    {"models": [{"id": "m", "price_in_per_1m": "free"}]},
    {"models": [{"id": "ft:private:org:x"}]},
    {"verified_on": None},
    {"verified_on": 5},
    {"verified_on": []},
    {"verified_on": "not-a-date"},
    {"verified_on": "9999-99-99"},
    {"verified_on": {"a": 1}},
    {"count": "many"},
    {"count": None},
    {"source": []},
    {"provider": 5},
)

#: Documents shaped like a result document the HTML report renders.
REPORT_DOCUMENTS: tuple[Any, ...] = (
    {"kind": "run"},
    {"kind": "run", "task": MARKUP_PAYLOAD, "model": ATTR_PAYLOAD},
    {"kind": "run", "task": None, "answer": MARKUP_PAYLOAD},
    {"kind": "run", "sources": [JS_URL, DATA_URL, MARKUP_PAYLOAD]},
    {"kind": "run", "sources": "not-a-list"},
    {"kind": "run", "sources": [None, 5, {"url": JS_URL}]},
    {"kind": "run", "citations": [{"index": MARKUP_PAYLOAD, "quote": CLOSE_PAYLOAD}]},
    {"kind": "run", "citations": "abc"},
    {"kind": "run", "trace": None},
    {"kind": "run", "trace": "abc"},
    {"kind": "run", "trace": {"children": "abc"}},
    {"kind": "run", "error": MARKUP_PAYLOAD},
    {"kind": "run", "error": {"message": MARKUP_PAYLOAD, "type": ATTR_PAYLOAD}},
    {"kind": "run", "metadata": None},
    {"kind": "run", "metadata": "abc"},
    {"kind": "run", "metadata": {"cost_usd": "free", "total_tokens": "many"}},
    # Shapes ``detect_report_kind`` recognizes, so each body renderer is reached
    # with its own collection wrong in one way.
    {"scores": {"m": {"accuracy": 1}}, "recommendations": {}},
    {"scores": "abc", "recommendations": {}},
    {"scores": 5, "recommendations": {}},
    {"scores": [None, 5, "text"], "recommendations": {}},
    {"scores": [{"model": MARKUP_PAYLOAD}], "recommendations": "abc"},
    {"scores": [{"model": "m", "responses": "abc"}], "recommendations": {}},
    {"scores": [{"model": "m", "responses": [None, 5]}], "recommendations": {}},
    {"suite": MARKUP_PAYLOAD, "results": None},
    {"suite": "s", "results": "abc"},
    {"suite": "s", "results": {"a": 1}},
    {"suite": "s", "results": [None, 5, "text"]},
    {"suite": "s", "results": [{"query": MARKUP_PAYLOAD, "cost_usd": "free"}]},
    {"period": MARKUP_PAYLOAD, "rows": None},
    {"period": "p", "rows": "abc"},
    {"period": "p", "rows": {"a": 1}},
    {"period": "p", "rows": [None, 5]},
    {"period": "p", "rows": [{"model": MARKUP_PAYLOAD, "cost_usd": "free"}]},
    {"period": "p", "rows": [{"cost_usd": 1}], "total_cost_usd": "free"},
    {"period": "p", "rows": [{"cost_usd": 1}], "daily_budget_usd": 1, "period_days": 1,
     "total_cost_usd": "lots"},
    {"latency": None, "throughput_rps": 1},
    {"latency": "abc", "throughput_rps": "fast"},
    {"latency": {"p95": "slow"}, "throughput_rps": 1, "errors": {"": MARKUP_PAYLOAD}},
    {"latency": {}, "throughput_rps": 1, "errors": "abc"},
    {"prompt": MARKUP_PAYLOAD, "contenders": None},
    {"prompt": "p", "contenders": "abc"},
    {"prompt": "p", "contenders": [None, 5]},
    {"prompt": "p", "contenders": [{"model": MARKUP_PAYLOAD}], "verdict": "abc"},
    {"output": MARKUP_PAYLOAD, "success": True, "sources": "abc"},
    {"output": "o", "success": True, "trace": None, "execution_tree": "abc"},
    {"output": "o", "success": True, "citations": "abc", "metadata": "abc"},
    {"output": "o", "success": True, "error": MARKUP_PAYLOAD},
    {"kind": MARKUP_PAYLOAD},
)

#: Everything a generic sweep should try against every surface.
HOSTILE_DOCUMENTS: tuple[Any, ...] = (
    GENERIC_DOCUMENTS
    + WORKFLOW_DOCUMENTS
    + STATE_DOCUMENTS
    + SNAPSHOT_DOCUMENTS
    + REPORT_DOCUMENTS
)


# ---------------------------------------------------------------------------
# Hypothesis strategy over the same value space
# ---------------------------------------------------------------------------

_TEXT = st.one_of(
    st.text(max_size=24),
    st.sampled_from([
        "", " ", "\x00", "‮", MARKUP_PAYLOAD, ATTR_PAYLOAD, CLOSE_PAYLOAD,
        JS_URL, DATA_URL, TEMPLATE_PAYLOAD, ENV_PAYLOAD, "id", "name", "models",
    ]),
)

_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    _TEXT,
)


def json_documents(max_leaves: int = 12) -> st.SearchStrategy[Any]:
    """A recursive strategy over arbitrary JSON-representable documents."""
    return st.recursive(
        _SCALARS,
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(_TEXT, children, max_size=4),
        ),
        max_leaves=max_leaves,
    )


def json_objects(max_leaves: int = 12) -> st.SearchStrategy[dict[str, Any]]:
    """A recursive strategy restricted to documents whose top level is a mapping."""
    return st.dictionaries(_TEXT, json_documents(max_leaves), max_size=5)


# ---------------------------------------------------------------------------
# Writers — put a document where a surface reads it
# ---------------------------------------------------------------------------


def write_yaml(path: Path, document: Any) -> Path:
    """Serialize *document* as YAML. Falls back to JSON, which YAML accepts."""
    try:
        text = yaml.safe_dump(document, allow_unicode=True, default_flow_style=False)
    except yaml.YAMLError:
        text = json.dumps(document, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, document: Any) -> Path:
    """Serialize *document* as JSON."""
    path.write_text(
        json.dumps(document, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# The surface table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """One document-reading entry point, and the errors its contract permits.

    ``drive`` receives a scratch directory and one document; it writes the
    document where the entry point reads it and calls the entry point (and the
    accessors a caller reaches for straight afterwards, since a loader that
    defers its crash to the first read has not handled the document).
    """

    name: str
    drive: Callable[[Path, Any], Any]
    contract: tuple[type[BaseException], ...]
    documents: tuple[Any, ...] = field(default=HOSTILE_DOCUMENTS)

    def permits(self, exc: BaseException) -> bool:
        return isinstance(exc, self.contract)


def _config_surface(work: Path, document: Any) -> Any:
    from effgen.config.loader import ConfigLoader

    path = write_yaml(work / "config.yaml", document)
    loader = ConfigLoader(config_dir=str(work))
    config = loader.load_config(path)
    config.to_dict()
    loader.get("models.name.temperature")
    return config


def _config_json_surface(work: Path, document: Any) -> Any:
    from effgen.config.loader import ConfigLoader

    path = write_json(work / "config.json", document)
    loader = ConfigLoader(config_dir=str(work))
    config = loader.load_config(path, merge=False)
    config.to_dict()
    return config


def _config_multi_surface(work: Path, document: Any) -> Any:
    from effgen.config.loader import ConfigLoader

    base = write_yaml(work / "base.yaml", {"models": {"a": {"temperature": 0.5}}})
    other = write_yaml(work / "over.yaml", document)
    loader = ConfigLoader(config_dir=str(work))
    config = loader.load_config([base, other])
    config.to_dict()
    return config


def _workflow_surface(work: Path, document: Any) -> Any:
    from effgen.core.workflow import WorkflowDAG

    path = write_yaml(work / "workflow.yaml", document)
    dag = WorkflowDAG.from_yaml(str(path))
    # Reach for the accessors a caller uses straight after loading, so a loader
    # cannot pass by deferring its crash to the first read.
    assert isinstance(dag.topological_order(), list)
    assert isinstance(dag.entry_nodes(), list)
    assert isinstance(dag.nodes, list)
    assert isinstance(dag.edges, list)
    assert isinstance(dag.to_dict(), dict)
    return dag


def _batch_surface(suffix: str) -> Callable[[Path, Any], Any]:
    def drive(work: Path, document: Any) -> Any:
        from effgen.core.batch import BatchRunner

        path = write_json(work / f"queries{suffix}", document)
        return BatchRunner._read_queries(path, "query")

    return drive


def _batch_roundtrip_surface(suffix: str) -> Callable[[Path, Any], Any]:
    """What ``write_results`` writes, ``_read_queries`` must be able to read back."""

    def drive(work: Path, document: Any) -> Any:
        from effgen.core.batch import BatchRunner

        queries, result = batch_result_for(document)
        out = work / f"results{suffix}"
        BatchRunner.write_results(result, out, query_list=queries)
        return BatchRunner._read_queries(out, "query")

    return drive


def batch_result_for(document: Any) -> tuple[list[str], Any]:
    """Build ``(queries, BatchResult)`` carrying *document* as its text."""
    from effgen.core.agent import AgentResponse
    from effgen.core.batch import BatchResult

    text = str(document)[:200] or "q"
    queries = [text, "plain query"]
    responses = [
        AgentResponse(
            output=text,
            success=True,
            metadata={"cost_usd": 0.5, "parsed": _as_mapping(document)},
        ),
        AgentResponse(output="ok", success=True),
    ]
    return queries, BatchResult(results=responses, total=2, succeeded=2)


def _session_surface(work: Path, document: Any) -> Any:
    from effgen.core.session import Session, SessionManager

    store = work / "sessions"
    store.mkdir(exist_ok=True)
    write_json(store / "fuzz.json", document)
    manager = SessionManager(str(store))
    manager.scan()
    manager.list_sessions()
    manager.cleanup(older_than_days=0)
    if (store / "fuzz.json").exists():
        session = Session.load("fuzz", str(store))
        session.to_dict()
        return session
    return None


def _session_export_surface(work: Path, document: Any) -> Any:
    from effgen.core.session import SessionManager

    store = work / "sessions"
    store.mkdir(exist_ok=True)
    write_json(store / "fuzz.json", document)
    manager = SessionManager(str(store))
    manager.export("fuzz", "json")
    return manager.export("fuzz", "text")


def _session_roundtrip_surface(work: Path, document: Any) -> Any:
    from effgen.core.session import Session

    store = work / "sessions"
    store.mkdir(exist_ok=True)
    session = Session(session_id="rt", metadata=_as_mapping(document))
    session.add_message("user", str(document)[:200], cost_usd=0.5, run_id="r1")
    session.save(str(store))
    return Session.load("rt", str(store))


def _checkpoint_surface(backend: str) -> Callable[[Path, Any], Any]:
    def drive(work: Path, document: Any) -> Any:
        from effgen.core.checkpoint import CheckpointManager

        store = work / f"cp-{backend}"
        manager = CheckpointManager(str(store), backend=backend)
        if backend == "filesystem":
            write_json(store / "fuzz.json", document)
        manager.list_checkpoints()
        try:
            return manager.load("fuzz")
        except FileNotFoundError:
            return None

    return drive


def _checkpoint_roundtrip_surface(work: Path, document: Any) -> Any:
    from effgen.core.checkpoint import Checkpoint, CheckpointManager

    manager = CheckpointManager(str(work / "cp-rt"))
    checkpoint = Checkpoint(
        checkpoint_id="rt",
        agent_name="fuzz",
        task=str(document)[:200],
        iteration=1,
        metadata=_as_mapping(document),
    )
    manager.save(checkpoint)
    manager.list_checkpoints()
    manager.load_latest()
    return manager.load("rt")


def _agent_state_surface(work: Path, document: Any) -> Any:
    from effgen.core.state import AgentState

    path = write_json(work / "state.json", document)
    state = AgentState.load(str(path))
    # An agent hydrates itself from these straight after loading.
    for message in state.conversation_history:
        message.get("role")
    for call in state.tool_history:
        call.get("tool")
    state.add_message("user", "after")
    state.to_dict()
    return state


def _agent_state_roundtrip_surface(work: Path, document: Any) -> Any:
    from effgen.core.state import AgentState

    path = work / "rt.json"
    state = AgentState(agent_id="rt", metadata=_as_mapping(document))
    state.add_message("user", str(document)[:200])
    state.add_tool_call("calculator", {"expression": "1+1"}, document)
    state.save(str(path))
    return AgentState.load(str(path))


def _snapshot_surface(work: Path, document: Any) -> Any:
    from effgen.models import _catalog

    path = write_json(work / "provider.json", document)
    _catalog.load_snapshot("openai", path=path)
    _catalog.snapshot_age_days("openai", path=path)
    return _catalog.load_snapshot_records("openai", path=path)


def _report_surface(work: Path, document: Any) -> Any:
    from effgen.ui.report_html import build_html_report, load_result_document

    path = write_json(work / "result.json", document)
    data = load_result_document(path)
    return build_html_report(data)


def _as_mapping(document: Any) -> dict[str, Any]:
    return document if isinstance(document, dict) else {"value": document}


_YAML_ERRORS: tuple[type[BaseException], ...] = (yaml.YAMLError,)


def _lazy_contract(*names: str) -> tuple[type[BaseException], ...]:
    from effgen.errors import CorruptStateError
    from effgen.ui.report_html import ReportError

    table = {"CorruptStateError": CorruptStateError, "ReportError": ReportError}
    return tuple(table[n] for n in names)


def build_surfaces() -> tuple[Surface, ...]:
    """Return the surface table.

    Built lazily so importing this module does not import half of effGen.
    """
    corrupt, report_error = _lazy_contract("CorruptStateError", "ReportError")
    parse = (ValueError, *_YAML_ERRORS)
    return (
        Surface("config.yaml", _config_surface, parse),
        Surface("config.json", _config_json_surface, parse),
        Surface("config.merge", _config_multi_surface, parse),
        Surface("workflow.yaml", _workflow_surface, parse),
        Surface("batch.jsonl", _batch_surface(".jsonl"), (ValueError,)),
        Surface("batch.json", _batch_surface(".json"), (ValueError,)),
        Surface("batch.csv", _batch_surface(".csv"), (ValueError,)),
        Surface("batch.txt", _batch_surface(".txt"), (ValueError,)),
        Surface("batch.roundtrip.jsonl", _batch_roundtrip_surface(".jsonl"), (ValueError,)),
        Surface("batch.roundtrip.json", _batch_roundtrip_surface(".json"), (ValueError,)),
        Surface("batch.roundtrip.csv", _batch_roundtrip_surface(".csv"), (ValueError,)),
        Surface("session.store", _session_surface, (ValueError, corrupt)),
        Surface("session.export", _session_export_surface, (ValueError, corrupt)),
        Surface("session.roundtrip", _session_roundtrip_surface, ()),
        Surface("checkpoint.filesystem", _checkpoint_surface("filesystem"), (ValueError, corrupt)),
        Surface("checkpoint.sqlite", _checkpoint_surface("sqlite"), (ValueError, corrupt)),
        Surface("checkpoint.roundtrip", _checkpoint_roundtrip_surface, ()),
        Surface("agentstate.file", _agent_state_surface, (ValueError, corrupt)),
        Surface("agentstate.roundtrip", _agent_state_roundtrip_surface, ()),
        Surface("catalog.snapshot", _snapshot_surface, ()),
        Surface("report.html", _report_surface, (report_error,)),
    )


def sweep(surface: Surface, work: Path) -> Iterator[tuple[Any, BaseException]]:
    """Drive *surface* over its documents, yielding every uncontracted escape."""
    for index, document in enumerate(surface.documents):
        scratch = work / f"{surface.name}-{index}"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            surface.drive(scratch, document)
        except Exception as exc:  # noqa: BLE001 - the point is to catch everything
            if not surface.permits(exc):
                yield document, exc
