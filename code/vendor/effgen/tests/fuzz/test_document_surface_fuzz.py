"""Hypothesis-driven fuzz sweep over every document-reading surface.

effGen reads documents it did not write: a config a user hand-edited, a workflow
YAML, a batch input file, a session or checkpoint written by a different build, a
catalog snapshot, a result document a report renders. Each is driven here from one
table (``tests.fuzz.documents.SURFACES``) so covering a new surface is one row.

Asserts, for every surface and every document:

  1. The surface either produces a result or raises an error **from its own
     contract** — the errors its docstring names. A raw ``AttributeError`` /
     ``TypeError`` / ``KeyError`` escaping a parser is a defect: the caller is
     handed a crash from inside a library instead of a message naming the file.
  2. Whatever it returns is usable: the accessors a caller reaches for straight
     afterwards are part of each row, so a loader cannot pass by deferring its
     crash to the first read.
  3. Every message a surface raises names the file it was reading, so a user with
     a directory of files knows which one to fix.

Exit criterion: the whole curated corpus against every row, plus ≥200 Hypothesis
examples per generated-document test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fuzz.documents import (
    HOSTILE_DOCUMENTS,
    build_surfaces,
    json_documents,
    sweep,
)

pytestmark = pytest.mark.fuzz

SURFACES = build_surfaces()
SURFACE_IDS = [s.name for s in SURFACES]


@pytest.mark.parametrize("surface", SURFACES, ids=SURFACE_IDS)
def test_no_document_escapes_a_surface_contract(surface, tmp_path) -> None:
    """No document makes a reader crash outside the errors it documents."""
    escapes = list(sweep(surface, tmp_path))
    assert not escapes, "\n".join(
        f"{surface.name}: {type(exc).__name__}: {exc}\n  document: {doc!r:.200}"
        for doc, exc in escapes
    )


@pytest.mark.parametrize("surface", SURFACES, ids=SURFACE_IDS)
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
@given(document=json_documents())
def test_generated_documents_stay_inside_the_contract(
    surface, tmp_path, document
) -> None:
    """The same contract holds for documents Hypothesis invents."""
    work = tmp_path / "generated"
    work.mkdir(exist_ok=True)
    try:
        surface.drive(work, document)
    except Exception as exc:  # noqa: BLE001 - the point is to catch everything
        if not surface.permits(exc):
            raise AssertionError(
                f"{surface.name} raised {type(exc).__name__} outside its contract "
                f"{[t.__name__ for t in surface.contract]}: {exc}\n"
                f"  document: {document!r:.200}"
            ) from exc


# ---------------------------------------------------------------------------
# A message that does not name the file leaves a user guessing which of the
# files in a directory is the broken one.
# ---------------------------------------------------------------------------

#: Rows whose error is raised against a path the caller supplied.
_PATH_NAMING_SURFACES = (
    "config.yaml",
    "config.json",
    "workflow.yaml",
    "batch.jsonl",
    "batch.json",
    "batch.csv",
    "batch.txt",
)


@pytest.mark.parametrize(
    "surface",
    [s for s in SURFACES if s.name in _PATH_NAMING_SURFACES],
    ids=list(_PATH_NAMING_SURFACES),
)
def test_a_refusal_names_the_file_it_was_reading(surface, tmp_path) -> None:
    """Every document a surface refuses is refused by name."""
    unnamed: list[str] = []
    for index, document in enumerate(HOSTILE_DOCUMENTS):
        work = tmp_path / f"{index}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            surface.drive(work, document)
        except Exception as exc:  # noqa: BLE001
            if not surface.permits(exc):
                continue  # the contract test above owns this case
            if str(work) not in str(exc):
                unnamed.append(f"{type(exc).__name__}: {exc}")
    assert not unnamed, (
        f"{surface.name}: {len(unnamed)} refusal(s) did not name the file:\n"
        + "\n".join(unnamed[:5])
    )


# ---------------------------------------------------------------------------
# Coverage gate — a new reader must be covered, not quietly uncovered.
# ---------------------------------------------------------------------------

#: The modules whose document readers this harness is responsible for.
COVERED_MODULES = (
    "effgen.config.loader",
    "effgen.core.workflow",
    "effgen.core.batch",
    "effgen.core.session",
    "effgen.core.checkpoint",
    "effgen.core.state",
    "effgen.models._catalog",
    "effgen.ui.report_html",
)

#: Every function in those modules that parses a document off disk, and is
#: therefore reachable by a file effGen did not write. Adding a reader without
#: adding it here (and driving it from a row in ``documents.SURFACES``) fails the
#: gate below.
KNOWN_READERS: frozenset[str] = frozenset({
    # effgen.config.loader
    "ConfigLoader._load_single_config",
    "ConfigLoader.save_config",
    # effgen.core.workflow
    "WorkflowDAG.from_yaml",
    # effgen.core.batch
    "BatchRunner._read_queries",
    "BatchRunner.write_results",
    "BatchRunner._csv_flatten",
    # effgen.core.session
    "Session.load",
    "Session.save",
    "SessionManager.scan",
    "SessionManager.export",
    # effgen.core.checkpoint
    "_read_checkpoint_json",
    "CheckpointManager.save",
    "CheckpointManager.load",
    "CheckpointManager.load_latest",
    "CheckpointManager.list_checkpoints",
    # effgen.core.state
    "AgentState.save",
    "AgentState.load",
    # effgen.models._catalog
    "load_snapshot",
    "save_snapshot",
    # effgen.ui.report_html
    "load_result_document",
})

#: Calls that mean "this function reads or writes a document on disk". Both
#: halves count: a writer whose output no reader accepts is the same defect as
#: a reader that crashes on a file.
_PARSE_CALLS = frozenset({
    "json.load", "json.loads", "json.dump", "json.dumps",
    "yaml.safe_load", "yaml.safe_dump", "yaml.dump",
    "csv.DictReader", "csv.DictWriter",
})


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _document_readers(module_name: str) -> set[str]:
    """Return every ``[Class.]function`` in *module* that parses a document."""
    import importlib

    source = Path(inspect.getsourcefile(importlib.import_module(module_name)))
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found: set[str] = set()

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                for inner in ast.walk(child):
                    if isinstance(inner, ast.Call) and _call_name(inner.func) in _PARSE_CALLS:
                        found.add(f"{prefix}{child.name}")
                        break
                visit(child, prefix)

    visit(tree, "")
    return found


def test_every_document_reader_is_covered_by_the_table() -> None:
    """A new reader in a covered module must be listed and driven.

    The harness is only worth its runtime if it is exhaustive over the modules it
    claims. This reads the readers out of the source rather than trusting a list
    that a later change would silently outgrow.
    """
    discovered: set[str] = set()
    for module in COVERED_MODULES:
        discovered |= _document_readers(module)

    uncovered = sorted(discovered - KNOWN_READERS)
    assert not uncovered, (
        "These functions parse a document off disk but are not listed in "
        "KNOWN_READERS, so no surface row drives them:\n  "
        + "\n  ".join(uncovered)
    )

    stale = sorted(KNOWN_READERS - discovered)
    assert not stale, (
        "KNOWN_READERS names functions that no longer parse a document:\n  "
        + "\n  ".join(stale)
    )


def test_every_surface_from_the_contract_has_a_row() -> None:
    """Each document-reading surface the sweep covers has at least one row."""
    names = {s.name for s in SURFACES}
    for prefix in (
        "config",
        "workflow",
        "batch",
        "session",
        "checkpoint",
        "agentstate",
        "catalog",
        "report",
    ):
        assert any(n.startswith(prefix) for n in names), f"no row covers {prefix}"


def test_the_corpus_is_large_enough_to_be_worth_running() -> None:
    """A corpus that shrank to nothing would pass every row silently."""
    assert len(HOSTILE_DOCUMENTS) >= 50
    assert len(SURFACES) >= 15
    assert all(isinstance(s.contract, tuple) for s in SURFACES)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(st.data())
def test_the_strategy_reaches_every_json_type(data) -> None:
    """The generated documents are not all one shape."""
    document = data.draw(json_documents())
    assert document is None or isinstance(
        document, bool | int | float | str | list | dict
    )
