"""Layout of the ``effgen.ui.report_html`` split.

The report renderer is three shared layers plus one builder per report kind:

* ``report_html_format`` — escaping, value formatting, document normalizing;
* ``report_html_components`` — cards, tables, badges, links, inline charts;
* ``report_html_page`` — the inline stylesheet, the inline script, the shell;
* ``report_html_<kind>`` — the builder for exactly one entry of ``REPORT_KINDS``;
* ``report_html`` — kind detection, the refusals, and the three entry points.

Four invariants keep that split safe:

* the kind-to-module mapping is mechanical, so adding a kind means adding a
  module rather than growing an existing one;
* every name a split module defines is re-exported from ``report_html``, and it
  is the same object, so the module stays the single import path;
* the layers only import downwards — a builder never imports the facade or
  another builder, and the formatting layer imports no report module at all —
  so the import graph stays acyclic;
* no module in the split emits a construct that would make a browser retrieve
  anything, which is what keeps a written report readable with no network.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

import effgen.ui.report_html as facade
from effgen.ui.report_html import _BODY_RENDERERS, REPORT_KINDS

PACKAGE = "effgen.ui"

#: The three shared layers, in dependency order.
SHARED_LAYERS = (
    "report_html_format",
    "report_html_components",
    "report_html_page",
)

#: Every module the split is made of, the facade last.
SPLIT_MODULES = (
    *SHARED_LAYERS,
    *(f"report_html_{kind}" for kind in REPORT_KINDS),
    "report_html",
)

#: What each shared helper is, and the layer that owns it. A helper drifting
#: back into the facade — or into a builder — is caught here.
OWNERS = {
    # escaping, value formatting and document normalizing
    "ReportError": "report_html_format",
    "_DASH": "report_html_format",
    "_esc": "report_html_format",
    "_pct": "report_html_format",
    "_secs": "report_html_format",
    "_ms": "report_html_format",
    "_usd": "report_html_format",
    "_rps": "report_html_format",
    "_int": "report_html_format",
    "_truncate": "report_html_format",
    "_fraction": "report_html_format",
    "_mapping": "report_html_format",
    "_number": "report_html_format",
    "_sequence": "report_html_format",
    # shared page components
    "_SERIES_ROLES": "report_html_components",
    "_card": "report_html_components",
    "_cards": "report_html_components",
    "_table": "report_html_components",
    "_badge": "report_html_components",
    "_link": "report_html_components",
    "_bar_chart": "report_html_components",
    "_donut": "report_html_components",
    "_meter": "report_html_components",
    # the page shell
    "_css": "report_html_page",
    "_THEME_SCRIPT": "report_html_page",
    "_provenance_items": "report_html_page",
    "_page": "report_html_page",
    # the run card's own helpers travel with it
    "_children": "report_html_run",
    "_sort_key": "report_html_run",
    "_tool_steps": "report_html_run",
    "_LOCAL_ENGINES": "report_html_run",
    "_unpriced_label": "report_html_run",
    "_run_command": "report_html_run",
}

#: The facade holds kind detection, the refusals, and the entry points — and
#: nothing a report kind renders with.
FACADE_DEFINITIONS = {
    "REPORT_KINDS",
    "_KIND_KEYS",
    "_BODY_RENDERERS",
    "__all__",
    "detect_report_kind",
    "_require_kind_data",
    "build_html_report",
    "write_html_report",
    "load_result_document",
}

#: What each module is allowed to import from within the split. The formatting
#: layer is the bottom: it imports nothing else here.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "report_html_format": set(),
    "report_html_components": {"report_html_format"},
    "report_html_page": {"report_html_format"},
    **{
        f"report_html_{kind}": {"report_html_format", "report_html_components"}
        for kind in REPORT_KINDS
    },
    "report_html": set(SPLIT_MODULES) - {"report_html"},
}

#: Constructs that make a browser retrieve something when the page opens. A
#: report is written to disk and read from disk, so none of them may be emitted.
_FETCHING_MARKUP = (
    re.compile(r"<link\b", re.I),
    re.compile(r"<(?:img|iframe|embed|object|video|audio|source|track)\b", re.I),
    re.compile(r"<script\b[^>]*\bsrc\b", re.I),
    re.compile(r"@import", re.I),
    re.compile(r"url\(\s*(?![\"']?data:)", re.I),
    re.compile(r"//cdn", re.I),
    re.compile(r"""\bsrc\s*=\s*["']?https?:""", re.I),
)


def module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


def source_path(name: str) -> Path:
    return Path(inspect.getsourcefile(module(name)))


def top_level_names(name: str) -> set[str]:
    """Every name the module binds at module scope, imports excluded."""
    tree = ast.parse(source_path(name).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def emitted_strings(name: str) -> list[str]:
    """Every string literal the module can write into a page.

    Docstrings are prose about the module, not markup it emits, so they are
    excluded — otherwise describing what a report must not contain would read
    as containing it.
    """
    tree = ast.parse(source_path(name).read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _import_module_argument(node: ast.Call) -> str | None:
    """The literal module name a dynamic import call names, if it is one."""
    func = node.func
    dynamic = (
        (isinstance(func, ast.Attribute) and func.attr == "import_module")
        or (isinstance(func, ast.Name) and func.id in ("import_module", "__import__"))
    )
    if not dynamic or not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def imported_split_modules(name: str) -> set[str]:
    """The modules of this split that *name* imports, however it spells them.

    All four spellings count: ``from .report_html_x import y``,
    ``from . import report_html_x`` (and ``from effgen.ui import report_html_x``,
    which names the module in the alias rather than in the module path),
    ``import effgen.ui.report_html_x``, and a dynamic
    ``importlib.import_module("effgen.ui.report_html_x")``.
    """
    tree = ast.parse(source_path(name).read_text(encoding="utf-8"))
    found: set[str] = set()

    def note(dotted: str) -> None:
        tail = dotted.split(".")[-1]
        if tail in SPLIT_MODULES:
            found.add(tail)

    for node in ast.walk(tree):
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
# One module per report kind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_each_report_kind_is_built_in_its_own_module(kind):
    builder = _BODY_RENDERERS[kind]
    assert builder.__module__ == f"{PACKAGE}.report_html_{kind}"
    assert builder.__name__ == f"_{kind}_body"


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_a_kind_module_defines_exactly_one_body_builder(kind):
    defined = {n for n in top_level_names(f"report_html_{kind}") if n.endswith("_body")}
    assert defined == {f"_{kind}_body"}


def test_every_declared_kind_has_a_builder_and_no_builder_is_orphaned():
    assert set(_BODY_RENDERERS) == set(REPORT_KINDS)
    assert len(REPORT_KINDS) == len(set(REPORT_KINDS))


def test_no_builder_module_is_missing_from_the_split():
    for name in SPLIT_MODULES:
        assert source_path(name).exists()


# ---------------------------------------------------------------------------
# Ownership: one definition, in one layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "owner"), sorted(OWNERS.items()))
def test_shared_helper_is_defined_in_its_own_layer(name, owner):
    assert name in top_level_names(owner), f"{name} is not defined in {owner}"
    for other in SPLIT_MODULES:
        if other != owner:
            assert name not in top_level_names(other), (
                f"{name} is also defined in {other}; it must live in {owner} alone"
            )


def test_no_name_is_defined_in_two_modules_of_the_split():
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for name in SPLIT_MODULES:
        for defined in top_level_names(name):
            if defined in seen:
                clashes.append(f"{defined}: {seen[defined]} and {name}")
            else:
                seen[defined] = name
    assert clashes == [], f"defined more than once: {clashes}"


def test_the_facade_defines_only_kind_detection_and_the_entry_points():
    assert top_level_names("report_html") == FACADE_DEFINITIONS


# ---------------------------------------------------------------------------
# The facade stays the one import path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "owner"), sorted(OWNERS.items()))
def test_the_facade_re_exports_every_moved_name(name, owner):
    assert hasattr(facade, name), f"{name} is no longer reachable on report_html"
    assert getattr(facade, name) is getattr(module(owner), name)


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_the_facade_re_exports_every_body_builder(kind):
    name = f"_{kind}_body"
    assert getattr(facade, name) is getattr(module(f"report_html_{kind}"), name)


def test_the_public_surface_is_unchanged_by_the_split():
    assert facade.__all__ == [
        "REPORT_KINDS",
        "ReportError",
        "build_html_report",
        "detect_report_kind",
        "load_result_document",
        "write_html_report",
    ]
    for name in facade.__all__:
        assert hasattr(facade, name)


# ---------------------------------------------------------------------------
# The layers only import downwards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_a_module_imports_only_the_layers_below_it(name):
    imported = imported_split_modules(name)
    allowed = ALLOWED_IMPORTS[name]
    assert imported <= allowed, (
        f"{name} imports {sorted(imported - allowed)}, which is not below it"
    )


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_a_kind_module_imports_neither_the_facade_nor_another_kind(kind):
    imported = imported_split_modules(f"report_html_{kind}")
    assert "report_html" not in imported
    siblings = {f"report_html_{k}" for k in REPORT_KINDS} - {f"report_html_{kind}"}
    assert not (imported & siblings)


def test_the_formatting_layer_imports_no_report_module():
    assert imported_split_modules("report_html_format") == set()


def test_the_facade_imports_every_module_of_the_split():
    assert imported_split_modules("report_html") == set(SPLIT_MODULES) - {"report_html"}


# ---------------------------------------------------------------------------
# Self-containment, at the source rather than only in a rendered page
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_no_module_of_the_split_emits_a_construct_that_fetches(name):
    hits = [
        (pattern.pattern, text[:80])
        for text in emitted_strings(name)
        for pattern in _FETCHING_MARKUP
        if pattern.search(text)
    ]
    assert hits == [], (
        f"{name} emits markup that would make the page retrieve something: {hits}"
    )


@pytest.mark.parametrize("kind", REPORT_KINDS)
def test_the_stylesheet_and_script_are_inlined_once_per_document(kind):
    from tests.cli.test_html_reports import DOCS

    html = facade.build_html_report(DOCS[kind], kind=kind)
    assert html.count("<style>") == 1
    assert html.count("<script>") == 1
    assert "<link" not in html
