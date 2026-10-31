"""Every module and public definition in ``effgen`` carries a docstring.

Walks the package source with ``ast`` and asserts that every module, and every
public class/function/method (name not starting with ``_``), has a docstring
whose first line stands alone as a sentence. Conditionally defined objects
(inside ``if``/``try``/``with`` blocks, e.g. optional-dependency fallbacks) are
included; definitions nested inside functions are local helpers and are
skipped.

A signature that is not self-evident carries more than a summary. A callable
taking ``_SELF_EVIDENT_ARITY`` or more documentable parameters (``self`` and
``cls`` do not count) must name each of them in its docstring, and must say what
it hands back when its return annotation is anything other than ``None``.
Properties and ``typing.overload`` stubs are exempt: the first take no
arguments, the second document themselves through the implementation they
precede. Sample code inside an ``Example`` block does not count as describing
the result: a ``return`` statement shown to illustrate usage says nothing about
what the documented callable itself hands back.

Sections are also read in order. Google-style docstrings put ``Args`` before
``Returns``/``Yields`` and both before ``Raises``, so a reader meets the inputs,
the result and then the failures; a block that appears out of that order is
reported.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import effgen

PACKAGE_ROOT = Path(effgen.__file__).parent

# A summary line reads as a sentence when it ends in one of these.
_TERMINATORS = (".", "!", "?", ":")

# Below this many documentable parameters, the names and annotations in the
# signature say everything a caller needs, and a summary line is enough.
_SELF_EVIDENT_ARITY = 3

# Any of these, in a docstring's own prose, counts as describing the result.
_RETURN_HINTS = ("returns", "return ", ":return", "yields", "yield ", "->")

_SIMPLE_NONE = ("None", "'None'", '"None"')

# Sample code is illustration, not documentation: a ``return`` inside one says
# nothing about what the documented callable hands back.
_EXAMPLE_HEADING = re.compile(r"^\s*(Example|Examples|Usage)s?\s*:\s*$")

# Sections read in this order: inputs, then the result, then the failures.
_SECTION_HEADING = re.compile(
    r"^\s*(Args|Arguments|Parameters|Returns|Yields|Raises)\s*:\s*$"
)
_SECTION_RANK = {
    "args": 0, "arguments": 0, "parameters": 0,
    "returns": 1, "yields": 1,
    "raises": 2,
}


def _summary_finding(doc: str, where: str) -> str | None:
    """Return a style finding for *doc*'s summary line, or ``None`` if it reads well."""
    lines = doc.strip().splitlines()
    if not lines:
        return f"{where}: empty docstring"
    first = lines[0].strip()
    if first.endswith(_TERMINATORS):
        return None
    if len(lines) > 1 and lines[1].strip():
        return f"{where}: summary wraps onto line 2: {first[:70]!r}"
    return f"{where}: summary has no terminal punctuation: {first[:70]!r}"


def _documentable_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the parameter names of *node* that a caller has to be told about."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args)]
    if args.vararg:
        names.append(args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(args.kwarg.arg)
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names


def _returns_a_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return ``True`` when *node* is annotated with a return type other than ``None``."""
    if node.returns is None:
        return False
    return ast.unparse(node.returns).strip() not in _SIMPLE_NONE


def _is_exempt(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return ``True`` for properties and overload stubs, which need no argument section."""
    for dec in node.decorator_list:
        src = ast.unparse(dec)
        if "overload" in src or "property" in src or src.endswith((".setter", ".deleter")):
            return True
    return False


def _only_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return ``True`` when *node*'s body is a single ``raise``, so nothing comes back."""
    body = [
        stmt
        for stmt in node.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    return len(body) == 1 and isinstance(body[0], ast.Raise)


def _mentions(doc: str, name: str) -> bool:
    """Return ``True`` when *doc* uses *name* as a standalone word."""
    return re.search(rf"(^|[^A-Za-z0-9_]){re.escape(name)}([^A-Za-z0-9_]|$)", doc) is not None


def _prose(doc: str) -> str:
    """Return *doc* without its sample code, which illustrates but does not document.

    Three shapes of sample code are dropped: a block under an ``Example:``
    heading, a doctest prompt, and a literal block opened by ``::`` — the last
    because a signature written inside one carries an arrow that would otherwise
    read as a documented result.

    Args:
        doc: The docstring to read.

    Returns:
        The docstring's remaining lines, joined, with sample code removed.
    """
    kept: list[str] = []
    heading_indent: int | None = None
    literal_indent: int | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if _EXAMPLE_HEADING.match(line):
            heading_indent = indent
            continue
        if heading_indent is not None:
            if not stripped:
                continue
            if indent > heading_indent:
                continue
            heading_indent = None
        if literal_indent is not None:
            if not stripped:
                continue
            if indent > literal_indent:
                continue
            literal_indent = None
        if stripped.startswith((">>>", "...")):
            continue
        if stripped.endswith("::"):
            # A reST literal block: everything indented under this line is code.
            literal_indent = indent
            kept.append(line)
            continue
        kept.append(line)
    return "\n".join(kept)


def _section_order_finding(doc: str, where: str) -> str | None:
    """Return a finding when *doc*'s sections do not read inputs, result, failures."""
    seen = [
        m.group(1).lower()
        for m in (_SECTION_HEADING.match(line) for line in doc.splitlines())
        if m
    ]
    for earlier, later in zip(seen, seen[1:]):
        if _SECTION_RANK[earlier] > _SECTION_RANK[later]:
            return f"{where}: {later.title()} section comes after {earlier.title()}"
    return None


def _signature_findings(
    node: ast.FunctionDef | ast.AsyncFunctionDef, doc: str, where: str
) -> list[str]:
    """Return findings for parameters and a result *doc* leaves undocumented."""
    findings = []
    order = _section_order_finding(doc, where)
    if order:
        findings.append(order)
    if _is_exempt(node):
        return findings
    params = _documentable_params(node)
    if len(params) < _SELF_EVIDENT_ARITY:
        return findings
    undocumented = [p for p in params if not _mentions(doc, p)]
    if undocumented:
        findings.append(f"{where}: parameter(s) not documented: {', '.join(undocumented)}")
    if (
        _returns_a_value(node)
        and not _only_raises(node)
        and not any(h in _prose(doc).lower() for h in _RETURN_HINTS)
    ):
        findings.append(f"{where}: return value not documented")
    return findings


def _walk(
    body: list[ast.stmt],
    qualname: list[str],
    rel: str,
    missing: list[str],
    style: list[str],
) -> None:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                name = ".".join(qualname + [node.name])
                where = f"{rel}:{node.lineno}: {name}"
                doc = ast.get_docstring(node)
                if not doc:
                    missing.append(where)
                else:
                    finding = _summary_finding(doc, where)
                    if finding:
                        style.append(finding)
                    style.extend(_signature_findings(node, doc, where))
            # Definitions nested inside functions are local helpers, not API.
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                name = ".".join(qualname + [node.name])
                doc = ast.get_docstring(node)
                if not doc:
                    missing.append(f"{rel}:{node.lineno}: {name}")
                else:
                    finding = _summary_finding(doc, f"{rel}:{node.lineno}: {name}")
                    if finding:
                        style.append(finding)
                _walk(node.body, qualname + [node.name], rel, missing, style)
        elif isinstance(node, ast.If):
            _walk(node.body, qualname, rel, missing, style)
            _walk(node.orelse, qualname, rel, missing, style)
        elif isinstance(node, ast.Try):
            for sub in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
                _walk(sub, qualname, rel, missing, style)
        elif isinstance(node, ast.With):
            _walk(node.body, qualname, rel, missing, style)


def test_every_module_and_public_definition_is_documented() -> None:
    missing_modules: list[str] = []
    missing: list[str] = []
    style: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = str(path.relative_to(PACKAGE_ROOT.parent))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing_modules.append(rel)
        _walk(tree.body, [], rel, missing, style)
    assert not missing_modules, (
        f"{len(missing_modules)} module(s) without a docstring:\n" + "\n".join(missing_modules)
    )
    assert not missing, (
        f"{len(missing)} public definition(s) without a docstring:\n" + "\n".join(missing)
    )
    assert not style, f"{len(style)} summary line(s) to reword:\n" + "\n".join(style)


def test_gate_detects_a_missing_and_a_wrapped_docstring() -> None:
    """The walker itself flags an undocumented def and a two-line summary (self-test)."""
    src = (
        "class Thing:\n"
        '    """Documented."""\n'
        "    def compute(self):\n"
        "        return 1\n"
        "def helper():\n"
        '    """A summary that keeps\n'
        '    going onto a second line."""\n'
    )
    missing: list[str] = []
    style: list[str] = []
    _walk(ast.parse(src).body, [], "sample.py", missing, style)
    assert len(missing) == 1 and "Thing.compute" in missing[0]
    assert len(style) == 1 and "wraps onto line 2" in style[0]


def test_gate_detects_undocumented_arguments_and_return() -> None:
    """A three-argument signature with a summary-only docstring is flagged (self-test)."""
    src = (
        "def render(rows, width, colour) -> str:\n"
        '    """Draw a table."""\n'
        "def keep(rows, width, colour) -> str:\n"
        '    """Draw a table.\n'
        "\n"
        "    Args:\n"
        "        rows: the rows.\n"
        "        width: the width.\n"
        "        colour: the colour.\n"
        "\n"
        "    Returns:\n"
        "        The rendered block.\n"
        '    """\n'
        "def pair(rows, width) -> str:\n"
        '    """Two arguments explain themselves."""\n'
    )
    missing: list[str] = []
    style: list[str] = []
    _walk(ast.parse(src).body, [], "sample.py", missing, style)
    assert not missing
    assert len(style) == 2, style
    assert "parameter(s) not documented: rows, width, colour" in style[0]
    assert "return value not documented" in style[1]


def test_gate_exempts_properties_and_overloads() -> None:
    """A property and an overload stub are not asked for an argument section (self-test)."""
    src = (
        "class Box:\n"
        '    """A box."""\n'
        "    @property\n"
        "    def size(self) -> int:\n"
        '        """The size."""\n'
        "    @overload\n"
        "    def get(self, a, b, c) -> str:\n"
        '        """Get a value."""\n'
    )
    missing: list[str] = []
    style: list[str] = []
    _walk(ast.parse(src).body, [], "sample.py", missing, style)
    assert not missing and not style


def test_sample_code_does_not_stand_in_for_a_return_description() -> None:
    """A ``return`` shown inside an Example block leaves the result undocumented (self-test)."""
    src = (
        "def decorate(name, schema, version) -> Callable:\n"
        '    """Register a handler.\n'
        "\n"
        "    Args:\n"
        "        name: the name.\n"
        "        schema: the schema.\n"
        "        version: the version.\n"
        "\n"
        "    Example:\n"
        "        @decorate(name='x', schema={}, version='1')\n"
        "        def handler(payload):\n"
        "            return payload\n"
        '    """\n'
    )
    missing: list[str] = []
    style: list[str] = []
    _walk(ast.parse(src).body, [], "sample.py", missing, style)
    assert not missing
    assert len(style) == 1, style
    assert "return value not documented" in style[0]


def test_gate_detects_sections_in_the_wrong_order() -> None:
    """Args after Yields and Returns after Raises are both reported (self-test)."""
    src = (
        "def stream(prompt, config, retries):\n"
        '    """Stream the answer.\n'
        "\n"
        "    Yields:\n"
        "        Successive chunks.\n"
        "\n"
        "    Args:\n"
        "        prompt: the prompt.\n"
        "        config: the config.\n"
        "        retries: the retries.\n"
        '    """\n'
        "def store(entry, table, ttl) -> int:\n"
        '    """Store the entry.\n'
        "\n"
        "    Args:\n"
        "        entry: the entry.\n"
        "        table: the table.\n"
        "        ttl: the ttl.\n"
        "\n"
        "    Raises:\n"
        "        ValueError: the entry is empty.\n"
        "\n"
        "    Returns:\n"
        "        The stored row id.\n"
        '    """\n'
    )
    missing: list[str] = []
    style: list[str] = []
    _walk(ast.parse(src).body, [], "sample.py", missing, style)
    assert not missing
    assert len(style) == 2, style
    assert "Args section comes after Yields" in style[0]
    assert "Returns section comes after Raises" in style[1]
