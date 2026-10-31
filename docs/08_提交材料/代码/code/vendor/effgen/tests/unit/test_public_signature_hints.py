"""Every public function signature in ``effgen`` carries complete type hints.

Walks the package source with ``ast`` and asserts that each public function or
method annotates every parameter and its return type. A definition keeps the
surrounding scope whatever statement encloses it, so the walk descends through
every compound statement — ``if``/``try`` (optional-dependency fallbacks),
``with``, ``for``, ``while`` and ``match`` alike — rather than through a chosen
few; functions nested inside other functions are local helpers, not public
surface, and are skipped.

**Every** dunder counts, not a chosen subset: a dunder is called by the language
itself, so it is reachable from any user's code whether or not it appears in an
API listing, and effGen ships ``py.typed``, which promises their signatures are
accurate too. An unannotated ``__str__`` or ``__post_init__`` degrades to
``Any`` in a user's type check exactly as an unannotated ``__call__`` would.
"""

from __future__ import annotations

import ast
from pathlib import Path

import effgen

PACKAGE_ROOT = Path(effgen.__file__).parent


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def _is_public(name: str) -> bool:
    return _is_dunder(name) or not name.startswith("_")


def _missing_hints(node: ast.FunctionDef | ast.AsyncFunctionDef, in_class: bool) -> list[str]:
    """Return a list of unannotated parameter names (plus ``return``)."""
    args = node.args
    params = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    decorators = {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
    if in_class and "staticmethod" not in decorators and params and params[0].arg in ("self", "cls"):
        params = params[1:]
    missing = [p.arg for p in params if p.annotation is None]
    if args.vararg is not None and args.vararg.annotation is None:
        missing.append("*" + args.vararg.arg)
    if args.kwarg is not None and args.kwarg.annotation is None:
        missing.append("**" + args.kwarg.arg)
    if node.returns is None:
        missing.append("return")
    return missing


def _nested_bodies(node: ast.AST) -> list[list[ast.stmt]]:
    """Return every statement list *node* holds, at any depth within itself.

    Derived from the node's own fields rather than from a list of statement
    types, so a definition placed inside a ``for``, ``while``, ``match`` or
    ``async with`` body is reached exactly as one inside an ``if`` is. The
    recursion into non-statement children is what picks up the bodies hanging
    off an ``except`` handler or a ``match`` case.
    """
    bodies: list[list[ast.stmt]] = []
    for field in node._fields:
        value = getattr(node, field, None)
        if not isinstance(value, list):
            continue
        if value and isinstance(value[0], ast.stmt):
            bodies.append(value)
        else:
            for item in value:
                if isinstance(item, ast.AST):
                    bodies.extend(_nested_bodies(item))
    return bodies


def _walk(body: list[ast.stmt], qualname: list[str], in_class: bool, rel: str, findings: list[str]) -> None:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _is_public(node.name):
                missing = _missing_hints(node, in_class)
                if missing:
                    name = ".".join(qualname + [node.name])
                    findings.append(f"{rel}:{node.lineno}: {name} missing {', '.join(missing)}")
            # Functions nested inside functions are local helpers, not API.
        elif isinstance(node, ast.ClassDef):
            if _is_public(node.name):
                _walk(node.body, qualname + [node.name], True, rel, findings)
        else:
            # Any other compound statement leaves the scope alone, so a
            # definition in its body is as public as one beside it.
            for sub in _nested_bodies(node):
                _walk(sub, qualname, in_class, rel, findings)


def test_public_signatures_fully_annotated() -> None:
    findings: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = str(path.relative_to(PACKAGE_ROOT.parent))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _walk(tree.body, [], False, rel, findings)
    assert not findings, (
        f"{len(findings)} public signature(s) missing type hints:\n" + "\n".join(findings)
    )


def test_gate_detects_missing_hints() -> None:
    """The walker itself flags an unannotated public def (self-test)."""
    src = (
        "class Thing:\n"
        "    def compute(self, x):\n"
        "        return x\n"
        "def helper(a, *b, **c):\n"
        "    pass\n"
    )
    findings: list[str] = []
    _walk(ast.parse(src).body, [], False, "sample.py", findings)
    assert len(findings) == 2
    assert "Thing.compute missing x, return" in findings[0]
    assert "helper missing a, *b, **c, return" in findings[1]


def test_gate_covers_every_dunder_not_a_chosen_subset() -> None:
    """Dunders outside any hand-picked list are checked too (self-test)."""
    src = (
        "class Thing:\n"
        "    def __str__(self):\n"
        "        return ''\n"
        "    def __post_init__(self):\n"
        "        pass\n"
        "    def __init_subclass__(cls, **kwargs):\n"
        "        pass\n"
        "def __getattr__(name):\n"
        "    raise AttributeError(name)\n"
    )
    findings: list[str] = []
    _walk(ast.parse(src).body, [], False, "sample.py", findings)
    flagged = {f.split(": ", 1)[1].split(" missing ")[0] for f in findings}
    assert flagged == {
        "Thing.__str__",
        "Thing.__post_init__",
        "Thing.__init_subclass__",
        "__getattr__",
    }, findings

    # A single-underscore private helper stays out of scope.
    private: list[str] = []
    _walk(ast.parse("def _helper(a):\n    pass\n").body, [], False, "s.py", private)
    assert private == []


def test_gate_reaches_a_definition_under_any_compound_statement() -> None:
    """Enclosing a def in a loop or a ``match`` does not hide it (self-test)."""
    enclosures = {
        "if": "if True:\n    def made(a):\n        pass\n",
        "else": "if True:\n    pass\nelse:\n    def made(a):\n        pass\n",
        "try": "try:\n    def made(a):\n        pass\nexcept Exception:\n    pass\n",
        "except": "try:\n    pass\nexcept Exception:\n    def made(a):\n        pass\n",
        "finally": "try:\n    pass\nfinally:\n    def made(a):\n        pass\n",
        "with": "with open('x') as f:\n    def made(a):\n        pass\n",
        "for": "for _i in ():\n    def made(a):\n        pass\n",
        "while": "while False:\n    def made(a):\n        pass\n",
        "match": "match 1:\n    case 1:\n        def made(a):\n            pass\n",
        "class in a loop": "for _i in ():\n    class C:\n        def made(self, a):\n            pass\n",
    }
    unreached = []
    for label, src in enclosures.items():
        findings: list[str] = []
        _walk(ast.parse(src).body, [], False, "sample.py", findings)
        if not findings:
            unreached.append(label)
    assert not unreached, f"the walk never descends into: {', '.join(unreached)}"

    # A def inside another def stays a local helper whatever encloses it.
    nested: list[str] = []
    _walk(
        ast.parse("def outer(a: int) -> None:\n    for _i in ():\n        def inner(b):\n            pass\n").body,
        [],
        False,
        "s.py",
        nested,
    )
    assert nested == []
