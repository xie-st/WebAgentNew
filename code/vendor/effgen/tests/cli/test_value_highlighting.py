"""A styled human line renders in the colours its own markup asks for.

A rich console runs a *value highlighter* over any string it prints: numbers,
brackets, paths, quoted words and repr-ish tokens are recoloured inside the line
regardless of the style the line already carries. On a styled line that splits
one message across several colours — ``effgen 0.3.2`` comes out with ``0.3`` in
cyan, the dot uncoloured and ``2`` in cyan again; ``(1.42 s, 3 tool calls)``
comes out with bold parentheses around cyan numbers; ``effgen sessions show
<id>`` reads ``<id>`` as a tag and renders it bold magenta.

The contract pinned here:

* every place that prints a **string** to a console states its highlighting,
  so what the reader sees is decided by the line's own markup and not by
  whichever token the highlighter recognised;
* **tables, panels and other renderables are untouched** — rich applies the
  highlighter to strings only, so their bytes do not depend on the setting;
* on a console with no colour at all (piped output, ``NO_COLOR``, ``--json``)
  nothing changes either way, because a style that is not written produces no
  bytes.

Two gates. The first drives the whole command surface enumerated from the
parser and compares the bytes it writes with the console's highlighter on and
off: identical means no human output depends on it. The second reads the source,
so a print added to a command the driver cannot reach is still covered.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "effgen"

SGR = re.compile(r"\x1b\[[0-9;]*m")

#: Offline-safe invocations that print real content, beyond the ``--help`` of
#: every command path. Kept in step with the surface driver in
#: ``tests/packaging/test_reduced_install``.
REAL_INVOCATIONS = [
    ["--version"],
    ["presets"],
    ["tools", "list"],
    ["tools", "info", "calculator"],
    ["models", "list"],
    ["models", "list", "--provider", "openai"],
    ["models", "list", "--free"],
    ["models", "status"],
    ["models", "info", "gpt-5-nano"],
    ["models", "browse", "--limit", "5"],
    ["config", "show"],
    ["examples", "list"],
    ["cost"],
    ["cost", "today"],
    ["cost", "by-provider"],
    ["prompts", "list"],
    ["sessions", "list"],
    ["runs", "list"],
    ["eval", "--suite", "list"],
    ["doctor"],
    ["top", "--once"],
    ["monitor", "--once"],
]


def _command_paths() -> list[list[str]]:
    """Every command path argparse knows about, read from the parser."""
    from effgen.cli._main import create_parser

    def walk(parser, prefix=()):
        out = [list(prefix)]
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    out += walk(sub, (*tuple(prefix), name))
        return out

    return walk(create_parser())


def _invocations() -> list[list[str]]:
    return [[*path, "--help"] for path in _command_paths()] + REAL_INVOCATIONS


def _run(argv: list[str]) -> str:
    """Run one command with both streams captured, and return what it wrote.

    The stream carries UTF-8, so the terminal surfaces draw what they would draw
    on a normal console rather than their ASCII stand-ins.
    """
    from effgen.cli._main import main as cli_main

    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict", newline="")
    argv_before = sys.argv
    sys.argv = ["effgen", *argv]
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            cli_main()
    except SystemExit:
        pass
    except BaseException as exc:  # noqa: BLE001 - recorded, then asserted on below
        stream.write(f"\n<<raised {type(exc).__name__}: {exc}>>")
    finally:
        sys.argv = argv_before
    stream.flush()
    return stream.buffer.getvalue().decode("utf-8", "replace")


@pytest.fixture
def coloured_console(monkeypatch):
    """Make every console this process builds write ANSI into the captured stream.

    Function-scoped on purpose: the suite-wide fixture that clears the
    colour-forcing variables runs before this one, so a higher-scoped setting
    would be undone before the test body ran.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLUMNS", "100")


@pytest.fixture
def highlighter_off(monkeypatch):
    """Build every themed console with its value highlighter disabled.

    A ``console.print`` that passes ``highlight=`` explicitly is unaffected; one
    that leaves the decision to the console changes, which is what the
    comparison detects.
    """
    from effgen.ui import theme

    real = theme._EffGenConsole

    class _NoHighlight(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("highlight", False)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(theme, "_EffGenConsole", _NoHighlight)


#: Driving the whole surface twice takes a few seconds; every test in this module
#: reads the same run.
_SURFACE: dict[str, tuple[str, str]] = {}


@pytest.fixture
def surface(coloured_console):
    """Drive every invocation twice, so a command that moves is visible as such."""
    if not _SURFACE:
        _SURFACE.update({" ".join(inv): (_run(inv), _run(inv)) for inv in _invocations()})
    return _SURFACE


def test_a_command_that_moves_between_runs_is_recognised(surface):
    """The comparison below leans on this: two runs of a fixed command agree.

    A dashboard that reads the clock or the host's GPUs does not, and is
    compared line by line instead of byte for byte.
    """
    assert surface["presets"][0] == surface["presets"][1]
    assert surface["tools list"][0] == surface["tools list"][1]


# --------------------------------------------------------------------------- #
# The whole command surface, with the highlighter on and off
# --------------------------------------------------------------------------- #
def test_the_surface_is_enumerated_from_the_parser(surface):
    """The driver covers every command argparse knows about, not a chosen subset."""
    driven = set(surface)
    help_paths = {name for name in driven if name.endswith("--help")}
    assert len(help_paths) == len(_command_paths())
    assert len(driven) > len(help_paths)


def test_something_is_actually_coloured(surface):
    """Guards the comparison: a run with no ANSI at all would compare equal."""
    assert sum(len(SGR.findall(a)) for a, _ in surface.values()) > 200


def test_no_human_line_depends_on_the_value_highlighter(
    surface, coloured_console, highlighter_off
):
    offenders = []
    for name, (first, second) in surface.items():
        argv = name.split(" ") if name else []
        without = _run(argv)
        if SGR.sub("", first) == SGR.sub("", without) == SGR.sub("", second):
            # Same text in every run, so any difference in bytes is colour.
            if without != first:
                offenders.append((name, first, without))
            continue
        # The command writes something that moves between runs — a clock, a GPU
        # reading, a host measurement — so whole runs cannot be compared. Its
        # colouring still cannot move, so every line whose text appears in both
        # runs must carry the same colour codes in both.
        if _lines_coloured_differently(first, without):
            offenders.append((name, first, without))

    report = "\n\n".join(
        f"effgen {name}\n{_colour_diff(a, b)}" for name, a, b in offenders[:6]
    )
    assert not offenders, (
        f"{len(offenders)} command(s) print a string whose colouring is decided by "
        f"the console's value highlighter rather than by the line's own markup. "
        f"Pass an explicit highlight= at the call site.\n\n{report}"
    )


def _by_text(output: str) -> dict[str, list[str]]:
    """Each line's colour codes, keyed on the line's text with the colour removed."""
    return {SGR.sub("", line): SGR.findall(line) for line in output.splitlines()}


def _lines_coloured_differently(with_highlighter: str, without: str) -> bool:
    left, right = _by_text(with_highlighter), _by_text(without)
    return any(left[text] != codes for text, codes in right.items() if text in left)


def _colour_diff(with_highlighter: str, without: str) -> str:
    """The first line whose colouring differs between the two runs."""
    left, right = _by_text(with_highlighter), _by_text(without)
    for text, codes in right.items():
        if text in left and left[text] != codes:
            return (
                f"  line:                 {text!r}\n"
                f"  with the highlighter: {left[text]}\n"
                f"  without it:           {codes}"
            )
    for one, two in zip(
        with_highlighter.splitlines(), without.splitlines(), strict=False
    ):
        if one != two:
            return f"  with the highlighter: {one!r}\n  without it:           {two!r}"
    return f"  with the highlighter: {with_highlighter[:200]!r}"


def test_a_command_that_prints_a_table_is_unchanged_either_way(
    surface, coloured_console, highlighter_off
):
    """Tables are data: rich highlights strings only, so their bytes do not move."""
    first, second = surface["tools list"]
    assert first == second
    assert "│" in first  # it really did draw a table
    assert "\x1b[" in first  # and coloured it
    assert _run(["tools", "list"]) == first


# --------------------------------------------------------------------------- #
# The source, so a print the driver cannot reach is covered too
# --------------------------------------------------------------------------- #
def _is_console_print(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "print":
        return False
    current, names = func.value, []
    while True:
        if isinstance(current, ast.Attribute):
            names.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Name):
            names.append(current.id)
            break
        elif isinstance(current, ast.Call):
            inner = current.func
            names.append(getattr(inner, "attr", getattr(inner, "id", "")))
            break
        else:
            break
    return any("console" in name.lower() for name in names)


#: Calls whose result is a string, for the purpose of reading an assignment.
_STRING_CALLS = frozenset({
    "ascii_fold", "str", "join", "format", "strip", "lstrip", "rstrip",
    "lower", "upper", "removeprefix", "removesuffix", "replace",
})

#: Annotations that say a name holds text.
_STRING_ANNOTATIONS = frozenset({"str", "str | None", "None | str", "Optional[str]"})


def _is_string_expression(node: ast.AST | None) -> bool:
    """Whether an expression evaluates to a string, as far as the source can say."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp):
        return _is_string_expression(node.left) or _is_string_expression(node.right)
    if isinstance(node, ast.IfExp):
        return _is_string_expression(node.body) or _is_string_expression(node.orelse)
    if isinstance(node, ast.BoolOp):
        return any(_is_string_expression(value) for value in node.values)
    if isinstance(node, ast.Call):
        return getattr(node.func, "id", getattr(node.func, "attr", "")) in _STRING_CALLS
    return False


def _annotated_as_text(annotation: ast.AST | None) -> bool:
    return annotation is not None and ast.unparse(annotation) in _STRING_ANNOTATIONS


def _text_names(scope: ast.AST) -> set[str]:
    """Names that hold text inside *scope*, read from its parameters and assignments.

    This is what lets the scan reach a print whose argument is a variable — the
    shape two of the original defects had, where a helper takes a formatted line
    and hands it straight to the console.
    """
    names: set[str] = set()
    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = scope.args
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            if _annotated_as_text(argument.annotation):
                names.add(argument.arg)
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and _is_string_expression(node.value):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _annotated_as_text(node.annotation) or _is_string_expression(node.value):
                names.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if _is_string_expression(node.value):
                names.add(node.target.id)
    return names


def _prints_a_string(node: ast.Call, text_names: set[str]) -> bool:
    """Whether the first argument is a string, as far as the source can say.

    A literal or an f-string is one for certain, and so is the ASCII fold, which
    takes markup and returns markup. A bare name counts when the enclosing
    function says it holds text. Anything else -- a table, a panel, a value the
    source cannot type -- is left to the driver above.
    """
    if not node.args:
        return False
    first = node.args[0]
    if _is_string_expression(first):
        return True
    return isinstance(first, ast.Name) and first.id in text_names


def string_print_sites(root: Path) -> list[tuple[str, int, str]]:
    """Every ``console.print`` of a string that states no ``highlight=``."""
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scopes = [tree] + [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        seen: set[int] = set()
        for scope in reversed(scopes):  # innermost function first
            text_names = _text_names(scope)
            for node in ast.walk(scope):
                if not (isinstance(node, ast.Call) and _is_console_print(node)):
                    continue
                if node.lineno in seen:
                    continue
                seen.add(node.lineno)
                if not _prints_a_string(node, text_names):
                    continue
                if any(keyword.arg == "highlight" for keyword in node.keywords):
                    continue
                offenders.append(
                    (str(path.relative_to(root.parent)), node.lineno,
                     ast.unparse(node).replace("\n", " ")[:120])
                )
    return sorted(offenders)


def test_every_printed_string_states_its_highlighting():
    offenders = string_print_sites(PACKAGE)
    report = "\n".join(f"  {path}:{line}: {src}" for path, line, src in offenders)
    assert not offenders, (
        f"{len(offenders)} console.print call site(s) print a string without saying "
        f"whether its values are highlighted. A styled human line takes "
        f"highlight=False; a line where recolouring values is wanted takes "
        f"highlight=True.\n{report}"
    )


def test_the_source_gate_reports_a_site_that_omits_it(tmp_path):
    """Plant a violation: the gate has to see one."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "clean.py").write_text(
        "def show(console, table):\n"
        "    console.print(table)\n"
        "    console.print('[bold]ready[/bold]', highlight=False)\n",
        encoding="utf-8",
    )
    assert string_print_sites(package) == []

    (package / "regressed.py").write_text(
        "def show(console, count):\n"
        "    console.print(f'[green]done[/green] {count} items')\n",
        encoding="utf-8",
    )
    found = string_print_sites(package)
    assert [(line, "regressed.py" in path) for path, line, _ in found] == [(2, True)]


def test_the_source_gate_reads_a_line_held_in_a_variable(tmp_path):
    """The two hardest sites passed the line through a helper, not a literal.

    A print whose argument is a name is covered when the enclosing function says
    the name holds text -- an annotated parameter, or an assignment from a
    formatted string.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "held.py").write_text(
        "def emit(console, line: str) -> None:\n"
        "    console.print(line)\n"
        "\n"
        "def summarise(console, count):\n"
        "    footer = f'[dim]{count} rows[/dim]'\n"
        "    console.print(footer)\n"
        "\n"
        "def tabulate(console, rows):\n"
        "    table = Table()\n"
        "    console.print(table)\n",
        encoding="utf-8",
    )
    assert [line for _path, line, _src in string_print_sites(package)] == [2, 6]
