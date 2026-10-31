"""The four tools the benchmarks need.

These are ports of the tools the paper's scripts gave every framework: a
calculator for the math sets, a Python runner for the algorithmic sets, a web
search for the agentic sets, and a lookup over the local knowledge database for
the retrieval sets.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import subprocess
import sys
import textwrap
from functools import lru_cache

from ..config import DATA_DIR
from .spec import ToolSpec, one_string_arg

# ---------------------------------------------------------------- calculator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "abs": abs, "round": round, "int": int, "float": float,
    "sum": sum, "min": min, "max": max, "pow": pow,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"unknown function {node.func.id!r}")
        return fn(*[_eval_node(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]
    raise ValueError("unsupported expression")


#: Trailing "= 9", "= ?" or a bare "=" a model writes when it states the sum it
#: expects rather than only the expression. The paper's own calculators were
#: given the same inputs; ours rejected them, which cost the columns that call
#: the tool most. Stripping this is not making the tool cleverer — the
#: expression to the left is what any calculator would evaluate.
_TRAILING_EQUALS_RE = re.compile(r"\s*=\s*[-+]?[\d.,\s?]*$")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression.

    Parsed with `ast`, not `eval`, so a model that writes something odd cannot
    reach the interpreter.

    Currency symbols, thousands separators and a trailing ``= <value>`` are
    removed first. Models write ``$50,000 / 20`` and ``16 - 3 - 4 = ?`` often
    enough that refusing them measured how tidily a framework reformats its
    arguments rather than whether it can do arithmetic: 33% of effGen 0.0.1's
    calls and 13% of effGen++'s failed this way, against 2% for a column that
    barely used the tool at all.
    """
    expr = str(expression).strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = expr.replace(",", "")
    # Currency markers carry no arithmetic meaning; "$80000 + $50000" is the
    # same sum as "80000 + 50000".
    expr = expr.replace("$", "").replace("£", "").replace("€", "")
    expr = _TRAILING_EQUALS_RE.sub("", expr).strip()
    if not expr:
        return "Error: empty expression"
    try:
        value = _eval_node(ast.parse(expr, mode="eval"))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# -------------------------------------------------------------- python_exec

PY_EXEC_TIMEOUT_S = 30


#: Driver that runs the model's code the way an interactive session would: the
#: program runs normally, and if its last statement is a bare expression, that
#: expression's value is echoed the way a REPL or a notebook cell echoes it.
#:
#: Without this the tool ran the code with `python -c`, where a trailing
#: `sum(numbers)` computes the answer and throws it away. Models write that
#: constantly — they have seen far more notebook transcripts than scripts — and
#: the tool answered "the program printed nothing", after which the model
#: usually gave up and answered from its head with a wrong number. On the
#: generated arithmetic set at 3B that was 41 of 79 wrong answers, 17 points of
#: a 25 point gap, and it was the single largest failure of any column that uses
#: this tool.
_PY_EXEC_DRIVER = """\
import ast, sys
source = sys.stdin.read()
tree = ast.parse(source)
tail = None
if tree.body and isinstance(tree.body[-1], ast.Expr):
    tail = tree.body.pop()
scope = {"__name__": "__main__"}
exec(compile(tree, "<code>", "exec"), scope)
if tail is not None:
    value = eval(compile(ast.Expression(tail.value), "<code>", "eval"), scope)
    if value is not None:
        print(repr(value) if isinstance(value, str) else value)
"""


def python_exec(code: str) -> str:
    """Run a short Python program and return what it printed.

    The value of a trailing bare expression is echoed too, as an interactive
    interpreter would, so `sum(numbers)` on the last line answers rather than
    returning nothing.

    Runs in a separate process with a timeout, so an infinite loop in generated
    code costs a few seconds instead of hanging the run.
    """
    source = textwrap.dedent(str(code)).strip()
    if not source:
        return "Error: no code given"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _PY_EXEC_DRIVER],
            input=source,
            capture_output=True,
            text=True,
            timeout=PY_EXEC_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"Error: code did not finish within {PY_EXEC_TIMEOUT_S}s"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"Error: {err[-2000:] or 'exited with code %d' % proc.returncode}"
    if not out:
        return "(the program printed nothing; print() the answer)"
    return out[:4000]


# --------------------------------------------------------------- web_search

WEB_SEARCH_RESULTS = 3


def web_search(query: str) -> str:
    """Search the web and return the top snippets."""
    q = str(query).strip()
    if not q:
        return "Error: empty query"
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older name
        except ImportError:
            return "Error: no search backend installed"
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(q, max_results=WEB_SEARCH_RESULTS))
    except Exception as exc:
        return f"Error: search failed ({exc})"
    if not hits:
        return "No results found."
    lines = []
    for i, h in enumerate(hits, 1):
        title = h.get("title", "")
        body = (h.get("body") or "")[:500]
        lines.append(f"[Result {i}] {title}\n{body}")
    return "\n\n".join(lines)


# --------------------------------------------------- knowledge_search (local)


@lru_cache(maxsize=1)
def _knowledge_tool():
    import importlib.util

    path = DATA_DIR / "knowledge_db" / "build_knowledge_db.py"
    spec = importlib.util.spec_from_file_location("effgen_bench_knowledge_db", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    tool = module.KnowledgeSearchTool()
    tool.load()
    return tool


def knowledge_search(query: str) -> str:
    """Look the query up in the local knowledge database.

    This is the stand-in for web search used by the retrieval benchmarks. The
    database is built from the ARC and CommonsenseQA train and validation splits
    with the test items filtered out, so it retrieves related knowledge without
    leaking the answer.
    """
    q = str(query).strip()
    if not q:
        return "Error: empty query"
    try:
        return _knowledge_tool().search_formatted(q, max_results=3)
    except FileNotFoundError:
        return (
            "Error: knowledge database missing. "
            "Build it with: python -m effgen_bench.cli prepare-data"
        )
    except Exception as exc:
        return f"Error: lookup failed ({exc})"


# ------------------------------------------------------------------ registry

TOOLS: dict[str, ToolSpec] = {
    "calculator": ToolSpec(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression and return the result. "
            "Use it for every calculation. "
            "Example input: (25 * 4) + 17 / 2"
        ),
        parameters=one_string_arg(
            "expression",
            "A single arithmetic expression, for example '12 * (3 + 4)'. "
            "No variables, no assignments, no text.",
        ),
        func=calculator,
    ),
    "python_exec": ToolSpec(
        name="python_exec",
        description=(
            "Run a short Python 3 program and return everything it printed. "
            "Use it to compute the answer instead of reasoning it out by hand. "
            "The program must print the final answer."
        ),
        parameters=one_string_arg(
            "code",
            "A complete Python 3 program. It must print the answer with print().",
        ),
        func=python_exec,
    ),
    "web_search": ToolSpec(
        name="web_search",
        description=(
            "Search the web and return the top result snippets. "
            "Use short, specific queries built from the key entities in the question."
        ),
        parameters=one_string_arg(
            "query", "A short search query, for example '2023 Nobel Prize physics winner'."
        ),
        func=web_search,
    ),
    "knowledge_search": ToolSpec(
        name="knowledge_search",
        description=(
            "Look a topic up in the reference knowledge base and return the top "
            "matching entries. Use it to check facts before answering."
        ),
        parameters=one_string_arg(
            "query", "A short query made of the key terms in the question."
        ),
        func=knowledge_search,
    ),
}


def get_tools(names) -> list[ToolSpec]:
    return [TOOLS[n] for n in names if n in TOOLS]
