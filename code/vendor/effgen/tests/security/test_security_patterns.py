"""Security-pattern *gate* test.

Scans the shipped ``effgen`` source for dangerous patterns so a hardened bug
class cannot silently return in a later change:

* un-gated ``pickle.load`` / ``pickle.loads`` / ``pickle.Unpickler`` outside the
  one explicitly-gated retrieval index loader;
* ``eval`` / ``exec`` outside the sandboxed REPL worker;
* raw ``urlopen`` / ``requests.get|post|put|head`` in built-in tools that bypass
  the shared SSRF guard (``tools/builtin/_net.py``);
* ``shell=True`` / ``create_subprocess_shell`` outside ``bash_tool``.

Each allowlist entry is a deliberate, reviewed exception. Adding a new dangerous
call without updating the allowlist (with justification) fails this test.

The final test plants a synthetic violation and asserts the detectors flag it,
so the gate itself is proven to work.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import effgen

PKG_ROOT = Path(effgen.__file__).resolve().parent


def _rel(path: Path) -> str:
    return path.relative_to(PKG_ROOT).as_posix()


def _py_files() -> list[Path]:
    return [p for p in PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


# Files allowed to contain each pattern (reviewed, justified exceptions).
PICKLE_ALLOW = {"tools/builtin/retrieval.py"}          # gated by allow_pickle + warning
EVAL_EXEC_ALLOW = {"tools/builtin/_repl_worker.py"}    # the sandboxed REPL worker
RAW_HTTP_ALLOW = {"tools/builtin/_net.py"}             # the shared SSRF guard itself
SHELL_ALLOW = {"tools/builtin/bash_tool.py"}           # the one shell tool (hardened)


# --------------------------------------------------------------------------- #
# Detectors (also exercised by the plant-and-detect self-test)                #
# --------------------------------------------------------------------------- #
def _find_pickle_load(tree: ast.AST) -> list[int]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("load", "loads", "Unpickler"):
                val = node.func.value
                if isinstance(val, ast.Name) and val.id == "pickle":
                    hits.append(node.lineno)
    return hits


def _find_eval_exec(tree: ast.AST) -> list[int]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("eval", "exec"):
                hits.append(node.lineno)
    return hits


_RAW_HTTP_RE = re.compile(
    r"(?<![\w.])urlopen\(|requests\.(?:get|post|put|head)\(|\.Session\(\)\.(?:get|post)\("
)


def _find_raw_http(source: str) -> list[int]:
    hits = []
    for i, line in enumerate(source.splitlines(), 1):
        if "safe_urlopen" in line or "safe_requests" in line or "def safe_" in line:
            continue
        if _RAW_HTTP_RE.search(line):
            hits.append(i)
    return hits


_SHELL_RE = re.compile(r"create_subprocess_shell|shell\s*=\s*True")


def _find_shell(source: str) -> list[int]:
    return [
        i for i, line in enumerate(source.splitlines(), 1) if _SHELL_RE.search(line)
    ]


# --------------------------------------------------------------------------- #
# Gate tests                                                                   #
# --------------------------------------------------------------------------- #
def test_no_ungated_pickle_load():
    offenders = []
    for path in _py_files():
        rel = _rel(path)
        if rel in PICKLE_ALLOW:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno in _find_pickle_load(tree):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"un-gated pickle.load outside allowlist: {offenders}"


def test_no_eval_or_exec_outside_repl_worker():
    offenders = []
    for path in _py_files():
        rel = _rel(path)
        if rel in EVAL_EXEC_ALLOW:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno in _find_eval_exec(tree):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"eval/exec outside the sandboxed REPL worker: {offenders}"


def test_no_raw_http_in_builtin_tools():
    tools_dir = PKG_ROOT / "tools" / "builtin"
    offenders = []
    for path in tools_dir.glob("*.py"):
        if _rel(path) in RAW_HTTP_ALLOW:
            continue
        for lineno in _find_raw_http(path.read_text(encoding="utf-8")):
            offenders.append(f"{_rel(path)}:{lineno}")
    assert not offenders, (
        "raw HTTP in builtin tools bypassing _net.py (use safe_urlopen / "
        f"safe_requests_get): {offenders}"
    )


def test_no_shell_outside_bash_tool():
    offenders = []
    for path in _py_files():
        rel = _rel(path)
        if rel in SHELL_ALLOW:
            continue
        for lineno in _find_shell(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"shell execution outside bash_tool: {offenders}"


# --------------------------------------------------------------------------- #
# Plant-and-detect: prove the detectors actually fire                          #
# --------------------------------------------------------------------------- #
def test_detectors_catch_planted_violations():
    pickle_src = "import pickle\npickle.load(open('x','rb'))\n"
    assert _find_pickle_load(ast.parse(pickle_src)) == [2]

    eval_src = "eval('1+1')\nexec('x=1')\n"
    assert _find_eval_exec(ast.parse(eval_src)) == [1, 2]

    http_src = "import requests\nrequests.get('http://x')\nurlopen(req)\n"
    assert _find_raw_http(http_src) == [2, 3]
    # the safe wrappers must NOT be flagged
    assert _find_raw_http("safe_urlopen(url)\nsafe_requests_get(requests, url)\n") == []

    shell_src = "subprocess.run(cmd, shell=True)\nawait create_subprocess_shell(c)\n"
    assert _find_shell(shell_src) == [1, 2]
