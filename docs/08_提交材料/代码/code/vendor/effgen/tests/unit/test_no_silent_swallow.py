"""Guard against silent exception swallowing across the library.

A bare ``except ...: pass`` that neither re-raises, narrows to a meaningful
recovery, logs, nor carries a human-readable justification hides real failures.
This gate scans the long-tail subsystems (``tools``, ``rag``, ``memory``,
``server``, ``utils``, ``observability``) and fails on any *unjustified*
pass-only handler — extending the earlier ``core``/``models`` sweep so the whole
package holds one bar: every silent swallow must say *why* it is safe.

A pass-only handler is **justified** when its ``except`` line carries a comment
that, after stripping lint/type directives (``noqa``/``pragma``/``type:``),
still contains a real reason (e.g. ``# best-effort cleanup``). Logging the
exception (``logger.debug(..., exc_info=True)``), re-raising, returning, or any
non-``pass`` recovery also counts as handled and is not flagged.
"""

from __future__ import annotations

import ast
import re
import tokenize
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2] / "effgen"
_SCOPE = ["tools", "rag", "memory", "server", "utils", "observability"]

# Directive tokens that do NOT count as a human justification on their own.
_DIRECTIVE_RE = re.compile(
    r"(noqa(:[\sA-Z0-9,]+)?|pragma:\s*no\s*cover|type:\s*ignore(\[[^\]]*\])?)",
    re.IGNORECASE,
)


def _line_comments(path: Path) -> dict[int, str]:
    """Map line number -> trailing comment text for a source file."""
    out: dict[int, str] = {}
    with open(path, "rb") as fh:
        try:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type == tokenize.COMMENT:
                    out[tok.start[0]] = tok.string
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
    return out


def _is_pass_only(handler: ast.ExceptHandler) -> bool:
    stmts = [
        s for s in handler.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return (not stmts) or all(isinstance(s, ast.Pass) for s in stmts)


def _has_real_reason(comment: str | None) -> bool:
    if not comment:
        return False
    text = comment.lstrip("#").strip()
    stripped = _DIRECTIVE_RE.sub("", text)
    # Drop leftover separators/codes; require some real words remain.
    stripped = re.sub(r"[-:,\s]+", " ", stripped).strip()
    return len(stripped) >= 3 and any(c.isalpha() for c in stripped)


def _unjustified_sites() -> list[str]:
    offenders: list[str] = []
    for sub in _SCOPE:
        base = _PKG / sub
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            comments = _line_comments(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                if not _is_pass_only(node):
                    continue
                # Justified if any line within the handler (the except line
                # through its body) carries a human-readable reason.
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                if any(
                    _has_real_reason(comments.get(ln))
                    for ln in range(node.lineno, end + 1)
                ):
                    continue
                rel = path.relative_to(_PKG.parent)
                offenders.append(f"{rel}:{node.lineno}")
    return sorted(offenders)


def test_no_unjustified_silent_swallow():
    offenders = _unjustified_sites()
    assert not offenders, (
        "Unjustified silent `except ...: pass` sites (add a re-raise, a "
        "logger.debug(..., exc_info=True), or a comment saying why swallowing is "
        "safe):\n  " + "\n  ".join(offenders)
    )
