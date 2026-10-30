"""Source files ingest as text, and file-writing tools honor a workspace root.

- The document ingester reads common source-code and plain-text config files as
  UTF-8 text, so handing an agent a ``.py``/``.js``/``.sql``/... file works the
  same way a ``.txt`` file does.
- ``EFFGEN_WORKSPACE`` sets the default directory the file and shell tools read
  and write, so generated files land there instead of the caller's directory. A
  relative path resolves against that root, and path confinement still holds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from effgen.rag.ingest import DocumentIngester
from effgen.tools.builtin.file_ops import FileOperations


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("ext", [".py", ".js", ".ts", ".go", ".rs", ".java", ".sql", ".yaml"])
def test_source_files_ingest_as_code(tmp_path, ext):
    p = tmp_path / f"sample{ext}"
    p.write_text("value = 42\n", encoding="utf-8")
    chunks = DocumentIngester(show_progress=False).ingest(p)
    assert len(chunks) == 1
    assert "value = 42" in chunks[0].content
    assert chunks[0].metadata.get("type") == "code"
    assert chunks[0].metadata.get("language") == ext.lstrip(".")


def test_source_extensions_are_advertised():
    exts = DocumentIngester(show_progress=False).supported_extensions()
    for ext in (".py", ".js", ".sh", ".sql"):
        assert ext in exts
    # Genuinely unknown / binary document types stay unsupported.
    for ext in (".xyz", ".pptx"):
        assert ext not in exts


def test_unknown_extension_is_reported_not_dropped(tmp_path):
    ing = DocumentIngester(show_progress=False)
    p = tmp_path / "thing.xyz"
    p.write_text("hello", encoding="utf-8")
    chunks = ing.ingest(p)
    assert chunks == []
    assert ing.last_skipped
    assert "unsupported extension '.xyz'" in ing.last_skipped[0][1]


def test_workspace_routes_writes_and_holds_confinement(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(ws))

    fo = FileOperations()
    _run(fo.initialize())
    assert fo._allowed_directories == [ws.resolve()]

    # A relative path resolves against the workspace root.
    r = _run(fo.execute(operation="write", path="step1.py", content="print(1)", format="text"))
    assert r.success is True, r.error
    assert (ws / "step1.py").exists()

    r2 = _run(fo.execute(operation="read", path="step1.py", format="text"))
    assert r2.success is True
    assert r2.output["data"] == "print(1)"

    # Escaping the workspace is still refused, fail-closed.
    r3 = _run(fo.execute(operation="read", path="../../../../etc/passwd", format="text"))
    assert r3.success is False


def test_default_without_workspace_uses_current_directory(monkeypatch):
    monkeypatch.delenv("EFFGEN_WORKSPACE", raising=False)
    fo = FileOperations()
    assert fo._allowed_directories == [Path.cwd()]
