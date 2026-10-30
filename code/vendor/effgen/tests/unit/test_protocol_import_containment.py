"""One incompatible protocol SDK must not take the package down with it.

``effgen/tools/protocols/__init__.py`` imported ``a2a``, ``acp``, ``mcp`` and
``mcp_official`` eagerly. The day the MCP SDK published 2.0.0 — which removes
``mcp.server.fastmcp``, the API the effGen MCP server is built on —
``mcp_official`` failed at import and collection of 2,284 unrelated unit tests
failed with it. The blast radius was the eager import, not the feature.

Each submodule is now imported on first use, and a protocol that cannot load
reports why when something reaches for it.
"""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def _fresh_protocols():
    for name in list(sys.modules):
        if name.startswith("effgen.tools.protocols"):
            del sys.modules[name]
    return importlib.import_module("effgen.tools.protocols")


def test_the_package_imports_without_importing_any_protocol():
    protocols = _fresh_protocols()
    loaded = [n for n in ("mcp", "mcp_official", "a2a", "acp") if n in vars(protocols)]
    assert loaded == [], f"imported eagerly: {loaded}"


@pytest.mark.parametrize("name", ["mcp", "mcp_official", "a2a", "acp"])
def test_each_protocol_still_imports_by_name(name):
    """The public spelling must keep working."""
    module = importlib.import_module(f"effgen.tools.protocols.{name}")
    assert module.__name__.endswith(name)
    protocols = importlib.import_module("effgen.tools.protocols")
    assert getattr(protocols, name).__name__.endswith(name)


def test_dir_lists_every_protocol_without_importing_them():
    protocols = _fresh_protocols()
    assert {"mcp", "mcp_official", "a2a", "acp"} <= set(dir(protocols))
    assert "mcp_official" not in vars(protocols)


def test_an_unknown_name_is_an_attribute_error():
    protocols = _fresh_protocols()
    with pytest.raises(AttributeError):
        _ = protocols.not_a_protocol


def test_a_broken_sdk_leaves_the_package_and_the_other_protocols_working(monkeypatch):
    """The property that matters: the damage stays inside the one protocol.

    ``mcp_official`` already degrades when the SDK is merely *absent* — it warns
    and leaves its names unbound. What it did not survive was an SDK that is
    present but incompatible: 2.0.0 removes ``mcp.server.fastmcp``, and the
    ``ImportError`` from a submodule travelled out through the eager package
    import. With the import deferred, reaching for the broken protocol is what
    fails, and nothing else does.
    """
    real_import = builtins.__import__

    def _refuse_mcp_sdk(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("cannot import name 'FastMCP' from 'mcp.server.fastmcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _refuse_mcp_sdk)

    # The package itself imports — this is what used to fail, taking every
    # unrelated test with it.
    protocols = _fresh_protocols()
    assert "mcp_official" not in vars(protocols)

    # Reaching for the broken protocol fails, and names it.
    with pytest.raises((ImportError, Warning)) as excinfo:
        _ = protocols.mcp_official
    assert "mcp" in str(excinfo.value).lower()

    # Another protocol still loads.
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert protocols.a2a.__name__.endswith("a2a")
