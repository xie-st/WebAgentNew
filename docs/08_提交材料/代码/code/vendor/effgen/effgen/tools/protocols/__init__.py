"""
Protocol integrations for effGen.

This package provides implementations for various agent communication protocols:

- **MCP (Model Context Protocol)** — Anthropic's protocol for context and tool sharing.
  Round-tripped against the official MCP client (stdio + HTTP).

  - ``mcp_official``: built on the official MCP SDK (FastMCP). **Recommended** —
    use it whenever ``pip install "mcp[cli]"`` is available.
  - ``mcp``: a self-contained implementation with no external SDK dependency.
    Use it only when the official ``mcp`` package cannot be installed.

- **A2A (Agent-to-Agent)** — Google's agent-communication protocol.
  *Experimental*: effGen ships the client side (``A2AClient``, ``AgentCard``,
  auth handlers) and the wire-protocol/task model; there is no bundled A2A
  server. See ``effgen.tools.protocols.a2a``.

- **ACP (Agent Communication Protocol)** — IBM's agent-interoperability protocol.
  *Experimental*: client + server smoke-tested locally; not yet validated against
  the external BeeAI ecosystem. See ``effgen.tools.protocols.acp``.

HTTP transports bind to 127.0.0.1 by default and warn when exposed to the
network. Apply authentication before binding to a public address.
"""

# Imported on first use, not at package import. Each of these pulls in an
# external SDK, and one incompatible SDK used to take the whole package down
# with it: the day the MCP SDK published 2.0.0, ``mcp_official`` failed at
# import and collection of 2,284 unrelated unit tests failed with it. The blast
# radius was the eager import, not the feature. A protocol that cannot load now
# reports why when something reaches for it, and the rest of the package works.
#
# ``from effgen.tools.protocols import mcp_official`` still works: Python falls
# back to this module-level ``__getattr__`` when the attribute is absent.
from types import ModuleType

_PROTOCOL_MODULES = ("mcp", "mcp_official", "a2a", "acp")

__all__ = [
    "mcp",
    "mcp_official",
    "a2a",
    "acp",
]


def __getattr__(name: str) -> "ModuleType":
    """Import a protocol submodule on first use.

    Args:
        name: The submodule being reached for.

    Returns:
        The imported submodule.

    Raises:
        AttributeError: *name* is not a protocol this package ships.
        ImportError: The protocol's SDK is absent or incompatible. The message
            names the protocol and keeps the underlying reason, so the failure
            is attributable rather than arriving as a bare import error from a
            module the caller never named.
    """
    if name not in _PROTOCOL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(f".{name}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"The '{name}' protocol integration could not be loaded: {exc}. "
            f"Install or repair its SDK — see docs/tools/protocols.md — or use "
            f"another protocol; the rest of effgen.tools.protocols is unaffected."
        ) from exc
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """List the protocol submodules without importing them."""
    return sorted(set(globals()) | set(_PROTOCOL_MODULES))
