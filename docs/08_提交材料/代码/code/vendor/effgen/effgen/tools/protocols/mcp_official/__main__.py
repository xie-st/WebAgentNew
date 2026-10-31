"""Package entry point for the official effGen MCP server.

Run with::

    python -m effgen.tools.protocols.mcp_official [stdio|http|sse] [host] [port]

Launching the package (rather than ``...mcp_official.server``) avoids the
``runpy`` double-import warning that ``python -m <pkg>.<module>`` emits when the
module has already been imported as part of its package — so the stdio transport
starts with a clean stderr, which matters for clients that surface server logs.
"""

from __future__ import annotations

import sys

from .server import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
