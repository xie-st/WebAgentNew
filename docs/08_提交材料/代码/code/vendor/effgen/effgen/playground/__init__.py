"""In-browser playground served by the effGen API server.

A self-contained page (``effgen/playground/static/``) mounted at ``/playground``
that drives the existing ``POST /v1/chat/completions`` endpoint: pick a model or
preset, type a prompt, optionally attach a tool, Run, and read the answer, the
token/cost stats, and the tool step trace — then copy the equivalent CLI or
Python. All assets are served locally (no external network); the page is behind
the same auth as the rest of the server and only becomes locally viewable when
an operator opts in via ``EFFGEN_PUBLIC_PLAYGROUND``.
"""

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

__all__ = ["STATIC_DIR"]
