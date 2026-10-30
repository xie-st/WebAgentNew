"""effGen local web dashboard.

Served at ``/dashboard`` by the API server.  Provides real-time panels:
- Live span stream (SSE endpoint ``/dashboard/spans``)
- Prometheus metrics summary
- Recent agent runs with token counts and cost
- SLO burn-rate indicators

The dashboard is a single-page application (HTML + JS + CSS) served from
``effgen/dashboard/static/``.  No build step is required, and every asset is
served locally — the page renders with no external network dependency (the
latency chart is drawn on a canvas, not loaded from a CDN).

Usage
-----
Start the effGen server with dev mode enabled::

    EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory --port 8080

Then open ``http://localhost:8080/dashboard`` in a browser.
"""
from __future__ import annotations

from pathlib import Path

STATIC_DIR: Path = Path(__file__).parent / "static"
