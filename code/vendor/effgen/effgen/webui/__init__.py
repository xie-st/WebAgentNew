"""Assets shared by effGen's web surfaces.

``effgen/webui/static/`` holds the files the dashboard and the playground both
load: the command palette and keyboard layer (``webui.js``) and its stylesheet
(``webui.css``). Each surface serves them from its own path — ``/dashboard/
webui.js`` and ``/playground/webui.js`` — so a shared asset follows the same
access rule as the page that loads it.
"""

from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

__all__ = ["STATIC_DIR"]
