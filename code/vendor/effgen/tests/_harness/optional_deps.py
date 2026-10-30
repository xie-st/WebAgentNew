"""Turn "this optional feature is not installed" into a skip, not a failure.

Install effGen with ``.[dev]`` — the documented contributor path — and about 94
tests fail. Every one of them is a tool whose optional dependency was never
installed. The library is right throughout: each tool raises a typed, actionable
``ImportError`` naming the extra. What is wrong is the *test's* response, which
treats a correctly installed environment as broken and buries any real
regression in ninety expected ones.

Rather than editing ninety call sites (and needing a new edit for every test
added afterwards), the decision is made once, here, and applied by a report hook
in ``tests/conftest.py``.

**The rule is deliberately narrow**, because a hook that turns failures into
skips can hide real breakage:

* only a module named by one of the project's own **optional** extras counts —
  the list is read from ``pyproject.toml`` at import time, never hand-written,
  so a dependency that moves into the core requirements stops qualifying on its
  own and its failures go back to being failures;
* only the two shapes that mean "not installed" count: Python's
  ``ModuleNotFoundError: No module named 'x'``, and the sentence effGen's own
  tools raise (``x is required: pip install x`` / ``x is not installed``), which
  reaches a test as an assertion on a ``ToolResult`` rather than as an
  exception;
* the skip reason names the extra to install, so the run says which feature was
  not exercised rather than going quiet.

An environment with the extras installed sees no change at all: nothing raises
these, so nothing is converted.
"""
from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Distribution name -> import name, where the two differ. Everything else is
# matched on its distribution name with '-' normalized to '_'.
_IMPORT_NAMES = {
    "youtube-transcript-api": "youtube_transcript_api",
    "yt-dlp": "yt_dlp",
    "sentence-transformers": "sentence_transformers",
    "beautifulsoup4": "bs4",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "pillow": "PIL",
    "faster-whisper": "faster_whisper",
    "duckduckgo-search": "duckduckgo_search",
    "google-genai": "google",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
}


def _requirement_name(spec: str) -> str:
    """The bare distribution name from a requirement string."""
    return re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0].strip().lower()


@lru_cache(maxsize=1)
def optional_dependency_names() -> frozenset[str]:
    """Every module name that belongs to an optional extra and not to the core.

    Read from ``pyproject.toml`` so the set cannot drift: a package promoted
    into the core requirements drops out of this set, and its absence becomes a
    failure again rather than a skip.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    core = {_requirement_name(spec) for spec in project.get("dependencies", [])}
    extras = project.get("optional-dependencies", {})
    # ``dev`` is the contributor install: anything in it is expected to be
    # present, so its absence is a real problem and must not be softened.
    core |= {
        _requirement_name(spec)
        for spec in extras.get("dev", [])
    }

    names: set[str] = set()
    for name, specs in extras.items():
        if name == "dev":
            continue
        for spec in specs:
            dist = _requirement_name(spec)
            if not dist or dist in core or dist.startswith("effgen"):
                continue
            names.add(dist)
            names.add(dist.replace("-", "_"))
            mapped = _IMPORT_NAMES.get(dist)
            if mapped:
                names.add(mapped)
    return frozenset(names)


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_NOT_INSTALLED_RE = re.compile(
    r"([\w.-]+) (?:is required|is not installed|not installed)", re.IGNORECASE
)


def absent_optional_dependency(text: str) -> str | None:
    """Return the optional package a failure blames, or ``None``.

    Args:
        text: The rendered failure — traceback, assertion message, or both.

    Returns:
        The distribution name to install, or ``None`` when the failure is not
        about an absent optional dependency.
    """
    if not text:
        return None
    known = optional_dependency_names()
    for match in _MISSING_MODULE_RE.finditer(text):
        top = match.group(1).split(".")[0]
        if top in known or top.replace("_", "-") in known:
            return top
    for match in _NOT_INSTALLED_RE.finditer(text):
        name = match.group(1)
        if name in known or name.replace("-", "_") in known:
            return name
    return None
