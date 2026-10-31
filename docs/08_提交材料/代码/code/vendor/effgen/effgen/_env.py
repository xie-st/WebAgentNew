"""Shared ``.env`` discovery and loading for the CLI and library callers.

The command-line tools load the nearest project ``.env`` before running so a
freshly installed user's first command works with no manual key export. The
same discovery is available to a script or notebook through
:func:`load_env`::

    from effgen import load_env
    load_env()  # picks up OPENAI_API_KEY / GROQ_API_KEY / ... from a nearby .env

so library code gets the same zero-setup key discovery the CLI has, instead of
failing with "API key not found" moments after ``effgen run`` worked.

Loading is non-overriding: a real environment variable always wins over a file,
and the search is skipped entirely when ``EFFGEN_NO_DOTENV=1`` (or
``EFFGEN_DOTENV=none``) so a deployed process only ever sees the environment its
orchestrator injected.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env", "env_search_paths", "dotenv_disabled"]


def dotenv_disabled() -> bool:
    """True when the ``.env`` filesystem search should be skipped entirely.

    Set ``EFFGEN_NO_DOTENV=1`` (or ``EFFGEN_DOTENV=none``) so a production
    process uses only the environment its orchestrator injected — a stray
    ``.env`` left in a deploy image or the current directory can otherwise
    supply provider keys nobody intended to expose to that process.
    """
    if os.environ.get("EFFGEN_NO_DOTENV", "0").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return (os.environ.get("EFFGEN_DOTENV") or "").strip().lower() == "none"


def env_search_paths() -> list[Path]:
    """The ordered list of ``.env`` locations that are loaded (earliest wins).

    Search order (documented so pip-installed users aren't surprised):
      0. Skipped entirely when ``EFFGEN_NO_DOTENV=1`` or ``EFFGEN_DOTENV=none``.
      1. ``$EFFGEN_DOTENV`` — explicit override, if set (and not ``"none"``).
      2. ``~/.effgen/.env`` — per-user effGen config.
      3. ``./.env`` and each parent directory up to the filesystem root — the
         nearest project ``.env`` to the current working directory.
    Values load non-overriding, so a real environment variable always wins over
    a file, and earlier files win over later ones.
    """
    if dotenv_disabled():
        return []
    paths: list[Path] = []
    override = os.environ.get("EFFGEN_DOTENV")
    if override:
        paths.append(Path(override))
    paths.append(Path.home() / ".effgen" / ".env")
    # Walk up from the cwd to find the nearest project .env (a checkout's repo
    # root, for example) instead of a confusing package-relative path.
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        paths.append(d / ".env")
    return paths


def load_env() -> list[str]:
    """Load ``.env`` files from the documented search paths (non-overriding).

    Reads the same locations the CLI loads on startup (see
    :func:`env_search_paths`), giving a script or notebook the CLI's zero-setup
    key discovery. Returns the list of paths actually loaded (for diagnostics);
    returns an empty list without touching the filesystem when the search is
    disabled (see :func:`dotenv_disabled`) or when ``python-dotenv`` is not
    installed. A value already present in the environment is never overwritten.
    """
    loaded: list[str] = []
    if dotenv_disabled():
        return loaded
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        return loaded
    seen: set[Path] = set()
    for ep in env_search_paths():
        try:
            rp = ep.resolve()
        except Exception:
            rp = ep
        if rp in seen:
            continue
        seen.add(rp)
        if ep.exists():
            _load_dotenv(ep, override=False)
            loaded.append(str(ep))
    return loaded
