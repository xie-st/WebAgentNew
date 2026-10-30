"""Compatibility shim for legacy ``setup.py`` invocations.

All packaging metadata — dependencies, optional extras, version, entry points,
package data — lives in ``pyproject.toml`` (PEP 621), which is the single source
of truth. This file exists only so tooling that still calls ``setup.py``
directly continues to work; it intentionally declares no metadata of its own so
there is nothing here to drift out of sync with ``pyproject.toml``.
"""

from setuptools import setup

setup()
