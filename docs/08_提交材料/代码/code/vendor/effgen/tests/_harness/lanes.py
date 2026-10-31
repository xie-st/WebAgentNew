"""Which lane a test belongs to.

A *lane* is the top-level directory under ``tests/`` that holds a test module —
``unit``, ``cli``, ``server`` and so on. Lanes are the unit CI schedules jobs in and
the unit per-lane timing is reported in, so both the timing plugin and the scripts
that drive the suite resolve them through this module rather than re-splitting node
ids by hand.
"""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent.parent

#: Lane name for a test module that sits directly in ``tests/``.
ROOT_LANE = "(root)"


def lane_of_nodeid(nodeid: str) -> str:
    """Return the lane a pytest node id belongs to.

    ``tests/unit/test_x.py::test_y`` and ``unit/test_x.py::test_y`` both resolve to
    ``unit``; a module directly under ``tests/`` resolves to :data:`ROOT_LANE`.
    """
    head = str(nodeid).split("::", 1)[0].replace("\\", "/")
    parts = [p for p in head.split("/") if p and p != "."]
    if parts and parts[0] == "tests":
        parts = parts[1:]
    if len(parts) > 1:
        return parts[0]
    return ROOT_LANE


def lane_of_path(path: str | Path) -> str:
    """Return the lane a filesystem path belongs to."""
    resolved = Path(str(path)).resolve()
    try:
        relative = resolved.relative_to(TESTS_ROOT)
    except ValueError:
        return ROOT_LANE
    if len(relative.parts) > 1:
        return relative.parts[0]
    return ROOT_LANE


def lanes_with_tests(tests_root: Path | None = None) -> list[str]:
    """Return every lane under ``tests/`` that contains at least one test module."""
    root = Path(tests_root) if tests_root is not None else TESTS_ROOT
    found: set[str] = set()
    for module in root.rglob("test_*.py"):
        found.add(lane_of_path(module))
    for module in root.rglob("*_test.py"):
        found.add(lane_of_path(module))
    return sorted(found)
