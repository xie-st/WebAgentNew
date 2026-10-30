"""Read and check the register of tests known to fail for a reason outside the tree.

A suite that is re-run until it goes green teaches nobody anything. The register
(``tests/flake_register.toml``) is the alternative: a test that can fail for a reason
the tree does not control is written down once, with the reason, the class of cause,
the part of the tree that has to deal with it, when it was first seen and the date the
entry stops being an answer.

Two things read this file. The scripts that drive the whole suite over several orders
use :func:`entry_for` to separate a known cause from a new one, so a run reports "one
known flake, zero new failures" rather than "one failure". A gate test uses
:func:`violations` so an entry cannot be added without a reason and an owner, cannot
outlive the test it names, and cannot sit past its expiry date.
"""

from __future__ import annotations

import ast
import datetime as _datetime
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TESTS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_ROOT.parent
REGISTER_PATH = TESTS_ROOT / "flake_register.toml"

#: The causes an entry may name. A cause outside this set is a new class and needs a
#: deliberate addition here rather than a free-text label nobody can search for.
CAUSE_CLASSES = (
    "external-service",  # a third-party endpoint: quota, rate limit, outage, retirement
    "host-resource",  # something the host provides: a port, a driver, a system library
    "artifact-cache",  # a download the host may or may not already hold
    "timing",  # a deadline or scheduling assumption on a machine under load
)

#: A reason shorter than this is a label, not an explanation.
MIN_REASON_CHARS = 40

REQUIRED_FIELDS = ("id", "cause", "reason", "owner", "first_seen", "expires")


@dataclass(frozen=True)
class FlakeEntry:
    """One registered test, as read from the register."""

    id: str
    cause: str
    reason: str
    owner: str
    first_seen: str
    expires: str

    @property
    def module(self) -> str:
        return self.id.split("::", 1)[0]


def _as_entry(raw: dict[str, Any]) -> FlakeEntry:
    return FlakeEntry(
        id=str(raw.get("id", "")),
        cause=str(raw.get("cause", "")),
        reason=str(raw.get("reason", "")),
        owner=str(raw.get("owner", "")),
        first_seen=str(raw.get("first_seen", "")),
        expires=str(raw.get("expires", "")),
    )


def load(path: Path | None = None) -> list[FlakeEntry]:
    """Read the register. An absent file is an empty register, not an error."""
    location = Path(path) if path is not None else REGISTER_PATH
    if not location.exists():
        return []
    document = tomllib.loads(location.read_text(encoding="utf-8"))
    return [_as_entry(raw) for raw in document.get("flake", [])]


def raw_entries(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the register without shaping the entries, so missing keys stay missing."""
    location = Path(path) if path is not None else REGISTER_PATH
    if not location.exists():
        return []
    document = tomllib.loads(location.read_text(encoding="utf-8"))
    return list(document.get("flake", []))


def entry_for(nodeid: str, entries: list[FlakeEntry] | None = None) -> FlakeEntry | None:
    """Return the entry covering a node id, matching the whole id or its module."""
    registered = load() if entries is None else entries
    node = str(nodeid).replace("\\", "/")
    if not node.startswith("tests/"):
        node = f"tests/{node.lstrip('/')}" if "/" in node else node
    for entry in registered:
        if entry.id == node:
            return entry
        if entry.id.endswith(".py") and node.startswith(entry.id + "::"):
            return entry
    return None


def declared_test_names(module: Path) -> set[str]:
    """Every test name a module declares, as ``name`` and ``Class::name``."""
    try:
        tree = ast.parse(Path(module).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    names.add(f"{node.name}::{child.name}")
    return names


def _selector(nodeid: str) -> str:
    """The part of a node id after the module, with any parametrisation removed."""
    if "::" not in nodeid:
        return ""
    tail = nodeid.split("::", 1)[1]
    head, sep, _ = tail.partition("[")
    return head if sep else tail


def _parse_date(value: str) -> _datetime.date | None:
    try:
        return _datetime.date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def violations(
    path: Path | None = None,
    repo_root: Path | None = None,
    today: _datetime.date | None = None,
    known_ids: set[str] | None = None,
) -> list[str]:
    """Return one message per problem with the register; an empty list means it is sound.

    ``known_ids`` is the set of node ids the tree currently collects. When it is given,
    an entry naming a test that no longer exists is reported, so the register cannot
    outlive the tests it describes.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    day = today or _datetime.date.today()
    found: list[str] = []

    for index, raw in enumerate(raw_entries(path)):
        label = raw.get("id") or f"entry #{index + 1}"
        for field in REQUIRED_FIELDS:
            if field not in raw or not str(raw[field]).strip():
                found.append(f"{label}: missing required field '{field}'")
        entry = _as_entry(raw)

        if entry.cause and entry.cause not in CAUSE_CLASSES:
            found.append(
                f"{label}: cause '{entry.cause}' is not one of {', '.join(CAUSE_CLASSES)}"
            )
        if entry.reason and len(entry.reason.strip()) < MIN_REASON_CHARS:
            found.append(
                f"{label}: reason is {len(entry.reason.strip())} characters; "
                f"at least {MIN_REASON_CHARS} are needed to explain a cause"
            )
        if entry.owner and not (root / entry.owner).exists():
            found.append(f"{label}: owner '{entry.owner}' is not a path in the repository")
        module_path = root / entry.module
        if entry.id and not module_path.exists():
            found.append(f"{label}: names '{entry.module}', which is not a file in the tree")
        elif entry.id:
            selector = _selector(entry.id)
            if selector and selector not in declared_test_names(module_path):
                found.append(
                    f"{label}: '{entry.module}' declares no test named '{selector}'"
                )

        first_seen = _parse_date(entry.first_seen)
        expires = _parse_date(entry.expires)
        if entry.first_seen and first_seen is None:
            found.append(f"{label}: first_seen '{entry.first_seen}' is not an ISO-8601 date")
        if entry.expires and expires is None:
            found.append(f"{label}: expires '{entry.expires}' is not an ISO-8601 date")
        if first_seen and expires and expires <= first_seen:
            found.append(f"{label}: expires {entry.expires} is not after first_seen {entry.first_seen}")
        if expires and expires < day:
            found.append(
                f"{label}: expired on {entry.expires}; re-measure the cause and either "
                f"renew the entry with what was seen or remove it"
            )
        if known_ids is not None and entry.id and entry.id not in known_ids:
            if not entry.id.endswith(".py"):
                found.append(f"{label}: no test with this id is collected from the tree")

    seen: set[str] = set()
    for entry in load(path):
        if entry.id in seen:
            found.append(f"{entry.id}: listed more than once")
        seen.add(entry.id)

    return found
