"""The allowlist: the only source of experience entries a run may load.

The allowlist is a local file named by configuration. It names each approved
entry by id, version and a path relative to the allowlist's own directory. It is
the complete set: a file that exists on disk but is not named here is never read,
which is what "only allowlist entries load" means in practice.

Entry paths are contained on purpose. A relative path that resolves outside the
allowlist directory is refused, so a local file cannot silently point the loader
at an arbitrary path or at page-derived data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import (
    ALLOWLIST_SCHEMA,
    ExperienceFormatError,
    ExperiencePathError,
    _check_keys,
    _expect_text,
    _expect_version,
)

ENTRY_SUFFIXES = frozenset({".yaml", ".yml", ".json"})


@dataclass(frozen=True)
class AllowlistItem:
    """One approved (id, version, relative entry path) triple."""

    id: str
    version: int
    path: str


@dataclass(frozen=True)
class Allowlist:
    """A parsed allowlist and the directory its relative paths resolve in."""

    source: Path
    items: tuple[AllowlistItem, ...]


def _read_data(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ExperienceFormatError(
            f"invalid allowlist syntax in {path.name}: {exc}"
        ) from exc


def load_allowlist(path: Path) -> Allowlist:
    """Read and validate one allowlist file.

    A missing file raises ``OSError`` and malformed content raises
    :class:`ExperienceFormatError`; the loader records either and fails closed.
    """

    if not isinstance(path, Path):
        path = Path(path)
    data = _read_data(path)
    mapping = data
    if not isinstance(mapping, Mapping):
        raise ExperienceFormatError("allowlist root must be a mapping")
    _check_keys(mapping, {"schema", "entries"}, "allowlist")
    schema = mapping.get("schema", ALLOWLIST_SCHEMA)
    if schema != ALLOWLIST_SCHEMA:
        raise ExperienceFormatError(
            f"unsupported allowlist schema {schema!r}; expected {ALLOWLIST_SCHEMA!r}"
        )
    raw_entries = mapping.get("entries")
    if not isinstance(raw_entries, (list, tuple)):
        raise ExperienceFormatError("allowlist.entries must be a list")
    items: list[AllowlistItem] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        where = f"allowlist.entries[{index}]"
        if not isinstance(raw, Mapping):
            raise ExperienceFormatError(f"{where} must be a mapping")
        _check_keys(raw, {"id", "version", "path"}, where)
        item = AllowlistItem(
            id=_expect_text(raw.get("id"), f"{where}.id"),
            version=_expect_version(raw.get("version"), f"{where}.version"),
            path=_expect_text(raw.get("path"), f"{where}.path"),
        )
        if item.id in seen_ids:
            raise ExperienceFormatError(
                f"{where}.id {item.id!r} is listed more than once"
            )
        seen_ids.add(item.id)
        items.append(item)
    return Allowlist(source=path, items=tuple(items))


def resolve_entry_path(root: Path, relative: str) -> Path:
    """Resolve an allowlisted relative path and keep it under ``root``."""

    raw = Path(relative)
    if raw.is_absolute():
        raise ExperiencePathError(
            f"entry path {relative!r} must be relative to the allowlist directory",
            reason="absolute_path",
        )
    if raw.suffix.lower() not in ENTRY_SUFFIXES:
        raise ExperiencePathError(
            f"entry path {relative!r} must end with one of {sorted(ENTRY_SUFFIXES)}",
            reason="unsupported_suffix",
        )
    root_resolved = root.resolve()
    candidate = (root_resolved / raw).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ExperiencePathError(
            f"entry path {relative!r} resolves outside {root_resolved}",
            reason="path_outside_allowlist",
        ) from None
    return candidate


__all__ = [
    "Allowlist",
    "AllowlistItem",
    "ENTRY_SUFFIXES",
    "load_allowlist",
    "resolve_entry_path",
]
