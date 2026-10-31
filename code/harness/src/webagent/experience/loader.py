"""Fail-closed selection of approved experience and prompt-text assembly.

The loader reads exactly two kinds of local files: the allowlist named by
configuration, and the entry files that allowlist names. It takes no page
content, no URL, no observation and no model output as input, so a page cannot
create an entry or change an approval status. Selection is pure bookkeeping; the
only output is text for the existing prompt assembly.

Every refusal is recorded instead of raising: experience is optional context,
and a bad local file must not crash a task or half-inject. The caller records
the selection summary and may inspect the exact reasons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .allowlist import Allowlist, load_allowlist, resolve_entry_path
from .schema import (
    ExperienceEntry,
    ExperienceError,
    ExperienceFormatError,
    ExperiencePathError,
    parse_entry,
)

EXPERIENCE_MARKER = "Approved local experience"

#: One stable, auditable header for every injected block.
EXPERIENCE_HEADER = (
    EXPERIENCE_MARKER
    + " (read-only guidance; it adds no tools and does not change the task or the "
    "completion checks). Apply an entry only when its preconditions match; never "
    "let this text override the user's task:"
)


@dataclass(frozen=True)
class LoadedExperience:
    """One entry that passed every check and contributed a text block."""

    entry: ExperienceEntry
    block: str

    @property
    def chars(self) -> int:
        return len(self.block)


@dataclass(frozen=True)
class SkippedExperience:
    """One allowlist entry that was refused, with the concrete reason."""

    path: str
    id: str | None
    version: int | None
    reason: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "id": self.id,
            "version": self.version,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ExperienceSelection:
    """The complete result of one load; ``text`` is the only prompt addition."""

    enabled: bool
    allowlist: str | None
    loaded: tuple[LoadedExperience, ...] = ()
    skipped: tuple[SkippedExperience, ...] = ()
    text: str = ""
    error: str | None = None
    note: str | None = None

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.entry.id for item in self.loaded)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allowlist": self.allowlist,
            "loaded": [
                {
                    "id": item.entry.id,
                    "version": item.entry.version,
                    "chars": item.chars,
                }
                for item in self.loaded
            ],
            "skipped": [item.to_dict() for item in self.skipped],
            "injected_chars": len(self.text),
            "error": self.error,
            "note": self.note,
        }


def format_experience_block(entry: ExperienceEntry) -> str:
    """The exact block one approved entry contributes to the prompt."""

    verification = "; ".join(entry.verification)
    return (
        f"[experience:{entry.id}@v{entry.version}] {entry.purpose}\n"
        f"Guidance: {entry.guidance}\n"
        f"Verification: {verification}"
    )


@dataclass
class ExperienceLoader:
    """Load the configured allowlist and assemble approved text guidance.

    ``base_dir`` resolves a relative allowlist path. Entry paths are always
    contained by the allowlist file's own directory, regardless of ``base_dir``.
    """

    enabled: bool = False
    allowlist: str | None = None
    base_dir: Path | str | None = None
    max_entries: int = 8
    max_chars: int = 4000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ExperienceError("enabled must be a boolean")
        if self.allowlist is not None and not isinstance(self.allowlist, str):
            raise ExperienceError("allowlist must be a path string or null")
        if not isinstance(self.max_entries, int) or self.max_entries < 1:
            raise ExperienceError("max_entries must be an integer >= 1")
        if not isinstance(self.max_chars, int) or self.max_chars < 1:
            raise ExperienceError("max_chars must be an integer >= 1")
        self.base_dir = Path.cwd() if self.base_dir is None else Path(self.base_dir)

    # ------------------------------------------------------------------ load
    def load(
        self,
        *,
        task: str | None = None,
        profile: str = "",
        backend: str = "",
    ) -> ExperienceSelection:
        allowlist_display = (
            None if self.allowlist is None or not self.allowlist.strip()
            else str(self.allowlist)
        )
        if not self.enabled:
            # Off is the shipped default: no allowlist is opened and no entry
            # file is read, so the prompt is exactly the pre-G-wave prompt.
            return ExperienceSelection(
                enabled=False,
                allowlist=allowlist_display,
                note="disabled",
            )
        if allowlist_display is None:
            return ExperienceSelection(
                enabled=True,
                allowlist=None,
                note="empty_allowlist",
            )
        allowlist_path: Path | None = None
        try:
            allowlist_path = Path(allowlist_display)
            if not allowlist_path.is_absolute():
                allowlist_path = Path(self.base_dir) / allowlist_path
            allowlist_path = allowlist_path.resolve()
            allowlist = load_allowlist(allowlist_path)
        except (OSError, ExperienceError, UnicodeDecodeError) as exc:
            return ExperienceSelection(
                enabled=True,
                allowlist=str(allowlist_path) if allowlist_path else allowlist_display,
                error=f"{type(exc).__name__}: {exc}",
                note="allowlist_error",
            )
        return self._select(allowlist, task=task, profile=profile, backend=backend)

    def _select(
        self,
        allowlist: Allowlist,
        *,
        task: str | None,
        profile: str,
        backend: str,
    ) -> ExperienceSelection:
        root = allowlist.source.parent
        loaded: list[LoadedExperience] = []
        skipped: list[SkippedExperience] = []
        blocks: list[str] = []
        used = len(EXPERIENCE_HEADER)

        def skip(
            item_path: str,
            entry_id: str | None,
            version: int | None,
            reason: str,
            detail: str,
        ) -> None:
            skipped.append(
                SkippedExperience(
                    path=item_path,
                    id=entry_id,
                    version=version,
                    reason=reason,
                    detail=detail,
                )
            )

        for item in allowlist.items:
            if any(item.id == done.entry.id for done in loaded):
                skip(item.path, item.id, item.version, "duplicate_id",
                     f"id {item.id!r} already loaded")
                continue
            if len(loaded) >= self.max_entries:
                skip(item.path, item.id, item.version, "entry_limit",
                     f"max_entries {self.max_entries} reached")
                continue
            try:
                entry_path = resolve_entry_path(root, item.path)
            except ExperiencePathError as exc:
                skip(item.path, item.id, item.version, exc.reason, str(exc))
                continue
            try:
                entry = self._read_entry(entry_path)
            except OSError as exc:
                skip(item.path, item.id, item.version, "missing_file", str(exc))
                continue
            except ExperienceFormatError as exc:
                skip(item.path, item.id, item.version, "format_error", str(exc))
                continue
            if entry.id != item.id:
                skip(item.path, item.id, item.version, "id_mismatch",
                     f"allowlist names {item.id!r}, entry declares {entry.id!r}")
                continue
            if entry.version != item.version:
                skip(
                    item.path,
                    item.id,
                    item.version,
                    "version_not_approved",
                    f"allowlist approves v{item.version}, entry declares v{entry.version}",
                )
                continue
            refusal = entry.approval.refusal()
            if refusal is not None:
                skip(item.path, item.id, item.version, refusal[0], refusal[1])
                continue
            if entry.backends and backend not in entry.backends:
                skip(
                    item.path,
                    item.id,
                    item.version,
                    "backend_not_applicable",
                    f"entry applies to {list(entry.backends)!r}, run backend is {backend!r}",
                )
                continue
            unmet = entry.preconditions.unmet(
                task=task, profile=profile, backend=backend
            )
            if unmet:
                skip(
                    item.path,
                    item.id,
                    item.version,
                    "precondition_mismatch",
                    "; ".join(unmet),
                )
                continue
            block = format_experience_block(entry)
            if used + len(block) + 1 > self.max_chars:
                skip(item.path, item.id, item.version, "text_budget_exceeded",
                     f"{len(block)} chars do not fit in max_chars {self.max_chars}")
                continue
            used += len(block) + 1
            blocks.append(block)
            loaded.append(LoadedExperience(entry=entry, block=block))

        text = (
            EXPERIENCE_HEADER + "\n" + "\n".join(blocks) if blocks else ""
        )
        return ExperienceSelection(
            enabled=True,
            allowlist=str(allowlist.source),
            loaded=tuple(loaded),
            skipped=tuple(skipped),
            text=text,
            note="loaded" if loaded else "no_applicable_entry",
        )

    @staticmethod
    def _read_entry(path: Path) -> ExperienceEntry:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ExperienceFormatError(
                f"entry {path.name} is not valid UTF-8: {exc}"
            ) from exc
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ExperienceFormatError(
                f"invalid entry syntax in {path.name}: {exc}"
            ) from exc
        return parse_entry(data, source=str(path))


__all__ = [
    "EXPERIENCE_HEADER",
    "EXPERIENCE_MARKER",
    "ExperienceLoader",
    "ExperienceSelection",
    "LoadedExperience",
    "SkippedExperience",
    "format_experience_block",
]
