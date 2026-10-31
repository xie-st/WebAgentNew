"""Format and validation for approved local text experience (G wave, G06).

An experience entry is local, human-approved *text* guidance. Loading it only
adds text to the existing prompt assembly: an entry never registers a tool,
executes a step, changes the completion checks or carries a page-derived value.

The parser is deliberately strict. An allowlist or entry that does not match the
documented format is refused and recorded as skipped, never partially applied.
That is what makes "only approved versions" checkable: the approval and version
that actually reached the prompt are the ones the file declared and the
allowlist named.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

ENTRY_SCHEMA = "webagent.experience.v1"
ALLOWLIST_SCHEMA = "webagent.experience.allowlist.v1"

#: Ids are prompt markers and event keys, so keep them small and path-free.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_APPROVED_STATUS = "approved"


class ExperienceError(ValueError):
    """Base error for a refused experience configuration or entry."""


class ExperienceFormatError(ExperienceError):
    """An allowlist or entry file does not match the documented format."""


class ExperiencePathError(ExperienceError):
    """An allowlisted entry path leaves the allowlist directory."""

    def __init__(self, message: str, *, reason: str = "invalid_path") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Approval:
    """The entry's recorded approval state, copied verbatim from its file."""

    approved: bool
    status: str
    approved_by: str | None = None

    @property
    def is_injectable(self) -> bool:
        return self.approved is True and self.status == _APPROVED_STATUS

    def refusal(self) -> tuple[str, str] | None:
        """``(reason, detail)`` when this approval cannot be injected."""

        if self.approved is not True:
            return (
                "not_approved",
                f"approval.approved is {self.approved!r}; only approved entries load",
            )
        if self.status != _APPROVED_STATUS:
            return (
                "approval_status_not_approved",
                f"approval.status is {self.status!r}; only {_APPROVED_STATUS!r} loads",
            )
        return None


@dataclass(frozen=True)
class Preconditions:
    """Deterministic applicability checks evaluated against the run.

    Empty lists mean "no constraint". The values are matched against program
    facts only (profile, backend, task text); page content is never an input.
    """

    backends: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    task_contains: tuple[str, ...] = ()
    task_excludes: tuple[str, ...] = ()

    def unmet(
        self,
        *,
        task: str | None,
        profile: str,
        backend: str,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.backends and backend not in self.backends:
            reasons.append(
                f"backend {backend!r} is not one of {list(self.backends)!r}"
            )
        if self.profiles and profile not in self.profiles:
            reasons.append(
                f"profile {profile!r} is not one of {list(self.profiles)!r}"
            )
        text = (task or "").casefold()
        if self.task_contains and not text:
            reasons.append("no task text is available for task_contains")
        else:
            for token in self.task_contains:
                if token.casefold() not in text:
                    reasons.append(f"task does not contain {token!r}")
        for token in self.task_excludes:
            if token.casefold() in text:
                reasons.append(f"task contains excluded token {token!r}")
        return tuple(reasons)


@dataclass(frozen=True)
class ExperienceEntry:
    """One parsed, still-unapproved entry; approval is checked by the loader."""

    id: str
    version: int
    purpose: str
    preconditions: Preconditions
    guidance: str
    verification: tuple[str, ...]
    source: str
    backends: tuple[str, ...]
    approval: Approval
    #: Where the entry was read from; provenance only, never injected verbatim.
    path: str = ""


def _expect_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperienceFormatError(f"{where} must be a mapping")
    return value


def _expect_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceFormatError(f"{where} must be a non-empty string")
    return value.strip()


def _expect_version(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExperienceFormatError(f"{where} must be an integer >= 1")
    return value


def _expect_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ExperienceFormatError(f"{where} must be a boolean")
    return value


def _expect_string_list(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ExperienceFormatError(f"{where} must be a list of strings")
    items: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str) or not raw.strip():
            raise ExperienceFormatError(f"{where}[{index}] must be a non-empty string")
        items.append(raw.strip())
    return tuple(items)


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ExperienceFormatError(
            f"unknown keys in {where}: {', '.join(unknown)}"
        )


def parse_preconditions(
    raw: Any, *, where: str = "preconditions"
) -> Preconditions:
    mapping = _expect_mapping(raw, where)
    _check_keys(
        mapping,
        {"backends", "profiles", "task_contains", "task_excludes"},
        where,
    )
    return Preconditions(
        backends=_expect_string_list(
            mapping.get("backends", []), f"{where}.backends"
        ),
        profiles=_expect_string_list(
            mapping.get("profiles", []), f"{where}.profiles"
        ),
        task_contains=_expect_string_list(
            mapping.get("task_contains", []), f"{where}.task_contains"
        ),
        task_excludes=_expect_string_list(
            mapping.get("task_excludes", []), f"{where}.task_excludes"
        ),
    )


def parse_approval(raw: Any, *, where: str = "approval") -> Approval:
    mapping = _expect_mapping(raw, where)
    _check_keys(mapping, {"approved", "status", "approved_by"}, where)
    approved_by = mapping.get("approved_by")
    if approved_by is not None:
        approved_by = _expect_text(approved_by, f"{where}.approved_by")
    return Approval(
        approved=_expect_bool(mapping.get("approved"), f"{where}.approved"),
        status=_expect_text(mapping.get("status"), f"{where}.status").lower(),
        approved_by=approved_by,
    )


def parse_entry(data: Any, *, source: str = "") -> ExperienceEntry:
    """Parse one experience entry, refusing any undocumented shape."""

    mapping = _expect_mapping(data, "experience entry")
    _check_keys(
        mapping,
        {
            "schema",
            "id",
            "version",
            "purpose",
            "preconditions",
            "guidance",
            "verification",
            "source",
            "backends",
            "approval",
        },
        "experience entry",
    )
    schema = mapping.get("schema", ENTRY_SCHEMA)
    if schema != ENTRY_SCHEMA:
        raise ExperienceFormatError(
            f"unsupported entry schema {schema!r}; expected {ENTRY_SCHEMA!r}"
        )
    entry_id = _expect_text(mapping.get("id"), "id")
    if not _ID.match(entry_id):
        raise ExperienceFormatError(
            f"id {entry_id!r} must match {_ID.pattern}"
        )
    verification_raw = mapping.get("verification")
    if isinstance(verification_raw, str):
        verification = (_expect_text(verification_raw, "verification"),)
    else:
        verification = _expect_string_list(verification_raw, "verification")
        if not verification:
            raise ExperienceFormatError("verification must not be empty")
    return ExperienceEntry(
        id=entry_id,
        version=_expect_version(mapping.get("version"), "version"),
        purpose=_expect_text(mapping.get("purpose"), "purpose"),
        preconditions=parse_preconditions(mapping.get("preconditions")),
        guidance=_expect_text(mapping.get("guidance"), "guidance"),
        verification=verification,
        source=_expect_text(mapping.get("source"), "source"),
        backends=_expect_string_list(mapping.get("backends", []), "backends"),
        approval=parse_approval(mapping.get("approval")),
        path=source,
    )


__all__ = [
    "ALLOWLIST_SCHEMA",
    "ENTRY_SCHEMA",
    "Approval",
    "ExperienceEntry",
    "ExperienceError",
    "ExperienceFormatError",
    "ExperiencePathError",
    "Preconditions",
    "parse_entry",
    "parse_preconditions",
]
