"""Approved local text experience (G wave, G06).

The package supplies three things and nothing else:

* :func:`webagent.experience.schema.parse_entry` — the documented entry format,
* :func:`webagent.experience.allowlist.load_allowlist` — the only list of files a
  run may read,
* :class:`webagent.experience.loader.ExperienceLoader` — approval/version and
  precondition checks plus the bounded prompt text.

It does not register tools, execute steps, read observations or write approval
state. The official entry only passes the assembled text into the existing
prompt assembly, so tool schema, registration and dispatch stay untouched.
"""

from .allowlist import Allowlist, AllowlistItem, load_allowlist, resolve_entry_path
from .loader import (
    EXPERIENCE_HEADER,
    EXPERIENCE_MARKER,
    ExperienceLoader,
    ExperienceSelection,
    LoadedExperience,
    SkippedExperience,
    format_experience_block,
)
from .schema import (
    ALLOWLIST_SCHEMA,
    ENTRY_SCHEMA,
    Approval,
    ExperienceEntry,
    ExperienceError,
    ExperienceFormatError,
    ExperiencePathError,
    Preconditions,
    parse_entry,
    parse_preconditions,
)

__all__ = [
    "ALLOWLIST_SCHEMA",
    "ENTRY_SCHEMA",
    "EXPERIENCE_HEADER",
    "EXPERIENCE_MARKER",
    "Allowlist",
    "AllowlistItem",
    "Approval",
    "ExperienceEntry",
    "ExperienceError",
    "ExperienceFormatError",
    "ExperienceLoader",
    "ExperiencePathError",
    "ExperienceSelection",
    "LoadedExperience",
    "Preconditions",
    "SkippedExperience",
    "format_experience_block",
    "load_allowlist",
    "parse_entry",
    "parse_preconditions",
    "resolve_entry_path",
]
