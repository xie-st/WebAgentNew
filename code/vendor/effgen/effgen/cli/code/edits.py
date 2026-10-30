"""Proposed edits and the per-workspace undo journal for ``effgen code``.

Two pieces:

- :class:`ProposedEdit` — a single file write the agent wants to make, captured
  before it happens: the path, the file's current content, the proposed content,
  and the unified diff between them. The gated file tool builds one of these so a
  write can be previewed and applied hunk-by-hunk against the file's live content
  rather than overwriting it blindly.
- :class:`EditJournal` — a bounded, on-disk stack of the edits a workspace has
  applied, so ``effgen code --undo`` can restore the previous content of the last
  change (or delete a file the run created). The journal lives outside the
  workspace, under the effGen state directory, keyed to the workspace path, so it
  never appears in the files the agent sees.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from effgen.utils.atomic_file import atomic_write_text

from .diffs import Hunk, apply_hunks, diff_stat, split_hunks, unified_diff_text

# Most edits a single undo would ever walk back. The journal keeps this many
# entries per workspace and drops the oldest beyond it, so history stays small.
_MAX_HISTORY = 100


@dataclass
class ProposedEdit:
    """One pending file write, with the diff it would produce.

    Attributes:
        rel_path: The write target, relative to the workspace root.
        old_content: The file's content now (``""`` when it does not yet exist).
        new_content: The content the agent proposes.
        is_new: True when the file does not exist yet.
    """

    rel_path: str
    old_content: str
    new_content: str
    is_new: bool

    @property
    def unchanged(self) -> bool:
        """True when the proposed content matches what is already on disk."""
        return self.old_content == self.new_content

    @property
    def hunks(self) -> list[Hunk]:
        """The hunks turning the current content into the proposed content."""
        return split_hunks(self.old_content, self.new_content)

    @property
    def diff_text(self) -> str:
        """The unified diff of this edit, labelled with the relative path."""
        return unified_diff_text(self.old_content, self.new_content, self.rel_path)

    @property
    def added(self) -> int:
        """Number of added lines."""
        return diff_stat(self.old_content, self.new_content).added

    @property
    def removed(self) -> int:
        """Number of removed lines."""
        return diff_stat(self.old_content, self.new_content).removed

    def stat(self) -> str:
        """A short ``+a/-b`` summary of the change size."""
        return str(diff_stat(self.old_content, self.new_content))

    def resolve_against_current(self, current: str) -> tuple[str, list[int], list[int]]:
        """Return ``(text, applied, failed)`` applying this edit to *current*.

        When *current* is what the edit was prepared against, the proposed
        content is produced exactly. When the file changed underneath, the hunks
        that still match are applied and the rest are reported as failed rather
        than overwriting the file — so the user's other changes are not lost.
        """
        if current == self.old_content:
            return self.new_content, list(range(len(self.hunks))), []
        hunks = self.hunks
        result = apply_hunks(current, hunks, list(range(len(hunks))))
        return result.text, result.applied, result.failed


def _history_root() -> Path:
    """Return the directory the undo journals live in.

    Honors ``EFFGEN_HOME`` (the base effGen state dir) like the session store,
    falling back to ``~/.effgen``. The journals sit in a ``code-history``
    subdirectory, one file per workspace.
    """
    home = os.environ.get("EFFGEN_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".effgen"
    return base / "code-history"


@dataclass
class AppliedEdit:
    """A recorded write, enough to reverse it."""

    rel_path: str
    before: str | None  # the content before the write; None when newly created
    after: str
    created: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        """Return the record as a JSON-serializable dict for the journal file."""
        return {
            "path": self.rel_path,
            "before": self.before,
            "after": self.after,
            "created": self.created,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AppliedEdit":
        """Rebuild a record from a journal entry, tolerating missing fields."""
        return cls(
            rel_path=str(data.get("path", "")),
            before=data.get("before"),  # type: ignore[arg-type]
            after=str(data.get("after", "")),
            created=bool(data.get("created", False)),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
        )


@dataclass
class UndoOutcome:
    """What one undo step did."""

    rel_path: str
    action: str  # "restored" (previous content written) or "removed" (created file deleted)
    detail: str = ""


class EditJournal:
    """A bounded, on-disk stack of a workspace's applied edits.

    Args:
        workspace: The directory whose edits this journal tracks.
        root: Where journals are stored; defaults to the effGen state dir. Tests
            point it elsewhere.
    """

    def __init__(self, workspace: Path, *, root: Path | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self._root = Path(root) if root is not None else _history_root()

    # -- storage ------------------------------------------------------------

    def _path(self) -> Path:
        key = hashlib.sha1(str(self.workspace).encode("utf-8")).hexdigest()[:16]
        return self._root / f"{key}.json"

    def _load(self) -> list[AppliedEdit]:
        path = self._path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries = data.get("entries", []) if isinstance(data, dict) else []
        return [AppliedEdit.from_dict(e) for e in entries if isinstance(e, dict)]

    def _save(self, entries: list[AppliedEdit]) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        trimmed = entries[-_MAX_HISTORY:]
        payload = {
            "workspace": str(self.workspace),
            "entries": [e.to_dict() for e in trimmed],
        }
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False))

    # -- recording ----------------------------------------------------------

    def record(self, edit: AppliedEdit) -> None:
        """Push one applied edit onto the stack."""
        entries = self._load()
        entries.append(edit)
        self._save(entries)

    def record_write(self, rel_path: str, before: str | None, after: str) -> None:
        """Record a write of *rel_path*, capturing what to restore on undo.

        Args:
            rel_path: Workspace-relative path that was written.
            before: The content before the write, or ``None`` when the run created
                the file — this is what an undo restores.
            after: The content the write left on disk.
        """
        self.record(
            AppliedEdit(
                rel_path=rel_path,
                before=before,
                after=after,
                created=before is None,
            )
        )

    # -- inspection ---------------------------------------------------------

    def entries(self) -> list[AppliedEdit]:
        """Return the recorded edits, oldest first."""
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    # -- undo ---------------------------------------------------------------

    def undo(self) -> UndoOutcome | None:
        """Reverse the most recent applied edit, or return ``None`` if empty.

        Restores the file's previous content, or deletes it when the edit
        created it. The target is resolved inside the workspace before anything
        is written; an entry that would escape the workspace is dropped and
        reported rather than restored.
        """
        entries = self._load()
        if not entries:
            return None
        edit = entries.pop()

        target = (self.workspace / edit.rel_path).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            self._save(entries)
            return UndoOutcome(
                rel_path=edit.rel_path,
                action="skipped",
                detail="target resolves outside the workspace; not restored.",
            )

        try:
            if edit.created:
                if target.exists():
                    target.unlink()
                outcome = UndoOutcome(edit.rel_path, "removed", "created by the run")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.before or "", encoding="utf-8")
                outcome = UndoOutcome(edit.rel_path, "restored", "previous content")
        except OSError as exc:
            # Put the entry back so a transient failure does not lose history.
            entries.append(edit)
            self._save(entries)
            return UndoOutcome(edit.rel_path, "failed", str(exc))

        self._save(entries)
        return outcome
