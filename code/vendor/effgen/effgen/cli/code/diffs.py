"""Unified diffs for the coding agent's file edits.

A write the agent proposes is shown as a unified diff before it touches disk, so
the change is reviewable. This module generates those diffs from
:mod:`difflib`, renders them with the CLI theme's colors, and applies a diff as
individual hunks against the file's current content — so a hunk that no longer
matches (the file changed since the edit was prepared) is reported precisely and
the hunks that still apply are kept, rather than the file being overwritten
whole.

Nothing here reads or writes the filesystem or calls a model: it operates on the
two strings (the current content and the proposed content) it is handed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# The context radius difflib puts around each changed region. Three lines is the
# de-facto default for ``diff -u`` and what a reviewer expects to see.
_CONTEXT = 3

# ``@@ -old_start,old_count +new_start,new_count @@`` — the count is omitted when
# it is 1, matching ``diff`` output.
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<os>\d+)(?:,(?P<oc>\d+))? \+(?P<ns>\d+)(?:,(?P<nc>\d+))? @@"
)


def _as_lines(text: str) -> list[str]:
    """Split *text* into lines, each keeping its trailing newline."""
    return text.splitlines(keepends=True)


@dataclass(frozen=True)
class DiffStat:
    """Added/removed line counts for one edit."""

    added: int
    removed: int

    def __str__(self) -> str:
        return f"+{self.added}/-{self.removed}"


def diff_stat(old: str, new: str) -> DiffStat:
    """Return the added/removed line counts between *old* and *new*."""
    added = removed = 0
    for line in difflib.unified_diff(_as_lines(old), _as_lines(new), n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return DiffStat(added=added, removed=removed)


def unified_diff_text(old: str, new: str, path: str, *, context: int = _CONTEXT) -> str:
    """Return a unified diff turning *old* into *new*, labelled with *path*.

    The header paths are ``a/<path>`` and ``b/<path>`` so the output reads like
    ``git diff``. An empty string is returned when the two are identical.

    Args:
        old: The content before the change.
        new: The content after the change.
        path: Workspace-relative path both header lines are labelled with.
        context: Unchanged lines kept on each side of a change.
    """
    diff = difflib.unified_diff(
        _as_lines(old),
        _as_lines(new),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    text = "".join(diff)
    # difflib omits the trailing newline flag; a diff whose last line lacks one
    # is annotated the way ``diff`` does so a reviewer sees it.
    if text and not text.endswith("\n"):
        text += "\n\\ No newline at end of file\n"
    return text


@dataclass
class Hunk:
    """One changed region of a unified diff.

    ``lines`` are the body lines with their leading marker (``' '`` context,
    ``'-'`` removed, ``'+'`` added). ``old_start``/``new_start`` are 1-based, as
    in the ``@@`` header.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)

    @property
    def old_block(self) -> list[str]:
        """The lines this hunk expects to find (context + removed), no markers."""
        return [ln[1:] for ln in self.lines if ln and ln[0] in " -"]

    @property
    def new_block(self) -> list[str]:
        """The lines this hunk produces (context + added), no markers."""
        return [ln[1:] for ln in self.lines if ln and ln[0] in " +"]

    @property
    def added(self) -> int:
        """Number of added lines in this hunk."""
        return sum(1 for ln in self.lines if ln.startswith("+"))

    @property
    def removed(self) -> int:
        """Number of removed lines in this hunk."""
        return sum(1 for ln in self.lines if ln.startswith("-"))


def split_hunks(old: str, new: str, *, context: int = _CONTEXT) -> list[Hunk]:
    """Return the hunks that turn *old* into *new*.

    Built from the same unified diff the preview renders, so the hunks a user
    sees are exactly the hunks :func:`apply_hunks` would apply.

    Args:
        old: The content before the change.
        new: The content after the change.
        context: Unchanged lines kept on each side of a change.

    Returns:
        One :class:`Hunk` per changed region, numbered from 1 in file order.
    """
    diff = list(
        difflib.unified_diff(_as_lines(old), _as_lines(new), n=context, lineterm="")
    )
    hunks: list[Hunk] = []
    current: Hunk | None = None
    for line in diff:
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        match = _HUNK_HEADER.match(line)
        if match:
            current = Hunk(
                old_start=int(match["os"]),
                old_count=int(match["oc"]) if match["oc"] is not None else 1,
                new_start=int(match["ns"]),
                new_count=int(match["nc"]) if match["nc"] is not None else 1,
            )
            hunks.append(current)
            continue
        if current is not None and line and line[0] in " -+":
            current.lines.append(line)
    return hunks


@dataclass
class ApplyResult:
    """The outcome of applying a set of hunks to a file's current content."""

    text: str
    applied: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when every requested hunk applied."""
        return not self.failed


def _find_block(lines: list[str], block: list[str], near: int) -> int | None:
    """Return the index in *lines* where *block* occurs closest to *near*.

    An empty block (a pure insertion) applies at ``near``. When the block is not
    present anywhere, ``None`` is returned so the caller can report the hunk as
    failed rather than corrupt the file.
    """
    if not block:
        return max(0, min(near, len(lines)))
    best: tuple[int, int] | None = None
    span = len(block)
    for i in range(len(lines) - span + 1):
        if lines[i : i + span] == block:
            distance = abs(i - near)
            if best is None or distance < best[1]:
                best = (i, distance)
    return best[0] if best is not None else None


def apply_hunks(original: str, hunks: list[Hunk], accepted: list[int]) -> ApplyResult:
    """Apply the *accepted* hunks to *original*, matching each by its context.

    Each accepted hunk's expected block (its context and removed lines) is
    located in the current content near where the hunk says it should be, then
    replaced by the hunk's produced lines. A hunk whose block is not found — the
    file changed underneath it — is left out and reported in ``failed``; the
    hunks that do match still apply. Rejected hunks (not in *accepted*) are
    ignored. The returned text never mixes a partially matched hunk in: a hunk
    is applied whole or not at all.

    Args:
        original: The file's current content.
        hunks: The hunks :func:`split_hunks` produced for that content.
        accepted: The 1-based numbers of the hunks to apply.

    Returns:
        An :class:`ApplyResult` carrying the new text, the hunk numbers that
        applied and the hunk numbers whose block was not found.
    """
    lines = _as_lines(original)
    applied: list[int] = []
    failed: list[int] = []
    # Apply in file order so running offsets from earlier splices stay correct.
    offset = 0
    order = sorted(i for i in accepted if 0 <= i < len(hunks))
    for index in order:
        hunk = hunks[index]
        near = max(0, hunk.old_start - 1 + offset)
        block = hunk.old_block
        found = _find_block(lines, block, near)
        if found is None:
            failed.append(index)
            continue
        new_block = hunk.new_block
        lines[found : found + len(block)] = new_block
        offset += len(new_block) - len(block)
        applied.append(index)
    return ApplyResult(text="".join(lines), applied=sorted(applied), failed=sorted(failed))


# Theme styles per diff line kind, resolved against the CLI palette.
_ADD = "effgen.success"
_DEL = "effgen.error"
_HUNK = "effgen.info"
_META = "effgen.muted"


def render_diff(diff_text: str, stream: object | None = None) -> tuple[str, str]:
    """Return ``(plain, markup)`` renderings of *diff_text*.

    ``plain`` is the diff unchanged, for a stream without color. ``markup`` wraps
    each line in the theme style for its kind (added green, removed red, hunk
    header cyan, file header muted), escaped so diff content with brackets does
    not read as markup.
    """
    plain_lines: list[str] = []
    markup_lines: list[str] = []
    for raw in diff_text.splitlines():
        plain_lines.append(raw)
        escaped = raw.replace("[", r"\[")
        if raw.startswith(("+++", "---", "diff ", "index ", "\\ ")):
            style = _META
        elif raw.startswith("@@"):
            style = _HUNK
        elif raw.startswith("+"):
            style = _ADD
        elif raw.startswith("-"):
            style = _DEL
        else:
            style = None
        markup_lines.append(f"[{style}]{escaped}[/{style}]" if style else escaped)
    return "\n".join(plain_lines), "\n".join(markup_lines)
