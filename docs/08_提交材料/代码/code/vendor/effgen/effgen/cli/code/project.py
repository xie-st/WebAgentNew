"""Repository detection and the project context ``effgen code`` starts from.

Before the first turn the coding agent needs to know what it is working on: is
this a git repository, which branch, what is already modified, and which files
exist. This module answers that with a few read-only git commands and a bounded
file inventory, and turns it into two things:

- a one-line summary for the banner
  (``Repo: effGen · branch main · 3 change(s) · 412 file(s)``),
- a compact block appended to the system prompt so the model can ask for the
  files it needs instead of guessing at paths.

Inside a repository the inventory comes from ``git ls-files`` with the standard
exclusions, so ``.gitignore`` is respected without reimplementing it. Outside one
the directory is walked with a small skip list and no error is raised — a plain
workspace is a supported way to work.

Everything here reads; nothing in this module changes a repository. The one
mutating action the coding agent offers lives in :mod:`effgen.cli.code.git_actions`
behind an explicit confirmation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effgen.tools.builtin._fs import is_credential_filename

#: Most files serialized by :meth:`ProjectContext.to_dict`.
MAX_FILES = 200

#: Up to this many files, the layout lists every file; beyond it, the layout
#: summarizes each top-level directory by how many files it holds.
DETAIL_THRESHOLD = 60

#: Most files inventoried in one workspace, git-listed or walked.
WALK_CAP = 2000

#: Character budget for the layout block, so a large repository stays compact.
MAX_LAYOUT_CHARS = 6000

#: Most ``git status`` lines carried into the context block.
MAX_STATUS_LINES = 12

#: Most characters read from a project instruction file.
MAX_BRIEF_CHARS = 4000

#: Filenames read as the project's instruction file, in order of preference.
BRIEF_FILENAMES: tuple[str, ...] = ("AGENTS.md", "agents.md")

# Directories skipped when walking a workspace that is not a git repository.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".eggs", ".idea",
    ".vscode", "dist", "build", ".ipynb_checkpoints",
})

#: Branch text used when the repository has no commit on HEAD yet.
NO_COMMITS = "no commits yet"

_GIT_TIMEOUT = 10


def _listable(rel: str) -> bool:
    """True when *rel* belongs in the inventory shown to the model.

    Credential-shaped filenames are left out: the write and read paths refuse
    them anyway, so listing one only invites a request that will be refused.
    """
    return not is_credential_filename(Path(rel).name)


def _git_read(root: Path, args: list[str], *, timeout: int = _GIT_TIMEOUT) -> str | None:
    """Run a read-only git command in *root* and return stdout, or ``None``.

    Any failure — git missing, not a repository, a timeout, a non-zero exit —
    returns ``None``. Project context is an aid, so a repository that cannot be
    read degrades to the plain workspace inventory instead of failing the run.
    """
    if shutil.which("git") is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@dataclass
class RepoInfo:
    """What the read-only git surface knows about the workspace's repository."""

    root: Path
    branch: str
    status_lines: list[str] = field(default_factory=list)
    staged: int = 0
    unstaged: int = 0
    untracked: int = 0
    status_truncated: int = 0

    @property
    def dirty(self) -> bool:
        """True when the working tree or index has changes."""
        return bool(self.staged or self.unstaged or self.untracked)

    @property
    def branch_label(self) -> str:
        """How the branch reads in a sentence.

        A named branch reads ``branch main``; a detached or not-yet-created HEAD
        is already a phrase (``detached at 1a2b3c``, ``no commits yet``) and is
        used as-is.
        """
        if self.branch == NO_COMMITS or self.branch.startswith("detached"):
            return self.branch
        return f"branch {self.branch}"

    @property
    def changed(self) -> int:
        """How many paths ``git status`` reported."""
        return self.staged + self.unstaged + self.untracked

    def to_dict(self) -> dict[str, Any]:
        """Return the repository facts as a JSON-serializable dict."""
        return {
            "root": str(self.root),
            "branch": self.branch,
            "dirty": self.dirty,
            "staged": self.staged,
            "unstaged": self.unstaged,
            "untracked": self.untracked,
        }


def detect_repo(workspace: Path) -> RepoInfo | None:
    """Return the repository containing *workspace*, or ``None`` if there is none.

    Reports the branch (or the short commit when HEAD is detached) and a bounded
    ``git status --porcelain`` reading, counted by staged / unstaged / untracked.
    """
    top = _git_read(Path(workspace), ["rev-parse", "--show-toplevel"])
    if not top or not top.strip():
        return None
    root = Path(top.strip())

    branch = (_git_read(root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "").strip()
    if not branch or branch == "HEAD":
        short = (_git_read(root, ["rev-parse", "--short", "HEAD"]) or "").strip()
        branch = f"detached at {short}" if short else NO_COMMITS

    porcelain = _git_read(root, ["status", "--porcelain"]) or ""
    lines = [line for line in porcelain.splitlines() if line.strip()]
    staged = unstaged = untracked = 0
    for line in lines:
        code = line[:2]
        if code == "??":
            untracked += 1
            continue
        if code[0] not in (" ", "?"):
            staged += 1
        if code[1] not in (" ", "?"):
            unstaged += 1

    return RepoInfo(
        root=root,
        branch=branch,
        status_lines=lines[:MAX_STATUS_LINES],
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        status_truncated=max(0, len(lines) - MAX_STATUS_LINES),
    )


def staged_diff(root: Path, *, max_chars: int = 20000) -> str:
    """Return the staged patch (``git diff --cached``), truncated to *max_chars*.

    An empty string means nothing is staged (or the diff could not be read).
    """
    text = _git_read(Path(root), ["diff", "--cached"], timeout=20) or ""
    if len(text) > max_chars:
        return text[:max_chars] + f"\n… (truncated at {max_chars:,} characters)"
    return text


#: Character budget for a review subject (the diff plus any named files).
MAX_REVIEW_CHARS = 60000

#: ``--review`` with no target: everything not yet committed, staged or not.
REVIEW_UNCOMMITTED = "uncommitted"

#: ``--review staged``: the index only.
REVIEW_STAGED = "staged"


def _truncate(text: str, max_chars: int) -> tuple[str, int]:
    """Return ``(text, cut_at)``, marking and counting anything left out."""
    if len(text) <= max_chars:
        return text, 0
    dropped = len(text) - max_chars
    kept = text[:max_chars]
    marked = (
        f"{kept}\n… (truncated at {max_chars:,} characters; "
        f"{dropped:,} more not shown)"
    )
    return marked, max_chars


def working_diff(root: Path, *, ref: str | None = None, max_chars: int = MAX_REVIEW_CHARS) -> str:
    """Return the patch for *ref*, truncated to *max_chars*.

    Args:
        root: The repository to read, or any directory inside it.
        ref: ``None`` means everything not yet committed — ``git diff HEAD``, so
            staged and unstaged changes are read together, falling back to
            ``git diff`` in a repository with no commit yet. Any other value is
            passed to ``git diff`` as given, so a revision (``HEAD~3``) and a
            range (``main...HEAD``) both work.
        max_chars: Budget for the patch; the cut is marked and counted.

    Returns:
        The patch text. An empty string means there is no such change (or the
        diff could not be read); use :func:`verify_rev` first to tell a bad
        revision from an empty one.
    """
    root = Path(root)
    if ref is None:
        text = _git_read(root, ["diff", "HEAD"], timeout=20)
        if text is None:
            text = _git_read(root, ["diff"], timeout=20) or ""
    else:
        text = _git_read(root, ["diff", ref], timeout=20) or ""
    return _truncate(text, max_chars)[0]


def verify_rev(root: Path, ref: str) -> str | None:
    """Return an error message when git does not recognize *ref*, else ``None``.

    Run before :func:`working_diff` so a mistyped revision is reported as such
    instead of reading as "nothing changed". A range (``main..HEAD``,
    ``main...HEAD``) is checked side by side, because a range that resolves to
    no commits is still a valid range and an empty diff is a real answer.
    """
    root = Path(root)
    separator = "..." if "..." in ref else (".." if ".." in ref else "")
    sides = ref.split(separator) if separator else [ref]
    for side in sides:
        # An omitted side (``main..``) means HEAD and needs no check.
        name = side.strip()
        if not name:
            continue
        if _git_read(root, ["rev-parse", "--verify", "--quiet", f"{name}^{{commit}}"]):
            continue
        return (
            f"'{name}' is not a revision this repository knows. Pass a commit, "
            "a branch, a range such as main...HEAD, 'staged', or omit the "
            "target to review everything that is not committed."
        )
    return None


@dataclass
class ReviewSubject:
    """What a read-only review run was asked to look at.

    Built by :func:`review_subject` and handed to the model as context, so the
    review never needs a shell to reach a diff. A non-empty :attr:`error` means
    no subject could be assembled and names how to give one.
    """

    kind: str = ""
    ref: str = ""
    files: list[str] = field(default_factory=list)
    text: str = ""
    truncated_at: int = 0
    line_count: int = 0
    error: str = ""

    def describe(self) -> str:
        """One line naming what is being reviewed."""
        if self.error:
            return self.error
        parts = []
        if self.ref == REVIEW_UNCOMMITTED:
            parts.append("uncommitted changes")
        elif self.ref == REVIEW_STAGED:
            parts.append("staged changes")
        elif self.ref:
            parts.append(f"diff of {self.ref}")
        if self.files:
            parts.append(f"{len(self.files)} file(s): {', '.join(self.files)}")
        body = " · ".join(parts) or "nothing"
        suffix = f" (truncated at {self.truncated_at:,} characters)" if self.truncated_at else ""
        return f"Reviewing {body}, {self.line_count:,} line(s){suffix}"

    def to_dict(self) -> dict[str, Any]:
        """Return the subject as a JSON-serializable dict (without its text)."""
        return {
            "kind": self.kind,
            "ref": self.ref,
            "files": list(self.files),
            "line_count": self.line_count,
            "truncated_at": self.truncated_at,
        }


def review_subject(
    workspace: Path,
    target: str | None = None,
    files: Iterable[str] = (),
    *,
    max_chars: int = MAX_REVIEW_CHARS,
) -> ReviewSubject:
    """Assemble the diff and/or file set a review run reads.

    Args:
        workspace: The directory the run is confined to.
        target: ``uncommitted``, ``staged``, or any revision/range git accepts.
            ``None`` means no diff was asked for.
        files: Workspace-relative paths to include in full. A path outside the
            workspace, or one that is not a readable text file, is reported
            rather than silently skipped.
        max_chars: Budget for the assembled subject; the cut is marked and counted.

    Returns:
        A :class:`ReviewSubject`. A non-empty ``error`` means there is nothing to
        review and says how to name a subject.
    """
    workspace = Path(workspace)
    named = [str(f) for f in files]
    repo = detect_repo(workspace) if target is not None else None

    if target is not None and repo is None:
        return ReviewSubject(error=(
            f"{workspace} is not inside a git repository, so there is no diff to "
            "review. Name the files instead (-f/--file PATH, repeatable), or run "
            "the review from inside a repository."
        ))

    sections: list[str] = []
    ref = ""
    kind = ""
    if repo is not None:
        if target == REVIEW_STAGED:
            ref = REVIEW_STAGED
            patch = staged_diff(repo.root, max_chars=max_chars)
            heading = "The staged changes (git diff --cached)"
            empty = "nothing is staged"
        elif target in (None, "", REVIEW_UNCOMMITTED):
            ref = REVIEW_UNCOMMITTED
            patch = working_diff(repo.root, max_chars=max_chars)
            heading = "Everything not yet committed (git diff HEAD)"
            empty = "everything is committed"
        else:
            bad = verify_rev(repo.root, target)
            if bad:
                return ReviewSubject(error=bad)
            ref = target
            patch = working_diff(repo.root, ref=target, max_chars=max_chars)
            heading = f"The diff of {target} (git diff {target})"
            empty = f"'{target}' covers no change"
        if patch.strip():
            kind = "diff"
            sections.append(f"{heading}:\n\n{patch}")
        elif not named:
            return ReviewSubject(ref=ref, error=(
                f"There is nothing to review in {repo.root}: {empty}. Make or "
                "stage a change, pass a revision (--review HEAD~1), or name "
                "files with -f/--file."
            ))

    included: list[str] = []
    for rel in named:
        try:
            resolved = (workspace / rel).resolve()
            resolved.relative_to(workspace.resolve())
        except (ValueError, OSError):
            return ReviewSubject(ref=ref, error=(
                f"'{rel}' is outside the workspace {workspace}; a review reads "
                "only files inside it."
            ))
        if is_credential_filename(resolved.name):
            return ReviewSubject(ref=ref, error=(
                f"'{rel}' reads as a credentials file, which is never opened. "
                "Name a source file instead."
            ))
        try:
            body = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ReviewSubject(ref=ref, error=(
                f"'{rel}' could not be read as text: {exc}"
            ))
        included.append(rel)
        sections.append(f"=== {rel} ===\n{body}")
    if included:
        kind = "diff+files" if kind else "files"

    if not sections:
        return ReviewSubject(ref=ref, error=(
            "Nothing to review. Pass --review with a target inside a git "
            "repository, or name files with -f/--file PATH."
        ))

    joined = "\n\n".join(sections)
    text, cut = _truncate(joined, max_chars)
    return ReviewSubject(
        kind=kind,
        ref=ref,
        files=included,
        text=text,
        truncated_at=cut,
        line_count=joined.count("\n") + 1,
    )


def _repo_files(workspace: Path) -> list[str] | None:
    """List the workspace's files via git, honoring ``.gitignore``.

    Tracked files plus untracked-but-not-ignored ones, relative to *workspace*,
    which is what a person sees when they look at the directory. ``None`` when
    git could not answer.
    """
    out = _git_read(workspace, ["ls-files", "--cached", "--others", "--exclude-standard"])
    if out is None:
        return None
    return sorted(
        {line.strip() for line in out.splitlines() if line.strip() and _listable(line.strip())}
    )


def _walked_files(workspace: Path) -> tuple[list[str], bool]:
    """Walk *workspace*, returning ``(paths, capped)`` sorted, skipping build dirs.

    Used when the workspace is not in a git repository. Build, cache and VCS
    directories are pruned as the walk descends, and the walk stops at
    :data:`WALK_CAP` entries so a huge tree does not turn startup into a
    filesystem crawl; ``capped`` says whether it stopped early.
    """
    root = Path(workspace)
    found: list[str] = []
    capped = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
        )
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            if name.endswith(".pyc") or not _listable(name):
                continue
            found.append(str(rel_dir / name) if rel_dir.parts else name)
            if len(found) >= WALK_CAP:
                return found, True
    return sorted(found), capped


def _file_size(workspace: Path, rel: str) -> int:
    try:
        return (workspace / rel).stat().st_size
    except OSError:
        return 0


def read_brief(workspace: Path, repo: RepoInfo | None) -> tuple[str | None, str | None]:
    """Return ``(path, text)`` of the project instruction file, or ``(None, None)``.

    A project can steer the coding agent with an ``AGENTS.md`` in the workspace
    (or at the repository root). It is read as text, truncated to
    :data:`MAX_BRIEF_CHARS`, and never executed.
    """
    roots = [Path(workspace)]
    if repo is not None and repo.root != Path(workspace):
        roots.append(repo.root)
    for root in roots:
        for name in BRIEF_FILENAMES:
            candidate = root / name
            try:
                if not candidate.is_file():
                    continue
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if len(text) > MAX_BRIEF_CHARS:
                text = text[:MAX_BRIEF_CHARS] + "\n… (truncated)"
            return str(candidate), text
    return None, None


@dataclass
class ProjectContext:
    """The workspace's repository state, file layout and project brief."""

    workspace: Path
    repo: RepoInfo | None = None
    files: list[str] = field(default_factory=list)
    total_files: int = 0
    count_capped: bool = False
    ignored_respected: bool = False
    brief_path: str | None = None
    brief: str | None = None

    def file_count_text(self) -> str:
        """The file count, marked ``+`` when the walk stopped at its cap."""
        return f"{self.total_files}{'+' if self.count_capped else ''} file(s)"

    def summary_line(self) -> str:
        """One line for the banner and the run header."""
        if self.repo is not None:
            bits = [f"Repo: {self.repo.root.name}", self.repo.branch_label]
            bits.append(
                f"{self.repo.changed} change(s)" if self.repo.dirty else "clean tree"
            )
        else:
            bits = ["No git repository"]
        bits.append(self.file_count_text())
        return " · ".join(bits)

    def layout_lines(self) -> list[str]:
        """The file layout, bounded by count and by a character budget.

        A small workspace is listed file by file with sizes — that is what a
        person would see in it. A large one is summarized instead: the files at
        the root, then each top-level directory with how many files it holds, so
        the model learns the shape of the tree and can ask for a directory rather
        than being handed a thousand paths. Whatever is left out is counted,
        never silently dropped.

        Both the prompt block and ``/context`` read this, so what the model is
        told and what the user is shown are the same listing.
        """
        if self.total_files <= DETAIL_THRESHOLD:
            return self._bounded(
                f"  {rel} ({_file_size(self.workspace, rel):,} B)" for rel in self.files
            )

        root_files: list[str] = []
        directories: dict[str, int] = {}
        for rel in self.files:
            head, _, tail = rel.replace("\\", "/").partition("/")
            if tail:
                directories[head] = directories.get(head, 0) + 1
            else:
                root_files.append(head)

        lines = [f"  {name} ({_file_size(self.workspace, name):,} B)" for name in root_files]
        lines += [
            f"  {name}/ — {count} file(s)"
            for name, count in sorted(directories.items())
        ]
        return self._bounded(lines)

    def _bounded(self, lines: Iterable[str]) -> list[str]:
        """Cut *lines* at :data:`MAX_LAYOUT_CHARS`, noting how many were dropped."""
        kept: list[str] = []
        used = 0
        dropped = 0
        for line in lines:
            if used + len(line) > MAX_LAYOUT_CHARS:
                dropped += 1
                continue
            kept.append(line)
            used += len(line) + 1
        if dropped:
            kept.append(f"  … and {dropped} more entries not listed")
        return kept

    def as_prompt(self) -> str:
        """Return the context block appended to the agent's system prompt."""
        parts: list[str] = []
        if self.repo is not None:
            head = [
                f"Git repository: {self.repo.root} ({self.repo.branch_label})",
            ]
            if self.repo.status_lines:
                head.append(
                    f"Uncommitted changes ({self.repo.changed} path(s), "
                    "git status --porcelain):"
                )
                head.extend(f"  {line}" for line in self.repo.status_lines)
                if self.repo.status_truncated:
                    head.append(f"  … and {self.repo.status_truncated} more")
            else:
                head.append("The working tree is clean.")
            parts.append("\n".join(head))
        else:
            parts.append(f"Not a git repository. Workspace: {self.workspace}")

        if self.files:
            listed = "Files in the workspace"
            if self.ignored_respected:
                listed += " (ignored files excluded)"
            parts.append(listed + ":\n" + "\n".join(self.layout_lines()))

        if self.brief:
            parts.append(
                f"Project instructions from {self.brief_path} — follow them:\n{self.brief}"
            )
        parts.append(
            "This listing is a starting point: read a file before changing it, and "
            "do not assume a path that is not listed exists."
        )
        return "Project context\n\n" + "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return the context as a JSON-serializable dict."""
        return {
            "workspace": str(self.workspace),
            "repo": self.repo.to_dict() if self.repo else None,
            "files": self.files[:MAX_FILES],
            "total_files": self.total_files,
            "count_capped": self.count_capped,
            "brief_path": self.brief_path,
        }


def build_project_context(
    workspace: Path, *, cap: int = WALK_CAP, include_brief: bool = True
) -> ProjectContext:
    """Assemble the project context for *workspace*.

    Inside a repository the inventory comes from git (so ``.gitignore`` is
    honored); otherwise the directory is walked. Either way it stops at *cap*
    entries, and the call never raises: a workspace that cannot be inspected
    yields an empty inventory rather than a failed run.

    Args:
        workspace: The directory to inspect.
        cap: Most entries to list before the inventory is marked capped.
        include_brief: Whether to read the project brief file when one exists.
    """
    workspace = Path(workspace)
    repo = detect_repo(workspace)

    files = _repo_files(workspace) if repo is not None else None
    ignored_respected = files is not None
    capped = False
    if files is None:
        try:
            files, capped = _walked_files(workspace)
        except OSError:
            files = []
    if len(files) > cap:
        files = files[:cap]
        capped = True

    brief_path, brief = read_brief(workspace, repo) if include_brief else (None, None)
    return ProjectContext(
        workspace=workspace,
        repo=repo,
        files=files,
        total_files=len(files),
        count_capped=capped,
        ignored_respected=ignored_respected,
        brief_path=brief_path,
        brief=brief,
    )
