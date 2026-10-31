"""Shared filesystem path-confinement helper for built-in tools.

Document/media tools (ocr, pdf, docx, excel, image_info, qr_read,
audio_transcribe, data_analysis, image_caption, …) take a filesystem path. Left
unconfined, a tool argument is an arbitrary file *read* primitive — and tools
that then upload the bytes to a third-party API (OCR, captioning) turn that into
exfiltration under prompt injection.

Two layers, both built on ``FileOperations``' resolve-then-check model:

* **Default (no ``allowed_directories``):** a *deny-list* blocks the sensitive
  locations an attacker actually wants — ``/etc`` (``/etc/passwd``,
  ``/etc/shadow``), ``/proc``/``/sys``/``/dev``, ``/root``, kernel/boot, mounted
  secrets, and per-user credential stores (``~/.ssh``, ``~/.aws``, ``~/.gnupg``,
  ``~/.kube``, ``~/.config/gcloud``, ``~/.docker``, ``~/.netrc``, …). It also
  refuses a path whose *filename* matches a common credentials-file shape
  (``.env``, ``id_rsa``, ``credentials``, …), the same names ``bash_tool``
  already refuses to ``cat``. Ordinary files (a report under the project, a
  scan in a temp dir) still work, so the tools stay usable for their intended
  purpose.
* **Confined (``allowed_directories=[...]``):** the documented opt-in tightens to
  a strict allow-list — only those roots are readable (and the deny-list still
  applies). This is the ``FileOperations`` posture for callers that want it.

Resolution follows symlinks (``Path.resolve``), so a symlink pointing at a denied
location (or outside the allow-list) is rejected too.

The filename check does not catch a credentials file renamed to an innocuous
extension (``.env`` saved as ``.csv``). :func:`check_content_not_credentials`
covers that case: a caller reads the file's text and passes it through before
returning parsed data or extracted text to the model, so a dotenv-shaped
``KEY=VALUE`` block or a private-key header is refused regardless of the
extension that got it past the filename check.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: Environment variable naming a directory where file-writing tools operate by
#: default. When set, ``file_operations`` confines to it and ``bash`` runs there,
#: so generated files land in a dedicated location instead of the caller's
#: current directory. Unset (the default), tools use the current directory.
WORKSPACE_ENV_VAR = "EFFGEN_WORKSPACE"


def default_workspace() -> Path | None:
    """Resolve the configured workspace directory, or ``None`` when unset.

    Reads ``EFFGEN_WORKSPACE``. The directory is created (including parents) if it
    does not yet exist so tools can write into it immediately. Returns ``None``
    when the variable is unset, empty, or the directory cannot be created, in
    which case callers fall back to the current directory.
    """
    raw = os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as exc:
        logger.warning(
            "%s=%r could not be used as a workspace (%s); using the current "
            "directory instead.", WORKSPACE_ENV_VAR, raw, exc,
        )
        return None


class PathNotAllowedError(ValueError):
    """Raised when a file path is denied (sensitive location or outside roots).

    Subclasses ValueError so existing tool error handling catches it.
    """


# Filenames that commonly hold credentials, checked against the path itself
# (mirrors ``bash_tool.SECRET_FILE_PATTERNS``' filename coverage). A path
# match is refused regardless of which directory it resolves under.
_DENY_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\.env(?:\.[\w.-]+)?$", re.I),
    re.compile(r"^\.netrc$", re.I),
    re.compile(r"^\.pgpass$", re.I),
    re.compile(r"^id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?$", re.I),
    re.compile(r"^credentials$", re.I),
    re.compile(r"^\.git-credentials$", re.I),
)


def is_credential_filename(name: str) -> bool:
    """True when the bare filename *name* looks like a credentials file.

    Matches common credentials-file shapes (``.env``, ``id_rsa``,
    ``credentials``, ...) — the same coverage :data:`_DENY_FILENAME_PATTERNS` gives the deny-list default
    above, exposed for tools (e.g. ``FileOperations``) that enforce their own
    allow-/deny-list confinement but want the same filename awareness.
    """
    return any(pattern.match(name) for pattern in _DENY_FILENAME_PATTERNS)


# A file whose *content* looks like a credentials store is refused regardless
# of its extension — a decoy secrets file renamed to ``.csv``/``.txt`` keeps
# its filename-based check from firing, but not this one. dotenv-shaped
# ``KEY=VALUE`` lines and a PEM private-key header are checked directly.
_DOTENV_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*\S+[ \t]*$")
_CREDENTIAL_KEY_HINT = re.compile(
    r"(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD|CREDENTIAL|PRIVATE)", re.I
)
_PRIVATE_KEY_HEADER = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")


def looks_like_credentials_content(text: str) -> bool:
    """True if *text* reads like a credentials/secrets file, any extension.

    Two shapes are treated as a credentials store: a PEM private-key header,
    or a block of dotenv-style ``KEY=VALUE`` lines where most non-blank,
    non-comment lines fit that shape and at least one key name hints at a
    credential (``*_KEY``, ``*_SECRET``, ``*_TOKEN``, ...). The ratio/hint
    combination keeps an ordinary data file with a stray ``A=1``-looking cell
    from being mistaken for a secrets file.
    """
    if _PRIVATE_KEY_HEADER.search(text):
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return False
    dotenv_lines = [ln for ln in lines if _DOTENV_LINE.match(ln)]
    if len(dotenv_lines) < 2:
        return False
    ratio = len(dotenv_lines) / len(lines)
    has_hint = any(_CREDENTIAL_KEY_HINT.search(ln.split("=", 1)[0]) for ln in dotenv_lines)
    return ratio >= 0.6 and (has_hint or len(dotenv_lines) >= 3)


def check_content_not_credentials(text: str, *, source: str = "") -> None:
    """Raise :class:`PathNotAllowedError` if *text* looks like a credentials file.

    Call after reading a file's text content and before returning it (or data
    derived from it) to the caller, so a secrets file cannot slip through a
    filename-only check by being renamed to an innocuous extension.
    """
    if looks_like_credentials_content(text):
        where = f" '{source}'" if source else ""
        raise PathNotAllowedError(
            f"Refusing to return the content of{where}: it reads like a "
            "credentials file (dotenv-style KEY=VALUE lines or a private-key "
            "header), regardless of its extension. Point the tool at the file "
            "you meant, or strip the credentials from this one."
        )


# Absolute directory trees that never hold legitimate tool input and commonly
# hold secrets / kernel interfaces. Reads resolving under these are refused.
_DENY_ABS_DIRS: tuple[str, ...] = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/root",
    "/run/secrets",
    "/var/run/secrets",
)

# Per-user credential stores (resolved against the *owning* user's home as well
# as the current user's home). Relative to a home directory.
_DENY_HOME_SUBPATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".azure",
    ".config/gcloud",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
)


def _denied_roots() -> list[Path]:
    roots = [Path(d) for d in _DENY_ABS_DIRS]
    # Block credential stores under /home/*, /root and the current home, so a
    # path is refused regardless of which user it belongs to.
    home_bases = {Path.home()}
    for base in (Path("/home"), Path("/Users")):
        if base.is_dir():
            try:
                home_bases.update(p for p in base.iterdir() if p.is_dir())
            except OSError:
                # Can't enumerate home roots (perms/race) — safe to skip: the
                # current user's home and /root are already in the deny set, and
                # the absolute deny dirs (/etc, /proc, …) are unaffected.
                pass
    home_bases.add(Path("/root"))
    for base in home_bases:
        for sub in _DENY_HOME_SUBPATHS:
            roots.append(base / sub)
    return roots


def _is_denied(resolved: Path) -> bool:
    if any(pattern.match(resolved.name) for pattern in _DENY_FILENAME_PATTERNS):
        return True
    for root in _denied_roots():
        try:
            rroot = root.resolve()
        except (OSError, RuntimeError):
            rroot = root
        if resolved == rroot or rroot in resolved.parents:
            return True
    return False


def normalize_allowed_dirs(directories: list[str] | None) -> list[Path] | None:
    """Resolve configured roots, or ``None`` for the default deny-list posture.

    Returning ``None`` (no ``allowed_directories`` given) selects the deny-list
    default; a non-empty list selects strict allow-list confinement.
    """
    if not directories:
        return None
    normalized: list[Path] = []
    for dir_path in directories:
        try:
            path = Path(dir_path).resolve()
        except (OSError, RuntimeError):
            logger.warning("Ignoring invalid allowed directory: %s", dir_path)
            continue
        if path.exists() and path.is_dir():
            normalized.append(path)
        else:
            logger.warning(
                "Allowed directory does not exist or is not a directory: %s", dir_path
            )
    return normalized or None


def is_path_allowed(path: Path, allowed_dirs: list[Path] | None) -> bool:
    """True if ``path`` (resolved) is permitted under the active policy."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    if _is_denied(resolved):
        return False
    if allowed_dirs is None:
        return True  # deny-list mode: anything not denied is allowed
    return any(resolved == root or root in resolved.parents for root in allowed_dirs)


def confine_path(
    path: str,
    allowed_dirs: list[Path] | None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve and validate a file path against the active confinement policy.

    Args:
        path: The user/model-supplied path.
        allowed_dirs: ``None`` for the deny-list default, or roots from
            :func:`normalize_allowed_dirs` for strict allow-list confinement.
        must_exist: When True, also require the file to exist (and be a file).

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        PathNotAllowedError: If the path is denied or outside the roots.
        FileNotFoundError: If ``must_exist`` and the file is missing.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError) as e:
        raise PathNotAllowedError(
            f"Invalid path '{path}': {e}. Pass a path this process can "
            "resolve — check for a broken symlink or an unreadable parent "
            "directory."
        )

    if not is_path_allowed(resolved, allowed_dirs):
        if allowed_dirs is None:
            raise PathNotAllowedError(
                f"Refusing to read '{path}': it resolves to a protected system "
                "or credential location. Reading secrets via a tool argument is "
                "blocked. Copy what the tool genuinely needs to a working "
                "directory and pass that path instead."
            )
        raise PathNotAllowedError(
            f"Path '{path}' is outside the allowed directories "
            f"({[str(d) for d in allowed_dirs]}). Pass allowed_directories=[…] "
            "to this tool to permit reading from another location."
        )

    if must_exist:
        if not resolved.exists():
            raise FileNotFoundError(
                f"File not found: {path}. Check the path and that the file "
                "exists on this machine."
            )
        if not resolved.is_file():
            raise PathNotAllowedError(
                f"Path '{path}' is not a regular file. Pass a file rather than "
                "a directory, socket or device node."
            )

    return resolved
