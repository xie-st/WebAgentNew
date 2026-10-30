"""Environment checks for a coding run, shared by ``doctor`` and the session.

``effgen code`` needs three things from the machine it runs on: a workspace
directory it can write to, a sandbox backend that can execute what it writes,
and — for the confirmed commit — a git installation. :func:`code_readiness`
reads all three without calling a model, and returns them as
:class:`ReadinessCheck` records carrying a status, a detail line and, when
something is not ready, the exact fix.

Nothing here changes state: the workspace is inspected but never created, the
sandbox is probed but not resolved (so a run still picks its own backend and
prints its own warning), and every git call is a read.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effgen.tools.builtin._fs import WORKSPACE_ENV_VAR

__all__ = [
    "ReadinessCheck",
    "ReadinessReport",
    "code_readiness",
]

#: How long the Docker probe may take before it is reported as unreachable.
_DOCKER_PROBE_TIMEOUT = 5.0

#: How long `git --version` may take.
_GIT_TIMEOUT = 5


@dataclass
class ReadinessCheck:
    """One environment check and what it found.

    ``status`` is a short word for a table cell (``ready``, ``limited``,
    ``missing``, ``unavailable``); ``ok`` is False only when the check would stop
    a coding run from working, so a usable-but-weaker setup (the subprocess
    sandbox, a workspace outside a repository) reports ``ok`` with a ``fix`` that
    explains how to improve it.
    """

    name: str
    ok: bool
    status: str
    detail: str = ""
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a JSON-serializable dict."""
        return {
            "name": self.name,
            "ok": self.ok,
            "status": self.status,
            "detail": self.detail,
            "fix": self.fix,
        }


@dataclass
class ReadinessReport:
    """The full set of coding checks, plus the workspace they were read for."""

    workspace: str = ""
    checks: list[ReadinessCheck] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when every check that could stop a run passed."""
        return all(check.ok for check in self.checks)

    @property
    def blocked(self) -> list[ReadinessCheck]:
        """The checks that would stop a coding run."""
        return [check for check in self.checks if not check.ok]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as the JSON document ``doctor --json`` embeds."""
        return {
            "ready": self.ready,
            "workspace": self.workspace,
            "checks": [check.to_dict() for check in self.checks],
        }


def _workspace_target(explicit: str | None) -> tuple[Path, str]:
    """Return the directory a run would use and where that setting came from.

    Mirrors :func:`~effgen.cli.code.engine.resolve_workspace` — explicit flag,
    then ``EFFGEN_WORKSPACE``, then the current directory — without creating
    anything, so a check never leaves a directory behind.
    """
    if explicit:
        return Path(explicit).expanduser().absolute(), "-w/--workspace"
    configured = os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser().absolute(), WORKSPACE_ENV_VAR
    return Path.cwd().absolute(), "current directory"


def _workspace_check(explicit: str | None) -> tuple[ReadinessCheck, Path | None]:
    """Report whether the workspace a run would use can be written to."""
    workspace, source = _workspace_target(explicit)

    if not workspace.exists():
        parent = next((p for p in workspace.parents if p.exists()), None)
        if parent is not None and os.access(parent, os.W_OK):
            return (
                ReadinessCheck(
                    name="workspace",
                    ok=True,
                    status="ready",
                    detail=f"{workspace} (from {source}) — created on first use",
                ),
                None,
            )
        return (
            ReadinessCheck(
                name="workspace",
                ok=False,
                status="unavailable",
                detail=f"{workspace} (from {source}) does not exist and cannot be created",
                fix=f"Point {WORKSPACE_ENV_VAR} (or -w/--workspace) at a directory you can write to.",
            ),
            None,
        )

    if not workspace.is_dir():
        return (
            ReadinessCheck(
                name="workspace",
                ok=False,
                status="not a directory",
                detail=f"{workspace} (from {source}) is a file",
                fix=f"Point {WORKSPACE_ENV_VAR} (or -w/--workspace) at a directory.",
            ),
            None,
        )

    if not os.access(workspace, os.W_OK):
        return (
            ReadinessCheck(
                name="workspace",
                ok=False,
                status="read-only",
                detail=f"{workspace} (from {source}) is not writable",
                fix=f"Choose a writable directory with -w/--workspace or export {WORKSPACE_ENV_VAR}.",
            ),
            workspace,
        )

    return (
        ReadinessCheck(
            name="workspace",
            ok=True,
            status="ready",
            detail=f"{workspace} (from {source})",
        ),
        workspace,
    )


def _probe_docker() -> bool:
    """True when a Docker daemon answers within the probe timeout."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_DOCKER_PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _subprocess_isolation() -> tuple[bool, bool]:
    """Return ``(network_isolated, writes_confined)`` for the subprocess backend.

    Asks the backend's own probes, so the line this check prints is the
    isolation a run on this host would actually get. The two are independent:
    a host can isolate the network through a user namespace yet still refuse
    the locked read-only remounts that confine writes. Any error answers
    ``(False, False)`` — understating isolation is safe, overstating it is not.
    """
    import asyncio

    from effgen.security.sandbox import SubprocessSandbox

    async def _probe() -> tuple[bool, bool]:
        await SubprocessSandbox._probe_caps()
        unshare_bin = shutil.which("unshare")
        if not (unshare_bin and SubprocessSandbox._userns_ok):
            return False, False
        if not SubprocessSandbox._mountns_ok:
            return True, False
        await SubprocessSandbox._probe_confinement(unshare_bin)
        return True, SubprocessSandbox._confine_ok

    try:
        return asyncio.run(_probe())
    except Exception:  # noqa: BLE001 - a diagnostic never fails the run
        return False, False


def _subprocess_readiness(fix_when_confined: str) -> ReadinessCheck:
    """Describe the subprocess backend as this host can actually run it."""
    network_isolated, writes_confined = _subprocess_isolation()
    if writes_confined:
        return ReadinessCheck(
            name="sandbox",
            ok=True,
            status="limited",
            detail="subprocess — network isolated, writes confined to the workspace",
            fix=fix_when_confined,
        )
    if network_isolated:
        return ReadinessCheck(
            name="sandbox",
            ok=True,
            status="limited",
            detail=(
                "subprocess — network isolated, but this host cannot confine "
                "writes: executed code can write anywhere you can"
            ),
            fix=(
                "Install Docker, or run on a host with unprivileged user "
                "namespaces, to confine writes by executed code."
            ),
        )
    return ReadinessCheck(
        name="sandbox",
        ok=True,
        status="limited",
        detail=(
            "subprocess — this host has no user namespaces, so neither the "
            "network nor writes by executed code are isolated"
        ),
        fix=(
            "Install Docker, or enable unprivileged user namespaces "
            "(kernel.unprivileged_userns_clone=1), to isolate executed code."
        ),
    )


def _sandbox_check() -> ReadinessCheck:
    """Report which backend would run the code a session writes.

    The configured backend is read from the environment and probed directly
    rather than through :func:`~effgen.security.sandbox.get_sandbox`, so the
    check never caches a backend or emits the fallback warning a real run owns.
    """
    configured = os.environ.get("EFFGEN_SANDBOX_BACKEND", "").strip().lower() or "auto"

    if configured == "off":
        return ReadinessCheck(
            name="sandbox",
            ok=False,
            status="disabled",
            detail="EFFGEN_SANDBOX_BACKEND=off — executed code runs unconfined",
            fix="Unset EFFGEN_SANDBOX_BACKEND (or set it to docker) so executed code is confined.",
        )

    if configured == "docker":
        if _probe_docker():
            return ReadinessCheck(
                name="sandbox",
                ok=True,
                status="ready",
                detail="docker — executed code has no network and its own filesystem",
            )
        return ReadinessCheck(
            name="sandbox",
            ok=False,
            status="unavailable",
            detail="EFFGEN_SANDBOX_BACKEND=docker but the Docker daemon did not answer",
            fix="Start Docker, or unset EFFGEN_SANDBOX_BACKEND to fall back to the subprocess sandbox.",
        )

    if configured == "subprocess":
        return _subprocess_readiness(
            "Install Docker and unset EFFGEN_SANDBOX_BACKEND to confine reads as well."
        )

    if configured == "firecracker":
        # The name resolves, but the backend is a stub that reports itself
        # unavailable, so a run pinned to it fails at the first execution.
        return ReadinessCheck(
            name="sandbox",
            ok=False,
            status="unavailable",
            detail="EFFGEN_SANDBOX_BACKEND=firecracker is not implemented in this release",
            fix="Set EFFGEN_SANDBOX_BACKEND to docker or subprocess, or unset it.",
        )

    if configured != "auto":
        return ReadinessCheck(
            name="sandbox",
            ok=False,
            status="unknown",
            detail=f"EFFGEN_SANDBOX_BACKEND={configured} is not a backend effGen knows",
            fix="Set EFFGEN_SANDBOX_BACKEND to docker or subprocess, or unset it.",
        )

    if _probe_docker():
        return ReadinessCheck(
            name="sandbox",
            ok=True,
            status="ready",
            detail="docker — executed code has no network and its own filesystem",
        )
    return _subprocess_readiness(
        "Install Docker to confine reads as well; the subprocess sandbox is used until then."
    )


def _git_check(workspace: Path | None) -> ReadinessCheck:
    """Report whether git is installed and whether the workspace is in a repository."""
    if shutil.which("git") is None:
        return ReadinessCheck(
            name="git",
            ok=True,
            status="missing",
            detail="git is not installed — a session runs without repository context",
            fix="Install git to get branch/status context and the confirmed --commit action.",
        )

    version = ""
    try:
        proc = subprocess.run(
            ["git", "--version"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        if proc.returncode == 0:
            version = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = ""

    if workspace is None:
        return ReadinessCheck(
            name="git",
            ok=True,
            status="ready",
            detail=version or "git is installed",
        )

    from .project import detect_repo

    repo = detect_repo(workspace)
    if repo is None:
        return ReadinessCheck(
            name="git",
            ok=True,
            status="no repository",
            detail=f"{version or 'git is installed'}; {workspace} is not inside a repository",
            fix="Run effgen code from a repository (or git init the workspace) for branch/status context.",
        )
    return ReadinessCheck(
        name="git",
        ok=True,
        status="ready",
        detail=f"{repo.root} on {repo.branch_label}"
               + (f", {repo.changed} path(s) changed" if repo.dirty else ", clean"),
    )


def code_readiness(workspace: str | None = None) -> ReadinessReport:
    """Read the environment a coding run needs and return what it found.

    *workspace* is an explicit directory (what ``-w/--workspace`` would set);
    without one the same resolution a run uses applies — ``EFFGEN_WORKSPACE``,
    then the current directory. No model is called and nothing is written.
    """
    target, _source = _workspace_target(workspace)
    workspace_check, existing = _workspace_check(workspace)
    checks = [workspace_check, _sandbox_check(), _git_check(existing)]
    return ReadinessReport(workspace=str(target), checks=checks)
