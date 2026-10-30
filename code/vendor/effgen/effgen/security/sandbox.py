"""
effgen.security.sandbox — Sandbox backends for CodeExecutor.

Threat model: attacker-controlled prompt → agent emits arbitrary Python/shell →
execution must be isolated from the host.

Backends (in order of preference):
  DockerSandbox     — Docker container with --read-only --network=none --cap-drop=ALL
                      --pids-limit=100 --memory=256m. Default when Docker is reachable.
  SubprocessSandbox — fork/exec with ulimit + unshare(1) on Linux. Used when Docker is
                      unavailable; emits a loud warning. Inside its mount namespace every
                      mount is remounted read-only except the run's scratch space (the
                      configured workdir, else the workspace directory, else the calling
                      process's directory) and a private /tmp and /dev/shm, so executed
                      code writes only inside that scratch space. Reads are not confined.
  FirecrackerSandbox — Stub only; not implemented in this release.

Selection:
  Auto (default): DockerSandbox if Docker daemon is reachable; SubprocessSandbox + warning otherwise.
  Override via env var EFFGEN_SANDBOX_BACKEND=docker|subprocess|firecracker|off
  Timeout via EFFGEN_SANDBOX_TIMEOUT=<seconds>  (default 10)

  EFFGEN_SANDBOX_BACKEND=off disables sandboxing entirely. This is UNSAFE: code
  runs directly on the host with the privileges of the effGen process. A loud
  warning is emitted whenever the off backend is used. Intended only for trusted
  code in controlled environments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shlex
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """Environment-driven sandbox configuration."""

    backend: str = field(
        default_factory=lambda: os.environ.get("EFFGEN_SANDBOX_BACKEND", "auto")
    )
    timeout: int = field(
        default_factory=lambda: int(os.environ.get("EFFGEN_SANDBOX_TIMEOUT", "10"))
    )
    memory_limit: str = field(
        default_factory=lambda: os.environ.get("EFFGEN_SANDBOX_MEMORY", "256m")
    )
    # Docker-specific
    docker_image: str = field(
        default_factory=lambda: os.environ.get(
            "EFFGEN_SANDBOX_IMAGE", "effgen/sandbox:python-3.11"
        )
    )
    # Fallback image if custom image not available
    docker_fallback_image: str = "python:3.11-slim"
    network_enabled: bool = False
    #: Working directory for the executed process. When set, a relative path in
    #: the code resolves against this directory, so code can read and write the
    #: files a caller placed there. Applies to the subprocess and off backends;
    #: DockerSandbox does not mount the host filesystem, so it ignores this.
    #: ``None`` (the default) leaves the calling process's directory in effect.
    workdir: str | None = None

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        """Create config from environment variables."""
        return cls()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """Result of a sandboxed code execution.

    :attr:`filesystem_confined` and :attr:`writable_root` report what the
    backend that ran this call actually enforced, so a caller never has to
    assume a guarantee the environment could not provide:

    ==============================  ====================  ==================
    Backend state                   ``filesystem_confined``  ``writable_root``
    ==============================  ====================  ==================
    docker                          ``True``              ``None``
    subprocess, namespaces present  ``True``              the scratch root
    subprocess, degraded            ``False``             ``None``
    off                             ``False``             ``None``
    ==============================  ====================  ==================
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    timed_out: bool = False
    backend_used: str = ""
    error: str | None = None
    #: True when writes by the executed code were confined by the backend.
    filesystem_confined: bool = False
    #: The only directory writable by the executed code, when the backend
    #: confined writes to a host path. ``None`` when the backend used its own
    #: filesystem (docker) or did not confine writes at all.
    writable_root: str | None = None
    #: True when the executed code had no outbound network access.
    network_isolated: bool = False
    #: True when the executed code ran in its own PID namespace with its own
    #: ``/proc``, so it saw only its own processes. When ``False`` the host
    #: process table was visible and ``/proc/<pid>/cmdline`` / ``environ`` for
    #: the calling user's other processes could be read.
    process_table_isolated: bool = False
    #: True when the per-user credential stores were masked for this run.
    #: Deliberately **not** called ``reads_confined``: reads are not confined,
    #: they are denied at named locations (``~/.ssh``, ``~/.aws``, ``~/.gnupg``,
    #: ``~/.kube``, ``~/.docker``, ``~/.azure``, ``~/.config/gcloud``, the
    #: credential files beside them, ``/etc/shadow`` and the mounted-secret
    #: directories). Executed code can still read ordinary files the calling
    #: user can read. Reported so a caller never assumes more than was enforced.
    credential_reads_masked: bool = False


@dataclass
class _CommandPlan:
    """The command to run plus what isolation it will actually enforce."""

    argv: list[str]
    network_isolated: bool = False
    filesystem_confined: bool = False
    writable_root: str | None = None
    credential_reads_masked: bool = False
    process_table_isolated: bool = False
    #: Directory to advertise as the temp directory, when the sandbox could not
    #: mount a private ``/tmp`` over the host one.
    tmpdir_override: str | None = None


def _resolve_workdir(config: SandboxConfig) -> str | None:
    """Return the directory to start the executed process in, or ``None``.

    A configured :attr:`SandboxConfig.workdir` is used only when it is an
    existing directory; a missing or unreadable one is reported once and the
    calling process's directory is used instead, so a stale path never turns
    every execution into a spawn failure.
    """
    workdir = config.workdir
    if not workdir:
        return None
    if os.path.isdir(workdir):
        return workdir
    logger.warning(
        "Sandbox workdir %r is not a directory; running in the current "
        "directory instead.", workdir,
    )
    return None


def _is_within(root: str, path: str) -> bool:
    """True when *path* is *root* itself or lives underneath it."""
    try:
        root_r = os.path.realpath(root).rstrip(os.sep) or os.sep
        path_r = os.path.realpath(path).rstrip(os.sep) or os.sep
    except OSError:
        return False
    return path_r == root_r or path_r.startswith(root_r + os.sep)


def _scratch_root(config: SandboxConfig) -> str | None:
    """Return the one directory executed code may write into, or ``None``.

    Resolved with the same rule the file and shell tools use, so executed code
    and the tools that wrote the files agree on a single root:
    :attr:`SandboxConfig.workdir`, else the configured workspace directory
    (``EFFGEN_WORKSPACE``), else the calling process's own directory.

    ``None`` means no root can be claimed — the candidate is missing, is not a
    directory, or is the filesystem root itself (confining writes to ``/``
    would confine nothing). Callers treat ``None`` as "do not claim
    confinement".
    """
    candidate = _resolve_workdir(config)
    if candidate is None:
        workspace = None
        try:
            from ..tools.builtin._fs import default_workspace
            workspace = default_workspace()
        except Exception:  # noqa: BLE001 - the workspace helper is optional here
            workspace = None
        if workspace is not None:
            candidate = str(workspace)
        else:
            try:
                candidate = os.getcwd()
            except OSError:
                return None
    try:
        root = os.path.realpath(candidate)
    except OSError:
        return None
    if root == os.sep:
        logger.warning(
            "Sandbox writes cannot be confined: the run directory is the "
            "filesystem root. Set EFFGEN_WORKSPACE to a project directory to "
            "narrow it."
        )
        return None
    if not os.path.isdir(root):
        return None
    return root


#: Per-user credential stores masked inside the sandbox. A deny-list, not a
#: confinement: executed code can still read ordinary files the calling user
#: can read. What it closes is the chain that makes such a read more than local
#: — an agent holding ``code_executor`` commonly also holds an outbound tool
#: that runs *outside* the sandbox, so a secret read inside can be printed to
#: stdout and handed out on the next turn. These are the paths that chain
#: actually wants. Kept in step with ``effgen/tools/builtin/_fs.py``, which
#: applies the same list to the *file* tools.
_SANDBOX_DENY_HOME_DIRS: tuple[str, ...] = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gcloud",
)
_SANDBOX_DENY_HOME_FILES: tuple[str, ...] = (
    ".netrc", ".git-credentials", ".npmrc", ".pypirc",
)
_SANDBOX_DENY_ABS_FILES: tuple[str, ...] = ("/etc/shadow", "/etc/gshadow")
_SANDBOX_DENY_ABS_DIRS: tuple[str, ...] = ("/run/secrets", "/var/run/secrets")


def _deny_reads_script() -> str:
    """Shell that masks the credential stores before the read-only remounts.

    Runs as root in the sandbox's own user and mount namespace, so each step is
    an unprivileged bind in a private namespace and nothing on the host changes.
    A directory is covered by an empty tmpfs and a file by ``/dev/null``, so a
    read succeeds and returns nothing rather than failing in a way that tells an
    attacker the path exists.

    This is a **deny-list**, and the docstring of the backend says so: reads are
    not confined, they are masked at the locations that matter.

    Returns:
        The shell prologue, or ``""`` when no masking can be set up.
    """
    dirs = " ".join(shlex.quote(d) for d in _SANDBOX_DENY_HOME_DIRS)
    files = " ".join(shlex.quote(f) for f in _SANDBOX_DENY_HOME_FILES)
    abs_files = " ".join(shlex.quote(f) for f in _SANDBOX_DENY_ABS_FILES)
    abs_dirs = " ".join(shlex.quote(d) for d in _SANDBOX_DENY_ABS_DIRS)
    return (
        '__deny=$(mktemp -d 2>/dev/null) || __deny=""; '
        'if [ -n "$__deny" ]; then mount -t tmpfs none "$__deny" 2>/dev/null; fi; '
        'if [ -n "$__deny" ]; then '
        'for __h in /root /home/* "$HOME"; do '
        '[ -d "$__h" ] || continue; '
        f"for __s in {dirs}; do "
        '[ -d "$__h/$__s" ] && mount --bind "$__deny" "$__h/$__s" 2>/dev/null; '
        "done; "
        f"for __s in {files}; do "
        '[ -f "$__h/$__s" ] && mount --bind /dev/null "$__h/$__s" 2>/dev/null; '
        "done; "
        "done; "
        f"for __p in {abs_dirs}; do "
        '[ -d "$__p" ] && mount --bind "$__deny" "$__p" 2>/dev/null; '
        "done; "
        "fi; "
        f"for __p in {abs_files}; do "
        '[ -f "$__p" ] && mount --bind /dev/null "$__p" 2>/dev/null; '
        "done; "
    )


def _confine_script(root: str, private_tmp: bool) -> str:
    """Build the shell prologue that makes *root* the only writable directory.

    Runs as root inside the sandbox's own user namespace, so every step is an
    unprivileged mount operation on the private mount namespace:

    1. bind *root* onto itself, giving it a mount entry of its own;
    2. remount every other mount read-only, skipping ``/proc`` (the nested
       ``unshare`` writes ``uid_map`` there) and mounts that already are;
    3. mount a fresh ``tmpfs`` over ``/tmp`` (unless *root* lives under the
       temp directory, where that would hide the workspace) and over
       ``/dev/shm``, so temp files and shared memory keep working;
    4. ``cd`` into *root* — the inherited working directory was opened before
       the bind, so relative writes would otherwise land on the read-only
       mount underneath.

    The caller appends ``exec unshare --map-root-user --mount <interpreter>``:
    handing the interpreter a nested user and mount namespace is what makes the
    kernel mark these mounts locked. Without it the executed code — which is
    root in the outer namespace — simply remounts them read-write again.
    """
    quoted = shlex.quote(root)
    parts = [
        f"mount --bind {quoted} {quoted} 2>/dev/null; ",
        # Mask the credential stores first, while globbing is still on and
        # before the read-only pass, which then covers the new mounts too.
        _deny_reads_script(),
        # No globbing while the mountinfo line is split into fields.
        "set -f; ",
        "while IFS= read -r __l; do ",
        'set -- $__l; ',
        # Field 5 is the mount point (\040-escaped), field 6 its options.
        '__m=$(printf "%b" "$5"); ',
        'case "$__m" in /proc|/proc/*) continue;; esac; ',
        'case "$6" in ro|ro,*) continue;; esac; ',
        f'if [ "$__m" = {quoted} ]; then continue; fi; ',
        'mount -o remount,bind,ro "$__m" 2>/dev/null; ',
        "done < /proc/self/mountinfo; ",
        "set +f; ",
    ]
    if private_tmp:
        parts.append("mount -t tmpfs none /tmp 2>/dev/null; ")
    parts.append("mount -t tmpfs none /dev/shm 2>/dev/null; ")
    parts.append(
        f"cd {quoted} 2>/dev/null || "
        "{ echo 'sandbox: the writable directory could not be entered' >&2; "
        "exit 1; }; "
    )
    return "".join(parts)


def _killed_at_deadline(exit_code: int | None, elapsed: float, timeout: float) -> bool:
    """True when a process was killed by a signal at/after its time budget.

    The subprocess backends enforce the wall-clock timeout two ways: an outer
    ``asyncio.wait_for`` (raises ``TimeoutError``) and an inner CPU-time
    ``ulimit -t`` that ``SIGXCPU``/``SIGKILL``s a busy loop *before* the outer
    guard fires. A negative exit code (signal kill) that lands at or after the
    requested deadline is therefore a timeout, not a clean exit — report it
    explicitly instead of as a silent success. (An OOM from ``ulimit -v`` surfaces
    as a Python ``MemoryError`` with a positive exit code, so it is not matched
    here.)
    """
    if exit_code is None or exit_code >= 0 or timeout <= 0:
        return False
    return elapsed >= timeout


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SandboxBase(ABC):
    """Abstract sandbox backend interface."""

    name: str = "base"

    @abstractmethod
    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Execute *code* in the sandbox and return a SandboxResult.

        Args:
            code: The source to execute.
            language: The language *code* is written in.
            config: Limits and isolation settings for the execution.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if this backend can be used in the current environment."""

    # ------------------------------------------------------------------
    # Helpers shared by all backends
    # ------------------------------------------------------------------

    _CODE_FILENAMES: dict[str, str] = {
        "python": "script.py",
        "javascript": "script.js",
        "bash": "script.sh",
        "sh": "script.sh",
    }

    _EXEC_COMMANDS: dict[str, list[str]] = {
        "python": ["python3", "script.py"],
        "javascript": ["node", "script.js"],
        "bash": ["bash", "script.sh"],
        "sh": ["sh", "script.sh"],
    }

    def _code_filename(self, language: str) -> str:
        return self._CODE_FILENAMES.get(language, "script.txt")

    def _exec_command(self, language: str) -> list[str]:
        return self._EXEC_COMMANDS.get(language, ["bash", "script.sh"])


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------

_DOCKER_IMAGES: dict[str, str] = {
    "python": "python:3.11-slim",
    "javascript": "node:20-slim",
    "bash": "bash:5",
    "sh": "bash:5",
}


class DockerSandbox(SandboxBase):
    """
    Run code inside a Docker container with strict isolation:

    - ``--read-only``         container FS is read-only (tmpfs for /tmp)
    - ``--network=none``      no outbound network connectivity
    - ``--cap-drop=ALL``      all Linux capabilities dropped
    - ``--no-new-privileges`` prevent privilege escalation via setuid/setgid
    - ``--pids-limit=100``    cap process/thread count
    - ``--memory=<limit>``    memory cap (default 256m)
    - ``--rm``                auto-remove container on exit

    The sandbox image is configurable via ``EFFGEN_SANDBOX_IMAGE``; falls back
    to the stock ``python:3.11-slim`` (or language-appropriate image) if the
    custom image is not available.
    """

    name = "docker"

    async def is_available(self) -> bool:
        """Return True if Docker daemon is reachable."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except (FileNotFoundError, PermissionError, OSError):
            return False

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Run *code* in a one-shot Docker container and return the captured result.

        Args:
            code: The source to execute.
            language: The language *code* is written in.
            config: Limits and isolation settings for the container.
        """
        start = time.monotonic()
        tmp_dir = Path(tempfile.mkdtemp(prefix="effgen_sandbox_"))
        try:
            # Write code to temp dir
            fname = self._code_filename(language)
            script_path = tmp_dir / fname
            script_path.write_text(code, encoding="utf-8")

            # The container process may run under a *different* uid than this
            # process — e.g. a Docker daemon with userns-remap, or an image whose
            # default USER is non-root. ``mkdtemp`` makes the dir 0700 and the
            # file 0600 (owner-only), so that uid cannot traverse the dir or read
            # the bind-mounted script and Python aborts with "[Errno 13]
            # Permission denied". Make the ephemeral, read-only-mounted code dir
            # and file world-readable/traversable so any container uid can read
            # them. Safe: the dir is a per-run temp holding only the user's own
            # code, is mounted read-only, and is removed in the finally below.
            os.chmod(tmp_dir, 0o755)
            os.chmod(script_path, 0o644)

            # Decide image
            image = await self._resolve_image(language, config)

            # Build docker run command
            cmd = self._build_docker_cmd(
                tmp_dir=tmp_dir,
                image=image,
                language=language,
                config=config,
            )

            logger.debug("DockerSandbox cmd: %s", " ".join(cmd))

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=config.timeout + 5
                )
                exit_code = proc.returncode
                timed_out = False
            except TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                stdout_b, stderr_b = b"", b"Execution timed out"
                exit_code = -1
                timed_out = True

            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                execution_time=time.monotonic() - start,
                timed_out=timed_out,
                backend_used="docker",
                # The container runs --read-only on its own filesystem and no
                # host path is mounted writable, so there is no host root to
                # name. Isolation itself is unchanged.
                filesystem_confined=True,
                network_isolated=not config.network_enabled,
            )

        except Exception as exc:
            logger.error("DockerSandbox error: %s", exc)
            return SandboxResult(
                exit_code=-1,
                execution_time=time.monotonic() - start,
                backend_used="docker",
                error=str(exc),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_image(self, language: str, config: SandboxConfig) -> str:
        """
        Pick the container image.

        The custom sandbox image (``effgen/sandbox:python-3.11``) only ships a
        Python runtime, so it is used solely for ``python``. Other languages
        always use their language-appropriate stock image.
        """
        if language == "python":
            custom = config.docker_image
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "image", "inspect", custom,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                if proc.returncode == 0:
                    return custom
            except Exception:
                pass
        # Fall back to language-appropriate stock image
        return _DOCKER_IMAGES.get(language, "python:3.11-slim")

    def _build_docker_cmd(
        self,
        tmp_dir: Path,
        image: str,
        language: str,
        config: SandboxConfig,
    ) -> list[str]:
        """Assemble the ``docker run`` command."""
        exec_cmd = self._exec_command(language)

        cmd: list[str] = [
            "docker", "run",
            "--rm",
            # Filesystem isolation
            "--read-only",
            "--tmpfs", "/tmp:size=64m,noexec",
            # Mount code (read-only)
            "-v", f"{tmp_dir}:/workspace:ro",
            "-w", "/workspace",
            # Security
            "--network", "none" if not config.network_enabled else "bridge",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            # Resource limits
            "--pids-limit", "100",
            "--memory", config.memory_limit,
            "--memory-swap", config.memory_limit,  # disable swap
            "--cpus", "1",
        ]

        cmd.append(image)
        cmd.extend(exec_cmd)
        return cmd


# ---------------------------------------------------------------------------
# SubprocessSandbox
# ---------------------------------------------------------------------------

class SubprocessSandbox(SandboxBase):
    """
    Best-effort isolation via subprocess with resource limits.

    On Linux, execution is wrapped in an *unprivileged* user namespace via
    ``unshare --map-root-user`` combined with:

    - ``--net``   a fresh network namespace with no interfaces → outbound
                  network is blocked (DNS + connect fail).
    - ``--mount`` a private mount namespace in which every mount is remounted
                  **read-only** except the run's scratch space, which is bound
                  back read-write, plus a fresh ``tmpfs`` over ``/tmp`` and
                  ``/dev/shm``. The interpreter is then handed off through a
                  *nested* ``unshare --map-root-user --mount``, which makes the
                  kernel lock those mounts: executed code cannot remount them
                  read-write or unmount them, from its own namespace or a
                  nested one it creates.
    - ``ulimit``  address-space, CPU-time and process-count caps.

    The scratch space is :attr:`SandboxConfig.workdir`, else the configured
    workspace directory (``EFFGEN_WORKSPACE``), else the calling process's own
    directory — the same root the file and shell tools use, so executed code
    can write the files the agent just created and nothing else.

    Code is fed to the interpreter over **stdin** (``python3 -`` / ``bash -s``),
    so no script file is written to the host filesystem.

    Capability probing happens once per process. Each isolation primitive
    degrades independently: if user namespaces are unavailable the sandbox
    falls back to ``ulimit``-only mode and emits a warning describing exactly
    which protections are NOT in effect. Write confinement is probed by running
    the real command and checking that a write inside the scratch space
    succeeds while one outside it fails; anything short of that leaves
    confinement off rather than claimed. Every result reports what was actually
    enforced in ``filesystem_confined`` / ``writable_root``.

    **Caveats / limitations:**
    - **Reads are not confined, but the credential stores are masked.** The
      per-user credential directories and files (``~/.ssh``, ``~/.aws``,
      ``~/.gnupg``, ``~/.kube``, ``~/.docker``, ``~/.azure``,
      ``~/.config/gcloud``, ``~/.netrc``, ``~/.git-credentials``, ``~/.npmrc``,
      ``~/.pypirc``), ``/etc/shadow`` and the mounted-secret directories are
      covered inside the sandbox — a directory by an empty tmpfs, a file by
      ``/dev/null`` — so a read succeeds and returns nothing. It is a deny-list,
      reported as ``SandboxResult.credential_reads_masked``, not confinement.
    - **Everything else is readable.** Executed code can read every other file the calling
      user can read. Use DockerSandbox when reads must be confined too.
    - ``/proc`` stays writable, because the nested ``unshare`` needs to write
      ``/proc/self/uid_map`` — but it is the sandbox's **own** ``/proc``: the
      run gets a private PID namespace (``--pid --fork --mount-proc``), so
      executed code sees only its own process and the writes reach nothing
      outside. Reported as ``SandboxResult.process_table_isolated``; a host that
      cannot create the namespace degrades to the shared process table and says
      so rather than claiming isolation.
    - Requires unprivileged user namespaces to be enabled
      (``kernel.unprivileged_userns_clone=1`` / ``user.max_user_namespaces>0``).
      Without them nothing is confined and a warning says so.
    - Memory limit is advisory (via ``ulimit -v``), not hard-enforced like cgroups.
    - Should only be used when Docker is unavailable.

    A loud warning is emitted at startup when this backend is selected as the
    default.
    """

    name = "subprocess"

    # Executables and the flag that makes each read program text from stdin.
    _LANG_STDIN_CMD: dict[str, list[str]] = {
        "python": ["python3", "-"],
        "javascript": ["node", "-"],
        "bash": ["bash", "-s"],
        "sh": ["sh", "-s"],
    }

    # Probed once per process: does ``unshare`` support each isolation mode?
    _caps_probed: bool = False
    _userns_ok: bool = False
    _mountns_ok: bool = False
    #: Whether a private PID namespace with its own ``/proc`` can be created.
    #: Without it the sandbox keeps the host process table visible, which is a
    #: read exposure (``/proc/<pid>/cmdline`` and ``environ`` can carry a
    #: secret), not a write one — so it degrades rather than refusing to run.
    _pidns_ok: bool = False

    # Probed once per process: does the read-only remount recipe actually hold?
    _confine_probed: bool = False
    _confine_ok: bool = False

    async def is_available(self) -> bool:
        """Always available (subprocess is a Python built-in)."""
        return True

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Run *code* in an isolated subprocess (user-namespace confinement when available).

        Args:
            code: The source to execute.
            language: The language *code* is written in.
            config: Limits and isolation settings for the subprocess.

        Returns:
            The captured output, exit status and timing of the run.
        """
        start = time.monotonic()
        try:
            plan = await self._build_cmd(language, config)
            cmd = plan.argv
            logger.debug(
                "SubprocessSandbox cmd: %s (net=%s, fs=%s, root=%s)",
                " ".join(cmd), plan.network_isolated,
                plan.filesystem_confined, plan.writable_root,
            )

            env = self._build_env(plan.tmpdir_override)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=_resolve_workdir(config),
                )
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")),
                    timeout=config.timeout + 2,
                )
                exit_code = proc.returncode
                elapsed = time.monotonic() - start
                # The CPU-time ulimit can SIGKILL a busy loop before the outer
                # wait_for fires; surface that as a timeout, not a clean exit.
                timed_out = _killed_at_deadline(exit_code, elapsed, config.timeout)
                if timed_out and not stderr_b:
                    stderr_b = b"Execution timed out"
            except TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                stdout_b, stderr_b = b"", b"Execution timed out"
                exit_code = -1
                timed_out = True

            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code if exit_code is not None else -1,
                execution_time=time.monotonic() - start,
                timed_out=timed_out,
                backend_used="subprocess",
                filesystem_confined=plan.filesystem_confined,
                writable_root=plan.writable_root,
                network_isolated=plan.network_isolated,
                credential_reads_masked=plan.credential_reads_masked,
                process_table_isolated=plan.process_table_isolated,
            )
        except Exception as exc:
            logger.error("SubprocessSandbox error: %s", exc)
            return SandboxResult(
                exit_code=-1,
                execution_time=time.monotonic() - start,
                backend_used="subprocess",
                error=str(exc),
            )

    # ------------------------------------------------------------------

    async def _build_cmd(
        self, language: str, config: SandboxConfig
    ) -> _CommandPlan:
        """Assemble the execution command and report what it will enforce."""
        stdin_cmd = self._LANG_STDIN_CMD.get(language, ["python3", "-"])
        interpreter = " ".join(stdin_cmd)
        mem_kb = self._parse_mem_kb(config.memory_limit)
        ulimit_prefix = (
            f"ulimit -v {mem_kb} -t {config.timeout} -u 256 2>/dev/null; "
        )
        plain = _CommandPlan(argv=["bash", "-c", f"{ulimit_prefix}exec {interpreter}"])

        if platform.system() != "Linux":
            return plain

        await self._probe_caps()
        unshare_bin = shutil.which("unshare")

        if not (unshare_bin and self._userns_ok):
            # Degraded: no user namespaces. ulimit-only, no net/fs isolation.
            logger.warning(
                "SubprocessSandbox: unprivileged user namespaces unavailable; "
                "running in ulimit-only mode. Network is NOT isolated and "
                "filesystem writes are NOT confined. Install Docker or enable "
                "kernel.unprivileged_userns_clone for stronger isolation."
            )
            return plain

        want_net = not config.network_enabled
        pid_isolated = self._pidns_ok
        root: str | None = None
        if self._mountns_ok:
            root = _scratch_root(config)
            if root is not None:
                await self._probe_confinement(unshare_bin)
                if not self._confine_ok:
                    _warn_confinement_degraded()
                    root = None

        inner = ulimit_prefix
        tmpdir_override: str | None = None
        if root is not None:
            # A scratch space under the system temp directory would be hidden by
            # a private tmpfs over /tmp; skip it there and point temp-file APIs
            # at the scratch space instead. Host /tmp is read-only either way.
            private_tmp = not _is_within(tempfile.gettempdir(), root)
            if not private_tmp:
                tmpdir_override = root
            inner += _confine_script(root, private_tmp)
            # The nested namespace is what locks the mounts above in place.
            inner += f"exec {unshare_bin} --map-root-user --mount {interpreter}"
        else:
            if self._mountns_ok:
                # /tmp gets a fresh tmpfs so host /tmp is untouched.
                inner += "mount -t tmpfs none /tmp 2>/dev/null; "
            inner += f"exec {interpreter}"

        unshare_flags = [unshare_bin, "--map-root-user"]
        if self._mountns_ok:
            unshare_flags.append("--mount")
        if want_net:
            unshare_flags.append("--net")
        # A private process table and a ``/proc`` to match. Without it executed
        # code reads the host's ``/proc/<pid>/cmdline`` and ``environ``, either
        # of which can carry a credential the caller never handed it. ``--fork``
        # is required: the process that creates a PID namespace does not enter
        # it, so ``unshare`` forks a child to be PID 1, and the kill still lands
        # because the timeout targets the process *group* from outside.
        if pid_isolated:
            unshare_flags += ["--pid", "--fork", "--mount-proc"]
        return _CommandPlan(
            argv=unshare_flags + ["bash", "-c", inner],
            network_isolated=want_net,
            process_table_isolated=pid_isolated,
            filesystem_confined=root is not None,
            writable_root=root,
            # Masked by the same prologue that confines writes, so it holds
            # exactly when that one ran.
            credential_reads_masked=root is not None,
            tmpdir_override=tmpdir_override,
        )

    @classmethod
    async def _probe_caps(cls) -> None:
        """Probe (once) whether unprivileged user/mount namespaces work."""
        if cls._caps_probed:
            return
        cls._userns_ok = await cls._unshare_succeeds(["--map-root-user", "true"])
        cls._mountns_ok = await cls._unshare_succeeds(
            ["--map-root-user", "--mount", "true"]
        )
        # A private process table needs the PID namespace *and* a ``/proc`` of
        # its own; ``--mount-proc`` is what supplies the second, and it is the
        # half a hardened runtime is most likely to withhold.
        cls._pidns_ok = cls._mountns_ok and await cls._unshare_succeeds(
            ["--map-root-user", "--mount", "--pid", "--fork", "--mount-proc", "true"]
        )
        cls._caps_probed = True

    @classmethod
    async def _probe_confinement(cls, unshare_bin: str) -> None:
        """Probe (once) whether write confinement really holds in this environment.

        Availability is not "does ``unshare`` exist": the recipe needs
        ``mount(8)``, a mount namespace *and* a nested user namespace, any of
        which a hardened container or a restricted runtime can withhold. So run
        the real command over two throwaway directories and require both halves
        of the contract — a write inside the scratch space succeeds, a write
        outside it fails. Anything else leaves confinement off, because
        remounts that executed code can undo are a boundary in name only.
        """
        if cls._confine_probed:
            return
        cls._confine_probed = True
        cls._confine_ok = False
        parent = None
        proc = None
        try:
            parent = tempfile.mkdtemp(prefix="effgen_confine_probe_")
            inside = os.path.join(parent, "in")
            outside = os.path.join(parent, "out")
            os.mkdir(inside)
            os.mkdir(outside)
            # ``outside`` sits under the system temp directory, so this also
            # checks that the host temp directory is read-only.
            body = (
                f"touch {shlex.quote(os.path.join(inside, 'x'))} 2>/dev/null "
                "|| exit 3; "
                f"touch {shlex.quote(os.path.join(outside, 'y'))} 2>/dev/null "
                "&& exit 4; "
                "exit 0"
            )
            inner = _confine_script(inside, private_tmp=False) + (
                f"exec {unshare_bin} --map-root-user --mount "
                f"bash -c {shlex.quote(body)}"
            )
            proc = await asyncio.create_subprocess_exec(
                unshare_bin, "--map-root-user", "--mount", "bash", "-c", inner,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
            cls._confine_ok = proc.returncode == 0
        except Exception as exc:  # noqa: BLE001 - a failed probe degrades, never raises
            logger.debug("Sandbox confinement probe failed: %s", exc)
            cls._confine_ok = False
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
        finally:
            if parent:
                shutil.rmtree(parent, ignore_errors=True)

    @staticmethod
    async def _unshare_succeeds(args: list[str]) -> bool:
        """Return True if ``unshare <args>`` exits 0 in this environment."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "unshare", *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _parse_mem_kb(mem_str: str) -> int:
        """Convert Docker-style memory string (e.g. '256m', '1g') to KB."""
        mem_str = mem_str.strip().lower()
        if mem_str.endswith("g"):
            return int(float(mem_str[:-1]) * 1024 * 1024)
        if mem_str.endswith("m"):
            return int(float(mem_str[:-1]) * 1024)
        if mem_str.endswith("k"):
            return int(float(mem_str[:-1]))
        return int(mem_str)  # assume bytes, convert to KB approximately

    @staticmethod
    def _build_env(tmpdir_override: str | None = None) -> dict[str, str]:
        """Return a minimal environment stripping sensitive vars.

        *tmpdir_override* names the writable directory temp-file APIs should
        use. It is set when the sandbox could not give the run a private
        ``/tmp``, so ``tempfile.mkstemp``, ``mktemp`` and ``os.tmpdir()`` keep
        working instead of hitting the read-only host temp directory.
        """
        safe_keys = {
            "PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE",
            "TERM", "USER", "LOGNAME", "SHELL",
            "TZ", "TMPDIR", "TEMP", "TMP",
            "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
        }
        env = {k: v for k, v in os.environ.items() if k in safe_keys}
        if tmpdir_override:
            env["TMPDIR"] = tmpdir_override
            env["TEMP"] = tmpdir_override
            env["TMP"] = tmpdir_override
        # Harden Python
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env


# ---------------------------------------------------------------------------
# FirecrackerSandbox (stub)
# ---------------------------------------------------------------------------

class FirecrackerSandbox(SandboxBase):
    """
    Stub interface for Firecracker MicroVM isolation.

    **Not implemented in this release.**

    Firecracker provides VM-level isolation with sub-second boot times,
    making it suitable for multi-tenant code execution with stronger guarantees
    than Docker containers. Implementation requires:
    - Firecracker binary + KVM access on the host
    - A minimal guest rootfs (e.g., Alpine-based initrd)
    - jailer for additional isolation
    - vsock or REST API for code injection + result retrieval

    This stub exists to define the interface and allow future implementation
    without breaking the sandbox dispatch machinery.
    """

    name = "firecracker"

    async def is_available(self) -> bool:
        """Not yet implemented — always returns False."""
        return False

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Not implemented: raises to report the Firecracker backend as unusable.

        Args:
            code: The source that would be executed.
            language: The language *code* is written in.
            config: Limits and isolation settings that would apply.

        Raises:
            NotImplementedError: This backend has no implementation.
        """
        raise NotImplementedError(
            "FirecrackerSandbox is not yet implemented in this release. "
            "Use 'docker' or 'subprocess' backend instead."
        )


# ---------------------------------------------------------------------------
# OffSandbox (explicit, unsafe)
# ---------------------------------------------------------------------------

class OffSandbox(SandboxBase):
    """NO sandbox: executes code directly on the host with the effGen process's privileges.

    **This is unsafe and must only be enabled explicitly** via
    ``EFFGEN_SANDBOX_BACKEND=off``. It is never selected by auto-resolution.
    A loud warning is emitted on every execution.

    Code is fed to the interpreter over stdin; ``ulimit`` caps are still applied
    on Linux as a courtesy, but provide no real isolation.
    """

    name = "off"

    _warned: bool = False

    async def is_available(self) -> bool:
        """Always available — but unsafe."""
        return True

    async def run(
        self,
        code: str,
        language: str,
        config: SandboxConfig,
    ) -> SandboxResult:
        """Run *code* directly on the host with no isolation.

        Args:
            code: The source to execute.
            language: The language *code* is written in.
            config: Limits that still apply, such as the timeout.

        Returns:
            The captured output, exit status and timing of the run.
        """
        if not OffSandbox._warned:
            logger.warning(
                "\n"
                "┌─────────────────────────────────────────────────────────────┐\n"
                "│  effGen SANDBOX DISABLED (EFFGEN_SANDBOX_BACKEND=off)         │\n"
                "│                                                              │\n"
                "│  Code is executing DIRECTLY ON THE HOST with no isolation.   │\n"
                "│  Network, filesystem, and resources are NOT restricted.      │\n"
                "│  Only use this for fully trusted code in controlled envs.    │\n"
                "└─────────────────────────────────────────────────────────────┘"
            )
            OffSandbox._warned = True

        start = time.monotonic()
        stdin_cmd = SubprocessSandbox._LANG_STDIN_CMD.get(language, ["python3", "-"])
        interpreter = " ".join(stdin_cmd)
        if platform.system() == "Linux":
            mem_kb = SubprocessSandbox._parse_mem_kb(config.memory_limit)
            cmd = [
                "bash", "-c",
                (f"ulimit -v {mem_kb} -t {config.timeout} -u 256 2>/dev/null; "
                f"exec {interpreter}"),
            ]
        else:
            cmd = ["bash", "-c", f"exec {interpreter}"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_resolve_workdir(config),
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")),
                    timeout=config.timeout + 2,
                )
                exit_code = proc.returncode
                elapsed = time.monotonic() - start
                timed_out = _killed_at_deadline(exit_code, elapsed, config.timeout)
                if timed_out and not stderr_b:
                    stderr_b = b"Execution timed out"
            except TimeoutError:
                proc.kill()
                await proc.wait()
                stdout_b, stderr_b = b"", b"Execution timed out"
                exit_code = -1
                timed_out = True

            return SandboxResult(
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                exit_code=exit_code if exit_code is not None else -1,
                execution_time=time.monotonic() - start,
                timed_out=timed_out,
                backend_used="off",
                filesystem_confined=False,
                writable_root=None,
                network_isolated=False,
            )
        except Exception as exc:
            logger.error("OffSandbox error: %s", exc)
            return SandboxResult(
                exit_code=-1,
                execution_time=time.monotonic() - start,
                backend_used="off",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Sandbox factory / dispatch
# ---------------------------------------------------------------------------

_BACKEND_MAP: dict[str, type[SandboxBase]] = {
    "docker": DockerSandbox,
    "subprocess": SubprocessSandbox,
    "firecracker": FirecrackerSandbox,
    "off": OffSandbox,
}

# Cache the resolved backend so availability checks run once per process
_resolved_backend: SandboxBase | None = None

# Emitted once per process, regardless of whether SubprocessSandbox was reached
# via "auto" fallback or an explicit EFFGEN_SANDBOX_BACKEND=subprocess.
_subprocess_fallback_warned: bool = False

# Emitted once per process when namespaces exist but write confinement does not.
_confinement_degraded_warned: bool = False


def _warn_subprocess_fallback() -> None:
    global _subprocess_fallback_warned
    if _subprocess_fallback_warned:
        return
    _subprocess_fallback_warned = True
    logger.warning(
        "\n"
        "┌─────────────────────────────────────────────────────────────┐\n"
        "│  effGen SANDBOX WARNING                                      │\n"
        "│                                                              │\n"
        "│  Code execution is using SubprocessSandbox, which provides   │\n"
        "│  PARTIAL isolation only.                                     │\n"
        "│                                                              │\n"
        "│  Limitations:                                                │\n"
        "│  • Executed code can READ any host file the calling          │\n"
        "│    process's user can read — reads are NOT confined          │\n"
        "│  • Writes are confined to the run's working directory only   │\n"
        "│    where the kernel allows it; each result reports what was  │\n"
        "│    enforced in 'filesystem_confined' / 'writable_root'       │\n"
        "│  • Network isolation via unshare (may require privileges)    │\n"
        "│  • Memory limit is advisory, not hard-enforced               │\n"
        "│                                                              │\n"
        "│  To confine reads as well, install Docker and ensure the     │\n"
        "│  daemon is running and accessible by the current user.       │\n"
        "└─────────────────────────────────────────────────────────────┘\n"
    )


def _warn_confinement_degraded() -> None:
    """Say once that writes are not confined, so no caller assumes they are."""
    global _confinement_degraded_warned
    if _confinement_degraded_warned:
        return
    _confinement_degraded_warned = True
    logger.warning(
        "SubprocessSandbox: this environment cannot confine writes by executed "
        "code (the read-only remount or the nested user namespace was "
        "refused), so executed code can write anywhere the calling user can. "
        "Results report filesystem_confined=False. Install Docker, or run on a "
        "host with unprivileged user namespaces, to confine writes."
    )


async def get_sandbox(config: SandboxConfig | None = None) -> SandboxBase:
    """
    Return the best available sandbox backend.

    Resolution order when ``config.backend == "auto"`` (the default):

    1. DockerSandbox — preferred; strong isolation.
    2. SubprocessSandbox — fallback; emits ``WARNING`` on first use.

    A ``WARNING`` is also emitted the first time ``SubprocessSandbox`` is
    resolved via an explicit ``EFFGEN_SANDBOX_BACKEND=subprocess``, since that
    path skips Docker probing entirely but carries the same filesystem
    exposure as the auto-fallback case.

    The resolved backend is cached for the lifetime of the process.
    """
    global _resolved_backend

    if config is None:
        config = SandboxConfig.from_env()

    if _resolved_backend is not None and config.backend == "auto":
        return _resolved_backend

    backend_name = config.backend.lower()

    if backend_name != "auto":
        cls = _BACKEND_MAP.get(backend_name)
        if cls is None:
            raise ValueError(
                f"Unknown EFFGEN_SANDBOX_BACKEND={backend_name!r}. "
                f"Valid choices: {sorted(_BACKEND_MAP)}"
            )
        # An explicitly pinned backend is cached for the process lifetime too:
        # reuse the resolved instance when it already matches the requested class.
        if isinstance(_resolved_backend, cls):
            return _resolved_backend
        backend = cls()
        if not await backend.is_available():
            raise RuntimeError(
                f"Sandbox backend {backend_name!r} is not available in this environment."
            )
        if cls is SubprocessSandbox:
            _warn_subprocess_fallback()
        _resolved_backend = backend
        return backend

    # Auto-selection
    docker = DockerSandbox()
    if await docker.is_available():
        logger.info("Sandbox: using DockerSandbox (--network=none --cap-drop=ALL).")
        _resolved_backend = docker
        return docker

    # Fallback
    _warn_subprocess_fallback()
    sub = SubprocessSandbox()
    _resolved_backend = sub
    return sub


def reset_sandbox_cache() -> None:
    """Reset the cached sandbox backend, probes and warning state (useful for testing)."""
    global _resolved_backend, _subprocess_fallback_warned
    global _confinement_degraded_warned
    _resolved_backend = None
    _subprocess_fallback_warned = False
    _confinement_degraded_warned = False
    SubprocessSandbox._caps_probed = False
    SubprocessSandbox._userns_ok = False
    SubprocessSandbox._mountns_ok = False
    SubprocessSandbox._pidns_ok = False
    SubprocessSandbox._confine_probed = False
    SubprocessSandbox._confine_ok = False


__all__ = [
    "SandboxConfig",
    "SandboxResult",
    "SandboxBase",
    "DockerSandbox",
    "SubprocessSandbox",
    "FirecrackerSandbox",
    "OffSandbox",
    "get_sandbox",
    "reset_sandbox_cache",
]
