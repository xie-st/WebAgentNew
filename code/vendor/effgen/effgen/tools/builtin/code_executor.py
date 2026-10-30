"""
Code execution sandbox tool.

This module provides an isolated code execution environment supporting
multiple languages (Python, JavaScript, Bash), resource limits, and security
measures whose strength depends on the resolved backend.

Sandbox dispatch:
  - DockerSandbox (default when Docker daemon is available): confines both
      filesystem and network —
      --read-only --network=none --cap-drop=ALL --pids-limit=100 --memory=256m
  - SubprocessSandbox (fallback): ulimit + unshare on Linux. Isolates the
      network and confines writes to the working directory; the rest of the
      host filesystem is readable but read-only. See
      ``effgen.security.sandbox.SubprocessSandbox``'s docstring.
  - Backend configurable via EFFGEN_SANDBOX_BACKEND=docker|subprocess
  - Timeout configurable via EFFGEN_SANDBOX_TIMEOUT=<seconds>

Every result carries ``sandbox_backend``, ``filesystem_confined`` and
``writable_root``, so a caller reads what the run actually enforced instead of
assuming a guarantee the host could not provide.
"""

from __future__ import annotations

import logging
from typing import Any

from ...security.sandbox import (
    SandboxBase,
    SandboxConfig,
    SubprocessSandbox,
    get_sandbox,
)
from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


class CodeExecutionError(Exception):
    """Raised when code execution fails."""
    pass


class CodeExecutor(BaseTool):
    """
    Sandboxed code execution tool.

    Features:
    - Multi-language support (Python, JavaScript, Bash)
    - Docker-based isolation (primary)
    - Subprocess fallback with ulimit + unshare
    - Resource limits (CPU, memory, time)
    - Network isolation (default: off)
    - Output capture (stdout, stderr, return values)
    - Comprehensive error handling

    Security:
    - Filesystem AND network isolation via DockerSandbox when Docker is
      available. Without Docker, the SubprocessSandbox fallback isolates the
      network and confines writes to the working directory; the rest of the
      host filesystem stays readable but read-only, so reads are not confined.
      A ``WARNING`` is logged the first time the fallback is used; check
      ``result["sandbox_backend"]`` to see which one ran, and
      ``result["filesystem_confined"]`` / ``result["writable_root"]`` for what
      it enforced.
    - Configurable resource limits
    - Network restricted by default
    - Timeout mechanisms
    - Automatic cleanup

    Configuration (env vars):
    - EFFGEN_SANDBOX_BACKEND: docker | subprocess | auto (default: auto)
    - EFFGEN_SANDBOX_TIMEOUT: seconds (default: 10)
    - EFFGEN_SANDBOX_MEMORY: memory limit (default: 256m)
    - EFFGEN_SANDBOX_IMAGE: custom Docker image (default: effgen/sandbox:python-3.11)
    """

    SUPPORTED_LANGUAGES = ["python", "javascript", "bash", "sh"]

    # Default resource limits
    DEFAULT_TIMEOUT = 30           # seconds (tool-level; sandbox timeout is shorter)
    DEFAULT_MEMORY_LIMIT = "256m"  # 256 MB
    DEFAULT_MAX_OUTPUT = 102400    # 100 KB

    # Minimum memory limit per language. Some runtimes reserve a large virtual
    # address space at startup regardless of the heap they use — node/V8 reserves
    # a CodeRange that exceeds the 256 MB default enforced by the subprocess
    # sandbox's ``ulimit -v``, so the interpreter aborts before user code runs.
    # A language listed here is given at least this much so it starts under the
    # default; an explicit higher limit is left untouched.
    LANGUAGE_MIN_MEMORY = {
        "javascript": "1g",
    }

    # Docker images for different languages (used by DockerSandbox fallback)
    DOCKER_IMAGES = {
        "python": "python:3.11-slim",
        "javascript": "node:20-slim",
        "bash": "bash:5",
        "sh": "bash:5",
    }

    def __init__(self, workdir: str | None = None) -> None:
        """Initialize the code executor.

        Args:
            workdir: Directory the executed process starts in, so a relative
                path in the code resolves against it. Defaults to the
                ``EFFGEN_WORKSPACE`` directory when that is set — the same root
                ``file_operations`` writes into and ``bash`` runs in, so code
                written by an agent and then executed sees its own files.
                Applies to the subprocess and off backends; DockerSandbox does
                not mount the host filesystem, so it is unaffected.
        """
        super().__init__(
            metadata=ToolMetadata(
                name="code_executor",
                description=(
                    "Execute code in an isolated environment with support for "
                    "Python, JavaScript, and Bash. Uses Docker (network-disabled, "
                    "read-only, capability-dropped) when the Docker daemon is "
                    "reachable — this confines the filesystem and network. "
                    "Without Docker, falls back to a subprocess sandbox that "
                    "isolates the network and allows writes ONLY inside the "
                    "working directory: the code starts there, so a relative "
                    "path reads and writes the files there, while every path "
                    "outside it is read-only and a write to one fails with a "
                    "read-only-filesystem error. Reading anywhere the user can "
                    "read still works, and CPU/process limits are enforced by "
                    "ulimit, not cgroups. The result reports which backend ran "
                    "('sandbox_backend'), whether writes were confined "
                    "('filesystem_confined') and the directory writes are "
                    "allowed in ('writable_root')."
                ),
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="The code to execute",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="language",
                        type=ParameterType.STRING,
                        description="Programming language to use",
                        required=True,
                        enum=["python", "javascript", "bash", "sh"],
                    ),
                    ParameterSpec(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description="Execution timeout in seconds",
                        required=False,
                        default=30,
                        min_value=1,
                        max_value=300,
                    ),
                    ParameterSpec(
                        name="memory_limit",
                        type=ParameterType.STRING,
                        description="Memory limit (e.g., '512m', '1g')",
                        required=False,
                        default="256m",
                    ),
                    ParameterSpec(
                        name="network_enabled",
                        type=ParameterType.BOOLEAN,
                        description="Whether to allow network access (default: False)",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="files",
                        type=ParameterType.OBJECT,
                        description="Additional files to make available (filename: content)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="env_vars",
                        type=ParameterType.OBJECT,
                        description="Environment variables to set (not supported in Docker mode)",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "exit_code": {"type": "integer"},
                        "execution_time": {"type": "number"},
                        "timed_out": {"type": "boolean"},
                        "sandbox_backend": {"type": "string"},
                        "filesystem_confined": {"type": "boolean"},
                        "writable_root": {"type": ["string", "null"]},
                    },
                },
                timeout_seconds=300,
                tags=["code", "execution", "sandbox", "docker", "security"],
                examples=[
                    {
                        "code": "print('Hello, World!')",
                        "language": "python",
                        "output": {
                            "stdout": "Hello, World!\n",
                            "exit_code": 0,
                            "sandbox_backend": "docker",
                        },
                    },
                    {
                        "code": "console.log('Hello from Node.js')",
                        "language": "javascript",
                        "output": {
                            "stdout": "Hello from Node.js\n",
                            "exit_code": 0,
                            "sandbox_backend": "docker",
                        },
                    },
                ],
            )
        )
        self._sandbox: SandboxBase | None = None
        self._sandbox_config: SandboxConfig = SandboxConfig.from_env()
        self._workdir = workdir

    def _resolve_workdir(self) -> str | None:
        """Return the directory executed code should start in, or ``None``.

        Read at call time (not at construction) so a workspace configured after
        the tool was created still takes effect.
        """
        if self._workdir:
            return self._workdir
        from ._fs import default_workspace
        workspace = default_workspace()
        return str(workspace) if workspace is not None else None

    async def initialize(self) -> None:
        """Initialize the executor and resolve the sandbox backend."""
        await super().initialize()
        try:
            self._sandbox = await get_sandbox(self._sandbox_config)
            logger.info(
                "CodeExecutor initialized with sandbox backend: %s",
                self._sandbox.name,
            )
        except Exception as exc:
            logger.error("Failed to initialize sandbox: %s", exc)
            # Hard fail — we must not execute unsandboxed code silently
            raise RuntimeError(
                f"CodeExecutor could not initialize a sandbox backend: {exc}"
            ) from exc

    async def _execute(
        self,
        code: str,
        language: str,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        network_enabled: bool = False,
        files: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute code in the configured sandbox.

        Args:
            code: Source code to execute.
            language: One of "python", "javascript", "bash", "sh".
            timeout: Wall-clock timeout in seconds (overrides sandbox default).
            memory_limit: Memory cap (e.g. "256m").
            network_enabled: Allow outbound network (default False).
            files: Extra files to write alongside the code (DockerSandbox: ignored).
            env_vars: Extra env vars (SubprocessSandbox only).

        Returns:
            Dict with stdout, stderr, exit_code, execution_time, timed_out,
            sandbox_backend, filesystem_confined and writable_root.
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise CodeExecutionError(
                f"Unsupported language: {language!r}. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )

        # Lazily initialize if not yet done (e.g. tool used without explicit init)
        if self._sandbox is None:
            await self.initialize()

        # Raise the memory cap to a language-appropriate floor when the requested
        # limit is below what the runtime needs to start (see LANGUAGE_MIN_MEMORY).
        memory_limit = self._resolve_memory_limit(language, memory_limit)

        # Build a per-call config (override from call params)
        call_config = SandboxConfig(
            backend=self._sandbox_config.backend,
            timeout=timeout,
            memory_limit=memory_limit,
            docker_image=self._sandbox_config.docker_image,
            docker_fallback_image=self._sandbox_config.docker_fallback_image,
            network_enabled=network_enabled,
            workdir=self._resolve_workdir(),
        )

        try:
            result = await self._sandbox.run(
                code=code,
                language=language,
                config=call_config,
            )
        except Exception as exc:
            logger.error("Sandbox execution error: %s", exc)
            raise CodeExecutionError(
                f"Execution failed: {exc}. Check the code for the error "
                "named above, then run it again."
            ) from exc

        if result.error:
            raise CodeExecutionError(
                f"Sandbox error: {result.error}. Check that the sandbox "
                "backend is available (`effgen doctor` reports it), then run "
                "the code again."
            )

        out: dict[str, Any] = {
            "stdout": result.stdout[: self.DEFAULT_MAX_OUTPUT],
            "stderr": result.stderr[: self.DEFAULT_MAX_OUTPUT],
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "timed_out": result.timed_out,
            "sandbox_backend": result.backend_used,
            "filesystem_confined": result.filesystem_confined,
            "writable_root": result.writable_root,
        }
        # The tool result mirrors the program's own outcome: it succeeds only when
        # the process exited 0 without timing out. A non-zero exit — an uncaught
        # exception, sys.exit(n), a syntax error, a non-zero shell command, or a
        # runtime abort such as an out-of-memory — is reported as a failure so a
        # caller keyed on the result does not treat a crash as a success. The full
        # stdout/stderr/exit_code stay in the payload on every path.
        succeeded = (result.exit_code == 0) and not result.timed_out
        out["success"] = succeeded
        if not succeeded:
            out["error"] = self._failure_message(language, result, timeout)
        return out

    def _resolve_memory_limit(self, language: str, memory_limit: str) -> str:
        """Raise ``memory_limit`` to the language floor when it is set too low.

        Returns the requested limit unchanged for languages without a floor or
        when the request already meets it.
        """
        floor = self.LANGUAGE_MIN_MEMORY.get(language)
        if not floor:
            return memory_limit
        try:
            requested_kb = SubprocessSandbox._parse_mem_kb(memory_limit)
            floor_kb = SubprocessSandbox._parse_mem_kb(floor)
        except (ValueError, AttributeError):
            return memory_limit
        if requested_kb < floor_kb:
            logger.info(
                "Raising memory_limit for %s from %s to %s "
                "(the runtime needs more address space to start)",
                language, memory_limit, floor,
            )
            return floor
        return memory_limit

    def _failure_message(self, language: str, result: Any, timeout: int) -> str:
        """Build an actionable one-line reason for a non-successful execution."""
        if result.timed_out:
            return (
                f"Execution timed out after {timeout}s "
                f"(sandbox '{result.backend_used}' killed the process)"
            )
        stderr = (result.stderr or "").strip()
        if language == "javascript" and "reserve virtual memory" in stderr.lower():
            return (
                "JavaScript runtime ran out of address space under the current "
                "memory_limit; raise memory_limit (node/V8 needs about 1g)"
            )
        base = f"Code exited with status {result.exit_code}"
        writable_root = getattr(result, "writable_root", None)
        if writable_root and self._is_read_only_filesystem_error(stderr):
            # Without the writable root named, the caller sees a bare
            # "[Errno 30] Read-only file system" and retries the same path.
            return (
                f"{base}: writes are confined to {writable_root}; write inside "
                "it (the rest of the filesystem is read-only)"
            )
        detail = self._stderr_summary(stderr)
        return f"{base}: {detail}" if detail else base

    @staticmethod
    def _is_read_only_filesystem_error(stderr: str) -> bool:
        """True when the run failed because it wrote outside the writable root."""
        lowered = stderr.lower()
        return (
            "read-only file system" in lowered
            or "errno 30" in lowered
            or "erofs" in lowered
        )

    @staticmethod
    def _stderr_summary(stderr: str) -> str:
        """Pick the most informative single line from a traceback/stderr dump."""
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
        # Drop the interpreter version footer node prints after an error.
        lines = [ln for ln in lines if not ln.startswith("Node.js v")]
        if not lines:
            return ""
        # Prefer an explicit error/exception line over a stack frame ("at ...",
        # 'File "...", line N'); Python puts it last, node puts it mid-dump.
        err_lines = [
            ln for ln in lines
            if ("error" in ln.lower() or "exception" in ln.lower())
            and not ln.startswith(("at ", "File "))
        ]
        return err_lines[-1] if err_lines else lines[-1]
