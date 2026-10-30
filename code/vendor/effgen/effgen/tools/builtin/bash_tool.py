"""
Bash/Shell command execution tool with security controls.

This module provides a tool for executing shell commands with
configurable security: allowed/blocked command lists, timeout,
environment variable filtering, and working directory control.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import signal
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)

_POSIX = os.name == "posix"

# Shell metacharacters that require a real shell to interpret (pipes,
# redirects, substitution, globbing, etc.). Commands without any of these are
# executed via argv (no shell), which is safer and avoids shell-injection
# surprises.
_SHELL_METACHARS = re.compile(r"[|&;<>$`(){}\[\]*?~!\n\\]")

# Default blocked commands/patterns that are dangerous
DEFAULT_BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/*",
    "dd",
    "mkfs",
    "fdisk",
    "parted",
    "mount",
    "umount",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    "systemctl",
    "service",
    "iptables",
    "nft",
    "useradd",
    "userdel",
    "usermod",
    "passwd",
    "chown -R /",
    "chmod -R 777 /",
    ":(){ :|:& };:",  # fork bomb
}

# Patterns that indicate dangerous intent
DANGEROUS_PATTERNS = [
    r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+/\s*$",  # rm -rf /
    r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+/\*",  # rm -rf /*
    r">\s*/dev/sd",  # overwrite disk devices
    r"mkfs\.",  # format filesystems
    r"dd\s+.*of=/dev/",  # dd to devices
    r":\(\)\s*\{",  # fork bomb
    r"\|\s*sh\b",  # piping to shell (potential injection)
    r"\|\s*bash\b",  # piping to bash
    r"curl\s+.*\|\s*(sh|bash)",  # curl | sh pattern
    r"wget\s+.*\|\s*(sh|bash)",  # wget | sh pattern
    r"eval\s+",  # eval
    r"`.*`",  # command substitution in backticks
    r"\$\(.*\)",  # command substitution  $(cmd) including nested
    r"\$\{[^}]*\$\(",  # parameter expansion with embedded substitution ${VAR:-$(cmd)}
    r"\$\{[^}]*`",  # parameter expansion with backtick substitution ${VAR:-`cmd`}
    r"<<[- \t]*\w+",  # heredoc injection (<<EOF, <<-EOF, << MARKER)
    r"<\(.*\)",  # process substitution input <(cmd)
    r">\(.*\)",  # process substitution output >(cmd)
]

# Common secret/credential files. Commands that reference these are refused so a
# shell command cannot read what the environment strip protects (the env-strip
# would be pointless if `cat .env` still worked). This is a best-effort deterrent
# against a direct, literal reference to one of these paths — not a guarantee: a
# shell tool that runs with the calling user's full privileges can be routed
# around a text match by an attacker willing to obfuscate the filename (quoted
# string concatenation, `$(printf ...)`-built names, decode-to-file-then-execute).
# The checks below close the concatenation and decode/execute obfuscations found
# in practice; a fully general defense would need to actually parse shell
# semantics, which this tool does not attempt.
SECRET_FILE_PATTERNS = [
    re.compile(r"(?:^|[\s/'\"=:;,])\.env(?:\.[\w.-]+)?(?:$|[\s'\"/:;,])", re.I),  # .env / .env.local
    re.compile(r"\.aws/credentials", re.I),
    re.compile(r"\.ssh/", re.I),
    re.compile(r"\bid_(?:rsa|dsa|ecdsa|ed25519)\b", re.I),
    re.compile(r"\.netrc\b", re.I),
    re.compile(r"\.pgpass\b", re.I),
    re.compile(r"\.kube/config\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

# A dotfile glob (`.e*`, `.en?`, `.ssh*`) sidesteps the literal-name patterns
# above while still matching credential files a shell would expand to. Refused
# whenever a wildcard is glued directly onto a dot-prefixed name.
_SECRET_FILE_GLOB_PATTERN = re.compile(r"(?:^|[\s/])\.[\w.-]*[*?\[]")

# `base64 -d`/`xxd -r`/`openssl ... -d` decodes an opaque payload to a file that
# is then executed in the same command — the outer command text never contains
# the decoded content, so the patterns above (which match on literal filenames)
# cannot see what the payload does. Blocked as a pattern in its own right,
# independent of what the decoded script turns out to do.
_DECODE_PATTERN = re.compile(
    r"\b(?:base64\s+(?:-d\b|--decode\b)|xxd\s+-r\b|openssl\s+(?:base64\s+-d\b|enc\s+.*-d\b))"
)
_EXEC_INTERPRETER_PATTERN = re.compile(r"\b(?:bash|sh|zsh|python3?|perl|ruby|node|source)\s+\S")

# Sensitive environment variables to strip. This explicit list is a floor; the
# generic name-pattern check in ``_is_sensitive_env_name`` strips anything that
# *looks* like a credential too, so providers added later are covered without
# editing this set.
SENSITIVE_ENV_VARS = {
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "SERPAPI_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "REPLICATE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "DATABASE_URL",
    "DB_PASSWORD",
    "SECRET_KEY",
    "PRIVATE_KEY",
    "SSH_PRIVATE_KEY",
}

# Generic credential-name patterns: any env var whose name ends with one of these
# suffixes, or contains one of these substrings, is treated as a secret and
# stripped from the shell environment — so a newly added provider key
# (``*_API_KEY`` / ``*_API_TOKEN`` / ``*SECRET*`` …) is covered automatically,
# never silently leaked because the explicit list drifted.
_SECRET_NAME_SUFFIXES = (
    "_API_KEY",
    "_API_TOKEN",
    "_API_SECRET",
    "_ACCESS_KEY",
    "_ACCESS_TOKEN",
    "_SECRET_KEY",
    "_AUTH_TOKEN",
    "_SECRET",
    "_TOKEN",
)
_SECRET_NAME_SUBSTRINGS = (
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "API_KEY",
)


def _is_sensitive_env_name(name: str) -> bool:
    """True if an env var name looks like a credential and must not be exposed."""
    upper = name.upper()
    if upper.endswith(_SECRET_NAME_SUFFIXES):
        return True
    return any(sub in upper for sub in _SECRET_NAME_SUBSTRINGS)


class BashTool(BaseTool):
    """
    Shell command execution tool with security controls.

    Features:
    - Configurable allowed/blocked command lists
    - Command timeout (default: 30s)
    - Output capture (stdout + stderr)
    - Working directory control
    - Environment variable filtering (strips sensitive vars)
    - Dangerous command detection and blocking

    Security:
    - Blocks dangerous commands by default (rm -rf /, dd, mkfs, etc.)
    - Optional allowed_commands whitelist mode
    - Strips sensitive environment variables
    - Command timeout to prevent hanging

    Resource limits (configurable):
    - CPU time: ``timeout`` seconds (default 30)
    - Output size: ``max_output_size`` bytes (default 100 KB)
    """

    # Standardized defaults (shared across execution tools)
    DEFAULT_TIMEOUT = 30          # seconds
    DEFAULT_MAX_OUTPUT = 102400   # 100 KB

    def __init__(
        self,
        allowed_commands: list[str] | None = None,
        blocked_commands: list[str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        working_directory: str | None = None,
        strip_env_vars: set[str] | None = None,
        allow_command_substitution: bool = False,
        max_output_size: int = DEFAULT_MAX_OUTPUT,
    ) -> None:
        """
        Initialize the Bash tool.

        Args:
            allowed_commands: If set, ONLY these commands can be run (whitelist mode).
            blocked_commands: Additional commands to block (added to defaults).
            timeout: Command timeout in seconds (default: 30).
            working_directory: Working directory for commands. Default: the
                ``EFFGEN_WORKSPACE`` directory when that environment variable is
                set, otherwise the current directory.
            strip_env_vars: Extra env vars to strip (added to defaults).
            allow_command_substitution: Allow $() and `` in commands (default: False).
            max_output_size: Maximum output size in bytes (default: 100 KB).
        """
        super().__init__(
            metadata=ToolMetadata(
                name="bash",
                description=(
                    "Run a shell command with the calling user's full privileges. "
                    "This is NOT a sandbox — do not enable it for untrusted input. "
                    "Dangerous commands and reads of common secret files (.env, "
                    "~/.ssh, ~/.aws/credentials) are refused, but a shell can still "
                    "touch anything the user can. Use it for system commands, listing "
                    "files, checking system info, and processing text with CLI tools."
                ),
                category=ToolCategory.SYSTEM,
                parameters=[
                    ParameterSpec(
                        name="command",
                        type=ParameterType.STRING,
                        description="The shell command to execute",
                        required=True,
                        min_length=1,
                        max_length=2000,
                    ),
                    ParameterSpec(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description=(
                            "Per-call wall-clock timeout in seconds, overriding the "
                            "tool default for this command only. Values <= 0 fall "
                            "back to the default."
                        ),
                        required=False,
                        default=None,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "return_code": {"type": "integer"},
                        "command": {"type": "string"},
                    },
                },
                timeout_seconds=timeout,
                tags=["bash", "shell", "system", "command"],
                examples=[
                    {
                        "command": "ls -la",
                        "output": {"stdout": "total 32\ndrwxr-xr-x ...", "return_code": 0},
                    },
                    {
                        "command": "echo 'Hello World'",
                        "output": {"stdout": "Hello World\n", "return_code": 0},
                    },
                    {
                        "command": "wc -l *.py",
                        "output": {"stdout": "  42 main.py\n  18 utils.py\n  60 total", "return_code": 0},
                    },
                ],
            )
        )

        self.allowed_commands = set(allowed_commands) if allowed_commands else None
        self.blocked_commands = set(DEFAULT_BLOCKED_COMMANDS)
        if blocked_commands:
            self.blocked_commands.update(blocked_commands)
        self.timeout = timeout
        self.working_directory = working_directory
        self.strip_env_vars = SENSITIVE_ENV_VARS.copy()
        if strip_env_vars:
            self.strip_env_vars.update(strip_env_vars)
        self.allow_command_substitution = allow_command_substitution
        self.max_output_size = max_output_size

    def _is_command_safe(self, command: str) -> tuple:
        """
        Check if a command is safe to execute.

        Returns:
            (is_safe, reason) tuple
        """
        cmd_stripped = command.strip()

        # Check blocked commands (exact match)
        for blocked in self.blocked_commands:
            if cmd_stripped == blocked or cmd_stripped.startswith(blocked + " "):
                return False, f"Command blocked: '{blocked}' is in the blocked commands list"

        # Refuse commands that reference common secret files, so the env-strip
        # can't be sidestepped by reading credentials off disk instead. Also
        # check a quote-stripped view of the command: adjacent quoted string
        # literals (`".e""nv"`) concatenate under the shell into a filename
        # that never appears as a literal substring of the raw command text.
        quote_stripped = command.replace('"', "").replace("'", "")
        for pattern in SECRET_FILE_PATTERNS:
            if pattern.search(command) or pattern.search(quote_stripped):
                return False, (
                    "Command blocked: references a sensitive credential file "
                    "(.env, ~/.ssh, ~/.aws/credentials, private keys)"
                )
        if _SECRET_FILE_GLOB_PATTERN.search(command):
            return False, (
                "Command blocked: a wildcard against a dotfile can expand to a "
                "credential file (.env, .ssh, .netrc, ...)"
            )
        if _DECODE_PATTERN.search(cmd_stripped) and _EXEC_INTERPRETER_PATTERN.search(cmd_stripped):
            return False, (
                "Command blocked: decodes a payload to a file and executes it "
                "in the same command — the decoded content cannot be inspected"
            )

        # Check dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, cmd_stripped):
                # Allow command substitution if explicitly enabled
                substitution_patterns = {
                    r"`.*`", r"\$\(.*\)", r"\$\{[^}]*\$\(",
                    r"\$\{[^}]*`", r"<\(.*\)", r">\(.*\)",
                }
                if self.allow_command_substitution and pattern in substitution_patterns:
                    continue
                return False, "Command blocked: matches dangerous pattern"

        # Whitelist mode: only allowed commands
        if self.allowed_commands is not None:
            try:
                parts = shlex.split(cmd_stripped)
                base_cmd = parts[0] if parts else ""
            except ValueError:
                base_cmd = cmd_stripped.split()[0] if cmd_stripped.split() else ""

            if base_cmd not in self.allowed_commands:
                return False, (
                    f"Command '{base_cmd}' not in allowed commands list. "
                    f"Allowed: {sorted(self.allowed_commands)}"
                )

        return True, ""

    def _get_safe_env(self) -> dict[str, str]:
        """Get environment with sensitive variables stripped.

        Strips the explicit ``strip_env_vars`` set *and* any variable whose name
        matches the generic credential-name patterns, so provider keys added
        later (or custom ``*_API_KEY`` / ``*SECRET*`` vars) are never leaked to a
        shell command just because the explicit list didn't list them.
        """
        env = os.environ.copy()
        for var in list(env):
            if var in self.strip_env_vars or _is_sensitive_env_name(var):
                env.pop(var, None)
        return env

    @staticmethod
    def _needs_shell(command: str) -> bool:
        """True when the command uses shell features (pipes/globs/redirects)."""
        return bool(_SHELL_METACHARS.search(command))

    async def _spawn(self, command: str, cwd: str, env: dict[str, str]):
        """Spawn the command, preferring argv execution over a shell.

        A new session is created so the whole process group (including any
        children the command spawns) can be killed together on timeout.
        """
        kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": cwd,
            "env": env,
        }
        if _POSIX:
            kwargs["start_new_session"] = True

        if not self._needs_shell(command):
            try:
                argv = shlex.split(command)
            except ValueError:
                argv = []
            if argv:
                return await asyncio.create_subprocess_exec(*argv, **kwargs)
        # Shell features present (or argv parse failed): use the shell. The
        # safety checks in ``_is_command_safe`` already blocked the dangerous
        # substitution/redirect patterns.
        return await asyncio.create_subprocess_shell(command, **kwargs)

    @staticmethod
    async def _kill_process_group(proc) -> None:
        """Hard-kill the process group and reap the child."""
        if proc.returncode is not None:
            return
        killed = False
        if _POSIX:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                killed = True
            except (ProcessLookupError, PermissionError, OSError):
                killed = False
        if not killed:
            try:
                proc.kill()
            except ProcessLookupError:  # process already gone
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:  # pragma: no cover - defensive
            pass

    def _resolve_timeout(self, timeout: float | int | None) -> float:
        """Resolve the effective wall-clock cap for one command.

        A positive per-call ``timeout`` overrides the instance default for this
        execution only; ``None`` or non-positive values fall back to the default
        so a caller cannot accidentally disable the deadline.
        """
        if timeout is not None:
            try:
                t = float(timeout)
                if t > 0:
                    return t
            except (TypeError, ValueError):  # invalid timeout value; fall back to the default
                pass
        return float(self.timeout)

    async def _execute(
        self, command: str, timeout: float | int | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: The shell command to execute.
            timeout: Per-call wall-clock cap (seconds) overriding the instance
                default for this command. A runaway command is hard-killed
                (process group) at this deadline; values <= 0 or ``None`` fall
                back to the default.

        Returns:
            Dict with stdout, stderr, return_code, and command.
        """
        # Safety check
        is_safe, reason = self._is_command_safe(command)
        if not is_safe:
            raise ValueError(f"Security: {reason}")

        # Determine working directory: an explicit setting wins, otherwise the
        # configured workspace (EFFGEN_WORKSPACE) when set, else the current
        # directory. Running in the workspace keeps generated files out of the
        # caller's project directory.
        from ._fs import default_workspace

        cwd = self.working_directory or str(default_workspace() or os.getcwd())
        if not os.path.isdir(cwd):
            raise ValueError(f"Working directory does not exist: {cwd}")

        env = self._get_safe_env()
        effective_timeout = self._resolve_timeout(timeout)

        try:
            proc = await self._spawn(command, cwd, env)
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
            except TimeoutError:
                # Kill the whole process group so backgrounded children die too.
                await self._kill_process_group(proc)
                raise TimeoutError(
                    f"Command timed out after {effective_timeout:g}s: {command}"
                )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            # Truncate very long outputs
            max_output = self.max_output_size
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... (output truncated)"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... (output truncated)"

            result = {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "return_code": proc.returncode,
                "command": command,
            }

            # Format output for agent consumption
            if proc.returncode == 0:
                return result
            else:
                # Include stderr info in the result so agent sees the error
                result["error_info"] = f"Command exited with code {proc.returncode}"
                return result

        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to execute command: {e}")
