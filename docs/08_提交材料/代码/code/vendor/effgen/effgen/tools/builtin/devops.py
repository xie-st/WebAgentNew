"""
DevOps tools for the effGen framework.

Provides read-only wrappers around git, docker, system info, and HTTP
requests. All subprocess calls are time-bounded and restricted to
read-only operations by default.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 15) -> dict[str, Any]:
    """Run a subprocess command and return stdout/stderr/returncode."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Command not found: {cmd[0]}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


class GitTool(BaseTool):
    """Read-only git operations: status, log, diff, branch."""

    ALLOWED = {"status", "log", "diff", "branch", "show", "remote"}

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="git",
                description=(
                    "Run read-only git commands (status, log, diff, branch, show, "
                    "remote) in a repository. Does NOT modify the repository."
                ),
                category=ToolCategory.SYSTEM,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Git operation",
                        required=True,
                        enum=["status", "log", "diff", "branch", "show", "remote"],
                    ),
                    ParameterSpec(
                        name="cwd",
                        type=ParameterType.STRING,
                        description="Working directory (repo root)",
                        required=False,
                        default=".",
                    ),
                    ParameterSpec(
                        name="n",
                        type=ParameterType.INTEGER,
                        description="Number of log entries (for log)",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=200,
                    ),
                    ParameterSpec(
                        name="ref",
                        type=ParameterType.STRING,
                        description="Optional git ref (branch, commit) for show/diff",
                        required=False,
                    ),
                ],
                timeout_seconds=20,
                tags=["devops", "git", "vcs", "read-only"],
                examples=[
                    {"operation": "status"},
                    {"operation": "log", "n": 5},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        cwd: str = ".",
        n: int = 10,
        ref: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if operation not in self.ALLOWED:
            raise ValueError(f"Disallowed git operation: {operation}")
        if shutil.which("git") is None:
            raise RuntimeError("git executable not found on PATH")

        if operation == "status":
            cmd = ["git", "status", "--short", "--branch"]
        elif operation == "log":
            cmd = ["git", "log", f"-n{n}", "--oneline", "--decorate"]
        elif operation == "diff":
            cmd = ["git", "diff", "--stat"]
            if ref:
                cmd.append(ref)
        elif operation == "branch":
            cmd = ["git", "branch", "-a"]
        elif operation == "show":
            cmd = ["git", "show", "--stat", ref or "HEAD"]
        elif operation == "remote":
            cmd = ["git", "remote", "-v"]
        else:
            raise ValueError(f"Unknown git operation: {operation}")

        result = _run(cmd, cwd=cwd, timeout=15)
        return {"operation": operation, "command": " ".join(cmd), **result}


class DockerTool(BaseTool):
    """Read-only docker operations: ps, images, logs."""

    ALLOWED = {"ps", "images", "logs", "version", "info"}

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="docker",
                description=(
                    "Run read-only Docker commands (ps, images, logs, version, info). "
                    "Requires Docker to be installed locally."
                ),
                category=ToolCategory.SYSTEM,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Docker operation",
                        required=True,
                        enum=["ps", "images", "logs", "version", "info"],
                    ),
                    ParameterSpec(
                        name="container",
                        type=ParameterType.STRING,
                        description="Container name/id (required for logs)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="tail",
                        type=ParameterType.INTEGER,
                        description="Number of log lines to tail",
                        required=False,
                        default=50,
                        min_value=1,
                        max_value=1000,
                    ),
                ],
                timeout_seconds=20,
                tags=["devops", "docker", "container", "read-only"],
                examples=[{"operation": "ps"}],
            )
        )

    async def _execute(
        self,
        operation: str,
        container: str | None = None,
        tail: int = 50,
        **kwargs,
    ) -> dict[str, Any]:
        if operation not in self.ALLOWED:
            raise ValueError(f"Disallowed docker operation: {operation}")
        if shutil.which("docker") is None:
            raise RuntimeError("docker executable not found on PATH")

        if operation == "ps":
            cmd = ["docker", "ps", "--format", "{{json .}}"]
        elif operation == "images":
            cmd = ["docker", "images", "--format", "{{json .}}"]
        elif operation == "logs":
            if not container:
                raise ValueError("logs requires 'container'")
            cmd = ["docker", "logs", "--tail", str(tail), container]
        elif operation == "version":
            cmd = ["docker", "version", "--format", "{{json .}}"]
        elif operation == "info":
            cmd = ["docker", "info", "--format", "{{json .}}"]
        else:
            raise ValueError(f"Unknown docker operation: {operation}")

        result = _run(cmd, timeout=15)
        return {"operation": operation, **result}


class SystemInfoTool(BaseTool):
    """CPU, memory, disk, and network info via psutil."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="system_info",
                description=(
                    "Get local system information: CPU usage, memory, disk, and "
                    "network statistics. Uses psutil."
                ),
                category=ToolCategory.SYSTEM,
                parameters=[
                    ParameterSpec(
                        name="kind",
                        type=ParameterType.STRING,
                        description="Type of info to return",
                        required=False,
                        default="all",
                        enum=["cpu", "memory", "disk", "network", "all"],
                    ),
                ],
                timeout_seconds=10,
                tags=["devops", "system", "psutil"],
                examples=[{"kind": "memory"}],
            )
        )

    async def _execute(self, kind: str = "all", **kwargs) -> dict[str, Any]:
        try:
            import psutil  # type: ignore
        except ImportError as e:
            raise ImportError(
                "psutil is not installed. Install with: pip install psutil"
            ) from e

        out: dict[str, Any] = {}
        if kind in ("cpu", "all"):
            out["cpu"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "count_logical": psutil.cpu_count(logical=True),
                "count_physical": psutil.cpu_count(logical=False),
            }
        if kind in ("memory", "all"):
            vm = psutil.virtual_memory()
            out["memory"] = {
                "total": vm.total,
                "available": vm.available,
                "used": vm.used,
                "percent": vm.percent,
            }
        if kind in ("disk", "all"):
            du = psutil.disk_usage("/")
            out["disk"] = {
                "total": du.total,
                "used": du.used,
                "free": du.free,
                "percent": du.percent,
            }
        if kind in ("network", "all"):
            net = psutil.net_io_counters()
            out["network"] = {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
            }
        return out


class HTTPTool(BaseTool):
    """Make simple HTTP GET/POST requests."""

    def __init__(self, allow_private: bool = False) -> None:
        """Args:
            allow_private: Allow requests to private/loopback/link-local/metadata
                addresses. Default ``False`` blocks them (SSRF protection); the
                target URL is user/model-suppliable.
        """
        self.allow_private = allow_private
        super().__init__(
            metadata=ToolMetadata(
                name="http",
                description=(
                    "Make an HTTP GET or POST request to a URL with optional "
                    "headers and JSON body. Returns status, headers, and body."
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="url",
                        type=ParameterType.STRING,
                        description="Target URL (http or https)",
                        required=True,
                    ),
                    ParameterSpec(
                        name="method",
                        type=ParameterType.STRING,
                        description="HTTP method",
                        required=False,
                        default="GET",
                        enum=["GET", "POST"],
                    ),
                    ParameterSpec(
                        name="headers",
                        type=ParameterType.OBJECT,
                        description="Request headers",
                        required=False,
                    ),
                    ParameterSpec(
                        name="params",
                        type=ParameterType.OBJECT,
                        description="Query parameters (appended to URL)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="json_body",
                        type=ParameterType.OBJECT,
                        description="JSON body (for POST)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description="Timeout in seconds",
                        required=False,
                        default=20,
                        min_value=1,
                        max_value=120,
                    ),
                ],
                timeout_seconds=30,
                tags=["devops", "http", "network"],
                examples=[
                    {"url": "https://httpbin.org/get", "method": "GET"},
                ],
            )
        )

    async def _execute(
        self,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        params: dict | None = None,
        json_body: dict | None = None,
        timeout: int = 20,
        **kwargs,
    ) -> dict[str, Any]:
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("url must start with http:// or https://")

        if params:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{urlencode(params)}"

        data = None
        req_headers = dict(headers or {})
        if method == "POST" and json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        try:
            with safe_urlopen(
                url,
                data=data,
                headers=req_headers,
                method=method,
                timeout=timeout,
                allow_private=self.allow_private,
            ) as resp:
                body_bytes = resp.read()
                status = resp.getcode()
                resp_headers = dict(resp.getheaders())
        except HTTPError as e:
            return {
                "status": e.code,
                "error": e.reason,
                "body": e.read().decode("utf-8", errors="replace")[:4096],
            }
        except URLError as e:
            raise ConnectionError(f"Request failed: {e.reason}")

        body_text = body_bytes.decode("utf-8", errors="replace")
        parsed: Any = None
        ct = resp_headers.get("Content-Type", "")
        if "application/json" in ct:
            try:
                parsed = json.loads(body_text)
            except json.JSONDecodeError:
                parsed = None

        return {
            "status": status,
            "headers": resp_headers,
            "body": body_text[:8192],
            "json": parsed,
            "truncated": len(body_text) > 8192,
        }
