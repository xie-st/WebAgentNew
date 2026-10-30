"""
Python REPL tool with persistent sessions.

This module provides a Python REPL (Read-Eval-Print Loop) tool with
persistent session state, safe evaluation, and comprehensive error handling.

User code never runs in the host process. Each session is backed by a
dedicated **worker subprocess** (:mod:`effgen.tools.builtin._repl_worker`)
started in its own process group. The advertised timeout is enforced by the
parent with a wall-clock deadline and a hard ``SIGKILL`` of the worker's
process group, so a runaway such as ``while True: pass`` is killed on time
instead of pinning a CPU. Memory and output caps are likewise enforced from
outside the executed code.

``restricted_mode`` (the default, and the only mode a model can reach — see
below) withholds ``eval``/``exec``/``compile``/``open``/``input``/
``__import__``/``breakpoint`` and imports outside a fixed allow-list. It
refuses process, shell, and native-code execution (``os.system``,
``subprocess``, ``os.exec``/``spawn``/``fork``, ``ctypes``) via an interpreter
audit hook, so the dangerous operation is rejected no matter how the code
reached the callable — including a subclass-graph walk that recovers a live
module reference through a getset descriptor or a generator frame. As a
first, cheaper layer it also rejects the common dunder-reflection routes to
such a reference (``__globals__``/``__subclasses__``/... spelled literally or
built at run time, whether via ``getattr`` or ``operator.attrgetter``).

This is not equivalent to OS-level process isolation (no filesystem/network
namespace, no seccomp); code that runs inside it still runs as the host user,
and the audit hook targets execution primitives, not every possible read of
process state. Untrusted or adversarial code should go through the sandboxed
:mod:`effgen.execution` code-executor tool instead, which adds process/
namespace isolation on top.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import queue
import select
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)

_WORKER_PATH = os.path.join(os.path.dirname(__file__), "_repl_worker.py")


_spawn_queue: queue.Queue = queue.Queue()
_spawn_thread: threading.Thread | None = None
_spawn_thread_lock = threading.Lock()


def _spawn_loop() -> None:  # pragma: no cover - runs for the interpreter's life
    while True:
        start, box, done = _spawn_queue.get()
        try:
            box.append((None, start()))
        except BaseException as exc:  # relayed to the caller that queued the work
            box.append((exc, None))
        finally:
            done.set()


def _start_on_spawn_thread(start: Any) -> Any:
    """Run ``start`` on the long-lived worker-spawning thread.

    The worker asks the kernel to kill it if its parent dies, and on Linux that
    signal is delivered when the *thread* that created the process exits — not
    when the process does. A worker started from a short-lived thread (an event
    loop's default executor, say) is therefore killed as soon as that thread is
    recycled, which silently empties the session's namespace: the next call
    lands on a fresh worker and the caller's variables are gone. Every worker is
    started from this thread instead, which lives as long as the interpreter.
    """
    global _spawn_thread
    with _spawn_thread_lock:
        if _spawn_thread is None or not _spawn_thread.is_alive():
            _spawn_thread = threading.Thread(
                target=_spawn_loop, name="effgen-repl-spawner", daemon=True
            )
            _spawn_thread.start()
    box: list[tuple[BaseException | None, Any]] = []
    done = threading.Event()
    _spawn_queue.put((start, box, done))
    done.wait()
    error, value = box[0]
    if error is not None:
        raise error
    return value


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:  # already closed
        pass


class _Worker:
    """Handle to a single persistent REPL worker subprocess."""

    __slots__ = ("proc", "resp_fd", "lock", "users", "_fd_closer", "__weakref__")

    def __init__(self, proc: subprocess.Popen, resp_fd: int):
        self.proc = proc
        self.resp_fd = resp_fd
        self.lock = threading.Lock()
        # Callers that have been handed this worker and have not finished with
        # it. Counted from hand-out (not from acquiring ``lock``) so eviction
        # never picks a worker a caller is about to send a request to.
        self.users = 0
        # The response fd is released when this handle is no longer referenced,
        # not when the worker is killed: a thread reading a reply holds the
        # handle for the whole request, so the fd cannot be closed — and its
        # number handed to another worker's pipe — while that read is waiting.
        self._fd_closer = weakref.finalize(self, _close_fd, resp_fd)

    def kill(self) -> None:
        """Hard-kill the worker's whole process group.

        Any thread waiting on a reply sees end-of-file on the response pipe and
        reports the failure; the pipe itself is closed once no thread holds
        this handle.
        """
        proc = self.proc
        try:
            if proc.poll() is None:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(proc.pid), signal_SIGKILL())
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                else:  # pragma: no cover - non-POSIX
                    proc.kill()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:  # pragma: no cover - best-effort worker fd cleanup
            pass
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover - best-effort worker reap
            pass


def signal_SIGKILL() -> int:
    """Return the platform's ``SIGKILL`` signal number."""
    import signal

    return signal.SIGKILL


def _kill_all_workers(workers: dict[str, _Worker]) -> None:
    """Kill every worker in ``workers`` and empty the mapping.

    Registered as the tool's finalizer so a REPL that is dropped without
    ``cleanup()`` being awaited does not leave its worker subprocesses running
    for the rest of the host process's life.
    """
    for worker in list(workers.values()):
        worker.kill()
    workers.clear()


class PythonREPL(BaseTool):
    """
    Python REPL tool with persistent session state.

    Features:
    - Persistent variable state across executions
    - Safe evaluation with restricted builtins
    - Import management
    - Comprehensive error handling
    - Output capture (stdout/stderr)
    - Expression vs statement handling
    - Session management (reset, save, restore), with a bounded number of live
      sessions: past ``max_sessions`` the least recently used idle session's
      worker is stopped and its variables are discarded

    Security:
    - User code runs in an isolated worker subprocess (its own process group)
    - Wall-clock timeout enforced by the parent with a hard process-group kill
    - Address-space (memory) limit applied to the worker via RLIMIT_AS
    - stdout/stderr output capped outside the executed code
    - Restricted builtins (no file operations, eval, exec by default)
    - Process/shell/native-code execution (``os.system``/``subprocess``/
      ``os.exec``/``spawn``/``fork``/``ctypes``) is refused by an interpreter
      audit hook while restricted code runs, regardless of how the code reached
      the callable
    - Dunder-attribute reflection (``getattr``/``setattr``/``delattr``/
      ``operator.attrgetter``/``operator.methodcaller``) is checked by value at
      the point of use, not by scanning the code text, so a dynamically built
      attribute name is treated the same as a literal one
    - Configurable allowed imports
    - Not a substitute for OS-level process isolation: see the code-executor
      tool for sandboxed execution of untrusted code
    """

    # Standardized resource limits
    DEFAULT_TIMEOUT = 30          # seconds (matches ToolMetadata.timeout_seconds)
    DEFAULT_MAX_OUTPUT = 102400   # 100 KB
    DEFAULT_MAX_MEMORY_MB = 1024  # per-worker address-space cap (RLIMIT_AS)
    # Live worker subprocesses kept at once. ``session_id`` is a model-supplied
    # parameter, so without a cap a caller (or a model varying the id) can
    # spawn one interpreter per session and hold them all.
    DEFAULT_MAX_SESSIONS = 8
    # Hard ceiling on bytes read back from a worker for a single request.
    _MAX_RESPONSE_BYTES = 4 * 1024 * 1024

    # Dangerous builtins to restrict
    RESTRICTED_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "vars",
    }

    # Default allowed imports
    DEFAULT_ALLOWED_IMPORTS = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "statistics",
        "decimal",
        "fractions",
    }

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_output: int = DEFAULT_MAX_OUTPUT,
        max_memory_mb: int | None = DEFAULT_MAX_MEMORY_MB,
        allow_unrestricted: bool = False,
        max_sessions: int | None = None,
    ) -> None:
        """Initialize the Python REPL.

        Args:
            timeout: Wall-clock seconds before a runaway worker is killed.
            max_output: Maximum captured stdout/stderr bytes per execution.
            max_memory_mb: Per-worker address-space cap in MiB (``None`` disables
                the RLIMIT_AS limit; the timeout kill still applies).
            allow_unrestricted: Developer opt-in that lets a per-call
                ``restricted_mode=False`` actually drop the sandbox. Off by
                default: the sandbox toggle is never reachable by the model, so a
                model-supplied ``restricted_mode=False`` is ignored and execution
                stays restricted. Can also be enabled out-of-band by setting the
                ``EFFGEN_REPL_ALLOW_UNRESTRICTED`` environment variable to a
                truthy value (``1``/``true``/``yes``/``on``).
            max_sessions: How many session worker subprocesses stay live at
                once (default 8, or the ``EFFGEN_REPL_MAX_SESSIONS``
                environment variable). When a new session would exceed the
                limit, the least recently used *idle* session's worker is
                stopped and its variables are discarded; using that session
                again starts a fresh worker with an empty namespace. A session
                whose call is still running is never stopped, so while more
                calls than the limit are in flight the count can exceed it
                until they finish. Values below 1 are raised to 1.
        """
        super().__init__(
            metadata=ToolMetadata(
                name="python_repl",
                description="Execute Python code in a persistent REPL session with safe evaluation",
                category=ToolCategory.CODE_EXECUTION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="Python code to execute",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="session_id",
                        type=ParameterType.STRING,
                        description="Session identifier for persistent state (default: 'default')",
                        required=False,
                        default="default",
                    ),
                    ParameterSpec(
                        name="reset_session",
                        type=ParameterType.BOOLEAN,
                        description="Whether to reset the session before execution",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="return_variables",
                        type=ParameterType.BOOLEAN,
                        description="Whether to return all session variables",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        # Fractional seconds are accepted, so the declared type
                        # is the one the implementation takes rather than a
                        # narrower one that would reject a working call.
                        name="timeout",
                        type=ParameterType.FLOAT,
                        description=(
                            "Wall-clock seconds this call may run before the "
                            f"worker is killed; a value that is not positive "
                            f"uses this tool's default ({int(timeout)}s)"
                        ),
                        required=False,
                        default=None,
                        # Caller-controlled bound, like restricted_mode: the
                        # value raises as well as lowers the wall-clock cap, so
                        # a model choosing its own would be able to lift the
                        # limit that stops a runaway program.
                        model_facing=False,
                    ),
                    ParameterSpec(
                        name="restricted_mode",
                        type=ParameterType.BOOLEAN,
                        description="Whether to use restricted builtins (safer but limited)",
                        required=False,
                        default=True,
                        # Developer-controlled sandbox toggle — never exposed to
                        # the model. A model-supplied restricted_mode=False is
                        # ignored unless the developer set allow_unrestricted=True
                        # at construction (or EFFGEN_REPL_ALLOW_UNRESTRICTED=1).
                        model_facing=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "result": {"type": "any"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "error": {"type": "string"},
                        "variables": {"type": "object"},
                    },
                },
                timeout_seconds=int(timeout),
                tags=["python", "repl", "code", "execution"],
                examples=[
                    {
                        "code": "x = 10\ny = 20\nx + y",
                        "session_id": "default",
                        "output": {
                            "result": 30,
                            "stdout": "",
                            "error": None,
                        },
                    },
                    {
                        "code": "import math\nmath.sqrt(16)",
                        "output": {
                            "result": 4.0,
                            "stdout": "",
                        },
                    },
                ],
            )
        )
        self._timeout = int(timeout)
        self._max_output = int(max_output)
        self._max_memory_bytes = (
            int(max_memory_mb) * 1024 * 1024 if max_memory_mb else None
        )
        env_optin = os.environ.get("EFFGEN_REPL_ALLOW_UNRESTRICTED", "").strip().lower()
        self._allow_unrestricted = bool(allow_unrestricted) or env_optin in (
            "1", "true", "yes", "on",
        )
        self._allowed_imports = self.DEFAULT_ALLOWED_IMPORTS.copy()
        self._max_sessions = self._resolve_max_sessions(max_sessions)
        # session_id -> _Worker (live worker subprocess holding the namespace),
        # in least-recently-used order so the oldest session is evicted first.
        self._workers: OrderedDict[str, _Worker] = OrderedDict()
        self._workers_lock = threading.Lock()
        # Stop the workers even if the tool is dropped without cleanup().
        self._finalizer = weakref.finalize(self, _kill_all_workers, self._workers)

    @classmethod
    def _resolve_max_sessions(cls, max_sessions: int | None) -> int:
        """Resolve the live-session cap from the argument or the environment."""
        if max_sessions is None:
            raw = os.environ.get("EFFGEN_REPL_MAX_SESSIONS", "").strip()
            if raw:
                try:
                    max_sessions = int(raw)
                except ValueError:
                    logger.warning(
                        "EFFGEN_REPL_MAX_SESSIONS=%r is not an integer; "
                        "using the default of %d",
                        raw, cls.DEFAULT_MAX_SESSIONS,
                    )
        if max_sessions is None:
            max_sessions = cls.DEFAULT_MAX_SESSIONS
        return max(1, int(max_sessions))

    async def initialize(self) -> None:
        """Initialize the REPL (workers are spawned lazily on first use)."""
        await super().initialize()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------
    def _spawn_worker(self) -> _Worker:
        """Start a fresh worker subprocess in its own process group.

        The process itself is started on the shared spawner thread — see
        :func:`_start_on_spawn_thread` for why it cannot be started on whatever
        thread happens to be running the call.
        """
        resp_r, resp_w = os.pipe()
        env = os.environ.copy()
        env["EFFGEN_REPL_RESP_FD"] = str(resp_w)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # A worker that may run developer-opted unrestricted code must be able
        # to spawn processes; every other worker runs only restricted code and
        # gets an always-on audit hook that refuses process/native execution.
        env["EFFGEN_REPL_ALLOW_UNRESTRICTED_WORKER"] = (
            "1" if self._allow_unrestricted else "0"
        )

        max_mem = self._max_memory_bytes

        def _preexec() -> None:  # pragma: no cover - runs in the child
            # New session => new process group, so the parent can kill the
            # whole tree (including grandchildren) with a single signal.
            os.setsid()
            if max_mem:
                try:
                    import resource

                    resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))
                except Exception:  # RLIMIT_AS unsupported here; continue without the memory cap
                    pass

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "pass_fds": (resp_w,),
            "env": env,
        }
        if hasattr(os, "setsid"):
            popen_kwargs["preexec_fn"] = _preexec
        try:
            proc = _start_on_spawn_thread(
                lambda: subprocess.Popen([sys.executable, _WORKER_PATH], **popen_kwargs)
            )
        finally:
            os.close(resp_w)  # the child owns its copy now
        return _Worker(proc, resp_r)

    def _get_worker(self, session_id: str) -> _Worker:
        """Reserve the worker for a session, spawning one if needed.

        Spawning past ``max_sessions`` evicts the least recently used idle
        session first, so the number of live worker subprocesses stays bounded
        however many distinct session ids arrive. The returned worker is
        checked out to the caller, who must call :meth:`_release_worker` when
        done with it.
        """
        evicted: list[_Worker] = []
        with self._workers_lock:
            worker = self._workers.get(session_id)
            if worker is None or worker.proc.poll() is not None:
                if worker is not None:
                    evicted.append(self._workers.pop(session_id))
                while len(self._workers) >= self._max_sessions:
                    old_id = self._evictable_session()
                    if old_id is None:
                        logger.debug(
                            "Every one of the %d REPL sessions is still running "
                            "a call; starting session %r alongside them.",
                            self._max_sessions, session_id,
                        )
                        break
                    logger.warning(
                        "REPL session limit of %d reached; stopping the least "
                        "recently used idle session %r. Its variables are "
                        "discarded; using it again starts an empty session. "
                        "Raise the limit with max_sessions= or "
                        "EFFGEN_REPL_MAX_SESSIONS.",
                        self._max_sessions, old_id,
                    )
                    evicted.append(self._workers.pop(old_id))
                worker = self._spawn_worker()
            self._workers[session_id] = worker
            self._workers.move_to_end(session_id)
            worker.users += 1
        for old_worker in evicted:
            old_worker.kill()
        return worker

    def _release_worker(self, worker: _Worker) -> None:
        """Give back a worker checked out by :meth:`_get_worker`.

        Trims the pool back to the limit if a burst of concurrent calls pushed
        it over, so extra workers do not stay alive after the burst.
        """
        evicted: list[_Worker] = []
        with self._workers_lock:
            worker.users -= 1
            while len(self._workers) > self._max_sessions:
                over_id = self._evictable_session()
                if over_id is None:
                    break
                evicted.append(self._workers.pop(over_id))
        for old_worker in evicted:
            old_worker.kill()

    def _evictable_session(self) -> str | None:
        """The least recently used session no caller is currently using.

        Only an unused session is evicted, so a call in flight always runs to
        completion or to its own timeout. If every session is in use the limit
        is exceeded for as long as those calls last — the extra workers are
        bounded by how many calls are in flight, and the next spawn trims back
        to the limit. Callers hold ``_workers_lock``.
        """
        for session_id, worker in self._workers.items():
            if worker.users == 0:
                return session_id
        return None

    def _kill_worker(self, session_id: str) -> None:
        with self._workers_lock:
            worker = self._workers.pop(session_id, None)
        if worker is not None:
            worker.kill()

    def _request(
        self,
        session_id: str,
        payload: dict[str, Any],
        _attempt: int = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one request to a session worker and read its response.

        Enforces the wall-clock timeout: if the worker does not answer within
        ``timeout`` (or ``self._timeout`` when no per-call override is given)
        seconds it is hard-killed (process group) and a typed timeout failure is
        returned. Runs blocking I/O — call via a thread.

        If a *reused* worker has exited between requests before producing any
        output, the request is transparently retried once on a fresh worker —
        this is safe (no output was emitted, so nothing double-runs) and fixes a
        rare race where a worker died after the previous request.
        """
        while True:
            worker = self._get_worker(session_id)
            try:
                result = self._request_on(worker, session_id, payload, _attempt, timeout)
            finally:
                self._release_worker(worker)
            if result is not None:
                return result
            _attempt = 1

    def _request_on(
        self,
        worker: _Worker,
        session_id: str,
        payload: dict[str, Any],
        _attempt: int,
        timeout: float | None,
    ) -> dict[str, Any] | None:
        """Send one request to a reserved worker and read its reply.

        Returns ``None`` when the worker turned out to be dead before anything
        was sent, so the caller can retry on a fresh one.
        """
        line = (json.dumps(payload) + "\n").encode("utf-8")
        with worker.lock:
            try:
                worker.proc.stdin.write(line)
                worker.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as e:
                # Worker died between requests. Nothing was sent, so retrying on
                # a fresh worker cannot double-run the code.
                self._kill_worker(session_id)
                if _attempt == 0:
                    return None
                return self._failure(f"REPL worker unavailable: {e}")

            effective_timeout = timeout if (timeout is not None and timeout > 0) else self._timeout
            deadline = time.monotonic() + effective_timeout
            buf = b""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_worker(session_id)
                    return self._failure(
                        f"Execution timed out after {effective_timeout:g}s "
                        "(worker process killed)",
                        timeout=True,
                    )
                try:
                    ready, _, _ = select.select([worker.resp_fd], [], [], remaining)
                except OSError:
                    self._kill_worker(session_id)
                    return self._failure("REPL worker channel closed")
                if not ready:
                    continue
                try:
                    chunk = os.read(worker.resp_fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    self._kill_worker(session_id)
                    # Worker died before answering. If it produced no partial
                    # output and we haven't retried yet, retry on a fresh worker
                    # (safe: nothing was emitted for this request).
                    if buf == b"" and _attempt == 0:
                        return None
                    return self._failure("REPL worker exited unexpectedly")
                buf += chunk
                if len(buf) > self._MAX_RESPONSE_BYTES:
                    self._kill_worker(session_id)
                    return self._failure("REPL produced too much output")
                nl = buf.find(b"\n")
                if nl != -1:
                    try:
                        return json.loads(buf[:nl].decode("utf-8", "replace"))
                    except Exception as e:
                        self._kill_worker(session_id)
                        return self._failure(f"REPL protocol error: {e}")

    @staticmethod
    def _failure(message: str, timeout: bool = False) -> dict[str, Any]:
        result = {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "",
            "error": message,
        }
        if timeout:
            result["timeout"] = True
        return result

    def _is_import_allowed(self, module_name: str) -> bool:
        """Check if an import is allowed."""
        # Get base module name (before any dots)
        base_module = module_name.split(".")[0]
        return base_module in self._allowed_imports

    def _check_imports(self, code: str) -> str | None:
        """
        Check if code contains only allowed imports.

        Returns:
            Optional[str]: Error message if disallowed import found, None otherwise
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_import_allowed(alias.name):
                            return f"Import of '{alias.name}' is not allowed"
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._is_import_allowed(node.module):
                        return f"Import from '{node.module}' is not allowed"
        except SyntaxError:
            # Let the actual execution handle syntax errors
            pass

        return None

    def _check_security(self, code: str) -> str | None:
        """
        Check for sandbox-escape patterns beyond import restrictions.

        Blocks:
        - Direct ``__import__`` calls
        - ``importlib`` usage
        - ``__builtins__`` dict manipulation
        - ``getattr``/``setattr`` targeting blocked modules or dunder attrs
        - String-concatenation bypasses  (e.g. ``eval("__" + "import__")``)

        Returns:
            Error message if dangerous pattern found, None otherwise.
        """
        # Fast string-level checks (catches dynamic construction too)
        blocked_strings = [
            "__import__",
            "importlib",
            "__builtins__",
            "__subclasses__",
            "__bases__",
            "__mro__",
            "__globals__",
            "__code__",
            # ``__dict__`` is the mappingproxy that exposes the getset
            # descriptors above; blocking the literal spelling here (and the
            # ``vars`` builtin in the worker) closes the descriptor route to
            # them that never names them to ``getattr``.
            "__dict__",
        ]
        for pattern in blocked_strings:
            if pattern in code:
                return f"Access to '{pattern}' is blocked in restricted mode"

        # AST-level checks
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None  # let execution handle it

        for node in ast.walk(tree):
            # Block getattr/setattr calls that could reach blocked objects
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ("getattr", "setattr", "delattr"):
                    # Check if second arg (attr name) is a blocked dunder
                    if len(node.args) >= 2:
                        attr_arg = node.args[1]
                        if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                            if attr_arg.value.startswith("__") and attr_arg.value.endswith("__"):
                                return (
                                    f"getattr/setattr with dunder attribute "
                                    f"'{attr_arg.value}' is blocked in restricted mode"
                                )

            # Block attribute access to __builtins__, __globals__ etc.
            if isinstance(node, ast.Attribute):
                if node.attr in ("__builtins__", "__globals__", "__subclasses__",
                                 "__bases__", "__mro__", "__code__", "__dict__"):
                    return f"Access to '{node.attr}' is blocked in restricted mode"

        return None

    async def _execute(
        self,
        code: str,
        session_id: str = "default",
        reset_session: bool = False,
        return_variables: bool = False,
        restricted_mode: bool = True,
        timeout: float | int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute Python code in a REPL session.

        Args:
            code: Python code to execute
            session_id: Session identifier for state persistence
            reset_session: Whether to reset session state before execution
            return_variables: Whether to include all variables in response
            restricted_mode: Whether to use restricted builtins
            timeout: Per-call wall-clock cap (seconds) overriding the instance
                default for this execution. A runaway block is hard-killed at
                this deadline; values <= 0 or ``None`` fall back to the default.

        Returns:
            Dict containing result, stdout, stderr, error, and optionally variables
        """
        # The sandbox toggle is developer-controlled, never model-controlled.
        # Unless the developer explicitly opted into unrestricted execution at
        # construction (allow_unrestricted=True / EFFGEN_REPL_ALLOW_UNRESTRICTED),
        # any restricted_mode=False — including one a prompt-injected model emits —
        # is ignored and execution stays sandboxed. Fail-closed by default.
        if not self._allow_unrestricted:
            restricted_mode = True

        # Check imports and security in restricted mode (cheap parent-side
        # gate that rejects obvious escapes before any subprocess work).
        if restricted_mode:
            import_error = self._check_imports(code)
            if import_error:
                resp = self._failure(import_error)
                if return_variables:
                    resp["variables"] = {}
                return resp
            security_error = self._check_security(code)
            if security_error:
                resp = self._failure(security_error)
                if return_variables:
                    resp["variables"] = {}
                return resp

        payload = {
            "code": code,
            "restricted_mode": restricted_mode,
            "allowed_imports": sorted(self._allowed_imports),
            "return_variables": return_variables,
            "reset": reset_session,
            "max_output": self._max_output,
        }

        # Resolve the effective wall-clock cap: an explicit positive per-call
        # ``timeout`` overrides the instance default for this execution only.
        effective_timeout: float | None = None
        if timeout is not None:
            try:
                t = float(timeout)
                if t > 0:
                    effective_timeout = t
            except (TypeError, ValueError):
                effective_timeout = None

        # The worker runs user code; do the blocking request off the event loop
        # so a slow (or killed-on-timeout) worker never stalls the caller.
        response = await asyncio.to_thread(
            self._request, session_id, payload, timeout=effective_timeout
        )

        if response.get("error"):
            logger.debug("REPL execution error: %s", response["error"])
        if return_variables and "variables" not in response:
            response["variables"] = {}
        return response

    def add_allowed_import(self, module_name: str) -> None:
        """
        Add a module to the allowed imports list.

        Args:
            module_name: Name of the module to allow
        """
        self._allowed_imports.add(module_name)

    def remove_allowed_import(self, module_name: str) -> None:
        """
        Remove a module from the allowed imports list.

        Args:
            module_name: Name of the module to disallow
        """
        self._allowed_imports.discard(module_name)

    def reset_session(self, session_id: str) -> None:
        """
        Reset a session to clean state by terminating its worker.

        The next execution for this session spawns a fresh worker with an empty
        namespace.

        Args:
            session_id: Session to reset
        """
        self._kill_worker(session_id)

    def get_session_variables(self, session_id: str) -> dict[str, str]:
        """
        Get all variables in a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict mapping variable names to their string representations
        """
        with self._workers_lock:
            live = session_id in self._workers
        if not live:
            return {}
        response = self._request(
            session_id,
            {
                "code": "",
                "restricted_mode": False,
                "allowed_imports": sorted(self._allowed_imports),
                "return_variables": True,
                "reset": False,
                "max_output": self._max_output,
            },
        )
        return response.get("variables") or {}

    async def cleanup(self) -> None:
        """Terminate every worker subprocess and clear state."""
        with self._workers_lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.kill()
        await super().cleanup()
