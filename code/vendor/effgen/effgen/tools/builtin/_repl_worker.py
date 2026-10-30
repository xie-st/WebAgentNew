"""Out-of-process worker that executes Python REPL code under hard limits.

This module is launched as a **standalone subprocess** by
:class:`effgen.tools.builtin.python_repl.PythonREPL`. It deliberately imports
nothing from ``effgen`` so start-up stays cheap and the worker has a minimal,
auditable surface.

Protocol
--------
* The parent writes one JSON request per line to the worker's **stdin**.
* The worker writes one JSON response per line to the descriptor named by the
  ``EFFGEN_REPL_RESP_FD`` environment variable (an inherited pipe). Standard
  out/err are redirected to ``/dev/null`` while user code runs so stray writes
  can never corrupt the protocol channel.

Each request is ``{"code", "restricted_mode", "allowed_imports",
"return_variables", "reset", "max_output"}``. The persistent namespace lives in
this process, so variables survive across requests — exactly like an in-process
REPL — but a runaway request (``while True: pass``) can be killed by the parent
sending ``SIGKILL`` to the worker's process group with no lasting harm.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import operator
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

_RESP_FD = int(os.environ.get("EFFGEN_REPL_RESP_FD", "3"))

# Builtins withheld in restricted mode (mirrors PythonREPL.RESTRICTED_BUILTINS).
# ``vars`` is withheld because ``vars(x)`` returns ``x.__dict__`` and so is an
# alternate route to the type/function ``__dict__`` mappingproxy that the
# literal ``__dict__`` block would otherwise close.
_RESTRICTED_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "vars",
}

# Captured before any guard ever runs, so introspecting ``dir(builtins)`` while
# a guard is active never trips on legitimate dunder builtins the interpreter
# itself relies on (``__build_class__``, ``__debug__``, ``__doc__``, ...).
_REAL_GETATTR = builtins.getattr


# The specific attributes that let code walk from any object to a live
# reference to an unrestricted module (``os``, ``subprocess``, ...) or its
# bytecode/namespace -- mirrors the literal-text blocklist in
# ``python_repl._check_security``. Ordinary user code (including the
# interpreter's own import machinery, which reads ``__name__``/``__spec__``/
# ``__loader__`` on every ``import`` statement) never needs these, so this set
# stays intentionally narrow rather than "any dunder" -- blocking every dunder
# at the reflection layer would also break the interpreter's own internal use
# of ``getattr`` for ordinary imports and class creation.
_BLOCKED_DUNDER_ATTRS = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__code__",
        "__bases__",
        "__base__",
        "__mro__",
        "__builtins__",
        "__closure__",
        "__import__",
        # ``X.__dict__`` is the mappingproxy that holds the getset descriptors
        # (``__globals__``/``__code__``/``__subclasses__``); blocking it closes
        # the ``type(f).__dict__[built_name].__get__(f)`` descriptor route that
        # reaches those without ever naming them to ``getattr``.
        "__dict__",
    }
)


def _is_blocked_dunder_name(name: object) -> bool:
    """True if *name* is one of the blocked dunder attributes, however built.

    A name built at run time (``chr(95) * 2 + "subclasses" + chr(95) * 2``)
    never appears as a literal string in source, so it cannot be caught by a
    text or AST scan of the code. Checking the resolved string value at the
    point of use catches it regardless of how it was constructed.
    """
    return isinstance(name, str) and name in _BLOCKED_DUNDER_ATTRS


def _reject_dunder(name: object, verb: str) -> None:
    if _is_blocked_dunder_name(name):
        raise AttributeError(
            f"{verb} dunder attribute {name!r} is blocked in restricted mode"
        )


class _ReflectionGuard:
    """Withhold dunder-attribute reflection process-wide, for one execution.

    ``restricted_mode`` replaces ``getattr``/``setattr``/``delattr`` in the
    namespace handed to user code, but that only governs names looked up
    directly in the executed code's own frame. Any already-imported standard-
    library helper that reaches a dunder attribute internally --
    ``operator.attrgetter``, ``operator.methodcaller``, ``functools.
    update_wrapper`` -- still calls the real, process-wide ``builtins.getattr``/
    ``setattr`` (or the C-level attribute protocol directly), so a namespace-
    local override alone does not stop code that reaches those attributes
    through such a helper instead of a bare ``getattr(...)`` call.

    This guard replaces the real ``builtins`` reflection functions and
    ``operator.attrgetter``/``operator.methodcaller`` with dunder-checked
    versions for the exact duration of one restricted-mode execution, then
    restores the originals -- closing that path for direct calls and for any
    standard-library code invoked transitively by the executed code.
    """

    def __init__(self) -> None:
        self._saved: dict[str, object] = {}

    def __enter__(self) -> "_ReflectionGuard":
        self._saved = {
            "getattr": builtins.getattr,
            "setattr": builtins.setattr,
            "delattr": builtins.delattr,
            "hasattr": builtins.hasattr,
            "attrgetter": operator.attrgetter,
            "methodcaller": operator.methodcaller,
        }
        real_getattr = self._saved["getattr"]
        real_setattr = self._saved["setattr"]
        real_delattr = self._saved["delattr"]
        real_hasattr = self._saved["hasattr"]
        real_attrgetter = self._saved["attrgetter"]
        real_methodcaller = self._saved["methodcaller"]

        def guarded_getattr(obj, name, *default):
            _reject_dunder(name, "reading")
            return real_getattr(obj, name, *default)

        def guarded_setattr(obj, name, value):
            _reject_dunder(name, "setting")
            return real_setattr(obj, name, value)

        def guarded_delattr(obj, name):
            _reject_dunder(name, "deleting")
            return real_delattr(obj, name)

        def guarded_hasattr(obj, name):
            if _is_blocked_dunder_name(name):
                return False
            return real_hasattr(obj, name)

        def guarded_attrgetter(*attrs):
            for attr in attrs:
                for part in str(attr).split("."):
                    _reject_dunder(part, "reading")
            return real_attrgetter(*attrs)

        def guarded_methodcaller(name, *args, **kwargs):
            _reject_dunder(name, "calling")
            return real_methodcaller(name, *args, **kwargs)

        builtins.getattr = guarded_getattr
        builtins.setattr = guarded_setattr
        builtins.delattr = guarded_delattr
        builtins.hasattr = guarded_hasattr
        operator.attrgetter = guarded_attrgetter
        operator.methodcaller = guarded_methodcaller
        return self

    def __exit__(self, *exc_info) -> None:
        builtins.getattr = self._saved["getattr"]
        builtins.setattr = self._saved["setattr"]
        builtins.delattr = self._saved["delattr"]
        builtins.hasattr = self._saved["hasattr"]
        operator.attrgetter = self._saved["attrgetter"]
        operator.methodcaller = self._saved["methodcaller"]


# Attribute-name filtering (the reflection guard and the parent-side text/AST
# scan) blocks the common routes to a live module reference, but it cannot be
# exhaustive: a name built at run time can be routed through a getset descriptor
# (``type(f).__dict__['__globals__'].__get__(f)``) or a generator/coroutine
# frame (``(x for x in []).gi_frame.f_globals``) that never calls ``getattr``.
# The audit hook below is the actual boundary: it refuses the dangerous OS
# operation itself -- spawning a process or shell -- no matter how the code
# reached the callable. Audit hooks cannot be removed once installed and fire
# from the interpreter's C layer, so restricted user code cannot disable this.
_DANGEROUS_AUDIT_EVENTS = frozenset(
    {
        # process / shell execution
        "os.system",
        "os.exec",
        "os.spawn",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "os.startfile",
        "subprocess.Popen",
        "pty.spawn",
        # native-code execution via ctypes (no legitimate use in the
        # allowed-import REPL; blocks a libc.system() route around subprocess)
        "ctypes.dlopen",
        "ctypes.dlsym",
        "ctypes.call_function",
    }
)

def _audit_hook(event: str, args: tuple) -> None:
    """Refuse process/shell/native-code execution unconditionally.

    Enforcement is deliberately not gated by a Python-level flag: once user
    code can reflect (reach ``sys.modules``), any module-global toggle it could
    reach it could also flip, so the hook must raise on every dangerous event
    for the worker's whole life. This is why the hook is installed only in a
    worker that never runs developer-opted unrestricted code (see
    :func:`_install_audit_hook`).
    """
    if event in _DANGEROUS_AUDIT_EVENTS:
        raise PermissionError(
            f"operation {event!r} is blocked in restricted mode"
        )


def _install_audit_hook() -> None:
    """Install the process-execution audit hook unless the worker is unrestricted.

    A worker spawned for a REPL with ``allow_unrestricted`` (developer opt-in)
    must be able to run ``subprocess``/``os.system``/etc., so the hook is not
    installed there. Every other worker only ever runs restricted code, so the
    always-on hook cannot break a legitimate operation. Audit hooks cannot be
    removed once installed and fire from the interpreter's C layer, so user
    code cannot disable this one.
    """
    if os.environ.get("EFFGEN_REPL_ALLOW_UNRESTRICTED_WORKER") == "1":
        return
    try:
        sys.addaudithook(_audit_hook)
    except Exception:  # pragma: no cover - audit hooks unavailable
        pass


def _install_parent_death_signal() -> None:
    """Ask the kernel to SIGKILL this worker if its parent dies (Linux only).

    Without this, a worker busy-executing user code (e.g. ``while True: pass``)
    never returns to read stdin, so an abrupt parent death would leave it
    running as an orphan. ``PR_SET_PDEATHSIG`` closes that gap; it is set here
    (post-exec) because the flag is cleared across ``execve``.

    Note for the parent side: Linux delivers this signal when the *thread* that
    created the process exits, not when the process does, so workers have to be
    started from a thread that lives as long as the interpreter.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        import signal

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        # Guard the race where the parent already exited before prctl ran.
        if os.getppid() == 1:
            os._exit(0)
    except Exception:  # best-effort PDEATHSIG race guard
        pass


def _send(obj: dict) -> None:
    """Write a single JSON response line on the protocol descriptor."""
    data = (json.dumps(obj) + "\n").encode("utf-8", errors="replace")
    os.write(_RESP_FD, data)


def _restricted_builtins(allowed_imports: set[str]) -> dict:
    """Build a restricted ``__builtins__`` mapping for user code.

    Uses the unpatched, pre-guard ``getattr`` reference to read every name off
    ``builtins`` (so introspecting dunder builtins like ``__build_class__``
    never trips the reflection guard); the four reflection entries
    (``getattr``/``setattr``/``delattr``/``hasattr``) still resolve to
    whatever ``builtins`` currently binds them to, so when a guard is active
    this dict picks up the guarded versions for those four names.
    """
    safe = {
        name: _REAL_GETATTR(builtins, name)
        for name in dir(builtins)
        if name not in _RESTRICTED_BUILTINS
    }
    original_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base_module = name.split(".")[0]
        if base_module not in allowed_imports:
            raise ImportError(f"Import of '{name}' is not allowed")
        return original_import(name, globals, locals, fromlist, level)

    safe["__import__"] = safe_import
    return safe


def _serialize_result(value):
    """Return a JSON-safe representation of an execution result.

    Plain JSON-serialisable values pass through unchanged (so ``2 + 2`` stays an
    ``int``); everything else degrades to ``repr`` so the protocol channel never
    breaks on an exotic object.
    """
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        try:
            return repr(value)
        except Exception:  # pragma: no cover - repr of a hostile object
            return "<unrepresentable result>"


def _run_one(namespace: dict, req: dict) -> dict:
    code = req.get("code", "")
    restricted = req.get("restricted_mode", True)
    allowed_imports = set(req.get("allowed_imports", ()))
    return_variables = req.get("return_variables", False)
    max_output = int(req.get("max_output", 102400))

    if req.get("reset"):
        namespace.clear()

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    reflection_guard = _ReflectionGuard() if restricted else contextlib.nullcontext()

    # Redirect the OS-level stdout/stderr to /dev/null during execution so a
    # stray ``os.write(1, ...)`` cannot corrupt the JSON protocol on fd 3.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        with reflection_guard, redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Built *after* the reflection guard is active, so the ``getattr``
            # entry captured here is already the dunder-checked one, not a
            # snapshot of the real function taken before the guard applied.
            if restricted:
                namespace["__builtins__"] = _restricted_builtins(allowed_imports)
            try:
                tree = ast.parse(code, mode="eval")
                compiled = compile(tree, "<repl>", mode="eval")
                result = eval(compiled, namespace, namespace)
            except SyntaxError:
                tree = ast.parse(code, mode="exec")
                compiled = compile(tree, "<repl>", mode="exec")
                exec(compiled, namespace, namespace)
                # Re-evaluate a trailing bare expression (but never a Call, which
                # would run side effects twice) to surface its value.
                if (
                    tree.body
                    and isinstance(tree.body[-1], ast.Expr)
                    and not isinstance(tree.body[-1].value, ast.Call)
                ):
                    last_expr = tree.body[-1].value
                    result = eval(
                        compile(ast.Expression(body=last_expr), "<repl>", mode="eval"),
                        namespace,
                        namespace,
                    )
    except Exception as e:  # noqa: BLE001 - user code may raise anything
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)

    stdout_value = stdout_capture.getvalue()
    stderr_value = stderr_capture.getvalue()
    if len(stdout_value) > max_output:
        stdout_value = stdout_value[:max_output] + "\n... (output truncated)"
    if len(stderr_value) > max_output:
        stderr_value = stderr_value[:max_output] + "\n... (output truncated)"

    response = {
        "success": error is None,
        "result": _serialize_result(result),
        "stdout": stdout_value,
        "stderr": stderr_value,
        "error": error,
    }

    if return_variables:
        variables = {}
        for k, v in namespace.items():
            if k.startswith("_") or k == "__builtins__":
                continue
            try:
                variables[k] = repr(v)
            except Exception:  # pragma: no cover
                variables[k] = "<unrepresentable>"
        response["variables"] = variables

    return response


def main() -> int:
    """Worker loop: read one JSON request per stdin line, write one JSON reply."""
    _install_parent_death_signal()
    _install_audit_hook()
    namespace: dict = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _send({"success": False, "result": None, "stdout": "", "stderr": "",
                   "error": "worker: malformed request"})
            continue
        try:
            _send(_run_one(namespace, req))
        except Exception as e:  # pragma: no cover - defensive
            _send({"success": False, "result": None, "stdout": "", "stderr": "",
                   "error": f"worker fatal: {type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
