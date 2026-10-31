"""Run the suite without the ambient state of the machine it runs on.

A test that reads the developer's home directory, an environment variable nobody in
the suite set, or a host on the network passes on the machine that has those things
and fails on the machine that does not. The failure lands on whoever runs the suite
next, usually in CI, and the cause is invisible from the test body.

Setting ``EFFGEN_TEST_HERMETIC=1`` removes all three before collection starts:

* ``HOME`` (with ``USERPROFILE`` and the ``XDG_*`` directories) points at an empty
  temporary directory, so a read of ``~/.effgen`` finds nothing rather than the
  developer's own configuration, history and cost database;
* the process environment is reduced to a declared allowlist — no provider
  credentials, no CI, editor, terminal, session or toolchain-adjacent variables — and
  the repository ``.env`` is not loaded, by the suite or by anything it starts;
* a connection to anything other than loopback is refused with a typed error, so a
  test that reaches the internet says so instead of depending on the runner's route.

Everything the suite itself sets afterwards (the isolated cost database, the run
history directory) survives, because the scrub runs first.

Two declared exceptions to the scrub:

* the artifact caches (``HF_HOME`` and friends) are re-pointed at their real
  locations. They are a content-addressed download store, not a decision input, and
  dropping them turns "this model is already on disk" into a network fetch, which
  the same guard then refuses. Set ``EFFGEN_TEST_HERMETIC_CACHES=0`` to drop them too
  and measure the suite against an empty cache.
* the interpreter's own search paths (``PATH``, ``LD_LIBRARY_PATH``, the conda and
  virtualenv prefixes) are kept, because a subprocess test needs to find the
  interpreter it was started with.
"""

from __future__ import annotations

import errno
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

ENABLE_VAR = "EFFGEN_TEST_HERMETIC"
KEEP_CACHES_VAR = "EFFGEN_TEST_HERMETIC_CACHES"

#: Environment variables kept verbatim: how to find the interpreter and its libraries,
#: the locale and temporary directory, and the harness's own controls.
ALLOWED_NAMES = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_EXE",
        "CONDA_PYTHON_EXE",
        "VIRTUAL_ENV",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_NUMERIC",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "PWD",
        "EFFGEN_TEST_HERMETIC",
        "EFFGEN_TEST_HERMETIC_CACHES",
        "EFFGEN_TEST_REVERSE_ORDER",
        "EFFGEN_LANE_TIMING",
        "EFFGEN_LANE_TIMING_TABLE",
        "EFFGEN_SOAK_SCALE",
    }
)

#: Prefixes kept verbatim: the interpreter's own configuration and pytest's.
ALLOWED_PREFIXES = ("PYTHON", "PYTEST_", "_PYTEST", "COV_CORE_", "COVERAGE_")

#: Artifact caches, re-pointed at their real locations unless caches are dropped.
#: ``HF_HUB_CACHE`` is where current huggingface_hub keeps downloaded weights and
#: it takes precedence over ``HF_HOME``; a setup that points it at a shared store
#: keeps nothing under ``$HF_HOME/hub``, so dropping it turns every cached model
#: into a download the network guard then refuses.
CACHE_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "HF_XET_CACHE",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "TORCH_HOME",
)

#: Home-shaped variables pointed at the empty temporary home.
HOME_NAMES = (
    "HOME",
    "USERPROFILE",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
)

_LOOPBACK_HOSTS = frozenset(
    {
        "",
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "127.0.0.1",
        "0.0.0.0",  # a bind address rather than a destination
        "::",
        "::1",
    }
)


class AmbientNetworkRefused(OSError):
    """A test reached for a host other than loopback while the scrub was active."""


_state: dict[str, Any] = {
    "active": False,
    "root": None,
    "home": None,
    "dropped": (),
    "blocked": [],
}


def requested() -> bool:
    """Whether the scrub was asked for through the environment."""
    return (os.environ.get(ENABLE_VAR) or "").strip().lower() in ("1", "true", "yes", "on")


def is_active() -> bool:
    """Whether the scrub has been applied to this process."""
    return bool(_state["active"])


def home() -> Path | None:
    """The empty home the scrub installed, if it is active."""
    return _state["home"]


def blocked_connections() -> list[str]:
    """Every destination the network guard refused, in the order it refused them."""
    return list(_state["blocked"])


def _keeps(name: str) -> bool:
    if name in ALLOWED_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _host_of(address: Any) -> str:
    if isinstance(address, tuple | list) and address:
        host = address[0]
    else:
        host = address
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    return str(host)


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_HOSTS:
        return True
    return host.startswith(("127.", "::ffff:127."))


def _address_allowed(family: Any, address: Any) -> bool:
    if family == getattr(socket, "AF_UNIX", None):
        return True
    if family not in (socket.AF_INET, socket.AF_INET6):
        return True
    return _is_loopback(_host_of(address))


def _install_network_guard() -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_getaddrinfo = socket.getaddrinfo

    def _refuse(target: str) -> AmbientNetworkRefused:
        _state["blocked"].append(target)
        return AmbientNetworkRefused(
            errno.ENETUNREACH,
            f"the test suite is running with the ambient environment removed "
            f"({ENABLE_VAR}=1), so only loopback is reachable; {target} was refused. "
            f"A test that needs a network peer should start one on loopback or skip.",
        )

    def connect(self: socket.socket, address: Any) -> Any:  # type: ignore[override]
        if not _address_allowed(self.family, address):
            raise _refuse(_host_of(address))
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:  # type: ignore[override]
        if not _address_allowed(self.family, address):
            _state["blocked"].append(_host_of(address))
            return errno.ENETUNREACH
        return real_connect_ex(self, address)

    def sendto(self: socket.socket, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        address = args[-1] if args else None
        if address is not None and not _address_allowed(self.family, address):
            raise _refuse(_host_of(address))
        return real_sendto(self, *args, **kwargs)

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        name = _host_of(host)
        if not _is_loopback(name):
            _state["blocked"].append(name)
            raise socket.gaierror(
                socket.EAI_NONAME,
                f"name resolution is unavailable while the ambient environment is "
                f"removed ({ENABLE_VAR}=1): {name}",
            )
        return real_getaddrinfo(host, port, *args, **kwargs)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = sendto  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]


def _stop_dotenv_loading(real_home: str) -> None:
    """Refuse the ``.env`` files that carry this machine's credentials.

    Around a hundred test modules call ``load_dotenv()`` at import to pick up the
    repository ``.env``. Clearing the process environment does not reach those, so
    the run would still hold provider credentials and the key-gated tests would
    still make live calls. The loader is wrapped rather than removed: a call that
    resolves to the repository's own ``.env``, or to anything under the real home
    directory, returns without reading it, and every other path — a ``.env`` a test
    wrote in its own temporary directory, for one — loads the way it always did.
    ``EFFGEN_NO_DOTENV`` covers anything the suite starts as a subprocess.
    """
    try:
        import dotenv
    except ModuleNotFoundError:
        return

    home_path = Path(real_home).resolve()
    repo_env = (Path(__file__).resolve().parent.parent.parent / ".env").resolve()

    def _carries_this_machines_credentials(path: Any) -> bool:
        if path is None:
            return True  # the search walk, which lands on the repository's own
        try:
            resolved = Path(str(path)).resolve()
        except OSError:
            return False
        if resolved == repo_env:
            return True
        return resolved == home_path or home_path in resolved.parents

    for module in (dotenv, getattr(dotenv, "main", None)):
        if module is None:
            continue
        for name, refused in (("load_dotenv", False), ("dotenv_values", {})):
            real = getattr(module, name, None)
            if real is None:
                continue
            setattr(module, name, _guard(real, refused, _carries_this_machines_credentials))


def _guard(real: Any, refused: Any, is_blocked: Any) -> Any:
    def wrapper(dotenv_path: Any = None, *args: Any, **kwargs: Any) -> Any:
        path = dotenv_path if dotenv_path is not None else kwargs.get("dotenv_path")
        if is_blocked(path):
            return refused
        return real(dotenv_path, *args, **kwargs)

    return wrapper


def activate() -> bool:
    """Apply the scrub when it was requested. Returns whether it is now active.

    Safe to call more than once; only the first call changes anything.
    """
    if _state["active"] or not requested():
        return bool(_state["active"])

    real_home = os.environ.get("HOME") or str(Path.home())
    keep_caches = (os.environ.get(KEEP_CACHES_VAR) or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    cache_values = {name: os.environ.get(name) for name in CACHE_NAMES}

    root = Path(tempfile.mkdtemp(prefix="effgen_hermetic_"))
    empty_home = root / "home"
    empty_home.mkdir(parents=True, exist_ok=True)

    dropped = tuple(sorted(name for name in os.environ if not _keeps(name)))
    for name in dropped:
        os.environ.pop(name, None)

    # The XDG directories live beside the empty home rather than inside it, so a read
    # of ``~`` finds nothing at all.
    for name in HOME_NAMES:
        if name in ("HOME", "USERPROFILE"):
            os.environ[name] = str(empty_home)
            continue
        target = root / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)

    if keep_caches:
        defaults = {
            "HF_HOME": Path(real_home) / ".cache" / "huggingface",
            "TORCH_HOME": Path(real_home) / ".cache" / "torch",
        }
        for name in CACHE_NAMES:
            value = cache_values.get(name) or defaults.get(name)
            if value is not None:
                os.environ[name] = str(value)

    # Anything the suite starts as a subprocess walks the filesystem for a .env of
    # its own; this is the switch that already turns that walk off.
    os.environ["EFFGEN_NO_DOTENV"] = "1"

    _stop_dotenv_loading(real_home)
    _install_network_guard()

    _state.update(active=True, root=root, home=empty_home, dropped=dropped)
    return True


def cleanup() -> None:
    """Remove the temporary home. The environment is not put back."""
    root = _state.get("root")
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)
        _state["root"] = None


def report_line() -> str:
    """One line describing the scrub, for the pytest header."""
    if not is_active():
        return ""
    dropped = _state["dropped"]
    caches = "kept" if os.environ.get("HF_HOME") else "dropped"
    return (
        f"ambient environment removed: HOME={_state['home']}, "
        f"{len(dropped)} variables dropped, artifact caches {caches}, "
        f"non-loopback sockets refused"
    )
