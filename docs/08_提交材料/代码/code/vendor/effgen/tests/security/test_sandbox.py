"""
Sandbox backend tests.

Test coverage:
  - SandboxConfig: defaults and env-var overrides
  - SubprocessSandbox: basic execution, timeout, resource limits, network block
  - DockerSandbox: availability detection, network block (when Docker available)
  - FirecrackerSandbox: stub correctly raises NotImplementedError
  - get_sandbox(): auto-selection logic
  - CodeExecutor: routes through sandbox, returns sandbox_backend field

Network-block test (DockerSandbox):
  Runs: python3 -c "import urllib.request; urllib.request.urlopen('https://example.com')"
  inside --network=none container → must fail (non-zero exit).

Skip markers:
  - docker tests skip if Docker daemon is unreachable
  - subprocess tests always run (subprocess is always available)
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import shutil
import subprocess as sp
import tempfile
from pathlib import Path

import pytest

from effgen.security.sandbox import (
    DockerSandbox,
    FirecrackerSandbox,
    OffSandbox,
    SandboxConfig,
    SubprocessSandbox,
    get_sandbox,
    reset_sandbox_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine in a fresh event loop (3.11–3.13 safe)."""
    return asyncio.run(coro)


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        result = sp.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def _userns_works() -> bool:
    """Return True if unprivileged user namespaces are usable on this host."""
    if platform.system() != "Linux" or not shutil.which("unshare"):
        return False
    try:
        return sp.run(
            ["unshare", "--map-root-user", "true"],
            capture_output=True, timeout=5,
        ).returncode == 0
    except Exception:
        return False


def _confinement_works() -> bool:
    """Return True if this **host** provides the primitives write confinement needs.

    Deliberately does not consult the sandbox's own probe. Asking the
    implementation whether it works makes the guard self-referential: a
    regression that breaks confinement also makes the probe report "cannot
    confine", so every test pinning the contract would skip instead of fail and
    the suite would stay green through the exact regression it exists to catch.

    So exercise the kernel primitives directly, in a namespace of this
    function's own making: bind-mount two throwaway directories, remount one
    read-only, hand the result to a nested user + mount namespace, and require
    that a write inside succeeds, a write outside fails, and the read-only
    mount cannot be remounted read-write. When all of that holds the host is
    capable, and any failure below is the sandbox's, not the environment's.
    """
    if not _userns_works() or not shutil.which("unshare"):
        return False
    parent = tempfile.mkdtemp(prefix="effgen_host_confine_cap_")
    try:
        inside = os.path.join(parent, "in")
        outside = os.path.join(parent, "out")
        os.mkdir(inside)
        os.mkdir(outside)
        q_in, q_out = shlex.quote(inside), shlex.quote(outside)
        nested = (
            f"touch {q_in}/x 2>/dev/null || exit 13; "
            f"touch {q_out}/y 2>/dev/null && exit 14; "
            f"mount -o remount,bind,rw {q_out} 2>/dev/null && exit 15; "
            "exit 0"
        )
        inner = (
            f"mount --bind {q_in} {q_in} 2>/dev/null || exit 10; "
            f"mount --bind {q_out} {q_out} 2>/dev/null || exit 11; "
            f"mount -o remount,bind,ro {q_out} 2>/dev/null || exit 12; "
            f"exec unshare --map-root-user --mount bash -c {shlex.quote(nested)}"
        )
        return sp.run(
            ["unshare", "--map-root-user", "--mount", "bash", "-c", inner],
            capture_output=True, timeout=60,
        ).returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(parent, ignore_errors=True)


DOCKER_AVAILABLE = _docker_available()
USERNS_AVAILABLE = _userns_works()
CONFINEMENT_AVAILABLE = _confinement_works()

docker_required = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker daemon not reachable",
)
userns_required = pytest.mark.skipif(
    not USERNS_AVAILABLE,
    reason="unprivileged user namespaces not available",
)
confinement_required = pytest.mark.skipif(
    not CONFINEMENT_AVAILABLE,
    reason="this host cannot confine sandbox writes (no locked mount namespace)",
)


# ---------------------------------------------------------------------------
# SandboxConfig tests
# ---------------------------------------------------------------------------

class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        # backend defaults to EFFGEN_SANDBOX_BACKEND env or "auto"
        assert cfg.backend in ("auto", "docker", "subprocess")
        assert cfg.timeout > 0
        assert cfg.memory_limit.endswith(("m", "g", "k")) or cfg.memory_limit.isdigit()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
        monkeypatch.setenv("EFFGEN_SANDBOX_TIMEOUT", "42")
        monkeypatch.setenv("EFFGEN_SANDBOX_MEMORY", "128m")
        cfg = SandboxConfig.from_env()
        assert cfg.backend == "subprocess"
        assert cfg.timeout == 42
        assert cfg.memory_limit == "128m"

    def test_network_disabled_by_default(self):
        cfg = SandboxConfig()
        assert cfg.network_enabled is False


# ---------------------------------------------------------------------------
# SubprocessSandbox tests
# ---------------------------------------------------------------------------

class TestSubprocessSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_is_always_available(self):
        sb = SubprocessSandbox()
        assert _run(sb.is_available()) is True

    def test_python_hello_world(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=15, memory_limit="256m")
        result = _run(sb.run("print('hello sandbox')", "python", cfg))
        assert result.exit_code == 0
        assert "hello sandbox" in result.stdout
        assert result.backend_used == "subprocess"

    def test_bash_execution(self):
        if not shutil.which("bash"):
            pytest.skip("bash not available")
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("echo 'bash works'", "bash", cfg))
        assert result.exit_code == 0
        assert "bash works" in result.stdout

    def test_python_syntax_error(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("def broken(:", "python", cfg))
        assert result.exit_code != 0
        assert result.stderr  # error message should be present

    def test_timeout_enforced(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=3, memory_limit="256m")
        # Python infinite loop should time out
        result = _run(sb.run("while True: pass", "python", cfg))
        assert result.timed_out is True or result.exit_code != 0

    def test_result_fields_present(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("print(1+1)", "python", cfg))
        assert isinstance(result.stdout, str)
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)
        assert isinstance(result.execution_time, float)
        assert isinstance(result.timed_out, bool)
        assert result.backend_used == "subprocess"

    def test_stderr_captured(self):
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=10)
        result = _run(sb.run("import sys; sys.stderr.write('err output')", "python", cfg))
        assert "err output" in result.stderr

    @userns_required
    def test_subprocess_network_block(self):
        """
        On Linux with unprivileged user namespaces, the subprocess sandbox runs
        inside `unshare --map-root-user --net`, so urllib.request.urlopen fails.
        """
        reset_sandbox_cache()
        SubprocessSandbox._caps_probed = False  # re-probe in this env
        sb = SubprocessSandbox()
        cfg = SandboxConfig(backend="subprocess", timeout=15, memory_limit="256m")
        code = (
            "import urllib.request, sys\n"
            "try:\n"
            "    urllib.request.urlopen('https://example.com', timeout=5)\n"
            "    print('NETWORK_OK')\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    print(f'NETWORK_BLOCKED: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        result = _run(sb.run(code, "python", cfg))
        assert result.exit_code != 0, (
            "Expected network to be blocked by unshare --net, but got exit_code="
            f"{result.exit_code}, stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
        assert "NETWORK_OK" not in result.stdout

    @userns_required
    def test_subprocess_fs_isolation_protects_host_tmp(self):
        """
        `rm -rf /tmp/<sentinel>` inside the sandbox must NOT touch the host:
        the sandbox mounts a private tmpfs over /tmp inside its mount namespace.

        Skipped if the host mount namespace cannot be created (mount isolation
        is the protection being tested).
        """
        SubprocessSandbox._caps_probed = False
        if not _run(SubprocessSandbox._unshare_succeeds(["--map-root-user", "--mount", "true"])):
            pytest.skip("mount namespace not available — FS isolation cannot apply")

        # The sandbox shields the literal /tmp path; place the sentinel there.
        sentinel = Path("/tmp/effgen_fs_isolation_sentinel.txt")
        sentinel.write_text("do-not-delete", encoding="utf-8")
        try:
            SubprocessSandbox._caps_probed = False
            sb = SubprocessSandbox()
            cfg = SandboxConfig(backend="subprocess", timeout=15, memory_limit="256m")
            code = (
                "import os\n"
                "os.system('rm -rf /tmp/effgen_fs_isolation_sentinel.txt')\n"
                "print('ran rm')\n"
            )
            result = _run(sb.run(code, "python", cfg))
            assert "ran rm" in result.stdout
            # The host sentinel must survive — the rm hit the namespace tmpfs only.
            assert sentinel.exists(), (
                "Host /tmp sentinel was deleted — FS isolation failed!"
            )
        finally:
            sentinel.unlink(missing_ok=True)

    @confinement_required
    def test_unmounting_the_private_tmp_does_not_reveal_the_host_tmp(self):
        """The private tmpfs must not be removable from inside the sandbox.

        Executed code is root in the sandbox's user namespace, so an unlocked
        tmpfs over /tmp is one ``umount`` away from the writable host /tmp
        underneath. The nested namespace locks it: the unmount is refused and
        a write to the host path leaves nothing behind.
        """
        sentinel = Path("/tmp/effgen_host_tmp_unmount_probe.txt")
        sentinel.unlink(missing_ok=True)
        workspace = Path(tempfile.mkdtemp(dir=str(Path.home())))
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            code = (
                "umount /tmp 2>&1 | head -1\n"
                f"echo escaped > {sentinel} 2>&1 || echo 'write refused'\n"
                "echo done\n"
            )
            result = _run(sb.run(code, "bash", cfg))
            assert "done" in result.stdout, result.stdout
            assert not sentinel.exists(), (
                "unmounting the private /tmp exposed the writable host /tmp"
            )
        finally:
            sentinel.unlink(missing_ok=True)
            shutil.rmtree(workspace, ignore_errors=True)

    @confinement_required
    def test_scratch_root_with_spaces_and_quotes_is_handled(self):
        """The scratch root is the first caller-controlled string in the
        sandbox command, so it must survive shell metacharacters."""
        parent = Path(tempfile.mkdtemp(dir=str(Path.home())))
        workspace = parent / "a b'c $d*"
        workspace.mkdir()
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            code = (
                "import os\n"
                "open('inside.txt', 'w').write('ok')\n"
                "print('cwd', os.getcwd())\n"
            )
            result = _run(sb.run(code, "python", cfg))
            assert result.exit_code == 0, result.stderr
            assert (workspace / "inside.txt").read_text() == "ok"
            assert result.writable_root == str(workspace.resolve())
        finally:
            shutil.rmtree(parent, ignore_errors=True)

    def test_result_reports_what_the_backend_enforced(self):
        """``filesystem_confined``/``writable_root`` describe the run that
        happened, never a guarantee the environment could not provide."""
        workspace = Path(tempfile.mkdtemp(dir=str(Path.home())))
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            result = _run(sb.run("print('ok')", "python", cfg))
            assert result.exit_code == 0, result.stderr
            assert result.network_isolated is USERNS_AVAILABLE
            if CONFINEMENT_AVAILABLE:
                assert result.filesystem_confined is True
                assert result.writable_root == str(workspace.resolve())
            else:
                assert result.filesystem_confined is False
                assert result.writable_root is None
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_confinement_is_not_claimed_when_the_probe_fails(self, monkeypatch):
        """A host that cannot lock the mounts gets the previous behavior and
        says so, rather than reporting a boundary that is not there."""
        # The degraded path still mounts a private tmpfs over /tmp, so a target
        # under the system temp directory is invisible to the executed code. The
        # home directory is the location outside the workspace that stays visible
        # — unless it is itself under the temp directory, where no such location
        # exists and the case has no premise.
        home = Path.home().resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if home == temp_root or temp_root in home.parents:
            pytest.skip(
                "the home directory is under the system temp directory, which the "
                "degraded path replaces with a private tmpfs"
            )
        outside = Path(tempfile.mkdtemp(dir=str(home), prefix="effgen_sandbox_"))
        workspace = outside / "workspace"
        workspace.mkdir()
        target = outside / "degraded_probe_target.txt"
        try:
            reset_sandbox_cache()

            async def _probe_fails(cls, unshare_bin):
                cls._confine_probed = True
                cls._confine_ok = False

            monkeypatch.setattr(
                SubprocessSandbox,
                "_probe_confinement",
                classmethod(_probe_fails),
            )
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            result = _run(
                sb.run(f"open({str(target)!r}, 'w').write('x')", "python", cfg)
            )
            assert result.filesystem_confined is False
            assert result.writable_root is None
            if USERNS_AVAILABLE:
                # Unconfined is exactly the previous behavior: the write lands.
                assert result.exit_code == 0, result.stderr
                assert target.exists()
        finally:
            shutil.rmtree(outside, ignore_errors=True)
            reset_sandbox_cache()

    def test_parse_mem_kb(self):
        assert SubprocessSandbox._parse_mem_kb("256m") == 256 * 1024
        assert SubprocessSandbox._parse_mem_kb("1g") == 1024 * 1024
        assert SubprocessSandbox._parse_mem_kb("512k") == 512

    @confinement_required
    def test_write_outside_the_scratch_root_is_refused(self):
        """The contract: executed code writes only inside its scratch space.

        The run's working directory is the one writable host path; every other
        mount is read-only, so a write to a directory the calling user owns
        outside it fails with a read-only-filesystem error and leaves no file
        behind.

        Uses a directory under the home tree (NOT pytest's tmp_path, so the
        refusal cannot be confused with the private /tmp shadowing the path)."""
        probe_dir = Path(tempfile.mkdtemp(dir=str(Path.home())))
        workspace = Path(tempfile.mkdtemp(dir=str(Path.home())))
        target = probe_dir / "outside_root_probe.txt"
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            code = f"""
with open({str(target)!r}, "w") as f:
    f.write("written-from-sandbox")
"""
            result = _run(sb.run(code, "python", cfg))
            assert result.exit_code != 0, (
                f"write outside the scratch root succeeded: {result.stdout!r}"
            )
            assert "Read-only file system" in result.stderr, result.stderr
            assert not target.exists(), "a host file was created outside the scratch root"
            assert result.filesystem_confined is True
            assert result.writable_root == str(workspace.resolve())
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
            shutil.rmtree(workspace, ignore_errors=True)

    @confinement_required
    def test_reads_outside_the_scratch_root_still_work(self):
        """Reads are deliberately NOT confined — only writes are.

        The companion to the test above: the same path that refuses a write is
        still readable, which is why the fallback warning names read exposure.
        """
        probe_dir = Path(tempfile.mkdtemp(dir=str(Path.home())))
        workspace = Path(tempfile.mkdtemp(dir=str(Path.home())))
        source = probe_dir / "readable.txt"
        source.write_text("readable-from-sandbox", encoding="utf-8")
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            result = _run(sb.run(f"print(open({str(source)!r}).read())", "python", cfg))
            assert result.exit_code == 0, result.stderr
            assert "readable-from-sandbox" in result.stdout
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
            shutil.rmtree(workspace, ignore_errors=True)

    @confinement_required
    def test_writes_inside_the_scratch_root_work_relative_and_absolute(self):
        """The behavior confinement must preserve: the agent's own files.

        Executed code starts in the scratch root, so both a relative path and
        the absolute form of the same directory are writable.
        """
        workspace = Path(tempfile.mkdtemp(dir=str(Path.home())))
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            code = (
                "open('relative.txt', 'w').write('r')\n"
                f"open({str(workspace / 'absolute.txt')!r}, 'w').write('a')\n"
                "print('both written')\n"
            )
            result = _run(sb.run(code, "python", cfg))
            assert result.exit_code == 0, result.stderr
            assert (workspace / "relative.txt").read_text() == "r"
            assert (workspace / "absolute.txt").read_text() == "a"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @confinement_required
    def test_executed_code_cannot_unlock_the_read_only_mounts(self):
        """Executed code is root in the sandbox's user namespace, so the
        read-only remounts are only a boundary while the kernel keeps them
        locked. Remounting read-write, unmounting, and doing either from a
        nested user namespace of the code's own making must all be refused —
        and no host file may appear afterwards."""
        probe_dir = Path(tempfile.mkdtemp(dir=str(Path.home())))
        workspace = Path(tempfile.mkdtemp(dir=str(Path.home())))
        target = probe_dir / "after_escape.txt"
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=25, workdir=str(workspace)
            )
            code = (
                "mount -o remount,bind,rw / 2>&1 | head -1\n"
                "umount / 2>&1 | head -1\n"
                "unshare --map-root-user --mount "
                "mount -o remount,bind,rw / 2>&1 | head -1\n"
                f"echo escaped > {target} 2>&1 || echo 'write still refused'\n"
            )
            result = _run(sb.run(code, "bash", cfg))
            assert "write still refused" in result.stdout, result.stdout
            assert not target.exists(), "escaped the scratch root after remount attempts"
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
            shutil.rmtree(workspace, ignore_errors=True)

    @confinement_required
    def test_scratch_root_under_the_temp_directory_is_usable(self):
        """A workspace inside the system temp directory is writable, and the
        host temp directory around it is not.

        The private tmpfs that shields /tmp would hide such a workspace, so it
        is skipped there; the read-only remount covers the host temp directory
        instead, and TMPDIR points at the workspace so temp files still work.
        """
        workspace = Path(tempfile.mkdtemp(prefix="effgen_tmp_workspace_"))
        host_probe = Path(tempfile.gettempdir()) / "effgen_host_tmp_write_probe.txt"
        host_probe.unlink(missing_ok=True)
        try:
            sb = SubprocessSandbox()
            cfg = SandboxConfig(
                backend="subprocess", timeout=15, workdir=str(workspace)
            )
            code = (
                "import tempfile\n"
                "open('relative.txt', 'w').write('r')\n"
                f"open({str(workspace / 'absolute.txt')!r}, 'w').write('a')\n"
                "print('scratch:', tempfile.mkstemp()[1])\n"
                "try:\n"
                f"    open({str(host_probe)!r}, 'w').write('x')\n"
                "    print('HOST TMP WRITABLE')\n"
                "except OSError as exc:\n"
                "    print('host tmp refused:', exc.strerror)\n"
            )
            result = _run(sb.run(code, "python", cfg))
            assert result.exit_code == 0, result.stderr
            assert (workspace / "relative.txt").exists()
            assert (workspace / "absolute.txt").exists()
            assert "host tmp refused: Read-only file system" in result.stdout
            assert not host_probe.exists()
            assert f"scratch: {workspace}" in result.stdout, result.stdout
        finally:
            host_probe.unlink(missing_ok=True)
            shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Subprocess fallback warning
# ---------------------------------------------------------------------------

class TestSubprocessFallbackWarning:
    """A WARNING naming the real filesystem exposure fires the first time
    SubprocessSandbox is resolved — whether via auto-fallback (Docker
    unavailable) or an explicit EFFGEN_SANDBOX_BACKEND=subprocess, which
    previously skipped the warning entirely.

    The exposure the banner must name is *reads*: the subprocess backend
    confines writes to the run's working directory but lets executed code read
    anything the calling user can read."""

    def setup_method(self):
        reset_sandbox_cache()

    def teardown_method(self):
        reset_sandbox_cache()

    def test_explicit_subprocess_backend_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="effgen.security.sandbox"):
            _run(get_sandbox(SandboxConfig(backend="subprocess")))
        assert any(
            "reads are NOT confined" in rec.message for rec in caplog.records
        ), "explicit subprocess backend must warn about filesystem exposure"

    def test_auto_fallback_warns_when_docker_unavailable(self, caplog, monkeypatch):
        import logging

        from effgen.security import sandbox as sandbox_mod

        async def _unavailable(self):
            return False

        monkeypatch.setattr(sandbox_mod.DockerSandbox, "is_available", _unavailable)
        with caplog.at_level(logging.WARNING, logger="effgen.security.sandbox"):
            _run(get_sandbox(SandboxConfig(backend="auto")))
        assert any(
            "reads are NOT confined" in rec.message for rec in caplog.records
        )

    def test_warning_fires_once_per_process(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="effgen.security.sandbox"):
            _run(get_sandbox(SandboxConfig(backend="subprocess")))
            _run(get_sandbox(SandboxConfig(backend="subprocess")))
        hits = [
            rec for rec in caplog.records if "reads are NOT confined" in rec.message
        ]
        assert len(hits) == 1

    def test_docker_backend_does_not_warn(self, caplog):
        import logging

        docker = DockerSandbox()
        if not _run(docker.is_available()):
            pytest.skip("Docker not available in this environment")
        with caplog.at_level(logging.WARNING, logger="effgen.security.sandbox"):
            _run(get_sandbox(SandboxConfig(backend="docker")))
        assert not any(
            "reads are NOT confined" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# DockerSandbox tests
# ---------------------------------------------------------------------------

class TestDockerSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_is_available_returns_bool(self):
        sb = DockerSandbox()
        result = _run(sb.is_available())
        assert isinstance(result, bool)
        assert result == DOCKER_AVAILABLE

    @docker_required
    def test_python_hello_world(self):
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=30,
            memory_limit="256m",
            network_enabled=False,
        )
        result = _run(sb.run("print('docker sandbox works')", "python", cfg))
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "docker sandbox works" in result.stdout
        assert result.backend_used == "docker"

    @docker_required
    def test_network_block(self):
        """
        Core network-block test:
        urllib.request.urlopen inside --network=none container must fail.
        """
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=30,
            memory_limit="256m",
            network_enabled=False,
        )
        code = (
            "import urllib.request, sys\n"
            "try:\n"
            "    urllib.request.urlopen('https://example.com', timeout=5)\n"
            "    print('NETWORK_OK')\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    print(f'NETWORK_BLOCKED: {type(e).__name__}: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        result = _run(sb.run(code, "python", cfg))
        assert result.exit_code != 0, (
            "Expected network request to FAIL inside --network=none container, "
            f"but exit_code={result.exit_code}, stdout={result.stdout!r}"
        )
        assert result.backend_used == "docker"

    @docker_required
    def test_syntax_error_captured(self):
        sb = DockerSandbox()
        cfg = SandboxConfig(backend="docker", timeout=30, memory_limit="256m")
        result = _run(sb.run("this is not valid python !!!", "python", cfg))
        assert result.exit_code != 0
        assert result.stderr  # Python error in stderr

    @docker_required
    def test_timeout_enforced(self):
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=5,
            memory_limit="256m",
            network_enabled=False,
        )
        result = _run(sb.run("while True: pass", "python", cfg))
        assert result.timed_out is True or result.exit_code != 0

    @docker_required
    def test_memory_limit_applied(self):
        """Verify container runs under memory cap (256m by default)."""
        sb = DockerSandbox()
        cfg = SandboxConfig(
            backend="docker",
            timeout=30,
            memory_limit="256m",
            network_enabled=False,
        )
        # A simple calculation should complete fine
        result = _run(sb.run("print(sum(range(1000000)))", "python", cfg))
        assert result.exit_code == 0
        assert "499999500000" in result.stdout


# ---------------------------------------------------------------------------
# FirecrackerSandbox stub tests
# ---------------------------------------------------------------------------

class TestFirecrackerSandbox:
    def test_is_not_available(self):
        sb = FirecrackerSandbox()
        assert _run(sb.is_available()) is False

    def test_run_raises_not_implemented(self):
        sb = FirecrackerSandbox()
        cfg = SandboxConfig()
        with pytest.raises(NotImplementedError, match="FirecrackerSandbox"):
            _run(sb.run("print('hi')", "python", cfg))


# ---------------------------------------------------------------------------
# OffSandbox tests (explicit, unsafe)
# ---------------------------------------------------------------------------

class TestOffSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        OffSandbox._warned = False
        yield
        reset_sandbox_cache()
        OffSandbox._warned = False

    def test_is_available(self):
        assert _run(OffSandbox().is_available()) is True

    def test_executes_directly(self):
        sb = OffSandbox()
        cfg = SandboxConfig(backend="off", timeout=10)
        result = _run(sb.run("print('off backend ran')", "python", cfg))
        assert result.exit_code == 0
        assert "off backend ran" in result.stdout
        assert result.backend_used == "off"

    def test_emits_loud_warning(self, caplog):
        import logging
        sb = OffSandbox()
        cfg = SandboxConfig(backend="off", timeout=10)
        with caplog.at_level(logging.WARNING, logger="effgen.security.sandbox"):
            _run(sb.run("print(1)", "python", cfg))
        assert any(
            "SANDBOX DISABLED" in rec.message or "directly on the host" in rec.message.lower()
            for rec in caplog.records
        ), "OffSandbox must emit a loud warning"

    def test_selected_via_env(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "off")
        cfg = SandboxConfig.from_env()
        sb = _run(get_sandbox(cfg))
        assert isinstance(sb, OffSandbox)

    def test_off_never_chosen_by_auto(self):
        """Auto-selection must never pick OffSandbox."""
        sb = _run(get_sandbox(SandboxConfig(backend="auto")))
        assert not isinstance(sb, OffSandbox)


# ---------------------------------------------------------------------------
# get_sandbox() auto-selection tests
# ---------------------------------------------------------------------------

class TestGetSandbox:
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_auto_selection_returns_sandbox(self):
        sandbox = _run(get_sandbox())
        assert isinstance(sandbox, DockerSandbox | SubprocessSandbox)

    def test_auto_returns_docker_when_available(self):
        if not DOCKER_AVAILABLE:
            pytest.skip("Docker not available")
        sb = _run(get_sandbox())
        assert isinstance(sb, DockerSandbox)

    def test_explicit_subprocess_selection(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
        cfg = SandboxConfig.from_env()
        sb = _run(get_sandbox(cfg))
        assert isinstance(sb, SubprocessSandbox)

    def test_invalid_backend_raises(self):
        cfg = SandboxConfig(backend="invalid_backend_xyz")
        with pytest.raises(ValueError, match="Unknown EFFGEN_SANDBOX_BACKEND"):
            _run(get_sandbox(cfg))

    def test_caching_returns_same_instance(self):
        sb1 = _run(get_sandbox())
        sb2 = _run(get_sandbox())
        assert sb1 is sb2

    def test_explicit_backend_is_cached_across_calls(self):
        # A pinned backend is cached for the process lifetime, the same as the
        # auto path — repeated resolutions return the one instance regardless of
        # whether the backend was chosen via config or the environment variable.
        cfg = SandboxConfig(backend="subprocess")
        sb1 = _run(get_sandbox(cfg))
        sb2 = _run(get_sandbox(cfg))
        assert isinstance(sb1, SubprocessSandbox)
        assert sb1 is sb2


# ---------------------------------------------------------------------------
# CodeExecutor integration test (routes through sandbox)
# ---------------------------------------------------------------------------

class TestCodeExecutorSandboxIntegration:
    """
    Verify CodeExecutor._execute() routes through the sandbox backend
    and returns the sandbox_backend field.
    """

    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        reset_sandbox_cache()
        yield
        reset_sandbox_cache()

    def test_code_executor_uses_sandbox(self):
        from effgen.tools.builtin.code_executor import CodeExecutor

        executor = CodeExecutor()
        # Force initialize with subprocess backend for reliability
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
            reset_sandbox_cache()
            executor._sandbox_config = SandboxConfig.from_env()
            _run(executor.initialize())

        result = _run(executor._execute(
            code="print('executor routed')",
            language="python",
            timeout=15,
            memory_limit="256m",
            network_enabled=False,
        ))

        assert result["exit_code"] == 0
        assert "executor routed" in result["stdout"]
        assert "sandbox_backend" in result
        assert result["sandbox_backend"] in ("docker", "subprocess")

    def test_code_executor_no_longer_executes_directly(self):
        """CodeExecutor must have a sandbox; direct subprocess calls are gone."""
        from effgen.tools.builtin.code_executor import CodeExecutor

        executor = CodeExecutor()
        # Verify the old _execute_docker / _execute_fallback are NOT present
        assert not hasattr(executor, "_execute_docker"), (
            "CodeExecutor should not have _execute_docker; sandbox handles this"
        )
        assert not hasattr(executor, "_execute_fallback"), (
            "CodeExecutor should not have _execute_fallback; sandbox handles this"
        )


class TestPublicCodeExecutorStatesItsIsolation:
    """`from effgen import CodeExecutor` documents its isolation level plainly."""

    def test_public_export_still_resolves(self):
        import effgen
        from effgen.execution.sandbox import CodeExecutor as _SandboxExecutor

        # Backward-compatible: the public name still points at the wrapper.
        assert effgen.CodeExecutor is _SandboxExecutor

    def test_local_backend_docstrings_state_they_are_not_os_isolation(self):
        from effgen.execution.sandbox import CodeExecutor, LocalSandbox

        for doc in (CodeExecutor.__doc__, LocalSandbox.__doc__):
            assert doc is not None
            low = doc.lower()
            # Names the hardened path and disclaims OS-level isolation.
            assert "code_executor" in low
            assert "not" in low and ("isolat" in low or "boundary" in low)

    def test_module_docstring_points_at_the_hardened_tool(self):
        import effgen.execution.sandbox as mod

        assert mod.__doc__ is not None
        assert "effgen.security.sandbox" in mod.__doc__
        assert "code_executor" in mod.__doc__

    def test_package_and_validator_docstrings_do_not_overclaim_safety(self):
        # The execution package defaults to LocalSandbox; a static validator
        # narrows the attack surface but is not an OS boundary, so neither the
        # package nor the validator module may claim it makes execution "safe"
        # or "secure" without naming the container/namespace isolation instead.
        import effgen.execution as pkg
        import effgen.execution.validators as validators

        pkg_doc = (pkg.__doc__ or "").lower()
        assert "secure code execution" not in pkg_doc
        assert "code_executor" in pkg_doc

        val_doc = (validators.__doc__ or "").lower()
        assert "ensuring safe execution" not in val_doc
        # It states plainly that the static screen is not an OS boundary.
        assert "static" in val_doc and "not" in val_doc
