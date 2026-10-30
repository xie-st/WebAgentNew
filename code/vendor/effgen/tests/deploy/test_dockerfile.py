"""Dockerfile tests for effGen.

Tests verify:
1. The Dockerfile syntax and structure are correct (always runs).
2. The image builds and /health responds (runs when Docker is accessible).
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "deploy" / "docker" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / "deploy" / "docker" / ".dockerignore"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


DOCKER_AVAILABLE = _docker_available()
requires_docker = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker daemon not accessible (add user to docker group or use rootless docker)",
)


# ---------------------------------------------------------------------------
# Static / syntax tests — always run
# ---------------------------------------------------------------------------

class TestDockerfileExists:
    def test_dockerfile_present(self) -> None:
        assert DOCKERFILE.exists(), f"Dockerfile not found at {DOCKERFILE}"

    def test_dockerignore_present(self) -> None:
        assert DOCKERIGNORE.exists(), f".dockerignore not found at {DOCKERIGNORE}"


class TestDockerfileStructure:
    """Parse and validate Dockerfile structure without running Docker."""

    def setup_method(self) -> None:
        self.content = DOCKERFILE.read_text()

    def test_multi_stage_build(self) -> None:
        """Must have at least two FROM stages."""
        from_lines = [ln for ln in self.content.splitlines() if ln.strip().startswith("FROM")]
        assert len(from_lines) >= 2, f"Expected multi-stage build, got FROM lines: {from_lines}"

    def test_builder_stage(self) -> None:
        assert "AS builder" in self.content, "Expected a 'builder' stage"

    def test_runtime_stage(self) -> None:
        assert "AS runtime" in self.content, "Expected a 'runtime' stage"

    def test_python_version(self) -> None:
        assert "python:3.11" in self.content, "Must use python:3.11 base image"

    def test_non_root_user(self) -> None:
        assert "USER effgen" in self.content or "useradd" in self.content, \
            "Dockerfile must create and switch to a non-root user"

    def test_healthcheck(self) -> None:
        assert "HEALTHCHECK" in self.content, "Dockerfile must include a HEALTHCHECK directive"

    def test_health_endpoint_in_healthcheck(self) -> None:
        assert "/health" in self.content, "HEALTHCHECK must probe the /health endpoint"

    def test_expose_port(self) -> None:
        assert "EXPOSE 8080" in self.content, "Container must EXPOSE port 8080"

    def test_no_secrets_in_env(self) -> None:
        """ENV instructions must not contain API keys or passwords."""
        secret_patterns = [r"API_KEY\s*=\s*\S+", r"PASSWORD\s*=\s*\S+", r"SECRET\s*=\s*\S+"]
        env_lines = [ln for ln in self.content.splitlines() if ln.strip().startswith("ENV")]
        for line in env_lines:
            for pattern in secret_patterns:
                assert not re.search(pattern, line, re.IGNORECASE), \
                    f"Possible secret in ENV directive: {line!r}"

    def test_no_root_default(self) -> None:
        """Final USER directive should not be root."""
        user_lines = [ln.strip() for ln in self.content.splitlines() if ln.strip().startswith("USER")]
        if user_lines:
            last_user = user_lines[-1]
            assert "root" not in last_user.lower(), \
                f"Final USER directive should not be root: {last_user!r}"

    def test_dev_mode_off_by_default(self) -> None:
        """EFFGEN_DEV_MODE default must be 0 in the image."""
        assert "EFFGEN_DEV_MODE=0" in self.content or 'EFFGEN_DEV_MODE="0"' in self.content, \
            "EFFGEN_DEV_MODE must default to 0 (auth on) in the image"

    def test_no_editable_install(self) -> None:
        """Image must NOT use an editable install.

        ``pip install -e`` bakes the absolute build path into a finder file,
        which breaks once the source tree is relocated/removed in the runtime
        stage. The image must use a regular install.
        """
        for line in self.content.splitlines():
            stripped = line.strip()
            if stripped.startswith("RUN") and "pip install" in stripped:
                assert " -e " not in stripped and "pip install -e" not in stripped, \
                    f"Image must not use editable install: {stripped!r}"

    def test_default_extras_are_real(self) -> None:
        """The Dockerfile's default EXTRAS must reference real pyproject extras.

        Guards against silently-skipped extras (pip only warns, exit 0) leaving
        required deps like authlib/prometheus-client out of the image.
        """
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py<3.11
            import tomli as tomllib  # type: ignore[no-redef]

        m = re.search(r'ARG\s+EXTRAS="([^"]*)"', self.content)
        assert m, "Dockerfile must declare a default ARG EXTRAS"
        requested = {e.strip() for e in m.group(1).split(",") if e.strip()}

        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        available = set(pyproject["project"]["optional-dependencies"].keys())
        bogus = requested - available
        assert not bogus, (
            f"Dockerfile default EXTRAS reference non-existent pyproject extras: "
            f"{sorted(bogus)}. Available: {sorted(available)}"
        )

    def test_server_extra_has_prometheus(self) -> None:
        """The /metrics endpoint (scraped by the HPA) needs prometheus-client.

        It must ship with the ``server`` extra the image installs.
        """
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py<3.11
            import tomli as tomllib  # type: ignore[no-redef]
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        server_deps = pyproject["project"]["optional-dependencies"]["server"]
        assert any(d.lower().startswith("prometheus-client") for d in server_deps), \
            "server extra must include prometheus-client for /metrics"

    def test_copies_readme_needed_by_setup_py(self) -> None:
        """The build context must include every readme the build reads.

        pyproject's ``readme`` is README_PYPI.md, but setup.py opens README.md
        for its long_description — omitting either fails metadata generation
        (FileNotFoundError) during ``docker build``.
        """
        copy_lines = " ".join(
            ln for ln in self.content.splitlines() if ln.strip().startswith("COPY")
        )
        for required in ("README_PYPI.md", "README.md"):
            assert required in copy_lines, \
                f"Dockerfile must COPY {required} (read during the build)"


class TestDockignoreStructure:
    def setup_method(self) -> None:
        self.content = DOCKERIGNORE.read_text()

    def test_git_excluded(self) -> None:
        assert ".git/" in self.content or ".git" in self.content

    def test_env_files_excluded(self) -> None:
        assert ".env" in self.content

    def test_pycache_excluded(self) -> None:
        assert "__pycache__/" in self.content


# ---------------------------------------------------------------------------
# Docker integration tests — require daemon access
# ---------------------------------------------------------------------------

IMAGE_TAG = "effgen:test-ci"


@pytest.mark.docker
@requires_docker
class TestDockerBuild:
    """Build the Docker image and verify it runs.

    Marked ``docker`` (not just gated by ``requires_docker``) so the offline /
    order-independence lanes that run ``-m "not docker"`` skip the real
    ``docker build`` entirely — on runners where the daemon exists the build
    would otherwise run and blow past the per-test timeout.
    """

    @pytest.fixture(scope="class", autouse=True)
    def build_image(self) -> None:  # type: ignore[override]
        """Build the effGen Docker image once for all tests in this class."""
        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(DOCKERFILE),
                "--build-arg", "EXTRAS=server",
                "-t", IMAGE_TAG,
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            pytest.fail(f"docker build failed:\n{result.stdout}\n{result.stderr}")
        yield
        # Cleanup: remove test image
        subprocess.run(["docker", "rmi", "-f", IMAGE_TAG], capture_output=True)

    def test_image_exists_after_build(self) -> None:
        result = subprocess.run(
            ["docker", "images", "-q", IMAGE_TAG],
            capture_output=True, text=True,
        )
        assert result.stdout.strip(), f"Image {IMAGE_TAG} not found after build"

    def test_health_endpoint(self) -> None:
        """Run container in dev mode, wait for readiness, poll /health."""
        container_name = "effgen-ci-test"
        # Ensure no stale container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        run_result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", "18080:8080",
                "-e", "EFFGEN_DEV_MODE=1",
                "--read-only",
                "--tmpfs", "/tmp",
                "--tmpfs", "/home/effgen/.effgen",
                IMAGE_TAG,
            ],
            capture_output=True, text=True,
        )
        try:
            assert run_result.returncode == 0, \
                f"docker run failed: {run_result.stderr}"

            # Wait for server to start (up to 30s)
            import urllib.error
            import urllib.request

            deadline = time.time() + 30
            healthy = False
            last_err = ""
            while time.time() < deadline:
                try:
                    resp = urllib.request.urlopen(
                        "http://localhost:18080/health", timeout=3
                    )
                    if resp.status == 200:
                        healthy = True
                        break
                except Exception as exc:
                    last_err = str(exc)
                    time.sleep(1)

            if not healthy:
                # Grab logs for diagnosis
                logs = subprocess.run(
                    ["docker", "logs", container_name],
                    capture_output=True, text=True,
                ).stdout
                pytest.fail(
                    f"/health did not respond within 30s (last error: {last_err})\n"
                    f"Container logs:\n{logs}"
                )
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    def test_non_root_user_in_container(self) -> None:
        """Verify container runs as non-root.

        The image declares ``ENTRYPOINT ["python", "-m", "uvicorn", ...]``, so a
        bare ``docker run IMAGE id -u`` would append ``id -u`` as *arguments to
        uvicorn* rather than running ``id``. Override the entrypoint to invoke
        ``id`` directly.
        """
        result = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "id",
             "-e", "EFFGEN_DEV_MODE=1", IMAGE_TAG, "-u"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"docker run id failed: {result.stderr}"
        uid = result.stdout.strip()
        assert uid != "0", "Container running as root (uid=0), expected non-root"
        assert uid == "1001", f"Expected uid 1001 (effgen user), got {uid!r}"
