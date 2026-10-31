"""CLI smoke tests — verify effgen CLI commands work."""

import subprocess
import sys


def _run_cli(*args, timeout=30):
    """Run effgen CLI command and return CompletedProcess."""
    cmd = [sys.executable, "-m", "effgen.cli"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class TestCLIVersion:
    """Test effgen --version."""

    def test_version_exits_zero(self):
        result = _run_cli("--version")
        assert result.returncode == 0

    def test_version_contains_effgen(self):
        result = _run_cli("--version")
        output = result.stdout + result.stderr
        assert "effGen" in output or "effgen" in output.lower()


class TestCLIHelp:
    """Test effgen --help."""

    def test_help_exits_zero(self):
        result = _run_cli("--help")
        assert result.returncode == 0

    def test_help_shows_usage(self):
        result = _run_cli("--help")
        output = result.stdout + result.stderr
        assert "usage" in output.lower() or "effgen" in output.lower()


class TestCLIPresets:
    """Test effgen presets command."""

    def test_presets_exits_zero(self):
        result = _run_cli("presets")
        assert result.returncode == 0

    def test_presets_lists_names(self):
        result = _run_cli("presets")
        output = result.stdout + result.stderr
        for name in ["math", "research", "coding", "general", "minimal"]:
            assert name in output.lower(), f"Preset '{name}' not found in output"


class TestCLIHealth:
    """Test effgen health command."""

    def test_health_runs(self):
        """Health command should run (may fail network checks but should not crash)."""
        result = _run_cli("health")
        # Accept 0 (all checks pass) or 1 (some checks fail due to network)
        assert result.returncode in (0, 1), f"Unexpected return code: {result.returncode}\nstderr: {result.stderr}"

    def test_health_produces_output(self):
        result = _run_cli("health")
        output = result.stdout + result.stderr
        assert len(output) > 0, "Health command produced no output"
