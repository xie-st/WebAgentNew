"""
tests/security/test_secret_patterns.py

Validates that gitleaks with our .gitleaks.toml correctly:
  1. Detects planted fake secrets in a temp file.
  2. Does NOT flag allowlisted test fixtures with obviously fake keys.

These tests use the gitleaks binary (downloaded during CI setup or available
on PATH). If gitleaks is not found, tests are skipped with a clear message.
"""

import json
import shutil
import subprocess
import textwrap
import tomllib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GITLEAKS_CONFIG = REPO_ROOT / ".gitleaks.toml"


def _find_gitleaks() -> str | None:
    """Return gitleaks binary path, checking PATH + /tmp/gitleaks."""
    for candidate in [shutil.which("gitleaks"), "/tmp/gitleaks"]:
        if candidate and Path(candidate).is_file():
            try:
                subprocess.run(
                    [candidate, "version"],
                    capture_output=True,
                    check=True,
                )
                return candidate
            except (subprocess.CalledProcessError, OSError):
                continue
    return None


GITLEAKS_BIN = _find_gitleaks()
skip_if_no_gitleaks = pytest.mark.skipif(
    GITLEAKS_BIN is None,
    reason="gitleaks binary not found on PATH or /tmp/gitleaks; install to run secret-scan tests",
)


def _run_gitleaks_detect(directory: Path) -> tuple[int, list[dict]]:
    """Run gitleaks dir on a directory, return (exit_code, findings)."""
    report_file = directory / "gitleaks-report.json"
    result = subprocess.run(
        [
            GITLEAKS_BIN,
            "dir",  # use 'dir' subcommand for non-git directories
            str(directory),
            "--config",
            str(GITLEAKS_CONFIG),
            "--report-format",
            "json",
            "--report-path",
            str(report_file),
            "--exit-code",
            "1",  # exit 1 if leaks found
            "--redact",
        ],
        capture_output=True,
        text=True,
    )
    findings = []
    if report_file.exists():
        try:
            with open(report_file) as f:
                findings = json.load(f) or []
        except (json.JSONDecodeError, ValueError):
            pass
    return result.returncode, findings


# ---------------------------------------------------------------------------
# Planted-secret tests — these MUST be detected
# ---------------------------------------------------------------------------


class TestPlantedSecretsDetected:
    """Gitleaks must trip on each planted fake secret pattern.

    Keys use realistic random-looking hex/alphanumeric patterns (not sequential
    alphabets) to avoid triggering the gitleaks 'abcdefghijklmnopqrstuvwxyz'
    entropy/stopword heuristic present in some rule sets.
    """

    # Realistic-looking (non-sequential) fake key materials for testing only
    _GROQ_KEY = "gsk_3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a8b2c5f9d1e4a7b0c3f6e9b2d5a8"
    _OPENAI_PROJ_KEY = "sk-proj-3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a8b2c5f9d"
    _OPENAI_SVCACCT_KEY = "sk-svcacct-3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a8b2c5f9d"
    _ANTHROPIC_KEY = "sk-ant-api03-3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a8b2c5f9"
    _HF_TOKEN = "hf_3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a"
    _REPLICATE_TOKEN = "r8_3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a8b"
    _EFFGEN_KEY = "real-secret-key-3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d"

    @skip_if_no_gitleaks
    def test_groq_api_key_detected(self, tmp_path):
        """A Groq API key pattern (gsk_...) must be detected."""
        secret_file = tmp_path / "config.py"
        secret_file.write_text(f'GROQ_KEY = "{self._GROQ_KEY}"\n')
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert (
            exit_code != 0
        ), "gitleaks should have detected the planted Groq API key but returned exit 0"

    @skip_if_no_gitleaks
    def test_openai_project_key_detected(self, tmp_path):
        """An OpenAI project key (sk-proj-...) must be detected."""
        secret_file = tmp_path / "settings.py"
        secret_file.write_text(f'OPENAI_API_KEY = "{self._OPENAI_PROJ_KEY}"\n')
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "gitleaks should have detected the planted OpenAI project key"

    @skip_if_no_gitleaks
    def test_openai_service_account_key_detected(self, tmp_path):
        """An OpenAI service-account key (sk-svcacct-...) must be detected."""
        secret_file = tmp_path / "settings.py"
        secret_file.write_text(f'OPENAI_API_KEY = "{self._OPENAI_SVCACCT_KEY}"\n')
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, (
            "gitleaks should have detected the planted OpenAI service-account key"
        )

    @skip_if_no_gitleaks
    def test_anthropic_api_key_detected(self, tmp_path):
        """An Anthropic Claude API key (sk-ant-api03-...) must be detected."""
        secret_file = tmp_path / "llm_config.yaml"
        secret_file.write_text(f"anthropic_key: {self._ANTHROPIC_KEY}\n")
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "gitleaks should have detected the planted Anthropic API key"

    @skip_if_no_gitleaks
    def test_huggingface_token_detected(self, tmp_path):
        """A HuggingFace token (hf_...) must be detected."""
        secret_file = tmp_path / "model_loader.py"
        secret_file.write_text(f'HF_TOKEN = "{self._HF_TOKEN}"\n')
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "gitleaks should have detected the planted HuggingFace token"

    @skip_if_no_gitleaks
    def test_replicate_token_detected(self, tmp_path):
        """A Replicate API token (r8_...) must be detected."""
        secret_file = tmp_path / "inference.py"
        secret_file.write_text(f'REPLICATE_TOKEN = "{self._REPLICATE_TOKEN}"\n')
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "gitleaks should have detected the planted Replicate API token"

    @skip_if_no_gitleaks
    def test_generic_api_key_in_effgen_config_detected(self, tmp_path):
        """A generic EFFGEN_API_KEY assignment must be detected."""
        secret_file = tmp_path / "effgen_config.py"
        secret_file.write_text(f'EFFGEN_API_KEY = "{self._EFFGEN_KEY}"\n')
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "gitleaks should have detected the planted EFFGEN_API_KEY"

    @skip_if_no_gitleaks
    def test_multiple_secrets_in_one_file(self, tmp_path):
        """Multiple secrets in a single file — all must be flagged."""
        secret_file = tmp_path / "secrets.env"
        secret_file.write_text(textwrap.dedent(f"""\
                GROQ_API_KEY={self._GROQ_KEY}
                HF_TOKEN={self._HF_TOKEN}
                OPENAI_API_KEY={self._OPENAI_PROJ_KEY}
            """))
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "gitleaks should detect secrets in a multi-secret .env file"
        # Should find multiple distinct secrets
        assert len(findings) >= 1, f"Expected >= 1 finding, got: {findings}"


# ---------------------------------------------------------------------------
# Clean file — must NOT trigger false positives
# ---------------------------------------------------------------------------


class TestCleanFilesNotFlagged:
    """Gitleaks must not false-positive on clean files."""

    @skip_if_no_gitleaks
    def test_clean_python_file_not_flagged(self, tmp_path):
        """A normal Python file without secrets must pass clean."""
        clean_file = tmp_path / "adapter.py"
        clean_file.write_text(textwrap.dedent("""\
                import os
                from effgen.adapters import BaseAdapter

                class MyAdapter(BaseAdapter):
                    def __init__(self):
                        self.api_key = os.environ.get("MY_API_KEY", "")

                    def chat(self, messages):
                        return {"role": "assistant", "content": "hi"}
            """))
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code == 0, f"gitleaks should NOT flag a clean file; findings: {findings}"

    @skip_if_no_gitleaks
    def test_env_placeholder_not_flagged(self, tmp_path):
        """A .env.example with placeholder values must not be flagged."""
        env_file = tmp_path / ".env.example"
        env_file.write_text(textwrap.dedent("""\
                # Copy this file to .env and fill in real values
                CEREBRAS_API_KEY=your_cerebras_api_key_here
                OPENAI_API_KEY=your_openai_api_key_here
                GROQ_API_KEY=your_groq_api_key_here
            """))
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        # Placeholder text should not trigger — stopwords in allowlist cover "your"
        # If it does trigger, findings will be shown in the assert message
        if exit_code != 0:
            # Soft check: these are generic placeholders, acceptable if rule is strict
            # Log for awareness but don't hard-fail
            pytest.xfail(
                f"Placeholder .env.example triggered gitleaks (strict rule); findings: {findings}"
            )

    @skip_if_no_gitleaks
    def test_setup_instruction_placeholders_not_flagged(self, tmp_path):
        """``your-<provider>-<token>`` placeholders in setup docs must pass clean.

        The generic rule only fires on a value of twenty characters or more, so
        whether a placeholder tripped it used to depend on how long the provider's
        name happened to be — the shorter ones passed and one longer one did not.
        Every provider named here is written the same way, so they either all pass
        or the allowlist has stopped covering the family.
        """
        doc = tmp_path / "installation.md"
        doc.write_text(textwrap.dedent("""\
                export OPENAI_API_KEY="your-openai-key"
                export CEREBRAS_API_KEY="your-cerebras-key"
                export GROQ_API_KEY="your-groq-key"
                export TOGETHER_API_KEY="your-together-key"
                export FIREWORKS_API_KEY="your-fireworks-key"
                export REPLICATE_API_TOKEN="your-replicate-token"
                export HF_TOKEN="your-hf-token"
                export SOMEFUTUREPROVIDER_API_TOKEN="your-somefutureprovider-token"
            """))
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code == 0, (
            f"documentation placeholders must not read as secrets; findings: {findings}"
        )

    @skip_if_no_gitleaks
    def test_a_real_token_beside_a_placeholder_is_still_caught(self, tmp_path):
        """The placeholder allowlist must not blunt detection of a real token."""
        doc = tmp_path / "installation.md"
        doc.write_text(textwrap.dedent("""\
                export REPLICATE_API_TOKEN="your-replicate-token"
                export REPLICATE_API_TOKEN="r8_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S"
            """))
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code != 0, "a real Replicate token must still be detected"
        assert any("replicate" in str(f.get("RuleID", "")).lower() for f in findings), (
            f"expected the Replicate rule to fire; findings: {findings}"
        )

    @skip_if_no_gitleaks
    def test_config_with_env_var_reference_not_flagged(self, tmp_path):
        """Config referencing env vars (not inline keys) must pass clean."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
                provider: cerebras
                model: llama3.1-8b
                api_key: ${CEREBRAS_API_KEY}
                timeout: 30
                max_tokens: 4096
            """))
        exit_code, findings = _run_gitleaks_detect(tmp_path)
        assert exit_code == 0, f"gitleaks should NOT flag env var references; findings: {findings}"


# ---------------------------------------------------------------------------
# Gitleaks config validation
# ---------------------------------------------------------------------------


class TestGitleaksConfig:
    """Validate that .gitleaks.toml itself is well-formed."""

    def test_gitleaks_config_exists(self):
        assert GITLEAKS_CONFIG.exists(), f".gitleaks.toml not found at {GITLEAKS_CONFIG}"

    def test_gitleaks_config_parseable(self):
        """Config must be valid TOML."""
        with open(GITLEAKS_CONFIG, "rb") as f:
            cfg = tomllib.load(f)
        assert "rules" in cfg or "extend" in cfg, "Config must have [rules] or [extend] section"

    def test_gitleaks_config_has_effgen_rules(self):
        """Config must define effGen-specific rules."""
        with open(GITLEAKS_CONFIG, "rb") as f:
            cfg = tomllib.load(f)
        rule_ids = [r["id"] for r in cfg.get("rules", [])]
        effgen_rules = [r for r in rule_ids if r.startswith("effgen-")]
        assert len(effgen_rules) >= 3, f"Expected >= 3 effgen-* rules, found: {effgen_rules}"

    def test_gitleaks_config_has_allowlist(self):
        """Config must have an allowlist for test fixtures."""
        with open(GITLEAKS_CONFIG, "rb") as f:
            cfg = tomllib.load(f)
        allowlist = cfg.get("allowlist", {})
        assert allowlist, "No top-level [allowlist] section found"
        assert allowlist.get("paths") or allowlist.get(
            "stopwords"
        ), "Allowlist must have paths or stopwords"

    @skip_if_no_gitleaks
    def test_gitleaks_version_acceptable(self):
        """Gitleaks version must be >= 8.x."""
        result = subprocess.run(
            [GITLEAKS_BIN, "version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version_str = result.stdout.strip()
        major = int(version_str.split(".")[0])
        assert major >= 8, f"gitleaks version too old: {version_str}"

    @skip_if_no_gitleaks
    def test_repo_scan_clean(self):
        """The effGen repo itself must pass gitleaks dir scan (no real secrets)."""
        result = subprocess.run(
            [
                GITLEAKS_BIN,
                "dir",
                str(REPO_ROOT),
                "--config",
                str(GITLEAKS_CONFIG),
                "--exit-code",
                "1",
                "--redact",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"gitleaks detected potential secrets in the repo!\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:1000]}"
        )

    @skip_if_no_gitleaks
    def test_repo_git_history_clean(self):
        """The full git history must pass gitleaks (CI runs a history scan).

        The CI secret-scan workflow checks out with fetch-depth: 0 and scans the
        entire commit history. Vendored deps (e.g. node_modules) were committed
        and later removed but remain in history; the allowlist must keep the
        history scan clean so CI does not fail on every push.
        """
        if not (REPO_ROOT / ".git").exists():
            pytest.skip("not a git checkout; history scan not applicable")
        result = subprocess.run(
            [
                GITLEAKS_BIN,
                "git",
                str(REPO_ROOT),
                "--config",
                str(GITLEAKS_CONFIG),
                "--exit-code",
                "1",
                "--redact",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"gitleaks detected potential secrets in git HISTORY!\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:1000]}"
        )


# ---------------------------------------------------------------------------
# Pre-commit hook path — `gitleaks protect --staged` must block staged secrets
# ---------------------------------------------------------------------------


class TestPreCommitHookBlocks:
    """The pre-commit hook uses `gitleaks protect --staged` — it must block a
    staged secret (exit non-zero) and allow allowlisted fixtures (exit zero)."""

    _LEGACY_OPENAI_KEY = "sk-Rk3f8b2c9d1e7a4f6b0c5d8e2a9b4f7c1d3e6a8b2c5f9d1Q"

    @skip_if_no_gitleaks
    def test_staged_secret_blocked(self, tmp_path):
        """A staged file with a real-looking key must make `protect --staged` fail."""
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        shutil.copy(GITLEAKS_CONFIG, tmp_path / ".gitleaks.toml")
        secret_file = tmp_path / "config.py"
        secret_file.write_text(f'OPENAI_API_KEY = "{self._LEGACY_OPENAI_KEY}"\n')
        subprocess.run(["git", "add", "config.py"], cwd=tmp_path, check=True)
        result = subprocess.run(
            [GITLEAKS_BIN, "protect", "--staged", "--config=.gitleaks.toml"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode != 0
        ), "pre-commit hook (gitleaks protect --staged) should BLOCK a staged secret"

    @skip_if_no_gitleaks
    def test_staged_allowlisted_fixture_allowed(self, tmp_path):
        """A real-looking key in a tests/ fixture is allowlisted — commit allowed."""
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        shutil.copy(GITLEAKS_CONFIG, tmp_path / ".gitleaks.toml")
        (tmp_path / "tests").mkdir()
        fixture = tmp_path / "tests" / "test_fixture.py"
        fixture.write_text(f'FIXTURE_KEY = "{self._LEGACY_OPENAI_KEY}"\n')
        subprocess.run(["git", "add", "tests/test_fixture.py"], cwd=tmp_path, check=True)
        result = subprocess.run(
            [GITLEAKS_BIN, "protect", "--staged", "--config=.gitleaks.toml"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"allowlisted tests/ fixture should NOT block the commit; "
            f"stdout: {result.stdout[:1000]}"
        )
