"""
End-to-end secret-redaction tests.

These set fake secrets and push them through the *real* code paths that surface
text to a user — structured logging, provider-failure errors, tool-execution
errors, and the ``effgen doctor`` CLI — asserting the raw secret value never
appears in any output. Covers the provider key shapes the redactor must handle,
including the ones a prior audit found missing (Together, Replicate, Fireworks,
AWS, Azure, and generic ``*_API_KEY=…`` env dumps).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest

from effgen.observability.redact import Redactor

# Fake secret values — each long enough to satisfy its pattern. None of these
# are real; they exist only so we can assert they never leak.
FAKE_SECRETS = {
    "openai": "sk-FAKEopenai0123456789abcdefghijABCD",
    "anthropic": "sk-ant-FAKE0123456789abcdefghijklmnop",
    "replicate": "r8_FAKEabcdefghijklmnopqrstuvwxyz0123456789",
    "fireworks": "fw_FAKEabcdefghijklmnop0123",
    "aws_id": "AKIAFAKE1234567890XY",
    "aws_secret_env": "AWS_SECRET_ACCESS_KEY=wJalrFAKE/K7MDENG/bPxRfiCYzEXAMPLEKEY",
    "together_env": "TOGETHER_API_KEY=FAKE1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f",
    "azure_env": "AZURE_OPENAI_API_KEY=FAKE0123456789abcdef0123456789ab",
    "generic_env": "MY_SERVICE_API_KEY=FAKEsupersecretvalue123",
}


def _raw_values() -> list[str]:
    """The exact substrings that must NOT appear anywhere in redacted output."""
    out = []
    for v in FAKE_SECRETS.values():
        # For "VAR=value" shapes only the value half is the secret.
        out.append(v.split("=", 1)[1] if "=" in v and v.split("=", 1)[0].isupper() else v)
    return out


def _assert_no_leak(text: str) -> None:
    for secret in _raw_values():
        assert secret not in text, f"LEAKED secret fragment {secret!r} in: {text!r}"


# ---------------------------------------------------------------------------
# 1. Redactor covers every provider key shape
# ---------------------------------------------------------------------------

class TestProviderKeyShapes:
    def test_all_shapes_redacted(self):
        r = Redactor()
        blob = " ".join(FAKE_SECRETS.values())
        out = r.scrub(blob)
        _assert_no_leak(out)

    def test_fallback_runs_after_user_pattern(self):
        # A user's own pattern must still win on its own span; the generic
        # env_secret fallback runs last.
        r = Redactor()
        r.add_pattern("my_token", r"mytok-[A-Za-z0-9]{8,}")
        out = r.scrub("AUTH_TOKEN=mytok-abcdef123")
        assert "mytok-abcdef123" not in out

    def test_benign_text_preserved(self):
        r = Redactor()
        for benign in (
            "status: success",
            "request_id: 12345 done",
            "the result is 42 tokens",
            "category: computation",
        ):
            assert r.scrub(benign) == benign


# ---------------------------------------------------------------------------
# 2. Structured logging path (server errors + general logs flow through here)
# ---------------------------------------------------------------------------

class TestStructuredLoggingRedaction:
    def test_attributes_and_exception_redacted(self, capsys):
        from effgen.observability.logs import StructuredFormatter

        formatter = StructuredFormatter(redact=True)
        try:
            raise RuntimeError(
                f"upstream call failed: {FAKE_SECRETS['replicate']} "
                f"{FAKE_SECRETS['together_env']}"
            )
        except RuntimeError:
            record = logging.LogRecord(
                name="effgen.server",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="request failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        # Attach secret-bearing extra= fields the way a caller would.
        record.authorization = f"Bearer {FAKE_SECRETS['openai']}"  # type: ignore[attr-defined]
        record.fireworks_key = FAKE_SECRETS["fireworks"]  # type: ignore[attr-defined]
        record.env_dump = FAKE_SECRETS["generic_env"]  # type: ignore[attr-defined]
        out = formatter.format(record)
        # It must be valid JSON and contain no raw secrets.
        json.loads(out)
        _assert_no_leak(out)


# ---------------------------------------------------------------------------
# 3. Provider-failure error path
# ---------------------------------------------------------------------------

class TestProviderFailureRedaction:
    def test_provider_runtime_error_redacted(self):
        from effgen.models._adapter_utils import provider_runtime_error

        raw = RuntimeError(
            f"401 Unauthorized: key {FAKE_SECRETS['openai']} / "
            f"{FAKE_SECRETS['aws_secret_env']}"
        )
        err = provider_runtime_error("openai", "gpt-5-nano", "chat", raw)
        _assert_no_leak(str(err))
        # The structured context carries no secret material either.
        _assert_no_leak(json.dumps(err.error_context))


# ---------------------------------------------------------------------------
# 4. Tool-execution error path
# ---------------------------------------------------------------------------

class TestToolErrorRedaction:
    def test_tool_redact_error(self):
        from effgen.tools.base_tool import _redact_error

        msg = (
            f"Tool execution failed: connecting with {FAKE_SECRETS['replicate']} "
            f"and {FAKE_SECRETS['azure_env']}"
        )
        _assert_no_leak(_redact_error(msg))


# ---------------------------------------------------------------------------
# 5. CLI doctor never echoes secret values
# ---------------------------------------------------------------------------

class TestDoctorCLIRedaction:
    def test_doctor_does_not_print_key_values(self):
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = FAKE_SECRETS["openai"]
        env["REPLICATE_API_TOKEN"] = FAKE_SECRETS["replicate"]
        env["TOGETHER_API_KEY"] = FAKE_SECRETS["together_env"].split("=", 1)[1]
        env["EFFGEN_TIPS"] = "0"
        proc = subprocess.run(
            [sys.executable, "-m", "effgen", "doctor"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        combined = proc.stdout + proc.stderr
        # doctor reports key *presence*, never the value.
        assert FAKE_SECRETS["openai"] not in combined
        assert FAKE_SECRETS["replicate"] not in combined
        assert FAKE_SECRETS["together_env"].split("=", 1)[1] not in combined


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
