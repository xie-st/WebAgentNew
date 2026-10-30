"""
Tests for effgen.observability.redact — secret redaction module.

Verifies that every built-in pattern fires correctly on known fixtures and
that redaction propagates through nested data structures.
"""

from __future__ import annotations

import pytest

from effgen.observability.redact import Redactor, get_redactor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh() -> Redactor:
    """Return a brand-new Redactor so tests are isolated from the singleton."""
    return Redactor()


# ---------------------------------------------------------------------------
# Fixture constants — use values that are long enough to satisfy each regex
# ---------------------------------------------------------------------------

OPENAI_KEY   = "sk-abcdefghijklmnopqrstuvwxyz123456"          # 28 chars after sk-
ANTHROPIC_KEY = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz"   # starts sk-ant-
CEREBRAS_KEY = "csk-AbCdEfGhIjKlMnOpQrStUvWxYz123456"       # starts csk-
GOOGLE_KEY   = "AIzaSyD" + "A" * 35                           # AIza + 35 chars
HF_KEY       = "hf_" + "a" * 25                               # hf_ + 25 chars
GROQ_KEY     = "gsk_" + "g" * 25                              # gsk_ + 25 chars
BEARER_TOKEN = "Bearer mysupersecrettoken123"
SLACK_HOOK   = "https://hooks.slack.com/services/ABC123/DEF456/GHIJklmnopqrst"
DISCORD_HOOK = "https://discord.com/api/webhooks/1234567890/abc-DEF_xyz"


# ---------------------------------------------------------------------------
# Pattern coverage tests
# ---------------------------------------------------------------------------

class TestOpenAIKey:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(f"my key is {OPENAI_KEY}")
        assert OPENAI_KEY not in out
        assert "<REDACTED:openai_key>" in out

    def test_not_consumed_by_other_pattern(self):
        # openai key must NOT survive when embedded in a larger string
        r = fresh()
        out = r.scrub(f"Authorization: {OPENAI_KEY}")
        assert OPENAI_KEY not in out

    def test_short_sk_not_redacted(self):
        # Keys shorter than 20 chars after "sk-" should NOT match
        r = fresh()
        short = "sk-short"
        assert r.scrub(short) == short  # unchanged


class TestAnthropicKey:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(f"Using {ANTHROPIC_KEY} today")
        assert ANTHROPIC_KEY not in out
        assert "<REDACTED:anthropic_key>" in out

    def test_not_classified_as_openai(self):
        # Anthropic key starts with sk-ant- which also matches sk- openai pattern.
        # The anthropic pattern is compiled first, so it should win.
        r = fresh()
        out = r.scrub(ANTHROPIC_KEY)
        assert "<REDACTED:anthropic_key>" in out
        # Must not also be replaced a second time
        assert out.count("<REDACTED:") == 1


class TestCerebrasKey:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(f"key={CEREBRAS_KEY}")
        assert CEREBRAS_KEY not in out
        assert "<REDACTED:cerebras_key>" in out


class TestGoogleKey:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(f"key={GOOGLE_KEY}")
        assert GOOGLE_KEY not in out
        assert "<REDACTED:google_key>" in out

    def test_exact_length(self):
        # AIza + exactly 35 chars → must match
        key = "AIza" + "B" * 35
        r = fresh()
        assert key not in r.scrub(key)


class TestHuggingFaceKey:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(HF_KEY)
        assert HF_KEY not in out
        assert "<REDACTED:hf_key>" in out


class TestGroqKey:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(GROQ_KEY)
        assert GROQ_KEY not in out
        assert "<REDACTED:groq_key>" in out


class TestBearerToken:
    def test_standalone(self):
        r = fresh()
        out = r.scrub(f"Authorization: {BEARER_TOKEN}")
        # The secret part must be gone
        assert "mysupersecrettoken123" not in out
        assert "<REDACTED:bearer_token>" in out

    def test_short_bearer_not_redacted(self):
        # 6 char minimum after "Bearer " — "Bearer short" (5 chars) stays
        r = fresh()
        assert r.scrub("Bearer x") == "Bearer x"  # 1 char — too short

    def test_bearer_with_key(self):
        # A bearer token that IS an openai key — only one replacement expected
        token = f"Bearer {OPENAI_KEY}"
        r = fresh()
        out = r.scrub(token)
        # The raw key must be gone
        assert OPENAI_KEY not in out


class TestSlackWebhook:
    def test_full_url(self):
        r = fresh()
        out = r.scrub(f"webhook={SLACK_HOOK}")
        assert SLACK_HOOK not in out
        assert "<REDACTED:slack_webhook>" in out


class TestDiscordWebhook:
    def test_full_url(self):
        r = fresh()
        out = r.scrub(f"url={DISCORD_HOOK}")
        assert DISCORD_HOOK not in out
        assert "<REDACTED:discord_webhook>" in out


# ---------------------------------------------------------------------------
# Nested structure redaction
# ---------------------------------------------------------------------------

class TestScrubDict:
    def test_flat(self):
        r = fresh()
        data = {"api_key": OPENAI_KEY, "name": "alice"}
        clean = r.scrub_dict(data)
        assert OPENAI_KEY not in clean["api_key"]
        assert clean["name"] == "alice"

    def test_nested(self):
        r = fresh()
        data = {"config": {"key": CEREBRAS_KEY, "debug": True}}
        clean = r.scrub_dict(data)
        assert CEREBRAS_KEY not in clean["config"]["key"]
        assert clean["config"]["debug"] is True

    def test_list_values(self):
        r = fresh()
        data = {"keys": [OPENAI_KEY, ANTHROPIC_KEY]}
        clean = r.scrub_dict(data)
        for k in clean["keys"]:
            assert OPENAI_KEY not in k
            assert ANTHROPIC_KEY not in k

    def test_non_string_values_pass_through(self):
        r = fresh()
        data = {"count": 42, "active": True, "ratio": 3.14}
        clean = r.scrub_dict(data)
        assert clean == data


class TestScrubValue:
    def test_string(self):
        r = fresh()
        assert OPENAI_KEY not in r.scrub_value(OPENAI_KEY)

    def test_non_string_unchanged(self):
        r = fresh()
        assert r.scrub_value(42) == 42
        assert r.scrub_value(True) is True
        assert r.scrub_value(None) is None

    def test_list(self):
        r = fresh()
        out = r.scrub_value([OPENAI_KEY, "safe"])
        assert OPENAI_KEY not in out[0]
        assert out[1] == "safe"

    def test_tuple(self):
        r = fresh()
        out = r.scrub_value((OPENAI_KEY,))
        assert isinstance(out, tuple)
        assert OPENAI_KEY not in out[0]


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestAddPattern:
    def test_custom_pattern_fires(self):
        r = fresh()
        r.add_pattern("custom_token", r"tok-[A-Za-z0-9]{10,}")
        secret = "tok-abcdef12345"
        out = r.scrub(f"token={secret}")
        assert secret not in out
        assert "<REDACTED:custom_token>" in out

    def test_pattern_names_includes_custom(self):
        r = fresh()
        r.add_pattern("my_secret", r"mysecret[0-9]+")
        assert "my_secret" in r.pattern_names()

    def test_builtin_names_present(self):
        r = fresh()
        names = r.pattern_names()
        for expected in [
            "openai_key", "anthropic_key", "cerebras_key",
            "google_key", "hf_key", "groq_key",
            "bearer_token", "slack_webhook", "discord_webhook",
        ]:
            assert expected in names, f"Missing pattern: {expected}"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_same_instance_returned(self):
        r1 = get_redactor()
        r2 = get_redactor()
        assert r1 is r2

    def test_add_pattern_persists(self):
        r = get_redactor()
        # Adding a pattern to the singleton — verify it redacts
        r.add_pattern("test_singleton_pat", r"TEST_SECRET_[A-Z]+")
        out = r.scrub("key=TEST_SECRET_ABCDEF")
        assert "TEST_SECRET_ABCDEF" not in out


# ---------------------------------------------------------------------------
# Regression: no false positives on safe strings
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    @pytest.mark.parametrize("safe", [
        "hello world",
        "sk-",           # too short
        "sk-short",      # too short (< 20 chars)
        "csk-",          # too short
        "hf_",           # too short
        "Bearer ",       # no token after space
        "Bearer x",      # 1 char — too short
        "https://example.com/api",
        "AIza" + "B" * 10,  # google key too short (< 35 chars)
    ])
    def test_safe_strings_unchanged(self, safe: str):
        r = fresh()
        # Safe strings should not be modified (barring coincidental matches)
        result = r.scrub(safe)
        assert "<REDACTED:" not in result, f"False positive for: {safe!r} → {result!r}"
