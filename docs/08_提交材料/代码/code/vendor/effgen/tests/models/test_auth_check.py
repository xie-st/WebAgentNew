"""
Tests for effgen.models.auth.check_keys.

Uses environment variable mocking (no network calls).
"""

from __future__ import annotations

import pytest

from effgen.models.auth import check_keys
from effgen.models.registry import ProviderRegistry


class _DummyAdapter:
    pass


DUMMY_MODELS = {"dummy-model": {"context": 4_096}}


@pytest.fixture(autouse=True)
def reset_registry():
    """Only the fake providers registered here may be visible to check_keys."""
    ProviderRegistry.clear()
    yield
    ProviderRegistry.reset()


class TestCheckKeys:
    def test_key_present_returns_available(self, monkeypatch):
        ProviderRegistry.register("testprov", _DummyAdapter, DUMMY_MODELS, env_keys=["TEST_API_KEY"])
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        result = check_keys(["testprov"])
        assert result["testprov"]["available"] is True
        assert result["testprov"]["env_key"] == "TEST_API_KEY"

    def test_key_absent_returns_not_available(self, monkeypatch):
        ProviderRegistry.register("testprov", _DummyAdapter, DUMMY_MODELS, env_keys=["TEST_API_KEY"])
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        result = check_keys(["testprov"])
        assert result["testprov"]["available"] is False
        assert result["testprov"]["env_key"] is None

    def test_first_env_key_wins(self, monkeypatch):
        ProviderRegistry.register(
            "testprov", _DummyAdapter, DUMMY_MODELS,
            env_keys=["FIRST_KEY", "SECOND_KEY"],
        )
        monkeypatch.delenv("FIRST_KEY", raising=False)
        monkeypatch.setenv("SECOND_KEY", "sk-second")
        result = check_keys(["testprov"])
        assert result["testprov"]["available"] is True
        assert result["testprov"]["env_key"] == "SECOND_KEY"

    def test_first_key_takes_priority(self, monkeypatch):
        ProviderRegistry.register(
            "testprov", _DummyAdapter, DUMMY_MODELS,
            env_keys=["FIRST_KEY", "SECOND_KEY"],
        )
        monkeypatch.setenv("FIRST_KEY", "sk-first")
        monkeypatch.setenv("SECOND_KEY", "sk-second")
        result = check_keys(["testprov"])
        assert result["testprov"]["env_key"] == "FIRST_KEY"

    def test_multiple_providers(self, monkeypatch):
        ProviderRegistry.register("prov_a", _DummyAdapter, DUMMY_MODELS, env_keys=["PROV_A_KEY"])
        ProviderRegistry.register("prov_b", _DummyAdapter, DUMMY_MODELS, env_keys=["PROV_B_KEY"])
        monkeypatch.setenv("PROV_A_KEY", "key_a")
        monkeypatch.delenv("PROV_B_KEY", raising=False)
        result = check_keys()
        assert result["prov_a"]["available"] is True
        assert result["prov_b"]["available"] is False

    def test_unregistered_provider_reports_error(self):
        result = check_keys(["nonexistent"])
        assert result["nonexistent"]["available"] is False
        assert "error" in result["nonexistent"]

    def test_no_env_keys_means_unavailable(self):
        ProviderRegistry.register("testprov", _DummyAdapter, DUMMY_MODELS, env_keys=[])
        result = check_keys(["testprov"])
        assert result["testprov"]["available"] is False
        assert result["testprov"]["env_key"] is None

    def test_env_keys_checked_field(self, monkeypatch):
        ProviderRegistry.register(
            "testprov", _DummyAdapter, DUMMY_MODELS,
            env_keys=["KEY_ONE", "KEY_TWO"],
        )
        monkeypatch.delenv("KEY_ONE", raising=False)
        monkeypatch.delenv("KEY_TWO", raising=False)
        result = check_keys(["testprov"])
        assert result["testprov"]["env_keys_checked"] == ["KEY_ONE", "KEY_TWO"]

    def test_check_keys_with_real_providers(self, monkeypatch):
        """Smoke: check_keys returns a dict with expected provider names."""
        import effgen.models.groq_adapter as _ga
        import effgen.models.together_adapter as _ta
        _ga._register()
        _ta._register()

        # Manually set one key to confirm availability detection
        monkeypatch.setenv("GROQ_API_KEY", "sk-smoke-test")
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        result = check_keys(["groq", "together"])
        assert result["groq"]["available"] is True
        assert result["together"]["available"] is False
