"""A backend that never answered is not a result.

Two behaviours together stop a dead endpoint from reading like a model that
could not solve the task: ``raise_on_error`` defaults to True, and a run that
never reached its backend raises whatever that flag says.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_config import AgentConfig
from effgen.models.errors import (
    UNREACHABLE_SIGNALS,
    BackendUnreachableError,
    classify_provider_error,
)


class TestTellingApartADeadBackend:
    """Which failures mean nothing answered at all."""

    @pytest.mark.parametrize("phrase", UNREACHABLE_SIGNALS)
    def test_a_backend_that_never_answered_is_labelled_unreachable(self, phrase):
        assert classify_provider_error(Exception(phrase)).category == "unreachable"

    def test_a_refused_connection_by_class_name_is_unreachable(self):
        assert classify_provider_error(
            ConnectionRefusedError("[Errno 111] Connection refused")
        ).category == "unreachable"

    @pytest.mark.parametrize("phrase", [
        "connection reset by peer",
        "connection aborted",
        "server disconnected",
        "service unavailable",
        "503 upstream overloaded",
    ])
    def test_a_server_that_answered_badly_is_still_transient(self, phrase):
        # These reached a server. Retrying against it can work, and the caller's
        # problem is the provider's health rather than the endpoint address.
        assert classify_provider_error(Exception(phrase)).category == "transient"

    def test_an_unreachable_backend_is_worth_another_attempt(self):
        assert classify_provider_error(Exception("connection refused")).should_retry

    def test_the_remediation_names_what_to_check(self):
        from effgen.models.errors import REMEDIATION_BY_CATEGORY

        guidance = REMEDIATION_BY_CATEGORY["unreachable"]
        assert "base_url" in guidance or "running" in guidance


class TestTheError:
    """What the caller is handed."""

    def test_it_names_the_endpoint_that_did_not_answer(self):
        error = BackendUnreachableError(
            "openai_compatible", "Qwen/Qwen2.5-7B-Instruct",
            "connection refused", endpoint="http://127.0.0.1:8100/v1",
        )
        assert "http://127.0.0.1:8100/v1" in str(error)

    def test_it_carries_the_classified_category(self):
        error = BackendUnreachableError("openai_compatible", "m", "connection refused")
        assert error.error_context["category"] == "unreachable"


class TestTheDefault:
    """``raise_on_error`` out of the box."""

    def test_a_failed_run_raises_by_default(self):
        assert AgentConfig(model="x", require_model=False).raise_on_error is True

    def test_the_old_behaviour_is_still_available(self):
        config = AgentConfig(model="x", require_model=False, raise_on_error=False)
        assert config.raise_on_error is False


class TestThroughARealRun:
    """What a run against nothing actually does."""

    def _agent(self, **overrides):
        from effgen import Agent
        from effgen.models import load_model

        model = load_model(
            "Qwen/Qwen2.5-7B-Instruct",
            base_url="http://127.0.0.1:1/v1",
            max_retries=0,
            timeout=5,
        )
        return Agent(AgentConfig(model=model, max_iterations=2, **overrides))

    def test_a_dead_endpoint_raises_rather_than_returning_a_string(self):
        agent = self._agent()
        try:
            with pytest.raises(Exception) as caught:
                agent.run("What is 6*7?")
        finally:
            agent.close()
        # The point is that something reached the caller at all — the old
        # behaviour returned "Maximum iterations reached without final answer."
        assert "42" not in str(caught.value)

    def test_it_raises_even_when_the_caller_asked_not_to(self):
        # A task that ran and failed is a result; a backend that never answered
        # is not, so opting out of raising does not apply to it.
        agent = self._agent(raise_on_error=False)
        try:
            with pytest.raises(BackendUnreachableError):
                agent.run("What is 6*7?")
        finally:
            agent.close()
