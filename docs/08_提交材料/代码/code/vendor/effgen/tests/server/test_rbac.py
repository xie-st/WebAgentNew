"""Tests for effgen.server.rbac — role definitions, policy resolution, enforcement."""
from __future__ import annotations

import pytest

from effgen.server.rbac import (
    EffectivePolicy,
    PolicyDenied,
    Role,
    get_role,
    list_roles,
    reset_registry,
    resolve_policy,
)


@pytest.fixture(autouse=True)
def reset_rbac_registry():
    """Reset the registry before and after each test for isolation."""
    reset_registry(None)  # revert to built-ins
    yield
    reset_registry(None)


# ---------------------------------------------------------------------------
# Role definition tests
# ---------------------------------------------------------------------------


class TestRoleDefinition:
    def test_role_allows_all_tools_when_empty(self):
        r = Role(name="open", allowed_tools=frozenset())
        assert r.allows_tool("any-tool") is True

    def test_role_blocks_unlisted_tool(self):
        r = Role(name="restricted", allowed_tools=frozenset({"web_search"}))
        assert r.allows_tool("web_search") is True
        assert r.allows_tool("code_exec") is False

    def test_role_allows_all_models_when_empty(self):
        r = Role(name="open", allowed_models=frozenset())
        assert r.allows_model("gpt-4") is True

    def test_role_blocks_unlisted_model(self):
        r = Role(name="restricted", allowed_models=frozenset({"llama3.1-8b"}))
        assert r.allows_model("llama3.1-8b") is True
        assert r.allows_model("gpt-4-turbo") is False

    def test_role_from_dict(self):
        data = {
            "name": "custom",
            "allowed_tools": ["web_search", "calculator"],
            "allowed_models": ["llama3.1-8b"],
            "max_cost_per_day": 10.0,
        }
        r = Role.from_dict(data)
        assert r.name == "custom"
        assert "web_search" in r.allowed_tools
        assert r.max_cost_per_day == 10.0

    def test_role_to_dict_round_trip(self):
        r = Role(
            name="t",
            allowed_tools=frozenset({"a", "b"}),
            allowed_models=frozenset({"m1"}),
            max_cost_per_day=5.0,
        )
        d = r.to_dict()
        r2 = Role.from_dict(d)
        assert r2.name == r.name
        assert r2.allowed_tools == r.allowed_tools
        assert r2.max_cost_per_day == r.max_cost_per_day


# ---------------------------------------------------------------------------
# Built-in roles
# ---------------------------------------------------------------------------


class TestBuiltinRoles:
    def test_admin_role_exists(self):
        role = get_role("admin")
        assert role is not None
        assert role.name == "admin"
        assert role.max_cost_per_day == 0.0  # unlimited
        assert role.allows_tool("any_tool") is True
        assert role.allows_model("any_model") is True

    def test_researcher_role_exists(self):
        role = get_role("researcher")
        assert role is not None
        assert role.max_cost_per_day == 50.0

    def test_viewer_role_exists(self):
        role = get_role("viewer")
        assert role is not None
        assert role.max_cost_per_day == 5.0

    def test_list_roles_includes_builtins(self):
        roles = list_roles()
        names = {r.name for r in roles}
        assert {"admin", "researcher", "viewer", "reader"} <= names

    def test_reader_role_denies_all_tools(self):
        """The built-in reader role permits NO tools (deny_tools)."""
        role = get_role("reader")
        assert role is not None
        assert role.deny_tools is True
        assert role.allows_tool("web_search") is False
        assert role.allows_tool("code_exec") is False
        # ...but still permits models (read-only chat without tools).
        assert role.allows_model("llama3.1-8b") is True
        assert role.max_cost_per_day == 1.0


class TestDenyTools:
    def test_deny_tools_overrides_empty_allowed_set(self):
        """deny_tools=True wins even when allowed_tools is empty (= all)."""
        r = Role(name="ro", allowed_tools=frozenset(), deny_tools=True)
        assert r.allows_tool("anything") is False

    def test_deny_tools_round_trips_through_dict(self):
        r = Role(name="ro", deny_tools=True)
        assert Role.from_dict(r.to_dict()).deny_tools is True


# ---------------------------------------------------------------------------
# Policy resolution — single role
# ---------------------------------------------------------------------------


class TestPolicyResolutionSingle:
    def test_admin_policy_allows_everything(self):
        policy = resolve_policy(["admin"])
        assert policy.allows_tool("any_tool") is True
        assert policy.allows_model("any_model") is True
        assert policy.max_cost_per_day == 0.0

    def test_researcher_policy(self):
        policy = resolve_policy(["researcher"])
        assert policy.allows_tool("any_tool") is True  # all tools allowed
        assert policy.max_cost_per_day == 50.0

    def test_unknown_role_strict_by_default(self):
        """Unknown roles are rejected in strict mode (the default)."""
        from effgen.server.rbac import PolicyDenied

        with pytest.raises(PolicyDenied):
            resolve_policy(["unknown-role"])

    def test_unknown_role_lenient_falls_back(self):
        """In lenient mode, unknown roles are skipped → read-only fallback."""
        policy = resolve_policy(["unknown-role"], strict=False)
        assert isinstance(policy, EffectivePolicy)
        # No recognized roles → read-only default (no tools).
        assert policy.allows_tool("web_search") is False

    def test_empty_roles_list_falls_back_to_viewer(self):
        policy = resolve_policy([])
        assert isinstance(policy, EffectivePolicy)


# ---------------------------------------------------------------------------
# Policy resolution — multiple roles (union)
# ---------------------------------------------------------------------------


class TestPolicyResolutionMultiple:
    def test_union_of_allowed_tools(self):
        """With two restricted roles, the union of tools is granted."""
        reset_registry([
            Role("r1", allowed_tools=frozenset({"web_search"}), allowed_models=frozenset()),
            Role("r2", allowed_tools=frozenset({"calculator"}), allowed_models=frozenset()),
        ])
        policy = resolve_policy(["r1", "r2"])
        assert policy.allows_tool("web_search") is True
        assert policy.allows_tool("calculator") is True
        assert policy.allows_tool("code_exec") is False  # neither role grants it

    def test_admin_plus_restricted_gives_all(self):
        """Admin role (empty = all) wins over a restricted role."""
        policy = resolve_policy(["viewer", "admin"])
        assert policy.allows_tool("anything") is True

    def test_max_cost_per_day_takes_maximum(self):
        """Effective cap is the maximum across roles."""
        reset_registry([
            Role("r1", max_cost_per_day=10.0),
            Role("r2", max_cost_per_day=25.0),
        ])
        policy = resolve_policy(["r1", "r2"])
        assert policy.max_cost_per_day == 25.0

    def test_zero_cost_cap_wins(self):
        """A zero cap (unlimited) on any role makes the effective cap unlimited."""
        reset_registry([
            Role("r1", max_cost_per_day=10.0),
            Role("r2", max_cost_per_day=0.0),  # unlimited
        ])
        policy = resolve_policy(["r1", "r2"])
        assert policy.max_cost_per_day == 0.0  # max(10, 0) = 10 but 0 means unlimited


# ---------------------------------------------------------------------------
# check_tool / check_model raise PolicyDenied
# ---------------------------------------------------------------------------


class TestPolicyChecks:
    def test_check_tool_raises_for_blocked_tool(self):
        reset_registry([
            Role("limited", allowed_tools=frozenset({"web_search"})),
        ])
        policy = resolve_policy(["limited"])
        with pytest.raises(PolicyDenied, match="code_exec"):
            policy.check_tool("code_exec")

    def test_check_tool_passes_for_allowed_tool(self):
        reset_registry([
            Role("limited", allowed_tools=frozenset({"web_search"})),
        ])
        policy = resolve_policy(["limited"])
        policy.check_tool("web_search")  # should not raise

    def test_check_model_raises_for_blocked_model(self):
        reset_registry([
            Role("limited", allowed_models=frozenset({"llama3.1-8b"})),
        ])
        policy = resolve_policy(["limited"])
        with pytest.raises(PolicyDenied, match="gpt-4"):
            policy.check_model("gpt-4")

    def test_admin_check_never_raises(self):
        policy = resolve_policy(["admin"])
        policy.check_tool("code_exec")
        policy.check_model("gpt-4-turbo")

    def test_reader_policy_denies_every_tool(self):
        """A reader-only principal is permitted no tools at all."""
        policy = resolve_policy(["reader"])
        assert policy.allows_tool("web_search") is False
        with pytest.raises(PolicyDenied, match="role reader does not permit tool web_search"):
            policy.check_tool("web_search")

    def test_reader_plus_researcher_grants_tools(self):
        """Most-permissive-wins: adding a tool-granting role lifts the deny."""
        policy = resolve_policy(["reader", "researcher"])
        assert policy.allows_tool("web_search") is True

    def test_unknown_role_defaults_to_read_only(self):
        """In lenient mode, unknown roles fall back to a read-only policy."""
        policy = resolve_policy(["does-not-exist"], strict=False)
        assert policy.allows_tool("web_search") is False


# ---------------------------------------------------------------------------
# Custom registry loaded from JSON (monkeypatch env)
# ---------------------------------------------------------------------------


class TestCustomPolicyFile:
    def test_policy_file_loaded(self, tmp_path):
        import json
        import os

        policy = [
            {
                "name": "ops",
                "allowed_tools": ["deploy", "rollback"],
                "allowed_models": [],
                "max_cost_per_day": 100.0,
            }
        ]
        policy_file = tmp_path / "policy.json"
        policy_file.write_text(json.dumps(policy))

        reset_registry(None)
        orig = os.environ.get("EFFGEN_RBAC_POLICY_FILE")
        os.environ["EFFGEN_RBAC_POLICY_FILE"] = str(policy_file)
        try:
            from effgen.server import rbac as _rbac

            _rbac._ROLE_REGISTRY = None  # force reload
            role = get_role("ops")
            assert role is not None
            assert "deploy" in role.allowed_tools
            assert role.max_cost_per_day == 100.0
        finally:
            if orig is None:
                os.environ.pop("EFFGEN_RBAC_POLICY_FILE", None)
            else:
                os.environ["EFFGEN_RBAC_POLICY_FILE"] = orig
            _rbac._ROLE_REGISTRY = None
