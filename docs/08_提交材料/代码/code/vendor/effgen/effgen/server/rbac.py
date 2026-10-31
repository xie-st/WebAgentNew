"""Role-Based Access Control (RBAC) for the effGen API server.

Roles are defined by name and carry:
  - allowed_tools:    set of tool names (empty = all allowed, unless deny_tools)
  - allowed_models:   set of model ids   (empty = all allowed)
  - max_cost_per_day: USD cap (0.0 = unlimited)
  - deny_tools:       when True the role permits NO tools (overrides the
                      "empty == all" rule). Used by read-only roles.

JWT claim "roles" (space-separated string or list) controls which roles apply.
Multiple roles are additive (union of allowed sets, most-permissive wins).

Environment configuration:

  EFFGEN_RBAC_POLICY_FILE — path to a JSON file with role definitions
                             (see docs/server/rbac.md for schema)

Built-in roles shipped with effGen:

  admin        — all tools, all models, no cost cap
  researcher   — all tools, all models, $50/day cap
  limited_user — all tools, all models, $5/day cap (low-budget tool user)
  viewer       — NO tools (deny_tools), all models, $5/day cap (read-only)
  reader       — NO tools (deny_tools), all models, $1/day cap (read-only)

``viewer`` is read-only despite the cost cap: it cannot execute tools (a
"viewer" should not be able to spend budget by invoking tools). Use
``limited_user`` for a low-budget role that *can* run tools.

These are overridden if a policy file is present.

Unknown roles: by default (outside dev mode) an unrecognized role is rejected
(:class:`PolicyDenied`) so identity-provider mapping mistakes fail loudly. Set
``EFFGEN_RBAC_STRICT_ROLES=0`` to fall back to the lenient "skip unknown roles"
behavior; dev mode is lenient unless ``EFFGEN_RBAC_STRICT_ROLES=1``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from effgen.errors import quote_for_message, with_next_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """Immutable role descriptor."""

    name: str
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    allowed_models: frozenset[str] = field(default_factory=frozenset)
    max_cost_per_day: float = 0.0  # 0 == unlimited
    deny_tools: bool = False  # when True, permits NO tools (read-only roles)

    def allows_tool(self, tool: str) -> bool:
        """Return True if this role permits the given tool."""
        if self.deny_tools:
            return False
        return not self.allowed_tools or tool in self.allowed_tools

    def allows_model(self, model: str) -> bool:
        """Return True if this role permits the given model."""
        return not self.allowed_models or model in self.allowed_models

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Role":
        """Build a role from its dict form (missing fields get defaults)."""
        return cls(
            name=data["name"],
            allowed_tools=frozenset(data.get("allowed_tools", [])),
            allowed_models=frozenset(data.get("allowed_models", [])),
            max_cost_per_day=float(data.get("max_cost_per_day", 0.0)),
            deny_tools=bool(data.get("deny_tools", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "name": self.name,
            "allowed_tools": sorted(self.allowed_tools),
            "allowed_models": sorted(self.allowed_models),
            "max_cost_per_day": self.max_cost_per_day,
            "deny_tools": self.deny_tools,
        }


# ---------------------------------------------------------------------------
# Built-in role registry
# ---------------------------------------------------------------------------

_BUILTIN_ROLES: dict[str, Role] = {
    "admin": Role(
        name="admin",
        allowed_tools=frozenset(),  # empty = all
        allowed_models=frozenset(),  # empty = all
        max_cost_per_day=0.0,
    ),
    "researcher": Role(
        name="researcher",
        allowed_tools=frozenset(),
        allowed_models=frozenset(),
        max_cost_per_day=50.0,
    ),
    "limited_user": Role(
        name="limited_user",
        allowed_tools=frozenset(),  # empty = all tools
        allowed_models=frozenset(),
        max_cost_per_day=5.0,
    ),
    "viewer": Role(
        name="viewer",
        allowed_tools=frozenset(),
        allowed_models=frozenset(),
        max_cost_per_day=5.0,
        deny_tools=True,  # read-only: a "viewer" must not execute tools
    ),
    "reader": Role(
        name="reader",
        allowed_tools=frozenset(),
        allowed_models=frozenset(),
        max_cost_per_day=1.0,
        deny_tools=True,  # read-only: no tool execution permitted
    ),
}

_ROLE_REGISTRY: dict[str, Role] | None = None


def _load_registry() -> dict[str, Role]:
    global _ROLE_REGISTRY
    if _ROLE_REGISTRY is not None:
        return _ROLE_REGISTRY

    policy_file = os.getenv("EFFGEN_RBAC_POLICY_FILE", "")
    if policy_file:
        path = Path(policy_file).expanduser()
        if path.exists():
            try:
                raw: list[dict[str, Any]] = json.loads(path.read_text())
                _ROLE_REGISTRY = {r["name"]: Role.from_dict(r) for r in raw}
                logger.info("Loaded %d RBAC roles from %s", len(_ROLE_REGISTRY), path)
                return _ROLE_REGISTRY
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load RBAC policy from %s: %s", policy_file, exc)
        else:
            logger.warning("EFFGEN_RBAC_POLICY_FILE=%s not found; using built-in roles", policy_file)

    _ROLE_REGISTRY = dict(_BUILTIN_ROLES)
    return _ROLE_REGISTRY


def get_role(name: str) -> Role | None:
    """Return the :class:`Role` for *name*, or ``None`` if not found."""
    return _load_registry().get(name)


def list_roles() -> list[Role]:
    """Return all registered roles."""
    return list(_load_registry().values())


def reset_registry(roles: list[Role] | None = None) -> None:
    """Reset the registry (primarily for testing)."""
    global _ROLE_REGISTRY
    _ROLE_REGISTRY = {r.name: r for r in roles} if roles else None


# ---------------------------------------------------------------------------
# Policy resolver
# ---------------------------------------------------------------------------


class PolicyDenied(Exception):
    """Raised when an RBAC policy check fails.

    The message names what the policy refused and then how to change the
    outcome, so a caller is not left guessing whether to change the request or
    the role.

    Attributes:
        status_code: The HTTP status this maps to (403 by default).
    """

    _FOLLOW_ON = (
        "Request only what the principal's roles allow, or grant the role the "
        "missing permission in the policy configuration."
    )

    def __init__(self, reason: str, status_code: int = 403) -> None:
        super().__init__(with_next_step(quote_for_message(reason), self._FOLLOW_ON))
        self.status_code = status_code


@dataclass
class EffectivePolicy:
    """Merged policy for a principal with multiple roles."""

    roles: list[Role]
    allowed_tools: frozenset[str]   # union across roles; empty = all
    allowed_models: frozenset[str]  # union across roles; empty = all
    max_cost_per_day: float          # max across roles; 0 = unlimited
    tools_unrestricted: bool = True  # True when some role grants all tools

    def allows_tool(self, tool: str) -> bool:
        """Return True if effective policy permits the given tool.

        Most-permissive-wins: a tool is allowed if any granted role permits
        it. A principal whose only roles are read-only (``deny_tools``)
        permits no tools at all.
        """
        for role in self.roles:
            if role.allows_tool(tool):
                return True
        return False

    def allows_model(self, model: str) -> bool:
        """Return True if effective policy permits the given model."""
        if not self.allowed_models:
            return True
        return model in self.allowed_models

    def check_tool(self, tool: str) -> None:
        """Raise :class:`PolicyDenied` if the tool is not permitted."""
        if not self.allows_tool(tool):
            role_names = [r.name for r in self.roles]
            primary = role_names[0] if role_names else "anonymous"
            raise PolicyDenied(
                f"role {primary} does not permit tool {tool}"
                f" (roles={role_names!r})"
            )

    def check_model(self, model: str) -> None:
        """Raise :class:`PolicyDenied` if the model is not permitted."""
        if not self.allows_model(model):
            raise PolicyDenied(
                f"Model {model!r} is not permitted for roles "
                f"{[r.name for r in self.roles]!r}"
            )


def _strict_roles() -> bool:
    """Return whether unknown roles should be rejected (strict mode).

    Strict by default outside dev mode; lenient in dev mode. Overridable with
    ``EFFGEN_RBAC_STRICT_ROLES`` (``1``/``0``).
    """
    raw = os.getenv("EFFGEN_RBAC_STRICT_ROLES", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    # Unset: strict unless dev mode.
    return os.getenv("EFFGEN_DEV_MODE", "0").strip() != "1"


def resolve_policy(
    role_names: list[str], *, strict: bool | None = None
) -> EffectivePolicy:
    """Build an :class:`EffectivePolicy` from a list of role name strings.

    In strict mode (default outside dev mode) an unrecognized role raises
    :class:`PolicyDenied` so identity-provider mapping mistakes fail loudly.
    In lenient mode unknown roles are logged and skipped.
    """
    if strict is None:
        strict = _strict_roles()
    registry = _load_registry()
    roles: list[Role] = []
    for name in role_names:
        role = registry.get(name)
        if role is None:
            if strict:
                raise PolicyDenied(
                    f"unknown role {name!r}; known roles: "
                    f"{sorted(registry)!r}"
                )
            logger.warning("Unknown role %r — skipping", name)
        else:
            roles.append(role)

    if not roles:
        # No recognized roles → read-only default (deny tool execution).
        logger.info("No recognized roles; applying default read-only policy")
        roles = [_BUILTIN_ROLES["reader"]]

    # A role with empty allowed_tools that is NOT deny_tools grants all tools.
    tools_unrestricted = any(
        (not r.deny_tools and not r.allowed_tools) for r in roles
    )

    # Union allowed sets; if any role has empty set (= all), result is empty = all
    all_tools: frozenset[str] = frozenset()
    has_restricted_tools = any(bool(r.allowed_tools) for r in roles)
    if has_restricted_tools:
        all_tools = frozenset().union(*(r.allowed_tools for r in roles if r.allowed_tools))

    all_models: frozenset[str] = frozenset()
    has_restricted_models = any(bool(r.allowed_models) for r in roles)
    if has_restricted_models:
        all_models = frozenset().union(*(r.allowed_models for r in roles if r.allowed_models))

    # Max cost cap: 0.0 means "unlimited". If any role is unlimited (0),
    # the effective policy is also unlimited. Otherwise, take the maximum
    # across all roles (most permissive wins).
    costs = [r.max_cost_per_day for r in roles]
    if any(c == 0.0 for c in costs):
        max_cost = 0.0  # unlimited
    else:
        max_cost = max(costs, default=0.0)

    return EffectivePolicy(
        roles=roles,
        allowed_tools=all_tools,
        allowed_models=all_models,
        max_cost_per_day=max_cost,
        tools_unrestricted=tools_unrestricted,
    )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_rbac_dependency(
    require_tool: str | None = None,
    require_model: str | None = None,
) -> Any:
    """Return a FastAPI Depends-compatible callable that enforces RBAC.

    Usage::

        from fastapi import Depends
        from effgen.server.rbac import get_rbac_dependency

        PolicyDep = Annotated[EffectivePolicy, Depends(get_rbac_dependency())]

        @router.post("/run-tool")
        async def run_tool(policy: PolicyDep, tool: str):
            policy.check_tool(tool)
            ...
    """
    try:
        from fastapi import HTTPException, Request
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fastapi is required for get_rbac_dependency()") from exc

    async def _rbac_dep(request: Request) -> EffectivePolicy:
        user = getattr(request.state, "user", None)
        roles: list[str] = []
        if user is not None:
            roles = getattr(user, "roles", [])

        try:
            policy = resolve_policy(roles)
        except PolicyDenied as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        if require_tool and not policy.allows_tool(require_tool):
            raise HTTPException(
                status_code=403,
                detail=f"Tool {require_tool!r} is not permitted for your role(s)",
            )
        if require_model and not policy.allows_model(require_model):
            raise HTTPException(
                status_code=403,
                detail=f"Model {require_model!r} is not permitted for your role(s)",
            )

        return policy

    return _rbac_dep
