"""effGen API server — Auth, RBAC, Audit.

Import the server components::

    from effgen.server.auth import verify_jwt, AuthMiddleware, TokenPayload
    from effgen.server.rbac import Role, resolve_policy, EffectivePolicy
    from effgen.server.audit import AuditRecord, write_audit_record, AuditMiddleware
    from effgen.server.budget import charge, check_budget, BudgetExceeded
    from effgen.server.app import create_app
"""
from __future__ import annotations

from effgen.server.app import create_app
from effgen.server.audit import AuditMiddleware, AuditRecord, write_audit_record
from effgen.server.auth import AuthError, AuthMiddleware, TokenPayload, verify_jwt
from effgen.server.budget import BudgetExceeded, charge, check_budget, get_spend
from effgen.server.rbac import (
    EffectivePolicy,
    PolicyDenied,
    Role,
    get_role,
    list_roles,
    resolve_policy,
)

__all__ = [
    "create_app",
    # auth
    "AuthError",
    "AuthMiddleware",
    "TokenPayload",
    "verify_jwt",
    # rbac
    "Role",
    "EffectivePolicy",
    "PolicyDenied",
    "get_role",
    "list_roles",
    "resolve_policy",
    # audit
    "AuditRecord",
    "AuditMiddleware",
    "write_audit_record",
    # budget
    "BudgetExceeded",
    "charge",
    "check_budget",
    "get_spend",
]
