# effGen API Server — Role-Based Access Control (RBAC)

effGen enforces RBAC at the API layer.  Each authenticated request carries a
list of role names (from the JWT `roles` or `scope` claim), and the server
resolves an **effective policy** by taking the union of all granted roles.

---

## Built-in roles

| Role | Allowed tools | Allowed models | Max cost/day |
|---|---|---|---|
| `admin` | all | all | unlimited |
| `researcher` | all | all | $50 |
| `viewer` | all | all | $5 |
| `reader` | **none** (read-only) | all | $1 |

> **Note:** `allowed_tools: []` and `allowed_models: []` mean "all" (no
> restriction).  Restrictions are expressed by listing the *permitted* items.
>
> To express "no tools at all" (a read-only role), set `deny_tools: true`.
> An empty `allowed_tools` would otherwise mean *all* tools — `deny_tools`
> overrides that. The built-in `reader` role uses this.

---

## Effective policy (union semantics)

When a principal has multiple roles, the **most-permissive role wins**:

- **Tools**: a tool is allowed if *any* granted role permits it. A role with
  `deny_tools: true` permits no tools; a principal whose only roles are
  read-only (e.g. `reader`) is therefore permitted no tools at all. Adding a
  tool-granting role (e.g. `researcher`) lifts the restriction.
- **Models**: a model is allowed if any role's `allowed_models` permits it
  (empty = all).
- **Cost cap**: the *maximum* across all roles.  A cap of `0.0` (unlimited)
  on any role makes the effective cap unlimited. See
  [audit.md](audit.md) and the cost-cap section below.

A principal with **no recognized roles** falls back to a read-only default
(no tools, models allowed) rather than failing open.

---

## Cost caps (daily budget)

Each role carries `max_cost_per_day` (USD). Spend is tracked per principal for
the current UTC day. The model-invoking endpoints
(`/v1/chat/completions`, `/v1/completions`) charge a small per-call estimate
(`EFFGEN_PER_CALL_COST_USD`, default `$0.01`) against the budget. Once the
accrued spend meets the cap, further calls are rejected with **HTTP 429** and
a `BudgetExceeded` detail. A cap of `0.0` means unlimited.

```bash
# Disable budget persistence (in-memory only)
export EFFGEN_BUDGET_PERSIST=0
# Override the budget snapshot directory
export EFFGEN_BUDGET_DIR=/var/lib/effgen/budget
```

---

## Custom policy file

Override built-in roles by pointing to a JSON file:

```bash
export EFFGEN_RBAC_POLICY_FILE=/etc/effgen/policy.json
```

Schema (`policy.json`):

```json
[
  {
    "name": "ops",
    "allowed_tools": ["deploy_tool", "rollback_tool"],
    "allowed_models": [],
    "max_cost_per_day": 100.0
  },
  {
    "name": "readonly",
    "allowed_tools": [],
    "allowed_models": ["gpt-oss-120b"],
    "max_cost_per_day": 1.0,
    "deny_tools": true
  }
]
```

---

## Python API

```python
from effgen.server.rbac import resolve_policy, Role, EffectivePolicy

# Resolve effective policy from JWT roles
policy: EffectivePolicy = resolve_policy(["researcher", "ops"])

# Check access
policy.check_tool("web_search")   # raises PolicyDenied if not allowed
policy.check_model("gpt-4")       # raises PolicyDenied if not allowed

if policy.allows_tool("code_exec"):
    ...
```

### Defining custom roles in code

```python
from effgen.server.rbac import Role, reset_registry

custom_roles = [
    Role("developer", allowed_tools=frozenset({"code_exec", "git"})),
    Role("analyst",   allowed_models=frozenset({"gpt-oss-120b"}), max_cost_per_day=20.0),
]
reset_registry(custom_roles)
```

---

## FastAPI dependency

```python
from fastapi import Depends
from effgen.server.rbac import get_rbac_dependency, EffectivePolicy
from typing import Annotated

PolicyDep = Annotated[EffectivePolicy, Depends(get_rbac_dependency())]

@app.post("/run-tool")
async def run_tool(policy: PolicyDep, tool: str):
    policy.check_tool(tool)   # 403 if denied
    ...
```

---

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /rbac/policy` | Effective policy for the current principal |
| `GET /rbac/roles` | All defined roles and their constraints |

---

## JWT roles claim

effGen reads the `roles` claim (list) or `scope` claim (space-separated
string) from the JWT.  Configure your IdP to include the role names.

**Auth0 example** (rules / actions):

```js
exports.onExecutePostLogin = async (event, api) => {
  api.accessToken.setCustomClaim("roles", event.authorization.roles);
};
```

**Keycloak**: add a *Token Claim* mapper of type *User Realm Role* with
claim name `roles` on the client scope.
