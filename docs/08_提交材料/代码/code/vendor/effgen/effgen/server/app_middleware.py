"""Request middleware for the effGen API server.

:class:`RBACBudgetMiddleware` enforces role-based tool/model access and a
per-principal daily cost cap on the model-invoking ``/v1`` endpoints;
:class:`MaxBodySizeMiddleware` bounds request bodies on body-accepting routes
the former does not cover. Both are pure-ASGI middlewares whose buffered-body
replay keeps streaming responses working behind ``BaseHTTPMiddleware`` layers.
"""
from __future__ import annotations

import os
from typing import Any

from effgen.server.app_runner import _tool_name


class RBACBudgetMiddleware:
    """Pure-ASGI middleware enforcing RBAC tool/model access + daily cost cap.

    Applies only to model-invoking endpoints (``/v1/chat/completions`` and
    ``/v1/completions``). It runs *after* :class:`AuthMiddleware` (so
    ``scope["state"]["user"]`` is set) and *just outside* the route, so its
    request-body replay survives the production ``BaseHTTPMiddleware`` layers.

    Rejections:
      * 403 ``role X does not permit tool Y`` — disallowed tool,
      * 403 ``role X does not permit model Y`` — disallowed model,
      * 429 ``BudgetExceeded`` — daily cost cap already met.

    Budget handling is reserve-then-reconcile: a per-call estimate
    (``EFFGEN_PER_CALL_COST_USD``, default ``0.01``) is *reserved* before the
    route runs (rejecting an over-cap principal with 429) and committed only if
    the call succeeds. Failed calls (HTTP >= 400 or an exception) release the
    reservation and are **not** charged.

    Request bodies are bounded by ``EFFGEN_MAX_BODY_BYTES`` (default 10 MiB)
    before buffering; oversized bodies are rejected with 413. Bodies are read
    only for the enforced ``/v1`` routes — all other paths pass straight
    through without touching the body.
    """

    _ENFORCED_PATHS = ("/v1/chat/completions", "/v1/completions")

    def __init__(self, app: Any) -> None:
        self.app = app
        self.per_call_cost = float(os.getenv("EFFGEN_PER_CALL_COST_USD", "0.01"))
        self.max_body_bytes = int(os.getenv("EFFGEN_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Skip body reads entirely for routes that don't need RBAC/budget.
        if scope.get("type") != "http" or scope.get("path") not in self._ENFORCED_PATHS:
            await self.app(scope, receive, send)
            return

        from effgen.server import budget as _budget
        from effgen.server.rbac import PolicyDenied, resolve_policy

        user = scope.get("state", {}).get("user")
        principal = getattr(user, "sub", "anonymous") if user else "anonymous"
        roles: list[str] = list(getattr(user, "roles", []) if user else [])
        try:
            policy = resolve_policy(roles)
        except PolicyDenied as exc:
            await _reject_json(send, exc.status_code, str(exc))
            return
        primary_role = roles[0] if roles else (policy.roles[0].name if policy.roles else "anonymous")

        rejected, raw = await _enforce_max_body_size(scope, receive, send, self.max_body_bytes)
        if rejected:
            return

        # Replay the buffered body to the route once. A StreamingResponse runs
        # a disconnect-listener concurrently with the body generator, looping on
        # receive(); if we kept handing it the request body it would spin
        # forever and starve the SSE generator (the /v1 streaming hang). After
        # the single replay we *park* — exactly as a real ASGI server's
        # receive() blocks until the client actually sends something — so the
        # disconnect-listener waits quietly and is cancelled when the response
        # finishes, instead of busy-looping or being told the client vanished.
        import asyncio as _asyncio

        _replay_state = {"sent": False}
        _parked = _asyncio.Event()

        async def _replay() -> dict[str, Any]:
            if not _replay_state["sent"]:
                _replay_state["sent"] = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await _parked.wait()  # cancelled when the route's response completes
            return {"type": "http.disconnect"}

        body: Any = {}
        if raw:
            try:
                import json as _json

                body = _json.loads(raw)
            except Exception:  # noqa: BLE001
                body = {}
        model = body.get("model") if isinstance(body, dict) else None
        tools = body.get("tools") if isinstance(body, dict) else None

        if model and not policy.allows_model(str(model)):
            await _reject_json(
                send, 403, f"role {primary_role} does not permit model {model}"
            )
            return

        for tool in tools or []:
            tname = _tool_name(tool)
            if not policy.allows_tool(tname):
                await _reject_json(
                    send, 403, f"role {primary_role} does not permit tool {tname}"
                )
                return

        # Reserve budget before the call; commit only on success.
        try:
            token = _budget.reserve(
                principal, self.per_call_cost, cap=policy.max_cost_per_day
            )
        except _budget.BudgetExceeded as exc:
            await _reject_json(send, exc.status_code, str(exc))
            return

        status_holder: dict[str, int] = {"status": 500}

        async def _send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 200))
            await send(message)

        committed = False
        try:
            await self.app(scope, _replay, _send_wrapper)
            if status_holder["status"] < 400:
                _budget.reconcile(principal, token)  # charge the reserved estimate
            else:
                _budget.release(principal, token)  # failed call → no charge
            committed = True
        finally:
            if not committed:
                # The route raised before we could settle the reservation.
                _budget.release(principal, token)


async def _reject_json(send: Any, status: int, detail: str) -> None:
    """Emit a JSON error response from an ASGI middleware.

    Uses the shared OpenAI error envelope so RBAC/budget rejections (403/413/429)
    carry the same ``{"error": {message, type, param, code}}`` shape as model
    errors, letting a client branch on ``err.type``/``err.code`` uniformly.
    """
    import json as _json

    from effgen.api.openai_compat import error_envelope

    payload = _json.dumps(error_envelope(status, detail)).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": payload})


async def _enforce_max_body_size(
    scope: Any, receive: Any, send: Any, max_body_bytes: int
) -> tuple[bool, bytes]:
    """Reject an oversized request body; otherwise buffer and return it.

    Checks the declared ``Content-Length`` first (cheap, no buffering), then
    streams the body while counting bytes so an unbounded/chunked body without
    a declared length is still capped. Returns ``(rejected, raw_body)`` — when
    ``rejected`` is ``True`` a 413 has already been sent and the caller must
    not read further from ``receive`` or write to ``send``.
    """
    headers = dict(scope.get("headers", []))
    declared = headers.get(b"content-length")
    if declared is not None:
        try:
            if int(declared) > max_body_bytes:
                await _reject_json(send, 413, "Request body too large")
                return True, b""
        except ValueError:  # non-integer Content-Length; the streamed check still applies
            pass

    chunks: list[bytes] = []
    total = 0
    more = True
    while more:
        msg = await receive()
        chunk = msg.get("body", b"")
        total += len(chunk)
        if total > max_body_bytes:
            await _reject_json(send, 413, "Request body too large")
            return True, b""
        chunks.append(chunk)
        more = msg.get("more_body", False)
    return False, b"".join(chunks)


class MaxBodySizeMiddleware:
    """Pure-ASGI middleware enforcing ``EFFGEN_MAX_BODY_BYTES`` on body routes.

    Covers routes that accept a body but are not already covered by
    :class:`RBACBudgetMiddleware` (which enforces the same cap for
    ``/v1/chat/completions`` and ``/v1/completions`` as part of its
    RBAC/budget replay). Add a path here
    whenever a new body-accepting route is mounted that doesn't need RBAC or
    budget enforcement, so the cap is never an allowlist of just the model
    endpoints.

    Buffers the body (bounded) and replays it once to the route, mirroring
    ``RBACBudgetMiddleware``'s SSE-safe replay-then-park pattern so a
    streaming response downstream isn't starved by a spinning ``receive()``.
    """

    _ENFORCED_PATHS = ("/v1/embeddings",)

    def __init__(self, app: Any) -> None:
        self.app = app
        self.max_body_bytes = int(os.getenv("EFFGEN_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") not in self._ENFORCED_PATHS:
            await self.app(scope, receive, send)
            return

        rejected, raw = await _enforce_max_body_size(scope, receive, send, self.max_body_bytes)
        if rejected:
            return

        import asyncio as _asyncio

        _replay_state = {"sent": False}
        _parked = _asyncio.Event()

        async def _replay() -> dict[str, Any]:
            if not _replay_state["sent"]:
                _replay_state["sent"] = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await _parked.wait()  # cancelled when the route's response completes
            return {"type": "http.disconnect"}

        await self.app(scope, _replay, send)
