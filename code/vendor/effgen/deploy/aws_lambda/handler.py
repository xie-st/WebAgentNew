"""effGen AWS Lambda handler — Mangum adapter wrapping the FastAPI app.

Cold-start warm path
--------------------
The module-level code runs exactly once per Lambda container lifetime.
We pre-initialise the ProviderRegistry and load environment variables from
``.env`` so that the first real request pays no per-call setup cost.

Environment variables
---------------------
All of the standard effGen server variables are supported:

  EFFGEN_DEV_MODE          — "1" disables JWT auth (dev/test only)
  EFFGEN_OIDC_ISSUER       — OIDC issuer URL
  EFFGEN_OIDC_CLIENT_ID    — JWT audience claim
  EFFGEN_OIDC_JWKS_URI     — JWKS endpoint (auto-discovered if omitted)
  EFFGEN_RBAC_POLICY_FILE  — path to JSON role policy file
  EFFGEN_AUDIT_DIR         — override audit log directory
  EFFGEN_TIMEOUT_SECONDS   — per-invocation timeout budget (default: 29)

The per-invocation budget is enforced: each request is run with a timeout of
``min(EFFGEN_TIMEOUT_SECONDS, remaining Lambda time - 0.5s)`` so the handler
returns a clean ``504`` rather than letting the Lambda runtime hard-kill the
container (which would surface to the client as an opaque 502 with no body).

AWS SecretsManager / SSM Parameter Store
-----------------------------------------
The SAM template wires provider API keys via SecretsManager references so
that secrets are injected as plain environment variables at container start —
this handler needs no special boto3 calls for key retrieval.

Usage (local smoke test)
------------------------
::

    import deploy.aws_lambda.handler as h
    import json, pathlib

    event = json.loads(
        pathlib.Path("tests/deploy/fixtures/apigw-http-event.json").read_text()
    )

    class FakeContext:
        function_name = "effgen"
        aws_request_id = "test-001"

        def get_remaining_time_in_millis(self):
            return 29000

    print(h.handler(event, FakeContext()))

A ready-to-run version of this smoke test lives in
``deploy/aws_lambda/_smoke_runner.py``.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cold-start: load .env (best-effort — Lambda envs won't have a .env file,
# but local smoke tests and SAM local invoke do benefit from this).
# ---------------------------------------------------------------------------

_cold_start_ts = time.monotonic()

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    # Try the project root first (local dev / SAM local invoke).
    for _dotenv_path in (
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.expanduser("~/.effgen/.env"),
    ):
        if os.path.exists(_dotenv_path):
            load_dotenv(_dotenv_path, override=False)
            logger.debug("Loaded .env from %s", _dotenv_path)
            break
except ImportError:
    pass  # python-dotenv not available in minimal Lambda layer — env vars set directly


def _timeout_seconds() -> int:
    """Per-invocation budget (seconds), read from EFFGEN_TIMEOUT_SECONDS."""
    try:
        return max(1, int(os.getenv("EFFGEN_TIMEOUT_SECONDS", "29")))
    except ValueError:
        return 29


# ---------------------------------------------------------------------------
# Cold-start: pre-initialise the model registry so first-request latency is
# dominated by the actual model call, not registry setup.
# ---------------------------------------------------------------------------


def _preload_registry() -> None:
    """Pre-load the ProviderRegistry during cold start.

    This is a best-effort warm-up: if any provider import fails (e.g. the
    optional ML dep is absent in the Lambda layer), we log a warning and
    continue — the registry will be populated on demand at request time.
    """
    try:
        from effgen.models.registry import ProviderRegistry

        # Trigger lazy provider registration by listing providers.
        providers = ProviderRegistry.list_providers()
        logger.info(
            "Cold-start registry preload: %d provider(s) registered in %.0fms",
            len(providers),
            (time.monotonic() - _cold_start_ts) * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Registry preload skipped: %s", exc)


_preload_registry()

# ---------------------------------------------------------------------------
# Cold-start: build the FastAPI application.
# The Mangum adapter is re-used across warm invocations.
# ---------------------------------------------------------------------------


def _build_adapter() -> Any:
    """Build and return the Mangum-wrapped FastAPI ASGI adapter.

    Called once at module load time so the same ASGI application object is
    reused across warm Lambda invocations — avoiding per-request FastAPI app
    construction overhead.
    """
    try:
        from mangum import Mangum  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "mangum is required for the Lambda handler: "
            "pip install 'effgen[lambda]' or pip install mangum"
        ) from exc

    from effgen.server.app import create_app

    app = create_app()

    # lifespan="off" — Lambda containers have no persistent event loop between
    # invocations; disabling lifespan avoids a stray startup/shutdown cycle on
    # every cold start and is the recommended setting for serverless deployments.
    #
    # api_gateway_base_path="/" — route all API Gateway paths directly.
    adapter = Mangum(
        app,
        lifespan="off",
        api_gateway_base_path="/",
    )

    _elapsed_ms = (time.monotonic() - _cold_start_ts) * 1000
    logger.info(
        "effGen Lambda adapter ready (timeout=%ds, cold-start=%.0fms)",
        _timeout_seconds(),
        _elapsed_ms,
    )

    return adapter


def _event_path(event: Any) -> str:
    """Best-effort extraction of the request path for logging."""
    if not isinstance(event, dict):
        return "<unknown>"
    return (
        event.get("rawPath")
        or event.get("path")
        or event.get("requestContext", {}).get("http", {}).get("path")
        or "<unknown>"
    )


def _error_body(status: int, message: str, code: str) -> str:
    """Serialize an error the same way the app's own routes do.

    The handler answers with the ``{"error": {message, type, param, code}}``
    envelope every effGen endpoint uses, so a client sees one error shape
    whether the failure came from a route or from the invocation wrapper. Falls
    back to the literal shape if the app package cannot be imported (the
    cold-start failure path).
    """
    import json

    try:
        from effgen.api.openai_compat import error_envelope

        payload = error_envelope(status, message, code=code, redact=False)
    except Exception:  # noqa: BLE001 - the failure path must never depend on the import
        err_type = "timeout" if status == 504 else "server_error"
        payload = {
            "error": {"message": message, "type": err_type, "param": None, "code": code}
        }
    return json.dumps(payload)


def _timeout_response(budget_s: float) -> dict:
    """API Gateway response returned when the invocation budget is exceeded."""
    return {
        "statusCode": 504,
        "headers": {"content-type": "application/json"},
        "body": _error_body(
            504,
            "Request exceeded the per-invocation timeout budget of "
            f"{budget_s:.0f}s",
            "invocation_timeout",
        ),
        "isBase64Encoded": False,
    }


def _make_handler(adapter: Any) -> Any:
    """Wrap the Mangum adapter with per-invocation timeout enforcement.

    The budget is ``min(EFFGEN_TIMEOUT_SECONDS, remaining Lambda time - 0.5s)``.
    Mangum is a synchronous callable that drives its own event loop, so we run
    it on a reusable worker thread and join with a timeout. This lets the
    handler return a clean 504 instead of letting the Lambda runtime hard-kill
    the container — which would otherwise surface to the client as an opaque
    502 with no body.
    """
    import asyncio
    import concurrent.futures

    def _initialise_worker() -> None:
        # Mangum's HTTP protocol calls ``asyncio.get_event_loop()``, which only
        # succeeds on a thread that already has an event loop. The worker thread
        # is not the main thread, so install one explicitly (once per thread).
        asyncio.set_event_loop(asyncio.new_event_loop())

    # A single reusable worker thread keeps warm-path overhead negligible.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="effgen-lambda",
        initializer=_initialise_worker,
    )

    def handler(event: Any, context: Any) -> dict:
        budget = float(_timeout_seconds())
        # Never exceed the remaining Lambda time; leave 0.5s headroom so we can
        # actually return the 504 before the runtime kills us.
        get_remaining = getattr(context, "get_remaining_time_in_millis", None)
        if callable(get_remaining):
            try:
                remaining_ms = get_remaining()
                budget = min(budget, max(0.5, remaining_ms / 1000.0 - 0.5))
            except Exception:  # noqa: BLE001
                pass

        future = executor.submit(adapter, event, context)
        try:
            return future.result(timeout=budget)
        except concurrent.futures.TimeoutError:
            logger.error(
                "Invocation exceeded %.1fs budget (path=%s) — returning 504",
                budget,
                _event_path(event),
            )
            # The orphaned future keeps running in the background until the
            # container is recycled; acceptable for a hard-timeout path.
            return _timeout_response(budget)

    return handler


# Module-level handler — Lambda runtime calls handler(event, context).
try:
    _adapter = _build_adapter()
    handler = _make_handler(_adapter)
except Exception as _init_exc:  # noqa: BLE001
    # If the app fails to initialise during cold start (e.g. missing dep),
    # return a 500 for every invocation rather than crashing the container.
    logger.exception("Failed to build Lambda handler: %s", _init_exc)

    def handler(event: Any, context: Any) -> dict:  # type: ignore[misc]
        """Fallback handler returned when cold-start initialisation failed."""
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": _error_body(
                500,
                "Lambda handler failed to initialise — check logs",
                "initialisation_failed",
            ),
            "isBase64Encoded": False,
        }
