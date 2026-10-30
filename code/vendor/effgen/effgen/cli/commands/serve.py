"""The ``effgen serve`` command: the API server launcher and convenience routes.

``_main`` parses arguments and dispatches; ``CLIInterface.serve_api`` /
``._register_convenience_routes`` delegate here, and the request models are
re-exported from ``_main``.

The server request models and the ``WebSocket`` type are defined at MODULE
level (not inside a function). FastAPI resolves a route's parameter types from
the handler's *module* globals via ``get_type_hints()``; a function-local
class/type is invisible there (acute under ``from __future__ import
annotations``, where every annotation is a string). A function-local body model
made FastAPI treat the body as a query param (422 on a JSON POST); a
function-local ``WebSocket`` annotation made it treat ``ws`` as a required query
param and reject the handshake (1008). Keeping both ``TaskRequest`` and
``WebSocket`` at module scope makes ``/run`` and ``/ws`` work and lets
``/openapi.json`` build. The agent classes and the version string are read from
``effgen.cli._main`` at call time so an override there still takes effect.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING, Any

try:
    from fastapi import WebSocket  # noqa: F401 - module-level for annotation resolution
except Exception:  # pragma: no cover - fastapi present only with [server]
    WebSocket = None  # type: ignore[assignment,misc]

try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import ConfigDict as _PydanticConfigDict

    class TaskRequest(_PydanticBaseModel):
        """Request body for the ``POST /run`` convenience route."""

        model_config = _PydanticConfigDict(extra="ignore")

        task: str
        model: str | None = "Qwen/Qwen2.5-3B-Instruct"
        tools: list[str] | None = None
        preset: str | None = None
        temperature: float | None = 0.7
        max_iterations: int | None = 10
        stream: bool = False

    class TaskResponse(_PydanticBaseModel):
        """Response body for the ``POST /run`` convenience route."""

        model_config = _PydanticConfigDict(extra="ignore")

        output: str
        success: bool
        metadata: dict[str, Any]
except Exception:  # pragma: no cover - pydantic always present with [server]
    _PydanticBaseModel = None  # type: ignore[assignment,misc]
    _PydanticConfigDict = None  # type: ignore[assignment,misc]
    TaskRequest = None  # type: ignore[assignment,misc]
    TaskResponse = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)


def _general_purpose_tool_names(registry: Any, limit: int = 5) -> list:
    """Pick up to *limit* model-agnostic tool names for the convenience routes.

    The ``/run`` and ``/ws`` endpoints attach a small default tool set so a bare
    task can still use tools. Provider-*native* tools (tagged ``*-native``, e.g.
    Anthropic computer-use, OpenAI/Gemini built-ins) are executed server-side by
    a specific provider and raise "incompatible with model" on any other model,
    so they must be excluded from a generic, any-model default set.
    """
    names = []
    for name in registry.list_tools():
        try:
            tags = getattr(registry.get_metadata(name), "tags", None) or []
            if any("native" in str(t).lower() for t in tags):
                continue
        except Exception:  # noqa: BLE001 - if metadata is unavailable, keep the tool
            pass
        names.append(name)
        if len(names) >= limit:
            break
    return names


def serve_api(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """
    Start the effGen API server.

    ``effgen serve`` and the ``effgen.server.app:create_app`` factory now
    converge on **one** secure application: the OpenAI-compatible ``/v1/*``
    endpoints (with SSE streaming), auth, RBAC/budget, audit, metrics, and
    the dashboard all come from :func:`effgen.server.app.create_app`. This
    method layers a few convenience routes (``/run``, ``/tools``, ``/``,
    ``/slo``, ``/ws``) on top.

    Auth posture (fail-closed by default):
      * ``EFFGEN_API_KEY`` set  → static API-key auth (Bearer or X-API-Key).
      * ``EFFGEN_DEV_MODE=1``   → auth disabled (loud warning; dev only).
      * OIDC env vars set       → JWT auth.
      * none of the above       → an **ephemeral** API key is generated and
        printed once, so the server is never unauthenticated by default.
    """
    from effgen.cli import _main

    cli.print_header(f"effGen v{_main.__version__} - API Server")

    try:
        import uvicorn
    except ImportError:
        cli.print_error("FastAPI and uvicorn are required for server mode.")
        cli.print("Install with: pip install 'effgen[server]'")
        return 1

    try:
        import secrets

        host = args.host
        port = args.port
        loopback = host in ("127.0.0.1", "localhost", "::1", "")
        verbose = getattr(args, "verbose", False)

        dev_mode = os.environ.get("EFFGEN_DEV_MODE", "0").strip() == "1"
        api_key = (os.environ.get("EFFGEN_API_KEY", "") or "").strip()
        oidc = bool(
            os.environ.get("EFFGEN_OIDC_ISSUER")
            or os.environ.get("EFFGEN_OIDC_JWKS_URI")
        )

        ephemeral_key = False
        if not api_key and not dev_mode and not oidc:
            # Secure by default: rather than serving unauthenticated, mint a
            # one-off key and print it. The operator copies it into their
            # client. Set EFFGEN_API_KEY (or EFFGEN_DEV_MODE=1) to override.
            api_key = secrets.token_urlsafe(24)
            os.environ["EFFGEN_API_KEY"] = api_key
            ephemeral_key = True

        if not loopback and not api_key and not oidc and dev_mode:
            cli.print_error(
                f"Binding to {host} exposes this server on all interfaces "
                "with auth DISABLED (EFFGEN_DEV_MODE=1). Unset dev mode and "
                "set EFFGEN_API_KEY, or bind to 127.0.0.1 (the default)."
            )

        cors_env = os.environ.get("EFFGEN_CORS_ORIGINS", "").strip()
        cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or None

        from effgen.server.app import create_app

        app = create_app(
            api_key=api_key or None,
            cors_origins=cors_origins,
            dev_mode=dev_mode,
            rate_limit_per_minute=getattr(args, "rate_limit", None),
            trust_proxy=getattr(args, "trust_proxy", None),
        )

        # Discover tools for the /run + /tools convenience routes and stash
        # the CLI instance on the app for those handlers to reach.
        cli.tool_registry.discover_builtin_tools()
        app.state.cli = cli

        cli._register_convenience_routes(app)

        # --- Auth posture banner ---
        if dev_mode:
            cli.print("Auth: DISABLED (EFFGEN_DEV_MODE=1) — do not use in production")
        elif ephemeral_key:
            cli.print_success("Auth: ephemeral API key (set EFFGEN_API_KEY to pin one)")
            cli.print(f"  API key: {api_key}")
            cli.print(f'  Example: curl -H "Authorization: Bearer {api_key}" '
                      f"http://{host}:{port}/v1/models")
        elif api_key:
            cli.print_success("Auth: static API key (EFFGEN_API_KEY)")
        elif oidc:
            cli.print_success("Auth: OIDC / JWT")

        public_dashboard = dev_mode or os.environ.get(
            "EFFGEN_PUBLIC_DASHBOARD", "0"
        ).strip() == "1"
        public_playground = dev_mode or os.environ.get(
            "EFFGEN_PUBLIC_PLAYGROUND", "0"
        ).strip() == "1"

        cli.print(f"Starting server on {host}:{port}")
        cli.print(f"  OpenAI-compatible API : http://{host}:{port}/v1")
        cli.print(f"  Interactive docs      : http://{host}:{port}/docs")
        dashboard_line = f"  Dashboard             : http://{host}:{port}/dashboard"
        if not public_dashboard:
            dashboard_line += "  (data requires an API key; set EFFGEN_PUBLIC_DASHBOARD=1 for local viewing)"
        cli.print(dashboard_line)
        playground_line = f"  Playground            : http://{host}:{port}/playground"
        if not public_playground:
            playground_line += "  (paste an API key, or set EFFGEN_PUBLIC_PLAYGROUND=1 for local viewing)"
        cli.print(playground_line)
        cli.print("  Both pages: Cmd/Ctrl-K opens the command palette, ? lists shortcuts.")
        cli.print()

        # Keep uvicorn's proxy-header handling consistent with the rate
        # limiter's trust decision. uvicorn rewrites scope["client"] from
        # X-Forwarded-For for any peer in ``forwarded_allow_ips`` (default
        # 127.0.0.1), which would let a loopback/same-host caller set the
        # rate-limit client IP even though trust_proxy defaults to off. When
        # the proxy is not trusted, disable that rewriting so the limiter
        # keys on the real socket peer; when it is trusted, effGen reads the
        # header itself and uvicorn's same-host default is left in place.
        from effgen.api.middleware import _resolve_trust_proxy
        uvicorn_kwargs: dict[str, Any] = {}
        if not _resolve_trust_proxy(getattr(args, "trust_proxy", None)):
            uvicorn_kwargs["forwarded_allow_ips"] = []

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info" if verbose else "warning",
            **uvicorn_kwargs,
        )
        return 0

    except Exception as e:
        cli.print_error(f"Error starting server: {e}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        return 1


def register_convenience_routes(cli: "CLIInterface", app: Any) -> None:
    """Attach the legacy convenience routes onto the secure app.

    Adds ``/run``, ``/tools``, ``/``, ``/slo`` and ``/ws`` to the app built by
    ``create_app``.

    Auth for these is provided by the app's ``AuthMiddleware`` (so they
    honor the same static-key / OIDC / dev posture as ``/v1``); they do not
    re-implement their own auth dependency.
    """
    # WebSocket is imported at module level (above) so FastAPI can resolve
    # the ``ws: WebSocket`` annotation under ``from __future__ import
    # annotations``; importing it only here would leave the annotation
    # unresolvable and break the /ws handshake.
    from fastapi import HTTPException, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    from effgen.cli import _main
    from effgen.server.app import _normalize_model_id

    @app.post("/run")
    async def run_task(request: TaskRequest) -> Any:
        """Run a task with an agent (convenience endpoint)."""
        try:
            # Normalize OpenAI-style ``provider/model`` ids to effGen's
            # ``provider:model`` routing, matching the /v1 endpoints — so a
            # ``groq/…`` id loads the provider adapter instead of falling
            # through to the local Transformers path and 500-ing.
            model_id = _normalize_model_id(request.model) if request.model else request.model
            if request.preset:
                from effgen.presets import create_agent as _create_preset_agent

                agent_instance = _create_preset_agent(
                    request.preset,
                    model_id,
                    temperature=request.temperature,
                    max_iterations=request.max_iterations,
                )
            else:
                tools = []
                for name in _general_purpose_tool_names(app.state.cli.tool_registry):
                    try:
                        tools.append(await app.state.cli.tool_registry.get_tool(name))
                    except Exception as tool_err:  # noqa: BLE001
                        logger.debug("Failed to load tool %s: %s", name, tool_err)
                agent_instance = _main.Agent(_main.AgentConfig(
                    name="api-agent",
                    # The route turns a failed run into a JSON error body itself.
                    raise_on_error=False,
                    model=model_id,
                    tools=tools,
                    temperature=request.temperature,
                    max_iterations=request.max_iterations,
                ))

            try:
                response = agent_instance.run(request.task)
            finally:
                agent_instance.close()  # release per-request agent resources
            return JSONResponse(content={
                "output": response.output,
                "success": response.success,
                "metadata": {
                    "mode": response.mode.value if hasattr(response.mode, "value") else str(response.mode),
                    "iterations": response.iterations,
                    "tool_calls": response.tool_calls,
                    "execution_time": response.execution_time,
                },
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("Error running task")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/slo")
    async def slo_endpoint() -> Any:
        """Burn-rate status for the SLO objectives registered in this process.

        Measured latency and availability for served traffic live in the
        ``slo`` block of ``/dashboard/data.json``; an empty list here says so
        rather than leaving the caller to guess.
        """
        try:
            from effgen.observability.slo import (
                EMPTY_SLO_DETAIL as _EMPTY_DETAIL,
            )
            from effgen.observability.slo import (
                get_tracker as _get_tracker,
            )

            statuses = _get_tracker().all_statuses()
            payload: dict[str, Any] = {"slos": statuses}
            if not statuses:
                payload["detail"] = _EMPTY_DETAIL
            return payload
        except Exception as exc:  # noqa: BLE001
            # A failure is reported with the server's error envelope and a
            # 500, not disguised as an empty result at 200.
            from effgen.api.openai_compat import error_envelope

            return JSONResponse(error_envelope(500, str(exc)), status_code=500)

    @app.get("/tools")
    async def list_tools_endpoint() -> Any:
        """List available tools."""
        tools = app.state.cli.tool_registry.list_tools()
        tool_info = []
        for tool_name in tools:
            try:
                metadata = app.state.cli.tool_registry.get_metadata(tool_name)
                tool_info.append({
                    "name": tool_name,
                    "description": metadata.description,
                    "category": metadata.category.value
                    if hasattr(metadata.category, "value") else str(metadata.category),
                })
            except Exception:  # noqa: BLE001
                tool_info.append({"name": tool_name, "description": "N/A", "category": "unknown"})
        return {"tools": tool_info, "count": len(tools)}

    @app.get("/")
    async def root() -> Any:
        """Root endpoint with API information."""
        return {
            "name": "effGen API",
            "version": _main.__version__,
            "endpoints": {
                "POST /v1/chat/completions": "OpenAI-compatible chat (SSE streaming)",
                "POST /v1/completions": "OpenAI-compatible text completion",
                "GET /v1/models": "List available model aliases",
                "POST /run": "Run a task with an agent",
                "WS /ws": "WebSocket streaming",
                "GET /health": "Health check",
                "GET /metrics": "Prometheus metrics (auth)",
                "GET /tools": "List available tools",
                "GET /docs": "OpenAPI documentation",
                "GET /dashboard": "Local metrics dashboard",
            },
        }

    @app.websocket("/ws")
    async def websocket_stream(ws: WebSocket) -> None:
        """WebSocket endpoint for streaming agent responses."""
        await ws.accept()
        try:
            while True:
                data = await ws.receive_json()
                task = data.get("task", "")
                model_id = _normalize_model_id(data.get("model", "Qwen/Qwen2.5-3B-Instruct"))
                preset_name = data.get("preset")
                if preset_name:
                    from effgen.presets import create_agent as _create_preset_agent

                    agent_instance = _create_preset_agent(preset_name, model_id)
                else:
                    tools = []
                    for name in _general_purpose_tool_names(app.state.cli.tool_registry):
                        try:
                            tools.append(await app.state.cli.tool_registry.get_tool(name))
                        except Exception:  # noqa: BLE001
                            pass
                    agent_instance = _main.Agent(_main.AgentConfig(
                        name="ws-agent", model=model_id, tools=tools,
                        enable_streaming=True, raise_on_error=False,
                    ))
                await ws.send_json({"type": "start", "task": task})
                try:
                    for token in agent_instance.stream(task):
                        await ws.send_json({"type": "token", "content": token})
                    await ws.send_json({"type": "done"})
                except Exception as e:  # noqa: BLE001
                    # Redacted like the HTTP error paths, so an upstream
                    # message carrying a key never reaches the client.
                    from effgen.api.openai_compat import _redact

                    await ws.send_json({"type": "error", "detail": _redact(str(e))})
                finally:
                    agent_instance.close()  # release per-turn agent resources
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
