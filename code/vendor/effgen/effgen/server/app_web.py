"""Dashboard and playground web surfaces for the effGen API server.

Mounts the dashboard SPA (static assets, ``data.json``, ``catalog.json``,
``history.json``, ``topology.json``, and the SSE span stream) at
``/dashboard`` and the in-browser playground at ``/playground``, with
path-confined static-asset resolution shared by both.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# The JSON-payload builders these routes serve are resolved through
# ``effgen.server.app`` at call time (not imported here) so a caller that
# rebinds one of those names on the app module (e.g. to stub the payload)
# stays effective.

# FastAPI Request must be importable at module level so route type annotations
# resolve correctly when `from __future__ import annotations` is active.
try:
    from fastapi import Request as _FastAPIRequest
except ImportError:  # pragma: no cover
    _FastAPIRequest = None  # type: ignore[assignment,misc]


def _asset_not_found(name: str) -> Any:
    """Return the shared 404 error envelope for a missing web asset.

    The dashboard and playground answer a missing file with the same
    ``{"error": {message, type, param, code}}`` shape as the ``/v1`` routes, so
    a caller never has to handle two error formats from one server.
    """
    from fastapi.responses import JSONResponse

    from effgen.api.openai_compat import error_envelope

    label = name[:100]
    return JSONResponse(
        error_envelope(
            404,
            f"{label} not found",
            code="not_found",
            error_type="invalid_request_error",
            redact=False,
        ),
        status_code=404,
    )


def _resolve_web_asset(asset_path: str, *roots: Any) -> Any:
    """Resolve a static asset request against the given directories in order.

    The first directory holding the file wins, so a surface can override a
    shared asset with its own copy. The resolved path must stay inside the
    directory it was found in; a request that escapes it (``..`` segments, an
    absolute path, a symlink out of the tree) resolves to ``None``.
    """
    from pathlib import Path as _Path

    if not asset_path or asset_path.startswith("/"):
        return None
    for root in roots:
        base = _Path(root).resolve()
        try:
            candidate = (base / asset_path).resolve()
            candidate.relative_to(base)
        except (ValueError, OSError):
            continue
        if candidate.is_file():
            return candidate
    return None


def _mount_dashboard(app: Any) -> None:
    """Mount the local dashboard SPA at /dashboard.

    Serves:
    - ``GET /dashboard``           — the SPA index.html
    - ``GET /dashboard/data.json`` — live metrics + run data as JSON
    - ``GET /dashboard/spans``     — SSE stream of recent span events
    - ``GET /dashboard/{path}``    — other static assets (JS, CSS, and the
      files shared with the playground)
    """
    try:
        from pathlib import Path as _Path

        from fastapi import APIRouter
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

        router = APIRouter(prefix="/dashboard", tags=["dashboard"])

        static_dir = _Path(__file__).parent.parent / "dashboard" / "static"
        # Assets both web surfaces load (command palette, keyboard layer).
        shared_dir = _Path(__file__).parent.parent / "webui" / "static"

        # ---- /dashboard → index.html ----------------------------------------
        @router.get("", include_in_schema=False)
        @router.get("/", include_in_schema=False)
        async def dashboard_index() -> Any:
            """Serve the dashboard SPA."""
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return _asset_not_found("dashboard")

        # ---- /dashboard/data.json -------------------------------------------
        @router.get("/data.json", include_in_schema=False)
        async def dashboard_data() -> Any:
            """Live metrics + recent runs as JSON consumed by the SPA."""
            from effgen.server import app as _app_module

            return JSONResponse(_app_module._build_dashboard_data())

        # ---- /dashboard/catalog.json ----------------------------------------
        @router.get("/catalog.json", include_in_schema=False)
        async def dashboard_catalog(provider: str | None = None) -> Any:
            """Model catalog (pricing, capabilities, provenance) for the SPA.

            Serves the same payload as ``GET /v1/models/catalog`` but under the
            dashboard's own access rule, so the catalog view follows the
            dashboard's local-viewing story (``EFFGEN_PUBLIC_DASHBOARD``) rather
            than requiring a separately-supplied API key.
            """
            from effgen.api.openai_compat import build_model_catalog

            return JSONResponse(build_model_catalog(provider))

        # ---- /dashboard/history.json ----------------------------------------
        @router.get("/history.json", include_in_schema=False)
        async def dashboard_history(
            limit: int = 50,
            status: str | None = None,
            search: str | None = None,
            run_id: str | None = None,
        ) -> Any:
            """Stored run history and saved sessions for the History view.

            Reads the same run store as ``effgen runs`` and the same session
            files as ``effgen sessions``, so the dashboard shows runs recorded
            by the CLI and by other processes, not only this server's traffic.
            Served under the dashboard's own access rule.
            """
            from effgen.server import app as _app_module

            return JSONResponse(_app_module._build_dashboard_history(
                limit=limit, status=status, search=search, run_id=run_id,
            ))

        # ---- /dashboard/topology.json ---------------------------------------
        @router.get("/topology.json", include_in_schema=False)
        async def dashboard_topology(limit: int = 6) -> Any:
            """Recent team and workflow executions as node-link graphs.

            Built from the durable run store plus the buffered spans, so a team
            or workflow run from a script or the CLI is included, not only work
            done inside this server process. Served under the dashboard's own
            access rule.
            """
            from effgen.observability.topology import build_topology

            return JSONResponse(build_topology(limit=max(1, min(int(limit or 6), 20))))

        # ---- /dashboard/spans (SSE) ----------------------------------------
        @router.get("/spans", include_in_schema=False)
        async def dashboard_spans(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
            """Server-sent events stream of recent trace spans.

            Emits all currently-buffered spans, then heartbeats while the
            client stays connected. The loop terminates as soon as the client
            disconnects (so it never leaks a worker thread) and is also bounded
            by ``EFFGEN_DASHBOARD_SSE_MAX_SECONDS`` (default 3600s) as a
            backstop — the SPA simply reconnects via ``EventSource``.
            """
            import asyncio
            import json as _json
            import time as _time

            max_seconds = float(os.getenv("EFFGEN_DASHBOARD_SSE_MAX_SECONDS", "3600"))
            heartbeat_s = float(os.getenv("EFFGEN_DASHBOARD_SSE_HEARTBEAT_SECONDS", "5"))

            async def _generate():  # type: ignore[return]
                # Stream any buffered spans first.
                from effgen.server import app as _app_module

                for span in _app_module._get_recent_spans():
                    yield f"data: {_json.dumps(span)}\n\n"
                # Heartbeat until the client disconnects or the cap is hit.
                deadline = _time.monotonic() + max_seconds
                while _time.monotonic() < deadline:
                    if await request.is_disconnected():
                        break
                    await asyncio.sleep(heartbeat_s)
                    yield ": heartbeat\n\n"

            return StreamingResponse(
                _generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # ---- /dashboard/{file} (static assets) ------------------------------
        @router.get("/{asset_path:path}", include_in_schema=False)
        async def dashboard_static(asset_path: str) -> Any:
            """Serve static assets (app.js, style.css, the shared webui files)."""
            asset = _resolve_web_asset(asset_path, static_dir, shared_dir)
            if asset is not None:
                return FileResponse(str(asset))
            return _asset_not_found(asset_path)

        app.include_router(router)
        logger.info("Mounted dashboard at /dashboard")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard not mounted: %s", exc)


def _mount_playground(app: Any) -> None:
    """Mount the in-browser playground SPA at /playground.

    Serves:
    - ``GET /playground``            — the SPA index.html
    - ``GET /playground/bootstrap``  — presets, tool options, defaults, and (in
      local-view mode) a session key, as JSON
    - ``GET /playground/{path}``     — other static assets (JS, CSS, and the
      files shared with the dashboard)

    The Run button drives the existing ``POST /v1/chat/completions`` endpoint;
    the playground adds no new model-execution path.
    """
    try:
        from pathlib import Path as _Path

        from fastapi import APIRouter
        from fastapi.responses import FileResponse, JSONResponse

        router = APIRouter(prefix="/playground", tags=["playground"])
        static_dir = _Path(__file__).parent.parent / "playground" / "static"
        shared_dir = _Path(__file__).parent.parent / "webui" / "static"

        @router.get("", include_in_schema=False)
        @router.get("/", include_in_schema=False)
        async def playground_index() -> Any:
            index = static_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return _asset_not_found("playground")

        @router.get("/bootstrap", include_in_schema=False)
        async def playground_bootstrap(request: _FastAPIRequest) -> Any:  # type: ignore[valid-type]
            """Config the page needs to render the pickers and (optionally) run."""
            from effgen.server import app as _app_module

            return JSONResponse(_app_module._build_playground_bootstrap(request))

        @router.get("/{asset_path:path}", include_in_schema=False)
        async def playground_static(asset_path: str) -> Any:
            asset = _resolve_web_asset(asset_path, static_dir, shared_dir)
            if asset is not None:
                return FileResponse(str(asset))
            return _asset_not_found(asset_path)

        app.include_router(router)
        logger.info("Mounted playground at /playground")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Playground not mounted: %s", exc)


# Tool names offered as one-click attachments in the playground. Kept to
# credential-free, side-effect-free tools so a first run works with no setup;
# a preset the user selects may pre-fill its own tools on top of these.
_PLAYGROUND_SAFE_TOOLS: tuple[str, ...] = ("calculator", "wikipedia")


def _build_playground_bootstrap(request: Any) -> dict[str, Any]:
    """Assemble the JSON served at /playground/bootstrap.

    Carries the presets (name + system prompt + tools, applied client-side), the
    always-safe tool options, sensible defaults, and — only when the operator
    opted into a public playground and a static key is configured — a session
    key so a local-view demo can Run without pasting one. When spend is not
    authorized locally the key is ``None`` and the page prompts for one.
    """
    app_state = getattr(request, "app", None)
    state = getattr(app_state, "state", None)
    public_playground = bool(getattr(state, "public_playground", False))
    dev_mode = bool(getattr(state, "dev_mode", False))
    configured_key = getattr(state, "api_key", "") or ""

    # Only hand the browser a key when the operator has authorized local spend
    # (public playground) AND a static key exists. In dev mode auth is bypassed,
    # so no key is needed for Run to work.
    session_key = configured_key if (public_playground and configured_key) else None

    presets: list[dict[str, Any]] = []
    try:
        from effgen.presets.registry import list_presets as _list_presets

        for name in _list_presets():
            try:
                from effgen.presets.registry import get_preset as _get_preset

                cfg = _get_preset(name)
            except Exception:  # noqa: BLE001 - skip a preset that won't load
                continue
            presets.append(
                {
                    "name": getattr(cfg, "name", name),
                    "description": getattr(cfg, "description", ""),
                    "system_prompt": getattr(cfg, "system_prompt", "") or "",
                    "tools": list(getattr(cfg, "tool_names", []) or []),
                    "temperature": getattr(cfg, "temperature", None),
                }
            )
    except Exception:  # noqa: BLE001 - presets are optional; page still works without them
        presets = []

    from effgen.api.openai_compat import default_model_id

    # Imported here (not at module level): the app factory module imports this
    # one at import time, so its helpers are reachable only after import.
    from effgen.server.app import _server_version

    return {
        "version": _server_version(),
        "presets": presets,
        "tools": list(_PLAYGROUND_SAFE_TOOLS),
        "default_model": default_model_id(),
        "spend_authorized": bool(session_key) or dev_mode,
        "session_key": session_key,
        "dev_mode": dev_mode,
        # Client-side spend guardrails the page applies by default.
        "defaults": {"max_tokens": 512, "temperature": 0.7, "stream": True},
        "catalog_url": "/v1/models/catalog",
    }
