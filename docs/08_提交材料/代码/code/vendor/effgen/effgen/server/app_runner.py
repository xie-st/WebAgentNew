"""Default agent-backed runner and model pool for the effGen API server.

Provides the runner the OpenAI-compatible ``/v1`` endpoints use when
:func:`effgen.server.app.create_app` is not given one: it drives an
:class:`~effgen.core.agent.Agent` around a pooled model, normalizes
OpenAI-style ``provider/model`` ids for effGen's loader, resolves tool specs
against the built-in registry, and mounts the ``/v1`` routers.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from effgen.errors import RunStoppedError

logger = logging.getLogger(__name__)

# Known provider prefixes — stable fallback when the dynamic ProviderRegistry
# is unavailable or has been reset (e.g. in test teardowns).
_KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "cerebras",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "together",
        "fireworks",
        "replicate",
        "hf",
    }
)

# ---------------------------------------------------------------------------
# Model pool — reuse loaded models across requests
# ---------------------------------------------------------------------------
#
# Constructing a fresh Agent per request is cheap, but each Agent used to spin
# up its own ModelLoader and *reload the model* every call — catastrophic for
# local GPU models and wasteful for cloud adapters. We instead pool the loaded
# *model* (the heavy object) keyed by its resolved id, bounded by an LRU, and
# build a lightweight, fresh-memory Agent around the pooled model per request.
# A fresh Agent per call keeps the endpoint stateless (no conversation bleed
# between unrelated API requests) while paying the model-load cost only once.

_MODEL_POOL: "OrderedDict[str, Any]" = OrderedDict()
_MODEL_POOL_LOCK = threading.Lock()
_LOADING_LOCKS: dict[str, Any] = {}

# Ids that produced at least one successful response this run. This drives the
# ``GET /v1/models`` listing so it advertises only models the server has really
# served — not every id that merely loaded a cloud adapter (a nonexistent
# ``provider:model`` can construct an adapter and only 404 on the actual call).
_SERVED_MODEL_IDS: "OrderedDict[str, None]" = OrderedDict()
_SERVED_MODEL_LOCK = threading.Lock()
_SERVED_MODEL_MAX = 64


def _record_served_model(resolved_model: str) -> None:
    """Record that *resolved_model* returned a successful response."""
    if not resolved_model:
        return
    with _SERVED_MODEL_LOCK:
        _SERVED_MODEL_IDS[resolved_model] = None
        _SERVED_MODEL_IDS.move_to_end(resolved_model)
        while len(_SERVED_MODEL_IDS) > _SERVED_MODEL_MAX:
            _SERVED_MODEL_IDS.popitem(last=False)


def _served_model_ids() -> list[str]:
    """Return the ids that have served a successful response this run."""
    with _SERVED_MODEL_LOCK:
        return list(_SERVED_MODEL_IDS.keys())


def _model_pool_max() -> int:
    try:
        return max(1, int(os.getenv("EFFGEN_MODEL_POOL_SIZE", "4")))
    except ValueError:
        return 4


def _get_pooled_model(resolved_model: str) -> Any:
    """Return a loaded model for *resolved_model*, reusing a pooled instance.

    Thread-safe with a bounded LRU. The (potentially slow) load happens outside
    the pool lock, serialized per-model so two concurrent first requests don't
    both load the same model.
    """
    with _MODEL_POOL_LOCK:
        model = _MODEL_POOL.get(resolved_model)
        if model is not None:
            _MODEL_POOL.move_to_end(resolved_model)
            return model
        load_lock = _LOADING_LOCKS.setdefault(resolved_model, threading.Lock())

    with load_lock:
        # Re-check: another thread may have loaded it while we waited.
        with _MODEL_POOL_LOCK:
            model = _MODEL_POOL.get(resolved_model)
            if model is not None:
                _MODEL_POOL.move_to_end(resolved_model)
                return model

        from effgen.models.model_loader import load_model as _load_model

        model = _load_model(resolved_model)

        evicted: list[Any] = []
        with _MODEL_POOL_LOCK:
            _MODEL_POOL[resolved_model] = model
            _MODEL_POOL.move_to_end(resolved_model)
            while len(_MODEL_POOL) > _model_pool_max():
                _, old = _MODEL_POOL.popitem(last=False)
                evicted.append(old)
            _LOADING_LOCKS.pop(resolved_model, None)

    for old in evicted:
        for closer in ("close", "unload", "shutdown"):
            fn = getattr(old, closer, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    logger.debug("Error closing evicted model", exc_info=True)
                break
    return model


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    """Pull (prompt_tokens, completion_tokens) from an AgentResponse.

    Prefers provider/tokenizer counts the agent recorded in metadata; falls
    back to the aggregate ``tokens_used`` for completion tokens.
    """
    meta = getattr(response, "metadata", None) or {}

    def _int_or_none(v: Any) -> int | None:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    prompt = _int_or_none(meta.get("input_tokens", meta.get("prompt_tokens")))
    completion = _int_or_none(meta.get("output_tokens", meta.get("completion_tokens")))
    if completion is None:
        completion = _int_or_none(getattr(response, "tokens_used", None)) or None
    return prompt, completion


def _extract_cost(response: Any) -> float | None:
    """Pull per-call ``cost_usd`` from an AgentResponse, if the model is priced.

    effGen records the dollar cost of a run in ``metadata["cost_usd"]`` (from the
    provider's pricing table). Returns ``None`` for unpriced/local models so the
    response can omit cost rather than report a misleading zero.
    """
    meta = getattr(response, "metadata", None) or {}
    raw = meta.get("cost_usd")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


_TRACE_ARG_MAX = 200
_TRACE_RESULT_MAX = 200


def _extract_tool_trace(response: Any) -> list[dict[str, Any]]:
    """Summarize the tools a run executed, in call order.

    The agent records each tool call as a start/complete (or failed) pair of
    events in ``response.execution_trace``. This pairs them into a compact
    per-tool summary — ``{tool, args, result_summary, ok, duration_ms}`` — so a
    caller can render what the agent actually did. Returns an empty list when no
    tool ran (a direct model answer).
    """
    trace = getattr(response, "execution_trace", None) or []
    if not isinstance(trace, list):
        return []
    steps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def _truncate(value: Any, limit: int) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= limit else text[: limit - 1] + "…"

    for event in trace:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        data = event.get("data") or {}
        if etype == "tool_call_start":
            if pending is not None:
                steps.append(pending)  # a start with no matching completion
            pending = {
                "tool": data.get("tool_name") or "tool",
                "args": _truncate(data.get("tool_input"), _TRACE_ARG_MAX),
                "result_summary": None,
                "ok": None,
                "duration_ms": None,
                "_start_ts": event.get("timestamp"),
            }
        elif etype in ("tool_call_complete", "tool_call_failed"):
            step = pending or {
                "tool": data.get("tool_name") or "tool",
                "args": _truncate(data.get("input"), _TRACE_ARG_MAX),
                "_start_ts": None,
            }
            pending = None
            ok = etype == "tool_call_complete"
            if ok:
                step["result_summary"] = _truncate(data.get("result"), _TRACE_RESULT_MAX)
            else:
                step["result_summary"] = _truncate(data.get("error"), _TRACE_RESULT_MAX)
            step["ok"] = ok
            start_ts = step.pop("_start_ts", None)
            end_ts = event.get("timestamp")
            if isinstance(start_ts, int | float) and isinstance(end_ts, int | float):
                step["duration_ms"] = round(max(0.0, (end_ts - start_ts) * 1000), 1)
            step.setdefault("result_summary", None)
            step.setdefault("ok", ok)
            step.setdefault("duration_ms", None)
            steps.append(step)
    if pending is not None:
        pending.pop("_start_ts", None)
        steps.append(pending)
    return steps


class _StreamWithUsage:
    """Token iterator that publishes the run's usage once it is exhausted.

    The OpenAI-compatible route reads ``usage`` after the last token to fill the
    ``stream_options.include_usage`` chunk, so a streamed request reports the
    same token counts and cost a non-streamed one does. It also reads
    ``finish_reason``, which is ``"length"`` when the loop stopped the run before
    the model wrote an answer — the same thing the non-streamed path reports.
    Iterating closes the ephemeral agent (releasing its memory handles and
    circuit breakers, but not the pooled model, which is shared and stays
    loaded).
    """

    def __init__(self, agent: Any, prompt: str, resolved_model: str) -> None:
        self._agent = agent
        self._prompt = prompt
        self._resolved_model = resolved_model
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str = "stop"

    def __iter__(self) -> Any:
        served = False
        try:
            for chunk in self._agent.stream(self._prompt):
                served = True
                yield chunk
            self.usage = self._agent.last_stream_usage
            record = getattr(self._agent, "last_stream_response", None)
            if getattr(record, "outcome", None) == "stopped":
                self.finish_reason = "length"
        finally:
            self._agent.close()
            if served:
                _record_served_model(self._resolved_model)


def _build_default_runner() -> Any:
    """Construct an agent-backed runner for the OpenAI-compatible endpoints.

    Returns a callable ``runner(prompt, *, model, tools, stream, **kw)`` that
    drives an :class:`~effgen.core.agent.Agent` around a *pooled* model. For
    ``stream=True`` it returns a lazy token generator (so the route can begin
    flushing SSE immediately); otherwise it returns a
    :class:`~effgen.api.openai_compat.RunnerResult` carrying real usage and the
    resolved model id.
    """
    def _runner(
        prompt: str,
        *,
        model: str,
        tools: Any = None,
        stream: bool = False,
        temperature: float | None = None,
        **_: Any,
    ) -> Any:
        from effgen.api.openai_compat import RunnerResult
        from effgen.core.agent import Agent, AgentConfig

        # Pool access resolves through ``effgen.server.app`` at call time so a
        # caller that rebinds ``app._get_pooled_model`` (e.g. to supply a stub
        # model) stays effective.
        from effgen.server import app as _app_module

        resolved_model = _normalize_model_id(model)
        pooled_model = _app_module._get_pooled_model(resolved_model)
        resolved_tools = _resolve_tools(tools)
        config = AgentConfig(
            # Name the served model in the agent's identity so a server-driven
            # run is distinguishable in traces and run history instead of every
            # request reading as the same "api" agent.
            name=f"api:{resolved_model}",
            model=pooled_model,  # pre-loaded instance → no per-request reload
            tools=resolved_tools,
            temperature=temperature if temperature is not None else 0.7,
            require_model=True,
            # A generation failure must reach the route as a typed exception so
            # it maps to a real HTTP status (404/502/503/...) via the same
            # taxonomy every other runner failure uses, instead of returning
            # HTTP 200 with the error text as the answer. ``stream()`` already
            # raises on failure unconditionally, so this only changes run().
            raise_on_error=True,
        )
        agent = Agent(config)

        if stream:
            # Close the ephemeral agent once the stream is exhausted. ``close()``
            # releases the agent's own resources (memory handles, circuit
            # breakers) but NOT the pooled model, which is shared and stays
            # loaded — and it silences the "garbage-collected without close()"
            # warning that would otherwise fire per streamed request.
            return _StreamWithUsage(agent, prompt, resolved_model)

        try:
            # A failure raises here (raise_on_error=True above), so a response
            # that reaches this point is always a success — except for a run the
            # loop stopped, which is served below as a completed request whose
            # generation was cut short.
            try:
                response = agent.run(prompt)
                finish_reason = "stop"
            except RunStoppedError as stopped:
                # The request was valid and the server did its job; the model
                # did not finish. That is a 200 with OpenAI's own vocabulary for
                # "cut off by a limit", not a 5xx blaming the server.
                response = stopped.response
                finish_reason = "length"
            _record_served_model(resolved_model)
            prompt_tokens, completion_tokens = _extract_usage(response)
            tool_trace = _extract_tool_trace(response)
            run_meta = getattr(response, "metadata", None) or {}
            # Surface which tools ran (and how many) so a caller can render the
            # step trace. Lives in the runner's ``metadata`` and is carried into
            # the response's non-standard ``effgen`` object by the OpenAI-compat
            # layer; standard OpenAI clients ignore unknown keys.
            extra_meta: dict[str, Any] = {
                "tool_calls": int(getattr(response, "tool_calls", 0) or 0),
            }
            if tool_trace:
                extra_meta["trace"] = tool_trace
            run_id = run_meta.get("run_id")
            if run_id:
                extra_meta["run_id"] = run_id
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason:
                extra_meta["stop_reason"] = stop_reason
            outcome = getattr(response, "outcome", None)
            if outcome:
                extra_meta["outcome"] = outcome
            partial = getattr(response, "partial", None)
            if partial is not None:
                extra_meta["partial"] = partial.to_dict()
            return RunnerResult(
                text=getattr(response, "output", "") or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                resolved_model=resolved_model,
                cost_usd=_extract_cost(response),
                finish_reason=finish_reason,
                metadata=extra_meta,
            )
        finally:
            agent.close()

    return _runner


def _normalize_model_id(model: str) -> str:
    """Normalize an OpenAI-style ``provider/model`` id for effGen's loader.

    OpenAI-compatible clients send ``"cerebras/llama3.1-8b"`` (slash), but
    effGen's ``ModelLoader`` routes API providers via the ``"provider:model"``
    (colon) prefix. Without this, a slash id falls through to the local
    Transformers path and fails with "not a valid model identifier". Rewrite
    the *provider* separator to a colon only when the prefix is a known
    provider, leaving bare ids and HF ``org/repo`` ids untouched.
    """
    if not isinstance(model, str) or "/" not in model:
        return model
    prefix, _, rest = model.partition("/")
    try:
        from effgen.models.registry import ProviderRegistry

        providers = ProviderRegistry.list_providers()
        if not providers:
            # Registry may have been reset (e.g. by test teardowns).
            # Fall back to the static provider set rather than attempting a
            # re-import that could cause circular-import or side-effect issues.
            providers = _KNOWN_PROVIDERS  # type: ignore[assignment]

        if prefix in providers:
            return f"{prefix}:{rest}"
    except Exception:  # noqa: BLE001 - registry optional; fall back to static set
        if prefix in _KNOWN_PROVIDERS:
            return f"{prefix}:{rest}"
    return model


def _resolve_tools(tools: Any) -> list[Any]:
    """Resolve OpenAI-style tool specs / bare names into effGen tool objects.

    The OpenAI chat schema sends tools as ``{"type": "function", "function":
    {"name": ...}}`` dicts; effGen's :class:`Agent` expects tool *instances*.
    Names are resolved against the built-in tool registry (RBAC has already
    authorized the names). A requested tool that the server does not host is
    **not** silently dropped — that would leave a client expecting OpenAI
    function-calling with prose and no ``tool_calls``. Instead the unhosted
    names are collected and surfaced as an :class:`UnknownToolError`, which the
    route turns into a clear ``400``.
    """
    if not tools:
        return []
    import asyncio

    from effgen.api.openai_compat import UnknownToolError
    from effgen.tools.registry import ToolRegistry

    registry = ToolRegistry()
    try:
        registry.discover_builtin_tools()
    except Exception:  # noqa: BLE001 - tool discovery is best-effort at startup; server still serves
        pass

    def _get(name: str) -> Any:
        """Resolve one tool name, working whether or not a loop is running."""
        coro = registry.get_tool(name)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # A loop is already running (FastAPI route): run in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()

    resolved: list[Any] = []
    unresolved: list[str] = []
    for tool in tools:
        if not isinstance(tool, str) and not isinstance(tool, dict):
            # Already a tool object.
            resolved.append(tool)
            continue
        name = _tool_name(tool)
        try:
            obj = _get(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve tool %r: %s", name, exc)
            obj = None
        if obj is not None:
            resolved.append(obj)
        else:
            unresolved.append(name)
    if unresolved:
        raise UnknownToolError(unresolved)
    return resolved


def _tool_name(tool: Any) -> str:
    """Extract a tool name from an OpenAI-style tool spec or a bare string."""
    if isinstance(tool, dict):
        return tool.get("function", {}).get("name") or tool.get("name") or "tool"
    return str(tool)


def _mount_existing_routers(
    app: Any, *, runner: Any = None, extra_models: Any = None
) -> None:
    """Mount the OpenAI-compat + embeddings routers."""
    try:
        from effgen.api.openai_compat import create_openai_router

        _runner = runner or _build_default_runner()
        router = create_openai_router(
            _runner, extra_models=extra_models or _served_model_ids
        )
        app.include_router(router)
        logger.info("Mounted OpenAI-compat router at /v1 (RBAC + budget enforced)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI-compat router not mounted: %s", exc)

    try:
        from effgen.api.embeddings import create_embeddings_router

        router = create_embeddings_router()
        # Router already has /v1 in path; mount at root
        app.include_router(router)
        logger.info("Mounted embeddings router")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Embeddings router not mounted: %s", exc)
