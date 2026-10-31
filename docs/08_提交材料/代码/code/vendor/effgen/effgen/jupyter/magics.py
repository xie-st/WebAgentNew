"""effGen IPython/Jupyter magics.

Magics
------
%effgen_chat <message>
    One-shot chat via the configured model.  Displays the response inline.

%%effgen_agent [preset]
    Treat the cell body as a task.  Runs an effGen agent (optionally with a
    named preset) and displays the final answer plus a tool-use trace.

%effgen_metrics
    Snapshot current Prometheus counters inline as a formatted table.

Configuration
-------------
Set environment variables or pass options on the magic line:

    EFFGEN_JUPYTER_MODEL      — model id for the magics (no built-in default;
                                falls back to EFFGEN_DEFAULT_MODEL when unset)
    EFFGEN_DEFAULT_MODEL      — the framework-wide default model id
    EFFGEN_JUPYTER_SERVER_URL — server URL for /v1 chat (default: none;
                                uses in-process agent when unset)

effGen never silently picks a paid cloud model, so if neither variable is set
you must pass ``--model``/``-m`` on the magic line. Run ``effgen models list``
to see options or ``effgen doctor`` to check which providers are usable.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Any

# IPython types are optional so the module is importable outside of a kernel
# (e.g. for unit tests that mock the kernel).
try:
    from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
    from IPython.display import HTML, Markdown, display
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "IPython is required for Jupyter magics: pip install ipython"
    ) from _exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_model() -> str | None:
    """Resolve the magics' default model id without ever guessing.

    Honours ``EFFGEN_JUPYTER_MODEL`` first (a Jupyter-specific override), then
    the framework-wide ``EFFGEN_DEFAULT_MODEL``. Returns ``None`` when neither
    is set — effGen never silently selects a paid cloud model, so callers show a
    friendly "pass --model" message instead of a stale hardcoded id.
    """
    for var in ("EFFGEN_JUPYTER_MODEL", "EFFGEN_DEFAULT_MODEL"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


def _no_model_message() -> str:
    """Friendly, actionable Markdown shown when no model could be resolved."""
    return (
        "**No model configured.** Pass one on the magic line, e.g.\n\n"
        "```\n"
        "%effgen_chat -m gpt-5-nano Hello!\n"
        "```\n\n"
        "or set a default once with `EFFGEN_DEFAULT_MODEL` (effGen never picks a "
        "paid cloud model for you). Run `effgen models list` to see options or "
        "`effgen doctor` to check which providers are usable."
    )


def _server_url() -> str | None:
    return os.environ.get("EFFGEN_JUPYTER_SERVER_URL") or None


def _chat_via_server(message: str, model: str, server_url: str) -> str:
    """Send a chat request to the effGen API server and return the reply."""
    try:
        import urllib.error
        import urllib.request

        url = server_url.rstrip("/") + "/v1/chat/completions"
        payload = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": message}]}
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return f"[effGen server error] {exc}"


def _trace_from_response(response: Any) -> list[dict[str, Any]]:
    """Extract a tool-use trace from an :class:`AgentResponse`.

    The agent records tool activity as events on ``response.execution_trace``
    (a list of dicts). We surface the ``tool_call_start`` / ``tool_call_complete``
    events as ``{"tool": name, "input": ..., "output": ...}`` rows.
    """
    events = getattr(response, "execution_trace", None) or []
    trace: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type", "")
        if not str(etype).startswith("tool_call"):
            continue
        data = ev.get("data", {}) or {}
        name = data.get("tool_name") or data.get("tool") or data.get("name") or ev.get("message", "tool")
        if etype == "tool_call_start":
            trace.append({"tool": name, "input": data.get("tool_input") or data.get("input") or {}, "output": None})
        else:  # tool_call_complete / tool_call_failed
            output = data.get("result") or data.get("output") or ev.get("message")
            # Attach the output to the most recent matching start row if present.
            for row in reversed(trace):
                if row["tool"] == name and row["output"] is None:
                    row["output"] = output
                    break
            else:
                trace.append({"tool": name, "input": {}, "output": output})
    return trace


def _chat_in_process(message: str, model: str) -> Any:
    """Run a one-shot chat in-process using effGen's Agent.

    Returns the :class:`AgentResponse` on success (so the caller can display its
    rich ``_repr_html_`` card in the notebook), or a plain error string when the
    agent could not be built or run.
    """
    try:
        from effgen.core.agent import Agent, AgentConfig

        config = AgentConfig(name="jupyter", model=model, require_model=True)
        agent = Agent(config)
        return agent.run(message)
    except Exception as exc:  # noqa: BLE001
        return f"[effGen error] {exc}"


def _resolve_tools(names: list[str]) -> list[Any]:
    """Resolve built-in tool names into tool instances (best-effort)."""
    if not names:
        return []
    objects: list[Any] = []
    try:
        from effgen.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.discover_builtin_tools()
        for name in names:
            try:
                if registry.is_registered(name):
                    objects.append(registry._tools[name]())  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return objects


def _run_agent(
    model: str,
    task: str,
    preset: str | None,
    extra_tools: list[str],
) -> Any:
    """Run an agentic task and return the :class:`AgentResponse`.

    When *preset* names a known effGen preset (math, research, coding, general,
    rag, minimal, …) the agent is built via :func:`effgen.presets.create_agent`
    so it is wired with that preset's tools (e.g. ``general``/``math`` include a
    calculator, so arithmetic tasks are computed, not guessed). Otherwise a
    plain :class:`Agent` is used, optionally augmented with ``extra_tools``.

    Returns the response object on success (so its rich card renders), or a
    plain string when a context-overflow hint applies or the run errored.
    """
    extra_tool_objs = _resolve_tools(extra_tools)
    try:
        if preset:
            from effgen.presets import create_agent, list_presets

            if preset in list_presets():
                agent = create_agent(
                    preset,
                    model,
                    agent_name="jupyter_agent",
                    extra_tools=extra_tool_objs or None,
                    require_model=True,
                )
                response = agent.run(task)
                hint = _context_overflow_hint(getattr(response, "output", ""), preset)
                return hint or response

        # No (recognised) preset: plain agent with any explicitly-requested tools.
        from effgen.core.agent import Agent, AgentConfig

        config = AgentConfig(
            name="jupyter_agent",
            model=model,
            tools=extra_tool_objs,
            require_model=True,
        )
        agent = Agent(config)
        response = agent.run(task)
        hint = _context_overflow_hint(getattr(response, "output", ""), preset)
        return hint or response
    except Exception as exc:  # noqa: BLE001
        hint = _context_overflow_hint(str(exc), preset)
        return hint or f"[effGen agent error] {exc}"


def _context_overflow_hint(text: str, preset: str | None) -> str | None:
    """Return a friendly, actionable message when *text* indicates the
    request (the prompt, or a large preset's tool-schema payload) exceeded
    a model's context window or a provider's token-rate limit.

    Shares its trigger detection with
    :func:`effgen.models.errors.context_overflow_hint` (the same signal set
    the CLI/``Agent.run`` error path uses), so a Groq per-minute rate
    message ("reduce your message size") is recognised here exactly as it
    is everywhere else. Returns ``None`` when *text* is not such a signal,
    so callers can fall through to the normal output.
    """
    from effgen.models.errors import context_overflow_hint

    if context_overflow_hint(text) is None:
        return None
    suggestion = (
        "a smaller preset (e.g. `math` for arithmetic) or a model with a larger "
        "context window (e.g. `--model cerebras:gpt-oss-120b`)"
    )
    label = f"`{preset}`" if preset else "this"
    return (
        f"⚠️ The {label} preset's tools don't fit this model's context window "
        f"or rate limit. Try {suggestion}."
    )


def _format_tool_trace(trace: list[dict[str, Any]]) -> str:
    """Render tool-use trace as Markdown."""
    if not trace:
        return ""
    lines = ["\n\n---\n**Tool calls:**\n"]
    for i, tc in enumerate(trace, 1):
        lines.append(f"**{i}. `{tc['tool']}`**")
        if tc.get("input"):
            inp = json.dumps(tc["input"], indent=2) if isinstance(tc["input"], dict) else str(tc["input"])
            lines.append(f"```json\n{inp}\n```")
        if tc.get("output") is not None:
            out = str(tc["output"])[:500]
            lines.append(f"*Output:* `{out}`")
        lines.append("")
    return "\n".join(lines)


def _display_outcome(header: str, result: Any) -> None:
    """Display a chat/agent outcome in the notebook.

    *result* is either an :class:`AgentResponse` (rendered via its rich
    ``_repr_html_`` card after a small Markdown header) or a plain string
    (an error or a server reply, rendered inline as Markdown).
    """
    if isinstance(result, str):
        display(Markdown(f"{header}\n\n{result}"))
        return
    # AgentResponse (or any object) — show the header, then let the object's
    # rich representation (Phase-aware _repr_html_ card) render the answer,
    # metrics and step trace.
    display(Markdown(header))
    display(result)


def _prometheus_snapshot() -> dict[str, float]:
    """Return a {metric_name: value} snapshot from prometheus_client."""
    try:
        from prometheus_client import REGISTRY

        snapshot: dict[str, float] = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                snapshot[sample.name] = sample.value
        return snapshot
    except Exception:  # noqa: BLE001
        return {}


def _format_metrics_html(snapshot: dict[str, float]) -> str:
    """Render the Prometheus snapshot as an HTML table."""
    if not snapshot:
        return "<p><em>No Prometheus metrics registered (is the effGen server running?)</em></p>"

    rows = "".join(
        f"<tr><td><code>{name}</code></td><td style='text-align:right'>{value:g}</td></tr>"
        for name, value in sorted(snapshot.items())
    )
    return f"""
<style>
.effgen-metrics {{
  border-collapse: collapse;
  font-family: monospace;
  font-size: 0.85em;
  max-height: 400px;
  overflow-y: auto;
  display: block;
}}
.effgen-metrics th, .effgen-metrics td {{
  border: 1px solid #ccc;
  padding: 3px 10px;
}}
.effgen-metrics th {{
  background: #f0f0f0;
}}
</style>
<table class="effgen-metrics">
  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Magics class
# ---------------------------------------------------------------------------

@magics_class
class EffgenMagics(Magics):  # type: ignore[misc]
    """effGen IPython/Jupyter magics."""

    # ------------------------------------------------------------------
    # %effgen_chat
    # ------------------------------------------------------------------

    @magic_arguments()  # type: ignore[misc]
    @argument("message", nargs="*", help="Chat message to send")
    @argument("--model", "-m", default=None, help="Model ID override")
    @argument("--server", "-s", default=None, help="effGen server URL override")
    @line_magic("effgen_chat")  # type: ignore[misc]
    def effgen_chat(self, line: str) -> None:  # type: ignore[override]
        """One-shot chat.

        Usage::

            %effgen_chat Tell me a one-sentence joke
            %effgen_chat --model cerebras:gpt-oss-120b Hello!
        """
        args = parse_argstring(self.effgen_chat, line)
        message = " ".join(args.message)
        if not message.strip():
            display(Markdown("*Usage: `%effgen_chat <message>`*"))
            return

        model = args.model or _default_model()
        server = args.server or _server_url()

        if not model:
            display(Markdown(_no_model_message()))
            return

        t0 = time.perf_counter()
        if server:
            result: Any = _chat_via_server(message, model, server)
        else:
            result = _chat_in_process(message, model)
        elapsed = time.perf_counter() - t0

        _display_outcome(f"**effGen** ({model}, {elapsed:.2f}s)", result)

    # ------------------------------------------------------------------
    # %%effgen_agent
    # ------------------------------------------------------------------

    @magic_arguments()  # type: ignore[misc]
    @argument("preset", nargs="?", default=None, help="Agent preset name")
    @argument("--model", "-m", default=None, help="Model ID override")
    @argument("--tools", "-t", nargs="*", help="Tool names to enable")
    @cell_magic("effgen_agent")  # type: ignore[misc]
    def effgen_agent(self, line: str, cell: str) -> None:  # type: ignore[override]
        """Run the cell body as an agentic task with tool use.

        Usage::

            %%effgen_agent coding
            Write a Python function to check if a number is prime.

            %%effgen_agent --model cerebras:gpt-oss-120b
            Search for the latest papers on mixture-of-experts.
        """
        args = parse_argstring(self.effgen_agent, line)
        task = textwrap.dedent(cell).strip()
        if not task:
            display(Markdown("*Cell body is empty — write a task for the agent.*"))
            return

        model = args.model or _default_model()
        preset_name: str | None = args.preset
        extra_tools: list[str] = args.tools or []

        if not model:
            display(Markdown(_no_model_message()))
            return

        t0 = time.perf_counter()
        result = _run_agent(model, task, preset_name, extra_tools)
        elapsed = time.perf_counter() - t0

        label = f"agent:{preset_name or 'default'}" if preset_name else "agent"
        _display_outcome(f"**effGen** ({label} · {model}, {elapsed:.2f}s)", result)

    # ------------------------------------------------------------------
    # %effgen_metrics
    # ------------------------------------------------------------------

    @line_magic("effgen_metrics")  # type: ignore[misc]
    def effgen_metrics(self, line: str) -> None:  # type: ignore[override]
        """Display a snapshot of current Prometheus counters.

        Usage::

            %effgen_metrics
        """
        snapshot = _prometheus_snapshot()
        html = _format_metrics_html(snapshot)
        display(HTML(f"<h4>effGen Metrics Snapshot</h4>{html}"))
