"""Terminal rendering for a team of agents — members, tools, and their edges.

Reads the topology a :class:`~effgen.core.orchestrator.TeamConfig` exposes
through ``to_dict()`` (and, after a run, the per-agent results) and lays it out
as an indented member list: the manager first when the pattern has one, then
each member with a status glyph and, when known, its model, duration, cost and
token count, followed by the tools it can reach and the members it feeds. The
output is plain text with status carried by the glyph, so it stays readable when
piped; callers on a color terminal can style each line.
"""

from __future__ import annotations

from typing import Any

from .palette import status_glyph, status_style


def _fmt_dur(seconds: float | None) -> str:
    if not seconds:
        return ""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_cost(cost: float | None) -> str:
    if cost is None or cost <= 0:
        return ""
    if cost >= 1e-4:
        return f"${cost:.4f}"
    if cost >= 1e-6:
        return f"${cost:.6f}"
    return f"${cost:.1e}"


def team_diagram_lines(
    topology: dict[str, Any],
    agent_responses: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Build ``(style, text)`` lines for a team's shape.

    Args:
        topology: The dict returned by ``TeamConfig.to_dict()`` — ``name``,
            ``pattern``, ``manager``, ``agents`` and ``edges``.
        agent_responses: Optional per-agent result dicts from a
            ``TeamResponse`` (``agent_name``, ``success``, ``execution_time``,
            ``cost_usd``, ``tokens_used``, ``error``). When absent, members
            render as pending structure only.

    Returns:
        A list of ``(style, text)`` pairs. ``style`` is a theme style name (or
        ``""`` for default); ``text`` is plain and safe to print without color.
    """
    results: dict[str, dict[str, Any]] = {}
    for entry in agent_responses or []:
        name = entry.get("agent_name")
        if name is None:
            continue
        prior = results.get(str(name))
        # A member that ran more than once (collaborative rounds, a fanned-out
        # subtask) shows its totals, and any failure wins over a success.
        if prior is None:
            results[str(name)] = dict(entry)
            continue
        prior["execution_time"] = (prior.get("execution_time") or 0) + (
            entry.get("execution_time") or 0
        )
        prior["cost_usd"] = (prior.get("cost_usd") or 0) + (entry.get("cost_usd") or 0)
        prior["tokens_used"] = (prior.get("tokens_used") or 0) + (entry.get("tokens_used") or 0)
        if entry.get("success") is False:
            prior["success"] = False
            prior["error"] = entry.get("error") or prior.get("error")

    manager = topology.get("manager")
    members: list[dict[str, Any]] = list(topology.get("agents") or [])
    edges: list[dict[str, Any]] = list(topology.get("edges") or [])

    downstream: dict[str, list[str]] = {}
    for edge in edges:
        downstream.setdefault(str(edge.get("source")), []).append(str(edge.get("target")))

    pattern = str(topology.get("pattern") or "sequential")
    out: list[tuple[str, str]] = []
    out.append(("effgen.heading", f"Team: {topology.get('name', 'team')}"))
    count = len(members) + (1 if manager else 0)
    out.append((
        "effgen.muted",
        f"{pattern} · {count} agent(s), {len(edges)} edge(s)",
    ))

    def _member_lines(node: dict[str, Any], label: str) -> None:
        name = str(node.get("name", "agent"))
        result = results.get(name)
        if result is None:
            status = "pending"
        elif result.get("success") is False:
            status = "failed"
        else:
            status = "completed"
        bits = [label, status]
        model = node.get("model")
        if model and model != "unknown":
            bits.append(str(model))
        if result is not None:
            dur = _fmt_dur(result.get("execution_time"))
            if dur:
                bits.append(dur)
            cost = _fmt_cost(result.get("cost_usd"))
            if cost:
                bits.append(cost)
            tokens = result.get("tokens_used")
            if tokens:
                bits.append(f"{int(tokens):,} tok")
        out.append((status_style(status), f"  {status_glyph(status)} {name}   " + "  ".join(bits)))
        if result is not None and result.get("error"):
            out.append(("effgen.error", f"      → {str(result['error'])[:100]}"))
        for tool in node.get("tools") or []:
            out.append(("effgen.muted", f"      ⚒ {tool}"))
        targets = downstream.get(name, [])
        for j, target in enumerate(targets):
            branch = "└─▶" if j == len(targets) - 1 else "├─▶"
            out.append(("effgen.muted", f"      {branch} {target}"))

    if manager:
        _member_lines(manager, "manager")
    for node in members:
        _member_lines(node, "member")
    if not members and not manager:
        out.append(("effgen.muted", "  (no agents)"))
    return out
