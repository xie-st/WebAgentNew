"""Terminal rendering for a workflow DAG — levels, edges, and per-node status.

Reads the topology a :class:`~effgen.core.workflow.WorkflowDAG` already exposes
(nodes, edges, execution levels) plus, after a run, the per-node results, and
lays it out as an indented dependency graph: nodes grouped by the level they run
in (so parallel steps sit together), each annotated with a status glyph and, when
known, its duration, cost, and token count, followed by the downstream nodes it
feeds. The output is plain text with status carried by the glyph, so it stays
readable when piped; callers on a color terminal can style each line.
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


def workflow_diagram_lines(
    name: str,
    node_ids: list[str],
    edges: list[dict[str, Any]],
    levels: list[list[str]],
    node_results: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Build ``(style, text)`` lines for a workflow dependency graph.

    Args:
        name: Workflow name for the header.
        node_ids: All node ids in the DAG.
        edges: Edge dicts with ``source``/``target`` (and optional ``key``).
        levels: Node ids grouped into parallel execution levels.
        node_results: Optional per-node result dicts (``id``, ``status``,
            ``execution_time``, ``metadata`` with ``cost_usd``/``tokens_used``).
            When absent, nodes render as pending structure only.

    Returns:
        A list of ``(style, text)`` pairs. ``style`` is a theme style name (or
        ``""`` for default); ``text`` is plain and safe to print without color.
    """
    results_by_id: dict[str, dict[str, Any]] = {}
    for nr in node_results or []:
        nid = nr.get("id")
        if nid is not None:
            results_by_id[str(nid)] = nr

    # Downstream targets per node (what each node feeds).
    downstream: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for e in edges:
        src, tgt = str(e.get("source")), str(e.get("target"))
        if src in downstream:
            downstream[src].append(tgt)

    out: list[tuple[str, str]] = []
    out.append(("effgen.heading", f"Workflow: {name}"))
    out.append(("effgen.muted", f"{len(node_ids)} node(s), {len(edges)} edge(s), {len(levels)} level(s)"))

    for lvl_idx, level in enumerate(levels):
        parallel = "  (parallel)" if len(level) > 1 else ""
        out.append(("effgen.label", f"Level {lvl_idx}{parallel}"))
        for nid in level:
            nr = results_by_id.get(nid, {})
            status = str(nr.get("status", "pending"))
            glyph, style = status_glyph(status), status_style(status)
            meta = nr.get("metadata") or {}
            bits = [status]
            dur = _fmt_dur(nr.get("execution_time"))
            if dur:
                bits.append(dur)
            cost = _fmt_cost(meta.get("cost_usd"))
            if cost:
                bits.append(cost)
            tokens = meta.get("tokens_used")
            if tokens:
                bits.append(f"{int(tokens):,} tok")
            err = nr.get("error")
            annot = "   " + "  ".join(bits)
            out.append((style, f"  {glyph} {nid}{annot}"))
            if err:
                out.append(("effgen.error", f"      → {str(err)[:100]}"))
            targets = downstream.get(nid, [])
            for j, tgt in enumerate(targets):
                branch = "└─▶" if j == len(targets) - 1 else "├─▶"
                out.append(("effgen.muted", f"      {branch} {tgt}"))
    return out
