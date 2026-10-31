"""Tests for the multi-agent topology graph — data, route, and rendering.

Covers:
1. A team serializes its topology before it runs (``TeamConfig.to_dict``).
2. Correlated spans and run records reassemble into one execution graph.
3. The dashboard route serves the graph and is protected by default.
4. The panel is self-contained (no external asset, no graph library), carries
   status as glyph + text rather than color alone, and its animation is behind
   ``prefers-reduced-motion``.
5. The terminal team diagram renders without color.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).parents[2] / "effgen" / "dashboard" / "static"


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    """Point run history at a temp dir and clear the in-memory buffers."""
    from effgen.observability import run_log, tracing

    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    run_log.clear()
    with tracing._SPAN_BUFFER_LOCK:
        tracing._SPAN_BUFFER.clear()
    yield
    run_log.clear()
    with tracing._SPAN_BUFFER_LOCK:
        tracing._SPAN_BUFFER.clear()


# ---------------------------------------------------------------------------
# Team topology, serialized before a run
# ---------------------------------------------------------------------------


class _StubAgent:
    def __init__(self, name, model="stub:model", tools=None):
        self.name = name
        self.model_name = model
        self.tools = dict.fromkeys(tools or [], object())


def _team(pattern, agents, manager=None):
    from effgen.core.orchestrator import TeamConfig

    return TeamConfig(name="desk", pattern=pattern, agents=agents, manager_agent=manager)


class TestTeamTopology:
    def test_hierarchical_team_serializes_delegation_edges(self):
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(
            OrchestrationPattern.HIERARCHICAL,
            [_StubAgent("worker-a", tools=["calculator"]), _StubAgent("worker-b")],
            manager=_StubAgent("lead"),
        )
        shape = team.to_dict()
        assert shape["pattern"] == "hierarchical"
        assert shape["manager"]["name"] == "lead"
        assert [a["name"] for a in shape["agents"]] == ["worker-a", "worker-b"]
        assert shape["agents"][0]["tools"] == ["calculator"]
        assert shape["edges"] == [
            {"source": "lead", "target": "worker-a", "kind": "delegation"},
            {"source": "lead", "target": "worker-b", "kind": "delegation"},
        ]

    def test_collaborative_team_serializes_peer_edges(self):
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(
            OrchestrationPattern.COLLABORATIVE,
            [_StubAgent("a"), _StubAgent("b"), _StubAgent("c")],
        )
        kinds = {e["kind"] for e in team.to_dict()["edges"]}
        assert kinds == {"peer"}
        assert len(team.to_dict()["edges"]) == 3

    def test_sequential_team_serializes_handoff_chain(self):
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(OrchestrationPattern.SEQUENTIAL, [_StubAgent("a"), _StubAgent("b")])
        assert team.to_dict()["edges"] == [
            {"source": "a", "target": "b", "kind": "handoff"},
        ]

    def test_parallel_team_has_no_inter_agent_edges(self):
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(OrchestrationPattern.PARALLEL, [_StubAgent("a"), _StubAgent("b")])
        assert team.to_dict()["edges"] == []

    def test_topology_is_available_without_running(self):
        """The shape is readable before anything is spent on a run."""
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(OrchestrationPattern.HIERARCHICAL, [_StubAgent("a")], manager=_StubAgent("m"))
        assert team.to_dict()["agents"][0]["model"] == "stub:model"


# ---------------------------------------------------------------------------
# Correlated telemetry → one execution graph
# ---------------------------------------------------------------------------


class TestExecutionCorrelation:
    def test_runs_of_one_execution_share_an_id(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="team", name="desk") as execution:
            run_log.record_run(model="m", agent="lead", run_id="r1")
            with execution_scope(role="worker", parent_agent="lead"):
                run_log.record_run(model="m", agent="worker-a", run_id="r2")

        rows = run_log.read_runs(limit=10)
        assert {r["execution_id"] for r in rows} == {execution["execution_id"]}
        worker = next(r for r in rows if r["agent"] == "worker-a")
        assert worker["parent_agent"] == "lead"
        assert worker["role"] == "worker"
        assert worker["execution_kind"] == "team"
        assert worker["execution_name"] == "desk"

    def test_nested_scope_does_not_rename_the_execution(self, isolated_history):
        """kind/name identify the execution, so a nested scope leaves them alone."""
        from effgen.observability import run_log
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="team", name="desk"):
            # A decomposition inside the team names itself, but the execution
            # it joined keeps its own identity.
            with execution_scope(kind="delegation", name="lead", role="sub-agent"):
                run_log.record_run(model="m", agent="helper", run_id="r1")

        row = run_log.read_runs(limit=1)[0]
        assert row["execution_kind"] == "team"
        assert row["execution_name"] == "desk"
        assert row["role"] == "sub-agent"

    def test_explicit_execution_id_keeps_ambient_kind_and_name(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="workflow", name="pipeline") as execution:
            run_log.record_run(
                model="m", agent="a", run_id="r1",
                execution_id=execution["execution_id"],
            )

        row = run_log.read_runs(limit=1)[0]
        assert row["execution_kind"] == "workflow"
        assert row["execution_name"] == "pipeline"

    def test_sub_agent_dispatch_keeps_the_caller_execution(self, isolated_history):
        """A sub-agent runs in a worker thread and must not lose its execution."""
        import asyncio

        from effgen.core.sub_agent_manager import SubAgentManager
        from effgen.observability.tracing import current_execution, execution_scope

        manager = SubAgentManager()
        seen: dict[str, object] = {}

        def _capture(agent_info, subtask, progress_callback=None):
            seen["execution"] = current_execution()
            return "done"

        manager._execute_sub_agent = _capture

        async def _drive():
            with execution_scope(kind="team", name="desk") as execution:
                await manager._execute_sub_agent_async({}, None, None)
                return execution["execution_id"]

        expected = asyncio.run(_drive())
        assert seen["execution"] is not None, "the worker thread lost the execution"
        assert seen["execution"]["execution_id"] == expected
        assert seen["execution"]["execution_kind"] == "team"

    def test_execution_id_survives_outside_the_scope(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="team", name="desk"):
            run_log.record_run(model="m", agent="a", run_id="r1")
        run_log.record_run(model="m", agent="solo", run_id="r2")

        solo = next(r for r in run_log.read_runs(limit=10) if r["agent"] == "solo")
        assert solo["execution_id"] is None

    def test_read_executions_groups_member_runs(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="workflow", name="pipeline"):
            run_log.record_run(model="m", agent="a", run_id="r1", cost_usd=0.001)
            run_log.record_run(model="m", agent="b", run_id="r2", cost_usd=0.002, error="boom")

        executions = run_log.read_executions()
        assert len(executions) == 1
        assert executions[0]["kind"] == "workflow"
        assert executions[0]["name"] == "pipeline"
        assert len(executions[0]["runs"]) == 2
        assert executions[0]["failed"] == 1
        assert executions[0]["cost_usd"] == pytest.approx(0.003)

    def test_build_topology_returns_nodes_and_edges(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.topology import build_topology
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="team", name="desk"):
            with execution_scope(role="manager"):
                run_log.record_run(model="m", agent="lead", run_id="r1")
            with execution_scope(role="worker", parent_agent="lead"):
                run_log.record_run(model="m", agent="worker-a", run_id="r2")

        graph = build_topology(limit=3)["executions"][0]
        node_ids = {n["id"] for n in graph["nodes"]}
        assert {"lead", "worker-a"} <= node_ids
        assert {"source": "lead", "target": "worker-a", "kind": "delegation", "count": 1} in graph["edges"]

    def test_build_topology_is_empty_without_multi_agent_work(self, isolated_history):
        from effgen.observability.topology import build_topology

        assert build_topology()["executions"] == []

    def test_failed_run_colors_its_node(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.topology import build_topology
        from effgen.observability.tracing import execution_scope

        with execution_scope(kind="team", name="desk"):
            run_log.record_run(model="m", agent="a", run_id="r1", error="ModelNotFoundError: nope")

        node = build_topology()["executions"][0]["nodes"][0]
        assert node["status"] == "error"
        assert "ModelNotFoundError" in node["error"]

    def test_tool_span_adds_a_tool_node_and_edge(self, isolated_history):
        from effgen.observability import run_log
        from effgen.observability.topology import build_topology
        from effgen.observability.tracing import (
            execution_scope,
            start_agent_run,
            start_tool_call,
        )

        with execution_scope(kind="team", name="desk"), \
                start_agent_run(preset="worker-a", run_id="r9"):
            with start_tool_call(tool_name="calculator", tool_input="6*7"):
                pass
            run_log.record_run(model="m", agent="worker-a", run_id="r9")

        graph = build_topology()["executions"][0]
        assert any(n["id"] == "tool:calculator" and n["type"] == "tool" for n in graph["nodes"])
        assert any(e["target"] == "tool:calculator" and e["kind"] == "tool_use" for e in graph["edges"])


# ---------------------------------------------------------------------------
# Span records carry identity and outcome as fields
# ---------------------------------------------------------------------------


class TestSpanFields:
    def test_span_carries_kind_and_identity_fields(self, isolated_history):
        from effgen.observability.tracing import (
            get_recent_spans,
            start_agent_run,
            start_tool_call,
        )

        with start_agent_run(preset="judge", run_id="r1"), \
                start_tool_call(tool_name="calculator"):
            pass

        spans = get_recent_spans(limit=5)
        tool = next(s for s in spans if s["kind"] == "tool")
        agent = next(s for s in spans if s["kind"] == "agent")
        assert tool["tool"] == "calculator"
        assert agent["agent"] == "judge"
        # The display name is kept for compatibility.
        assert agent["name"].startswith("effgen.agent.run")

    def test_run_that_fails_without_raising_records_the_error(self, isolated_history):
        from effgen.observability.tracing import (
            get_recent_spans,
            mark_run_error,
            start_agent_run,
        )

        with start_agent_run(preset="broken", run_id="r1"):
            mark_run_error("ModelNotFoundError: no such model")

        span = get_recent_spans(limit=1)[0]
        assert span["status"] == "error"
        assert "ModelNotFoundError" in span["error"]

    def test_skipped_step_is_recorded(self, isolated_history):
        from effgen.observability.tracing import get_recent_spans, record_skipped_step

        record_skipped_step("final", reason="upstream 'broken' did not complete (failed)")
        span = get_recent_spans(limit=1)[0]
        assert span["status"] == "skipped"
        assert "did not complete" in span["note"]


# ---------------------------------------------------------------------------
# Dashboard panel: self-contained, accessible, behind auth
# ---------------------------------------------------------------------------


class TestTopologyPanel:
    def test_index_contains_topology_panel(self):
        html = (STATIC_DIR / "index.html").read_text()
        for element_id in ("panel-topology", "topo-svg", "topo-select", "topo-detail"):
            assert f'id="{element_id}"' in html

    def test_panel_has_an_empty_state(self):
        html = (STATIC_DIR / "index.html").read_text()
        assert "No multi-agent activity recorded yet" in html

    def test_graph_is_hand_drawn_svg_without_a_library(self):
        js = (STATIC_DIR / "app.js").read_text()
        html = (STATIC_DIR / "index.html").read_text()
        assert "<svg" in html
        assert "topoNodeSvg" in js and "topoEdgeSvg" in js
        for library in ("d3", "cytoscape", "vis-network", "mermaid", "sigma"):
            assert f"{library}." not in js.lower().replace("topology.", "")

    def test_no_external_assets_in_topology_code(self):
        """The panel adds no remote host reference to any static file."""
        offenders: list[str] = []
        for fname in ("index.html", "app.js", "style.css"):
            text = (STATIC_DIR / fname).read_text()
            for match in re.findall(r"""https?://[^\s"'()]+|(?<![:\w])//[^\s"'()]+""", text):
                offenders.append(f"{fname}: {match}")
        assert not offenders, f"External asset references found: {offenders}"

    def test_status_is_not_color_only(self):
        """Each status pairs a glyph and a text label with its color."""
        js = (STATIC_DIR / "app.js").read_text()
        html = (STATIC_DIR / "index.html").read_text()
        assert "TOPO_GLYPH" in js
        for glyph in ("●", "◐", "⊘", "✗"):
            assert glyph in js
            assert glyph in html  # the legend repeats them
        # The node label states the status in words.
        assert "status ${status}" in js

    def test_nodes_are_focusable_and_labeled(self):
        js = (STATIC_DIR / "app.js").read_text()
        assert 'tabindex="0"' in js
        assert 'role="button"' in js
        assert "aria-label" in js
        assert 'ev.key === "Enter"' in js

    def test_animation_is_behind_reduced_motion(self):
        js = (STATIC_DIR / "app.js").read_text()
        css = (STATIC_DIR / "style.css").read_text()
        # The pulse class is only attached when motion is allowed …
        assert "prefers-reduced-motion" in js
        assert "reducedMotion()" in js
        assert "animate ?" in js
        # … and the stylesheet disables animation for a reduced-motion viewer.
        assert "@media (prefers-reduced-motion: reduce)" in css

    def test_arrowhead_is_not_styled_as_a_connector(self):
        """The marker needs its own class: .topo-edge sets fill:none, and a CSS
        rule outranks a fill presentation attribute, so a shared class would
        render the arrowheads invisible."""
        js = (STATIC_DIR / "app.js").read_text()
        css = (STATIC_DIR / "style.css").read_text()
        marker = js[js.index('<defs><marker id="topo-arrow"'):]
        marker = marker[: marker.index("</marker>")]
        assert 'class="topo-edge"' not in marker
        assert "topo-arrowhead" in marker
        assert re.search(r"\.topo-arrowhead\s*\{[^}]*fill:", css)

    def test_escaping_covers_attribute_positions(self):
        """Agent names reach the page inside attributes, so quotes must escape."""
        js = (STATIC_DIR / "app.js").read_text()
        body = js[js.index("function esc(s)"):]
        body = body[: body.index("function cssEscape")]
        assert "&quot;" in body and "&#39;" in body

    def test_rerender_restores_keyboard_focus(self):
        """Polling replaces the SVG; a focused node must not drop the reader."""
        js = (STATIC_DIR / "app.js").read_text()
        assert "document.activeElement" in js
        assert "restore.focus()" in js

    def test_theme_variables_used_not_hardcoded_colors(self):
        css = (STATIC_DIR / "style.css").read_text()
        topo_block = css[css.index("Multi-agent topology graph"):]
        assert "var(--ok)" in topo_block and "var(--err)" in topo_block
        assert not re.search(r"\.topo-[\w-]+\s*\{[^}]*#[0-9a-fA-F]{3,6}", topo_block)


class TestTopologyRoute:
    @pytest.fixture(autouse=True)
    def client(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from effgen.server.app import create_app

        self._client = TestClient(create_app(dev_mode=True), raise_server_exceptions=False)
        return self._client

    def test_topology_json_serves_expected_shape(self):
        resp = self._client.get("/dashboard/topology.json")
        assert resp.status_code == 200
        payload = resp.json()
        assert "executions" in payload and isinstance(payload["executions"], list)
        assert "count" in payload

    def test_topology_json_requires_auth_by_default(self):
        from fastapi.testclient import TestClient

        from effgen.server.app import create_app

        client = TestClient(create_app(dev_mode=False), raise_server_exceptions=False)
        assert client.get("/dashboard/topology.json").status_code == 401


# ---------------------------------------------------------------------------
# Terminal team diagram
# ---------------------------------------------------------------------------


class TestTeamDiagram:
    def test_diagram_lists_manager_members_and_tools(self):
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(
            OrchestrationPattern.HIERARCHICAL,
            [_StubAgent("worker-a", tools=["calculator"])],
            manager=_StubAgent("lead"),
        )
        text = team.diagram()
        assert "Team: desk" in text
        assert "lead" in text and "manager" in text
        assert "worker-a" in text
        assert "calculator" in text
        assert "└─▶ worker-a" in text

    def test_diagram_is_plain_text_with_status_glyphs(self):
        from effgen.core.orchestrator import OrchestrationPattern

        team = _team(OrchestrationPattern.SEQUENTIAL, [_StubAgent("a")])
        text = team.diagram()
        assert "\x1b[" not in text  # no escape codes when captured
        assert "○" in text  # pending glyph before a run

    def test_diagram_annotates_a_run(self):
        from effgen.core.orchestrator import (
            OrchestrationPattern,
            TeamResponse,
        )

        team = _team(OrchestrationPattern.SEQUENTIAL, [_StubAgent("a"), _StubAgent("b")])
        response = TeamResponse(
            output="done",
            agent_responses=[
                {"agent_name": "a", "success": True, "execution_time": 0.5,
                 "cost_usd": 0.0001, "tokens_used": 120},
                {"agent_name": "b", "success": False, "error": "boom"},
            ],
        )
        text = team.diagram(response)
        assert "completed" in text and "failed" in text
        assert "120 tok" in text
        assert "boom" in text

    def test_manager_status_reflects_the_team_outcome(self):
        """A manager has no subtask entry, so it takes the team's own status."""
        from effgen.core.orchestrator import OrchestrationPattern, TeamResponse

        team = _team(
            OrchestrationPattern.HIERARCHICAL,
            [_StubAgent("worker-a")],
            manager=_StubAgent("lead"),
        )
        ok = TeamResponse(output="done", success=True, agent_responses=[
            {"agent_name": "worker-a", "success": True},
        ])
        assert "lead   manager  completed" in team.diagram(ok)

        worker_failed = TeamResponse(
            output="Error", success=False,
            agent_responses=[{"agent_name": "worker-a", "success": False, "error": "boom"}],
            metadata={"reason": "sub_agent_failed"},
        )
        text = team.diagram(worker_failed)
        assert "lead   manager  completed" in text
        assert "worker-a   member  failed" in text
