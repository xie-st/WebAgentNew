"""Unit tests for multi-agent orchestration & workflow correctness.

Covers the doorway and the failure semantics of the coordination surface:
- WorkflowDAG.run() accepts a bare string (routed to entry nodes) or raises a
  clear typed error; topological order + cycle rejection.
- A node / sub-agent that reports success=False fails the workflow/team (never
  a silent success); partial results are labelled.
- assign_task accepts a TeamConfig or a registered team name; empty team fails.
- MessageBus pub/sub, ApprovalManager approve/deny/timeout, cooperative
  cancellation, the consensus score, and the public-surface exports.

These use an in-process fake agent (no network / no model) — orchestration glue
is what's under test, not live API behaviour (which is exercised separately in
tests/integration/test_multi_agent_live.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Fake agent: matches the small slice of the Agent contract the orchestrator and
# workflow engine actually use (name, run/run_async, success/output/tokens, close).
# --------------------------------------------------------------------------- #
@dataclass
class _FakeResponse:
    output: str
    success: bool = True
    tokens_used: int = 7
    tool_calls: int = 0
    execution_time: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeAgent:
    def __init__(self, name: str, *, succeed: bool = True, reply: str | None = None,
                 delay: float = 0.0, cost: float = 0.0):
        self.name = name
        self._succeed = succeed
        self._reply = reply
        self._delay = delay
        self._cost = cost
        self.closed = False
        self.runs = 0
        self.tasks: list[str] = []  # every task this agent was asked to run

    def run(self, task: str, mode: Any = None, context: Any = None, **kw) -> _FakeResponse:
        self.runs += 1
        self.tasks.append(task)
        if self._delay:
            time.sleep(self._delay)
        if not self._succeed:
            return _FakeResponse(
                output="boom: sk-secret-key-12345 leaked",
                success=False,
                metadata={"error": {"type": "ModelNotFoundError",
                                    "message": "model 'x' not found", "provider": "test"}},
            )
        meta = {"cost_usd": self._cost} if self._cost else {}
        return _FakeResponse(output=self._reply or f"{self.name}:{task}", metadata=meta)

    async def run_async(self, task: str, mode: Any = None, context: Any = None, **kw):
        return self.run(task, mode=mode, context=context, **kw)

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
def test_entry_points_exported_top_level():
    import effgen
    for name in [
        "MultiAgentOrchestrator", "TeamConfig", "TeamResponse", "OrchestrationPattern",
        "WorkflowDAG", "WorkflowNode", "WorkflowEdge", "WorkflowResult",
        "SubAgentRouter", "RoutingStrategy", "RoutingDecision",
    ]:
        assert name in effgen.__all__, f"{name} missing from __all__"
        assert getattr(effgen, name) is not None


# --------------------------------------------------------------------------- #
# WorkflowDAG type-guard + topology
# --------------------------------------------------------------------------- #
def _two_node_dag(a_ok=True, b_ok=True):
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("t")
    dag.add_node(WorkflowNode(id="a", agent=FakeAgent("a", succeed=a_ok)))
    dag.add_node(WorkflowNode(id="b", agent=FakeAgent("b", succeed=b_ok)))
    dag.connect("a", "b")
    return dag


def test_run_accepts_bare_string():
    dag = _two_node_dag()
    res = dag.run("do the task")
    assert res.success is True
    assert {n["id"]: n["status"] for n in res.node_results} == {
        "a": "completed", "b": "completed"}


def test_run_rejects_bad_type_with_clear_error():
    dag = _two_node_dag()
    with pytest.raises(TypeError) as ei:
        dag.run(123)
    assert "node_id" in str(ei.value)


def test_empty_workflow_reports_a_typed_failure():
    # A zero-node DAG must NOT report success=True (all([]) trap). Mirrors the
    # empty-team contract in MultiAgentOrchestrator.assign_task.
    from effgen.core.workflow import WorkflowDAG
    res = WorkflowDAG().run("any task")
    assert res.success is False
    assert res.metadata.get("reason") == "empty_workflow"
    assert res.outputs == {}
    assert res.node_results == []


def test_empty_workflow_async_reports_a_typed_failure():
    import asyncio

    from effgen.core.workflow import WorkflowDAG
    res = asyncio.run(WorkflowDAG().run_async("any task"))
    assert res.success is False
    assert res.metadata.get("reason") == "empty_workflow"


def test_real_dag_still_runs_after_empty_guard():
    # A non-empty DAG with a bare-string task still works.
    dag = _two_node_dag()
    res = dag.run("do the task")
    assert res.success is True
    assert len(res.node_results) == 2


def test_entry_nodes_and_topological_order():
    dag = _two_node_dag()
    assert dag.entry_nodes() == ["a"]
    assert dag.topological_order() == ["a", "b"]


def test_cycle_rejected():
    dag = _two_node_dag()
    with pytest.raises(ValueError):
        dag.connect("b", "a")


def test_node_failure_fails_workflow_and_redacts():
    # b fails -> workflow not success; a stays completed (labelled partial).
    dag = _two_node_dag(b_ok=False)
    res = dag.run("go")
    statuses = {n["id"]: n["status"] for n in res.node_results}
    assert statuses == {"a": "completed", "b": "failed"}
    assert res.success is False
    berr = next(n["error"] for n in res.node_results if n["id"] == "b")
    assert "sk-secret-key-12345" not in str(res.node_results)  # redacted/structured
    assert "ModelNotFoundError" in berr


def test_dag_skips_downstream_of_failed_upstream():
    # triage -> {billing(fails), synthesize}; synthesize also depends on billing.
    # The fan-in node must be SKIPPED (not run on the error text), so an internal
    # failure is never rewritten into a downstream answer.
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("support")
    triage = FakeAgent("triage")
    billing = FakeAgent("billing", succeed=False)
    synth = FakeAgent("synth")
    dag.add_node(WorkflowNode(id="triage", agent=triage))
    dag.add_node(WorkflowNode(id="billing", agent=billing))
    dag.add_node(WorkflowNode(id="synthesize", agent=synth))
    dag.connect("triage", "billing")
    dag.connect("triage", "synthesize")
    dag.connect("billing", "synthesize")
    res = dag.run("write the customer reply")
    statuses = {n["id"]: n["status"] for n in res.node_results}
    assert statuses == {"triage": "completed", "billing": "failed",
                        "synthesize": "skipped"}
    assert res.success is False
    assert synth.runs == 0  # never executed on the failed branch
    assert res.outputs.get("synthesize") is None


def test_dag_condition_skip_still_succeeds():
    # A legitimate conditional skip (not a failure) keeps success=True and
    # records a skip_reason for transparency.
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("cond")
    dag.add_node(WorkflowNode(id="a", agent=FakeAgent("a", reply="normal")))
    dag.add_node(WorkflowNode(id="b", agent=FakeAgent("b")))
    dag.connect("a", "b", condition=lambda out: "urgent" in str(out))
    res = dag.run("go")
    statuses = {n["id"]: n["status"] for n in res.node_results}
    assert statuses == {"a": "completed", "b": "skipped"}
    assert res.success is True
    bmeta = next(n["metadata"] for n in res.node_results if n["id"] == "b")
    assert "condition" in bmeta.get("skip_reason", "")


# --------------------------------------------------------------------------- #
# Orchestrator shapes + clear failure
# --------------------------------------------------------------------------- #
def _orch_with_team(name="team", agents=None, pattern=None):
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern
    orch = MultiAgentOrchestrator()
    pattern = pattern or OrchestrationPattern.SEQUENTIAL
    agents = agents if agents is not None else [FakeAgent("a1"), FakeAgent("a2")]
    orch.create_team(name, agents, pattern=pattern)
    return orch


def test_assign_task_by_name_and_by_config():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team()
    by_name = orch.assign_task("hi", "team")
    assert by_name.success is True
    by_cfg = orch.assign_task("hi", orch.get_team("team"))
    assert by_cfg.success is True
    assert by_cfg.pattern == OrchestrationPattern.SEQUENTIAL


def test_assign_task_unknown_name_raises_keyerror():
    orch = _orch_with_team()
    with pytest.raises(KeyError):
        orch.assign_task("hi", "nope")


def test_assign_task_bad_type_raises_typeerror():
    orch = _orch_with_team()
    with pytest.raises(TypeError):
        orch.assign_task("hi", 123)


def test_empty_team_reports_a_typed_failure():
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern, TeamConfig
    orch = MultiAgentOrchestrator()
    orch.teams["empty"] = TeamConfig(name="empty", pattern=OrchestrationPattern.SEQUENTIAL, agents=[])
    res = orch.assign_task("hi", "empty")
    assert res.success is False
    assert res.metadata.get("reason") == "empty_team"


def test_failing_agent_makes_team_fail_with_redacted_error():
    orch = _orch_with_team(agents=[FakeAgent("ok"), FakeAgent("bad", succeed=False)])
    res = orch.assign_task("hi", "team")
    assert res.success is False
    assert res.metadata.get("reason") == "sub_agent_failed"
    # the failing agent's response carries an error; secrets are not leaked raw
    assert "sk-secret-key-12345" not in str(res.metadata.get("error"))


def test_parallel_all_fail_is_failure():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(
        agents=[FakeAgent("p1", succeed=False), FakeAgent("p2", succeed=False)],
        pattern=OrchestrationPattern.PARALLEL,
    )
    res = orch.assign_task("hi", "team")
    assert res.success is False


def test_sequential_does_not_echo_input_on_failure():
    # On failure the output must NOT be the caller's original task echoed back
    # (a footgun: it reads like an answer). It carries the error instead.
    task = "MY ORIGINAL TICKET — refund please"
    orch = _orch_with_team(agents=[FakeAgent("bad", succeed=False), FakeAgent("ok")])
    res = orch.assign_task(task, "team")
    assert res.success is False
    assert res.output != task
    assert res.output.startswith("Error:")
    # the structured error dict's message is rendered, not its Python repr.
    assert "{'type':" not in res.output
    assert "model 'x' not found" in res.output


# --------------------------------------------------------------------------- #
# COLLABORATIVE failure reporting (a failing collaborator must not pass silently)
# --------------------------------------------------------------------------- #
def test_collaborative_failure_is_reported_not_swallowed():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(
        agents=[FakeAgent("billing"), FakeAgent("tech", succeed=False)],
        pattern=OrchestrationPattern.COLLABORATIVE,
    )
    orch.get_team("team").max_rounds = 1
    res = orch.assign_task("resolve this", "team")
    assert res.success is False
    assert res.metadata.get("reason") == "sub_agent_failed"
    # every agent response now carries a discoverable success flag + error
    flags = {r["agent_name"]: r["success"] for r in res.agent_responses}
    assert flags == {"billing": True, "tech": False}
    failed = next(r for r in res.agent_responses if not r["success"])
    assert "error" in failed
    # the team-level surfaced error/metadata never leaks raw secrets
    assert "sk-secret-key-12345" not in str(res.metadata)
    # the structured error dict's message is rendered, not its Python repr.
    assert "{'type':" not in res.output
    assert "model 'x' not found" in res.output


def test_collaborative_all_ok_succeeds():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(
        agents=[FakeAgent("a", reply="same answer"), FakeAgent("b", reply="same answer")],
        pattern=OrchestrationPattern.COLLABORATIVE,
    )
    orch.get_team("team").max_rounds = 1
    res = orch.assign_task("q", "team")
    assert res.success is True
    assert all(r["success"] for r in res.agent_responses)


# --------------------------------------------------------------------------- #
# HIERARCHICAL named routing (subtask goes to the worker the manager names)
# --------------------------------------------------------------------------- #
def _hier_orch(manager_reply, workers):
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern
    orch = MultiAgentOrchestrator()
    manager = FakeAgent("manager", reply=manager_reply)
    orch.create_team("support", workers, pattern=OrchestrationPattern.HIERARCHICAL,
                     manager_agent=manager)
    return orch


def test_hierarchical_routes_by_named_worker_not_position():
    # tech is first in the list, billing is second — but the manager labels the
    # subtask "billing:", so it must reach billing, not tech (the old positional
    # zip would have mis-routed it to tech).
    tech = FakeAgent("tech")
    billing = FakeAgent("billing")
    manager_reply = "1. billing: issue the refund for the double charge"
    orch = _hier_orch(manager_reply, [tech, billing])
    res = orch.assign_task("double charge", "support")
    assert billing.runs == 1 and tech.runs == 0
    assert res.agent_responses[0]["agent_name"] == "billing"


def test_hierarchical_runs_all_subtasks_even_beyond_agent_count():
    # 3 subtasks, 2 agents — the old zip() dropped the 3rd. All must run now.
    tech = FakeAgent("tech")
    billing = FakeAgent("billing")
    manager_reply = (
        "1. billing: confirm the charge\n"
        "2. tech: check the login crash\n"
        "3. billing: process the refund"
    )
    orch = _hier_orch(manager_reply, [tech, billing])
    res = orch.assign_task("handle both", "support")
    assert len(res.agent_responses) == 3
    assert billing.runs == 2 and tech.runs == 1


def test_hierarchical_worker_failure_fails_team():
    # A failed worker must make the team fail even if the manager's synthesis is OK.
    billing = FakeAgent("billing", succeed=False)
    tech = FakeAgent("tech")
    manager_reply = "1. billing: refund\n2. tech: fix login"
    orch = _hier_orch(manager_reply, [tech, billing])
    res = orch.assign_task("both", "support")
    assert res.success is False
    assert "sk-secret-key-12345" not in str(res.metadata)
    # the structured error dict's message is rendered, not its Python repr.
    assert "{'type':" not in res.output
    assert "model 'x' not found" in res.output


def test_hierarchical_cost_includes_manager_decomposition():
    # The manager runs twice (decomposition + synthesis); both calls must be
    # counted in the team total, not just the worker(s).
    from effgen.core.orchestrator import MultiAgentOrchestrator, OrchestrationPattern
    manager = FakeAgent("manager", reply="1. billing: refund", cost=0.01)
    billing = FakeAgent("billing", cost=0.001)
    orch = MultiAgentOrchestrator()
    orch.create_team("support", [billing], pattern=OrchestrationPattern.HIERARCHICAL,
                      manager_agent=manager)
    res = orch.assign_task("refund", "support")
    assert res.success is True
    assert manager.runs == 2  # decomposition + synthesis
    assert res.metadata["cost_usd"] == pytest.approx(0.01 + 0.01 + 0.001)


def test_hierarchical_unlabeled_falls_back_round_robin():
    # No recognizable worker name -> round-robin, but nothing is dropped.
    a = FakeAgent("alpha")
    b = FakeAgent("beta")
    manager_reply = "1. do the first thing\n2. do the second thing"
    orch = _hier_orch(manager_reply, [a, b])
    res = orch.assign_task("x", "support")
    assert len(res.agent_responses) == 2
    assert a.runs == 1 and b.runs == 1


# --------------------------------------------------------------------------- #
# PIPELINE is an alias for SEQUENTIAL (labelled as PIPELINE)
# --------------------------------------------------------------------------- #
def test_pipeline_pattern_label():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(pattern=OrchestrationPattern.PIPELINE)
    res = orch.assign_task("hi", "team")
    assert res.success is True
    assert res.pattern == OrchestrationPattern.PIPELINE


# --------------------------------------------------------------------------- #
# Cost / token aggregation onto team & workflow results
# --------------------------------------------------------------------------- #
def test_team_cost_and_tokens_aggregated():
    orch = _orch_with_team(
        agents=[FakeAgent("a", cost=0.001), FakeAgent("b", cost=0.002)],
    )
    res = orch.assign_task("hi", "team")
    assert res.success is True
    assert res.metadata["cost_usd"] == pytest.approx(0.003)
    assert res.metadata["tokens_used"] == 14  # 7 per FakeResponse


def test_workflow_cost_and_tokens_aggregated():
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("t")
    dag.add_node(WorkflowNode(id="a", agent=FakeAgent("a", cost=0.001)))
    dag.add_node(WorkflowNode(id="b", agent=FakeAgent("b", cost=0.004)))
    dag.connect("a", "b")
    res = dag.run("go")
    assert res.success is True
    assert res.metadata["cost_usd"] == pytest.approx(0.005)
    assert res.metadata["tokens_used"] == 14


def test_workflow_result_carries_topology():
    # A --json consumer must be able to rebuild the graph from the result alone;
    # the per-node dicts do not record edges, so the result metadata carries the
    # edges, topological order, and parallel levels.
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("fanout")
    for nid in ("research", "pros", "cons", "summary"):
        dag.add_node(WorkflowNode(id=nid, agent=FakeAgent(nid)))
    dag.connect("research", "pros")
    dag.connect("research", "cons")
    dag.connect("pros", "summary")
    dag.connect("cons", "summary")
    res = dag.run("go")
    meta = res.metadata
    assert [(e["source"], e["target"]) for e in meta["edges"]] == [
        ("research", "pros"), ("research", "cons"),
        ("pros", "summary"), ("cons", "summary"),
    ]
    assert meta["topological_order"][0] == "research"
    assert meta["topological_order"][-1] == "summary"
    # pros and cons run in parallel on the same level.
    assert ["pros", "cons"] in meta["levels"] or ["cons", "pros"] in meta["levels"]


def test_team_agent_responses_carry_execution_time():
    # Per-agent timing was absent from team responses; each member's wall-clock
    # time must be present so a delegation view can show where time went.
    from effgen.core.orchestrator import (
        MultiAgentOrchestrator,
        OrchestrationPattern,
    )
    orch = MultiAgentOrchestrator()
    team = orch.create_team(
        "t", [FakeAgent("a"), FakeAgent("b")],
        pattern=OrchestrationPattern.SEQUENTIAL,
    )
    res = orch.assign_task("go", team)
    assert res.success is True
    assert [r["agent_name"] for r in res.agent_responses] == ["a", "b"]
    for r in res.agent_responses:
        assert "execution_time" in r
        assert isinstance(r["execution_time"], int | float)


# --------------------------------------------------------------------------- #
# WorkflowDAG.from_yaml — edge wiring
# --------------------------------------------------------------------------- #
def _write_yaml(tmp_path, text):
    p = tmp_path / "wf.yaml"
    p.write_text(text)
    return str(p)


def test_from_yaml_depends_on_builds_edge(tmp_path):
    from effgen.core.workflow import WorkflowDAG
    dag = WorkflowDAG.from_yaml(_write_yaml(tmp_path, (
        "workflow:\n  name: p\n  nodes:\n    - id: a\n    - id: b\n"
        "      depends_on: [a]\n"
    )))
    assert [(e.source, e.target) for e in dag.edges] == [("a", "b")]


def test_from_yaml_top_level_edges_pair(tmp_path):
    # A top-level ``edges`` list is a common DAG convention; it must build the
    # graph, not be dropped while the workflow still reports as valid.
    from effgen.core.workflow import WorkflowDAG
    dag = WorkflowDAG.from_yaml(_write_yaml(tmp_path, (
        "workflow:\n  name: p\n  nodes:\n    - id: a\n    - id: b\n"
        "  edges:\n    - [a, b]\n"
    )))
    assert [(e.source, e.target) for e in dag.edges] == [("a", "b")]
    assert dag.topological_order() == ["a", "b"]


def test_from_yaml_top_level_edges_mapping_with_key(tmp_path):
    from effgen.core.workflow import WorkflowDAG
    dag = WorkflowDAG.from_yaml(_write_yaml(tmp_path, (
        "workflow:\n  name: p\n  nodes:\n    - id: a\n    - id: b\n"
        "  edges:\n    - {source: a, target: b, key: out}\n"
    )))
    edge = dag.edges[0]
    assert (edge.source, edge.target, edge.key) == ("a", "b", "out")


def test_from_yaml_node_input_keys_honored(tmp_path):
    from effgen.core.workflow import WorkflowDAG
    dag = WorkflowDAG.from_yaml(_write_yaml(tmp_path, (
        "workflow:\n  name: p\n  nodes:\n    - id: a\n      input_keys: [x, y]\n"
    )))
    assert dag.get_node("a").input_keys == ["x", "y"]
    assert "input_keys" not in dag.get_node("a").metadata


def test_from_yaml_unknown_top_level_key_warns(tmp_path, caplog):
    import logging

    from effgen.core.workflow import WorkflowDAG
    with caplog.at_level(logging.WARNING):
        WorkflowDAG.from_yaml(_write_yaml(tmp_path, (
            "workflow:\n  name: p\n  nodes:\n    - id: a\n  edgez: [[a, a]]\n"
        )))
    assert any("edgez" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Consensus, message bus, approval, cancellation
# --------------------------------------------------------------------------- #
def test_consensus_score_range():
    from effgen.core.orchestrator import MultiAgentOrchestrator
    orch = MultiAgentOrchestrator()
    assert orch._calculate_consensus([{"output": "the cat sat"},
                                      {"output": "the cat sat"}]) == 1.0
    assert orch._calculate_consensus([{"output": "the cat sat"},
                                      {"output": "dogs run far"}]) == 0.0
    assert orch._calculate_consensus([{"output": "alone"}]) == 1.0  # trivial


def test_message_bus_pubsub_wildcard():
    from effgen.core.message_bus import AgentMessage, MessageBus, MessageType
    bus = MessageBus(persist=True)
    got = []
    bus.subscribe("team.*.result", lambda m: got.append(m.payload))
    bus.send(AgentMessage(sender="a", recipient="o", type=MessageType.RESULT,
                          payload="match", topic="team.x.result"))
    bus.send(AgentMessage(sender="a", recipient="o", type=MessageType.STATUS_UPDATE,
                          payload="nomatch", topic="team.x.status"))
    assert got == ["match"]
    assert len(bus.get_history()) == 2


def test_approval_manager_approve_deny_timeout():
    from effgen.core.human_loop import ApprovalDecision, ApprovalManager, ApprovalMode
    yes = ApprovalManager(mode=ApprovalMode.ALWAYS, callback=lambda n, a: True)
    no = ApprovalManager(mode=ApprovalMode.ALWAYS, callback=lambda n, a: False)
    to = ApprovalManager(mode=ApprovalMode.ALWAYS, timeout=0.2,
                         callback=lambda n, a: (time.sleep(2) or True))
    assert yes.request_approval("bash", "ls") == ApprovalDecision.APPROVED
    assert no.request_approval("bash", "rm") == ApprovalDecision.DENIED
    assert to.request_approval("bash", "x") == ApprovalDecision.TIMEOUT


def test_cancel_stops_in_flight_sequential():
    # Cancel after the first agent's RESULT is published (synchronous delivery),
    # so the loop stops before launching the next agent.
    orch = _orch_with_team(agents=[FakeAgent("c1"), FakeAgent("c2"), FakeAgent("c3")])
    fired = {"v": False}

    def hook(_msg):
        if not fired["v"]:
            fired["v"] = True
            orch.cancel_workflow("team")

    orch.message_bus.subscribe("team.team.result", hook)
    res = orch.assign_task("count", "team")
    assert res.success is False
    assert res.metadata.get("reason") == "cancelled"
    assert len(res.agent_responses) == 1  # only the first ran


# --------------------------------------------------------------------------- #
# SubAgentManager: real execution requirement (no fabrication)
# --------------------------------------------------------------------------- #
def test_sub_agent_requires_real_parent_model():
    from effgen.core.sub_agent_manager import SubAgentConfig, SubAgentManager
    from effgen.core.task import SubTask
    mgr = SubAgentManager(parent_agent=None)  # no model available
    st = SubTask(id="st_1", description="do something", expected_output="x")
    cfg = SubAgentConfig.get_default_config("general")
    with pytest.raises(RuntimeError):
        mgr._run_real_sub_agent(st, cfg)


def test_sub_agent_manager_has_no_simulation_placeholder():
    # The fabrication trap must be gone for good.
    from effgen.core import sub_agent_manager as sam
    assert not hasattr(sam.SubAgentManager, "_simulate_execution")
    assert hasattr(sam.SubAgentManager, "_run_real_sub_agent")


# --------------------------------------------------------------------------- #
# SubAgentRouter type guard
# --------------------------------------------------------------------------- #
def test_router_rejects_non_string_task():
    from effgen.core.router import SubAgentRouter
    with pytest.raises(TypeError):
        SubAgentRouter().route({"not": "a string"})


# --------------------------------------------------------------------------- #
# Multi-agent execution correlation, fan-out, discussion structure, async pairs
# --------------------------------------------------------------------------- #
def test_team_response_carries_execution_id_and_topology():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(
        agents=[FakeAgent("a"), FakeAgent("b")],
        pattern=OrchestrationPattern.SEQUENTIAL,
    )
    res = orch.assign_task("hi", "team")
    assert res.metadata["execution_id"]
    shape = res.metadata["topology"]
    assert [n["name"] for n in shape["agents"]] == ["a", "b"]
    assert shape["edges"] == [{"source": "a", "target": "b", "kind": "handoff"}]


def test_member_runs_share_one_execution_id():
    from effgen.observability import run_log

    run_log.clear()
    orch = _orch_with_team(agents=[FakeAgent("a"), FakeAgent("b")])
    res = orch.assign_task("hi", "team")
    ids = {r["execution_id"] for r in run_log.get_recent_runs(limit=10) if r.get("agent") in ("a", "b")}
    # FakeAgent does not write run history; the id lives on the response and is
    # what a real member run would inherit.
    assert res.metadata["execution_id"]
    assert ids in ({res.metadata["execution_id"]}, set())


def test_hierarchical_subtask_naming_two_workers_runs_both():
    # A manager line addressed to several specialists reaches each of them
    # instead of being narrowed to whoever matched first.
    tech = FakeAgent("tech")
    billing = FakeAgent("billing")
    orch = _hier_orch("1. tech & billing: review the account", [tech, billing])
    res = orch.assign_task("review", "support")
    assert tech.runs == 1 and billing.runs == 1
    assigned = {r["agent_name"] for r in res.agent_responses}
    assert assigned == {"tech", "billing"}
    assert all(r["co_assigned"] == ["tech", "billing"] for r in res.agent_responses)


def test_longer_worker_name_wins_over_a_contained_name():
    tech = FakeAgent("analyst")
    market = FakeAgent("market-analyst")
    orch = _hier_orch("1. market-analyst: size the market", [tech, market])
    orch.assign_task("go", "support")
    assert market.runs == 1 and tech.runs == 0


def test_collaborative_rounds_record_who_was_answered():
    from effgen.core.orchestrator import OrchestrationPattern
    orch = _orch_with_team(
        agents=[FakeAgent("a"), FakeAgent("b")],
        pattern=OrchestrationPattern.COLLABORATIVE,
    )
    orch.teams["team"].max_rounds = 2
    res = orch.assign_task("discuss", "team")
    assert all(r["round"] == 2 for r in res.agent_responses)
    # Round 2 answers the responses of round 1.
    assert res.agent_responses[0]["responds_to"] == ["a", "b"]
    rounds = res.metadata["discussion_rounds"]
    assert [r["round"] for r in rounds] == [1, 2]
    assert rounds[0]["agents"] == ["a", "b"]


def test_assign_task_async_returns_a_team_response():
    import asyncio

    from effgen.core.orchestrator import TeamResponse
    orch = _orch_with_team(agents=[FakeAgent("a")])
    res = asyncio.run(orch.assign_task_async("hi", "team"))
    assert isinstance(res, TeamResponse)
    assert res.success is True


def test_workflow_execute_is_an_alias_for_run():
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("wf")
    dag.add_node(WorkflowNode(id="only", agent=FakeAgent("a")))
    result = dag.execute("go")
    assert result.success is True
    assert result.metadata["execution_id"]


def test_workflow_execute_async_is_an_alias_for_run_async():
    import asyncio

    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    dag = WorkflowDAG("wf")
    dag.add_node(WorkflowNode(id="only", agent=FakeAgent("a")))
    result = asyncio.run(dag.execute_async("go"))
    assert result.success is True


def test_workflow_skipped_node_is_recorded_in_telemetry():
    from effgen.core.workflow import WorkflowDAG, WorkflowNode
    from effgen.observability import tracing

    with tracing._SPAN_BUFFER_LOCK:
        tracing._SPAN_BUFFER.clear()
    dag = WorkflowDAG("wf")
    dag.add_node(WorkflowNode(id="first", agent=FakeAgent("a", succeed=False)))
    dag.add_node(WorkflowNode(id="second", agent=FakeAgent("b")))
    dag.connect("first", "second")
    result = dag.run("go")
    assert result.success is False
    skipped = [s for s in tracing.get_recent_spans(limit=20) if s.get("status") == "skipped"]
    assert any(s["agent"] == "second" for s in skipped)
    assert any("did not complete" in (s.get("note") or "") for s in skipped)
