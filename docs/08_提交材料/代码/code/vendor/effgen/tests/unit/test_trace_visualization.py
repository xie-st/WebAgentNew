"""Trace/workflow visualization data + renderers.

Covers the data an agent already records being made drawable:
- the execution tree finalizes (terminal status, real duration, populated
  children) and event ids stay unique so parent/child links are unambiguous;
- the terminal workflow diagram groups nodes by level, draws edges, and
  annotates per-node status/duration/cost;
- buffered spans carry a run correlation id and a start offset so a consumer
  can lay them out on a per-run timeline.
"""

from __future__ import annotations

import time


# --------------------------------------------------------------------------- #
# Execution tree finalization
# --------------------------------------------------------------------------- #
def _run_events(tracker):
    from effgen.core.execution_tracker import EventType, ExecutionEvent

    base = time.time()
    tracker.track_event(ExecutionEvent(
        type=EventType.TASK_START, agent_id="a", timestamp=base,
        data={"task": "compute"},
    ))
    tracker.track_event(ExecutionEvent(
        type=EventType.REASONING_STEP, agent_id="a", timestamp=base + 0.1,
    ))
    tracker.track_event(ExecutionEvent(
        type=EventType.TOOL_CALL_START, agent_id="a", timestamp=base + 1.0,
        data={"tool_name": "calculator", "tool_input": "1+1"},
    ))
    tracker.track_event(ExecutionEvent(
        type=EventType.TOOL_CALL_COMPLETE, agent_id="a", timestamp=base + 1.2,
        data={"tool_name": "calculator", "result": "2"},
    ))
    tracker.track_event(ExecutionEvent(
        type=EventType.TASK_COMPLETE, agent_id="a", timestamp=base + 2.0,
        data={"execution_time": 2.0, "tokens_used": 30, "tool_calls": 1},
    ))


def test_execution_tree_finalizes_with_children():
    from effgen.core.execution_tracker import ExecutionTracker

    tracker = ExecutionTracker()
    _run_events(tracker)
    tree = tracker.generate_execution_tree()

    assert tree["status"] == "completed"          # not stuck at "running"
    assert tree["completed_at"] is not None
    assert abs(tree["duration"] - 2.0) < 0.01      # real duration, not now-start
    # The tool step hangs under the task, not lost at the top level.
    assert len(tree["children"]) == 1
    child = tree["children"][0]
    assert child["node_type"] == "tool"
    assert child["status"] == "completed"
    assert child["metadata"].get("result") == "2"


def test_event_ids_are_unique_even_in_the_same_millisecond():
    from effgen.core.execution_tracker import EventType, ExecutionEvent

    ids = {ExecutionEvent(type=EventType.REASONING_STEP).event_id for _ in range(500)}
    assert len(ids) == 500


def test_flat_trace_carries_parent_links():
    from effgen.core.execution_tracker import ExecutionTracker

    tracker = ExecutionTracker()
    _run_events(tracker)
    trace = tracker.get_trace()
    root = trace[0]
    assert root["type"] == "task_start"
    assert root["parent_event_id"] is None
    # Every non-root event points at a real event id (the flat list is a tree).
    tool = next(e for e in trace if e["type"] == "tool_call_start")
    assert tool["parent_event_id"] == root["event_id"]


def test_failed_task_marks_tree_failed():
    from effgen.core.execution_tracker import EventType, ExecutionEvent, ExecutionTracker

    tracker = ExecutionTracker()
    base = time.time()
    tracker.track_event(ExecutionEvent(type=EventType.TASK_START, timestamp=base, data={"task": "x"}))
    tracker.track_event(ExecutionEvent(type=EventType.TASK_FAILED, timestamp=base + 0.5, data={"error": "boom"}))
    tree = tracker.generate_execution_tree()
    assert tree["status"] == "failed"
    assert tree["completed_at"] is not None


# --------------------------------------------------------------------------- #
# Terminal workflow diagram
# --------------------------------------------------------------------------- #
def test_workflow_diagram_lines_show_levels_edges_and_metrics():
    from effgen.ui.workflow_viz import workflow_diagram_lines

    node_ids = ["research", "pros", "cons", "summary"]
    edges = [
        {"source": "research", "target": "pros"},
        {"source": "research", "target": "cons"},
        {"source": "pros", "target": "summary"},
        {"source": "cons", "target": "summary"},
    ]
    levels = [["research"], ["pros", "cons"], ["summary"]]
    node_results = [
        {"id": "research", "status": "completed", "execution_time": 0.29,
         "metadata": {"cost_usd": 0.000009, "tokens_used": 130}},
        {"id": "pros", "status": "completed", "execution_time": 0.15,
         "metadata": {"cost_usd": 0.000008, "tokens_used": 155}},
        {"id": "cons", "status": "completed", "execution_time": 0.15,
         "metadata": {"cost_usd": 0.000008, "tokens_used": 153}},
        {"id": "summary", "status": "completed", "execution_time": 0.17,
         "metadata": {"cost_usd": 0.000007, "tokens_used": 128}},
    ]
    lines = workflow_diagram_lines("fanout", node_ids, edges, levels, node_results)
    text = "\n".join(t for _s, t in lines)
    assert "Level 0" in text and "Level 1" in text and "Level 2" in text
    assert "parallel" in text            # pros/cons are drawn on the same level
    assert "▶ pros" in text              # an edge is drawn
    assert "completed" in text and "130 tok" in text
    assert "$" in text                   # per-node cost annotated


def test_workflow_diagram_pending_without_results():
    from effgen.ui.workflow_viz import workflow_diagram_lines

    lines = workflow_diagram_lines(
        "p", ["a", "b"], [{"source": "a", "target": "b"}], [["a"], ["b"]],
    )
    text = "\n".join(t for _s, t in lines)
    assert "pending" in text
    assert "▶ b" in text


# --------------------------------------------------------------------------- #
# Span enrichment for the per-run waterfall
# --------------------------------------------------------------------------- #
def test_buffered_spans_carry_run_id_and_offset():
    from effgen.observability import tracing as T

    with T._SPAN_BUFFER_LOCK:
        T._SPAN_BUFFER.clear()
    with T.start_agent_run(preset="p", task="t", run_id="run-42"):
        with T.start_model_call("openai", "gpt-5-nano"):
            pass
        with T.start_tool_call("calculator", "1+1"):
            pass
    spans = T.get_recent_spans()
    assert spans, "expected buffered spans"
    for s in spans:
        assert s["run_id"] == "run-42"
        assert "offset_ms" in s and s["offset_ms"] >= 0.0
    # The agent.run span starts the run, so its offset is 0.
    run_span = next(s for s in spans if "agent.run" in s["name"])
    assert run_span["offset_ms"] == 0.0


def test_span_without_run_context_has_null_run_id():
    from effgen.observability import tracing as T

    with T._SPAN_BUFFER_LOCK:
        T._SPAN_BUFFER.clear()
    # A model call outside any agent run still records, just without correlation.
    with T.start_model_call("openai", "gpt-5-nano"):
        pass
    spans = T.get_recent_spans()
    assert spans and spans[0]["run_id"] is None
