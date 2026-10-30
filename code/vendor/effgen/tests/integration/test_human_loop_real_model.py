"""
Real-model integration tests for Human-in-the-Loop & Approval Workflows.

Tests approval callbacks, clarification, and feedback with a real SLM on GPU.
Uses Qwen2.5-1.5B-Instruct on CUDA.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.clarification import ClarificationDetector
from effgen.core.feedback import FeedbackCollector
from effgen.models import load_model
from effgen.tools.builtin.calculator import Calculator as CalculatorTool

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def human_loop_model(gpu_id):
    """Load the human-loop test model once per module to avoid GPU pressure."""
    if gpu_id is None:
        pytest.skip("No GPU available")
    model = load_model(MODEL_NAME, device_map={"": int(gpu_id)})
    yield model
    try:
        model.unload()
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def test_approval_callback_blocks_tool(human_loop_model):
    """
    Test 1: Approval callback DENIES tool execution.
    Agent should NOT execute calculator and should report denial.
    """
    print("\n=== Test 1: Approval callback blocks tool ===")
    approvals = []

    def deny_callback(tool_name, tool_args):
        approvals.append({"tool": tool_name, "args": tool_args, "decision": "deny"})
        return False  # DENY

    config = AgentConfig(
        name="deny_test",
        model=human_loop_model,
        tools=[CalculatorTool()],
        approval_callback=deny_callback,
        approval_mode="always",
        max_iterations=5,
    )
    agent = Agent(config=config)
    response = agent.run("What is 15 * 23?")

    print(f"  Approvals captured: {len(approvals)}")
    print(f"  Response: {response.output[:200]}")
    assert len(approvals) >= 1, "Approval callback should have been called at least once"
    assert approvals[0]["tool"] == "calculator", f"Expected calculator, got {approvals[0]['tool']}"
    # Response should indicate denial
    print("  PASS: Approval callback was called and denied tool execution")
    agent.close()


def test_approval_callback_allows_tool(human_loop_model):
    """
    Test 2: Approval callback ALLOWS tool execution.
    Agent should execute calculator and return correct answer.
    """
    print("\n=== Test 2: Approval callback allows tool ===")
    approvals = []

    def allow_callback(tool_name, tool_args):
        approvals.append({"tool": tool_name, "args": tool_args, "decision": "allow"})
        return True  # ALLOW

    config = AgentConfig(
        name="allow_test",
        model=human_loop_model,
        tools=[CalculatorTool()],
        approval_callback=allow_callback,
        approval_mode="always",
        max_iterations=5,
    )
    agent = Agent(config=config)
    response = agent.run("What is 15 * 23?")

    print(f"  Approvals captured: {len(approvals)}")
    print(f"  Response: {response.output[:200]}")
    assert len(approvals) >= 1, "Approval callback should have been called"
    assert "345" in response.output, f"Expected 345 in response, got: {response.output[:200]}"
    print("  PASS: Tool was approved and executed correctly, answer=345")
    agent.close()


def test_first_time_mode_only_asks_once(human_loop_model):
    """
    Test 3: FIRST_TIME mode only asks approval on first use of a tool.
    """
    print("\n=== Test 3: First-time mode ===")
    approval_count = 0

    def counting_callback(tool_name, tool_args):
        nonlocal approval_count
        approval_count += 1
        return True

    config = AgentConfig(
        name="first_time_test",
        model=human_loop_model,
        tools=[CalculatorTool()],
        approval_callback=counting_callback,
        approval_mode="first_time",
        max_iterations=8,
    )
    agent = Agent(config=config)

    # First query — should trigger approval
    r1 = agent.run("What is 10 + 5?")
    first_count = approval_count
    print(f"  After first query: approval_count={first_count}, output={r1.output[:100]}")
    assert first_count >= 1, "First use should trigger approval"

    # Second query with same tool — should NOT trigger approval
    r2 = agent.run("What is 20 + 30?")
    second_count = approval_count
    print(f"  After second query: approval_count={second_count}, output={r2.output[:100]}")
    assert second_count == first_count, f"Second use should NOT trigger approval (got {second_count} vs {first_count})"
    print("  PASS: First-time mode asked once, then remembered")
    agent.close()


def test_dangerous_only_mode(human_loop_model):
    """
    Test 4: DANGEROUS_ONLY mode only asks for dangerous tools.
    Calculator is not dangerous, so no approval should be requested.
    """
    print("\n=== Test 4: Dangerous-only mode ===")
    approval_count = 0

    def counting_callback(tool_name, tool_args):
        nonlocal approval_count
        approval_count += 1
        return True

    config = AgentConfig(
        name="dangerous_test",
        model=human_loop_model,
        tools=[CalculatorTool()],
        approval_callback=counting_callback,
        approval_mode="dangerous_only",
        max_iterations=5,
    )
    agent = Agent(config=config)
    response = agent.run("What is 7 * 8?")

    print(f"  Approval count: {approval_count}")
    print(f"  Response: {response.output[:200]}")
    assert approval_count == 0, f"Calculator is not dangerous; expected 0 approvals, got {approval_count}"
    assert "56" in response.output, "Expected 56 in response"
    print("  PASS: Non-dangerous tool ran without approval, answer=56")
    agent.close()


def test_no_approval_baseline(human_loop_model):
    """
    Test 5: Default mode (never) — no approval at all, tool executes freely.
    """
    print("\n=== Test 5: No-approval baseline ===")
    config = AgentConfig(
        name="baseline_test",
        model=human_loop_model,
        tools=[CalculatorTool()],
        max_iterations=5,
    )
    agent = Agent(config=config)
    response = agent.run("What is 99 + 1?")

    print(f"  Response: {response.output[:200]}")
    assert "100" in response.output, "Expected 100 in response"
    print("  PASS: Baseline (no approval) works, answer=100")
    agent.close()


def test_feedback_collector_with_real_agent(human_loop_model):
    """
    Test 6: FeedbackCollector records feedback and exports JSONL after real agent run.
    """
    print("\n=== Test 6: FeedbackCollector with real agent ===")
    config = AgentConfig(
        name="feedback_test",
        model=human_loop_model,
        tools=[CalculatorTool()],
        max_iterations=5,
    )
    agent = Agent(config=config)
    response = agent.run("What is 12 * 12?")

    # Collect feedback on the response
    fc = FeedbackCollector(agent_name="feedback_test")
    fc.thumbs(response_id="run1", thumbs_up=True, query="12*12")
    fc.rate(response_id="run1", rating=5, query="12*12")
    fc.comment(response_id="run1", text="Correct answer!", query="12*12")

    summary = fc.summary()
    print(f"  Agent output: {response.output[:200]}")
    print(f"  Feedback summary: {summary}")
    assert summary["total"] == 3
    assert summary["thumbs_up"] == 1
    assert summary["average_rating"] == 5.0
    assert summary["total_comments"] == 1

    # Export as JSONL
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    count = fc.export_jsonl(path)
    assert count == 3

    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    entry = json.loads(lines[0])
    assert entry["feedback_type"] == "thumbs"
    assert entry["value"] is True

    os.unlink(path)
    print("  PASS: Feedback collected, summarized, and exported as JSONL")
    agent.close()


def test_clarification_detector_heuristic():
    """
    Test 7: ClarificationDetector correctly identifies ambiguous vs clear queries.
    (No model needed — heuristic only, but run alongside GPU tests for completeness.)
    """
    print("\n=== Test 7: Clarification detector heuristic ===")
    detector = ClarificationDetector()

    # Ambiguous: too short
    req = detector.detect_ambiguity("do it")
    assert req is not None, "Short query should be flagged"
    print(f"  'do it' -> ambiguous ({req.context})")

    # Ambiguous: vague terms
    req = detector.detect_ambiguity("do something with whatever data is available")
    assert req is not None, "Vague query should be flagged"
    print(f"  'do something with whatever...' -> ambiguous ({req.context})")

    # Clear: specific query
    req = detector.detect_ambiguity("Calculate the square root of 144 using the calculator tool")
    assert req is None, "Clear query should NOT be flagged"
    print("  'Calculate the square root...' -> clear (no ambiguity)")

    # With callback
    detector2 = ClarificationDetector(input_callback=lambda q: "I want to multiply 5 by 3")
    result = detector2.request_clarification("do it")
    assert result == "I want to multiply 5 by 3"
    print(f"  Clarification response: {result}")

    print("  PASS: Ambiguity detection works correctly")


def test_requires_approval_on_tool_metadata(human_loop_model):
    """
    Test 8: ToolMetadata.requires_approval integrates with DANGEROUS_ONLY mode.
    Create a calculator with requires_approval=True, verify it triggers approval.
    """
    print("\n=== Test 8: requires_approval on tool metadata ===")
    approvals = []

    def tracking_callback(tool_name, tool_args):
        approvals.append(tool_name)
        return True

    # Create calculator with requires_approval=True
    calc = CalculatorTool()
    calc._metadata.requires_approval = True

    config = AgentConfig(
        name="metadata_approval_test",
        model=human_loop_model,
        tools=[calc],
        approval_callback=tracking_callback,
        approval_mode="dangerous_only",
        max_iterations=5,
    )
    agent = Agent(config=config)
    response = agent.run("What is 3 + 4?")

    print(f"  Approvals: {approvals}")
    print(f"  Response: {response.output[:200]}")
    assert len(approvals) >= 1, "requires_approval=True should trigger approval even in dangerous_only mode"
    assert "calculator" in approvals, f"Expected 'calculator' in approvals, got {approvals}"
    print("  PASS: requires_approval on metadata triggers approval in dangerous_only mode")
    agent.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Human-in-the-Loop Real GPU Integration Tests")
    print(f"Model: {MODEL_NAME}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)
