"""
Tests for Human-in-the-Loop & Approval Workflows.

Covers human_loop, clarification, and feedback modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from effgen.core.clarification import (
    ClarificationDetector,
    ClarificationRequest,
)
from effgen.core.feedback import (
    FeedbackCollector,
    FeedbackType,
)
from effgen.core.human_loop import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalMode,
    HumanApproval,
    HumanChoice,
    HumanInput,
    is_tool_dangerous,
)

# ============================================================
# HumanApproval tests
# ============================================================

class TestHumanApproval:
    def test_approval_with_callback_approved(self):
        approval = HumanApproval(tool_name="calculator", tool_args='{"expr":"1+1"}')
        decision = approval.request(callback=lambda name, args: True)
        assert decision == ApprovalDecision.APPROVED

    def test_approval_with_callback_denied(self):
        approval = HumanApproval(tool_name="calculator", tool_args='{"expr":"1+1"}')
        decision = approval.request(callback=lambda name, args: False)
        assert decision == ApprovalDecision.DENIED

    def test_approval_no_callback_returns_default(self):
        approval = HumanApproval(
            tool_name="bash", tool_args="rm -rf /",
            default_decision=ApprovalDecision.DENIED,
        )
        assert approval.request(callback=None) == ApprovalDecision.DENIED

    def test_approval_no_callback_approved_default(self):
        approval = HumanApproval(
            tool_name="safe_tool", tool_args="x",
            default_decision=ApprovalDecision.APPROVED,
        )
        assert approval.request(callback=None) == ApprovalDecision.APPROVED

    def test_approval_callback_error_returns_default(self):
        def bad_callback(name, args):
            raise RuntimeError("broken")

        approval = HumanApproval(
            tool_name="test", tool_args="",
            default_decision=ApprovalDecision.DENIED,
        )
        assert approval.request(callback=bad_callback) == ApprovalDecision.DENIED

    def test_approval_timeout(self):
        import time

        def slow_callback(name, args):
            time.sleep(5)
            return True

        approval = HumanApproval(
            tool_name="test", tool_args="",
            timeout=0.1,
            default_decision=ApprovalDecision.DENIED,
        )
        decision = approval.request(callback=slow_callback)
        assert decision == ApprovalDecision.TIMEOUT


# ============================================================
# HumanInput tests
# ============================================================

class TestHumanInput:
    def test_input_with_callback(self):
        hi = HumanInput(prompt="Enter name")
        result = hi.request(callback=lambda p: "Alice")
        assert result == "Alice"

    def test_input_no_callback_returns_default(self):
        hi = HumanInput(prompt="Enter name", default="Bob")
        assert hi.request(callback=None) == "Bob"

    def test_input_callback_error_returns_default(self):
        hi = HumanInput(prompt="x", default="fallback")
        result = hi.request(callback=lambda p: (_ for _ in ()).throw(RuntimeError("fail")))
        assert result == "fallback"

    def test_input_timeout(self):
        import time
        hi = HumanInput(prompt="x", timeout=0.1, default="timed_out")
        result = hi.request(callback=lambda p: time.sleep(5) or "late")
        assert result == "timed_out"


# ============================================================
# HumanChoice tests
# ============================================================

class TestHumanChoice:
    def test_choice_with_callback(self):
        hc = HumanChoice(prompt="Pick", options=["A", "B", "C"])
        idx = hc.request(callback=lambda p, opts: 2)
        assert idx == 2

    def test_choice_no_callback_returns_default(self):
        hc = HumanChoice(prompt="Pick", options=["A", "B"], default=1)
        assert hc.request(callback=None) == 1

    def test_choice_invalid_index_returns_default(self):
        hc = HumanChoice(prompt="Pick", options=["A", "B"], default=0)
        assert hc.request(callback=lambda p, opts: 99) == 0

    def test_choice_empty_options(self):
        hc = HumanChoice(prompt="Pick", options=[])
        assert hc.request(callback=lambda p, opts: 0) == -1

    def test_choice_timeout(self):
        import time
        hc = HumanChoice(prompt="Pick", options=["A", "B"], timeout=0.1, default=1)
        idx = hc.request(callback=lambda p, opts: time.sleep(5) or 0)
        assert idx == 1


# ============================================================
# ApprovalManager tests
# ============================================================

class TestApprovalManager:
    def test_never_mode_skips_approval(self):
        mgr = ApprovalManager(mode=ApprovalMode.NEVER, callback=lambda n, a: False)
        assert not mgr.should_request_approval("bash", requires_approval=True)

    def test_always_mode_requires_approval(self):
        mgr = ApprovalManager(mode=ApprovalMode.ALWAYS)
        assert mgr.should_request_approval("calculator")

    def test_first_time_mode_remembers(self):
        approvals = []
        mgr = ApprovalManager(
            mode=ApprovalMode.FIRST_TIME,
            callback=lambda n, a: (approvals.append(n) or True),
        )
        assert mgr.should_request_approval("calc")
        mgr.request_approval("calc", "1+1")
        assert not mgr.should_request_approval("calc")
        assert approvals == ["calc"]

    def test_dangerous_only_mode(self):
        mgr = ApprovalManager(mode=ApprovalMode.DANGEROUS_ONLY)
        assert mgr.should_request_approval("bash_executor")
        assert not mgr.should_request_approval("calculator")
        assert mgr.should_request_approval("safe_tool", requires_approval=True)

    def test_request_approval_denied(self):
        mgr = ApprovalManager(
            mode=ApprovalMode.ALWAYS,
            callback=lambda n, a: False,
        )
        decision = mgr.request_approval("tool", "args")
        assert decision == ApprovalDecision.DENIED
        # Denied tools should not be remembered as approved
        assert "tool" not in mgr._approved_tools

    def test_reset(self):
        mgr = ApprovalManager(mode=ApprovalMode.FIRST_TIME, callback=lambda n, a: True)
        mgr.request_approval("t", "")
        assert "t" in mgr._approved_tools
        mgr.reset()
        assert "t" not in mgr._approved_tools


class TestIsDangerous:
    def test_dangerous_tools(self):
        assert is_tool_dangerous("bash")
        assert is_tool_dangerous("code_executor")
        assert is_tool_dangerous("shell_runner")
        assert is_tool_dangerous("file_delete_tool")

    def test_safe_tools(self):
        assert not is_tool_dangerous("calculator")
        assert not is_tool_dangerous("web_search")
        assert not is_tool_dangerous("json_tool")


# ============================================================
# ClarificationRequest tests
# ============================================================

class TestClarificationRequest:
    def test_ask_with_options(self):
        req = ClarificationRequest(
            question="Which tool?",
            options=["A", "B"],
        )
        result = req.ask(choice_callback=lambda q, opts: 1)
        assert result == "B"

    def test_ask_with_options_no_callback(self):
        req = ClarificationRequest(question="Which?", options=["X", "Y"], default=1)
        assert req.ask() == "Y"

    def test_ask_free_text(self):
        req = ClarificationRequest(question="What do you mean?")
        result = req.ask(input_callback=lambda q: "I want to search")
        assert result == "I want to search"

    def test_ask_free_text_no_callback(self):
        req = ClarificationRequest(question="What?", default="nothing")
        assert req.ask() == "nothing"


# ============================================================
# ClarificationDetector tests
# ============================================================

class TestClarificationDetector:
    def test_short_query_detected(self):
        detector = ClarificationDetector()
        req = detector.detect_ambiguity("do it")
        assert req is not None
        assert "short" in req.context

    def test_ambiguous_words_detected(self):
        detector = ClarificationDetector()
        req = detector.detect_ambiguity("do something with whatever data")
        assert req is not None
        assert "vague" in req.context

    def test_clear_query_no_ambiguity(self):
        detector = ClarificationDetector()
        req = detector.detect_ambiguity("Calculate the square root of 144 using the calculator tool")
        assert req is None

    def test_multiple_tool_matches(self):
        class FakeTool:
            description = "search the web for information"
        tools = {
            f"search_tool_{i}": FakeTool() for i in range(4)
        }
        detector = ClarificationDetector()
        req = detector.detect_ambiguity("search for data", tools)
        assert req is not None
        assert req.options  # Should have tool options

    def test_request_clarification_returns_none_for_clear(self):
        detector = ClarificationDetector()
        result = detector.request_clarification(
            "Calculate the factorial of 10 using the math tool"
        )
        assert result is None

    def test_request_clarification_returns_response(self):
        detector = ClarificationDetector(
            input_callback=lambda q: "I want to add numbers"
        )
        result = detector.request_clarification("do it")
        assert result == "I want to add numbers"


# ============================================================
# FeedbackCollector tests
# ============================================================

class TestFeedbackCollector:
    def test_thumbs_up(self):
        fc = FeedbackCollector(agent_name="test_agent")
        entry = fc.thumbs("resp1", thumbs_up=True, query="Hello")
        assert entry.feedback_type == FeedbackType.THUMBS
        assert entry.value is True
        assert entry.response_id == "resp1"
        assert entry.agent_name == "test_agent"

    def test_thumbs_down(self):
        fc = FeedbackCollector()
        entry = fc.thumbs("resp2", thumbs_up=False)
        assert entry.value is False

    def test_rating_valid(self):
        fc = FeedbackCollector()
        entry = fc.rate("resp3", rating=4)
        assert entry.feedback_type == FeedbackType.RATING
        assert entry.value == 4

    def test_rating_invalid(self):
        fc = FeedbackCollector()
        with pytest.raises(ValueError, match="Rating must be 1-5"):
            fc.rate("resp", rating=0)
        with pytest.raises(ValueError, match="Rating must be 1-5"):
            fc.rate("resp", rating=6)

    def test_comment(self):
        fc = FeedbackCollector()
        entry = fc.comment("resp4", text="Great response!")
        assert entry.feedback_type == FeedbackType.TEXT
        assert entry.value == "Great response!"

    def test_entries_list(self):
        fc = FeedbackCollector()
        fc.thumbs("r1", True)
        fc.rate("r2", 5)
        fc.comment("r3", "nice")
        assert len(fc.entries) == 3

    def test_get_by_response(self):
        fc = FeedbackCollector()
        fc.thumbs("r1", True)
        fc.rate("r1", 5)
        fc.thumbs("r2", False)
        assert len(fc.get_by_response("r1")) == 2
        assert len(fc.get_by_response("r2")) == 1

    def test_summary(self):
        fc = FeedbackCollector()
        fc.thumbs("r1", True)
        fc.thumbs("r2", True)
        fc.thumbs("r3", False)
        fc.rate("r4", 3)
        fc.rate("r5", 5)
        fc.comment("r6", "ok")

        s = fc.summary()
        assert s["total"] == 6
        assert s["thumbs_up"] == 2
        assert s["thumbs_down"] == 1
        assert s["average_rating"] == 4.0
        assert s["total_ratings"] == 2
        assert s["total_comments"] == 1

    def test_export_jsonl(self):
        fc = FeedbackCollector(agent_name="test")
        fc.thumbs("r1", True, query="hi")
        fc.rate("r2", 4)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="r") as f:
            path = f.name

        count = fc.export_jsonl(path)
        assert count == 2

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["response_id"] == "r1"
        assert entry1["feedback_type"] == "thumbs"
        assert entry1["value"] is True

        entry2 = json.loads(lines[1])
        assert entry2["response_id"] == "r2"
        assert entry2["feedback_type"] == "rating"
        assert entry2["value"] == 4

        Path(path).unlink()

    def test_clear(self):
        fc = FeedbackCollector()
        fc.thumbs("r1", True)
        fc.clear()
        assert len(fc.entries) == 0

    def test_to_dict(self):
        fc = FeedbackCollector()
        entry = fc.thumbs("r1", True, metadata={"source": "cli"})
        d = entry.to_dict()
        assert d["response_id"] == "r1"
        assert d["feedback_type"] == "thumbs"
        assert d["metadata"] == {"source": "cli"}


# ============================================================
# AgentConfig integration test
# ============================================================

class TestAgentConfigApproval:
    """Test that AgentConfig accepts approval fields."""

    def test_agent_config_approval_fields(self):
        from effgen.core.agent import AgentConfig

        approvals = []
        def my_callback(name, args):
            approvals.append(name)
            return True

        config = AgentConfig(
            name="test",
            model="dummy",
            approval_callback=my_callback,
            approval_mode="always",
            approval_timeout=30.0,
        )
        assert config.approval_callback is my_callback
        assert config.approval_mode == "always"
        assert config.approval_timeout == 30.0

    def test_agent_config_defaults(self):
        from effgen.core.agent import AgentConfig
        config = AgentConfig(name="test", model="dummy")
        assert config.approval_callback is None
        assert config.approval_mode == "never"
        assert config.approval_timeout == 0.0
        assert config.clarification_callback is None
        assert config.input_callback is None


class TestToolMetadataApproval:
    """Test that ToolMetadata accepts requires_approval."""

    def test_requires_approval_default(self):
        from effgen.tools.base_tool import ToolCategory, ToolMetadata
        meta = ToolMetadata(name="test", description="test", category=ToolCategory.COMPUTATION)
        assert meta.requires_approval is False

    def test_requires_approval_set(self):
        from effgen.tools.base_tool import ToolCategory, ToolMetadata
        meta = ToolMetadata(
            name="bash", description="run shell",
            category=ToolCategory.SYSTEM, requires_approval=True,
        )
        assert meta.requires_approval is True
