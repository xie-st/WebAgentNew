"""Unit tests for the effgen.eval module."""

from __future__ import annotations

import json

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.eval.comparison import ComparisonMatrix, ModelComparison, ModelScore
from effgen.eval.evaluator import (
    AgentEvaluator,
    Difficulty,
    EvalResult,
    ScoringMode,
    SuiteResults,
    TestCase,
    _compute_tool_accuracy,
    _score_contains,
    _score_exact_match,
    _score_regex,
)
from effgen.eval.regression import RegressionAlert, RegressionTracker
from effgen.eval.suites import (
    ConversationSuite,
    MathSuite,
    ReasoningSuite,
    SafetySuite,
    ToolUseSuite,
    get_suite,
    list_suites,
)
from tests.fixtures.mock_models import MockModel

# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------

class TestTestCase:
    def test_from_dict_basic(self):
        tc = TestCase.from_dict({
            "query": "What is 2+2?",
            "expected": "4",
            "tools": ["calculator"],
            "difficulty": "easy",
        })
        assert tc.query == "What is 2+2?"
        assert tc.expected_output == "4"
        assert tc.expected_tools == ["calculator"]
        assert tc.difficulty == Difficulty.EASY

    def test_from_dict_defaults(self):
        tc = TestCase.from_dict({"query": "hello"})
        assert tc.expected_output == ""
        assert tc.expected_tools == []
        assert tc.difficulty == Difficulty.MEDIUM
        assert tc.tags == []

    def test_from_dict_with_metadata(self):
        tc = TestCase.from_dict({
            "query": "test",
            "expected": "result",
            "metadata": {"turns": 3},
        })
        assert tc.metadata["turns"] == 3

    @pytest.mark.parametrize("key", ["query", "input", "prompt", "question"])
    def test_from_dict_accepts_query_aliases(self, key):
        # A custom .jsonl may use any of the documented field names for the
        # prompt; all resolve to ``query``.
        tc = TestCase.from_dict({key: "What is 2+2?", "expected": "4"})
        assert tc.query == "What is 2+2?"
        assert tc.expected_output == "4"

    def test_from_dict_missing_query_names_the_field(self):
        # No recognized field -> an actionable ValueError that names the field
        # it needs and the keys it saw, not a raw KeyError('query').
        with pytest.raises(ValueError) as exc:
            TestCase.from_dict({"foo": "bar"})
        msg = str(exc.value)
        assert "query" in msg
        assert "input/prompt/question" in msg
        assert "foo" in msg


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

class TestScoring:
    def test_exact_match(self):
        assert _score_exact_match("hello", "hello") == 1.0
        assert _score_exact_match("Hello", "hello") == 1.0  # case insensitive
        assert _score_exact_match("hello", "world") == 0.0

    def test_contains(self):
        assert _score_contains("world", "hello world") == 1.0
        assert _score_contains("xyz", "hello world") == 0.0
        assert _score_contains("WORLD", "hello world") == 1.0  # case insensitive

    def test_regex(self):
        assert _score_regex(r"\d+", "the answer is 42") == 1.0
        assert _score_regex(r"^\d+$", "not a number") == 0.0
        assert _score_regex(r"[invalid", "test") == 0.0  # bad regex

    def test_semantic_similarity_falls_back_to_contains_without_dependency(self, monkeypatch):
        """Without sentence-transformers, scoring must still work — via `contains`
        — and report that it did, so a caller can tell the score came from a
        different metric than requested."""
        import builtins

        from effgen.eval.evaluator import _score_semantic_similarity

        real_import = builtins.__import__

        def _no_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated missing optional dependency")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_sentence_transformers)
        score, used_fallback = _score_semantic_similarity("hello world", "hello world")
        assert used_fallback is True
        assert score == 1.0  # contains scoring on an exact substring match

    def test_semantic_similarity_loads_embedding_model_once_per_process(self, monkeypatch):
        """A multi-case suite must reuse one loaded model instead of paying the
        disk-load cost (and printing a progress bar) on every single case."""
        from effgen.eval import evaluator as evaluator_mod
        from effgen.eval.evaluator import _score_semantic_similarity

        monkeypatch.setattr(evaluator_mod, "_SEMANTIC_MODEL_CACHE", {})
        construct_calls = []

        class _FakeModel:
            def __init__(self, name):
                construct_calls.append(name)

            def encode(self, texts, convert_to_tensor=True):
                return [[1.0], [1.0]]

        class _FakeUtil:
            @staticmethod
            def cos_sim(a, b):
                return [[1.0]]

        import sys
        fake_st = type(sys)("sentence_transformers")
        fake_st.SentenceTransformer = _FakeModel
        fake_st.util = _FakeUtil
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
        monkeypatch.setitem(sys.modules, "sentence_transformers.util", _FakeUtil)

        for _ in range(5):
            score, used_fallback = _score_semantic_similarity("a", "a")
            assert used_fallback is False
            assert score == 1.0
        assert construct_calls == ["all-MiniLM-L6-v2"]  # constructed exactly once

    def test_tool_accuracy_all_match(self):
        assert _compute_tool_accuracy(["calc", "json"], ["calc", "json"]) == 1.0

    def test_tool_accuracy_partial(self):
        assert _compute_tool_accuracy(["calc", "json"], ["calc"]) == 0.5

    def test_tool_accuracy_no_expected(self):
        assert _compute_tool_accuracy([], ["calc"]) == 1.0

    def test_tool_accuracy_none_match(self):
        assert _compute_tool_accuracy(["calc"], ["json"]) == 0.0


# ---------------------------------------------------------------------------
# EvalResult & SuiteResults
# ---------------------------------------------------------------------------

class TestTestCaseAliases:
    def test_expected_is_alias_for_expected_output(self):
        # The natural ``expected=`` keyword must populate expected_output rather
        # than raising a raw "unexpected keyword argument" TypeError.
        tc = TestCase(query="2+2?", expected="4")
        assert tc.expected_output == "4"

    def test_expected_output_takes_precedence_over_expected_alias(self):
        tc = TestCase(query="x", expected_output="canonical", expected="alias")
        assert tc.expected_output == "canonical"

    def test_input_alias_still_works(self):
        tc = TestCase(input="hello", expected="hi")
        assert tc.query == "hello" and tc.expected_output == "hi"


class TestEvalResult:
    def test_defaults(self):
        tc = TestCase(query="test")
        r = EvalResult(test_case=tc)
        assert r.score == 0.0
        assert r.passed is False
        assert r.latency == 0.0
        assert r.cost_usd is None


class TestSuiteResults:
    def test_summary(self):
        tc1 = TestCase(query="q1", difficulty=Difficulty.EASY)
        tc2 = TestCase(query="q2", difficulty=Difficulty.HARD)
        results = SuiteResults(
            suite_name="test",
            results=[
                EvalResult(test_case=tc1, score=1.0, passed=True, latency=0.1, tokens_used=10),
                EvalResult(test_case=tc2, score=0.0, passed=False, latency=0.2, tokens_used=20),
            ],
            accuracy=0.5,
            avg_latency=0.15,
            total_tokens=30,
            avg_tool_accuracy=1.0,
        )
        s = results.summary()
        assert s["suite"] == "test"
        assert s["total"] == 2
        assert s["passed"] == 1
        assert s["accuracy"] == 0.5
        assert "easy" in s["by_difficulty"]
        assert "hard" in s["by_difficulty"]
        assert s["total_cost_usd"] is None

    def test_summary_reports_total_cost_when_known(self):
        tc = TestCase(query="q1")
        results = SuiteResults(
            suite_name="test",
            results=[EvalResult(test_case=tc, score=1.0, passed=True, cost_usd=0.001234567)],
            total_cost_usd=0.001234567,
        )
        assert results.summary()["total_cost_usd"] == round(0.001234567, 8)

    def test_to_json(self):
        results = SuiteResults(suite_name="json_test", accuracy=0.75)
        j = json.loads(results.to_json())
        assert j["suite"] == "json_test"

    def test_summary_results_array_carries_per_case_detail(self):
        # A CI job that only captures `--json`/`-o` must be able to see which
        # case failed and why without also scraping the human-readable
        # terminal render.
        tc1 = TestCase(query="what is 2+2", expected_output="4", difficulty=Difficulty.EASY)
        tc2 = TestCase(query="what is 3+3", expected_output="6", difficulty=Difficulty.HARD)
        results = SuiteResults(
            suite_name="test",
            results=[
                EvalResult(
                    test_case=tc1, agent_output="4", score=1.0, passed=True,
                    latency=0.1, tokens_used=10, cost_usd=0.0001,
                    tools_called=["calculator"],
                ),
                EvalResult(
                    test_case=tc2, agent_output="7", score=0.0, passed=False,
                    latency=0.2, tokens_used=20,
                    details={"scoring_fallback": "contains"},
                ),
            ],
        )
        s = results.summary()
        assert "results" in s
        assert len(s["results"]) == 2
        first, second = s["results"]
        assert first["query"] == "what is 2+2"
        assert first["expected_output"] == "4"
        assert first["agent_output"] == "4"
        assert first["passed"] is True
        assert first["cost_usd"] == 0.0001
        assert first["tools_called"] == ["calculator"]
        assert second["query"] == "what is 3+3"
        assert second["agent_output"] == "7"
        assert second["passed"] is False
        assert second["cost_usd"] is None
        assert second["details"]["scoring_fallback"] == "contains"
        # Round-trips through real JSON (what a CI job actually captures).
        parsed = json.loads(results.to_json())
        assert parsed["results"][1]["agent_output"] == "7"

    def test_to_json_emits_raw_utf8_not_escaped(self):
        # Non-Latin metadata (e.g. a localized suite label) stays readable
        # in the piped/CI-gate JSON instead of turning into \uXXXX escapes.
        results = SuiteResults(suite_name="json_test", metadata={"note": "翻訳テスト"})
        raw = results.to_json()
        assert "翻訳テスト" in raw
        assert "\\u" not in raw
        assert json.loads(raw)["metadata"]["note"] == "翻訳テスト"


# ---------------------------------------------------------------------------
# AgentEvaluator with MockModel
# ---------------------------------------------------------------------------

class TestAgentEvaluator:
    def _make_agent(self, responses):
        model = MockModel(responses=responses)
        return Agent(config=AgentConfig(
            name="eval-test",
            model=model,
            tools=[],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
        ))

    def test_run_case_contains(self):
        agent = self._make_agent(["Thought: done\nFinal Answer: The answer is 42"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)
        tc = TestCase(query="What is the answer?", expected_output="42")
        result = evaluator.run_case(tc)
        assert result.passed is True
        assert result.score == 1.0
        assert result.latency > 0

    def test_run_case_exact_match_fail(self):
        agent = self._make_agent(["Thought: done\nFinal Answer: The answer is 42"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.EXACT_MATCH)
        tc = TestCase(query="What is 42?", expected_output="42")
        result = evaluator.run_case(tc)
        # "The answer is 42" != "42" exact match
        assert result.passed is False

    def test_run_case_regex(self):
        agent = self._make_agent(["Thought: done\nFinal Answer: The value is 3.14"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.REGEX)
        tc = TestCase(query="What is pi?", expected_output=r"3\.14")
        result = evaluator.run_case(tc)
        assert result.passed is True

    def test_run_suite(self):
        agent = self._make_agent([
            "Thought: done\nFinal Answer: 5",
            "Thought: done\nFinal Answer: 10",
        ])
        evaluator = AgentEvaluator(agent)

        class FakeSuite:
            name = "fake"
            test_cases = [
                TestCase(query="2+3?", expected_output="5"),
                TestCase(query="5+5?", expected_output="10"),
            ]

        results = evaluator.run_suite(FakeSuite())
        assert results.suite_name == "fake"
        assert results.accuracy == 1.0
        assert len(results.results) == 2

    def test_run_case_agent_error(self):
        """Agent that raises should produce a failed result, not crash."""
        agent = self._make_agent(["Thought: done\nFinal Answer: ok"])
        evaluator = AgentEvaluator(agent)
        # Use an expected output that won't match
        tc = TestCase(query="test", expected_output="XYZNONEXISTENT")
        result = evaluator.run_case(tc)
        assert result.passed is False
        assert result.score == 0.0

    def test_a_run_that_failed_is_reported_as_failed_not_scored(self, monkeypatch):
        """A run that could not reach the model records the failure instead of
        scoring its error text, which could otherwise contain the expected
        string by coincidence and read as a partly-correct model."""
        agent = self._make_agent(["Thought: done\nFinal Answer: ok"])

        class _Failed:
            output = "Generation failed: model `x-4` does not exist. Did you mean 42?"
            success = False
            tokens_used = 0
            metadata = {"error": {"type": "ModelNotFoundError",
                                  "message": "model `x-4` does not exist"}}

        monkeypatch.setattr(agent, "run", lambda *a, **kw: _Failed())
        evaluator = AgentEvaluator(agent)
        # The expected answer appears inside the failure text.
        result = evaluator.run_case(TestCase(query="q", expected_output="42"))
        assert result.passed is False
        assert result.score == 0.0
        assert result.error is not None
        # The message is the classified sentence, not the raw error dict.
        assert "does not exist" in result.error
        assert "{" not in result.error

    def test_a_suite_where_nothing_ran_carries_the_failure(self, monkeypatch):
        agent = self._make_agent(["Thought: done\nFinal Answer: ok"])

        def _boom(*a, **kw):
            raise RuntimeError("provider unreachable")

        monkeypatch.setattr(agent, "run", _boom)
        evaluator = AgentEvaluator(agent)

        class FakeSuite:
            name = "fake"
            test_cases = [TestCase(query="a", expected_output="1"),
                          TestCase(query="b", expected_output="2")]

        results = evaluator.run_suite(FakeSuite())
        assert results.error_count == 2
        assert results.error is not None and "unreachable" in results.error
        assert "ERROR" in results.to_markdown()
        assert results.summary()["error"] is not None

    def test_a_named_judge_grades_instead_of_the_model_itself(self):
        """Under llm_judge the caller's judge does the grading, and the output
        names it so a reader can weigh the score."""
        agent = self._make_agent(["Thought: done\nFinal Answer: blue"])
        judge = self._make_agent(
            ['Thought: done\nFinal Answer: {"score": 0.25, "reasoning": "incomplete"}']
        )
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.LLM_JUDGE, judge_agent=judge)
        result = evaluator.run_case(TestCase(query="color?", expected_output="blue"))
        assert result.score == 0.25, "the judge's score, not the model's own"
        assert result.details["self_judged"] is False
        assert result.details["judge_model"]

    def test_without_a_judge_the_model_grades_itself_and_says_so(self):
        agent = self._make_agent(
            ['Thought: done\nFinal Answer: {"score": 1.0, "reasoning": "perfect"}']
        )
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.LLM_JUDGE)
        result = evaluator.run_case(TestCase(query="color?", expected_output="blue"))
        assert result.details["self_judged"] is True

    def test_run_case_cost_usd_is_none_for_a_model_with_no_price_data(self):
        """MockModel reports no cost_usd metadata — the same shape a real
        local-model run reports — and that must read as None, not $0."""
        agent = self._make_agent(["Thought: done\nFinal Answer: 42"])
        evaluator = AgentEvaluator(agent)
        result = evaluator.run_case(TestCase(query="q", expected_output="42"))
        assert result.cost_usd is None

    def test_run_case_captures_cost_usd_from_response_metadata(self, monkeypatch):
        agent = self._make_agent(["Thought: done\nFinal Answer: 42"])
        real_run = agent.run

        def _run_with_cost(query, *a, **kw):
            resp = real_run(query, *a, **kw)
            resp.metadata["cost_usd"] = 0.002
            return resp

        monkeypatch.setattr(agent, "run", _run_with_cost)
        evaluator = AgentEvaluator(agent)
        result = evaluator.run_case(TestCase(query="q", expected_output="42"))
        assert result.cost_usd == 0.002

    def test_aggregate_sums_known_costs_and_ignores_none(self, monkeypatch):
        agent = self._make_agent([
            "Thought: done\nFinal Answer: 5",
            "Thought: done\nFinal Answer: 10",
        ])
        real_run = agent.run
        costs = iter([0.001, None])

        def _run_with_cost(query, *a, **kw):
            resp = real_run(query, *a, **kw)
            cost = next(costs)
            if cost is not None:
                resp.metadata["cost_usd"] = cost
            return resp

        monkeypatch.setattr(agent, "run", _run_with_cost)
        evaluator = AgentEvaluator(agent)

        class FakeSuite:
            name = "mixed"
            test_cases = [
                TestCase(query="2+3?", expected_output="5"),
                TestCase(query="5+5?", expected_output="10"),
            ]

        results = evaluator.run_suite(FakeSuite())
        assert results.total_cost_usd == pytest.approx(0.001)

    def test_aggregate_total_cost_is_none_when_every_case_is_unpriced(self):
        agent = self._make_agent([
            "Thought: done\nFinal Answer: 5",
            "Thought: done\nFinal Answer: 10",
        ])
        evaluator = AgentEvaluator(agent)

        class FakeSuite:
            name = "unpriced"
            test_cases = [
                TestCase(query="2+3?", expected_output="5"),
                TestCase(query="5+5?", expected_output="10"),
            ]

        results = evaluator.run_suite(FakeSuite())
        assert results.total_cost_usd is None

    def test_semantic_similarity_fallback_surfaces_in_suite_metadata(self, monkeypatch):
        """A silent scoring-mode fallback must be visible in the suite summary
        (which reaches the `--json` CI document), not just a log line."""
        import builtins

        real_import = builtins.__import__

        def _no_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated missing optional dependency")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_sentence_transformers)

        agent = self._make_agent(["Thought: done\nFinal Answer: 42"])
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.SEMANTIC_SIMILARITY)
        tc = TestCase(query="What is the answer?", expected_output="42")
        result = evaluator.run_case(tc)
        assert result.details.get("scoring_fallback") == "contains"

        results = evaluator._aggregate("fake", [result])
        assert results.metadata["scoring_fallback"] == "contains"
        assert results.summary()["metadata"]["scoring_fallback"] == "contains"


# ---------------------------------------------------------------------------
# Test Suites loading
# ---------------------------------------------------------------------------

class TestSuites:
    def test_list_suites(self):
        suites = list_suites()
        assert "math" in suites
        assert "tool_use" in suites
        assert "reasoning" in suites
        assert "safety" in suites
        assert "conversation" in suites

    def test_list_suites_reports_real_case_counts(self):
        # The advertised description must carry the suite's ACTUAL case count,
        # not a stale hardcoded number, so a user can budget an eval run.
        suites = list_suites()
        for name, desc in suites.items():
            actual = len(get_suite(name))
            assert desc.startswith(f"{actual} cases"), (name, actual, desc)

    def test_get_suite(self):
        suite = get_suite("math")
        assert isinstance(suite, MathSuite)
        assert len(suite) > 0

    def test_get_suite_unknown(self):
        with pytest.raises(KeyError):
            get_suite("nonexistent_suite")


class TestResolveEvalSuite:
    """`eval`/`compare` accept a named suite OR a path, plus subsampling."""

    def _resolver(self):
        from effgen.cli._main import _resolve_eval_suite
        return _resolve_eval_suite

    def test_named_suite_with_max_cases(self):
        suite = self._resolver()("math", max_cases=5)
        assert len(suite.test_cases) == 5

    def test_named_suite_difficulty_filter(self):
        suite = self._resolver()("math", difficulty="easy")
        assert len(suite.test_cases) > 0
        assert all(tc.difficulty == Difficulty.EASY for tc in suite.test_cases)

    def test_custom_jsonl_file(self, tmp_path):
        p = tmp_path / "mine.jsonl"
        p.write_text(
            '{"query":"2+2?","expected_output":"4","difficulty":"easy"}\n'
            '{"query":"cap of France?","expected_output":"Paris","difficulty":"hard"}\n',
            encoding="utf-8",
        )
        suite = self._resolver()(str(p))
        assert len(suite.test_cases) == 2
        assert suite.name == "mine"
        # difficulty filter applies to a file suite too
        easy = self._resolver()(str(p), difficulty="easy")
        assert len(easy.test_cases) == 1

    def test_custom_json_array_file(self, tmp_path):
        p = tmp_path / "mine.json"
        p.write_text(
            json.dumps([{"query": "q1", "expected_output": "a1"}]), encoding="utf-8",
        )
        suite = self._resolver()(str(p))
        assert len(suite.test_cases) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._resolver()(str(tmp_path / "nope.jsonl"))

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            self._resolver()("not_a_real_suite")

    def test_math_suite_loads(self):
        suite = MathSuite()
        assert len(suite.test_cases) >= 50
        # Check difficulty distribution
        easy = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.EASY]
        medium = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.MEDIUM]
        hard = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.HARD]
        assert len(easy) > 0
        assert len(medium) > 0
        assert len(hard) > 0

    def test_tool_use_suite_loads(self):
        suite = ToolUseSuite()
        assert len(suite.test_cases) >= 30
        # Verify tools are specified
        tools_seen = set()
        for tc in suite.test_cases:
            tools_seen.update(tc.expected_tools)
        assert len(tools_seen) > 5  # multiple tool types

    def test_reasoning_suite_loads(self):
        suite = ReasoningSuite()
        assert len(suite.test_cases) >= 20

    def test_safety_suite_loads(self):
        suite = SafetySuite()
        assert len(suite.test_cases) >= 20
        tags_seen = set()
        for tc in suite.test_cases:
            tags_seen.update(tc.tags)
        assert "prompt_injection" in tags_seen
        assert "jailbreak" in tags_seen
        assert "harmful_request" in tags_seen

    def test_conversation_suite_loads(self):
        suite = ConversationSuite()
        assert len(suite.test_cases) >= 10

    def test_suite_filter(self):
        suite = MathSuite()
        easy = suite.filter(difficulty=Difficulty.EASY)
        assert all(tc.difficulty == Difficulty.EASY for tc in easy)
        assert len(easy) < len(suite.test_cases)

    def test_suite_iteration(self):
        suite = MathSuite()
        cases = list(suite)
        assert len(cases) == len(suite.test_cases)


# ---------------------------------------------------------------------------
# Regression Tracker
# ---------------------------------------------------------------------------

class TestRegressionTracker:
    def test_save_and_load_baseline(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        results = SuiteResults(
            suite_name="math",
            accuracy=0.8,
            avg_latency=0.5,
            total_tokens=1000,
            avg_tool_accuracy=0.9,
        )
        path = tracker.save_baseline("math", results, version="0.1.0")
        assert path.exists()

        loaded = tracker.load_baseline("math")
        assert loaded is not None
        assert loaded["version"] == "0.1.0"
        assert loaded["summary"]["accuracy"] == 0.8

    def test_compare_no_baseline(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        results = SuiteResults(suite_name="new", accuracy=0.7)
        report = tracker.compare("new", results, version="0.1.0")
        assert not report.has_regressions  # first run = no regression

    def test_compare_no_regression(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.82, avg_latency=0.45)
        report = tracker.compare("math", current, version="0.2.0")
        assert not report.has_regressions

    def test_compare_accuracy_regression(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.7, avg_latency=0.5)
        report = tracker.compare("math", current, version="0.2.0")
        assert report.has_regressions
        assert any("accuracy" in str(a) for a in report.alerts)

    def test_compare_latency_is_advisory_not_regression(self, tmp_path):
        """A latency increase is reported as a notice but must NOT block.

        Free-tier API latency jitter previously filed a false-positive
        "regression" issue almost every night even when accuracy was flat
        or improving. Latency is now advisory-only.
        """
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.7)
        report = tracker.compare("math", current, version="0.2.0")
        # Latency drift alone is not a regression.
        assert not report.has_regressions
        assert not report.blocking_alerts
        # ...but it is surfaced as an advisory notice.
        assert any("avg_latency" in str(a) for a in report.notices)

    def test_accuracy_regression_blocks_even_with_latency_improvement(self, tmp_path):
        """Accuracy drop blocks regardless of latency; latency improving alone passes."""
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.85, avg_latency=4.0)
        tracker.save_baseline("math", baseline, version="0.1.0")

        # Mirrors the real nightly report: accuracy UP, latency UP — must pass.
        improved = SuiteResults(suite_name="math", accuracy=0.95, avg_latency=6.1)
        report = tracker.compare("math", improved, version="ci-nightly")
        assert not report.has_regressions
        assert "PASS" in report.to_markdown()

    def test_report_to_markdown(self, tmp_path):
        tracker = RegressionTracker(baselines_dir=tmp_path)
        baseline = SuiteResults(suite_name="math", accuracy=0.8, avg_latency=0.5)
        tracker.save_baseline("math", baseline, version="0.1.0")

        current = SuiteResults(suite_name="math", accuracy=0.7, avg_latency=0.5)
        report = tracker.compare("math", current, version="0.2.0")
        md = report.to_markdown()
        assert "REGRESSION DETECTED" in md
        assert "Baseline" in md

    def test_save_baseline_suite_name_with_path_separators_does_not_crash(self, tmp_path):
        """A suite name that is a filesystem path (the CLI's raw --suite
        argument for a custom dataset) must never turn into a nested/missing
        directory when building the baseline filename."""
        tracker = RegressionTracker(baselines_dir=tmp_path)
        results = SuiteResults(suite_name="mycases", accuracy=0.9)
        path = tracker.save_baseline("data/nested/mycases.jsonl", results, version="0.1.0")
        assert path.exists()
        assert path.parent == tmp_path  # no nested subdirectory was created
        loaded = tracker.load_baseline("data/nested/mycases.jsonl")
        assert loaded is not None
        assert loaded["summary"]["accuracy"] == 0.9

    def test_default_baselines_dir_is_cwd_relative_not_package_tree(self, tmp_path, monkeypatch):
        """Baselines must live under the caller's own working directory by
        default, not the installed effGen package tree."""
        monkeypatch.chdir(tmp_path)
        tracker = RegressionTracker()
        assert tracker.baselines_dir == tmp_path / ".effgen" / "baselines"
        assert tracker.baselines_dir.is_dir()

    def test_load_baseline_falls_back_to_legacy_package_location(self, tmp_path, monkeypatch):
        """A baseline saved before --baseline-dir existed (under the package
        tree) must still be found when the new default directory has none."""
        from effgen.eval import regression as regression_mod

        legacy_dir = tmp_path / "legacy_benchmarks"
        legacy_dir.mkdir()
        monkeypatch.setattr(regression_mod, "_LEGACY_BASELINES_DIR", legacy_dir)

        legacy_tracker = RegressionTracker(baselines_dir=legacy_dir)
        legacy_tracker.save_baseline("math", SuiteResults(suite_name="math", accuracy=0.9), version="0.1.0")

        new_dir = tmp_path / "new_baselines"
        tracker = RegressionTracker(baselines_dir=new_dir)
        assert not (new_dir / "eval_baseline_math.json").exists()
        loaded = tracker.load_baseline("math")
        assert loaded is not None
        assert loaded["summary"]["accuracy"] == 0.9


class TestRegressionAlert:
    def test_severity_warning(self):
        alert = RegressionAlert("accuracy", 0.8, 0.74, 0.05, "math")
        assert alert.severity == "warning"

    def test_severity_high(self):
        alert = RegressionAlert("accuracy", 0.8, 0.68, 0.05, "math")
        assert alert.severity == "high"

    def test_severity_critical(self):
        alert = RegressionAlert("accuracy", 0.8, 0.55, 0.05, "math")
        assert alert.severity == "critical"

    def test_str(self):
        alert = RegressionAlert("accuracy", 0.8, 0.7, 0.05, "math")
        s = str(alert)
        assert "math" in s
        assert "accuracy" in s


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

class TestModelComparison:
    def _make_agent(self, responses):
        model = MockModel(responses=responses)
        return Agent(config=AgentConfig(
            name="cmp-test",
            model=model,
            tools=[],
            max_iterations=3,
            enable_memory=False,
            enable_sub_agents=False,
        ))

    def test_run_comparison(self):
        agent_a = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)
        agent_b = self._make_agent(["Thought: done\nFinal Answer: wrong"] * 5)

        class FakeSuite:
            name = "mini"
            test_cases = [
                TestCase(query="2+3?", expected_output="5"),
                TestCase(query="1+4?", expected_output="5"),
            ]

        comparison = ModelComparison()
        matrix = comparison.run(
            agents={"model-a": agent_a, "model-b": agent_b},
            suites=[FakeSuite()],
        )
        assert len(matrix.scores) == 2
        assert matrix.recommendations.get("mini") == "model-a"

    def test_scores_carry_what_each_model_answered(self):
        """A bake-off can show its work: the answers are in the result, not
        only the aggregate percentages."""
        agent = self._make_agent(["Thought: done\nFinal Answer: 5"] * 4)

        class FakeSuite:
            name = "mini"
            test_cases = [TestCase(query="2+3?", expected_output="5")]

        matrix = ModelComparison().run({"model-a": agent}, [FakeSuite()])
        responses = matrix.scores[0].responses
        assert [r["query"] for r in responses] == ["2+3?"]
        assert "5" in responses[0]["output"]
        assert responses[0]["passed"] is True
        assert matrix.to_dict()["scores"][0]["responses"] == responses

    def test_a_model_that_never_ran_is_reported_as_failed_not_zero(self, monkeypatch):
        """An unreachable model reads as failed everywhere, and cannot be
        recommended over a model that actually answered."""
        good = self._make_agent(["Thought: done\nFinal Answer: 5"] * 4)
        broken = self._make_agent(["Thought: done\nFinal Answer: 5"] * 4)
        monkeypatch.setattr(
            broken, "run",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model not found")),
        )

        class FakeSuite:
            name = "mini"
            test_cases = [TestCase(query="2+3?", expected_output="5")]

        matrix = ModelComparison().run({"good": good, "broken": broken}, [FakeSuite()])
        by_model = {s.model_name: s for s in matrix.scores}
        assert by_model["broken"].error is not None
        assert by_model["broken"].error_count == 1
        assert matrix.recommendations["mini"] == "good"
        markdown = matrix.to_markdown()
        assert "ERROR" in markdown
        assert "## Failures" in markdown
        assert "did not run" in markdown

    def test_recommendation_breaks_accuracy_tie_on_latency(self):
        # Both agents always answer correctly -> tied accuracy. The recommendation
        # must pick the lower-latency model, not whichever was listed first.
        agent_a = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)
        agent_b = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)

        class FakeSuite:
            name = "mini"
            test_cases = [TestCase(query="2+3?", expected_output="5")]

        matrix = ModelComparison().run(
            agents={"a": agent_a, "b": agent_b}, suites=[FakeSuite()],
        )
        assert all(s.accuracy == 1.0 for s in matrix.scores)
        fastest = min(matrix.scores, key=lambda s: s.avg_latency).model_name
        assert matrix.recommendations["mini"] == fastest

    def test_recommendation_key_order_independent_and_token_tiebreak(self):
        from effgen.eval.comparison import _recommendation_key

        def pick(scores):
            best = {}
            for s in scores:
                cur = best.get(s.suite_name)
                if cur is None or _recommendation_key(s) > _recommendation_key(cur):
                    best[s.suite_name] = s
            return {k: v.model_name for k, v in best.items()}

        slow = ModelScore("slow", "math", accuracy=1.0, avg_latency=0.222, total_tokens=500)
        fast = ModelScore("fast", "math", accuracy=1.0, avg_latency=0.059, total_tokens=400)
        # Faster model wins regardless of listing order.
        assert pick([slow, fast])["math"] == "fast"
        assert pick([fast, slow])["math"] == "fast"
        # Accuracy still dominates latency.
        acc = ModelScore("acc", "math", accuracy=1.0, avg_latency=0.5, total_tokens=900)
        quick = ModelScore("quick", "math", accuracy=0.8, avg_latency=0.01, total_tokens=100)
        assert pick([acc, quick])["math"] == "acc"
        # Latency tie -> fewer tokens wins.
        a = ModelScore("a", "math", accuracy=1.0, avg_latency=0.1, total_tokens=500)
        b = ModelScore("b", "math", accuracy=1.0, avg_latency=0.1, total_tokens=300)
        assert pick([a, b])["math"] == "b"

    def test_comparison_matrix_to_markdown(self):
        matrix = ComparisonMatrix(
            scores=[
                ModelScore("model-a", "math", accuracy=0.9, avg_latency=1.0),
                ModelScore("model-b", "math", accuracy=0.7, avg_latency=0.5),
            ],
            recommendations={"math": "model-a"},
        )
        md = matrix.to_markdown()
        assert "model-a" in md
        assert "model-b" in md
        assert "Accuracy" in md

    def test_comparison_matrix_to_json(self):
        matrix = ComparisonMatrix(
            scores=[ModelScore("m", "s", accuracy=0.5)],
        )
        j = json.loads(matrix.to_json())
        assert len(j["scores"]) == 1

    def test_comparison_matrix_to_json_emits_raw_utf8_not_escaped(self):
        matrix = ComparisonMatrix(
            scores=[ModelScore("m", "s", accuracy=0.0, error="翻訳に失敗しました")],
        )
        raw = matrix.to_json()
        assert "翻訳に失敗しました" in raw
        assert "\\u" not in raw

    def test_empty_matrix(self):
        matrix = ComparisonMatrix()
        assert "No scores" in matrix.to_markdown()

    def test_comparison_matrix_to_markdown_includes_cost_table(self):
        matrix = ComparisonMatrix(
            scores=[
                ModelScore("model-a", "math", accuracy=0.9, avg_latency=1.0, avg_cost_usd=0.002),
                ModelScore("model-b", "math", accuracy=0.7, avg_latency=0.5, avg_cost_usd=None),
            ],
            recommendations={"math": "model-a"},
        )
        md = matrix.to_markdown()
        assert "Avg Cost (USD/run)" in md
        assert "$0.002000" in md
        assert "unpriced" in md

    def test_comparison_matrix_to_dict_includes_cost_and_optimize(self):
        matrix = ComparisonMatrix(
            scores=[ModelScore("m", "s", accuracy=0.5, avg_cost_usd=0.001)],
            optimize="cost",
        )
        d = matrix.to_dict()
        assert d["scores"][0]["avg_cost_usd"] == 0.001
        assert d["optimize"] == "cost"

    def test_optimize_cost_prefers_cheaper_qualifying_model(self, monkeypatch):
        # Both meet the accuracy threshold; "cheap" costs less than "pricey"
        # even though both answer correctly — cost should decide.
        agent_cheap = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)
        agent_pricey = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)

        def _inject(agent, cost):
            real_run = agent.run

            def _run(query, *a, **kw):
                resp = real_run(query, *a, **kw)
                resp.metadata["cost_usd"] = cost
                return resp
            monkeypatch.setattr(agent, "run", _run)

        _inject(agent_cheap, 0.0001)
        _inject(agent_pricey, 0.01)

        class FakeSuite:
            name = "mini"
            test_cases = [TestCase(query="2+3?", expected_output="5")]

        comparison = ModelComparison(pass_threshold=0.5)
        matrix = comparison.run(
            agents={"cheap": agent_cheap, "pricey": agent_pricey},
            suites=[FakeSuite()],
            optimize="cost",
        )
        assert all(s.accuracy == 1.0 for s in matrix.scores)  # both qualify
        by_name = {s.model_name: s.avg_cost_usd for s in matrix.scores}
        assert by_name["cheap"] < by_name["pricey"]
        assert matrix.recommendations["mini"] == "cheap"

    def test_optimize_cost_falls_back_to_full_field_when_none_qualify(self):
        from effgen.eval.comparison import _select_recommendation

        low_acc_cheap = ModelScore("cheap", "math", accuracy=0.1, avg_latency=0.1, avg_cost_usd=0.0001)
        low_acc_pricey = ModelScore("pricey", "math", accuracy=0.2, avg_latency=0.1, avg_cost_usd=0.01)
        best = _select_recommendation(
            [low_acc_cheap, low_acc_pricey], pass_threshold=0.5, optimize="cost",
        )
        # Neither meets the 0.5 threshold, so the full field is considered —
        # cost still decides among the fallback pool.
        assert best.model_name == "cheap"

    def test_optimize_latency_prefers_faster_qualifying_model(self):
        from effgen.eval.comparison import _select_recommendation

        fast = ModelScore("fast", "math", accuracy=1.0, avg_latency=0.01, total_tokens=50)
        slow = ModelScore("slow", "math", accuracy=1.0, avg_latency=5.0, total_tokens=50)
        best = _select_recommendation([fast, slow], pass_threshold=0.5, optimize="latency")
        assert best.model_name == "fast"

    def test_optimize_latency_restricts_to_qualifying_candidates(self):
        from effgen.eval.comparison import _select_recommendation

        fast_but_wrong = ModelScore("fast", "math", accuracy=0.0, avg_latency=0.01)
        slower_but_right = ModelScore("slow", "math", accuracy=1.0, avg_latency=2.0)
        best = _select_recommendation(
            [fast_but_wrong, slower_but_right], pass_threshold=0.5, optimize="latency",
        )
        assert best.model_name == "slow"

    def test_default_optimize_is_accuracy(self):
        agent_a = self._make_agent(["Thought: done\nFinal Answer: 5"] * 5)
        agent_b = self._make_agent(["Thought: done\nFinal Answer: wrong"] * 5)

        class FakeSuite:
            name = "mini"
            test_cases = [TestCase(query="2+3?", expected_output="5")]

        matrix = ModelComparison().run(
            agents={"model-a": agent_a, "model-b": agent_b}, suites=[FakeSuite()],
        )
        assert matrix.optimize == "accuracy"


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

class TestImports:
    def test_top_level_import(self):
        from effgen.eval import (
            AgentEvaluator,
        )
        # Just verify they are importable
        assert AgentEvaluator is not None

    def test_effgen_level_import(self):
        from effgen import AgentEvaluator
        assert AgentEvaluator is not None


class TestJudgeReporting:
    """Whatever graded the answers is named in the result."""

    def _agent(self, responses):
        model = MockModel(responses=responses)
        return Agent(config=AgentConfig(
            name="judge-report", model=model, tools=[], max_iterations=3,
            enable_memory=False, enable_sub_agents=False,
        ))

    class _Suite:
        name = "mini"
        test_cases = [TestCase(query="color?", expected_output="blue")]

    def test_a_named_judge_is_recorded_in_the_result(self):
        judge = self._agent(
            ['Thought: done\nFinal Answer: {"score": 0.5, "reasoning": "partial"}'] * 4
        )
        matrix = ModelComparison(
            scoring=ScoringMode.LLM_JUDGE, judge_agent=judge,
        ).run({"model-a": self._agent(["Thought: done\nFinal Answer: blue"] * 4)},
              [self._Suite()])
        assert matrix.self_judged is False
        assert matrix.judge_model
        doc = matrix.to_dict()
        assert doc["self_judged"] is False and doc["judge_model"] == matrix.judge_model
        assert matrix.judge_model in matrix.to_markdown()

    def test_self_grading_is_stated_when_no_judge_was_named(self):
        matrix = ModelComparison(scoring=ScoringMode.LLM_JUDGE).run(
            {"model-a": self._agent(
                ['Thought: done\nFinal Answer: {"score": 1.0, "reasoning": "perfect"}'] * 4
            )},
            [self._Suite()],
        )
        assert matrix.self_judged is True
        assert matrix.judge_model is None
        assert "grading its own answers" in matrix.to_markdown()

    def test_a_scoring_mode_without_a_model_says_nothing_about_a_judge(self):
        matrix = ModelComparison().run(
            {"model-a": self._agent(["Thought: done\nFinal Answer: blue"] * 4)},
            [self._Suite()],
        )
        assert matrix.self_judged is None and matrix.judge_model is None
        assert "judge" not in matrix.to_markdown().lower()
