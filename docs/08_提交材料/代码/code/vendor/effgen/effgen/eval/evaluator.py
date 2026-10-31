"""
Agent evaluation framework.

Run agents against test suites and collect structured results with
multiple scoring modes.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Difficulty(Enum):
    """Test case difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ScoringMode(Enum):
    """Available scoring strategies."""
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"


@dataclass
class TestCase:
    """A single evaluation test case.

    Attributes:
        query: The input query for the agent. ``input=`` is accepted as an
            ergonomic alias when constructing a TestCase; :meth:`from_dict`
            additionally reads ``prompt`` / ``question``.
        expected_output: Expected output text (used for exact/contains/regex).
            ``expected=`` is accepted as an ergonomic alias.
        expected_tools: Tool names the agent should invoke.
        tags: Arbitrary tags for filtering / grouping.
        difficulty: Difficulty level (a plain string like ``"easy"`` is coerced).
        metadata: Extra data (e.g. multi-turn conversation history).
    """

    __test__ = False  # not a pytest test class

    query: str = ""
    expected_output: str = ""
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)
    # Ergonomic aliases; never stored, reconciled in __post_init__.
    input: str | None = field(default=None, repr=False, compare=False)
    expected: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.input is not None and not self.query:
            self.query = self.input
        self.input = None
        if self.expected is not None and not self.expected_output:
            self.expected_output = self.expected
        self.expected = None
        if isinstance(self.difficulty, str):
            self.difficulty = Difficulty(self.difficulty)
        if not self.query:
            raise ValueError(
                "TestCase requires a non-empty 'query' "
                "(aliases: input/prompt/question)."
            )

    # Field names accepted for the query text, in priority order.
    _QUERY_KEYS = ("query", "input", "prompt", "question")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        """Create a TestCase from a dict (e.g. parsed JSONL line).

        The query text is read from ``query`` or any of the aliases
        ``input`` / ``prompt`` / ``question``. A dict with none of these keys
        raises a ``ValueError`` naming the field it needs and the keys it saw.
        """
        query = next(
            (data[k] for k in cls._QUERY_KEYS if data.get(k)), None
        )
        if query is None:
            raise ValueError(
                "each test case needs a 'query' field "
                "(aliases: input/prompt/question); got keys: "
                f"{sorted(data.keys())}"
            )
        difficulty = data.get("difficulty", "medium")
        if isinstance(difficulty, str):
            difficulty = Difficulty(difficulty)
        return cls(
            query=query,
            expected_output=data.get("expected", data.get("expected_output", "")),
            expected_tools=data.get("tools", data.get("expected_tools", [])),
            tags=data.get("tags", []),
            difficulty=difficulty,
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalResult:
    """Result from evaluating a single test case.

    Attributes:
        test_case: The test case that was evaluated.
        agent_output: Raw agent output text.
        score: Numeric score in [0, 1].
        passed: Whether the test case passed.
        latency: Wall-clock seconds for agent.run().
        tokens_used: Total tokens consumed.
        cost_usd: USD cost of the run, or ``None`` when the model/provider
            publishes no per-token price (e.g. a local model).
        tool_accuracy: Fraction of expected tools that were called.
        tools_called: Names of tools the agent actually invoked.
        scoring_mode: Which scoring mode produced the score.
        details: Extra details (e.g. judge reasoning).
        error: The classified failure message when the run itself failed (the
            model is unreachable, the id does not exist, the key is rejected),
            otherwise ``None``. A case with an ``error`` scored zero because it
            never produced an answer — not because the answer was wrong.
        stop_reason: What ended the run, when the response carried it — so a
            case that scored zero because the loop stopped it is told apart from
            one that scored zero on a wrong answer.
    """
    test_case: TestCase
    agent_output: str = ""
    score: float = 0.0
    passed: bool = False
    latency: float = 0.0
    tokens_used: int = 0
    cost_usd: float | None = None
    tool_accuracy: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    scoring_mode: ScoringMode = ScoringMode.CONTAINS
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    stop_reason: str | None = None


def _md_cell(text: str, limit: int = 120) -> str:
    """Flatten *text* into one Markdown table cell (pipes escaped, newlines folded)."""
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass
class SuiteResults:
    """Aggregated results from running a full test suite.

    Attributes:
        suite_name: Name of the suite.
        results: Per-test-case results.
        accuracy: Fraction of test cases that passed.
        avg_latency: Mean latency across test cases (seconds).
        total_tokens: Sum of tokens consumed.
        total_cost_usd: Sum of per-case cost, or ``None`` when no case
            reported a known price (e.g. every case ran on a local model).
        avg_tool_accuracy: Mean tool accuracy.
        metadata: Extra info (model name, timestamp, etc.).
        error_count: Cases whose run failed before producing an answer.
        error: The first failure message when **every** case failed — the model
            could not be run at all, so its accuracy is not a measurement.
    """
    suite_name: str = ""
    results: list[EvalResult] = field(default_factory=list)
    accuracy: float = 0.0
    avg_latency: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float | None = None
    avg_tool_accuracy: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        ``results`` carries one entry per test case (query, expected/actual
        output, pass/fail, score, and any extra ``details`` such as
        ``scoring_fallback``/``judge_reasoning``) so a consumer that only
        captures this JSON — the documented way to gate a build in CI — can
        see exactly which case failed and why, not just the suite-level
        totals above it.
        """
        by_difficulty: dict[str, dict[str, Any]] = {}
        for r in self.results:
            d = r.test_case.difficulty.value
            if d not in by_difficulty:
                by_difficulty[d] = {"total": 0, "passed": 0}
            by_difficulty[d]["total"] += 1
            if r.passed:
                by_difficulty[d]["passed"] += 1
        for v in by_difficulty.values():
            v["accuracy"] = v["passed"] / v["total"] if v["total"] else 0.0
        return {
            "suite": self.suite_name,
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "accuracy": self.accuracy,
            "avg_latency": round(self.avg_latency, 4),
            "total_tokens": self.total_tokens,
            "total_cost_usd": (
                round(self.total_cost_usd, 8) if self.total_cost_usd is not None else None
            ),
            "avg_tool_accuracy": round(self.avg_tool_accuracy, 4),
            "by_difficulty": by_difficulty,
            "error_count": self.error_count,
            "error": self.error,
            "metadata": self.metadata,
            "results": [
                {
                    "query": r.test_case.query,
                    "expected_output": r.test_case.expected_output,
                    "agent_output": r.agent_output,
                    "stop_reason": r.stop_reason,
                    "score": round(r.score, 4),
                    "passed": r.passed,
                    "latency": round(r.latency, 4),
                    "tokens_used": r.tokens_used,
                    "cost_usd": round(r.cost_usd, 8) if r.cost_usd is not None else None,
                    "tool_accuracy": round(r.tool_accuracy, 4),
                    "tools_called": r.tools_called,
                    "difficulty": r.test_case.difficulty.value,
                    "details": r.details,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2, *, ensure_ascii: bool = False) -> str:
        """Return the JSON rendering of :meth:`summary`.

        The rendering carries raw UTF-8 by default. Pass ``ensure_ascii=True``
        for a destination that cannot accept it, which escapes non-ASCII
        characters as ``\\uXXXX`` rather than losing them.
        """
        return json.dumps(self.summary(), indent=indent, ensure_ascii=ensure_ascii)

    def to_markdown(self) -> str:
        """Render the suite results as Markdown: totals, difficulty split, cases.

        A case with no published price reads ``unpriced`` rather than ``$0``.
        """
        s = self.summary()
        cost = (
            f"${s['total_cost_usd']:.6f}" if s["total_cost_usd"] is not None else "unpriced"
        )
        lines = [
            f"# Evaluation Results — {s['suite'] or 'suite'}",
            "",
            f"- **Accuracy**: {s['accuracy']:.1%} ({s['passed']}/{s['total']} cases)",
            f"- **Avg latency**: {s['avg_latency']:.4f}s",
            f"- **Total tokens**: {s['total_tokens']:,}",
            f"- **Total cost**: {cost}",
        ]
        if s["metadata"].get("scoring"):
            lines.append(f"- **Scoring**: {s['metadata']['scoring']}")

        if s["by_difficulty"]:
            lines.extend(["", "## By difficulty", "",
                          "| Difficulty | Passed | Total | Accuracy |",
                          "|------------|--------|-------|----------|"])
            for name, info in sorted(s["by_difficulty"].items()):
                lines.append(
                    f"| {name} | {info['passed']} | {info['total']} | {info['accuracy']:.1%} |"
                )

        if s["results"]:
            lines.extend(["", "## Cases", "",
                          "| Query | Expected | Got | Result | Latency | Cost |",
                          "|-------|----------|-----|--------|---------|------|"])
            for r in s["results"]:
                case_cost = (
                    f"${r['cost_usd']:.6f}" if r["cost_usd"] is not None else "unpriced"
                )
                verdict = "ERROR" if r["error"] else ("PASS" if r["passed"] else "FAIL")
                lines.append(
                    f"| {_md_cell(r['query'])} | {_md_cell(r['expected_output'], 80)} "
                    f"| {_md_cell(r['agent_output'], 120)} "
                    f"| {verdict} | {r['latency']:.3f}s "
                    f"| {case_cost} |"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_exact_match(expected: str, actual: str) -> float:
    return 1.0 if expected.strip().lower() == actual.strip().lower() else 0.0


def _score_contains(expected: str, actual: str) -> float:
    return 1.0 if expected.strip().lower() in actual.strip().lower() else 0.0


def _score_regex(pattern: str, actual: str) -> float:
    try:
        return 1.0 if re.search(pattern, actual, re.IGNORECASE) else 0.0
    except re.error:
        logger.warning("Invalid regex pattern: %s", pattern)
        return 0.0


#: One loaded ``SentenceTransformer`` per model name, reused across every
#: case in every suite run in this process instead of reloading the weights
#: from disk on each call — a multi-hundred-case suite otherwise pays that
#: load cost (and prints a progress bar) once per case.
_SEMANTIC_MODEL_CACHE: dict[str, Any] = {}


def _get_semantic_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Return a cached ``SentenceTransformer``, loading it once per process."""
    model = _SEMANTIC_MODEL_CACHE.get(model_name)
    if model is None:
        from effgen.utils.embedding_backend import load_sentence_transformer
        model = load_sentence_transformer(model_name)
        _SEMANTIC_MODEL_CACHE[model_name] = model
    return model


def _score_semantic_similarity(expected: str, actual: str) -> tuple[float, bool]:
    """Score via sentence-transformers cosine similarity (optional dep).

    Returns ``(score, used_fallback)``. ``used_fallback`` is True when
    sentence-transformers is unavailable and the score was computed with
    the ``contains`` heuristic instead — the caller records this so a result
    scored this way is distinguishable from a real similarity score.
    """
    try:
        from effgen.utils.embedding_backend import import_sentence_transformers

        (st_util,) = import_sentence_transformers("util")
        model = _get_semantic_model()
    except ImportError as exc:
        logger.warning("%s Falling back to contains scoring.", exc)
        return _score_contains(expected, actual), True
    emb = model.encode([expected, actual], convert_to_tensor=True)
    sim = float(st_util.cos_sim(emb[0], emb[1])[0][0])
    return max(0.0, min(1.0, sim)), False


def _score_llm_judge(agent: Any, query: str, expected: str, actual: str) -> tuple[float, str]:
    """Grade *actual* with the judge *agent*. Returns (score, reasoning).

    The caller chooses the judge. When it is the same agent that produced the
    answer, the model is grading its own work — pass a separate judge agent to
    :class:`AgentEvaluator` for a grade from a model with no stake in it.
    """
    judge_prompt = (
        "You are an evaluation judge. Score the following answer on a scale of 0 to 1.\n"
        "Respond ONLY with a JSON object: {\"score\": <float>, \"reasoning\": \"<text>\"}\n\n"
        f"Question: {query}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}\n"
    )
    try:
        response = agent.run(judge_prompt)
        text = response.output if hasattr(response, "output") else str(response)
        # Try to parse JSON from the response
        match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', text)
        if match:
            parsed = json.loads(match.group(0))
            return float(parsed.get("score", 0.0)), parsed.get("reasoning", "")
        return 0.0, f"Could not parse judge response: {text[:200]}"
    except Exception as exc:
        return 0.0, f"LLM judge error: {exc}"


def _classified_message(error: Any) -> str | None:
    """Reduce a run's recorded error to its readable message.

    A failed run records a classified error as a dict (type, category,
    provider, message, retryable); a reader wants the sentence, not the dict.
    """
    if isinstance(error, dict):
        message = error.get("message")
        kind = error.get("type")
        if message:
            return f"{kind}: {message}" if kind else str(message)
        return str(kind) if kind else None
    return str(error) if error else None


def _agent_model_id(agent: Any) -> str | None:
    """Return the model id an agent runs on, for labelling a judged score."""
    model = getattr(agent, "model", None)
    name = getattr(model, "model_name", None)
    if isinstance(name, str) and name:
        return name
    return getattr(agent, "name", None)


def _compute_tool_accuracy(expected_tools: list[str], called_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    expected_set = {t.lower() for t in expected_tools}
    called_set = {t.lower() for t in called_tools}
    return len(expected_set & called_set) / len(expected_set)


# ---------------------------------------------------------------------------
# AgentEvaluator
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """Run an agent against a test suite and collect results.

    Usage::

        evaluator = AgentEvaluator(agent)
        results = evaluator.run_suite(MathSuite())
        print(results.accuracy)
    """

    def __init__(
        self,
        agent: Any,
        scoring: ScoringMode = ScoringMode.CONTAINS,
        pass_threshold: float = 0.5,
        judge_agent: Any = None,
    ) -> None:
        """
        Args:
            agent: The agent under evaluation.
            scoring: How answers are scored.
            pass_threshold: Score at or above which a case passes.
            judge_agent: Agent that grades answers under
                ``ScoringMode.LLM_JUDGE``. Defaults to *agent*, which means the
                model grades its own answers; supply a different agent to have
                a model with no stake in the result do the grading. The judge's
                model id is recorded in each case's ``details["judge_model"]``
                and in the suite metadata, so a reader can weigh the grade.
        """
        self.agent = agent
        self.scoring = scoring
        self.pass_threshold = pass_threshold
        self.judge_agent = judge_agent if judge_agent is not None else agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_suite(
        self,
        suite: Any,
        max_cases: int | None = None,
        progress_callback: Any = None,
    ) -> SuiteResults:
        """Run all test cases in *suite* and return aggregated results.

        Args:
            suite: A TestSuite instance or iterable of TestCase objects.
            max_cases: If set, run only the first *max_cases* cases (useful for
                quick smoke runs in CI).
            progress_callback: Optional ``callback(completed, total)`` invoked
                after each case finishes (used by the CLI to drive a progress
                bar). Purely observational; does not change results.
        """
        test_cases: list[TestCase] = suite.test_cases if hasattr(suite, "test_cases") else list(suite)
        if max_cases is not None:
            test_cases = test_cases[:max_cases]
        total = len(test_cases)
        results: list[EvalResult] = []
        for idx, tc in enumerate(test_cases, 1):
            results.append(self.run_case(tc))
            if progress_callback is not None:
                try:
                    progress_callback(idx, total)
                except Exception:  # noqa: BLE001 - never let a callback break eval
                    pass
        return self._aggregate(suite.name if hasattr(suite, "name") else "custom", results)

    def run_case(self, tc: TestCase) -> EvalResult:
        """Evaluate a single test case."""
        start = time.perf_counter()
        error: str | None = None
        stop_reason: str | None = None
        try:
            response = self.agent.run(tc.query)
            stop_reason = getattr(response, "stop_reason", None)
            output = response.output if hasattr(response, "output") else str(response)
            tokens = response.tokens_used if hasattr(response, "tokens_used") else 0
            metadata = getattr(response, "metadata", None) or {}
            cost_usd = metadata.get("cost_usd")
            # Extract tool names from execution trace
            tools_called = self._extract_tools(response)
            # A run that reports success=False produced no answer — its output
            # is the classified failure message. Scoring that text would let a
            # provider error coincidentally match the expected string and read
            # as a partly-correct model.
            if getattr(response, "success", True) is False:
                error = _classified_message(metadata.get("error")) or output or "run failed"
        except Exception as exc:
            logger.warning("Agent error on query %r: %s", tc.query[:60], exc)
            error = str(exc)
            stop_reason = getattr(exc, "stop_reason", None)
            output = f"ERROR: {exc}"
            tokens = 0
            cost_usd = None
            tools_called = []
        latency = time.perf_counter() - start

        # A run that failed is reported as failed, not scored: scoring its error
        # text would read as a model that answered badly rather than one that
        # never answered.
        if error is not None:
            return EvalResult(
                test_case=tc,
                agent_output=output,
                score=0.0,
                passed=False,
                latency=latency,
                scoring_mode=self.scoring,
                error=error,
                stop_reason=stop_reason,
            )

        score, details = self._score(tc, output)
        tool_acc = _compute_tool_accuracy(tc.expected_tools, tools_called)

        return EvalResult(
            test_case=tc,
            agent_output=output,
            score=score,
            passed=score >= self.pass_threshold,
            latency=latency,
            tokens_used=tokens,
            cost_usd=cost_usd,
            tool_accuracy=tool_acc,
            tools_called=tools_called,
            scoring_mode=self.scoring,
            details=details,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score(self, tc: TestCase, output: str) -> tuple[float, dict[str, Any]]:
        details: dict[str, Any] = {}
        if self.scoring == ScoringMode.EXACT_MATCH:
            score = _score_exact_match(tc.expected_output, output)
        elif self.scoring == ScoringMode.CONTAINS:
            score = _score_contains(tc.expected_output, output)
        elif self.scoring == ScoringMode.REGEX:
            score = _score_regex(tc.expected_output, output)
        elif self.scoring == ScoringMode.SEMANTIC_SIMILARITY:
            score, used_fallback = _score_semantic_similarity(tc.expected_output, output)
            if used_fallback:
                details["scoring_fallback"] = "contains"
        elif self.scoring == ScoringMode.LLM_JUDGE:
            score, reasoning = _score_llm_judge(
                self.judge_agent, tc.query, tc.expected_output, output,
            )
            details["judge_reasoning"] = reasoning
            details["judge_model"] = _agent_model_id(self.judge_agent)
            details["self_judged"] = self.judge_agent is self.agent
        else:
            score = _score_contains(tc.expected_output, output)
        return score, details

    @staticmethod
    def _extract_tools(response: Any) -> list[str]:
        """Best-effort extraction of tool names from an AgentResponse."""
        tools: list[str] = []
        trace = getattr(response, "execution_trace", None)
        if trace:
            for event in trace:
                name = None
                if hasattr(event, "tool_name"):
                    name = event.tool_name
                elif isinstance(event, dict):
                    name = event.get("tool_name") or event.get("tool")
                if name and name not in tools:
                    tools.append(name)
        return tools

    def _aggregate(self, suite_name: str, results: list[EvalResult]) -> SuiteResults:
        n = len(results) or 1
        metadata: dict[str, Any] = {
            "scoring": self.scoring.value,
            "pass_threshold": self.pass_threshold,
            "num_cases": len(results),
        }
        # Surface a scoring-mode fallback in the suite metadata (reaches the
        # JSON CI document via SuiteResults.summary()) so a run silently
        # scored on a different metric than requested is never invisible.
        if any(r.details.get("scoring_fallback") for r in results):
            metadata["scoring_fallback"] = "contains"
        # Name the judge (and whether it graded its own answers) in the suite
        # metadata so a reader of the JSON knows what produced the score.
        if self.scoring == ScoringMode.LLM_JUDGE:
            metadata["judge_model"] = _agent_model_id(self.judge_agent)
            metadata["self_judged"] = self.judge_agent is self.agent
        known_costs = [r.cost_usd for r in results if r.cost_usd is not None]
        total_cost_usd = sum(known_costs) if known_costs else None
        errored = [r for r in results if r.error]
        # When nothing ran, the suite carries the failure rather than a 0%
        # score, so a caller never reads "unreachable model" as "bad model".
        suite_error = errored[0].error if errored and len(errored) == len(results) else None
        return SuiteResults(
            error_count=len(errored),
            error=suite_error,
            suite_name=suite_name,
            results=results,
            accuracy=sum(1 for r in results if r.passed) / n,
            avg_latency=sum(r.latency for r in results) / n,
            total_tokens=sum(r.tokens_used for r in results),
            total_cost_usd=total_cost_usd,
            avg_tool_accuracy=sum(r.tool_accuracy for r in results) / n,
            metadata=metadata,
        )
