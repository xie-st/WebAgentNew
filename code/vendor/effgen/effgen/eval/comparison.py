"""
Model comparison utilities.

Run the same evaluation suite across multiple models and generate
comparison reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .evaluator import AgentEvaluator, ScoringMode, SuiteResults

logger = logging.getLogger(__name__)


def _judge_note(judge_model: str | None, self_judged: bool) -> str:
    """State what graded the answers, so the scores can be weighed."""
    if self_judged:
        return (
            "Scored by each model grading its own answers. "
            "Name a judge (`--judge MODEL`) to have one model grade the field."
        )
    return f"Scored by {judge_model or 'a named judge'}, which is not one of the compared models."


@dataclass
class ModelScore:
    """Per-model results for a single suite.

    Attributes:
        error: Set when the model could not be run at all (unknown id, rejected
            key, unreachable provider). Such a model is reported as failed and
            is never eligible to win a recommendation — its zeroed metrics are
            not a measurement of the model.
        error_count: Cases that failed to run, when only some of them did.
        responses: What the model actually answered, one entry per case —
            ``{"query", "output", "score", "passed", "error"}`` — so a bake-off
            can show its work rather than only the aggregate percentages.
    """
    model_name: str
    suite_name: str
    accuracy: float = 0.0
    avg_latency: float = 0.0
    total_tokens: int = 0
    avg_cost_usd: float | None = None
    avg_tool_accuracy: float = 0.0
    error: str | None = None
    error_count: int = 0
    responses: list[dict[str, Any]] = field(default_factory=list)


def _recommendation_key(score: "ModelScore", optimize: str = "accuracy") -> tuple[float, ...]:
    """Sort key for "best model" under *optimize* — higher is better.

    - ``"accuracy"`` (default): accuracy first, tie-broken on lower
      ``avg_latency`` then lower ``total_tokens`` (negated so "higher is
      better" still holds). When latency and tokens are also equal, the
      first-seen model is kept (strict ``>`` comparison), preserving the
      prior stable ordering for genuine ties.
    - ``"cost"``: lower ``avg_cost_usd`` first (unpriced/free models sort as
      ``$0``), tie-broken on higher accuracy then lower latency.
    - ``"latency"``: lower ``avg_latency`` first, tie-broken on higher
      accuracy then lower ``total_tokens``.
    """
    cost = score.avg_cost_usd if score.avg_cost_usd is not None else 0.0
    if optimize == "cost":
        return (-cost, score.accuracy, -float(score.avg_latency))
    if optimize == "latency":
        return (-float(score.avg_latency), score.accuracy, -float(score.total_tokens))
    return (score.accuracy, -float(score.avg_latency), -float(score.total_tokens))


@dataclass
class ComparisonMatrix:
    """Comparison of multiple models across one or more suites.

    Attributes:
        scores: List of per-model-per-suite scores.
        recommendations: ``{suite: model_name}`` best model per suite.
        optimize: What the recommendation optimized for (``"accuracy"``,
            ``"cost"``, or ``"latency"``).
        recommendation_rationale: ``{suite: sentence}`` — why each recommended
            model won, stated with the numbers behind the choice.
        num_cases: Test cases each model ran per suite, or ``None`` when the
            matrix was assembled without that context.
        scoring: Scoring mode the comparison used (e.g. ``"contains"``).
        suite_sizes: ``{suite: case_count}`` for every compared suite.
        generated_at: ISO-8601 UTC timestamp of when the run completed.
        judge_model: Model that graded the answers under ``llm_judge`` scoring,
            or ``None`` for a scoring mode that needs no model.
        self_judged: ``True`` when each model graded its own answers, which is
            the default when no judge was named.
    """
    scores: list[ModelScore] = field(default_factory=list)
    recommendations: dict[str, str] = field(default_factory=dict)
    optimize: str = "accuracy"
    recommendation_rationale: dict[str, str] = field(default_factory=dict)
    num_cases: int | None = None
    scoring: str | None = None
    suite_sizes: dict[str, int] = field(default_factory=dict)
    generated_at: str | None = None
    judge_model: str | None = None
    self_judged: bool | None = None

    def to_markdown(self) -> str:
        """Render the matrix as a Markdown table."""
        if not self.scores:
            return "_No scores recorded._"

        suites = sorted({s.suite_name for s in self.scores})
        models = sorted({s.model_name for s in self.scores})

        # Build lookup
        lookup: dict[tuple[str, str], ModelScore] = {}
        for s in self.scores:
            lookup[(s.model_name, s.suite_name)] = s

        lines: list[str] = ["# Model Comparison", ""]

        # Accuracy table
        lines.append("## Accuracy")
        lines.append("")
        header = "| Model | " + " | ".join(suites) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(suites)) + "|"
        lines.extend([header, sep])
        for m in models:
            cells = []
            for su in suites:
                sc = lookup.get((m, su))
                if sc and sc.error:
                    cells.append("ERROR")
                elif sc:
                    cells.append(f"{sc.accuracy:.1%}")
                else:
                    cells.append("—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

        # Latency table
        lines.extend(["", "## Avg Latency (s)", ""])
        lines.extend([header.replace("Accuracy", "Latency"), sep])
        for m in models:
            cells = []
            for su in suites:
                sc = lookup.get((m, su))
                if sc and not sc.error:
                    cells.append(f"{sc.avg_latency:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

        # Cost table — "unpriced"/free models (avg_cost_usd is None) read as
        # "—" rather than a fabricated $0, mirroring `effgen cost`'s labeling.
        lines.extend(["", "## Avg Cost (USD/run)", ""])
        lines.extend([header.replace("Accuracy", "Cost"), sep])
        for m in models:
            cells = []
            for su in suites:
                sc = lookup.get((m, su))
                if sc and not sc.error and sc.avg_cost_usd is not None:
                    cells.append(f"${sc.avg_cost_usd:.6f}")
                elif sc and not sc.error:
                    cells.append("unpriced")
                else:
                    cells.append("—")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

        # Name every model that could not be run, and every model whose score
        # is based on fewer cases than it was given, so neither reads as a
        # measured result.
        failures = [s for s in self.scores if s.error or s.error_count]
        if failures:
            lines.extend(["", "## Failures", ""])
            for s in sorted(failures, key=lambda x: (x.model_name, x.suite_name)):
                if s.error:
                    lines.append(f"- **{s.model_name}** ({s.suite_name}): did not run — {s.error}")
                else:
                    lines.append(
                        f"- **{s.model_name}** ({s.suite_name}): {s.error_count} "
                        "case(s) failed to run and scored zero"
                    )

        if self.self_judged is not None:
            lines.extend(["", _judge_note(self.judge_model, self.self_judged)])

        # Recommendations
        if self.recommendations:
            lines.extend(["", f"## Recommendations (optimized for {self.optimize})", ""])
            for su, model in sorted(self.recommendations.items()):
                why = self.recommendation_rationale.get(su)
                lines.append(f"- **{su}**: {model}" + (f" — {why}" if why else ""))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return per-model scores plus the recommendations as a serializable dict."""
        return {
            "scores": [
                {
                    "model": s.model_name,
                    "suite": s.suite_name,
                    "accuracy": s.accuracy,
                    "avg_latency": s.avg_latency,
                    "total_tokens": s.total_tokens,
                    "avg_cost_usd": s.avg_cost_usd,
                    "avg_tool_accuracy": s.avg_tool_accuracy,
                    "error": s.error,
                    "error_count": s.error_count,
                    "responses": s.responses,
                }
                for s in self.scores
            ],
            "recommendations": self.recommendations,
            "recommendation_rationale": self.recommendation_rationale,
            "optimize": self.optimize,
            "scoring": self.scoring,
            "num_cases": self.num_cases,
            "suite_sizes": self.suite_sizes,
            "generated_at": self.generated_at,
            "judge_model": self.judge_model,
            "self_judged": self.self_judged,
        }

    def to_json(self, indent: int = 2, *, ensure_ascii: bool = False) -> str:
        """Return the JSON rendering of :meth:`to_dict`.

        The rendering carries raw UTF-8 by default. Pass ``ensure_ascii=True``
        for a destination that cannot accept it, which escapes non-ASCII
        characters as ``\\uXXXX`` rather than losing them.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)


class ModelComparison:
    """Run the same suite(s) across multiple models and compare.

    Usage::

        comparison = ModelComparison()
        matrix = comparison.run(
            agents={"qwen-3b": agent_a, "llama-3b": agent_b},
            suites=[MathSuite()],
        )
        print(matrix.to_markdown())
    """

    def __init__(
        self,
        scoring: ScoringMode = ScoringMode.CONTAINS,
        pass_threshold: float = 0.5,
        judge_agent: Any = None,
    ) -> None:
        """
        Args:
            scoring: How answers are scored.
            pass_threshold: Score at or above which a case passes.
            judge_agent: Agent that grades answers under
                ``ScoringMode.LLM_JUDGE``. Without it each contender grades its
                own answers; supplying one has a single model with no stake in
                the outcome grade every contender.
        """
        self.scoring = scoring
        self.pass_threshold = pass_threshold
        self.judge_agent = judge_agent

    def run(
        self,
        agents: dict[str, Any],
        suites: list[Any],
        optimize: str = "accuracy",
    ) -> ComparisonMatrix:
        """Evaluate all *agents* on all *suites*.

        Args:
            agents: ``{model_name: agent_instance}``
            suites: List of ``TestSuite`` instances.
            optimize: What the per-suite recommendation optimizes for —
                ``"accuracy"`` (default, highest accuracy wins), ``"cost"``
                (cheapest average run wins among candidates meeting
                ``pass_threshold`` accuracy), or ``"latency"`` (fastest
                average run wins among the same qualifying candidates). If no
                candidate meets the threshold, the full field is considered
                so a recommendation is always produced.

        Returns:
            A :class:`ComparisonMatrix` with scores and recommendations.
        """
        matrix = ComparisonMatrix(optimize=optimize)
        matrix.scoring = getattr(self.scoring, "value", str(self.scoring))
        for suite in suites:
            suite_name = suite.name if hasattr(suite, "name") else "custom"
            try:
                matrix.suite_sizes[suite_name] = len(suite)
            except TypeError:
                pass
        if len(set(matrix.suite_sizes.values())) == 1:
            matrix.num_cases = next(iter(matrix.suite_sizes.values()))

        for model_name, agent in agents.items():
            evaluator = AgentEvaluator(
                agent,
                scoring=self.scoring,
                pass_threshold=self.pass_threshold,
                judge_agent=self.judge_agent,
            )
            for suite in suites:
                suite_name = suite.name if hasattr(suite, "name") else "custom"
                logger.info("Evaluating %s on %s ...", model_name, suite_name)
                try:
                    results: SuiteResults = evaluator.run_suite(suite)
                    num_cases = len(results.results) or 1
                    avg_cost_usd = (
                        results.total_cost_usd / num_cases
                        if results.total_cost_usd is not None
                        else None
                    )
                    matrix.scores.append(ModelScore(
                        model_name=model_name,
                        suite_name=suite_name,
                        accuracy=results.accuracy,
                        avg_latency=results.avg_latency,
                        total_tokens=results.total_tokens,
                        avg_cost_usd=avg_cost_usd,
                        avg_tool_accuracy=results.avg_tool_accuracy,
                        # Every case failing means the model never ran; report
                        # that instead of a 0% score it did not earn.
                        error=results.error,
                        error_count=results.error_count,
                        responses=[
                            {
                                "query": r.test_case.query,
                                "output": r.agent_output,
                                "score": round(r.score, 4),
                                "passed": r.passed,
                                "error": r.error,
                            }
                            for r in results.results
                        ],
                    ))
                except Exception as exc:
                    logger.error("Error evaluating %s on %s: %s", model_name, suite_name, exc)
                    matrix.scores.append(ModelScore(
                        model_name=model_name,
                        suite_name=suite_name,
                        error=str(exc),
                    ))

        # Generate recommendations per suite. For "accuracy" (default): highest
        # accuracy, breaking ties on lower latency then fewer tokens — so a
        # "good enough" tie recommends the cheaper/faster model instead of
        # whichever happened to be listed first. For "cost"/"latency": restrict
        # to candidates that meet pass_threshold accuracy (falling back to the
        # full field if none qualify) and pick the cheapest/fastest among them.
        by_suite: dict[str, list[ModelScore]] = {}
        for s in matrix.scores:
            if s.error:
                continue
            by_suite.setdefault(s.suite_name, []).append(s)

        recommendations: dict[str, str] = {}
        rationale: dict[str, str] = {}
        for suite_name, candidates in by_suite.items():
            best = _select_recommendation(candidates, self.pass_threshold, optimize)
            recommendations[suite_name] = best.model_name
            rationale[suite_name] = _recommendation_rationale(best, candidates, optimize)
        matrix.recommendations = recommendations
        matrix.recommendation_rationale = rationale
        matrix.generated_at = datetime.now(UTC).isoformat(timespec="seconds")
        # Name what graded the answers, so a reader knows whether the scores
        # came from a model with a stake in them.
        if self.scoring == ScoringMode.LLM_JUDGE:
            from .evaluator import _agent_model_id

            matrix.self_judged = self.judge_agent is None
            matrix.judge_model = (
                _agent_model_id(self.judge_agent) if self.judge_agent is not None else None
            )

        return matrix


def _cost_phrase(score: ModelScore) -> str:
    """Render a model's average cost per run, or say it is unpriced."""
    if score.avg_cost_usd is None:
        return "unpriced"
    return f"${score.avg_cost_usd:.6f}/run"


def _recommendation_rationale(
    best: ModelScore,
    candidates: list[ModelScore],
    optimize: str,
) -> str:
    """Describe why *best* was recommended, using the numbers behind the choice.

    Names the winning metric and, when there is a runner-up, the margin over
    it — so a shared report states the reason rather than only the model id.
    A model that failed to run is left out of the comparison, so its zeroed
    metrics are never quoted as a margin.
    """
    others = [c for c in candidates if c is not best and not c.error]
    if optimize == "cost":
        priced = [c for c in others if c.avg_cost_usd is not None]
        if best.avg_cost_usd is None:
            # An unpriced model ranks as $0 for the cost objective. Say so, so
            # the choice is not read as a measured price of zero.
            lead = (
                f"no published per-token price, which ranks first on cost, "
                f"at {best.accuracy:.0%} accuracy"
            )
            if priced:
                runner = min(priced, key=lambda c: c.avg_cost_usd or 0.0)
                lead += f"; cheapest priced model is {runner.model_name} at {_cost_phrase(runner)}"
        else:
            lead = f"cheapest at {best.accuracy:.0%} accuracy — {_cost_phrase(best)}"
            if priced:
                runner = min(priced, key=lambda c: c.avg_cost_usd or 0.0)
                lead += f" vs {_cost_phrase(runner)} for {runner.model_name}"
    elif optimize == "latency":
        lead = f"fastest at {best.accuracy:.0%} accuracy — {best.avg_latency:.3f}s/run"
        if others:
            runner = min(others, key=lambda c: c.avg_latency)
            lead += f" vs {runner.avg_latency:.3f}s for {runner.model_name}"
    else:
        lead = f"highest accuracy at {best.accuracy:.0%}"
        tied = [c for c in others if abs(c.accuracy - best.accuracy) < 1e-9]
        if tied:
            plural = "s" if len(tied) > 1 else ""
            # The accuracy ranking breaks ties on lower latency, but two models
            # can tie on latency too. Only claim the speed margin when it holds.
            if all(best.avg_latency < c.avg_latency for c in tied):
                lead += (
                    f", tied with {len(tied)} other model{plural} "
                    f"and faster at {best.avg_latency:.3f}s/run"
                )
            else:
                lead += f", tied with {len(tied)} other model{plural} at {best.avg_latency:.3f}s/run"
            lead += f" ({_cost_phrase(best)})"
        else:
            lead += f" ({_cost_phrase(best)}, {best.avg_latency:.3f}s/run)"
    return lead


def _select_recommendation(
    candidates: list[ModelScore],
    pass_threshold: float,
    optimize: str,
) -> ModelScore:
    """Pick the recommended model for one suite from its *candidates*.

    ``"accuracy"`` considers the full field. ``"cost"``/``"latency"``
    restrict to candidates meeting *pass_threshold* accuracy first (falling
    back to the full field if none qualify) so a recommendation is always
    produced, then rank by :func:`_recommendation_key`. The first-seen
    candidate is kept on an exact tie.
    """
    if optimize in ("cost", "latency"):
        qualified = [c for c in candidates if c.accuracy >= pass_threshold]
        pool = qualified or candidates
    else:
        pool = candidates
    best = pool[0]
    for c in pool[1:]:
        if _recommendation_key(c, optimize) > _recommendation_key(best, optimize):
            best = c
    return best
