#!/usr/bin/env python
"""Run the math eval against a provider and decide what the result means.

The scheduled workflows use this to catch a drop in answer quality. The
complication is that the model is served on a metered free tier: when the
account's tokens-per-minute budget runs out, the provider refuses the remaining
requests. A refused case never reached an answer, so scoring it as wrong
measures the account's quota rather than effGen, and comparing a suite full of
refusals against the baseline reports a regression that is not one.

So the verdict is taken from the cases that actually ran:

* no refusals — compare against the baseline, exactly as before;
* some refusals, enough cases left — compare the accuracy over the cases that
  ran, and report the refusals alongside it;
* too few cases left to say anything — ``inconclusive``, which is not a failure;
* nothing ran at all — ``infra``, which is.

Writes ``eval-summary.json``, ``regression-report.md`` and ``eval-status.txt``
in the working directory. Exit codes: 0 pass, skipped or inconclusive; 1
regression, infrastructure failure or crash.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

STATUS_FILE = Path("eval-status.txt")
REPORT_FILE = Path("regression-report.md")
SUMMARY_FILE = Path("eval-summary.json")


def decide(
    summary: dict[str, Any],
    baseline_accuracy: float | None,
    accuracy_threshold: float,
    min_answered_fraction: float,
) -> tuple[str, str]:
    """Return ``(status, report_markdown)`` for a finished suite run.

    ``summary`` is ``SuiteResults.summary()``. ``baseline_accuracy`` is the
    accuracy recorded for the suite, or ``None`` when no baseline exists.
    """
    results = summary.get("results", [])
    total = summary.get("total", len(results))
    refused = [r for r in results if r.get("error")]
    answered = [r for r in results if not r.get("error")]
    passed = sum(1 for r in answered if r.get("passed"))
    accuracy = passed / len(answered) if answered else 0.0

    if not refused:
        # Nothing to explain away; the caller compares against the baseline.
        return "compare", ""

    recorded = (
        f"{baseline_accuracy:.2f}" if baseline_accuracy is not None
        else "none recorded"
    )
    body = [
        "",
        (
            f"{len(refused)} of {total} cases never reached the model. A refused "
            "request produced no answer to score, so the suite accuracy is not a "
            "measurement of effGen and the cases that ran are used instead."
        ),
        "",
        f"- cases that ran: **{len(answered)}** of {total}",
        f"- of those, passed: **{passed}** (accuracy {accuracy:.2f})",
        f"- baseline accuracy: {recorded}",
        "",
        "First error:",
        "",
        "```",
        str(refused[0].get("error"))[:400],
        "```",
    ]

    def report(heading: str, *extra: str) -> str:
        return "\n".join([f"## Eval {heading}", *body, *extra])

    if not answered:
        return "infra", report("could not run")

    if len(answered) < min_answered_fraction * total:
        return "inconclusive", report("inconclusive")

    if (
        baseline_accuracy is not None
        and accuracy < baseline_accuracy - accuracy_threshold
    ):
        return "regression", report(
            "regression",
            "",
            f"The cases that ran scored {accuracy:.2f} against a baseline of "
            f"{baseline_accuracy:.2f}, past the {accuracy_threshold:.0%} "
            "threshold, so this is reported as a regression.",
        )

    return "passed", report("passed on the cases that ran")


def _write(status: str, report: str) -> None:
    STATUS_FILE.write_text(status, encoding="utf-8")
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="label for this run")
    parser.add_argument("--suite", default="math-20")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("EVAL_CASE_DELAY", "8")),
        help="seconds to wait between cases, to stay inside the per-minute "
        "token budget of a free tier",
    )
    parser.add_argument("--min-answered-fraction", type=float, default=0.75)
    args = parser.parse_args(argv)

    if not os.getenv("GROQ_API_KEY", ""):
        _write("skipped", "SKIPPED: GROQ_API_KEY secret not set.")
        return 0

    try:
        from effgen.core.agent import Agent, AgentConfig
        from effgen.eval import AgentEvaluator, MathSuite, RegressionTracker
        from effgen.eval.evaluator import ScoringMode
        from effgen.models.groq_adapter import GroqAdapter
        from effgen.tools.builtin.calculator import Calculator

        adapter = GroqAdapter(args.model)
        adapter.load()
        agent = Agent(
            AgentConfig(
                name=f"{args.version}-eval",
                model=adapter,
                tools=[Calculator()],
                system_prompt=(
                    "You are a math assistant. Use the calculator tool for "
                    "each computation. Give a concise final numerical answer."
                ),
                max_iterations=5,
                temperature=0.1,
            )
        )
        evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)

        def pace(completed: int, total: int) -> None:
            if completed < total and args.delay > 0:
                time.sleep(args.delay)

        results = evaluator.run_suite(
            MathSuite(), max_cases=args.max_cases, progress_callback=pace
        )
        summary = results.summary()
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        tracker = RegressionTracker()
        baseline = tracker.load_baseline(args.suite) or {}
        baseline_accuracy = (baseline.get("summary") or {}).get("accuracy")

        status, report = decide(
            summary,
            baseline_accuracy,
            tracker.accuracy_threshold,
            args.min_answered_fraction,
        )

        if status == "compare":
            report_obj = tracker.compare(
                args.suite, results, version=args.version
            )
            report = report_obj.to_markdown()
            status = "regression" if report_obj.has_regressions else "passed"

        _write(status, report)
        return 1 if status in ("regression", "infra") else 0

    except Exception:
        _write("crashed", f"## Eval crashed\n\n```\n{traceback.format_exc()}\n```\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
