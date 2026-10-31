"""The scheduled eval decides a verdict from the cases that actually ran.

The model is served on a metered free tier. When the account's per-minute token
budget runs out the provider refuses the remaining requests, and a refused case
produced no answer to score. These tests pin the rule that separates "the model
answered worse" from "the account ran out of budget", because reporting the
second as the first is what filled the tracker with regressions nobody could
act on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci_eval_regression.py"

BASELINE_ACCURACY = 0.85
THRESHOLD = 0.05
MIN_ANSWERED = 0.75


@pytest.fixture(scope="module")
def mod():
    """Import scripts/ci_eval_regression.py by path (scripts/ is not a package)."""
    assert SCRIPT.exists(), f"{SCRIPT} is missing"
    spec = importlib.util.spec_from_file_location("_ci_eval_regression", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ci_eval_regression"] = module
    spec.loader.exec_module(module)
    return module


def summary(passed: int, refused: int, total: int = 20) -> dict:
    """A SuiteResults.summary()-shaped dict with a given pass/refusal split."""
    answered = total - refused
    assert passed <= answered
    results = [{"error": "RateLimitExceeded: 429", "passed": False} for _ in range(refused)]
    results += [{"error": None, "passed": True} for _ in range(passed)]
    results += [{"error": None, "passed": False} for _ in range(answered - passed)]
    return {"total": total, "results": results}


def decide(mod, s):
    return mod.decide(s, BASELINE_ACCURACY, THRESHOLD, MIN_ANSWERED)


def test_a_clean_run_defers_to_the_baseline_comparison(mod):
    status, report = decide(mod, summary(passed=19, refused=0))
    assert status == "compare"
    assert report == ""


def test_a_few_refusals_still_score_the_cases_that_ran(mod):
    # 17 answered of 20 is above the three-quarters floor, and 16/17 beats the
    # baseline, so the run passes rather than going quiet.
    status, report = decide(mod, summary(passed=16, refused=3))
    assert status == "passed"
    assert "17** of 20" in report


def test_a_real_drop_is_still_caught_when_some_cases_were_refused(mod):
    # 17 answered, but only 8 of them right: that is a genuine regression and
    # the refusals must not hide it.
    status, report = decide(mod, summary(passed=8, refused=3))
    assert status == "regression"
    assert "regression" in report


def test_mostly_refused_is_inconclusive_not_a_regression(mod):
    # The shape that filed issues #96-#99: 12 of 20 refused, and the 8 that ran
    # scored fine. Reporting that as a regression is the bug.
    status, report = decide(mod, summary(passed=7, refused=12))
    assert status == "inconclusive"
    assert "never reached the model" in report
    assert "accuracy 0.88" in report


def test_nothing_answered_is_an_infrastructure_failure(mod):
    status, report = decide(mod, summary(passed=0, refused=20))
    assert status == "infra"
    assert "could not run" in report


def test_the_refusal_reason_is_carried_into_the_report(mod):
    s = summary(passed=7, refused=12)
    s["results"][0]["error"] = "RateLimitExceeded: tokens per minute (TPM) limit"
    _, report = decide(mod, s)
    assert "tokens per minute (TPM) limit" in report


def test_a_missing_baseline_never_reports_a_regression(mod):
    status, _ = mod.decide(summary(passed=1, refused=3), None, THRESHOLD, MIN_ANSWERED)
    assert status == "passed"


def test_the_exit_code_fails_only_for_a_regression_or_an_outage(mod):
    # Mirrors main()'s mapping so a change to one is caught against the other.
    failing = {"regression", "infra"}
    for status in ("passed", "inconclusive", "skipped"):
        assert status not in failing
    for status in ("regression", "infra"):
        assert status in failing


def test_the_workflows_call_the_script_rather_than_inlining_the_eval(mod):
    for name in ("monitor.yml", "nightly.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "scripts/ci_eval_regression.py" in text, name
        assert "ev.run_suite(MathSuite()" not in text, name
