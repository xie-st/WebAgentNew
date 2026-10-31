"""
effGen Evaluation, Benchmarking & Regression Testing.

Provides tools to measure agent quality, detect regressions, and compare
models/configurations.
"""

from __future__ import annotations

from .battle import BattleResult, Contender, run_battle
from .comparison import ModelComparison
from .evaluator import (
    AgentEvaluator,
    Difficulty,
    EvalResult,
    ScoringMode,
    SuiteResults,
    TestCase,
)
from .regression import RegressionTracker
from .suites import (
    ConversationSuite,
    MathSuite,
    ReasoningSuite,
    SafetySuite,
    TestSuite,
    ToolUseSuite,
    get_suite,
    list_suites,
)

__all__ = [
    "AgentEvaluator",
    "Difficulty",
    "EvalResult",
    "ScoringMode",
    "SuiteResults",
    "TestCase",
    "TestSuite",
    "MathSuite",
    "ToolUseSuite",
    "ReasoningSuite",
    "SafetySuite",
    "ConversationSuite",
    "get_suite",
    "list_suites",
    "RegressionTracker",
    "ModelComparison",
    "BattleResult",
    "Contender",
    "run_battle",
]
