"""MATH-500: competition maths, calculator tool.

Answers are LaTeX, not plain numbers, so this one compares normalized LaTeX
rather than floats.
"""

from __future__ import annotations

from typing import Any

from ..types import Sample
from .base import Benchmark
from .scoring import extract_math_answer, math_answers_match

RAW_SYSTEM = (
    "You are a mathematics expert. Solve the problem step by step and put the "
    "final answer in \\boxed{answer}."
)

TOOL_SYSTEM = (
    "You are a mathematics expert. Solve the problem step by step.\n"
    "Use the calculator tool for numeric steps instead of doing them in your head.\n"
    "Put the final answer in \\boxed{answer}. Keep the same form the problem "
    "asks for: a fraction stays a fraction, a radical stays a radical."
)


class Math500(Benchmark):
    key = "math500"
    label = "MATH-500"
    category = "Calculator"
    tools = ("calculator",)

    def system_prompt(self, with_tools: bool) -> str:
        return TOOL_SYSTEM if with_tools else RAW_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return (
            f"Problem: {sample.question}\n\n"
            "Solve it step by step and put the final answer in \\boxed{answer}."
        )

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = self._slice(rows, limit, offset)
        return [
            Sample(
                sample_id=f"math500-{offset + i}",
                question=row["problem"],
                answer=str(row["answer"]),
                meta={"subject": row.get("subject"), "level": row.get("level")},
            )
            for i, row in enumerate(rows)
        ]

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        predicted = extract_math_answer(output)
        return math_answers_match(predicted, sample.answer), predicted
