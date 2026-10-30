"""GSM8K and GSM-PLUS: grade school word problems, calculator tool."""

from __future__ import annotations

from typing import Any

from ..types import Sample
from .base import Benchmark
from .scoring import extract_gsm8k_truth, extract_number, numbers_match, _to_float

RAW_SYSTEM = (
    "You are a math problem solver. Solve the given math problem step by step "
    "and give your final answer in \\boxed{answer} at the end of your response."
)

TOOL_SYSTEM = (
    "You are a math problem solver. Work through the problem step by step.\n"
    "Use the calculator tool for every arithmetic step instead of doing the "
    "arithmetic in your head.\n"
    "When you have the result, end your reply with the final answer in "
    "\\boxed{answer}. The answer must be a single number."
)


class _GSMBase(Benchmark):
    category = "Calculator"
    tools = ("calculator",)

    def system_prompt(self, with_tools: bool) -> str:
        return TOOL_SYSTEM if with_tools else RAW_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return (
            f"Question: {sample.question}\n\n"
            "Solve this problem step by step and give your final answer in "
            "\\boxed{answer} format."
        )

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        predicted = extract_number(output)
        return numbers_match(predicted, sample.answer, self.tolerance(sample.answer)), predicted

    @staticmethod
    def tolerance(truth: float | None) -> float:
        return 1e-5


class GSM8K(_GSMBase):
    key = "gsm8k"
    label = "GSM8K"

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="test")
        rows = self._slice(rows, limit, offset)
        samples = []
        for i, row in enumerate(rows):
            truth = extract_gsm8k_truth(row["answer"])
            samples.append(
                Sample(
                    sample_id=f"gsm8k-{offset + i}",
                    question=row["question"],
                    answer=truth,
                    meta={"answer_text": row["answer"]},
                )
            )
        return samples


class GSMPlus(_GSMBase):
    key = "gsmplus"
    label = "GSM-PLUS"

    @staticmethod
    def tolerance(truth: float | None) -> float:
        # GSM-PLUS has decimal answers, so the paper's scripts allowed 1%.
        if truth is None:
            return 1e-5
        return abs(truth) * 0.01 if abs(truth) > 1 else 0.01

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        # testmini (2400), not test (10552). The full split is eight benchmarks'
        # worth of runtime on its own and testmini is the authors' own balanced
        # subset, so it is what this table uses.
        rows = load_dataset("qintongli/GSM-Plus", split="testmini")
        rows = self._slice(rows, limit, offset)
        samples = []
        for i, row in enumerate(rows):
            samples.append(
                Sample(
                    sample_id=f"gsmplus-{offset + i}",
                    question=row["question"],
                    answer=_to_float(str(row["answer"])),
                    meta={"answer_text": row["answer"], "perturbation": row.get("perturbation_type")},
                )
            )
        return samples
