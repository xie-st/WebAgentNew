"""ARC-Challenge, ARC-Easy and CommonsenseQA: multiple choice with retrieval.

The raw baseline answers from what it already knows. Frameworks get the
knowledge_search tool over the local database built from the train and
validation splits, with test items filtered out.
"""

from __future__ import annotations

from typing import Any

from ..types import Sample
from .base import Benchmark
from .scoring import extract_choice

RAW_SYSTEM = (
    "You are a helpful assistant. Answer the multiple choice question with just "
    "the letter of the correct answer (A, B, C, D, or E)."
)

TOOL_SYSTEM = (
    "You are a helpful assistant answering multiple choice questions.\n"
    "Use the knowledge_search tool to look up facts that help you decide.\n"
    "Retrieved text is background, not the answer. If it does not settle the "
    "question, answer from what you already know.\n"
    "Finish with the single letter of the correct option, like: Answer: B"
)


class _MultipleChoice(Benchmark):
    category = "Retrieval"
    tools = ("knowledge_search",)

    def system_prompt(self, with_tools: bool) -> str:
        return TOOL_SYSTEM if with_tools else RAW_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return (
            f"Question: {sample.question}\n\n"
            f"Options:\n{sample.context}\n\n"
            "Answer with the letter of the correct option."
        )

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        valid = "".join(sample.meta["labels"])
        predicted = extract_choice(output, valid or "ABCDE")
        return predicted == sample.answer, predicted

    @staticmethod
    def _make_sample(sample_id: str, row: dict) -> Sample:
        choices = row["choices"]
        labels = [str(label) for label in choices["label"]]
        texts = list(choices["text"])
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        return Sample(
            sample_id=sample_id,
            question=row["question"],
            answer=str(row["answerKey"]).strip(),
            context=options,
            meta={"labels": labels, "options": texts},
        )


class ARCChallenge(_MultipleChoice):
    key = "arc_c"
    label = "ARC-C"

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        rows = self._slice(rows, limit, offset)
        return [self._make_sample(f"arc_c-{offset + i}", r) for i, r in enumerate(rows)]


class ARCEasy(_MultipleChoice):
    key = "arc_e"
    label = "ARC-E"

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        rows = self._slice(rows, limit, offset)
        return [self._make_sample(f"arc_e-{offset + i}", r) for i, r in enumerate(rows)]


class CommonsenseQA(_MultipleChoice):
    key = "csqa"
    label = "CSQA"

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        # The test split has no labels, so the paper scored the validation split.
        rows = load_dataset("tau/commonsense_qa", split="validation")
        rows = self._slice(rows, limit, offset)
        return [self._make_sample(f"csqa-{offset + i}", r) for i, r in enumerate(rows)]
