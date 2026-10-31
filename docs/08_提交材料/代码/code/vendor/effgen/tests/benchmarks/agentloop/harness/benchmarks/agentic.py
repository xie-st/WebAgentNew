"""GAIA and SimpleQA: the two sets that need real tool use.

Both come from the smolagents benchmark bundle, which is the same source the
paper used. GAIA questions need several tools; SimpleQA needs one web search
and a clean read of the result.
"""

from __future__ import annotations

from typing import Any

from ..types import Sample
from .base import Benchmark
from .scoring import extract_short_answer, short_answers_match

RAW_SYSTEM = (
    "You are a helpful assistant. Answer the question as accurately as you can.\n"
    "Give the shortest answer that is complete: a name, a number or a short "
    "phrase. End your reply with 'Answer: <your answer>'."
)

SEARCH_SYSTEM = (
    "You are a research assistant with a web search tool.\n"
    "Search before you answer. Build the query from the key names, dates and "
    "terms in the question rather than pasting the whole question in.\n"
    "If the first search does not answer it, try a different query.\n"
    "Give the shortest answer that is complete: a name, a number or a short "
    "phrase. End your reply with 'Answer: <your answer>'."
)

GAIA_SYSTEM = (
    "You are a research assistant with a web search tool and a Python tool.\n"
    "Work in steps: search for the facts you need, compute with python_exec when "
    "the question needs arithmetic or string handling, then answer.\n"
    "Follow the exact output format the question asks for. If it asks for a "
    "number, give a bare number with no units or thousands separators.\n"
    "End your reply with 'Answer: <your answer>'."
)


class _ShortAnswer(Benchmark):
    category = "Agentic"
    subset = ""

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return f"Question: {sample.question}"

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        from datasets import load_dataset

        rows = load_dataset("smolagents/benchmark-v1", self.subset)["test"]
        rows = self._slice(rows, limit, offset)
        return [
            Sample(
                sample_id=f"{self.key}-{offset + i}",
                question=row["question"],
                answer=str(row["true_answer"]),
                meta={"source": row.get("source")},
            )
            for i, row in enumerate(rows)
        ]

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        predicted = extract_short_answer(output)
        return short_answers_match(predicted, sample.answer), predicted


class SimpleQA(_ShortAnswer):
    key = "simpleqa"
    label = "SimpleQA"
    subset = "simpleqa"
    tools = ("web_search",)

    def system_prompt(self, with_tools: bool) -> str:
        return SEARCH_SYSTEM if with_tools else RAW_SYSTEM


class GAIA(_ShortAnswer):
    key = "gaia"
    label = "GAIA"
    subset = "gaia"
    tools = ("web_search", "python_exec")

    def system_prompt(self, with_tools: bool) -> str:
        return GAIA_SYSTEM if with_tools else RAW_SYSTEM
