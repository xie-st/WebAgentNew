"""What a benchmark has to provide.

A benchmark turns its raw data into `Sample`s, says which prompt the agent
should get, and decides whether an answer is right. It knows nothing about the
frameworks, and the frameworks know nothing about it.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..types import Sample


class Benchmark:
    key: str = ""
    label: str = ""
    category: str = ""
    # Tool names from effgen_bench.tools. The raw baseline ignores them.
    tools: tuple[str, ...] = ()
    # A floor on the completion budget for this benchmark, or None for the run's
    # own setting. A benchmark whose correct answer is simply long — a 63-move
    # Tower of Hanoi sequence, say — is not measuring reasoning if every
    # framework is cut off mid-answer. The floor is applied to the endpoint
    # before any framework is built, so every column gets the same budget.
    min_max_tokens: int | None = None

    def load(self, limit: int | None = None, offset: int = 0, seed: int = 42) -> list[Sample]:
        """Return the samples to run, already trimmed to `limit`."""
        raise NotImplementedError

    # ------------------------------------------------------------ prompting

    def system_prompt(self, with_tools: bool) -> str:
        """Instructions the agent gets up front.

        `with_tools` is False for the raw baseline, which is prompted directly
        with no tools, exactly as in the paper.
        """
        raise NotImplementedError

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        """The question as the agent sees it."""
        parts = []
        if sample.context:
            parts.append(sample.context)
        parts.append(sample.question)
        return "\n\n".join(parts)

    # -------------------------------------------------------------- scoring

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        """Return (correct, the answer we parsed out of the output)."""
        raise NotImplementedError

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _slice(items: Iterable[Any], limit: int | None, offset: int) -> list[Any]:
        items = list(items)
        if offset:
            items = items[offset:]
        if limit is not None and limit > 0:
            items = items[:limit]
        return items
