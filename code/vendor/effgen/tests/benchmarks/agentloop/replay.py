"""Tables over recorded cells, and the counterfactual.

Three things live here.

**Aggregation.** A column's accuracy is the unweighted mean of its cells'
accuracies; its cost columns — latency, model calls, tokens — are means over
samples, so a set with 2,376 samples counts more than one with 70. Those are two
different rules and mixing them up moves the numbers by whole points, so the
rule that produced a table is printed in that table's header.

**The counterfactual.** ``rescore`` scores the same records twice, with and
without a pure function of the answer text. Both arms are the same samples, so
sampling noise cancels exactly and the result is a per-sample count rather than
a delta with an error bar. It reports how often the change *fired* before it
reports any accuracy, because a change that fires on every sample is a rewrite
of every answer whatever the accuracy does.

**Firing counts.** How often a code path ran, and — where a run captured its log
per sample — how the samples it fired on scored against the ones it did not.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .harness.benchmarks import CATEGORIES
from .records import Cell, CellStats, Record

#: Trailing reference markers on an answer: ``"C [1]"``, ``"D [1], [2]"``.
#: Only at the end, and only markers — a bracketed number in the middle of a
#: sentence is part of the answer.
TRAILING_CITATIONS = re.compile(r"(?:\s*\[\d+\]\s*,?)+\s*$")

#: The same markers wherever they appear. Kept because it is the *wrong* answer,
#: and a useful one to keep: it removes a marker from the middle of a sentence
#: too, and it leaves behind the comma that separated two of them, so an answer
#: of "D [1], [2]" becomes "D," which no longer reads as a choice. It therefore
#: recovers fewer samples than the trailing-only form, and a check that expects
#: the trailing-only numbers can tell the two apart.
ANY_CITATIONS = re.compile(r"\s*\[\d+\]")


def strip_trailing_citations(output: str) -> str:
    return TRAILING_CITATIONS.sub("", output).strip()


def strip_any_citations(output: str) -> str:
    return ANY_CITATIONS.sub("", output).strip()


# ---------------------------------------------------------------- aggregation

#: Accuracy is averaged over cells, each cell counting once.
UNWEIGHTED = "unweighted mean of cell accuracies"
#: Cost is averaged over samples, so a big set counts for more.
SAMPLE_WEIGHTED = "sample-weighted mean over all cells"


@dataclass(frozen=True)
class ColumnStats:
    """One system, summarised over the cells it completed."""

    name: str
    cells: int
    n: int
    accuracy: float
    mean_latency_s: float
    mean_llm_calls: float
    mean_total_tokens: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_tool_calls: float
    roundtrip_mismatches: int
    partial_cells: tuple[str, ...] = ()

    accuracy_rule: str = UNWEIGHTED
    cost_rule: str = SAMPLE_WEIGHTED


def column_stats(name: str, stats: Sequence[CellStats]) -> ColumnStats:
    if not stats:
        raise ValueError(f"{name}: no cells")
    n = sum(s.n for s in stats)

    def weighted(pick: Callable[[CellStats], float]) -> float:
        return sum(pick(s) * s.n for s in stats) / n

    return ColumnStats(
        name=name,
        cells=len(stats),
        n=n,
        accuracy=sum(s.accuracy for s in stats) / len(stats),
        mean_latency_s=weighted(lambda s: s.mean_latency_s),
        mean_llm_calls=weighted(lambda s: s.mean_llm_calls),
        mean_total_tokens=weighted(lambda s: s.mean_total_tokens),
        mean_prompt_tokens=weighted(lambda s: s.mean_prompt_tokens),
        mean_completion_tokens=weighted(lambda s: s.mean_completion_tokens),
        mean_tool_calls=weighted(lambda s: s.mean_tool_calls),
        roundtrip_mismatches=sum(s.roundtrip_mismatches for s in stats),
        partial_cells=tuple(str(s.key) for s in stats if not s.complete),
    )


def category_of(bench: str) -> str | None:
    for category, keys in CATEGORIES.items():
        if bench in keys:
            return category
    return None


def category_accuracy(stats: Sequence[CellStats]) -> dict[str, float]:
    """Per-category accuracy, each cell counting once inside its category."""
    buckets: dict[str, list[float]] = {}
    for cell in stats:
        category = category_of(cell.key.bench)
        if category:
            buckets.setdefault(category, []).append(cell.accuracy)
    return {
        category: sum(values) / len(values)
        for category, values in buckets.items()
    }


@dataclass(frozen=True)
class StopReasonTotal:
    reason: str | None
    n: int
    accuracy: float
    share: float


def stop_reason_table(stats: Sequence[CellStats]) -> tuple[StopReasonTotal, ...]:
    """Every stop reason across a set of cells, most common first.

    ``None`` — the run did not say why it stopped — is a bucket of its own. Most
    systems report nothing at all, and folding that into "answered normally"
    would claim something the records do not say.
    """
    counts: dict[Any, int] = {}
    scored: dict[Any, float] = {}
    for cell in stats:
        for row in cell.stop_reasons:
            counts[row.reason] = counts.get(row.reason, 0) + row.n
            scored[row.reason] = scored.get(row.reason, 0.0) + row.accuracy / 100 * row.n
    total = sum(counts.values())
    return tuple(
        StopReasonTotal(
            reason=reason,
            n=count,
            accuracy=scored[reason] / count * 100,
            share=count / total if total else 0.0,
        )
        for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    )


# ------------------------------------------------------------- counterfactual


@dataclass(frozen=True)
class RescoreExample:
    sample_id: str
    ground_truth: Any
    before_output: str
    after_output: str
    before_score: float
    after_score: float


@dataclass(frozen=True)
class RescoreResult:
    """The same records scored with and without a change to the answer text."""

    key: str
    n: int
    fired: int
    gained: int
    broken: int
    unchanged_correct: int
    unchanged_wrong: int
    before: float
    after: float
    examples: tuple[RescoreExample, ...] = field(default=())
    #: True when the transform was allowed to read the record, not only the
    #: answer text. Such a transform can see things the agent could not, so its
    #: result is labelled wherever it is printed.
    uses_record: bool = False

    @property
    def fired_share(self) -> float:
        return self.fired / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def rewrites_everything(self) -> bool:
        """Fired on more than half the samples.

        Worth saying out loud before any accuracy is read: a change that touches
        most answers is not a fix to a failure mode, it is a new answer format,
        and it has to be checked against every shape of task rather than the one
        it was written for.
        """
        return self.fired_share > 0.5


def rescore(
    cell: Cell,
    transform: Callable[[str], str] | None = None,
    *,
    examples: int = 12,
) -> RescoreResult:
    """Score this cell's records again with ``transform`` applied to each answer.

    ``transform`` sees the answer text and nothing else. It cannot see the
    sample, the ground truth or the score — that restriction is what keeps a
    counterfactual from quietly becoming an oracle.
    """
    if transform is None:
        return _rescore(cell, lambda output, _record: output, examples, False)
    return _rescore(cell, lambda output, _record: transform(output), examples, False)


def rescore_with_record(
    cell: Cell,
    transform: Callable[[str, Record], str],
    *,
    examples: int = 12,
) -> RescoreResult:
    """As ``rescore``, for a change that needs more than the answer text.

    Recovering a result out of the tool calls is the obvious case. The result
    carries ``uses_record``, because a transform with the record in front of it
    can reach information the agent had no access to at the time and a table
    that does not say so invites the wrong comparison.
    """
    return _rescore(cell, transform, examples, True)


def _rescore(
    cell: Cell,
    transform: Callable[[str, Record], str],
    examples: int,
    uses_record: bool,
) -> RescoreResult:
    fired = gained = broken = unchanged_correct = unchanged_wrong = 0
    before_sum = after_sum = 0.0
    shown: list[RescoreExample] = []
    for record in cell.records:
        after_output = transform(record.output, record)
        before_score = record.score
        if after_output == record.output:
            after_score = before_score
        else:
            after_score = cell.score_output(record, after_output)
            fired += 1
            if len(shown) < examples:
                shown.append(
                    RescoreExample(
                        sample_id=record.sample_id,
                        ground_truth=record.ground_truth,
                        before_output=record.output,
                        after_output=after_output,
                        before_score=before_score,
                        after_score=after_score,
                    )
                )
        before_sum += before_score
        after_sum += after_score
        was_right, is_right = before_score >= 0.5, after_score >= 0.5
        if was_right and is_right:
            unchanged_correct += 1
        elif not was_right and not is_right:
            unchanged_wrong += 1
        elif is_right:
            gained += 1
        else:
            broken += 1
    n = len(cell.records)
    return RescoreResult(
        key=str(cell.key),
        n=n,
        fired=fired,
        gained=gained,
        broken=broken,
        unchanged_correct=unchanged_correct,
        unchanged_wrong=unchanged_wrong,
        before=before_sum / n * 100 if n else 0.0,
        after=after_sum / n * 100 if n else 0.0,
        examples=tuple(shown),
        uses_record=uses_record,
    )


# ------------------------------------------------------------------- compare


@dataclass(frozen=True)
class SampleChange:
    sample_id: str
    before_output: str
    after_output: str
    before_score: float
    after_score: float


@dataclass(frozen=True)
class RunDiff:
    """Two runs of the same set, sample by sample.

    ``churn`` is the fraction of samples that changed their answer text. Two
    runs of unchanged code at a non-zero temperature churn a substantial share of
    their samples, and that share is the reason a single-run accuracy delta on a
    small set means nothing: it is the size of the thing a change has to beat.
    """

    left: str
    right: str
    shared: int
    only_left: tuple[str, ...]
    only_right: tuple[str, ...]
    changed: tuple[SampleChange, ...]
    gained: tuple[str, ...]
    lost: tuple[str, ...]
    left_accuracy: float
    right_accuracy: float

    @property
    def churn(self) -> float:
        return len(self.changed) / self.shared if self.shared else 0.0

    @property
    def delta(self) -> float:
        return self.right_accuracy - self.left_accuracy


def compare_runs(left: Cell, right: Cell) -> RunDiff:
    """Diff two runs of the same set: which samples moved, and which way."""
    a = {r.sample_id: r for r in left.records}
    b = {r.sample_id: r for r in right.records}
    shared = sorted(set(a) & set(b))
    changed: list[SampleChange] = []
    gained: list[str] = []
    lost: list[str] = []
    for sid in shared:
        before, after = a[sid], b[sid]
        if before.output != after.output:
            changed.append(
                SampleChange(
                    sample_id=sid,
                    before_output=before.output,
                    after_output=after.output,
                    before_score=before.score,
                    after_score=after.score,
                )
            )
        if after.score >= 0.5 > before.score:
            gained.append(sid)
        elif before.score >= 0.5 > after.score:
            lost.append(sid)
    n = len(shared) or 1
    return RunDiff(
        left=str(left.key),
        right=str(right.key),
        shared=len(shared),
        only_left=tuple(sorted(set(a) - set(b))),
        only_right=tuple(sorted(set(b) - set(a))),
        changed=tuple(changed),
        gained=tuple(gained),
        lost=tuple(lost),
        left_accuracy=sum(a[s].score for s in shared) / n * 100,
        right_accuracy=sum(b[s].score for s in shared) / n * 100,
    )


# -------------------------------------------------------------- firing counts


@dataclass(frozen=True)
class FiringCount:
    """How often a code path ran, and how the samples it ran on scored.

    ``per_sample`` says whether the run captured its log per sample. When it did
    not — every recorded run in the 2026-08 sweep — the count is the number of
    times the phrase appears in the cell's log, the accuracy fields are ``None``,
    and a reader is told the attribution is missing rather than being shown a
    number that looks like one.
    """

    phrase: str
    fired: int
    n: int
    per_sample: bool
    accuracy_fired: float | None = None
    accuracy_not_fired: float | None = None
    examples: tuple[str, ...] = ()

    @property
    def fired_share(self) -> float | None:
        if not self.per_sample or not self.n:
            return None
        return self.fired / self.n

    @property
    def attribution(self) -> str:
        return "per-sample" if self.per_sample else "cell-level (no per-sample attribution)"


def firing_counts(
    cell: Cell,
    phrases: Iterable[str],
    *,
    log_text: str = "",
    examples: int = 6,
) -> dict[str, FiringCount]:
    """Count each phrase, per sample where the records carry their own log."""
    has_per_sample = any(record.log for record in cell.records)
    out: dict[str, FiringCount] = {}
    for phrase in phrases:
        if has_per_sample:
            hit = [r for r in cell.records if phrase in r.log]
            miss = [r for r in cell.records if phrase not in r.log]
            out[phrase] = FiringCount(
                phrase=phrase,
                fired=len(hit),
                n=len(cell.records),
                per_sample=True,
                accuracy_fired=(
                    sum(r.score for r in hit) / len(hit) * 100 if hit else None
                ),
                accuracy_not_fired=(
                    sum(r.score for r in miss) / len(miss) * 100 if miss else None
                ),
                examples=tuple(r.sample_id for r in hit[:examples]),
            )
        else:
            out[phrase] = FiringCount(
                phrase=phrase,
                fired=log_text.count(phrase),
                n=len(cell.records),
                per_sample=False,
            )
    return out


__all__ = [
    "ANY_CITATIONS",
    "ColumnStats",
    "FiringCount",
    "RescoreExample",
    "RescoreResult",
    "RunDiff",
    "SampleChange",
    "SAMPLE_WEIGHTED",
    "StopReasonTotal",
    "TRAILING_CITATIONS",
    "UNWEIGHTED",
    "category_accuracy",
    "category_of",
    "column_stats",
    "compare_runs",
    "firing_counts",
    "rescore",
    "rescore_with_record",
    "stop_reason_table",
    "strip_any_citations",
    "strip_trailing_citations",
]
