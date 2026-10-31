"""Reading a recorded cell: one model, one benchmark, one system.

A cell is a directory holding ``records.jsonl`` (one scored sample per line),
``summary.json`` (the totals the run wrote) and ``run.log``. This module turns
that into objects, checks the two guards that stop a wrong number leaving the
building, and nothing else — the tables live in ``replay.py``.

The two guards:

* **The round-trip identity.** Every record already carries the score the run
  gave it. The reader re-scores each record's own output with its own copy of
  the scorer and compares. A mismatch means the copied scorer has drifted from
  the one that wrote the file, and every number that cell could produce is void.
  It is reported on every table and refused by default.
* **Completeness.** A cell whose record count disagrees with its summary, or
  which has no summary at all, is refused. A partial run is not a result.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .harness.benchmarks import load_benchmark
from .harness.types import Sample

#: Short names for the served models, as the recorded directories spell them.
MODEL_ALIASES: dict[str, str] = {
    "1.5B": "Qwen_Qwen2.5-1.5B-Instruct",
    "3B": "Qwen_Qwen2.5-3B-Instruct",
    "7B": "Qwen_Qwen2.5-7B-Instruct",
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
}


def resolve_model(name: str) -> str:
    """Accept either the short form (``14B``) or the directory name."""
    return MODEL_ALIASES.get(name, name)


# --------------------------------------------------------------------- errors


class CellIncomplete(RuntimeError):
    """A cell that did not finish, or whose files disagree about how far it got."""


class ScorerDrift(RuntimeError):
    """The copied scorer no longer reproduces the scores stored in the records."""


class ScorerUnavailable(RuntimeError):
    """The copied scorer cannot run here because a package it imports is missing.

    Distinct from a scoring failure: the run scored a sample its scorer choked
    on as 0.0, and a replay reproduces that. A scorer that cannot import at all
    would score *every* sample 0.0 and report the whole cell as drift, which
    reads as a broken instrument rather than a missing package.
    """


# --------------------------------------------------------------------- pieces


@dataclass(frozen=True)
class CellKey:
    model: str
    bench: str
    framework: str

    def __str__(self) -> str:
        return f"{self.model}/{self.bench}/{self.framework}"

    @property
    def model_short(self) -> str:
        for short, full in MODEL_ALIASES.items():
            if full == self.model:
                return short
        return self.model


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str
    #: True when ``arguments`` arrived as text that would not parse as JSON. The
    #: text is kept under ``raw`` so nothing is lost, and the count of these is
    #: reported rather than swallowed.
    unparsed: bool = False

    @property
    def raw(self) -> str:
        return str(self.arguments.get("raw", ""))


def normalise_arguments(value: Any) -> tuple[dict[str, Any], bool]:
    """Tool-call arguments as a dict, however the run happened to write them.

    Runs recorded these three ways: as a mapping, as the JSON text the model
    emitted, and as nothing at all. Text that will not parse is kept verbatim
    under ``raw`` and counted, because a reader that quietly dropped it would
    under-report how often a model wrote an unusable call.
    """
    if isinstance(value, dict):
        return dict(value), False
    if value is None:
        return {}, False
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}, False
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return {"raw": value}, True
        if isinstance(parsed, dict):
            return parsed, False
        return {"raw": value}, True
    return {"raw": str(value)}, True


@dataclass(frozen=True)
class Record:
    sample_id: str
    question: str
    ground_truth: Any
    prediction: Any
    score: float
    correct: bool
    output: str
    meta: dict[str, Any]
    tool_calls: tuple[ToolCall, ...]
    tool_call_count: int | None
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None
    stop_reason: str | None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def outcome(self) -> str | None:
        """``answered`` / ``stopped`` / ``failed``, when the run recorded it.

        ``None`` for a record written before a run carried the field, which is
        what makes a strict accuracy unavailable rather than wrong for it.
        """
        value = (self.raw.get("attempt") or {}).get("outcome")
        return None if value is None else str(value)

    @property
    def tools_used(self) -> int:
        if self.tool_call_count is not None:
            return self.tool_call_count
        return len(self.tool_calls)

    @property
    def model_wall_s(self) -> float | None:
        """Seconds inside HTTP calls to the model, when the run measured it."""
        value = (self.raw.get("attempt") or {}).get("model_wall_s")
        return None if value is None else float(value)

    @property
    def framework_wall_s(self) -> float | None:
        value = (self.raw.get("attempt") or {}).get("framework_wall_s")
        return None if value is None else float(value)

    @property
    def framework_cpu_s(self) -> float | None:
        value = (self.raw.get("attempt") or {}).get("framework_cpu_s")
        return None if value is None else float(value)

    @property
    def streaming_seen(self) -> bool:
        return bool((self.raw.get("attempt") or {}).get("streaming_seen"))

    @property
    def log(self) -> str:
        return str((self.raw.get("attempt") or {}).get("log") or "")

    @classmethod
    def from_json(cls, line: dict[str, Any]) -> "Record":
        attempt = line.get("attempt") or {}
        calls: list[ToolCall] = []
        for call in attempt.get("tool_calls") or []:
            args, unparsed = normalise_arguments(call.get("arguments"))
            calls.append(
                ToolCall(
                    name=str(call.get("name") or "?"),
                    arguments=args,
                    result=str(call.get("result") or ""),
                    unparsed=unparsed,
                )
            )
        score = line.get("score")
        if score is None:
            # A run old enough to predate the score column recorded only the
            # bool. Reading it as 1.0/0.0 is what that bool meant.
            score = 1.0 if line.get("correct") else 0.0
        return cls(
            sample_id=str(line.get("sample_id", "")),
            question=str(line.get("question", "")),
            ground_truth=line.get("ground_truth"),
            prediction=line.get("prediction"),
            score=float(score),
            correct=bool(line.get("correct", float(score) >= 0.5)),
            output=str(line.get("output") or ""),
            meta=dict(line.get("meta") or {}),
            tool_calls=tuple(calls),
            tool_call_count=(
                None if attempt.get("tool_call_count") is None
                else int(attempt["tool_call_count"])
            ),
            llm_calls=int(attempt.get("llm_calls") or 0),
            prompt_tokens=int(attempt.get("prompt_tokens") or 0),
            completion_tokens=int(attempt.get("completion_tokens") or 0),
            latency_s=float(attempt.get("latency_s") or 0.0),
            error=attempt.get("error"),
            stop_reason=attempt.get("stop_reason"),
            raw=line,
        )


@dataclass(frozen=True)
class StopReasonRow:
    reason: str | None
    n: int
    accuracy: float
    share: float


@dataclass(frozen=True)
class CellStats:
    key: CellKey
    n: int
    complete: bool
    partial_reason: str | None
    accuracy: float
    #: Accuracy counting only runs that answered — a run the loop stopped scores
    #: zero however good the progress it kept was. ``None`` for a cell whose
    #: records predate the outcome being carried.
    accuracy_strict: float | None
    answered: int | None
    exact_correct: int
    mean_llm_calls: float
    mean_tool_calls: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_total_tokens: float
    mean_latency_s: float
    errors: int
    empty_outputs: int
    unparsed_tool_arguments: int
    roundtrip_mismatches: int
    stop_reasons: tuple[StopReasonRow, ...]
    mean_model_wall_s: float | None = None
    mean_framework_wall_s: float | None = None
    mean_framework_cpu_s: float | None = None
    streaming_seen: bool = False

    @property
    def split_is_reliable(self) -> bool:
        """Whether the framework/model time split can be quoted as a number.

        A streamed response returns from ``send()`` before its body has been
        read, so the model side of the split is short by however long the body
        took. Any run that saw one reports the split as unreliable instead.
        """
        return self.mean_model_wall_s is not None and not self.streaming_seen


# ----------------------------------------------------------------------- cell


class Cell:
    """One recorded cell, with its scorer attached."""

    def __init__(
        self,
        key: CellKey,
        records: list[Record],
        summary: dict[str, Any] | None,
        *,
        allow_partial: bool = False,
        sibling_counts: dict[str, int] | None = None,
    ) -> None:
        self.key = key
        self.records = records
        self.summary = summary
        self.benchmark = load_benchmark(key.bench)
        self.partial_reason = self._completeness(sibling_counts)
        if self.partial_reason and not allow_partial:
            raise CellIncomplete(f"{key}: {self.partial_reason}")
        self._roundtrip: list[tuple[str, float, float]] | None = None

    # ------------------------------------------------------------- guards

    def _completeness(self, sibling_counts: dict[str, int] | None) -> str | None:
        if self.summary is None:
            return "no summary.json, so there is nothing that says the run finished"
        state = self.summary.get("state")
        if state != "done":
            return f"summary state is {state!r}, not 'done'"
        completed = self.summary.get("completed")
        if completed is not None and int(completed) != len(self.records):
            return (
                f"summary says {completed} samples, records.jsonl holds "
                f"{len(self.records)}"
            )
        if sibling_counts:
            others = {
                name: count
                for name, count in sibling_counts.items()
                if name != self.key.framework
            }
            if others and len(self.records) != max(others.values()):
                return (
                    f"{len(self.records)} samples against "
                    f"{max(others.values())} for the other systems on this set"
                )
        return None

    def rebuild_sample(self, record: Record) -> Sample:
        """The benchmark item a record came from.

        Everything the scorer reads is on the record. ``context`` is not, and is
        rebuilt for the multiple-choice sets from the labels and option texts the
        record carries — the same string the run built. No other set's scorer
        looks at it.
        """
        context = ""
        labels = record.meta.get("labels")
        options = record.meta.get("options")
        if labels and options:
            context = "\n".join(
                f"{label}. {text}" for label, text in zip(labels, options)
            )
        return Sample(
            sample_id=record.sample_id,
            question=record.question,
            answer=record.ground_truth,
            context=context,
            meta=dict(record.meta),
        )

    def score_output(self, record: Record, output: str) -> float:
        """Score an answer for this record's sample, with this cell's scorer."""
        try:
            score, _prediction = self.benchmark.score(self.rebuild_sample(record), output)
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            raise ScorerUnavailable(
                f"NOT MEASURED: the {self.key.bench} scorer imports {missing!r}, which is "
                f"not installed. Install the eval extra (pip install 'effgen[eval]') "
                f"and re-run."
            ) from exc
        except Exception:
            # The run recorded a failed scoring as 0.0 rather than dropping the
            # sample, so a replay has to do the same or the two disagree on n.
            return 0.0
        return float(score)

    def roundtrip(self) -> list[tuple[str, float, float]]:
        """Records whose stored score the copied scorer does not reproduce.

        Each entry is ``(sample_id, stored, rescored)``. An empty list is the
        only result that lets this cell's numbers be quoted.
        """
        if self._roundtrip is None:
            drift = []
            for record in self.records:
                again = self.score_output(record, record.output)
                if not math.isclose(again, record.score, rel_tol=0, abs_tol=1e-9):
                    drift.append((record.sample_id, record.score, again))
            self._roundtrip = drift
        return self._roundtrip

    def check_roundtrip(self) -> None:
        drift = self.roundtrip()
        if drift:
            head = ", ".join(f"{sid} {a}!={b}" for sid, a, b in drift[:5])
            raise ScorerDrift(
                f"{self.key}: the scorer reproduces {len(self.records) - len(drift)} of "
                f"{len(self.records)} stored scores; {len(drift)} disagree ({head}). "
                "Every number this cell could produce is void until that is explained."
            )

    # -------------------------------------------------------------- tables

    def stats(self) -> CellStats:
        n = len(self.records)
        if n == 0:
            raise CellIncomplete(f"{self.key}: no records")
        counts: Counter[Any] = Counter(r.stop_reason for r in self.records)
        totals: Counter[Any] = Counter()
        for record in self.records:
            totals[record.stop_reason] += record.score
        rows = tuple(
            StopReasonRow(
                reason=reason,
                n=count,
                accuracy=totals[reason] / count * 100,
                share=count / n,
            )
            for reason, count in sorted(
                counts.items(), key=lambda kv: (-kv[1], str(kv[0]))
            )
        )
        model_wall = [r.model_wall_s for r in self.records if r.model_wall_s is not None]
        fw_wall = [r.framework_wall_s for r in self.records if r.framework_wall_s is not None]
        fw_cpu = [r.framework_cpu_s for r in self.records if r.framework_cpu_s is not None]
        return CellStats(
            key=self.key,
            n=n,
            complete=self.partial_reason is None,
            partial_reason=self.partial_reason,
            accuracy=sum(r.score for r in self.records) / n * 100,
            accuracy_strict=(
                sum(r.score for r in self.records if r.outcome == "answered") / n * 100
                if any(r.outcome is not None for r in self.records)
                else None
            ),
            answered=(
                sum(1 for r in self.records if r.outcome == "answered")
                if any(r.outcome is not None for r in self.records)
                else None
            ),
            exact_correct=sum(1 for r in self.records if r.correct),
            mean_llm_calls=sum(r.llm_calls for r in self.records) / n,
            mean_tool_calls=sum(r.tools_used for r in self.records) / n,
            mean_prompt_tokens=sum(r.prompt_tokens for r in self.records) / n,
            mean_completion_tokens=sum(r.completion_tokens for r in self.records) / n,
            mean_total_tokens=sum(r.total_tokens for r in self.records) / n,
            mean_latency_s=sum(r.latency_s for r in self.records) / n,
            errors=sum(1 for r in self.records if r.error),
            empty_outputs=sum(1 for r in self.records if not r.output.strip()),
            unparsed_tool_arguments=sum(
                1 for r in self.records for c in r.tool_calls if c.unparsed
            ),
            roundtrip_mismatches=len(self.roundtrip()),
            stop_reasons=rows,
            mean_model_wall_s=(sum(model_wall) / len(model_wall)) if model_wall else None,
            mean_framework_wall_s=(sum(fw_wall) / len(fw_wall)) if fw_wall else None,
            mean_framework_cpu_s=(sum(fw_cpu) / len(fw_cpu)) if fw_cpu else None,
            streaming_seen=any(r.streaming_seen for r in self.records),
        )

    # --------------------------------------------------------------- files

    @classmethod
    def load(
        cls,
        directory: str | Path,
        key: CellKey | None = None,
        *,
        allow_partial: bool = False,
        sibling_counts: dict[str, int] | None = None,
    ) -> "Cell":
        path = Path(directory)
        if key is None:
            key = CellKey(path.parent.parent.name, path.parent.name, path.name)
        records = [Record.from_json(line) for line in read_jsonl(path / "records.jsonl")]
        summary_path = path / "summary.json"
        summary = (
            json.loads(summary_path.read_text()) if summary_path.exists() else None
        )
        return cls(
            key,
            records,
            summary,
            allow_partial=allow_partial,
            sibling_counts=sibling_counts,
        )

    @classmethod
    def load_run(
        cls,
        directory: str | Path,
        model: str = "live",
        *,
        allow_partial: bool = True,
    ) -> "Cell":
        """A run this rig wrote, keyed from what the run says it is.

        A recorded cell lives at ``<model>/<benchmark>/<system>/`` and its key
        can be read off the path. A run from :mod:`live` does not: it is one
        directory named for the run, so the benchmark has to come from the
        manifest the run wrote. Reading it off the path instead picks up the
        parent directory's name and the cell is refused for naming a benchmark
        that does not exist.
        """
        path = Path(directory)
        described = None
        for name in ("manifest.json", "summary.json"):
            candidate = path / name
            if candidate.exists():
                described = json.loads(candidate.read_text())
                break
        if described is None or not described.get("benchmark"):
            raise CellIncomplete(
                f"{path}: no manifest.json or summary.json naming the benchmark, "
                "so there is nothing that says which scorer this run should be "
                "read with"
            )
        key = CellKey(model, described["benchmark"], described.get("run_id") or path.name)
        return cls.load(path, key, allow_partial=allow_partial)

    # ---------------------------------------------------------- log firings

    def log_text(self, directory: str | Path) -> str:
        path = Path(directory) / "run.log"
        return path.read_text(errors="replace") if path.exists() else ""


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Every well-formed object in a JSON-lines file, in order."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


# ------------------------------------------------------------------ the tree


class RecordTree:
    """A directory of cells, laid out ``<model>/<benchmark>/<system>/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def exists(self) -> bool:
        return self.root.is_dir()

    def cell_dir(self, model: str, bench: str, framework: str) -> Path:
        return self.root / resolve_model(model) / bench / framework

    def cells(self) -> list[CellKey]:
        found: list[CellKey] = []
        if not self.root.is_dir():
            return found
        for model_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for fw_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                    if (fw_dir / "records.jsonl").exists():
                        found.append(
                            CellKey(model_dir.name, bench_dir.name, fw_dir.name)
                        )
        return found

    def sibling_counts(self, model: str, bench: str) -> dict[str, int]:
        """How many records each system recorded for one model and set."""
        base = self.root / resolve_model(model) / bench
        counts: dict[str, int] = {}
        if not base.is_dir():
            return counts
        for fw_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            path = fw_dir / "records.jsonl"
            if path.exists():
                with open(path, encoding="utf-8") as handle:
                    counts[fw_dir.name] = sum(1 for line in handle if line.strip())
        return counts

    def load(
        self,
        model: str,
        bench: str,
        framework: str,
        *,
        allow_partial: bool = False,
        check_siblings: bool = True,
    ) -> Cell:
        key = CellKey(resolve_model(model), bench, framework)
        return Cell.load(
            self.cell_dir(model, bench, framework),
            key,
            allow_partial=allow_partial,
            sibling_counts=(
                self.sibling_counts(model, bench) if check_siblings else None
            ),
        )

    def load_many(
        self,
        keys: Iterable[CellKey],
        *,
        allow_partial: bool = False,
        on_refused: Callable[[CellKey, Exception], None] | None = None,
    ) -> list[Cell]:
        cells: list[Cell] = []
        for key in keys:
            try:
                cells.append(
                    self.load(
                        key.model, key.bench, key.framework, allow_partial=allow_partial
                    )
                )
            except (CellIncomplete, OSError) as exc:
                if on_refused is None:
                    raise
                on_refused(key, exc)
        return cells


__all__ = [
    "Cell",
    "CellIncomplete",
    "CellKey",
    "CellStats",
    "MODEL_ALIASES",
    "Record",
    "RecordTree",
    "ScorerDrift",
    "StopReasonRow",
    "ToolCall",
    "normalise_arguments",
    "read_jsonl",
    "resolve_model",
    "write_jsonl",
]
