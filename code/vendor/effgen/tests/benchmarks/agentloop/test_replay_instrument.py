"""Planting defects in the replay reader.

An instrument nobody has planted against is a claim. Every check here plants
something whose answer is known in advance and asserts the reader reports it —
including the four whose expected outcome is a refusal.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from .conftest import FIXTURES, fixture_cell, fixture_expected, require_record_tree
from .harness import COPIED_FILES
from .records import (
    Cell,
    CellIncomplete,
    CellKey,
    ScorerDrift,
    normalise_arguments,
)
from .replay import (
    SAMPLE_WEIGHTED,
    UNWEIGHTED,
    column_stats,
    compare_runs,
    rescore,
    stop_reason_table,
    strip_any_citations,
    strip_trailing_citations,
)

CLOSE = {"rel_tol": 0, "abs_tol": 1e-9}


# ------------------------------------------- 1: a cell whose answer is known


def test_known_cell_reports_every_number_it_was_built_with(known_answer):
    expected = fixture_expected("known_answer")
    stats = known_answer.stats()

    assert stats.n == expected["n"]
    assert math.isclose(stats.accuracy, expected["accuracy"], **CLOSE)
    assert stats.exact_correct == expected["exact_correct"]
    assert math.isclose(stats.mean_llm_calls, expected["mean_llm_calls"], **CLOSE)
    assert math.isclose(stats.mean_prompt_tokens, expected["mean_prompt_tokens"], **CLOSE)
    assert math.isclose(
        stats.mean_completion_tokens, expected["mean_completion_tokens"], **CLOSE
    )
    assert math.isclose(stats.mean_total_tokens, expected["mean_total_tokens"], **CLOSE)
    assert math.isclose(stats.mean_latency_s, expected["mean_latency_s"], **CLOSE)
    assert math.isclose(stats.mean_tool_calls, expected["mean_tool_calls"], **CLOSE)
    assert stats.errors == expected["errors"]
    assert stats.empty_outputs == expected["empty_outputs"]
    assert stats.roundtrip_mismatches == 0
    assert stats.complete is True


def test_known_cell_reproduces_its_stop_reason_table(known_answer):
    expected = fixture_expected("known_answer")["stop_reasons"]
    rows = {str(row.reason): row for row in known_answer.stats().stop_reasons}
    assert set(rows) == set(expected)
    for reason, want in expected.items():
        assert rows[reason].n == want["n"]
        assert math.isclose(rows[reason].accuracy, want["accuracy"], **CLOSE)
        assert math.isclose(rows[reason].share, want["share"], **CLOSE)


# ------------------------------------------------------ 2: a planted regression


def test_a_planted_regression_is_reported_and_its_samples_named(known_answer):
    planted = json.loads((FIXTURES / "planted_regression.summary.json").read_text())
    regressed = fixture_cell("planted_regression")
    diff = compare_runs(known_answer, regressed)

    assert diff.shared == known_answer.stats().n
    assert diff.right_accuracy < diff.left_accuracy
    assert sorted(c.sample_id for c in diff.changed) == sorted(planted["planted_sample_ids"])
    assert len(diff.changed) == planted["planted_k"]
    assert sorted(diff.lost) == sorted(planted["planted_sample_ids"])
    assert diff.gained == ()
    # The share of samples whose answer moved is what a later run has to beat
    # before its delta means anything.
    assert math.isclose(diff.churn, planted["planted_k"] / diff.shared, **CLOSE)


# ------------------------------------------------------------ 3: scorer drift


def test_a_drifted_scorer_voids_the_cell(monkeypatch, known_answer):
    """The reader's own alarm: re-score each record's own answer and compare.

    The planted break takes the *first* match instead of the last, which is what
    the parser did before a response that lists the options and then settles on
    one was scored on the option it happened to mention first.
    """
    from .harness.benchmarks import retrieval, scoring

    def first_match(response: str, valid: str = "ABCDE") -> str | None:
        import re

        if not response:
            return None
        text = response.strip().upper()
        letters = f"[{valid}]"
        for pattern in (
            rf"(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s*(?:IS)?\s*[:\-]?\s*\(?({letters})\)?\b",
            rf"^\(?({letters})\)?$",
            rf"\b({letters})\b",
        ):
            found = re.findall(pattern, text, re.MULTILINE)
            if found:
                return found[0]
        return None

    monkeypatch.setattr(scoring, "extract_choice", first_match)
    monkeypatch.setattr(retrieval, "extract_choice", first_match)

    drifted = fixture_cell("known_answer")
    mismatches = drifted.roundtrip()
    assert mismatches, "the planted parser change produced no disagreement at all"
    with pytest.raises(ScorerDrift) as raised:
        drifted.check_roundtrip()
    assert "void" in str(raised.value)
    assert drifted.stats().roundtrip_mismatches == len(mismatches)


def test_the_undrifted_scorer_reproduces_every_stored_score(known_answer):
    """The other half of the drift check: with nothing planted, it is silent."""
    assert known_answer.roundtrip() == []
    known_answer.check_roundtrip()


# ------------------------------------------------------ 4, 5: the counterfactual


def test_the_identity_transform_changes_nothing(known_answer):
    result = rescore(known_answer, lambda output: output)
    assert (result.fired, result.gained, result.broken) == (0, 0, 0)
    assert result.after == result.before
    assert result.fired_share == 0.0
    assert not result.rewrites_everything


def test_a_transform_that_writes_the_answer_in_is_reported_as_a_rewrite(known_answer):
    stats = known_answer.stats()
    truths = {r.sample_id: r.ground_truth for r in known_answer.records}
    order = iter(known_answer.records)

    def oracle(output: str) -> str:
        return f"Answer: {truths[next(order).sample_id]}"

    result = rescore(known_answer, oracle)
    assert math.isclose(result.after, 100.0, **CLOSE)
    assert result.fired == stats.n - _already_written(known_answer, truths)
    assert result.gained == stats.n - stats.exact_correct
    assert result.broken == 0
    # Firing on most of the set is the fact to read first: it says the change is
    # a new answer format, not a fix to a failure mode.
    assert result.rewrites_everything


def _already_written(cell: Cell, truths: dict) -> int:
    return sum(1 for r in cell.records if r.output == f"Answer: {truths[r.sample_id]}")


def test_the_counterfactual_shows_the_records_it_fired_on(known_answer):
    expected = fixture_expected("known_answer")["citation_counterfactual"]
    result = rescore(known_answer, strip_trailing_citations)
    assert result.fired == expected["fired"]
    assert result.gained == expected["gained"]
    assert result.broken == expected["broken"]
    assert math.isclose(result.before, expected["before"], **CLOSE)
    assert math.isclose(result.after, expected["after"], **CLOSE)
    assert [e.sample_id for e in result.examples] == expected["fired_ids"]
    for example in result.examples:
        assert example.after_output != example.before_output


def test_the_transform_cannot_see_the_answer_it_is_scored_against(known_answer):
    """A transform takes the answer text and nothing else.

    The restriction is the whole reason a counterfactual can be believed: a
    function that could read the ground truth would be an oracle wearing a
    counterfactual's clothes.
    """
    seen: list[tuple] = []

    def watcher(output, *extra):
        seen.append(extra)
        return output

    rescore(known_answer, watcher)
    assert seen and all(extra == () for extra in seen)


# ------------------------------------------------------- 6: fractional scores


def test_partial_credit_is_averaged_over_scores_not_over_the_correct_flag():
    expected = fixture_expected("fractional_scores")
    cell = fixture_cell("fractional_scores", bench="locomo")
    stats = cell.stats()
    assert stats.roundtrip_mismatches == 0
    assert math.isclose(stats.accuracy, expected["accuracy_from_scores"], **CLOSE)
    # The fixture only means something while the two rules disagree.
    assert not math.isclose(
        expected["accuracy_from_scores"], expected["accuracy_from_correct_flags"], abs_tol=0.5
    )
    assert expected["records_with_partial_credit"] > 0
    assert stats.exact_correct / stats.n * 100 == expected["accuracy_from_correct_flags"]


def test_a_scorer_missing_its_package_refuses_instead_of_scoring_zero(monkeypatch):
    """A scorer that cannot import must say so, not report every record as drift.

    The memory scorer stems with ``nltk``. Without it, scoring raised, the
    replay read the raise as a score of 0.0, and the cell reported six of eight
    records as scorer drift — a broken instrument, when the truth was a package.
    """
    from .harness.benchmarks import memory

    monkeypatch.setitem(sys.modules, "nltk", None)
    monkeypatch.setitem(sys.modules, "nltk.stem", None)
    memory._stemmer.cache_clear()
    try:
        cell = fixture_cell("fractional_scores", bench="locomo")
        with pytest.raises(RuntimeError, match="nltk") as raised:
            cell.stats()
        assert raised.type.__name__ == "ScorerUnavailable"
    finally:
        memory._stemmer.cache_clear()


# --------------------------------------------------------- 7: the ragged cell


def test_a_cell_that_did_not_finish_is_refused():
    with pytest.raises(CellIncomplete) as raised:
        fixture_cell("ragged_cell")
    assert "records.jsonl" in str(raised.value)


def test_a_cell_with_no_summary_is_refused():
    with pytest.raises(CellIncomplete) as raised:
        fixture_cell("no_summary")
    assert "summary" in str(raised.value)


def test_a_refused_cell_can_be_read_but_every_number_is_stamped():
    cell = fixture_cell("ragged_cell", allow_partial=True)
    stats = cell.stats()
    assert stats.complete is False
    assert stats.partial_reason and "9" in stats.partial_reason
    assert stats.n == 9


# ------------------------------------------------ 8: the four argument shapes


@pytest.mark.parametrize(
    "given,expected,unparsed",
    [
        ({"query": "a"}, {"query": "a"}, False),
        ('{"query": "a"}', {"query": "a"}, False),
        (None, {}, False),
        ("", {}, False),
        ("query=not json", {"raw": "query=not json"}, True),
        ("[1, 2]", {"raw": "[1, 2]"}, True),
    ],
)
def test_tool_arguments_are_normalised_however_they_were_written(given, expected, unparsed):
    assert normalise_arguments(given) == (expected, unparsed)


def test_unparseable_tool_arguments_are_counted_not_dropped(known_answer):
    expected = fixture_expected("known_answer")
    stats = known_answer.stats()
    assert stats.unparsed_tool_arguments == expected["unparsed_tool_arguments"]
    kept = [
        call.raw
        for record in known_answer.records
        for call in record.tool_calls
        if call.unparsed
    ]
    assert len(kept) == expected["unparsed_tool_arguments"]
    assert all(text for text in kept)


# ------------------------------------------------- 9: an unreported stop reason


def test_a_run_that_does_not_say_why_it_stopped_gets_its_own_row(known_answer):
    rows = known_answer.stats().stop_reasons
    unreported = [row for row in rows if row.reason is None]
    assert len(unreported) == 1, "None must be a bucket of its own, not folded away"
    assert unreported[0].n == 5
    assert math.isclose(sum(row.share for row in rows), 1.0, **CLOSE)
    assert sum(row.n for row in rows) == known_answer.stats().n


def test_the_stop_reason_table_over_several_cells_keeps_the_unreported_bucket(known_answer):
    rows = stop_reason_table([known_answer.stats(), known_answer.stats()])
    by_reason = {row.reason: row for row in rows}
    assert None in by_reason
    assert by_reason[None].n == 10
    assert math.isclose(sum(row.share for row in rows), 1.0, **CLOSE)


# ------------------------------------------------------- 10: the two averages


def test_accuracy_and_cost_are_averaged_by_different_rules(known_answer):
    """A big set counts once for accuracy and for all its samples for cost.

    Mixing the two up moves a table by whole points, so the rule is named in the
    result rather than being left to the reader.
    """
    big = known_answer.stats()
    small = fixture_cell("fractional_scores", bench="locomo").stats()
    column = column_stats("two cells", [big, small])

    assert column.accuracy_rule == UNWEIGHTED
    assert column.cost_rule == SAMPLE_WEIGHTED
    assert math.isclose(column.accuracy, (big.accuracy + small.accuracy) / 2, **CLOSE)

    weighted = (
        big.mean_prompt_tokens * big.n + small.mean_prompt_tokens * small.n
    ) / (big.n + small.n)
    unweighted = (big.mean_prompt_tokens + small.mean_prompt_tokens) / 2
    assert math.isclose(column.mean_prompt_tokens, weighted, **CLOSE)
    assert not math.isclose(weighted, unweighted, abs_tol=0.5)


# --------------------------------------------- the transform is pinned by name


def test_the_two_citation_transforms_are_not_the_same_function():
    """Trailing markers only, not markers anywhere.

    Removing a marker from the middle of an answer also removes the comma that
    separated two of them, leaving ``"D,"`` where the answer was ``D``. The two
    therefore recover different samples, and a check that expects one of them can
    tell it from the other.
    """
    assert strip_trailing_citations("D [1], [2]") == "D"
    assert strip_any_citations("D [1], [2]") == "D,"
    assert strip_trailing_citations("see [1] and [2] for why B") == "see [1] and [2] for why B"
    assert strip_any_citations("see [1] and [2] for why B") == "see and for why B"


# ------------------------------- the copies are copies, checked against source


def test_the_copied_scorers_are_byte_for_byte_copies():
    source_root = _harness_source()
    here = Path(__file__).parent / "harness"
    drifted = []
    for relative, source_relative in COPIED_FILES.items():
        ours = hashlib.sha256((here / relative).read_bytes()).hexdigest()
        theirs = hashlib.sha256((source_root / source_relative).read_bytes()).hexdigest()
        if ours != theirs:
            drifted.append(relative)
    assert not drifted, f"no longer copies of the harness that scored the records: {drifted}"


def _harness_source() -> Path:
    import os

    value = os.environ.get("AGENTLOOP_HARNESS_SOURCE")
    if value and Path(value).is_dir():
        return Path(value)
    pytest.skip(
        "NOT MEASURED: the harness these scorers were copied from is not reachable. "
        "Set AGENTLOOP_HARNESS_SOURCE to its package directory to check for drift."
    )
    raise AssertionError("unreachable")


# -------------------------------------- against the recorded runs, when present


@pytest.mark.parametrize(
    "bench,n,stored,after,gained",
    [
        ("arc_e", 2376, 86.74, 94.82, 192),
        ("arc_c", 1172, 89.68, 95.31, 66),
        ("csqa", 1221, 85.75, 90.01, 52),
    ],
)
def test_recorded_cells_replay_to_the_numbers_they_were_published_with(
    bench, n, stored, after, gained
):
    from .records import RecordTree

    tree = RecordTree(require_record_tree())
    cell = tree.load("14B", bench, "effgen_plus")
    stats = cell.stats()
    assert stats.n == n
    assert round(stats.accuracy, 2) == stored
    assert stats.roundtrip_mismatches == 0

    result = rescore(cell, strip_trailing_citations)
    assert round(result.after, 2) == after
    assert result.gained == gained
    assert result.broken == 0


def test_the_wrong_citation_transform_reports_different_numbers():
    from .records import RecordTree

    tree = RecordTree(require_record_tree())
    cell = tree.load("14B", "arc_e", "effgen_plus")
    trailing = rescore(cell, strip_trailing_citations)
    anywhere = rescore(cell, strip_any_citations)
    assert round(trailing.after, 2) == 94.82
    assert round(anywhere.after, 2) == 93.60
    assert anywhere.gained < trailing.gained


def test_a_cell_short_of_its_siblings_is_refused():
    from .records import RecordTree

    tree = RecordTree(require_record_tree())
    counts = tree.sibling_counts("1.5B", "bb_hard")
    if counts.get("smolagents") == max(counts.values()):
        pytest.skip("NOT MEASURED: the ragged cell in the recorded runs has been refilled")
    with pytest.raises(CellIncomplete):
        tree.load("1.5B", "bb_hard", "smolagents")
    cell = tree.load("1.5B", "bb_hard", "smolagents", allow_partial=True)
    assert cell.stats().complete is False
    assert cell.stats().partial_reason


# ------------------------------- a run this rig wrote, read back by its manifest


def _write_run(directory: Path, bench: str, run_id: str, *, manifest: bool = True) -> Path:
    directory.mkdir(parents=True)
    rows = [
        {
            "sample_id": f"{bench}-{i}",
            "question": "1 + 1?",
            "ground_truth": "2",
            "prediction": "2",
            "score": 1.0,
            "correct": True,
            "output": "The answer is 2",
            "meta": {},
            "attempt": {
                "output": "The answer is 2",
                "tool_calls": [],
                "llm_calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "latency_s": 0.5,
                "error": None,
                "stop_reason": "final_answer",
            },
        }
        for i in range(2)
    ]
    (directory / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    described = {
        "run_id": run_id,
        "benchmark": bench,
        "state": "done",
        "completed": len(rows),
        "total": len(rows),
    }
    if manifest:
        (directory / "manifest.json").write_text(json.dumps(described))
    (directory / "summary.json").write_text(json.dumps(described))
    return directory


def test_a_run_is_keyed_by_the_benchmark_it_says_it_ran(tmp_path):
    # The directory above a live run is named for the batch, not the benchmark,
    # so a key taken from the path names a benchmark that has no scorer.
    run = _write_run(tmp_path / "live" / "gsm8k-a", "gsm8k", "gsm8k-a")
    cell = Cell.load_run(run)
    assert cell.key.bench == "gsm8k"
    assert cell.key.framework == "gsm8k-a"
    assert cell.stats().n == 2
    assert cell.stats().accuracy == 100.0


def test_a_run_keyed_from_its_path_names_a_benchmark_that_does_not_exist(tmp_path):
    run = _write_run(tmp_path / "live" / "gsm8k-b", "gsm8k", "gsm8k-b")
    with pytest.raises(KeyError):
        Cell.load(run, CellKey("live", run.parent.name, run.name))


def test_a_run_that_does_not_say_which_set_it_ran_is_refused(tmp_path):
    run = _write_run(tmp_path / "live" / "mystery", "gsm8k", "mystery", manifest=False)
    (run / "summary.json").write_text(json.dumps({"state": "done", "completed": 2}))
    with pytest.raises(CellIncomplete):
        Cell.load_run(run)


def test_two_runs_of_one_set_are_compared_through_their_manifests(tmp_path):
    a = _write_run(tmp_path / "live" / "gsm8k-a", "gsm8k", "gsm8k-a")
    b = _write_run(tmp_path / "live" / "gsm8k-b", "gsm8k", "gsm8k-b")
    diff = compare_runs(Cell.load_run(a), Cell.load_run(b))
    assert diff.shared == 2
    assert diff.churn == 0.0
    assert diff.delta == 0.0
