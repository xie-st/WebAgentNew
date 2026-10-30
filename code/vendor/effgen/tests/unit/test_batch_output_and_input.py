"""Batch output files carry cost/tokens/parsed/error, and malformed input rows
do not abort the whole job.

These exercise the file-I/O and aggregation logic directly with constructed
``AgentResponse`` objects (no model call): the output writer keeps per-row cost,
token counts, the validated ``parsed`` object, and a failure reason; the
``BatchResult`` sums a per-job spend/token figure; and ``_read_queries`` skips a
malformed line (reporting file + line number) or hard-fails under ``strict``.
"""

from __future__ import annotations

import csv
import json

import pytest
from pydantic import BaseModel

from effgen.core.agent import AgentResponse
from effgen.core.batch import BatchResult, BatchRunner


class _Parsed(BaseModel):
    product: str
    urgency: str


def _resp(output, *, success=True, metadata=None, exec_time=1.0, tokens=0):
    return AgentResponse(
        output=output,
        success=success,
        metadata=metadata or {},
        execution_time=exec_time,
        tokens_used=tokens,
    )


def _batch(responses):
    total = len(responses)
    succeeded = sum(1 for r in responses if r is not None and r.success)
    # Route through the aggregation path so totals mirror a real run.
    runner = BatchRunner(agent=None)
    result = BatchResult(
        results=responses, total=total, succeeded=succeeded,
        failed=total - succeeded, total_time=0.0,
    )
    # Recompute totals the way _run_async does, via the shared row-metric reader.
    from effgen.core.batch import _row_metrics
    cost_seen = False
    total_cost = 0.0
    tok = tok_p = tok_c = 0
    for r in responses:
        m = _row_metrics(r)
        if "cost_usd" in m:
            cost_seen = True
            total_cost += m["cost_usd"]
        tok += m.get("total_tokens", 0)
        tok_p += m.get("prompt_tokens", 0)
        tok_c += m.get("completion_tokens", 0)
    result.total_cost_usd = total_cost if cost_seen else None
    result.total_tokens = tok
    result.total_prompt_tokens = tok_p
    result.total_completion_tokens = tok_c
    return runner, result


def test_jsonl_output_carries_cost_tokens_and_parsed(tmp_path):
    r0 = _resp(
        '{"product":"Widget","urgency":"high"}',
        metadata={
            "cost_usd": 0.00031065,
            "prompt_tokens": 77, "completion_tokens": 767, "total_tokens": 844,
            "parsed": _Parsed(product="Widget", urgency="high"),
        },
    )
    runner, result = _batch([r0])
    out = tmp_path / "out.jsonl"
    runner.write_results(result, out, query_list=["extract this"])
    row = json.loads(out.read_text().splitlines()[0])
    assert row["query"] == "extract this"
    assert row["cost_usd"] == 0.00031065
    assert row["total_tokens"] == 844
    assert row["prompt_tokens"] == 77
    assert row["parsed"] == {"product": "Widget", "urgency": "high"}
    assert "error" not in row


def test_failed_row_carries_error_reason(tmp_path):
    ok = _resp("apple", metadata={"total_tokens": 5})
    bad = _resp(
        "", success=False,
        metadata={"error": {"message": "provider timed out", "type": "TimeoutError"}},
    )
    runner, result = _batch([ok, bad])
    out = tmp_path / "out.jsonl"
    runner.write_results(result, out, query_list=["a", "b"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["success"] is True and "error" not in rows[0]
    assert rows[1]["success"] is False
    assert rows[1]["error"] == "provider timed out"


def test_batch_result_totals_sum_cost_and_tokens():
    r0 = _resp("x", metadata={"cost_usd": 0.001, "total_tokens": 100})
    r1 = _resp("y", metadata={"cost_usd": 0.002, "total_tokens": 250})
    _, result = _batch([r0, r1])
    d = result.to_dict()
    assert d["total_cost_usd"] == 0.003
    assert d["total_tokens"] == 350


def test_totals_cost_is_none_for_unpriced_rows():
    # Local/unpriced models report tokens but no cost.
    r0 = _resp("x", metadata={"total_tokens": 49})
    r1 = _resp("y", metadata={"total_tokens": 100})
    _, result = _batch([r0, r1])
    d = result.to_dict()
    assert d["total_cost_usd"] is None
    assert d["total_tokens"] == 149


def test_csv_output_serializes_parsed_as_json_cell(tmp_path):
    r0 = _resp(
        "out",
        metadata={"parsed": {"product": "Widget", "urgency": "low"}, "total_tokens": 12},
    )
    runner, result = _batch([r0])
    out = tmp_path / "out.csv"
    runner.write_results(result, out, query_list=["q"])
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["parsed"] == '{"product": "Widget", "urgency": "low"}'
    assert rows[0]["total_tokens"] == "12"


def test_csv_output_no_bom_by_default(tmp_path):
    r0 = _resp("out", metadata={"total_tokens": 12})
    runner, result = _batch([r0])
    out = tmp_path / "out.csv"
    runner.write_results(result, out, query_list=["q"])
    assert out.read_bytes()[:3] != b"\xef\xbb\xbf"


def test_csv_output_excel_bom_opt_in(tmp_path):
    r0 = _resp("保存更改", metadata={"total_tokens": 12})
    runner, result = _batch([r0])
    out = tmp_path / "out.csv"
    runner.write_results(result, out, query_list=["Save changes"], excel_bom=True)
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    # Still valid UTF-8 content once the BOM is stripped (utf-8-sig decodes it).
    text = raw.decode("utf-8-sig")
    assert "保存更改" in text


def test_read_queries_skips_malformed_line(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text(
        '{"query": "apple"}\n'
        '{"query": "pear", bad}\n'
        '{"query": "kiwi"}\n'
    )
    skipped: list[tuple[int, str]] = []
    queries = BatchRunner._read_queries(
        p, "query", on_skip=lambda ln, msg: skipped.append((ln, msg)),
    )
    assert queries == ["apple", "kiwi"]
    assert len(skipped) == 1
    assert skipped[0][0] == 2  # the real file line number, not a byte offset


def test_read_queries_strict_raises_with_file_and_line(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"query": "apple"}\n{"query": "pear", bad}\n')
    try:
        BatchRunner._read_queries(p, "query", strict=True)
    except ValueError as e:
        assert f"{p}:2" in str(e)
    else:
        raise AssertionError("strict mode should have raised on the malformed line")


def test_read_queries_resolves_common_field_aliases(tmp_path):
    # A file keyed on prompt/input/question/text works without --query-field.
    p = tmp_path / "in.jsonl"
    p.write_text(
        '{"prompt": "a"}\n'
        '{"input": "b"}\n'
        '{"question": "c"}\n'
        '{"text": "d"}\n'
    )
    assert BatchRunner._read_queries(p, "query") == ["a", "b", "c", "d"]


def test_read_queries_explicit_field_wins_over_alias(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"q": "explicit", "prompt": "alias"}\n')
    assert BatchRunner._read_queries(p, "q") == ["explicit"]


def test_read_queries_reports_empty_row_with_keys(tmp_path):
    # A row carrying no query text is reported with the fields it did have and
    # is not returned as an empty query. With no usable row left, the read fails
    # instead of sending empty prompts to the model.
    p = tmp_path / "in.jsonl"
    p.write_text('{"note": "hi", "id": 3}\n')
    empties: list[tuple[int, list[str]]] = []
    with pytest.raises(ValueError, match="No queries found"):
        BatchRunner._read_queries(
            p, "query", on_empty=lambda ln, keys: empties.append((ln, keys)),
        )
    assert empties == [(1, ["id", "note"])]


def test_read_queries_empty_row_dropped_but_others_kept(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text('{"note": "hi"}\n{"query": "real"}\n')
    empties: list[tuple[int, list[str]]] = []
    queries = BatchRunner._read_queries(
        p, "query", on_empty=lambda ln, keys: empties.append((ln, keys)),
    )
    assert queries == ["real"]
    assert empties == [(1, ["note"])]


def test_read_queries_alias_resolution_in_csv(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("prompt\nhello\nworld\n")
    assert BatchRunner._read_queries(p, "query") == ["hello", "world"]
