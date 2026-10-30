# Agent-loop measurement

Two instruments for answering "did that change make the agent better, or only
different?".

Neither is part of the installed package. Both live under `tests/` so they run
in the same suite as everything else and break loudly when the code they measure
moves.

## Replay — re-reading runs that already happened

A benchmark run writes one scored record per sample. `records.py` reads a
directory of them; `replay.py` turns them into tables.

```python
from tests.benchmarks.agentloop.records import RecordTree
from tests.benchmarks.agentloop.replay import rescore, strip_trailing_citations

tree = RecordTree("/path/to/runs")          # <model>/<benchmark>/<system>/
cell = tree.load("14B", "arc_e", "effgen_plus")

print(cell.stats().accuracy)                # 86.74
result = rescore(cell, strip_trailing_citations)
print(result.fired, result.gained, result.broken)   # 205 192 0
```

That last call is the point of the whole thing. `rescore` scores the same
records twice — once as they are, once with a pure function applied to the
answer text — so both arms are the same samples and the run-to-run variation
cancels exactly. The result is a count of samples that changed direction, with
no error bar, instead of a difference between two accuracies that may be smaller
than the noise.

Three habits are built into it rather than left to the reader:

* **firing count before accuracy.** `fired` and `fired_share` come first in the
  result and first in every printed table. A change that fires on most of a set
  is a new answer format, not a fix to a failure mode, whatever the accuracy
  does; the count says so long before the accuracy does.
* **examples, unprompted.** `result.examples` carries the first dozen records
  the change fired on, with the answer before and after and the truth beside
  them. Metrics that counted something other than what their name said have been
  built on twice; reading six matched samples is what catches it.
* **the reader checks itself.** For every cell it loads it re-scores each
  record's own answer and compares against the score already stored on it. Any
  disagreement means the scorer has drifted from the one that produced the file,
  and the cell is refused — not warned about. That single property is what makes
  the two arms of a counterfactual comparable: they come from the same code, so
  a difference between them can only be the transform.

A cell that did not finish is refused too. `allow_partial=True` reads it anyway
and stamps every number it touches with the reason and the sample count.

## Live — running a set and recording what it cost

`live.py` runs one set against one endpoint and writes records in the same
format, so a fresh run reads back through the replay reader with no special
case.

```python
from pathlib import Path
from tests.benchmarks.agentloop.live import LiveRun, run_cell

run_cell(LiveRun(
    bench="gsm8k", model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:8400/v1",       # always explicit
    out_dir=Path("runs/gsm8k-a"), n=200, concurrency=8,
))
```

Beside the fields a recorded run carries, each sample gains three:

| field | what it is |
|---|---|
| `model_wall_s` | seconds inside HTTP calls to the endpoint |
| `framework_wall_s` | `latency_s - model_wall_s` — everything else |
| `framework_cpu_s` | CPU this worker burned, nearly all of it outside those calls |

The split is measured at the HTTP boundary, in `timing.py`, and not by asking
the framework how long it took. The framework is what is being measured; a
number it reports about itself cannot be used to judge it.

Two cautions the code enforces or reports rather than leaving to a footnote:

* **a streamed response is flagged, not estimated.** `send()` returns before a
  server-sent-event body has been read, so the model side of the split would be
  short. Such a run sets `streaming_seen` and `CellStats.split_is_reliable`
  becomes false.
* **above concurrency 1 the wall split holds contention** as well as the
  framework's own work. Both the wall and the CPU figure are always printed,
  with the concurrency in the header; the CPU figure is the stable one.

The endpoint is an argument. If `EFFGEN_BASE_URL`, `OPENAI_BASE_URL` or
`OPENAI_API_BASE` is set — to anything, including the empty string — the runner
exits and names the variable, because a leftover value redirects every
OpenAI-protocol call in the process and the failures then read as an outage at
the provider.

## `harness/` — copied, on purpose

`harness/` holds the answer parsers, the thirteen scorers and the four tools
from the benchmark harness that produced the records this package re-reads. They
are byte-for-byte copies, listed in `harness/__init__.py`, and they are copies
rather than improvements deliberately: a better parser is a different parser,
and a different parser changes every number for reasons that have nothing to do
with the code under measurement.

`test_replay_instrument.py` hashes each copied file against its origin when the
origin is reachable, and says which check it could not run when it is not.

The only files there that are not copies are `config.py` (where the copies look
for their data), `benchmarks/specs.py` (the set list, which the origin kept in a
YAML file in another repository) and the two `__init__.py` files.

## Sample sets

`fixtures/samples/<set>.jsonl` holds the items each set runs, with their
original ids and in their original order. `fixtures/samples/SOURCE.md` says
where each came from and how it was checked.

## Running the tests

```bash
pytest tests/benchmarks/agentloop -q
```

They are offline and take about fifteen seconds: no network beyond a stub server
on a loopback port, no GPU, no recorded runs, no home directory. Checks that
need recorded runs or the origin of the copies are skipped with a printed reason
naming the environment variable that would enable them —
`AGENTLOOP_RECORDS` and `AGENTLOOP_HARNESS_SOURCE` — because a guard that skips
quietly reads exactly like a guard that passed.

The endpoint the live tests use is a stub with a stated delay and a stated usage
block. It is not a stand-in for a model: it makes no claim about how an agent
behaves, and every such claim is checked against a real model. It is a measuring
standard — the only way to plant a number a clock has to recover, since a real
model cannot be asked to take exactly 250 milliseconds.
