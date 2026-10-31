"""Reading, re-scoring and re-running agent benchmark cells.

Two instruments live here.

**Replay** (``records.py``, ``replay.py``) reads the per-sample records a
benchmark run wrote and re-derives everything from them: accuracy, the
stop-reason distribution, token and latency means, and — the reason it exists —
a *counterfactual*, which scores the same records with and without a change to
the answer text. A counterfactual has no sampling noise in it at all, because
both arms are the same samples, so a change that post-processes an answer can be
settled as "+27 right, 0 broken" instead of as a delta inside the run-to-run
band.

**Live** (``timing.py``, ``live.py``) runs one set against one endpoint and
writes the same record format, so a fresh run and a recorded one are read by the
same code. It also separates the time spent inside HTTP calls to the model from
the time spent in the framework around them, which is the only way to say what
an agent's own code costs.

The scorers under ``harness/`` are copies of the ones that produced the recorded
runs, kept identical on purpose: a changed parser changes the numbers for
reasons that have nothing to do with the agent.
"""

from __future__ import annotations

#: Bumped when the record or summary format the live runner writes changes in a
#: way a reader has to know about. Readers accept a record with no version, which
#: is what an older recorded run looks like.
INSTRUMENT_VERSION = 1

__all__ = ["INSTRUMENT_VERSION"]
