"""Answer scoring and tools, copied from the harness that produced the records.

The modules under this package are byte-for-byte copies of the benchmark
harness they came from. They are copies rather than improvements on purpose: a
changed parser changes the numbers for reasons that have nothing to do with the
framework, and every comparison against an already-recorded run dies with it.

``test_replay_instrument.py`` re-derives that claim by hashing each copied file
against its source when the source tree is reachable, and says loudly when it
is not.

The only files here that are *not* copies are this one, ``config.py`` and
``benchmarks/__init__.py`` and ``benchmarks/specs.py``, which replace a YAML
registry and a repository layout that do not exist in this tree.
"""

from __future__ import annotations

#: Files copied verbatim, relative to this package, and their path in the
#: source harness. Read by the drift check.
COPIED_FILES: dict[str, str] = {
    "types.py": "types.py",
    "tools/__init__.py": "tools/__init__.py",
    "tools/spec.py": "tools/spec.py",
    "tools/builtin.py": "tools/builtin.py",
    "benchmarks/base.py": "benchmarks/base.py",
    "benchmarks/scoring.py": "benchmarks/scoring.py",
    "benchmarks/retrieval.py": "benchmarks/retrieval.py",
    "benchmarks/gsm8k.py": "benchmarks/gsm8k.py",
    "benchmarks/math500.py": "benchmarks/math500.py",
    "benchmarks/agentic.py": "benchmarks/agentic.py",
    "benchmarks/beyondbench.py": "benchmarks/beyondbench.py",
    "benchmarks/memory.py": "benchmarks/memory.py",
}

__all__ = ["COPIED_FILES"]
