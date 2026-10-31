"""The thirteen sets: key, label, category, tools, and the size of the split.

These values came from the run configuration of the sweep whose records this
package replays. They are data, not policy: the label and category decide how a
table groups, and the tool list decides what an agent is handed, so they have to
match the recorded runs exactly or a fresh run is not comparable to an old one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    category: str
    module: str
    cls: str
    tools: tuple[str, ...]
    full_size: int


SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("bb_easy", "BB-Easy", "Coding", "beyondbench", "BeyondBenchEasy",
                  ("python_exec",), 240),
    BenchmarkSpec("bb_med", "BB-Med", "Coding", "beyondbench", "BeyondBenchMedium",
                  ("python_exec",), 125),
    BenchmarkSpec("bb_hard", "BB-Hard", "Coding", "beyondbench", "BeyondBenchHard",
                  ("python_exec",), 70),
    BenchmarkSpec("gaia", "GAIA", "Agentic", "agentic", "GAIA",
                  ("web_search", "python_exec"), 32),
    BenchmarkSpec("simpleqa", "SimpleQA", "Agentic", "agentic", "SimpleQA",
                  ("web_search",), 50),
    BenchmarkSpec("gsm8k", "GSM8K", "Calculator", "gsm8k", "GSM8K",
                  ("calculator",), 1319),
    BenchmarkSpec("gsmplus", "GSM-PLUS", "Calculator", "gsm8k", "GSMPlus",
                  ("calculator",), 2400),
    BenchmarkSpec("math500", "MATH-500", "Calculator", "math500", "Math500",
                  ("calculator",), 500),
    BenchmarkSpec("locomo", "LoCoMo", "Memory", "memory", "LoCoMo", (), 1986),
    BenchmarkSpec("longmemeval", "LongMemEval", "Memory", "memory", "LongMemEval", (), 500),
    BenchmarkSpec("arc_c", "ARC-C", "Retrieval", "retrieval", "ARCChallenge",
                  ("knowledge_search",), 1172),
    BenchmarkSpec("arc_e", "ARC-E", "Retrieval", "retrieval", "ARCEasy",
                  ("knowledge_search",), 2376),
    BenchmarkSpec("csqa", "CSQA", "Retrieval", "retrieval", "CommonsenseQA",
                  ("knowledge_search",), 1221),
)

BY_KEY: dict[str, BenchmarkSpec] = {spec.key: spec for spec in SPECS}

#: Which sets belong to which category, in the order a table prints them.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "calculator": ("gsm8k", "gsmplus", "math500"),
    "coding": ("bb_easy", "bb_med", "bb_hard"),
    "agentic": ("gaia", "simpleqa"),
    "memory": ("locomo", "longmemeval"),
    "retrieval": ("arc_c", "arc_e", "csqa"),
}

__all__ = ["BenchmarkSpec", "SPECS", "BY_KEY", "CATEGORIES"]
