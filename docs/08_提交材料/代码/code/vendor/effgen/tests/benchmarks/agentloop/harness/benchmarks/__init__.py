"""Look a benchmark up by key.

The harness these scorers were copied from resolved the same table out of a
YAML file in its own repository. The table is vendored in ``specs.py`` instead,
so a lookup here needs nothing outside this package.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from .base import Benchmark
from .specs import BY_KEY, CATEGORIES, SPECS, BenchmarkSpec


def find_spec(key: str) -> BenchmarkSpec:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown benchmark {key!r}; known: {sorted(BY_KEY)}") from None


@lru_cache(maxsize=None)
def load_benchmark(key: str) -> Benchmark:
    """The scorer for one set, with its label, category and tool list attached."""
    spec = find_spec(key)
    module = importlib.import_module(f"{__name__}.{spec.module}")
    bench: Benchmark = getattr(module, spec.cls)()
    bench.key = spec.key
    bench.label = spec.label
    bench.category = spec.category
    bench.tools = spec.tools
    return bench


__all__ = [
    "Benchmark",
    "BenchmarkSpec",
    "CATEGORIES",
    "SPECS",
    "find_spec",
    "load_benchmark",
]
