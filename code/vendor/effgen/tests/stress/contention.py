"""Harness for driving one shared object from many threads and counting exactly.

A contention scenario runs N workers against a single shared object — the tool
registry, the provider registry, one adapter, the cost tracker, the session
store, a running server — and reports quantities that have an exact right
answer: how many calls were recorded, how many tokens were folded onto a total,
how many distinct instances a lazy accessor handed out, how many reads failed.

The assertion is equality, not a bound. A counter that loses one update in three
thousand is a counter no invoice can be built from, so
:meth:`ContentionReport.assert_exact` fails on any difference and prints the
whole table so the failing row is visible next to the ones that held.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ContentionReport",
    "Measurement",
    "barrier_map",
    "collect_exceptions",
    "tight_switching",
]


@contextmanager
def tight_switching(interval: float = 1e-6) -> Iterator[None]:
    """Shorten the interpreter's thread switch interval for the block.

    A read-modify-write that is not atomic still produces the right answer most
    of the time at the default 5 ms interval, because a worker usually finishes
    its whole loop body inside one slice. Shortening the interval makes the
    interleaving that a loaded machine, a slower call, or a free-threaded build
    produces anyway happen reliably here, so the counter is measured rather
    than sampled. The previous value is restored on the way out.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


@dataclass(frozen=True)
class Measurement:
    """One counted quantity and the value it has to equal.

    Args:
        name: What was counted, e.g. ``"total_tokens"``.
        measured: What the run actually produced.
        expected: What it must equal for the scenario to pass.
        detail: Optional context shown beside the row when it fails.
    """

    name: str
    measured: Any
    expected: Any
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the measured value equals the expected one."""
        if isinstance(self.measured, float) or isinstance(self.expected, float):
            return abs(float(self.measured) - float(self.expected)) < 1e-9
        return bool(self.measured == self.expected)

    def row(self) -> str:
        """One line of the report table."""
        mark = "ok  " if self.ok else "FAIL"
        line = f"  {mark} {self.name:<28} measured={self.measured!r:<18} expected={self.expected!r}"
        return f"{line}  {self.detail}" if self.detail else line


@dataclass
class ContentionReport:
    """The measurements one contention scenario produced.

    Args:
        scenario: Scenario name, used in the printed table.
        workers: How many threads (or tasks) contended.
        iterations: How many operations each worker performed.
        measurements: The counted quantities.
        notes: Free-form context printed under the table (timings, samples of
            the errors that were raised, the provider that was used).
    """

    scenario: str
    workers: int
    iterations: int
    measurements: list[Measurement] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, measured: Any, expected: Any, detail: str = "") -> None:
        """Record one measured quantity and the value it must equal."""
        self.measurements.append(Measurement(name, measured, expected, detail))

    @property
    def failures(self) -> list[Measurement]:
        """The measurements whose value differs from what was expected."""
        return [m for m in self.measurements if not m.ok]

    def table(self) -> str:
        """The full report: every row, then the notes."""
        head = f"[{self.scenario}] workers={self.workers} iterations={self.iterations}"
        lines = [head, *(m.row() for m in self.measurements)]
        for key, value in self.notes.items():
            lines.append(f"  note {key}: {value}")
        return "\n".join(lines)

    def assert_exact(self) -> None:
        """Fail with the whole table when any measured value is off."""
        if self.failures:
            raise AssertionError(
                f"{len(self.failures)} of {len(self.measurements)} measurements "
                f"are off in {self.scenario}:\n{self.table()}"
            )


def barrier_map(
    workers: int,
    fn: Callable[[int], Any],
    *,
    timeout: float = 300.0,
) -> list[Any]:
    """Run ``fn(i)`` on *workers* threads that all start at the same instant.

    The barrier matters: without it the pool starts threads far enough apart
    that the first one finishes before the last one begins, and a scenario that
    never overlaps cannot observe a race.

    Args:
        workers: Number of threads.
        fn: Called once per worker with the worker's index.
        timeout: Seconds to wait at the barrier before giving up.

    Returns:
        One result per worker, in worker order.
    """
    barrier = threading.Barrier(workers, timeout=timeout)

    def run(index: int) -> Any:
        barrier.wait()
        return fn(index)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, range(workers)))


def collect_exceptions(results: Iterable[Any]) -> list[str]:
    """Return ``"Type: message"`` for every exception in *results*."""
    return [
        f"{type(r).__name__}: {r}"
        for r in results
        if isinstance(r, BaseException)
    ]
