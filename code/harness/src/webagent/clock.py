"""Clock helpers.

Wall-clock time is used for ISO timestamps and deadlines; durations and event
elapsed times use a monotonic clock so wall-clock adjustments do not corrupt
them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def monotonic(self) -> float: ...

    def iso(self, epoch_seconds: float) -> str: ...


def iso_utc(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat(timespec="milliseconds")


class SystemClock:
    def now(self) -> float:
        import time

        return time.time()

    def monotonic(self) -> float:
        import time

        return time.monotonic()

    def iso(self, epoch_seconds: float) -> str:
        return iso_utc(epoch_seconds)


class FakeClock:
    """Deterministic clock for offline tests."""

    def __init__(self, epoch_seconds: float = 1_793_000_000.0, monotonic_seconds: float = 1000.0) -> None:
        self._epoch = float(epoch_seconds)
        self._monotonic = float(monotonic_seconds)

    def now(self) -> float:
        return self._epoch

    def monotonic(self) -> float:
        return self._monotonic

    def iso(self, epoch_seconds: float) -> str:
        return iso_utc(epoch_seconds)

    def advance(self, seconds: float) -> None:
        self._epoch += float(seconds)
        self._monotonic += float(seconds)

    def set_epoch(self, epoch_seconds: float) -> None:
        self._epoch = float(epoch_seconds)
