"""Independent JSONL log for control/handoff events.

The phase-1 event log (``events.jsonl``) is owned by the shared event module and
its event-type table is closed. Control events therefore live in their own file
next to it: ``runtime/runs/<run_id>/control_events.jsonl``. They can be joined
to session events by run id and time, and they are not fed back into the model.

The shared integration request (adding ``control_epoch``/``page_epoch`` and
handoff fields to ``events.jsonl``) is recorded in the batch report rather than
implemented here.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from ..clock import Clock, SystemClock
from ..redaction import DEFAULT_REDACTOR, Redactor

CONTROL_EVENT_SCHEMA_VERSION = "webagent.control.v1"


class ControlEventLog:
    """Append-only control event log with an explicit schema."""

    def __init__(
        self,
        run_id: str,
        path: str | Path,
        *,
        clock: Clock | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self.run_id = run_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self.redactor = redactor or DEFAULT_REDACTOR
        self._start_monotonic = self.clock.monotonic()
        self._lock = threading.RLock()
        self._closed = False
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def append(
        self,
        event_type: str,
        *,
        control_epoch: int | None = None,
        page_epoch: int | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        reason: str | None = None,
        source: str | None = None,
        action_id: str | None = None,
        observation_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": CONTROL_EVENT_SCHEMA_VERSION,
            "event_id": "ctl_" + uuid.uuid4().hex,
            "run_id": self.run_id,
            "time": self.clock.iso(self.clock.now()),
            "elapsed_ms": max(
                0, int(round((self.clock.monotonic() - self._start_monotonic) * 1000))
            ),
            "event_type": str(event_type),
            "control_epoch": control_epoch,
            "page_epoch": page_epoch,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "source": source,
            "action_id": action_id,
            "observation_id": observation_id,
            "details": dict(details or {}),
        }
        line = json.dumps(
            self.redactor.redact(record), ensure_ascii=False, separators=(",", ":")
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("control event log is closed")
            self._handle.write(line + "\n")
            self._handle.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._handle.flush()
            finally:
                self._handle.close()

    def __enter__(self) -> "ControlEventLog":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_control_events(path: str | Path) -> list[dict[str, Any]]:
    """Read control events, rejecting a malformed line instead of guessing."""

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid control event JSON on line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"control event line {line_number} is not an object")
        events.append(value)
    return events


__all__ = [
    "CONTROL_EVENT_SCHEMA_VERSION",
    "ControlEventLog",
    "read_control_events",
]
