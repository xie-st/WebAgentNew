"""Structured run events (``events.jsonl``).

Every line carries the phase-1 mandatory fields, including the five contest
fields used for auditing: operation time, target page, action type, result and
model-call count. Proposed and dispatched events never carry an execution
result. All strings pass through :class:`~webagent.redaction.Redactor`.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from .clock import Clock, SystemClock
from .redaction import DEFAULT_REDACTOR, Redactor

EVENT_SCHEMA_VERSION = "webagent.events.v1"

REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "event_id",
    "run_id",
    "time",
    "elapsed_ms",
    "event_type",
    "page_id",
    "url",
    "observation_id",
    "action_id",
    "action_type",
    "result",
    "model_calls_total",
    "model_request_id",
)

# Phase-2 ledger fields. They are optional context on every event: older readers
# ignore them, and a run without a task ledger writes the same nulls it already
# writes for page_id/action_id. The ledger attaches a provider after its first
# snapshot; values are merged for later events only.
STATE_FIELDS: tuple[str, ...] = (
    "state_revision",
    "goal_id",
    "branch_id",
    "plan_version",
)

EVENT_TYPES = frozenset(
    {
        "run_start",
        "run_end",
        "reset",
        "task_start",
        "observation",
        "observation_projection",
        "proposed",
        "dispatched",
        "feedback",
        "expanded",
        "subaction",
        "agent_claim",
        "model_call",
        "paused",
        "resumed",
        "cancelled",
        "deadline",
        "deadline_margin",
        "error",
        "format_correction",
        "artifact_saved",
        "artifact_error",
        "verification",
        "behaviors",
        "recovery",
        "close",
    }
)

RUN_STATUS_SCHEMA_VERSION = "webagent.run_status.v1"


class RunStatusWriter:
    """Atomically maintain a small ``run_status.json`` next to ``events.jsonl``.

    The SDK may kill this process at the platform deadline, so a final
    ``result.json`` or ``run_end`` cannot be guaranteed. This file is rewritten
    after every event and carries the step count, the last event and the model
    call counters, which is what a later reconciliation with the platform result
    needs.
    """

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

    def write(self, status: Mapping[str, Any]) -> Path:
        if not isinstance(status, Mapping):
            raise ValueError("run status must be a mapping")
        payload = dict(status)
        payload.setdefault("schema_version", RUN_STATUS_SCHEMA_VERSION)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("updated_at", self.clock.iso(self.clock.now()))
        data = json.dumps(
            self.redactor.redact(payload), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.part")
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                import os

                os.fsync(handle.fileno())
            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return self.path


class EventLogger:
    def __init__(
        self,
        run_id: str,
        path: str | Path,
        *,
        clock: Clock | None = None,
        redactor: Redactor | None = None,
        schema_version: str = EVENT_SCHEMA_VERSION,
        model_calls_total: Callable[[], int] | None = None,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
        state_context: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.run_id = run_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self.redactor = redactor or DEFAULT_REDACTOR
        self.schema_version = schema_version
        self._model_calls_total_provider = model_calls_total
        self._model_calls = 0
        self._on_event = on_event
        self._state_context = state_context
        self._start_monotonic = self.clock.monotonic()
        self._lock = threading.RLock()
        self._closed = False
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    @property
    def model_calls_total(self) -> int:
        return self._model_calls

    def _calls(self) -> int:
        if self._model_calls_total_provider is not None:
            return int(self._model_calls_total_provider())
        return self._model_calls

    def set_state_context(
        self,
        provider: Callable[[], Mapping[str, Any]] | None,
        *,
        goal_id: str | None = None,
    ) -> None:
        """Attach the task ledger's current revision/branch/plan to later events.

        The provider is read on every emit so revision changes made between an
        action's proposal and its feedback are visible without mutating earlier
        lines. A failing provider never breaks the event log or the run; events
        simply keep the explicit nulls. ``goal_id`` is the active requirement the
        caller is working toward, not a claim that it is verified.
        """

        def provide() -> Mapping[str, Any]:
            values: dict[str, Any] = {}
            if goal_id is not None:
                values["goal_id"] = goal_id
            if provider is not None:
                try:
                    supplied = provider()
                except Exception:  # noqa: BLE001 - diagnostics must survive bad context
                    supplied = None
                if isinstance(supplied, Mapping):
                    values.update(supplied)
            return values

        self._state_context = provide if (provider is not None or goal_id is not None) else None

    def _state_values(self) -> Mapping[str, Any]:
        provider = self._state_context
        if provider is None:
            return {}
        try:
            values = provider()
        except Exception:  # noqa: BLE001 - diagnostics must survive bad context
            return {}
        return values if isinstance(values, Mapping) else {}

    def emit(
        self,
        event_type: str,
        *,
        page_id: str | None = None,
        url: str | None = None,
        observation_id: str | None = None,
        action_id: str | None = None,
        action_type: str | None = None,
        result: Any = None,
        model_request_id: str | None = None,
        model_calls_total: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {event_type!r}")
        self._validate_event(event_type, action_id, action_type, result)
        now = self.clock.now()
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event_id": "evt_" + uuid.uuid4().hex,
            "run_id": self.run_id,
            "time": self.clock.iso(now),
            "elapsed_ms": max(0, int(round((self.clock.monotonic() - self._start_monotonic) * 1000))),
            "event_type": event_type,
            "page_id": page_id,
            "url": url,
            "observation_id": observation_id,
            "action_id": action_id,
            "action_type": action_type,
            "result": result,
            "model_calls_total": self._calls() if model_calls_total is None else int(model_calls_total),
            "model_request_id": model_request_id,
            "state_revision": None,
            "goal_id": None,
            "branch_id": None,
            "plan_version": None,
            # Phase-2 E1 additive context. Older readers ignore these fields;
            # verification/recovery writers may supply their result object here.
            "verification": None,
            "recovery": None,
        }
        record.update(extra)
        state_values = self._state_values()
        for field in STATE_FIELDS:
            if record.get(field) is None and state_values.get(field) is not None:
                record[field] = state_values[field]
        redacted = self.redactor.redact(record)
        line = json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("event log is closed")
            self._handle.write(line + "\n")
            self._handle.flush()
        if self._on_event is not None:
            self._on_event(redacted)
        return redacted

    @staticmethod
    def _validate_event(
        event_type: str,
        action_id: str | None,
        action_type: str | None,
        result: Any,
    ) -> None:
        if event_type in {"proposed", "dispatched"}:
            if result is not None:
                raise ValueError(
                    f"{event_type} must not carry an execution result; use a feedback event"
                )
            if action_id is None or action_type is None:
                raise ValueError(f"{event_type} needs action_id and action_type")
        elif event_type == "feedback":
            if action_id is None or action_type is None:
                raise ValueError("feedback needs action_id and action_type")
            if not isinstance(result, str) or result not in {"observed", "error", "unknown"}:
                raise ValueError("feedback result must be observed, error or unknown")
        elif event_type == "agent_claim":
            if not isinstance(result, Mapping):
                raise ValueError("agent_claim result must describe the claim and reason")
            if any(key in result for key in ("score", "official_score", "passed")):
                raise ValueError("agent_claim must not invent an official result")
        elif event_type == "format_correction":
            if not isinstance(result, Mapping) or "reason" not in result:
                raise ValueError(
                    "format_correction result must name the reason it was sent"
                )
        elif event_type == "artifact_error":
            if not isinstance(result, Mapping) or "kind" not in result or "reason" not in result:
                raise ValueError(
                    "artifact_error result must name the artifact kind and failure reason"
                )

    def emit_model_call(
        self,
        *,
        request_id: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        token_count_source: str,
        latency_ms: float | None,
        failure_type: str | None = None,
        model_name: str | None = None,
        page_id: str | None = None,
        url: str | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._model_calls += 1
            calls = self._model_calls
        return self.emit(
            "model_call",
            page_id=page_id,
            url=url,
            observation_id=observation_id,
            result=None,
            model_request_id=request_id,
            model_calls_total=calls,
            model_input_tokens=input_tokens,
            model_output_tokens=output_tokens,
            token_count_source=token_count_source,
            latency_ms=latency_ms,
            failure_type=failure_type,
            model_name=model_name,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._handle.flush()
            finally:
                self._handle.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read an events file, validating that every line is a JSON object."""

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on event line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"event line {line_number} is not an object")
        events.append(value)
    return events
