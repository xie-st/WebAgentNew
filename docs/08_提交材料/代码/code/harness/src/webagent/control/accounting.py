"""Model-call accounting that survives a pause.

The three counters keep the phase-1 meanings:

* ``platform_calls`` — distinct platform request ids this process saw.
* ``local_attempts`` — every attempted model/SDK call, including failures.
* ``failures`` — attempts with a failure type.

Latency is summed separately. A proposal returned during a pause is recorded as
discarded (never a successful action) but its call and latency still count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelCallRecord:
    call_id: str
    control_epoch: int | None
    request_id: str | None
    latency_ms: float
    failure_type: str | None
    paused_at_return: bool
    discarded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "control_epoch": self.control_epoch,
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "failure_type": self.failure_type,
            "paused_at_return": self.paused_at_return,
            "discarded": self.discarded,
        }


@dataclass
class ModelCallAccounting:
    platform_calls: int = 0
    local_attempts: int = 0
    failures: int = 0
    latency_ms_total: float = 0.0
    discarded_proposals: int = 0
    records: list[ModelCallRecord] = field(default_factory=list)

    def record(
        self,
        *,
        call_id: str,
        control_epoch: int | None,
        request_id: str | None,
        latency_ms: float,
        failure_type: str | None = None,
        paused_at_return: bool = False,
    ) -> ModelCallRecord:
        record = ModelCallRecord(
            call_id=str(call_id),
            control_epoch=control_epoch,
            request_id=request_id,
            latency_ms=float(latency_ms),
            failure_type=failure_type,
            paused_at_return=bool(paused_at_return),
        )
        self.local_attempts += 1
        if request_id:
            self.platform_calls += 1
        if failure_type:
            self.failures += 1
        self.latency_ms_total += float(latency_ms)
        self.records.append(record)
        return record

    def mark_discarded(self, call_id: str | None) -> None:
        self.discarded_proposals += 1
        if call_id is None:
            return
        for record in reversed(self.records):
            if record.call_id == call_id:
                record.discarded = True
                return

    def snapshot(self) -> dict[str, Any]:
        return {
            "platform_calls": self.platform_calls,
            "local_attempts": self.local_attempts,
            "failures": self.failures,
            "latency_ms_total": round(self.latency_ms_total, 3),
            "discarded_proposals": self.discarded_proposals,
            "calls": [record.to_dict() for record in self.records],
            "paused_at_return": sum(
                1 for record in self.records if record.paused_at_return
            ),
        }


__all__ = ["ModelCallAccounting", "ModelCallRecord"]
