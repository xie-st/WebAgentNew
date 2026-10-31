"""Fault-plan validation and deterministic fault injection.

A case manifest may contain a ``fault_plan``. Faults are activated only by
observed site activity after the configured event count, never by the Agent
calling a control endpoint. Activation writes a hidden truth row and changes
only site behaviour (delay, auth overlay, duplicate control). No fault API is
rendered and no fault label is placed in the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .store import SiteStore

SUPPORTED_FAULT_TYPES = frozenset(
    {
        "slow_response",
        "feedback_delay",
        "login_overlay",
        "verification_prompt",
        "duplicate_button",
    }
)
RESERVED_FAULT_TYPES = frozenset(
    {
        "missing_control",
        "renumber",
        "draft_lost",
        "shared_object_modified",
        "cool_down",
        "tab_closed",
    }
)
ALL_FAULT_TYPES = SUPPORTED_FAULT_TYPES | RESERVED_FAULT_TYPES

TRIGGER_EVENTS = frozenset(
    {
        "page_view",
        "list_view",
        "view_item",
        "draft_saved",
        "draft_cancelled",
        "comment_sent",
        "auth_blocked",
    }
)


class FaultPlanError(ValueError):
    """Invalid or not-yet-enabled fault definition."""


@dataclass(frozen=True)
class FaultDefinition:
    fault_id: str
    type: str
    enabled: bool
    trigger_event: str
    trigger_after: int
    params: Mapping[str, Any] = field(default_factory=dict)
    note: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FaultDefinition":
        if not isinstance(raw, Mapping):
            raise FaultPlanError("fault must be a mapping")
        fault_id = str(raw.get("fault_id", "")).strip()
        if not fault_id:
            raise FaultPlanError("fault_id is required")
        fault_type = str(raw.get("type", "")).strip()
        if fault_type not in ALL_FAULT_TYPES:
            raise FaultPlanError(f"unknown fault type {fault_type!r}")
        enabled = bool(raw.get("enabled", False))
        if enabled and fault_type in RESERVED_FAULT_TYPES:
            raise FaultPlanError(
                f"fault type {fault_type!r} is reserved for a later batch; "
                "keep it disabled"
            )
        trigger = raw.get("trigger")
        if trigger is None:
            trigger_event = str(raw.get("trigger_event", ""))
            trigger_after = int(raw.get("trigger_after", 0))
        elif isinstance(trigger, Mapping):
            trigger_event = str(trigger.get("event", ""))
            trigger_after = int(trigger.get("after", 0))
        else:
            raise FaultPlanError("fault trigger must be a mapping")
        params = raw.get("params") or {}
        if not isinstance(params, Mapping):
            raise FaultPlanError("fault params must be a mapping")
        if enabled:
            if trigger_event not in TRIGGER_EVENTS:
                raise FaultPlanError(
                    f"fault {fault_id!r} has unsupported trigger event {trigger_event!r}"
                )
            if trigger_after < 1:
                raise FaultPlanError(
                    f"fault {fault_id!r} trigger.after must be a positive integer"
                )
        definition = cls(
            fault_id=fault_id,
            type=fault_type,
            enabled=enabled,
            trigger_event=trigger_event,
            trigger_after=trigger_after,
            params=dict(params),
            note=None if raw.get("note") is None else str(raw.get("note")),
        )
        definition._validate_params()
        return definition

    def _validate_params(self) -> None:
        if self.type in ("slow_response", "feedback_delay"):
            delay = self.params.get("delay_ms", 250)
            if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
                raise FaultPlanError("delay_ms must be a positive integer")
            if delay > 60_000:
                raise FaultPlanError("delay_ms is unreasonably large")
            times = self.params.get("times", 1)
            if not isinstance(times, int) or isinstance(times, bool) or times < 1:
                raise FaultPlanError("times must be a positive integer")
        elif self.type == "duplicate_button":
            count = self.params.get("count", 2)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise FaultPlanError("duplicate_button count must be a positive integer")

    def to_store_row(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "type": self.type,
            "enabled": self.enabled,
            "trigger_event": self.trigger_event,
            "trigger_after": self.trigger_after,
            "params": dict(self.params),
            "note": self.note,
        }


def normalize_fault_plan(
    raw_faults: Iterable[Mapping[str, Any]] | None,
) -> tuple[FaultDefinition, ...]:
    definitions: list[FaultDefinition] = []
    seen: set[str] = set()
    for raw in raw_faults or ():
        definition = FaultDefinition.from_mapping(raw)
        if definition.fault_id in seen:
            raise FaultPlanError(f"duplicate fault_id {definition.fault_id!r}")
        seen.add(definition.fault_id)
        definitions.append(definition)
    return tuple(definitions)


class FaultController:
    """Applies activated faults using only persistent site state."""

    def __init__(self, store: SiteStore) -> None:
        self.store = store

    # ------------------------------------------------------------- activation
    def after_event(self, event: str) -> list[dict[str, Any]]:
        if event not in TRIGGER_EVENTS:
            return []
        activated: list[dict[str, Any]] = []
        count = self.store.activity_count(event)
        for fault in self.store.faults():
            if not fault["enabled"] or fault["trigger_event"] != event:
                continue
            if count < fault["trigger_after"]:
                continue
            activated.append(self._activate(fault, count))
        return activated

    def _activate(self, fault: Mapping[str, Any], observed_count: int) -> dict[str, Any]:
        fault_type = str(fault["type"])
        params = dict(fault.get("params") or {})
        truth = {
            "fault_id": fault["fault_id"],
            "type": fault_type,
            "trigger_event": fault["trigger_event"],
            "trigger_after": fault["trigger_after"],
            "observed_count": observed_count,
            "params": params,
        }
        self.store.record_fault_event(
            fault_id=str(fault["fault_id"]),
            fault_type=fault_type,
            trigger_event=str(fault["trigger_event"]),
            observed_count=observed_count,
            truth=truth,
        )
        self.store.mark_fault_fired(str(fault["fault_id"]))
        if fault_type == "login_overlay":
            self.store.state_set("auth.login_required", "1")
        elif fault_type == "verification_prompt":
            self.store.state_set("auth.verification_required", "1")
        elif fault_type in ("slow_response", "feedback_delay"):
            self.store.state_set(
                f"fault.{fault['fault_id']}.remaining", str(int(params.get("times", 1)))
            )
        elif fault_type == "duplicate_button":
            self.store.state_set(
                f"fault.{fault['fault_id']}.duplicate_count", str(int(params.get("count", 2)))
            )
        return truth

    def inject(self, raw: Mapping[str, Any]) -> FaultDefinition:
        definition = FaultDefinition.from_mapping(raw)
        if not definition.enabled:
            raise FaultPlanError("injected fault must be enabled")
        self.store.add_fault(definition.to_store_row())
        return definition

    # ---------------------------------------------------------------- effects
    def consume_delay_ms(self, channel: str) -> int:
        """Consume one pending delay for ``channel`` ('page' or 'feedback')."""

        wanted = "slow_response" if channel == "page" else "feedback_delay"
        total = 0
        for fault in self.store.faults():
            if fault["type"] != wanted:
                continue
            key = f"fault.{fault['fault_id']}.remaining"
            remaining = int(self.store.state_get(key, "0") or "0")
            if remaining <= 0:
                continue
            self.store.state_set(key, str(remaining - 1))
            total += int((fault.get("params") or {}).get("delay_ms", 250))
        return total

    def duplicate_button_count(self) -> int:
        total = 0
        for fault in self.store.faults():
            if fault["type"] != "duplicate_button":
                continue
            key = f"fault.{fault['fault_id']}.duplicate_count"
            total += int(self.store.state_get(key, "0") or "0")
        return total

    def auth_kind(self) -> str | None:
        if self.store.state_get("auth.login_required", "0") == "1":
            return "login"
        if self.store.state_get("auth.verification_required", "0") == "1":
            return "verification"
        return None

    def force_auth(self, kind: str) -> None:
        if kind == "login":
            self.store.state_set("auth.login_required", "1")
        elif kind == "verification":
            self.store.state_set("auth.verification_required", "1")
        else:
            raise FaultPlanError(f"unknown auth kind {kind!r}")

    def clear_auth(self) -> None:
        self.store.state_delete("auth.login_required")
        self.store.state_delete("auth.verification_required")


__all__ = [
    "ALL_FAULT_TYPES",
    "FaultController",
    "FaultDefinition",
    "FaultPlanError",
    "RESERVED_FAULT_TYPES",
    "SUPPORTED_FAULT_TYPES",
    "normalize_fault_plan",
]
