"""E07: separate timing buckets, explicit resume, cancellable wait, deadline.

All durations use ``FakeClock``; no real waiting is needed except the small
timeout/cancel coordination checks, which use sub-second sleeps.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from webagent.clock import FakeClock
from webagent.control import (
    ControlCoordinator,
    ControlEventLog,
    ControlState,
    ControlWaitTimeout,
    FileControlSource,
    MemoryControlSource,
    read_control_events,
)
from webagent.errors import DeadlineExceeded


class AdvancingResumeSource(MemoryControlSource):
    """Returns an explicit resume after advancing the fake clock."""

    def __init__(self, clock: FakeClock, seconds: float, command: str = "resume") -> None:
        super().__init__()
        self.clock = clock
        self.seconds = seconds
        self.command = command
        self.sent = False

    def poll(self) -> str | None:
        command = super().poll()
        if command is not None:
            return command
        if not self.sent:
            self.sent = True
            self.clock.advance(self.seconds)
            return self.command
        return None


def _entered_handoff(clock: FakeClock, source) -> tuple[ControlCoordinator, float]:
    coordinator = ControlCoordinator(
        "e07-timing",
        clock=clock,
        source=source,
        platform_deadline_epoch=clock.now() + 60,
    )
    coordinator.start()
    clock.advance(2)
    assert coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()
    return coordinator, clock.now() + 60


def test_total_human_and_execution_time_reconcile_and_deadline_is_unchanged() -> None:
    clock = FakeClock()
    source = AdvancingResumeSource(clock, 30)
    coordinator, deadline_epoch = _entered_handoff(clock, source)

    assert coordinator.wait_for_user_signal() == "resume"
    coordinator.begin_refresh()
    clock.advance(3)
    coordinator.finish_refresh()
    clock.advance(1)
    snapshot = coordinator.finish("completed").to_dict()

    assert snapshot["total_seconds"] == pytest.approx(36.0)
    assert snapshot["human_wait_seconds"] == pytest.approx(30.0)
    assert snapshot["execution_seconds"] == pytest.approx(6.0)
    assert snapshot["total_seconds"] == pytest.approx(
        snapshot["human_wait_seconds"] + snapshot["execution_seconds"]
    )
    assert snapshot["platform_deadline_epoch"] == pytest.approx(deadline_epoch)
    assert snapshot["official_remaining_seconds"] == pytest.approx(24.0)
    assert snapshot["deadline_extended"] is False
    assert coordinator.timing.platform_deadline_epoch == pytest.approx(deadline_epoch)


def test_timeout_is_not_an_automatic_resume() -> None:
    clock = FakeClock()
    coordinator = ControlCoordinator(
        "e07-timeout", clock=clock, source=MemoryControlSource()
    )
    coordinator.start()
    coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()

    with pytest.raises(ControlWaitTimeout):
        coordinator.wait_for_user_signal(timeout_seconds=0.15)
    assert coordinator.state is ControlState.USER_CONTROL
    snapshot = coordinator.snapshot().to_dict()
    assert snapshot["state"] == "user_control"
    assert snapshot["total_seconds"] >= 0

    coordinator.source.send("resume")
    assert coordinator.wait_for_user_signal(timeout_seconds=0.5) == "resume"
    coordinator.begin_refresh()
    assert coordinator.state is ControlState.REFRESHING


def test_cancel_interrupts_a_handoff_wait() -> None:
    clock = FakeClock()
    coordinator = ControlCoordinator(
        "e07-cancel", clock=clock, source=MemoryControlSource()
    )
    coordinator.start()
    coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()
    outcome: dict[str, str] = {}

    def waiter() -> None:
        outcome["signal"] = coordinator.wait_for_user_signal()

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    time.sleep(0.06)
    assert coordinator.state is ControlState.USER_CONTROL
    coordinator.cancel("user_cancel")
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcome == {"signal": "cancel"}
    assert coordinator.state is ControlState.CANCELLED
    assert coordinator.snapshot().finished is True


def test_file_control_source_resume_and_cancel(tmp_path: Path) -> None:
    clock = FakeClock()
    control_path = tmp_path / "control.jsonl"
    source = FileControlSource(control_path)
    coordinator = ControlCoordinator(
        "e07-file",
        clock=clock,
        source=source,
        event_log=ControlEventLog(
            "e07-file", tmp_path / "control_events.jsonl", clock=clock
        ),
        platform_deadline_epoch=clock.now() + 60,
    )
    coordinator.start()
    coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()
    control_path.write_text("resume\n", encoding="utf-8")
    assert coordinator.wait_for_user_signal(timeout_seconds=1.0) == "resume"
    coordinator.begin_refresh()
    coordinator.finish_refresh()
    assert coordinator.state is ControlState.RUNNING

    coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()
    control_path.write_text("resume\ncancel\n", encoding="utf-8")
    assert coordinator.wait_for_user_signal(timeout_seconds=1.0) == "cancel"
    events = read_control_events(Path(tmp_path) / "control_events.jsonl")
    assert [event["event_type"] for event in events].count("pause_requested") == 2
    assert any(event["event_type"] == "cancel_signal" for event in events)


def test_platform_deadline_still_expires_during_handoff() -> None:
    clock = FakeClock()
    source = MemoryControlSource()
    deadline_epoch = clock.now() + 10
    coordinator = ControlCoordinator(
        "e07-deadline",
        clock=clock,
        source=source,
        platform_deadline_epoch=deadline_epoch,
    )
    coordinator.start()
    clock.advance(2)
    coordinator.request_pause("user_pause", source="user")
    coordinator.begin_drain()
    coordinator.begin_user_control()
    clock.advance(20)

    with pytest.raises(DeadlineExceeded, match="must not be extended|official"):
        coordinator.check_platform_deadline("resume")
    snapshot = coordinator.block("deadline_during_handoff").to_dict()
    assert snapshot["platform_deadline_epoch"] == pytest.approx(deadline_epoch)
    assert snapshot["deadline_extended"] is False
    assert snapshot["official_remaining_seconds"] == pytest.approx(-12.0)
