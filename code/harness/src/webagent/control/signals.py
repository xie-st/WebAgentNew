"""Explicit control signals (pause / resume / cancel).

Control is signal-driven on purpose: after a handoff there is no timer that
resumes the task. ``resume`` must be an explicit command from the user's control
surface (or the equivalent test signal). A missing signal leaves the task in
``user_control``; the caller can keep waiting or cancel.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Protocol

from .file_control import RunControl

SIGNAL_COMMANDS = frozenset({"pause", "resume", "cancel"})


class ControlSource(Protocol):
    def poll(self) -> str | None:
        """Return the next unread valid command, or None."""

    def close(self) -> None:
        """Release this source's resources; safe to call twice."""


class FileControlSource:
    """The phase-1 ``control.jsonl`` surface, reused unchanged."""

    def __init__(self, path: str | Path) -> None:
        self.run_control = RunControl(path)
        self.path = Path(path)

    def poll(self) -> str | None:
        command = self.run_control.poll()
        return command if command in SIGNAL_COMMANDS else None

    def close(self) -> None:
        return


class MemoryControlSource:
    """Thread-safe queue source used by tests and programmatic control."""

    def __init__(self) -> None:
        self._commands: deque[str] = deque()
        self._lock = threading.Lock()

    def send(self, command: str) -> None:
        if command not in SIGNAL_COMMANDS:
            raise ValueError(f"unknown control command {command!r}")
        with self._lock:
            self._commands.append(command)

    def poll(self) -> str | None:
        with self._lock:
            if not self._commands:
                return None
            return self._commands.popleft()

    def close(self) -> None:
        return


__all__ = [
    "ControlSource",
    "FileControlSource",
    "MemoryControlSource",
    "SIGNAL_COMMANDS",
]
