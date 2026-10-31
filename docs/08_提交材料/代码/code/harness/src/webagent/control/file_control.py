"""File-based run control for pause / resume / cancel.

The CLI never reads the keyboard in a background thread. A user (or a test)
appends one command per line to ``runtime/runs/<run_id>/control.jsonl``:

    pause
    resume
    cancel

JSON objects with a ``command`` field are accepted too. Commands are processed
between actions, so pause can only be accepted when the session has no in-flight
action.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .errors import ControlError


class RunControl:
    _COMMANDS = frozenset({"pause", "resume", "cancel"})

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self._offset = 0
        self._incomplete = ""

    def _read_new(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        if not chunk:
            return []
        data = self._incomplete + chunk
        lines = data.splitlines(keepends=True)
        self._incomplete = ""
        commands: list[str] = []
        for line in lines:
            if not line.endswith(("\n", "\r")):
                self._incomplete = line
                continue
            command = self._parse(line)
            if command is not None:
                commands.append(command)
        return commands

    def _parse(self, line: str) -> str | None:
        text = line.strip().lstrip("\ufeff").strip()
        if not text:
            return None
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict):
                text = str(payload.get("command", "")).strip().lower()
        else:
            text = text.lower()
        return text if text in self._COMMANDS else None

    def poll(self) -> str | None:
        """Return the latest unread valid command, if any."""

        commands = self._read_new()
        return commands[-1] if commands else None

    def wait_for(
        self,
        allowed: set[str],
        *,
        timeout_seconds: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
        interval_seconds: float = 0.2,
    ) -> str | None:
        """Block until one of ``allowed`` arrives, timeout, or no command."""

        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            command = self.poll()
            if command in allowed:
                return command
            if deadline is not None and time.monotonic() >= deadline:
                return None
            sleep(interval_seconds)


__all__ = ["ControlError", "RunControl"]
