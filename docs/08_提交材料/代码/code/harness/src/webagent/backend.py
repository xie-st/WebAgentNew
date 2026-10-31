"""Backend interface shared by the official, Playwright and fake backends.

A backend only observes and executes; it never plans and never consumes the
session's proposed action. A2 calls ``execute`` after ``session.consume``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .protocol import ExecutionResult, Observation, WebAction


@runtime_checkable
class Backend(Protocol):
    backend_id: str

    def observe(self) -> Observation:
        """Return a fresh observation of the current page."""

    def execute(self, action: WebAction) -> ExecutionResult:
        """Execute one consumed action and return feedback plus a fresh observation."""

    def close(self) -> None:
        """Release only this backend's own resources; repeated calls are safe."""
