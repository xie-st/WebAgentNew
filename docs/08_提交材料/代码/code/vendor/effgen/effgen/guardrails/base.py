"""
Base guardrails framework for effGen.

Provides abstract base classes and core infrastructure for input/output
validation, content filtering, and safety guardrails.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GuardrailPosition(Enum):
    """Where in the pipeline a guardrail applies."""
    INPUT = "input"            # Before agent processes user input
    OUTPUT = "output"          # Before returning agent output to user
    TOOL_INPUT = "tool_input"  # Before tool execution
    TOOL_OUTPUT = "tool_output"  # After tool execution


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: Whether the content passed the guardrail check.
        reason: Human-readable explanation (empty string if passed).
        modified_content: If the guardrail modified the content instead of
            rejecting it, this contains the modified version. None if
            content was not modified.
        guardrail_name: Name of the guardrail that produced this result.
        metadata: Additional metadata about the check.
    """
    passed: bool
    reason: str = ""
    modified_content: str | None = None
    guardrail_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Guardrail(ABC):
    """Abstract base class for all guardrails.

    Subclasses must implement ``check()`` which inspects content and returns
    a ``GuardrailResult`` indicating whether the content is acceptable.
    """

    def __init__(
        self,
        name: str = "",
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.positions = positions or [
            GuardrailPosition.INPUT,
            GuardrailPosition.OUTPUT,
        ]
        self.enabled = enabled

    @abstractmethod
    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        """Check content against this guardrail.

        Args:
            content: The text content to check.
            **kwargs: Additional context (e.g. tool_name, position).

        Returns:
            GuardrailResult indicating pass/fail and reason.
        """
        ...

    def applies_to(self, position: GuardrailPosition) -> bool:
        """Return True if this guardrail applies at the given position."""
        return self.enabled and position in self.positions


class GuardrailChain:
    """Runs multiple guardrails in sequence, short-circuiting on failure.

    Guardrails are executed in order. If any guardrail fails, the chain
    stops immediately and returns that failure result. If a guardrail
    modifies content (passes but provides ``modified_content``), subsequent
    guardrails receive the modified version.
    """

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self.guardrails: list[Guardrail] = guardrails or []

    def add(self, guardrail: Guardrail) -> GuardrailChain:
        """Add a guardrail to the chain. Returns self for chaining."""
        self.guardrails.append(guardrail)
        return self

    def check(
        self,
        content: str,
        position: GuardrailPosition | None = None,
        **kwargs: Any,
    ) -> GuardrailResult:
        """Run all applicable guardrails on content.

        Args:
            content: Text to check.
            position: If provided, only run guardrails that apply to this position.
            **kwargs: Passed through to each guardrail's check().

        Returns:
            GuardrailResult — passed=True only if ALL guardrails pass.
        """
        start = time.monotonic()
        current_content = content
        results: list[GuardrailResult] = []
        # Aggregate redaction detail from any modifying guardrail (e.g.
        # PIIGuardrail in redact mode) so a caller can audit what a chain removed
        # without reaching into each individual result.
        pii_types: list[str] = []
        pii_counts: dict[str, int] = {}

        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue
            if position is not None and not guardrail.applies_to(position):
                continue

            result = guardrail.check(current_content, **kwargs)
            result.guardrail_name = guardrail.name
            results.append(result)

            if not result.passed:
                elapsed = time.monotonic() - start
                logger.debug(
                    f"Guardrail '{guardrail.name}' rejected content "
                    f"({elapsed*1000:.1f}ms): {result.reason}"
                )
                result.metadata["chain_index"] = len(results) - 1
                result.metadata["chain_elapsed_ms"] = elapsed * 1000
                return result

            for t in result.metadata.get("pii_types", []):
                if t not in pii_types:
                    pii_types.append(t)
            for t, n in (result.metadata.get("pii_counts") or {}).items():
                pii_counts[t] = pii_counts.get(t, 0) + n

            # If guardrail modified content, pass modified version to next
            if result.modified_content is not None:
                current_content = result.modified_content

        elapsed = time.monotonic() - start
        metadata: dict[str, Any] = {
            "checks_run": len(results),
            "chain_elapsed_ms": elapsed * 1000,
        }
        if pii_types:
            metadata["pii_types"] = pii_types
        if pii_counts:
            metadata["pii_counts"] = pii_counts
        return GuardrailResult(
            passed=True,
            reason="",
            modified_content=current_content if current_content != content else None,
            guardrail_name="GuardrailChain",
            metadata=metadata,
        )
