"""
Tool safety guardrails for effGen.

Provides guardrails for validating tool inputs against parameter specs,
sanitizing tool outputs, and enforcing per-tool permissions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import Guardrail, GuardrailPosition, GuardrailResult

logger = logging.getLogger(__name__)


class ToolInputGuardrail(Guardrail):
    """Validate tool inputs against the tool's ParameterSpec before execution.

    When a tool has defined parameter specifications, this guardrail
    validates that the input JSON conforms to them (types, required fields,
    value ranges).
    """

    def __init__(
        self,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="ToolInputGuardrail",
            positions=[GuardrailPosition.TOOL_INPUT],
            enabled=enabled,
        )

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        """Validate the JSON tool input against the tool's parameter schema."""
        tool = kwargs.get("tool")
        if tool is None:
            return GuardrailResult(passed=True)

        # Parse input JSON
        try:
            input_dict = json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, TypeError):
            # If content is not JSON, let the tool handle it
            return GuardrailResult(passed=True)

        if not isinstance(input_dict, dict):
            return GuardrailResult(passed=True)

        # Get parameter specs from tool metadata
        metadata = getattr(tool, "metadata", None)
        if metadata is None:
            return GuardrailResult(passed=True)
        parameters = getattr(metadata, "parameters", None)
        if not parameters:
            return GuardrailResult(passed=True)

        # Validate each parameter spec
        errors: list[str] = []
        for param_spec in parameters:
            value = input_dict.get(param_spec.name)
            is_valid, error_msg = param_spec.validate(value)
            if not is_valid:
                errors.append(error_msg)

        if errors:
            return GuardrailResult(
                passed=False,
                reason=f"Tool input guardrail: {'; '.join(errors)}",
                metadata={"validation_errors": errors},
            )

        return GuardrailResult(passed=True)


class ToolOutputGuardrail(Guardrail):
    """Sanitize tool outputs — optionally strip PII and enforce size limits."""

    def __init__(
        self,
        max_output_length: int = 100_000,
        strip_pii: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="ToolOutputGuardrail",
            positions=[GuardrailPosition.TOOL_OUTPUT],
            enabled=enabled,
        )
        self.max_output_length = max_output_length
        self.strip_pii = strip_pii

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        """Truncate over-long tool output and optionally strip PII from it."""
        modified = content

        # Truncate if too long
        if len(modified) > self.max_output_length:
            modified = modified[: self.max_output_length] + "\n... [output truncated]"

        # Optionally strip PII patterns from output
        if self.strip_pii:
            # SSN
            modified = re.sub(
                r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
                "[SSN REDACTED]",
                modified,
            )
            # Email
            modified = re.sub(
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                "[EMAIL REDACTED]",
                modified,
            )

        if modified != content:
            return GuardrailResult(
                passed=True,
                modified_content=modified,
                reason="Tool output guardrail: output was sanitized",
            )

        return GuardrailResult(passed=True)


class ToolPermissionGuardrail(Guardrail):
    """Per-tool allow/deny/require_approval permission system.

    Configure which tools are allowed, denied, or require user approval.
    The approval callback is optional — if ``require_approval`` tools are
    called without a callback, they are denied by default.
    """

    def __init__(
        self,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        require_approval: list[str] | None = None,
        approval_callback: Any | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="ToolPermissionGuardrail",
            positions=[GuardrailPosition.TOOL_INPUT],
            enabled=enabled,
        )
        self.allow = set(allow) if allow else None  # None = allow all
        self.deny = set(deny) if deny else set()
        self.require_approval = set(require_approval) if require_approval else set()
        self.approval_callback = approval_callback

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        """Enforce the tool allow/deny lists and the approval requirement."""
        tool_name = kwargs.get("tool_name", "")

        # Check deny list first
        if tool_name in self.deny:
            return GuardrailResult(
                passed=False,
                reason=f"Tool permission guardrail: tool '{tool_name}' is denied",
                metadata={"tool_name": tool_name, "action": "denied"},
            )

        # Check allow list (if set, only listed tools are permitted)
        if self.allow is not None and tool_name not in self.allow:
            return GuardrailResult(
                passed=False,
                reason=f"Tool permission guardrail: tool '{tool_name}' is not in the allow list",
                metadata={"tool_name": tool_name, "action": "not_allowed"},
            )

        # Check require_approval
        if tool_name in self.require_approval:
            if self.approval_callback is not None:
                try:
                    approved = self.approval_callback(tool_name, content)
                    if not approved:
                        return GuardrailResult(
                            passed=False,
                            reason=f"Tool permission guardrail: approval denied for tool '{tool_name}'",
                            metadata={"tool_name": tool_name, "action": "approval_denied"},
                        )
                except Exception as e:
                    logger.warning(f"Approval callback error for '{tool_name}': {e}")
                    return GuardrailResult(
                        passed=False,
                        reason=f"Tool permission guardrail: approval callback failed for '{tool_name}'",
                        metadata={"tool_name": tool_name, "action": "approval_error"},
                    )
            else:
                # No callback — deny by default
                return GuardrailResult(
                    passed=False,
                    reason=f"Tool permission guardrail: tool '{tool_name}' requires approval but no callback configured",
                    metadata={"tool_name": tool_name, "action": "no_approval_callback"},
                )

        return GuardrailResult(passed=True)
