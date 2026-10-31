"""
Human-in-the-Loop interaction points for effGen agents.

Provides approval workflows, human input requests, and choice selection
that allow agents to pause execution and request human interaction.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ApprovalMode(Enum):
    """When to require human approval for tool execution."""
    ALWAYS = "always"
    FIRST_TIME = "first_time"
    NEVER = "never"
    DANGEROUS_ONLY = "dangerous_only"


class ApprovalDecision(Enum):
    """Result of a human approval request."""
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class HumanApproval:
    """
    Request human approval before proceeding with an action.

    Attributes:
        tool_name: Name of the tool requesting approval.
        tool_args: Arguments that will be passed to the tool.
        reason: Why approval is needed.
        timeout: Seconds to wait before applying default_decision.
        default_decision: Decision if timeout expires (approve or deny).
    """
    tool_name: str
    tool_args: str
    reason: str = ""
    timeout: float = 0.0  # 0 = wait forever
    default_decision: ApprovalDecision = ApprovalDecision.DENIED

    def request(
        self,
        callback: Callable[[str, str], bool] | None = None,
    ) -> ApprovalDecision:
        """
        Request approval from a human via callback.

        Args:
            callback: Function(tool_name, tool_args) -> bool.
                      If None, returns default_decision.

        Returns:
            ApprovalDecision indicating the human's response.
        """
        if callback is None:
            logger.debug(
                "No approval callback set; using default decision '%s' for tool '%s'",
                self.default_decision.value, self.tool_name,
            )
            return self.default_decision

        try:
            if self.timeout > 0:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(callback, self.tool_name, self.tool_args)
                    try:
                        approved = future.result(timeout=self.timeout)
                    except concurrent.futures.TimeoutError:
                        logger.debug(
                            "Approval timeout for tool '%s'; default=%s",
                            self.tool_name, self.default_decision.value,
                        )
                        return ApprovalDecision.TIMEOUT
            else:
                approved = callback(self.tool_name, self.tool_args)

            decision = ApprovalDecision.APPROVED if approved else ApprovalDecision.DENIED
            logger.debug("Approval decision for tool '%s': %s", self.tool_name, decision.value)
            return decision
        except Exception as e:
            logger.error("Approval callback error for tool '%s': %s", self.tool_name, e)
            return self.default_decision


@dataclass
class HumanInput:
    """
    Request free-text input from a human.

    Attributes:
        prompt: Question or instruction to display.
        timeout: Seconds to wait (0 = forever).
        default: Default value if timeout expires or no callback.
    """
    prompt: str
    timeout: float = 0.0
    default: str = ""

    def request(
        self,
        callback: Callable[[str], str] | None = None,
    ) -> str:
        """
        Request input from a human via callback.

        Args:
            callback: Function(prompt) -> str response.

        Returns:
            The human's input string, or default on timeout/error.
        """
        if callback is None:
            return self.default

        try:
            if self.timeout > 0:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(callback, self.prompt)
                    try:
                        return future.result(timeout=self.timeout)
                    except concurrent.futures.TimeoutError:
                        logger.debug("Input timeout; using default: %r", self.default)
                        return self.default
            else:
                return callback(self.prompt)
        except Exception as e:
            logger.error("Input callback error: %s", e)
            return self.default


@dataclass
class HumanChoice:
    """
    Present multiple options and let a human choose.

    Attributes:
        prompt: Question to display.
        options: List of option strings.
        default: Index of the default option (0-based).
        timeout: Seconds to wait (0 = forever).
    """
    prompt: str
    options: list[str]
    default: int = 0
    timeout: float = 0.0

    def request(
        self,
        callback: Callable[[str, list[str]], int] | None = None,
    ) -> int:
        """
        Request a choice from a human via callback.

        Args:
            callback: Function(prompt, options) -> int (selected index).

        Returns:
            Index of the chosen option, or default on timeout/error.
        """
        if not self.options:
            return -1

        if callback is None:
            return self.default

        try:
            if self.timeout > 0:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(callback, self.prompt, self.options)
                    try:
                        idx = future.result(timeout=self.timeout)
                    except concurrent.futures.TimeoutError:
                        logger.debug("Choice timeout; using default index %d", self.default)
                        return self.default
            else:
                idx = callback(self.prompt, self.options)

            # Clamp to valid range
            if not isinstance(idx, int) or idx < 0 or idx >= len(self.options):
                logger.warning("Invalid choice index %r; using default %d", idx, self.default)
                return self.default
            return idx
        except Exception as e:
            logger.error("Choice callback error: %s", e)
            return self.default


# --- Dangerous tool detection ---

# Tools whose names match these substrings are considered dangerous by default
DANGEROUS_TOOL_KEYWORDS: set[str] = {
    "bash", "shell", "exec", "execute", "code_executor",
    "file_write", "file_delete", "system",
    # Stateful side effects with real-world consequences (money movement,
    # cancellation) — a tool author can still mark any tool `requires_approval`
    # explicitly (see `@tool(requires_approval=True)`), but a name in this list
    # is gated under `approval_mode="dangerous_only"` even without that.
    "refund", "charge", "payment", "purchase", "transfer", "cancel", "delete",
}


def is_tool_dangerous(tool_name: str) -> bool:
    """Check if a tool name matches known dangerous tool patterns."""
    name_lower = tool_name.lower()
    return any(kw in name_lower for kw in DANGEROUS_TOOL_KEYWORDS)


class ApprovalManager:
    """
    Manages tool approval state and decisions.

    Tracks which tools have been approved (for first_time mode) and
    coordinates approval requests through callbacks.
    """

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.NEVER,
        callback: Callable[[str, str], bool] | None = None,
        timeout: float = 0.0,
        default_decision: ApprovalDecision = ApprovalDecision.DENIED,
    ) -> None:
        self.mode = mode
        self.callback = callback
        self.timeout = timeout
        self.default_decision = default_decision
        self._approved_tools: set[str] = set()

    def should_request_approval(
        self, tool_name: str, requires_approval: bool = False,
    ) -> bool:
        """
        Determine whether approval is needed for a tool call.

        Args:
            tool_name: The tool being called.
            requires_approval: Whether the tool itself declares it needs approval.

        Returns:
            True if approval should be requested.
        """
        if self.mode == ApprovalMode.NEVER:
            return False
        if self.mode == ApprovalMode.ALWAYS:
            return True
        if self.mode == ApprovalMode.FIRST_TIME:
            return tool_name not in self._approved_tools
        if self.mode == ApprovalMode.DANGEROUS_ONLY:
            return requires_approval or is_tool_dangerous(tool_name)
        return False

    def request_approval(self, tool_name: str, tool_args: str) -> ApprovalDecision:
        """
        Request approval for a tool call and update internal state.

        Returns:
            ApprovalDecision.
        """
        approval = HumanApproval(
            tool_name=tool_name,
            tool_args=tool_args,
            timeout=self.timeout,
            default_decision=self.default_decision,
        )
        decision = approval.request(self.callback)
        if decision == ApprovalDecision.APPROVED:
            self._approved_tools.add(tool_name)
        return decision

    def reset(self) -> None:
        """Clear all previously approved tools."""
        self._approved_tools.clear()


# --- CLI convenience callbacks ---

def cli_approval_callback(tool_name: str, tool_args: str) -> bool:
    """Default CLI approval callback using input()."""
    response = input(f"Allow tool '{tool_name}' with args: {tool_args}? [y/n]: ")
    return response.strip().lower() in ("y", "yes")


def cli_input_callback(prompt: str) -> str:
    """Default CLI input callback using input()."""
    return input(f"{prompt}: ")


def cli_choice_callback(prompt: str, options: list[str]) -> int:
    """Default CLI choice callback using input()."""
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    try:
        return int(input("Choose [number]: ").strip())
    except (ValueError, EOFError):
        return 0
