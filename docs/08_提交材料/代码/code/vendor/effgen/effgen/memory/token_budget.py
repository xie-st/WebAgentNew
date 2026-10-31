"""Token budget allocation for context window management.

The :class:`TokenBudget` distributes a model's context window among
several logical sections (system prompt, tool descriptions, history,
response headroom) and provides smart truncation helpers that try to
preserve the *most relevant* parts of the context rather than blindly
dropping from the head.

Default split:
    system   20%
    tools    30%
    history  40%
    response 10%
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

DEFAULT_SHARES: dict[str, float] = {
    "system": 0.20,
    "tools": 0.30,
    "history": 0.40,
    "response": 0.10,
}


@dataclass
class TokenBudget:
    """Distributes a context window across named sections."""

    context_length: int
    shares: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        total = sum(self.shares.values())
        if total <= 0:
            raise ValueError("shares must sum to a positive value")
        if abs(total - 1.0) > 1e-6:
            self.shares = {k: v / total for k, v in self.shares.items()}

    def allocate(self, section: str) -> int:
        """Return the token cap for *section* (0 for an unknown section)."""
        share = self.shares.get(section, 0.0)
        return max(0, int(self.context_length * share))

    def allocations(self) -> dict[str, int]:
        """Return the token cap for every configured section."""
        return {k: self.allocate(k) for k in self.shares}

    def reserve(self, section: str, tokens: int) -> None:
        """Override one section's token cap directly (in absolute tokens)."""
        if tokens < 0 or tokens > self.context_length:
            raise ValueError("tokens out of range")
        self.shares[section] = tokens / self.context_length
        # Re-normalize the *other* shares so the total is still 1.0.
        others = [k for k in self.shares if k != section]
        remaining_share = max(0.0, 1.0 - self.shares[section])
        old_other_total = sum(DEFAULT_SHARES.get(k, 0.0) for k in others) or 1.0
        for k in others:
            base = DEFAULT_SHARES.get(k, 0.0)
            self.shares[k] = remaining_share * (base / old_other_total)


def smart_truncate(
    items: list[str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
    *,
    keep_head: int = 1,
    keep_tail: int = 2,
    summary_marker: str = "[... earlier turns summarized ...]",
) -> list[str]:
    """Truncate *items* to fit *max_tokens*, preserving head and tail.

    Strategy: keep the first ``keep_head`` items (typically system / first
    user message) and the last ``keep_tail`` items (most recent context),
    then fill in middle items from newest-to-oldest until the budget is hit.
    Anything dropped is replaced by a single summary marker.

    Args:
        items: The turns to fit, oldest first.
        max_tokens: The budget the result must fit in.
        count_tokens: Counts the tokens in one item.
        keep_head: Items kept from the start whatever the budget.
        keep_tail: Items kept from the end whatever the budget.
        summary_marker: The single item that stands in for what was dropped.

    Returns:
        The kept items, in their original order.
    """
    if max_tokens <= 0 or not items:
        return []
    total = sum(count_tokens(x) for x in items)
    if total <= max_tokens:
        return list(items)

    n = len(items)
    head = items[:keep_head] if keep_head > 0 else []
    tail = items[max(keep_head, n - keep_tail):] if keep_tail > 0 else []
    middle = items[len(head): n - len(tail)] if (len(head) + len(tail)) <= n else []

    fixed_tokens = sum(count_tokens(x) for x in head) + sum(count_tokens(x) for x in tail)
    marker_tokens = count_tokens(summary_marker)
    budget_left = max_tokens - fixed_tokens - marker_tokens

    kept_middle: list[str] = []
    for item in reversed(middle):  # prefer recent middle items
        cost = count_tokens(item)
        if cost <= budget_left:
            kept_middle.insert(0, item)
            budget_left -= cost
        else:
            break

    truncated = list(head)
    if len(kept_middle) < len(middle):
        truncated.append(summary_marker)
    truncated.extend(kept_middle)
    truncated.extend(tail)
    return truncated


def fit_to_budget(
    sections: dict[str, Iterable[str]],
    budget: TokenBudget,
    count_tokens: Callable[[str], int],
) -> dict[str, list[str]]:
    """Apply per-section truncation according to *budget*.

    Args:
        sections: The items to fit, keyed by section name.
        budget: The per-section token allowances.
        count_tokens: Counts the tokens in one item.

    Returns:
        The kept items per section, in their original order.
    """
    out: dict[str, list[str]] = {}
    for name, items in sections.items():
        cap = budget.allocate(name)
        out[name] = smart_truncate(list(items), cap, count_tokens)
    return out
