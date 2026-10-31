"""
Prompt caching helpers for the Anthropic Claude adapter.

Anthropic supports explicit prompt caching via ``cache_control`` markers on
content blocks. Cached prefixes are reused across requests, reducing latency
and cost for long, repeated content (system prompts, tool specs, documents).

Cache TTL
---------
- ``"5m"`` (default) — ephemeral 5-minute cache; write cost 1.25× input.
- ``"1h"`` — extended 1-hour cache; write cost 2× input; read cost 0.1× input.

Limits
------
- Maximum **4** ``cache_control`` breakpoints per request
  (tools + system + messages combined).
- Minimum cacheable block size varies by model (see ``MIN_CACHE_TOKENS``).
  Shorter blocks are silently billed normally (no error returned).

Cache evaluation order: tools → system → messages.
Changes at a given level invalidate that level and everything after it.

Priority when breakpoints risk exceeding 4
------------------------------------------
1. System-prompt last block (reused on every turn — highest ROI).
2. Last tool spec (tool list seldom changes).
3. Long document / user-message blocks (situational).
"""

from __future__ import annotations

from typing import Any, Literal

MAX_CACHE_BREAKPOINTS = 4

CacheTTL = Literal["5m", "1h"]

# Minimum token count for a content block to be eligible for caching.
# Blocks below the threshold are accepted by the API but not cached —
# cache_creation_input_tokens / cache_read_input_tokens will remain 0.
# Source: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
# (verified 2026-04-27)
MIN_CACHE_TOKENS: dict[str, int] = {
    # Claude 4.x flagship / latest
    "claude-opus-4-7":              4096,
    "claude-sonnet-4-6":            2048,
    "claude-haiku-4-5":             4096,
    "claude-haiku-4-5-20251001":    4096,
    # Legacy 4.x
    "claude-opus-4-6":              4096,
    "claude-sonnet-4-5":            1024,
    "claude-sonnet-4-5-20250929":   1024,
    "claude-opus-4-5":              1024,
    "claude-opus-4-5-20251101":     1024,
    "claude-opus-4-1":              1024,
    "claude-opus-4-1-20250805":     1024,
    # Claude 3.7 / 3.5
    "claude-3-7-sonnet-20250219":   1024,
    "claude-3-5-sonnet-20241022":   2048,
    "claude-3-5-sonnet-20240620":   2048,
    "claude-3-5-haiku-20241022":    2048,
    # Claude 3
    "claude-3-opus-20240229":       1024,
    "claude-3-haiku-20240307":      1024,
    "claude-3-sonnet-20240229":     1024,
}

_DEFAULT_MIN_CACHE_TOKENS = 1024


def get_min_cache_tokens(model_name: str) -> int:
    """Return the minimum token count for a cached block for *model_name*."""
    return MIN_CACHE_TOKENS.get(model_name, _DEFAULT_MIN_CACHE_TOKENS)


def mark_cached(
    block: dict[str, Any] | str,
    ttl: CacheTTL = "5m",
) -> dict[str, Any]:
    """
    Attach a ``cache_control`` marker to a content block.

    Parameters
    ----------
    block:
        A content block dict (e.g. ``{"type": "text", "text": "..."}``), or a
        plain string (converted to a text block automatically).
    ttl:
        Cache lifetime — ``"5m"`` (default, 1.25× write cost) or ``"1h"``
        (extended, 2× write cost).  Both hit at 0.1× read cost.

    Returns a **new** dict — the original is not mutated.

    Example::

        system = [
            {"type": "text", "text": "You are a helpful assistant."},
            mark_cached({"type": "text", "text": long_context}, ttl="1h"),
        ]
    """
    if isinstance(block, str):
        block = {"type": "text", "text": block}
    result = dict(block)
    result["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    return result


def count_cache_breakpoints(
    system: str | list[dict] | None,
    messages: list[dict] | None,
    tools: list[dict] | None,
) -> int:
    """Count total ``cache_control`` markers across all parts of a request.

    Parameters mirror the Anthropic ``messages.create`` fields so callers can
    validate before sending.  Evaluation order: tools → system → messages.

    Args:
        system: The system prompt, as text or content blocks.
        messages: The conversation messages.
        tools: The tool definitions sent with the request.

    Returns:
        How many ``cache_control`` markers the request carries.
    """
    count = 0

    for tool in (tools or []):
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1

    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                count += 1

    for msg in (messages or []):
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1

    return count


def validate_breakpoint_count(
    system: str | list[dict] | None,
    messages: list[dict] | None,
    tools: list[dict] | None,
) -> None:
    """Reject requests carrying more than ``MAX_CACHE_BREAKPOINTS`` (4) cache markers.

    Raises ``ValueError`` when the total ``cache_control`` marker count across
    system/messages/tools exceeds the limit.

    Removal priority if you need to trim: system-prompt last block > last tool
    spec > message blocks.
    """
    count = count_cache_breakpoints(system, messages, tools)
    if count > MAX_CACHE_BREAKPOINTS:
        raise ValueError(
            f"Too many cache_control breakpoints: {count} (max {MAX_CACHE_BREAKPOINTS}). "
            "Remove low-priority markers — recommended removal order: "
            "message blocks first, then last tool spec, keep system-prompt last block."
        )


def apply_cache_to_system(
    system_prompt: str | list[dict],
    ttl: CacheTTL = "5m",
) -> list[dict]:
    """Convert *system_prompt* to content blocks with ``cache_control`` on the last one.

    If the input is already a list, the last block is marked and a new list is
    returned (originals are not mutated).  A plain string becomes a single
    ``{"type": "text", ...}`` block with ``cache_control`` attached.
    """
    if isinstance(system_prompt, str):
        return [mark_cached({"type": "text", "text": system_prompt}, ttl=ttl)]

    if not system_prompt:
        return []

    blocks = list(system_prompt)
    blocks[-1] = mark_cached(blocks[-1], ttl=ttl)
    return blocks


def apply_cache_to_last_tool(
    tools: list[dict],
    ttl: CacheTTL = "5m",
) -> list[dict]:
    """
    Return a copy of *tools* with ``cache_control`` on the **last** entry.

    The original list and dicts are not mutated.
    """
    if not tools:
        return tools
    result = list(tools)
    last = dict(result[-1])
    last["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    result[-1] = last
    return result
