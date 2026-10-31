"""
Guardrail presets for effGen.

Provides pre-configured guardrail chains for common use cases.
"""

from __future__ import annotations

from typing import Any

from .base import GuardrailChain, GuardrailPosition
from .content import LengthGuardrail, PIIGuardrail, ToxicityGuardrail
from .injection import PromptInjectionGuardrail, SystemPromptLeakGuardrail
from .tool_safety import ToolInputGuardrail, ToolOutputGuardrail, ToolPermissionGuardrail

# Positions the injection guardrail runs at in the "strict"/"standard"/"phi"
# presets: the first user turn AND a tool's return value, since an indirect
# prompt injection carried in a scraped page, a ticket body, or a RAG passage
# reaches the model through TOOL_OUTPUT, not INPUT. "minimal" stays
# input-only (a deliberate, lighter-weight tradeoff for that preset).
_INJECTION_POSITIONS_WITH_TOOL_OUTPUT = [
    GuardrailPosition.INPUT,
    GuardrailPosition.TOOL_OUTPUT,
]


def strict_guardrails(
    max_length: int = 50_000,
    tool_deny: list[str] | None = None,
) -> GuardrailChain:
    """All guardrails enabled, high sensitivity.

    Includes: length, injection (high, checked on the first user turn and on
    every tool's return value), a system-prompt-leak check on the agent's own
    answer, toxicity, PII (block), topic (none by default), tool input
    validation, tool output sanitization with PII stripping, and tool
    permissions.
    """
    chain = GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(
            sensitivity="high", positions=_INJECTION_POSITIONS_WITH_TOOL_OUTPUT
        ),
        SystemPromptLeakGuardrail(),
        ToxicityGuardrail(),
        PIIGuardrail(action="block"),
        ToolInputGuardrail(),
        ToolOutputGuardrail(strip_pii=True, max_output_length=max_length),
    ])
    if tool_deny:
        chain.add(ToolPermissionGuardrail(deny=tool_deny))
    return chain


def standard_guardrails(
    max_length: int = 100_000,
) -> GuardrailChain:
    """PII + injection + tool safety, medium sensitivity.

    A balanced preset. PII is redacted, not blocked: a conversational agent
    routinely receives a user's own identifying details (an email for
    verification, a phone number for a callback), and refusing the whole turn
    over that is worse than stripping it before it reaches the model. Use the
    "strict" preset when any PII in the input should refuse the turn outright.

    The injection guardrail checks both the first user turn
    (``GuardrailPosition.INPUT``) and every tool's return value
    (``GuardrailPosition.TOOL_OUTPUT``): content a tool fetches (an email
    body, a scraped page, a RAG passage) can carry an embedded instruction
    that steers the agent into calling a second, sensitive tool with
    attacker-chosen arguments, and this preset is the one most callers use by
    default. Use "strict" or "phi" for the same screening at higher
    sensitivity plus PII stripping on tool output.
    """
    return GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(
            sensitivity="medium", positions=_INJECTION_POSITIONS_WITH_TOOL_OUTPUT
        ),
        PIIGuardrail(action="redact"),
        ToolInputGuardrail(),
        ToolOutputGuardrail(max_output_length=max_length),
    ])


def phi_guardrails(
    max_length: int = 100_000,
    strict: bool = False,
) -> GuardrailChain:
    """De-identification posture for records containing personal identifiers.

    Redacts PII before it reaches a model: the built-in detectors plus the
    label-anchored fields common in semi-structured records (name, date of
    birth, medical record number, address, member/policy ID). Also runs length,
    injection, and tool-output safety with PII stripping.

    Coverage is regex-based: it removes labeled and well-formed identifiers, not
    every free-text personal name. For site-specific identifiers, build a
    ``PIIGuardrail`` with ``custom_patterns``/``custom_terms`` and use it
    directly. Pass ``strict=True`` to fail closed — any detection then reports
    ``passed=False`` instead of forwarding redacted text. Injection screening
    runs on the first user turn and on every tool's return value; a
    system-prompt-leak check also screens the agent's own answer.
    """
    return GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(
            sensitivity="medium", positions=_INJECTION_POSITIONS_WITH_TOOL_OUTPUT
        ),
        SystemPromptLeakGuardrail(),
        PIIGuardrail(action="redact", detect_labeled=True, strict=strict),
        ToolInputGuardrail(),
        ToolOutputGuardrail(strip_pii=True, max_output_length=max_length),
    ])


def minimal_guardrails(
    max_length: int = 200_000,
) -> GuardrailChain:
    """Basic length + injection only, first user turn only.

    Lightweight preset for low-risk applications. Injection screening runs at
    ``sensitivity="low"``, which does not detect plaintext role-label spoofing
    (e.g. a message that opens with "system:" or "assistant:" to impersonate
    another turn) — that pattern only loads at ``"medium"`` and above. Do not
    pick this preset by name alone for adversarial or regulated input; use
    "standard" or "phi" for a role-label-spoof check.
    """
    return GuardrailChain([
        LengthGuardrail(max_length=max_length),
        PromptInjectionGuardrail(sensitivity="low"),
    ])


def no_guardrails() -> GuardrailChain:
    """No guardrails — empty chain. For development/testing only."""
    return GuardrailChain([])


# Named preset constants for convenience
STRICT = "strict"
STANDARD = "standard"
PHI = "phi"
MINIMAL = "minimal"
NONE = "none"

_PRESET_FACTORIES = {
    STRICT: strict_guardrails,
    STANDARD: standard_guardrails,
    PHI: phi_guardrails,
    MINIMAL: minimal_guardrails,
    NONE: no_guardrails,
}

# Friendly aliases for the canonical preset names. ``"default"`` is the name most
# users reach for first, so it maps to the balanced "standard" preset.
_PRESET_ALIASES = {
    "default": STANDARD,
    "balanced": STANDARD,
    "hipaa": PHI,
    "deidentify": PHI,
    "off": NONE,
    "disabled": NONE,
}


def get_guardrail_preset(name: str, **kwargs: Any) -> GuardrailChain:
    """Get a guardrail chain by preset name.

    Args:
        name: One of "strict", "standard", "phi", "minimal", "none". The aliases
            "default"/"balanced" map to "standard", "hipaa"/"deidentify" map to
            "phi", and "off"/"disabled" map to "none".
        **kwargs: Passed to the preset factory function.

    Returns:
        Configured GuardrailChain.
    """
    key = name.lower()
    key = _PRESET_ALIASES.get(key, key)
    factory = _PRESET_FACTORIES.get(key)
    if factory is None:
        available = ", ".join(_PRESET_FACTORIES.keys())
        raise ValueError(f"Unknown guardrail preset: {name!r}. Available: {available}")
    return factory(**kwargs)
