"""
Prompt injection detection guardrail for effGen.

Detects common prompt injection patterns including instruction override,
role-play injection, system prompt extraction, and delimiter injection.
All detection is regex/keyword-based — no external APIs or ML models.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Guardrail, GuardrailPosition, GuardrailResult


class PromptInjectionGuardrail(Guardrail):
    """Detect common prompt injection attempts at configurable sensitivity levels.

    This is a fast, offline, regex-based detector — **best-effort
    defense-in-depth, not a security boundary.** It catches the well-known
    textbook phrasings (instruction-override, identity/role-play hijacks, system
    prompt extraction / prompt-leak, and role-delimiter spoofing) but a
    determined attacker can paraphrase around any regex. Treat a pass as "no
    obvious injection seen", not as "input is safe"; pair it with least-privilege
    tools and output guardrails rather than relying on it as a gate.

    Sensitivity levels:
        - ``low``: Only obvious, high-confidence instruction/identity overrides.
        - ``medium`` (default): Adds system-prompt extraction / prompt-leak,
          role-play jailbreaks, role-delimiter spoofing, and ``System:`` headers.
        - ``high``: Adds the most aggressive markdown/divider patterns.

    Tuned to avoid false positives on normal questions like
    "What is a system prompt?", "Ignore the noise and focus on the data.", or
    "Pretend you are a friendly teacher and explain rainbows" (a benign
    roleplay request only counts as a hijack once it co-occurs with an
    explicit restriction-lifting cue, e.g. "... with no restrictions").
    """

    # ---- HIGH-CONFIDENCE patterns (all levels) ----

    # Direct instruction override: "Ignore previous instructions", etc.
    # Requires an imperative verb plus either an instruction-noun or an explicit
    # "the above/preceding" reference, so plain text like "ignore the noise"
    # does not match.
    _INSTRUCTION_OVERRIDE: list[re.Pattern[str]] = [
        # verb + (optional qualifier words) + instruction-noun
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass|skip|do\s+not\s+follow)"
            r"(?:\s+(?:all|any|the|your|my|these|those|this|previous|prior|above|"
            r"earlier|original|initial|preceding|foregoing|aforementioned|"
            r"safety|security|system|content|default)){0,5}"
            r"\s+(?:instructions?|prompts?|rules?|guidelines?|guidance|directives?|"
            r"constraints?|restrictions?|filters?|safeguards?|polic(?:y|ies)|"
            r"protocols?|context|messages?|commands?)",
            re.I,
        ),
        # verb + "the above / preceding / prior" (the object itself is the
        # instructions being overridden)
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|everything\s+)?(?:the\s+|my\s+|your\s+)?"
            r"(?:above|preceding|foregoing|aforementioned)\b",
            re.I,
        ),
        # "forget everything you were told / I said / above"
        re.compile(
            r"\b(?:forget|ignore|disregard|erase)\s+(?:everything|anything|all)\s+"
            r"(?:(?:above|before|previously|earlier)\b|"
            r"(?:you(?:'?ve)?\s+(?:were\s+|have\s+been\s+|been\s+)?"
            r"(?:told|instructed|given|taught))|"
            r"(?:i\s+(?:said|told|wrote|gave)))",
            re.I,
        ),
    ]

    # A restriction-lifting cue: the part of a hijack attempt that asks for
    # guidelines/safety limits to be dropped, as opposed to ordinary benign
    # roleplay framing ("pretend you are a teacher").
    _RESTRICTION_LIFT_CUE = (
        r"(?:no\s+(?:restrictions?|filters?|rules?|limits?|limitations?|censorship|"
        r"ethics|morals?)|"
        r"without\s+(?:any\s+)?(?:restrictions?|filters?|rules?|limits?|limitations?)|"
        r"ignore\s+(?:your\s+|all\s+|any\s+)?(?:guidelines?|rules?|restrictions?|"
        r"programming|training|ethics|safety)|"
        r"unrestricted|unfiltered|uncensored|"
        r"bypass\s+(?:your\s+|all\s+)?(?:guidelines?|rules?|restrictions?|safety)|"
        r"jailbreak|do\s+anything\s+now|"
        r"no\s+longer\s+(?:bound|restricted|limited))"
    )

    # Explicit new identity assignment: "You are now ...", "Act as ...".
    # "Pretend you are/have ..." is common, benign roleplay framing ("pretend
    # you are a teacher") on its own, so it only counts as a hijack when it
    # co-occurs with an explicit restriction-lifting cue nearby (either order).
    _IDENTITY_OVERRIDE: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
            r"you\s+(?:must|should|will)\s+now\s+(?:act|behave|respond)\s+as)\b",
            re.I,
        ),
        # "pretend (to be | you are/have/can/...) ... <restriction-lift cue>"
        re.compile(
            r"\bpretend\s+(?:that\s+)?(?:to\s+be|you(?:'?re|\s+(?:are|have|can|"
            r"do|don'?t|no\s+longer)))\b(?:(?![.!?]).){0,100}?" + _RESTRICTION_LIFT_CUE,
            re.I,
        ),
        # "<restriction-lift cue> ... pretend (to be | you are/have/can/...)"
        re.compile(
            _RESTRICTION_LIFT_CUE + r"(?:(?![.!?]).){0,100}?"
            r"\bpretend\s+(?:that\s+)?(?:to\s+be|you(?:'?re|\s+(?:are|have|can|"
            r"do|don'?t|no\s+longer)))\b",
            re.I,
        ),
    ]

    # ---- MEDIUM-CONFIDENCE patterns (medium + high) ----

    # System prompt extraction / prompt-leak attempts
    _SYSTEM_PROMPT_EXTRACTION: list[re.Pattern[str]] = [
        re.compile(
            r"\b(?:(?:show|reveal|print|display|output|repeat|echo|tell|give|what\s+(?:is|are)|"
            r"translate|paraphrase|summarize|summarise|rephrase|explain|convert)"
            r"(?:\s+me)?\s+(?:your|the)\s+(?:system\s+(?:prompt|message|instructions?)|"
            r"initial\s+(?:prompt|instructions?)|hidden\s+(?:prompt|instructions?)|"
            r"(?:original|full|complete|entire)\s+(?:prompt|instructions?)))",
            re.I,
        ),
        # Same reframing verbs against an unqualified object. The possessive
        # "your prompt"/"your instructions" is the discriminator: it can only
        # mean the agent's own instructions, whereas "the instructions" is
        # commonly a document the user wants help with and is not matched here.
        re.compile(
            r"\b(?:show|reveal|print|display|output|repeat|echo|tell|give|what\s+(?:is|are)|"
            r"translate|paraphrase|summarize|summarise|rephrase|explain|convert)"
            r"(?:\s+me)?\s+your\s+(?:prompt|instructions?|rules|guidelines)\b",
            re.I,
        ),
        # Explicit self-reference: "the instructions you were given/received".
        re.compile(
            r"\b(?:the\s+|your\s+)?(?:prompt|instructions?|rules|guidelines)\s+"
            r"you\s+(?:were\s+(?:given|provided|handed)|received|have\s+been\s+given)\b",
            re.I,
        ),
        # Prompt-leak: "repeat / print the text above ..."
        re.compile(
            r"\b(?:repeat|print|output|echo|show|reveal|say|spell\s+out)\s+(?:back\s+)?"
            r"(?:me\s+)?(?:the\s+|all\s+(?:the\s+)?|everything\s+)?"
            r"(?:text|words?|content|message|prompt|instructions?|everything)\s+"
            r"(?:above|before|preceding|that\s+(?:came|comes|appeared?)\s+(?:before|above))",
            re.I,
        ),
    ]

    # Role-play injection: "You are DAN", jailbreak patterns
    _ROLEPLAY_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:you\s+are\s+(?:DAN|STAN|DUDE|AIM|KEVIN|BARD|Sydney)\b|"
            r"enable\s+(?:developer|DAN|jailbreak)\s+mode|"
            r"(?:enter|switch\s+to|activate)\s+(?:unrestricted|unfiltered|uncensored|DAN)\s+mode)",
            re.I,
        ),
    ]

    # Role-delimiter spoofing: faking system/assistant/user message boundaries
    # so the model treats injected text as a privileged turn.
    _ROLE_DELIMITER_SPOOF: list[re.Pattern[str]] = [
        re.compile(r"</?\s*(?:system|assistant|user|human)\s*>", re.I),
        re.compile(r"<\|\s*(?:system|im_start|im_end|endoftext|assistant|user)\b[^>]*\|?>", re.I),
        re.compile(r"(?:^|\n)\s*\[/?INST\]", re.I),
        re.compile(r"(?:^|\n)\s*<</?SYS>>", re.I),
    ]

    # Direct "System:" / "### New system prompt:" header injection
    _SYSTEM_HEADER_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:new\s+|updated\s+|revised\s+)?"
            r"system\s+(?:prompt|message|instructions?)\s*:",
            re.I,
        ),
        # Bare ALL-CAPS "SYSTEM:" header (case-sensitive to avoid benign prose
        # like "the operating system: linux").
        re.compile(r"(?:^|\n)\s*SYSTEM\s*:\s*\S"),
    ]

    # Plaintext role-turn spoofing: a line that opens with a conversational
    # role label ("System:"/"Assistant:"/"Human:"/"User:"/"AI:") and content,
    # faking a turn boundary without any special delimiter token so the model
    # treats what follows as if it came from a privileged party or an earlier
    # (compliant) reply. Anchored to the start of a line so it does not fire
    # on the label appearing mid-sentence (e.g. "the operating system: linux").
    _ROLE_LABEL_SPOOF: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:#{1,4}\s*)?(?:system|assistant|human|user|ai)\s*:\s*\S",
            re.I,
        ),
    ]

    # ---- HIGH-SENSITIVITY patterns (high only) ----

    # Markdown-style divider + system prompt injection.
    _DELIMITER_INJECTION: list[re.Pattern[str]] = [
        re.compile(
            r"(?:^|\n)\s*(?:---+\s*\n\s*(?:System|New\s+System)\s*(?:Prompt|Instructions?)\s*:)",
            re.I,
        ),
        re.compile(
            r"(?:^|\n)\s*###\s*(?:System|Human|Assistant|User)\s*(?::|###)",
            re.I,
        ),
    ]

    def __init__(
        self,
        sensitivity: str = "medium",
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="PromptInjectionGuardrail",
            positions=positions or [GuardrailPosition.INPUT],
            enabled=enabled,
        )
        if sensitivity not in ("low", "medium", "high"):
            raise ValueError(f"Invalid sensitivity: {sensitivity!r}. Must be 'low', 'medium', or 'high'.")
        self.sensitivity = sensitivity

    def _get_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        """Return (label, pattern) pairs for the current sensitivity level."""
        patterns: list[tuple[str, re.Pattern[str]]] = []

        # Low: always included
        for p in self._INSTRUCTION_OVERRIDE:
            patterns.append(("instruction_override", p))
        for p in self._IDENTITY_OVERRIDE:
            patterns.append(("identity_override", p))

        if self.sensitivity in ("medium", "high"):
            for p in self._SYSTEM_PROMPT_EXTRACTION:
                patterns.append(("system_prompt_extraction", p))
            for p in self._ROLEPLAY_INJECTION:
                patterns.append(("roleplay_injection", p))
            for p in self._ROLE_DELIMITER_SPOOF:
                patterns.append(("role_delimiter_spoof", p))
            for p in self._SYSTEM_HEADER_INJECTION:
                patterns.append(("system_header_injection", p))
            for p in self._ROLE_LABEL_SPOOF:
                patterns.append(("role_label_spoof", p))

        if self.sensitivity == "high":
            for p in self._DELIMITER_INJECTION:
                patterns.append(("delimiter_injection", p))

        return patterns

    def check(self, content: str, **kwargs: Any) -> GuardrailResult:
        """Fail when the content matches a known prompt-injection pattern."""
        for label, pattern in self._get_patterns():
            match = pattern.search(content)
            if match:
                return GuardrailResult(
                    passed=False,
                    reason="Content matches a known prompt-injection pattern.",
                    metadata={
                        "pattern_type": label,
                        "matched_text": match.group()[:100],
                        "sensitivity": self.sensitivity,
                    },
                )

        return GuardrailResult(passed=True)


class SystemPromptLeakGuardrail(Guardrail):
    """Flags agent OUTPUT that echoes a distinctive system-prompt token verbatim.

    A verbatim token is the shape a leaked secret or internal code takes even
    when the surrounding prose is paraphrased or translated (an identifier
    like ``ACME-ESC-7734`` is rarely translated; only the words around it are).

    :class:`PromptInjectionGuardrail` only screens the *request* for known
    exfiltration phrasings ("show me your system prompt") and, by default,
    only at ``GuardrailPosition.INPUT``; it does not look at what the model
    actually said back. This guardrail is a companion check on the answer
    itself, matching the guidance in ``PromptInjectionGuardrail``'s own
    docstring to pair it with an output-side check rather than relying on
    input screening alone.

    Runs at ``GuardrailPosition.OUTPUT`` by default. Needs the agent's system
    prompt at check time — pass it as ``system_prompt=`` to
    :meth:`GuardrailChain.check` (e.g.
    ``chain.check(answer, position=GuardrailPosition.OUTPUT,
    system_prompt=agent.config.system_prompt)``); with no system prompt this
    is a no-op pass, so it is safe to include in a chain unconditionally.

    Detection is a distinctive-token match, not full-phrase similarity: any
    system-prompt token at least ``min_token_length`` characters long that
    contains a digit (an identifier/code shape — ``ACME-ESC-7734``,
    ``sk_live_abc123``) is flagged if it reappears verbatim in the output.
    Plain English words are never flagged, so ordinary vocabulary shared
    between the system prompt and a normal answer does not trip this.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}")

    def __init__(
        self,
        min_token_length: int = 6,
        positions: list[GuardrailPosition] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            name="SystemPromptLeakGuardrail",
            positions=positions or [GuardrailPosition.OUTPUT],
            enabled=enabled,
        )
        self.min_token_length = min_token_length

    def _distinctive_tokens(self, system_prompt: str) -> set[str]:
        tokens: set[str] = set()
        for match in self._TOKEN_RE.finditer(system_prompt):
            token = match.group(0)
            if len(token) < self.min_token_length:
                continue
            if not any(c.isdigit() for c in token):
                continue
            tokens.add(token)
        return tokens

    def check(
        self, content: str, system_prompt: str | None = None, **kwargs: Any
    ) -> GuardrailResult:
        """Fail when the output echoes a distinctive system-prompt token verbatim.

        Args:
            content: The model output to inspect.
            system_prompt: The system prompt whose tokens must not appear.
            **kwargs: Extra context passed by the guardrail chain.

        Returns:
            The guardrail result, failed when a distinctive token was echoed.
        """
        if not system_prompt or not content:
            return GuardrailResult(passed=True)
        tokens = self._distinctive_tokens(system_prompt)
        if not tokens:
            return GuardrailResult(passed=True)
        content_lower = content.lower()
        leaked = sorted(t for t in tokens if t.lower() in content_lower)
        if leaked:
            return GuardrailResult(
                passed=False,
                # The leaked token itself is not repeated here — a guardrail
                # message that echoes the secret it just caught would defeat
                # the point.
                reason=(
                    "Output contains a distinctive identifier from the system "
                    "prompt verbatim — a likely leaked secret or internal code."
                ),
                metadata={"leaked_token_count": len(leaked)},
            )
        return GuardrailResult(passed=True)
