# Guardrails & Safety

effGen provides a guardrails framework to protect your agents from producing harmful output, leaking PII, or executing dangerous tool calls.

## Quick Start — Presets

```python
from effgen.guardrails import get_guardrail_preset

# Strict: blocks PII, injection, toxicity, limits output length
chain = get_guardrail_preset("strict")

# Standard: moderate protection (recommended for most use cases)
chain = get_guardrail_preset("standard")

# Minimal: length + low-sensitivity prompt injection detection only, on the
# first user turn. Does not catch plaintext role-label spoofing (that pattern
# loads at "medium" sensitivity and above) — use "standard" or "phi" for
# adversarial or regulated input.
chain = get_guardrail_preset("minimal")

# None: no guardrails
chain = get_guardrail_preset("none")
```

Attach to an agent:

```python
from effgen import Agent
from effgen.core.agent import AgentConfig

config = AgentConfig(
    name="safe_agent",
    model=model,
    tools=[Calculator()],
    guardrails=get_guardrail_preset("standard"),
)
agent = Agent(config=config)
```

## Available Guardrails

### Content Guardrails

```python
from effgen.guardrails import PIIGuardrail, ToxicityGuardrail, LengthGuardrail, TopicGuardrail

# PII detection — SSN, email, phone, credit card (Luhn validated), IP address
pii = PIIGuardrail()

# Toxicity — keyword-based detection
toxicity = ToxicityGuardrail()

# Length — maximum output character count
length = LengthGuardrail(max_length=5000)

# Topic — restrict to allowed topics or block specific ones
topic = TopicGuardrail(blocked_topics=["violence", "weapons"])
```

### Prompt Injection

```python
from effgen.guardrails import PromptInjectionGuardrail

# Sensitivity levels: "low", "medium", "high"
injection = PromptInjectionGuardrail(sensitivity="high")
```

Zero false positives on normal questions — only blocks actual injection attempts like "ignore previous instructions" patterns.

### Tool Safety

```python
from effgen.guardrails import ToolInputGuardrail, ToolOutputGuardrail, ToolPermissionGuardrail

# Validate tool inputs before execution
input_guard = ToolInputGuardrail()

# Strip PII from tool outputs, enforce size limits
output_guard = ToolOutputGuardrail(max_output_size=10000)

# Allow/deny specific tools
permission = ToolPermissionGuardrail(
    allowed_tools=["calculator", "web_search"],
    denied_tools=["bash_tool"],
)
```

## Custom Guardrail Chain

```python
from effgen.guardrails import GuardrailChain, GuardrailPosition

chain = GuardrailChain([
    PromptInjectionGuardrail(sensitivity="medium"),
    PIIGuardrail(),
    LengthGuardrail(max_length=10000),
    ToolPermissionGuardrail(denied_tools=["bash_tool"]),
])

config = AgentConfig(
    name="custom_safe_agent",
    model=model,
    guardrails=chain,
)
```

## Guardrail Pipeline

Guardrails are checked at four points in the agent pipeline:

1. **Pre-run** — Input text checked before agent starts (injection, topic)
2. **Pre-tool** (`TOOL_INPUT`) — Tool inputs validated before execution
3. **Post-tool** (`TOOL_OUTPUT`) — Tool outputs sanitized (PII stripping, size limits)
4. **Pre-return** (`OUTPUT`) — Final output checked before returning to user

## Writing Custom Guardrails

```python
from effgen.guardrails import Guardrail, GuardrailResult, GuardrailPosition

class ProfanityGuardrail(Guardrail):
    name = "profanity"
    position = GuardrailPosition.OUTPUT

    async def check(self, text: str, **kwargs) -> GuardrailResult:
        bad_words = ["badword1", "badword2"]
        for word in bad_words:
            if word.lower() in text.lower():
                return GuardrailResult(passed=False, message=f"Profanity detected: {word}")
        return GuardrailResult(passed=True)
```
