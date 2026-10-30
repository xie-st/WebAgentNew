"""
Business prompt: Email Draft — few-shot with 2 tone variants (formal, casual).

Inputs: purpose (str), recipient (str), key_points (list[str]), tone (str)
Tones: formal, casual
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {
            "type": "string",
            "description": "The main goal of the email.",
            "minLength": 5,
        },
        "recipient": {
            "type": "string",
            "description": "Recipient name/role, e.g. 'the engineering team', 'Dr. Smith'.",
            "minLength": 1,
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main points or information to convey.",
            "minItems": 1,
        },
        "tone": {
            "type": "string",
            "description": "Email tone: 'formal' or 'casual'.",
            "enum": ["formal", "casual"],
        },
    },
    "required": ["purpose", "recipient", "key_points", "tone"],
}

_FIXTURE = {
    "purpose": "request a one-week deadline extension on a project deliverable",
    "recipient": "project manager",
    "key_points": [
        "unexpected technical complexity discovered during testing",
        "need 5 additional business days to ensure quality",
        "happy to set up a call to discuss",
    ],
    "tone": "formal",
}

_FORMAL_EXAMPLE = """\
Tone: Formal
---
Subject: Request for Extension — Q3 Analytics Report

Dear Ms. Johnson,

I am writing to request a brief extension on the Q3 Analytics Report, currently due on July 15th. During the final review stage, our team identified a data discrepancy in the revenue reconciliation module that requires additional verification to ensure accuracy.

We estimate that a five-business-day extension (to July 22nd) would allow us to resolve this issue thoroughly and deliver a report you can present with confidence.

I would welcome the opportunity to discuss this further at your convenience. Please let me know if a brief call this week would work for you.

Thank you for your understanding.

Best regards,
[Your Name]
---"""

_CASUAL_EXAMPLE = """\
Tone: Casual
---
Subject: Quick heads-up — need a bit more time on the analytics report

Hi Sarah,

Hope you're doing well! Just wanted to give you a quick heads-up — we hit a snag in the data reconciliation piece of the Q3 report, so I'm hoping we can push the deadline back by about a week.

Nothing catastrophic, just want to make sure the numbers are solid before we ship it. Would July 22nd work instead of the 15th?

Happy to jump on a quick call if it's easier to chat. Let me know!

Cheers,
[Your Name]
---"""


def _email_draft_few_shot(
    purpose: str, recipient: str, key_points: list[str], tone: str
) -> str:
    points_str = "\n".join(f"  - {p}" for p in key_points) if key_points else "  - (no specific points)"
    return (
        f"You are an expert business communicator. Study these two email tone examples, "
        f"then draft a new email in the requested tone.\n\n"
        f"{_FORMAL_EXAMPLE}\n\n"
        f"{_CASUAL_EXAMPLE}\n\n"
        f"Now draft a {tone} email for the following situation:\n"
        f"Purpose: {purpose}\n"
        f"To: {recipient}\n"
        f"Key points to address:\n{points_str}\n\n"
        f"Match the {tone} tone from the example above. "
        f"Include Subject line, greeting, body paragraphs, and closing. "
        f"Use [Your Name] as the sender placeholder."
    )


email_draft_v1 = LibraryPrompt(
    name="business.email_draft.v1",
    domain="business",
    variant="few_shot",
    description=(
        "Few-shot email drafting with formal and casual tone exemplars. "
        "Inputs: purpose, recipient, key_points, tone (formal/casual)."
    ),
    template=_email_draft_few_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(?i)subject:|dear |hi |hello ",
    },
    tags=["business", "email", "communication", "few_shot", "formal", "casual"],
)

registry.register(email_draft_v1)
