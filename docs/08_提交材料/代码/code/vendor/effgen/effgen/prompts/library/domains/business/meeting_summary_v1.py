"""
Business prompt: Meeting Summary — structured output variant.

Returns JSON: {decisions, action_items[{owner, item, due}], risks}

Inputs: transcript (str), meeting_title (str), attendees (list[str])
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.eval import PromptEval
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {
            "type": "string",
            "description": "Full or partial meeting transcript or notes.",
            "minLength": 20,
        },
        "meeting_title": {
            "type": "string",
            "description": "Title or topic of the meeting.",
            "minLength": 1,
        },
        "attendees": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of attendee names.",
        },
    },
    "required": ["transcript", "meeting_title"],
}

_FIXTURE = {
    "meeting_title": "Q3 Product Roadmap Review",
    "attendees": ["Alice (PM)", "Bob (Eng Lead)", "Carol (Design)", "Dave (Sales)"],
    "transcript": (
        "Alice: Let's kick off the Q3 roadmap review. First agenda item is the mobile app redesign.\n\n"
        "Carol: The new design system is 80% complete. We'll have mockups ready by end of next week.\n\n"
        "Bob: Engineering estimates 6 weeks for implementation once designs are final. "
        "Main risk is the legacy API — we might need to refactor the auth layer.\n\n"
        "Dave: Sales needs the new onboarding flow by August 15th for the enterprise deals we're closing.\n\n"
        "Alice: Okay, decision: we'll prioritize the onboarding flow first. Carol, can you fast-track "
        "those screens? Bob, please start the auth layer analysis this week.\n\n"
        "Bob: Will do. I'll have an assessment by Friday.\n\n"
        "Carol: I can have onboarding mockups by Wednesday, July 10th.\n\n"
        "Alice: Great. Any other risks we should flag?\n\n"
        "Bob: If the auth refactor is bigger than expected, we might slip the August 15th deadline.\n\n"
        "Dave: That would be a problem — we have three signed LOIs contingent on the feature.\n\n"
        "Alice: Noted. Bob, if auth looks risky by Friday, escalate immediately. "
        "We'll revisit the timeline if needed. Meeting adjourned."
    ),
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "minLength": 1},
                    "item": {"type": "string", "minLength": 1},
                    "due": {"type": "string"},
                },
                "required": ["owner", "item", "due"],
            },
            "minItems": 1,
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["decisions", "action_items", "risks"],
}


def _meeting_summary_structured(
    transcript: str, meeting_title: str, attendees: list[str] | None = None
) -> str:
    attendees = attendees or []
    attendees_str = ", ".join(attendees) if attendees else "Not specified"
    return (
        f"You are an expert meeting facilitator and note-taker. "
        f"Analyze the following meeting transcript and produce a structured summary.\n\n"
        f"Meeting: {meeting_title}\n"
        f"Attendees: {attendees_str}\n\n"
        f"Transcript:\n---\n{transcript}\n---\n\n"
        f"Return a JSON object with exactly these three keys:\n"
        f'  "decisions"    — array of decision strings (what was agreed upon)\n'
        f'  "action_items" — array of objects, each with:\n'
        f'                     "owner" (string), "item" (string), "due" (string)\n'
        f'  "risks"        — array of strings identifying flagged risks or blockers\n\n'
        f"Be specific and capture real names, dates, and commitments from the transcript. "
        f"Output only the JSON object, no markdown fences, no extra text."
    )


def _json_schema_check(output: str) -> bool | str:
    try:
        from jsonschema import Draft202012Validator
        from jsonschema import exceptions as jsonschema_exceptions
    except ImportError:
        return "jsonschema required"

    parsed = PromptEval._extract_json_object(output)
    if parsed is None:
        return "output is not valid JSON"
    try:
        Draft202012Validator(_EXPECTED_SCHEMA).validate(parsed)
    except jsonschema_exceptions.ValidationError as exc:
        return f"JSON schema validation failed: {exc.message}"
    return True


meeting_summary_v1 = LibraryPrompt(
    name="business.meeting_summary.v1",
    domain="business",
    variant="structured",
    description=(
        "Extract structured meeting summary as JSON: decisions, action_items (with owner, "
        "item, due), and risks. Input: transcript + meeting title + attendees."
    ),
    template=_meeting_summary_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_schema_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["business", "meeting", "structured", "json", "action-items"],
)

registry.register(meeting_summary_v1)
