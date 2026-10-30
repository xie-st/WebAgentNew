"""
Creative prompt: Character Bio — structured output variant.

Returns JSON: {name, age, background, personality_traits, goals, flaws, relationships}

Inputs: character_seed (str), genre (str), role (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.eval import PromptEval
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "character_seed": {
            "type": "string",
            "description": "Brief description or concept for the character.",
            "minLength": 5,
        },
        "genre": {
            "type": "string",
            "description": "Story genre, e.g. 'fantasy', 'contemporary', 'sci-fi'.",
            "minLength": 1,
        },
        "role": {
            "type": "string",
            "description": "Character role: 'protagonist', 'antagonist', 'supporting'.",
            "minLength": 1,
        },
    },
    "required": ["character_seed", "genre", "role"],
}

_FIXTURE = {
    "character_seed": "a retired detective haunted by an unsolved case",
    "genre": "noir thriller",
    "role": "protagonist",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 1, "maximum": 200},
        "background": {"type": "string", "minLength": 10},
        "personality_traits": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "goals": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "flaws": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "relationships": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["name", "age", "background", "personality_traits", "goals", "flaws"],
}


def _character_bio_structured(character_seed: str, genre: str, role: str) -> str:
    return (
        f"You are a professional character designer for {genre} fiction. "
        f"Create a vivid, believable character profile for a {role} based on this concept:\n\n"
        f'"{character_seed}"\n\n'
        f"Return a JSON object with exactly these keys:\n"
        f'  "name"               — full character name (string)\n'
        f'  "age"                — age in years (integer)\n'
        f'  "background"         — 2-3 sentence backstory (string)\n'
        f'  "personality_traits" — array of 3-5 trait strings\n'
        f'  "goals"              — array of 1-3 goal strings\n'
        f'  "flaws"              — array of 1-3 flaw strings\n'
        f'  "relationships"      — array of strings describing key relationships (can be empty)\n\n'
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


character_bio_v1 = LibraryPrompt(
    name="creative.character_bio.v1",
    domain="creative",
    variant="structured",
    description=(
        "Structured character biography generator. Returns JSON with name, age, background, "
        "personality traits, goals, flaws, and relationships."
    ),
    template=_character_bio_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_schema_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["creative", "character", "structured", "json"],
)

registry.register(character_bio_v1)
