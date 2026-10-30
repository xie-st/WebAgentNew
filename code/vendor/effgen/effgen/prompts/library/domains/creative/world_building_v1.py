"""
Creative prompt: World Building — CoT variant.

Inputs: world_concept (str), genre (str), focus_areas (list[str])
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "world_concept": {
            "type": "string",
            "description": "High-level concept or premise for the world.",
            "minLength": 5,
        },
        "genre": {
            "type": "string",
            "description": "Genre: 'fantasy', 'sci-fi', 'post-apocalyptic', etc.",
            "minLength": 1,
        },
        "focus_areas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Aspects to develop: 'geography', 'politics', 'magic system', 'technology', 'culture', 'economy'.",
            "minItems": 1,
        },
    },
    "required": ["world_concept", "genre", "focus_areas"],
}

_FIXTURE = {
    "world_concept": "a world where music is a form of magic that physically reshapes reality",
    "genre": "fantasy",
    "focus_areas": ["magic system", "culture", "politics"],
}


def _world_building_cot(world_concept: str, genre: str, focus_areas: list[str]) -> str:
    areas_str = ", ".join(focus_areas) if focus_areas else "general world elements"
    return (
        f"You are a master world-builder for {genre} fiction. "
        f"Develop a rich, internally consistent world based on this concept:\n\n"
        f'"{world_concept}"\n\n'
        f"Think through your world step by step:\n\n"
        f"Step 1 — Core Premise: Establish the fundamental rules and logic of this world. "
        f"What makes it unique? What are the key tensions or paradoxes?\n\n"
        f"Step 2 — Focus Areas: For each of these aspects ({areas_str}), develop 2-3 "
        f"concrete, specific details that feel lived-in and real. Avoid generic descriptions.\n\n"
        f"Step 3 — Interconnections: Show how each focus area affects the others. "
        f"How does the core premise ripple through every layer of this society?\n\n"
        f"Step 4 — Story Hooks: Identify 2-3 inherent conflicts or questions this world "
        f"raises that could drive compelling narratives.\n\n"
        f"Step 5 — Synthesis: Write a 1-paragraph 'elevator pitch' for this world that "
        f"captures what makes it feel fresh and essential.\n\n"
        f"Work through each step before writing the final synthesis."
    )


world_building_v1 = LibraryPrompt(
    name="creative.world_building.v1",
    domain="creative",
    variant="cot",
    description=(
        "Chain-of-thought world building prompt. Develops geography, politics, magic/tech, "
        "culture, and story hooks step by step."
    ),
    template=_world_building_cot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(?:step|premise|concept|world|hook|conflict)",
    },
    tags=["creative", "world-building", "fiction", "cot"],
)

registry.register(world_building_v1)
