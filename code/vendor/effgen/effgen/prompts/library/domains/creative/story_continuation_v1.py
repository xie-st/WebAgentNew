"""
Creative prompt: Story Continuation — zero-shot and few-shot variants.

Inputs: story_so_far (str), genre (str), desired_length (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "story_so_far": {
            "type": "string",
            "description": "The existing story text to continue.",
            "minLength": 10,
        },
        "genre": {
            "type": "string",
            "description": "Story genre, e.g. 'fantasy', 'sci-fi', 'thriller'.",
            "minLength": 1,
        },
        "desired_length": {
            "type": "string",
            "description": "Target length description, e.g. '1-2 paragraphs', '300 words'.",
            "minLength": 1,
        },
    },
    "required": ["story_so_far", "genre", "desired_length"],
}

_FIXTURE = {
    "story_so_far": (
        "The old lighthouse had been dark for thirty years. Maya climbed the rusted spiral "
        "staircase, each step groaning under her weight. When she reached the top, she "
        "found the lantern room filled with strange symbols scratched into the glass."
    ),
    "genre": "mystery",
    "desired_length": "2-3 paragraphs",
}


def _story_continuation_zero_shot(story_so_far: str, genre: str, desired_length: str) -> str:
    return (
        f"You are a skilled fiction writer specializing in {genre} stories.\n\n"
        f"Continue the following story naturally, maintaining the established tone, "
        f"voice, and narrative thread. Write approximately {desired_length}.\n\n"
        f"Story so far:\n---\n{story_so_far}\n---\n\n"
        f"Continue the story:"
    )


story_continuation_zero_shot = LibraryPrompt(
    name="creative.story_continuation.v1.zero_shot",
    domain="creative",
    variant="zero_shot",
    description="Zero-shot story continuation maintaining genre and tone.",
    template=_story_continuation_zero_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"[\w\s,.'\"!?]{50,}",
    },
    tags=["creative", "story", "fiction", "zero_shot"],
)

registry.register(story_continuation_zero_shot)


_FEW_SHOT_EXAMPLES = """\
Example 1 — Fantasy continuation:
Original: "The dragon's scales shimmered gold in the fading light."
Continuation: "She had not expected kindness from a creature of legend. Yet here it was, lowering its great head to her level, ancient eyes reflecting a sorrow older than the kingdom itself. Elara reached out a trembling hand. The dragon exhaled slowly, warming the cold mountain air."

Example 2 — Sci-fi continuation:
Original: "The colony ship's engines had been silent for three days."
Continuation: "Commander Reyes ran diagnostics for the forty-seventh time. The numbers hadn't changed: no fuel, no thrust, no hope — at least by the physics she'd been taught. Then Dr. Park emerged from the engine room, grease on her face and something impossible in her eyes. 'I think I found what the aliens left behind,' she said."
"""


def _story_continuation_few_shot(story_so_far: str, genre: str, desired_length: str) -> str:
    return (
        f"You are a skilled fiction writer specializing in {genre} stories. "
        f"Study the following continuation examples, then apply the same craft "
        f"to continue the given story.\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n"
        f"Now continue this {genre} story (approximately {desired_length}):\n\n"
        f"Story so far:\n---\n{story_so_far}\n---\n\n"
        f"Continuation:"
    )


story_continuation_few_shot = LibraryPrompt(
    name="creative.story_continuation.v1.few_shot",
    domain="creative",
    variant="few_shot",
    description="Few-shot story continuation with craft exemplars from multiple genres.",
    template=_story_continuation_few_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"[\w\s,.'\"!?]{50,}",
    },
    tags=["creative", "story", "fiction", "few_shot"],
)

registry.register(story_continuation_few_shot)
