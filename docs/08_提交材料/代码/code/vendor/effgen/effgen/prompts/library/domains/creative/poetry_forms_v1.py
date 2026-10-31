"""
Creative prompt: Poetry Forms — few-shot with 3 form exemplars.

Inputs: theme (str), form (str), mood (str)
Forms: haiku, sonnet, free_verse
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "theme": {
            "type": "string",
            "description": "The central theme or subject of the poem.",
            "minLength": 1,
        },
        "form": {
            "type": "string",
            "description": "Poetic form: 'haiku', 'sonnet', or 'free_verse'.",
            "enum": ["haiku", "sonnet", "free_verse"],
        },
        "mood": {
            "type": "string",
            "description": "Emotional tone, e.g. 'melancholic', 'joyful', 'contemplative'.",
            "minLength": 1,
        },
    },
    "required": ["theme", "form", "mood"],
}

_FIXTURE = {
    "theme": "the passage of time",
    "form": "haiku",
    "mood": "contemplative",
}

_FORM_EXAMPLES = {
    "haiku": (
        "Form: Haiku\n"
        "Rules: Three lines — 5 syllables, 7 syllables, 5 syllables. "
        "Capture a single moment; often a seasonal or nature reference (kigo).\n"
        "Example:\n"
        "  Old silent pond...\n"
        "  A frog jumps into the pond.\n"
        "  Splash! Silence again.\n"
        "  (— Matsuo Bashō, transl.)\n"
    ),
    "sonnet": (
        "Form: Shakespearean Sonnet\n"
        "Rules: 14 lines of iambic pentameter, rhyme scheme ABAB CDCD EFEF GG. "
        "Three quatrains develop the theme; the couplet delivers a twist or resolution.\n"
        "Example (opening):\n"
        "  Shall I compare thee to a summer's day? (A)\n"
        "  Thou art more lovely and more temperate: (B)\n"
        "  Rough winds do shake the darling buds of May, (A)\n"
        "  And summer's lease hath all too short a date. (B)\n"
        "  (— Shakespeare, Sonnet 18)\n"
    ),
    "free_verse": (
        "Form: Free Verse\n"
        "Rules: No fixed meter or rhyme. Use line breaks, imagery, rhythm, and white space "
        "deliberately. Let the meaning and emotion drive the form.\n"
        "Example:\n"
        "  I hear America singing, the varied carols I hear,\n"
        "  Those of mechanics, each one singing his as it should be blithe and strong,\n"
        "  The carpenter singing his as he measures his plank or beam,\n"
        "  (— Walt Whitman, 'I Hear America Singing')\n"
    ),
}


def _poetry_forms_few_shot(theme: str, form: str, mood: str) -> str:
    form_key = form.lower().replace(" ", "_") if form else "haiku"
    form_guide = _FORM_EXAMPLES.get(form_key, _FORM_EXAMPLES["free_verse"])

    return (
        f"You are a poet with mastery of classical and contemporary forms. "
        f"Write a poem on the theme '{theme}' with a {mood} mood.\n\n"
        f"Here are three form references to guide you:\n\n"
        f"{_FORM_EXAMPLES['haiku']}\n"
        f"{_FORM_EXAMPLES['sonnet']}\n"
        f"{_FORM_EXAMPLES['free_verse']}\n"
        f"Now write your poem using the {form} form:\n"
        f"Theme: {theme}\n"
        f"Mood: {mood}\n\n"
        f"Follow these form constraints:\n{form_guide}\n\n"
        f"Your poem:"
    )


poetry_forms_v1 = LibraryPrompt(
    name="creative.poetry_forms.v1",
    domain="creative",
    variant="few_shot",
    description=(
        "Few-shot poetry generation with exemplars for haiku, sonnet, and free verse. "
        "Inputs: theme, form (haiku/sonnet/free_verse), mood."
    ),
    template=_poetry_forms_few_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"[\w\s]{20,}",
    },
    tags=["creative", "poetry", "haiku", "sonnet", "few_shot"],
)

registry.register(poetry_forms_v1)
