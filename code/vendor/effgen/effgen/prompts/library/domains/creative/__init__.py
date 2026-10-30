"""Creative domain prompts: story continuation, poetry, character bio, world building."""

from .character_bio_v1 import character_bio_v1
from .poetry_forms_v1 import poetry_forms_v1
from .story_continuation_v1 import story_continuation_few_shot, story_continuation_zero_shot
from .world_building_v1 import world_building_v1

__all__ = [
    "story_continuation_zero_shot",
    "story_continuation_few_shot",
    "poetry_forms_v1",
    "character_bio_v1",
    "world_building_v1",
]
