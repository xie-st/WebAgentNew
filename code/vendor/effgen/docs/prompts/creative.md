# Creative Domain Prompts

Templates for fiction writing, poetry generation, character design, and world-building.

---

## Templates

### `creative.story_continuation.v1.zero_shot`

**Variant:** zero_shot  
**Inputs:** `story_so_far` (str), `genre` (str), `desired_length` (str)

Continues an existing story naturally, preserving established tone and voice.

```python
from effgen.prompts.library.registry import registry

prompt = registry.get("creative.story_continuation.v1.zero_shot")
rendered = prompt.render(
    story_so_far="The lighthouse had been dark for thirty years...",
    genre="mystery",
    desired_length="2-3 paragraphs",
)
```

---

### `creative.story_continuation.v1.few_shot`

**Variant:** few_shot  
**Inputs:** `story_so_far` (str), `genre` (str), `desired_length` (str)

Few-shot story continuation with craft exemplars from fantasy and sci-fi. Guides the model toward publishable prose quality.

---

### `creative.poetry_forms.v1`

**Variant:** few_shot  
**Inputs:** `theme` (str), `form` (`haiku` | `sonnet` | `free_verse`), `mood` (str)

Generates a poem using one of three forms. Includes exemplars (Basho, Shakespeare, Whitman) to anchor the model's understanding of each form's constraints.

```python
prompt = registry.get("creative.poetry_forms.v1")
rendered = prompt.render(
    theme="loss and renewal",
    form="sonnet",
    mood="bittersweet",
)
```

---

### `creative.character_bio.v1`

**Variant:** structured  
**Inputs:** `character_seed` (str), `genre` (str), `role` (str)

Generates a complete character biography as structured JSON:

```json
{
  "name": "...",
  "age": 42,
  "background": "...",
  "personality_traits": ["...", "..."],
  "goals": ["..."],
  "flaws": ["..."],
  "relationships": ["..."]
}
```

The output is validated against a JSON schema.

---

### `creative.world_building.v1`

**Variant:** cot  
**Inputs:** `world_concept` (str), `genre` (str), `focus_areas` (list[str])

Chain-of-thought world building through 5 steps:
1. Core Premise
2. Focus Areas (magic system, politics, culture, technology, etc.)
3. Interconnections
4. Story Hooks
5. Synthesis paragraph

```python
prompt = registry.get("creative.world_building.v1")
rendered = prompt.render(
    world_concept="a world where music physically reshapes reality",
    genre="fantasy",
    focus_areas=["magic system", "culture", "politics"],
)
```

---

## Eval

```bash
effgen prompts eval --domain creative --live --model gpt-oss-120b
```

- `character_bio.v1` — live output validated against JSON schema
- `story_continuation.*` — output must contain ≥50 chars of story content
- `world_building.v1` — output must reference step/premise/concept/hook terminology
- `poetry_forms.v1` — output must contain ≥20 chars of poem content

---

## Running Tests

```bash
pytest tests/prompts/test_creative.py -v
```
