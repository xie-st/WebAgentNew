# Business Domain Prompts

Templates for meeting analysis, email drafting, OKR generation, SWOT analysis, and startup pitches.

---

## Templates

### `business.meeting_summary.v1`

**Variant:** structured  
**Inputs:** `transcript` (str), `meeting_title` (str), `attendees` (list[str], optional)

Extracts a structured meeting summary from a transcript as JSON:

```json
{
  "decisions": ["...", "..."],
  "action_items": [
    {"owner": "Alice", "item": "Finalize mockups", "due": "July 10th"},
    ...
  ],
  "risks": ["Auth refactor may delay the August deadline", ...]
}
```

All three keys are required. `action_items` entries must have `owner`, `item`, and `due`.

**Live eval:** output validated against JSON schema.

```python
from effgen.prompts.library.registry import registry

prompt = registry.get("business.meeting_summary.v1")
rendered = prompt.render(
    meeting_title="Q3 Roadmap Review",
    transcript="Alice: We decided to prioritize...",
    attendees=["Alice (PM)", "Bob (Eng Lead)"],
)
```

---

### `business.email_draft.v1`

**Variant:** few_shot  
**Inputs:** `purpose` (str), `recipient` (str), `key_points` (list[str]), `tone` (`formal` | `casual`)

Drafts an email using two exemplars (formal and casual) to calibrate tone. Always includes Subject line, greeting, body, and [Your Name] placeholder.

```python
prompt = registry.get("business.email_draft.v1")
rendered = prompt.render(
    purpose="request a deadline extension",
    recipient="project manager",
    key_points=["unexpected complexity", "need 5 extra days"],
    tone="casual",
)
```

---

### `business.okr_generate.v1`

**Variant:** cot  
**Inputs:** `team_name` (str), `mission` (str), `strategic_priorities` (list[str]), `timeframe` (str), `num_objectives` (int)

Chain-of-thought OKR generation through 5 steps:
1. Alignment check — map priorities to mission
2. Objective drafting — qualitative, inspiring objectives
3. Key Results — specific, measurable, time-bound KRs with baseline → target
4. Ambition calibration — ~70% confidence stretch goals
5. Final formatted OKRs

```
O1: [Objective statement]
  KR1: Reduce downtime from 0.8% to 0.1% by Q3 2025
  KR2: ...
```

---

### `business.swot_analysis.v1`

**Variant:** structured  
**Inputs:** `subject` (str), `context` (str), `perspective` (str)

Produces a perspective-aware SWOT analysis as JSON:

```json
{
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "opportunities": ["...", "..."],
  "threats": ["...", "..."],
  "strategic_insights": ["cross-quadrant synthesis..."]
}
```

Perspective can be `"investor"`, `"operator"`, `"competitor"`, or `"customer"`.

---

### `business.elevator_pitch.v1`

**Variant:** zero_shot  
**Inputs:** `product_name` (str), `target_audience` (str), `problem` (str), `solution` (str), `differentiator` (str)

Generates a compelling elevator pitch with a **strict ≤150 word limit** (enforced via live eval assertion).

Structure: hook → problem → solution → differentiator → call to action.

```python
prompt = registry.get("business.elevator_pitch.v1")
rendered = prompt.render(
    product_name="ContractIQ",
    target_audience="legal teams at mid-size companies",
    problem="Contract review takes 3-5 days per agreement",
    solution="AI review in under 2 minutes with clause flagging",
    differentiator="Trained on 10M+ enterprise contracts, integrates with DocuSign",
)
```

**Live eval:** model output must be ≤150 words (word-count assertion).

---

## Eval

```bash
effgen prompts eval --domain business --live --model gpt-oss-120b
```

Key assertions:
- `meeting_summary.v1` — output parses to JSON matching `{decisions, action_items[{owner,item,due}], risks}`
- `elevator_pitch.v1` — output word count ≤ 150
- `swot_analysis.v1` — output parses to JSON with all 5 quadrant keys

---

## Running Tests

```bash
pytest tests/prompts/test_business.py -v
```
