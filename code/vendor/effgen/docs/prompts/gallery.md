# Prompt Library Gallery

All 31 prompts in the curated library, grouped by domain. Regenerate this file with:

```bash
effgen prompts list --format markdown
```

For domain-specific guides and examples see the linked doc pages below each table.

---

## Research

[→ Research domain guide](research.md)

| Name | Variant | Description |
|------|---------|-------------|
| `research.literature_review.v1.zero_shot` | zero_shot | Zero-shot literature review covering a topic and year range. |
| `research.literature_review.v1.cot` | cot | Chain-of-thought literature review with step-by-step reasoning. |
| `research.paper_summary.v1` | structured | Structured paper summary returning JSON with abstract_summary, key_findings, limitations, and future_work. |
| `research.citation_extract.v1` | tool | Tool-augmented citation extraction using ArXiv/PubMed APIs. Instructs agents to retrieve live paper metadata before answering. |
| `research.methodology_critique.v1` | cot | Chain-of-thought methodology critique covering design, sampling, measurement, analysis, and generalizability. |

---

## Coding

[→ Coding domain guide](coding.md)

| Name | Variant | Description |
|------|---------|-------------|
| `coding.code_review.v1` | structured | Structured code review returning JSON {issues: [{severity, location, suggestion}]}. |
| `coding.bug_diagnose.v1` | cot | Chain-of-thought bug diagnosis: traces execution, identifies root cause, and proposes a minimal fix. |
| `coding.refactor_plan.v1` | tool | Tool-augmented refactoring plan: reads the source file then produces a structured plan with risk assessment and test strategy. |
| `coding.test_generate.v1` | few_shot | Few-shot test generation: two exemplar pytest suites guide output style. Live eval asserts ast.parse() passes on generated Python. |
| `coding.docstring_fill.v1` | zero_shot | Zero-shot docstring generator: adds Google/NumPy/Sphinx-style docstrings to undocumented Python functions. |

---

## Data / SQL

[→ Data domain guide](data.md)

| Name | Variant | Description |
|------|---------|-------------|
| `data.sql_from_nl.v1` | structured | Translate a natural-language question to SQL given a DDL schema. Returns JSON {sql, warnings[]}. Live eval validates via sqlglot.parse(). |
| `data.sql_explain.v1` | zero_shot | Explain a SQL query in plain English, targeting either a technical developer or a business stakeholder. |
| `data.sql_optimize.v1` | cot | Chain-of-thought SQL optimization: identify anti-patterns, explain execution impact, produce a rewritten query, and suggest indexes. |
| `data.data_profile.v1` | tool | Tool-augmented data profiling: takes ExcelTool/CSV column statistics and produces a structured data-quality report (completeness, uniqueness, range, issues, actions). |
| `data.etl_plan.v1` | few_shot | Few-shot ETL pipeline design: two exemplar designs guide output style. Covers Extract → Transform → Load → Validate → Cleanup with technology choices. |

---

## Legal

> **Disclaimer:** All legal templates include the verbatim text: *"This output is for informational purposes only and does not constitute legal advice. Consult a qualified attorney for guidance specific to your situation."*

[→ Legal domain guide](legal.md)

| Name | Variant | Description |
|------|---------|-------------|
| `legal.contract_summarize.v1` | structured | Summarize a contract into structured JSON with parties, term, obligations, termination, and risks. Includes mandatory legal disclaimer. |
| `legal.clause_classify.v1` | zero_shot | Classify a contract clause by type and flag notable characteristics. Zero-shot. Includes mandatory legal disclaimer. |
| `legal.legal_research_brief.v1` | tool | Produce a structured legal research brief grounded in pre-retrieved sources. Tool-augmented. Includes mandatory legal disclaimer. |

---

## Medical

> **Disclaimer:** All medical templates include the verbatim text: *"This output is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."*

[→ Medical domain guide](medical.md)

| Name | Variant | Description |
|------|---------|-------------|
| `medical.symptom_triage.v1` | structured | Triage reported symptoms into urgency level with see_doctor_if conditions. Structured JSON with mandatory disclaimer field. Includes mandatory medical disclaimer. |
| `medical.drug_interaction_query.v1` | structured | Query known drug interactions for a list of medications. Structured JSON with severity levels and recommendations. Includes mandatory medical disclaimer. |
| `medical.medical_literature.v1` | tool | Synthesize retrieved PubMed abstracts into a structured clinical evidence brief. Tool-augmented. Includes mandatory medical disclaimer. |

---

## Creative

[→ Creative domain guide](creative.md)

| Name | Variant | Description |
|------|---------|-------------|
| `creative.story_continuation.v1.zero_shot` | zero_shot | Zero-shot story continuation maintaining genre and tone. |
| `creative.story_continuation.v1.few_shot` | few_shot | Few-shot story continuation with craft exemplars from multiple genres. |
| `creative.poetry_forms.v1` | few_shot | Few-shot poetry generation with exemplars for haiku, sonnet, and free verse. Inputs: theme, form (haiku/sonnet/free_verse), mood. |
| `creative.character_bio.v1` | structured | Structured character biography generator. Returns JSON with name, age, background, personality traits, goals, flaws, and relationships. |
| `creative.world_building.v1` | cot | Chain-of-thought world building prompt. Develops geography, politics, magic/tech, culture, and story hooks step by step. |

---

## Business

[→ Business domain guide](business.md)

| Name | Variant | Description |
|------|---------|-------------|
| `business.meeting_summary.v1` | structured | Extract structured meeting summary as JSON: decisions, action_items (with owner, item, due), and risks. Input: transcript + meeting title + attendees. |
| `business.email_draft.v1` | few_shot | Few-shot email drafting with formal and casual tone exemplars. Inputs: purpose, recipient, key_points, tone (formal/casual). |
| `business.okr_generate.v1` | cot | Chain-of-thought OKR generator. Produces aligned objectives and measurable key results from team mission and strategic priorities. |
| `business.swot_analysis.v1` | structured | Structured SWOT analysis returning JSON with strengths, weaknesses, opportunities, threats, and strategic insights. Perspective-aware. |
| `business.elevator_pitch.v1` | zero_shot | Zero-shot elevator pitch generator with strict ≤150 word constraint. Inputs: product name, target audience, problem, solution, differentiator. |

---

## Quick Reference

```bash
# List all templates
effgen prompts list

# Filter
effgen prompts list --domain research
effgen prompts list --variant cot
effgen prompts list --domain coding --variant structured

# Inspect
effgen prompts show data.sql_from_nl.v1

# Evaluate (no model needed)
effgen prompts eval

# Live evaluation
effgen prompts eval --domain medical --live --model gpt-oss-120b

# Interactive playground
effgen prompts playground
```

```python
from effgen.prompts.library import registry, LibraryPrompt

# Browse
for p in registry.all():
    print(f"{p.name:50s}  {p.variant:12s}  {p.domain}")

# Get
p = registry.get("business.meeting_summary.v1")
rendered = p.template(
    transcript="Alice: We'll ship v1 by Friday. Bob: I'll own QA.",
    meeting_title="Sprint planning",
    attendees=["Alice", "Bob"],
)

# Search
legal = registry.search(domain="legal")
few_shot = registry.search(variant="few_shot")
```
