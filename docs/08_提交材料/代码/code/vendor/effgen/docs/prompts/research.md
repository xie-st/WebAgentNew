# Research Prompts

The `research` domain provides four prompt templates covering literature review,
paper summarization, citation extraction, and methodology critique.

---

## Templates

### `research.literature_review.v1.zero_shot`

**Variant:** zero\_shot  
**Inputs:** `topic` (str), `years_range` (str), `max_papers` (int)

Generates a structured literature review covering a topic and year range. Asks the
model to organize findings into Overview, Key Papers, Major Themes, and Research Gaps.

```python
from effgen.prompts.library import registry

p = registry.get("research.literature_review.v1.zero_shot")
prompt = p.render(
    topic="protein language models",
    years_range="2021-2024",
    max_papers=15,
)
```

---

### `research.literature_review.v1.cot`

**Variant:** cot  
**Inputs:** `topic` (str), `years_range` (str), `max_papers` (int)

Chain-of-thought variant. Guides the model through five explicit reasoning steps:
sub-topic identification → representative papers → synthesis → research gaps →
future directions. Produces higher-quality reviews on capable models.

---

### `research.paper_summary.v1`

**Variant:** structured  
**Inputs:** `title` (str), `abstract` (str), `authors` (str, optional)

Returns a JSON object with exactly four keys:

| Key | Type | Description |
|-----|------|-------------|
| `abstract_summary` | string | 2–3 sentence plain-language summary |
| `key_findings` | list[str] | 3–5 main contributions or results |
| `limitations` | list[str] | 2–4 noted limitations or caveats |
| `future_work` | list[str] | 2–3 suggested future research directions |

The prompt explicitly requests raw JSON (no markdown fences). The `expected_shape`
spec validates JSON and checks for all four required keys on live eval.

```python
p = registry.get("research.paper_summary.v1")
prompt = p.render(
    title="Attention Is All You Need",
    abstract="...",
    authors="Vaswani et al.",
)
# Feed `prompt` to a model; parse JSON response.
```

---

### `research.citation_extract.v1`

**Variant:** tool  
**Inputs:** `query` (str), `source` (`"arxiv"` | `"pubmed"` | `"auto"`), `max_results` (int)

Tool-augmented template. The render function is deterministic and tells the agent
which effGen tool to call (`ArXivTool` or `PubMedTool`) before producing the final
citation analysis. The model is asked to format citations (APA), classify paper
types, and synthesize relationships.

```python
p = registry.get("research.citation_extract.v1")
prompt = p.render(
    query="diffusion models protein design",
    source="arxiv",
    max_results=5,
)
# Feed `prompt` to an agent that has ArXivTool/PubMedTool available.
```

---

### `research.methodology_critique.v1`

**Variant:** cot  
**Inputs:** `methodology_text` (str), `field` (str)

Chain-of-thought methodology critique. Works through six dimensions sequentially:
Study Design → Sampling → Measurement → Analysis → Generalizability → Overall Verdict.
Useful for peer review assistance and methods-section feedback.

```python
p = registry.get("research.methodology_critique.v1")
prompt = p.render(
    methodology_text="We surveyed 200 students...",
    field="social psychology",
)
```

---

## Fixtures

Test fixtures live in `tests/prompts/fixtures/research/`:

| File | Content |
|------|---------|
| `attention_abstract.txt` | "Attention Is All You Need" abstract (arXiv:1706.03762) |
| `diffusion_abstract.txt` | DDPM abstract (arXiv:2006.11239) |
| `survey_methodology.txt` | Short methodology snippet for critique tests |

---

## Evaluation

Run golden + live eval:

```bash
effgen prompts eval --domain research --live --model gpt-oss-120b
```

The `paper_summary.v1` structured-output check asserts that the live output is
valid JSON with all four required keys (`abstract_summary`, `key_findings`,
`limitations`, `future_work`).

---

## Running Tests

```bash
pytest tests/prompts/test_research.py -v
```
