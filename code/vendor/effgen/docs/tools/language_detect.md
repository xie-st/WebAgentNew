# LanguageDetectTool

Detect the language of text. Fully offline — no external API or network required.

**Tool name:** `language_detect`  
**Category:** Data Processing  
**Auth required:** No  
**Network:** Not required (uses langdetect, a pure-Python library)

---

## Backend

Uses [langdetect](https://pypi.org/project/langdetect/) — a pure-Python port of Google's
language-detection library. Supports 55+ languages. Confidence scores are returned for the
top candidate and all other likely candidates.

Empty or whitespace-only input fails with `InvalidInputError` through the returned
`ToolResult` metadata instead of producing a silent empty result.

---

## Operations

### `detect`

Detect the language of a single text.

| Parameter | Type   | Default  | Description                             |
|-----------|--------|----------|-----------------------------------------|
| `text`    | string | required | Text to detect (up to 5000 characters)  |

**Response:**
```json
{
  "success": true,
  "language": "fr",
  "language_name": "French",
  "confidence": 0.9999,
  "data": {
    "language": "fr",
    "language_name": "French",
    "confidence": 0.9999,
    "all_candidates": [
      {"language": "fr", "language_name": "French", "confidence": 0.9999}
    ],
    "text_preview": "Bonjour le monde"
  },
  "error": null
}
```

### `detect_batch`

Detect languages for multiple texts in one call.

| Parameter | Type         | Default  | Description               |
|-----------|--------------|----------|---------------------------|
| `texts`   | list[string] | required | List of texts to detect   |

**Response:**
```json
{
  "success": true,
  "count": 3,
  "results": [
    {"index": 0, "text_preview": "Hello", "language": "en", "confidence": 0.71, "success": true},
    {"index": 1, "text_preview": "Bonjour", "language": "fr", "confidence": 1.00, "success": true},
    {"index": 2, "text_preview": "Hola", "language": "es", "confidence": 1.00, "success": true}
  ]
}
```

---

## Examples

```python
import asyncio
from effgen.tools.builtin import LanguageDetectTool

tool = LanguageDetectTool()

# Single text
result = asyncio.run(tool.execute(operation="detect", text="Bonjour le monde"))
print(result.output["language"])       # "fr"
print(result.output["confidence"])     # e.g. 0.9999
print(result.output["language_name"])  # "French"

# Batch
texts = ["Hello", "Bonjour", "Hola", "こんにちは"]
result = asyncio.run(tool.execute(operation="detect_batch", texts=texts))
for item in result.output["results"]:
    print(item["language"], item["confidence"])
```

### With an agent

```python
from effgen.presets import create_agent

agent = create_agent("general", model)
response = agent.run("Detect the language of: 'Guten Morgen, wie geht es Ihnen?'")
```

---

## Supported Languages (partial list)

`af`, `ar`, `bg`, `bn`, `ca`, `cs`, `cy`, `da`, `de`, `el`, `en`, `es`, `et`, `fa`,
`fi`, `fr`, `gu`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `kn`, `ko`, `lt`, `lv`,
`mk`, `ml`, `mr`, `ms`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `so`,
`sq`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`, `uk`, `ur`, `vi`, `zh-cn`, `zh-tw`

---

## Notes

- `langdetect` uses non-deterministic output by default. Results are stable for longer texts
  but may vary for very short texts (< 10 characters).
- For short strings like single words, provide more context for better accuracy.
- For the best accuracy, pass complete sentences rather than fragments.

---

## Preset Integration

`LanguageDetectTool` is included in the **general** preset.
