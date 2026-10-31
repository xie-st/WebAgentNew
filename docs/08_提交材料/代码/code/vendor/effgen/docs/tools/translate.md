# TranslateTool

Translate text between languages using LibreTranslate (with automatic offline fallback via argostranslate).

**Tool name:** `translate`  
**Category:** Data Processing  
**Auth required:** No  
**Network:** Optional (Argos fallback works fully offline)

---

## Backends

1. **LibreTranslate** (primary) — Sends requests to a LibreTranslate instance.
   Without configuration, the tool tries a small built-in list of free public
   mirrors in order (`translate.fedilab.app`, `translate.argosopentech.com`,
   `libretranslate.de`) and uses the first responsive one. Public mirrors are
   rate-limited and can return 403/429 — the tool falls through to the next
   mirror and then to Argos automatically.
   Override the entire list with the `LIBRE_TRANSLATE_URL` environment variable
   to pin a single instance (recommended for production — host your own).

2. **argostranslate** (fallback) — Fully local neural translation.  
   Language packs are downloaded on first use and cached in `~/.effgen/argos/`.  
   Subsequent calls reuse the cached packs with no network required.

The tool automatically falls back to Argos when LibreTranslate is unavailable.  
Translation latency depends on which backend is used: LibreTranslate is fast for common pairs;
Argos may be slower on first use due to pack download, but is instant once cached.

If both backends are unavailable, `tool.execute(...)` returns a failed `ToolResult`
whose `metadata["error_type"]` is `TranslateServiceUnavailable` and whose error
message explains how to fix the backend configuration.

---

## Operations

### `translate`

Translate text from one language to another.

| Parameter | Type   | Default  | Description                                      |
|-----------|--------|----------|--------------------------------------------------|
| `text`    | string | required | Text to translate (up to 5000 characters)        |
| `source`  | string | `"auto"` | Source language code (`"en"`, `"fr"`, …), or `"auto"` to detect |
| `target`  | string | `"en"`   | Target language code                             |

**Response:**
```json
{
  "success": true,
  "translated_text": "Bonjour le monde",
  "source_language": "en",
  "target_language": "fr",
  "backend": "libretranslate",
  "data": {
    "translated_text": "Bonjour le monde",
    "source_language": "en",
    "target_language": "fr",
    "backend": "libretranslate",
    "original_text": "Hello world"
  },
  "error": null
}
```

### `available_pairs`

List supported language pairs.

**Response:**
```json
{
  "success": true,
  "pairs": [{"source": "en", "target": "fr"}, ...],
  "count": 50,
  "source": "libretranslate"
}
```

---

## Examples

```python
import asyncio
from effgen.tools.builtin import TranslateTool

tool = TranslateTool()

# EN → FR
result = asyncio.run(tool.execute(operation="translate", text="Hello world", source="en", target="fr"))
print(result.output["translated_text"])  # "Bonjour le monde"

# Auto-detect source
result = asyncio.run(tool.execute(operation="translate", text="Guten Morgen", source="auto", target="en"))
print(result.output["translated_text"])  # "Good morning"

# List available pairs
result = asyncio.run(tool.execute(operation="available_pairs"))
print(result.output["count"])  # e.g. 50
```

### With an agent

```python
from effgen.presets import create_agent

agent = create_agent("general", model)
response = agent.run("Translate 'The quick brown fox' to Spanish and Japanese")
```

---

## Language Codes

Common codes: `en` (English), `fr` (French), `es` (Spanish), `de` (German), `ja` (Japanese),
`zh-Hans` (Chinese Simplified), `ar` (Arabic), `ru` (Russian), `pt` (Portuguese), `it` (Italian).

Use `available_pairs` to get the full list supported by the current backend.

---

## Offline Setup (argostranslate)

Language packs download automatically on first use per pair. To pre-install a pack:

```python
import argostranslate.package
argostranslate.package.update_package_index()
argostranslate.package.install_package_for_language_pair("en", "fr")
```

Packs are cached at `~/.effgen/argos/` and reused on subsequent calls.

---

## Environment Variables

| Variable              | Default                              | Description                        |
|-----------------------|--------------------------------------|------------------------------------|
| `LIBRE_TRANSLATE_URL` | (built-in mirror list, see Backends) | Pin LibreTranslate to a single URL |

---

## Preset Integration

`TranslateTool` is included in the **general** preset.
