# effGen Cookbook — Multimodal Walkthroughs

This cookbook provides five end-to-end walkthroughs for multimodal input in effGen v0.2.8. Each walkthrough is a self-contained Python snippet with prose explanation. All examples are also extracted into `tests/cookbook/test_cookbook_runs.py` and can be executed with `pytest -m live`.

---

## Walkthroughs

| # | Notebook | Modality | Provider | Description |
|---|----------|----------|----------|-------------|
| 01 | [multimodal_01_image_qa.md](multimodal_01_image_qa.md) | Image | Gemini + OpenAI | Ask natural-language questions about an image |
| 02 | [multimodal_02_audio_transcribe_reason.md](multimodal_02_audio_transcribe_reason.md) | Audio | Gemini + OpenAI Whisper | Transcribe an audio clip then analyze its content |
| 03 | [multimodal_03_video_summarize.md](multimodal_03_video_summarize.md) | Video | Gemini (native) + OpenAI (frames) | Sample keyframes and produce a narrative summary |
| 04 | [multimodal_04_ocr_plus_llm.md](multimodal_04_ocr_plus_llm.md) | Image + Text | Gemini + prompt library | OCR text extraction then structured LLM extraction |
| 05 | [multimodal_05_bullet_chart_read.md](multimodal_05_bullet_chart_read.md) | Image | Gemini + OpenAI | Read a bar chart and answer comparison questions |

---

## Prerequisites

```bash
pip install "effgen[all]"

# For video frame-sampling (optional)
sudo apt-get install ffmpeg   # Ubuntu/Debian
brew install ffmpeg           # macOS
```

Set API keys in `.env` or environment:

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
```

---

## Quick Start

```python
from effgen import image_from, audio_from, load_model
from effgen.core.messages import Message, Role
from effgen.presets import create_agent

# Create multimodal agent
model = load_model("gemini-3.1-flash-lite", provider="gemini")
agent = create_agent("multimodal", model)

# Image Q&A
img = image_from("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg")
msg = Message(role=Role.USER, content=[img, "What objects are visible in this image?"])
result = agent.run(msg)
print(result.output)
```

---

## Running Cookbook Tests

```bash
# Run all cookbook tests (requires API keys)
pytest tests/cookbook/ -m live -v

# Run a specific walkthrough
pytest tests/cookbook/test_cookbook_runs.py::test_cookbook_01_image_qa -v
```

---

## Architecture Reference

For the full multimodal architecture — message schema, capability gating, preprocessing, and provider support matrix — see [docs/multimodal/overview.md](../multimodal/overview.md).
