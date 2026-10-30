# FAQ & Troubleshooting

## General

**Q: Do I need a GPU to run effGen?**
A: For local models, a GPU is strongly recommended. Use `quantization="4bit"` to fit models on smaller GPUs. You can also use API backends (OpenAI, Anthropic, Gemini) which don't require a GPU.

**Q: What models work best with effGen?**
A: effGen is optimized for Small Language Models (1B-7B params). Recommended:
- `Qwen/Qwen2.5-3B-Instruct` — best balance of speed and quality
- `Qwen/Qwen2.5-1.5B-Instruct` — fastest, good for simple tasks
- `Qwen/Qwen2.5-7B-Instruct` — highest quality, needs more VRAM

**Q: Does effGen require any paid APIs?**
A: No. All features work without paid APIs by default. WebSearch uses DuckDuckGo (free), Wikipedia uses free APIs, and models run locally.

**Q: How do I use effGen with OpenAI/Claude instead of local models?**
A:
```python
model = load_model("openai:gpt-4o")       # Requires OPENAI_API_KEY
model = load_model("anthropic:claude-3-haiku")  # Requires ANTHROPIC_API_KEY
```

## Troubleshooting

**Q: `CUDA out of memory` error**
A: Use 4-bit quantization:
```python
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
```
Or select a specific GPU: `CUDA_VISIBLE_DEVICES=0 python script.py`

**Q: Agent doesn't call any tools**
A: Check your system prompt encourages tool use. The model may be too small to follow tool-calling format reliably. Try a 3B+ model.

**Q: Agent loops without producing a final answer**
A: Increase `max_iterations` or use a more capable model. The ReAct loop may need more steps for complex tasks.

**Q: ImportError when importing effGen**
A: Ensure you installed with `pip install effgen` (or `pip install -e .` for development). Check `python -c "import effgen; print(effgen.__version__)"`.

**Q: How do I add my own tools?**
A: See the [Custom Tools Guide](tutorials/custom-tools.md). Extend `BaseTool`, implement `metadata` and `_execute()`.

**Q: How do I distribute my custom tools?**
A: Package them as a plugin. See [Plugin Development Guide](guides/plugin-development.md).
