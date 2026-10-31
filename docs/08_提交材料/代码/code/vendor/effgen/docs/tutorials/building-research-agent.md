# Building a Research Agent

Create an agent that searches the web, academic literature, and reference
sources, then synthesizes information.

## Quick Way

```python
from effgen.presets import create_agent
from effgen import load_model

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("research", model)

result = agent.run("What are the latest developments in quantum computing?")
print(result.output)
```

## Custom Configuration

```python
from effgen import Agent, AgentConfig, load_model
from effgen.tools.builtin import (
    ArXivTool,
    PubMedTool,
    SemanticScholarTool,
    URLFetchTool,
    WebSearch,
    WikipediaTool,
)

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(AgentConfig(
    name="research-agent",
    model=model,
    tools=[
        WebSearch(),
        URLFetchTool(),
        WikipediaTool(),
        ArXivTool(),
        PubMedTool(),
        SemanticScholarTool(),
    ],
    system_prompt=(
        "You are a thorough research assistant. When answering questions:\n"
        "1. Search the web for current information\n"
        "2. Consult Wikipedia for background context\n"
        "3. Use arXiv, PubMed, or Semantic Scholar for academic sources\n"
        "4. Fetch specific URLs for detailed information\n"
        "5. Synthesize findings and cite your sources"
    ),
    max_iterations=10,
    temperature=0.5,
))

result = agent.run("Compare Python and Rust for systems programming")
```

## Notes

- `WebSearch` uses DuckDuckGo (free, no API key required)
- `WikipediaTool` queries Wikipedia's free API
- `URLFetchTool` downloads and extracts text from web pages
- `ArXivTool`, `PubMedTool`, and `SemanticScholarTool` are included in the
  `research` preset for paper discovery and metadata lookup
- Set higher `max_iterations` for research — the agent may need multiple search rounds
