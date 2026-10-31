# RAG Pipeline

effGen v0.2.0 includes a RAG (Retrieval Augmented Generation) pipeline with document ingestion, advanced chunking, hybrid search, reranking, and source attribution.

## Quick Start — RAG Preset

The fastest way to use RAG:

```python
from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = create_agent("rag", model, knowledge_base="./docs/")

result = agent.run("What does the architecture document say about scaling?")
print(result.output)
print(result.citations)  # [Citation(source="architecture.md", ...)]
```

## Step-by-Step Pipeline

### 1. Document Ingestion

```python
from effgen.rag import DocumentIngester

ingester = DocumentIngester()
chunks = ingester.ingest("./docs/")

# Supported formats (built-in): txt, md, json, jsonl, csv, html
# Optional formats: pdf (requires pypdf), docx (requires python-docx), epub (requires ebooklib)

# Options
chunks = ingester.ingest(
    "./docs/",
    recursive=True,          # Recurse into subdirectories
    deduplicate=True,        # SHA-256 content deduplication
)
```

### 2. Advanced Chunking

```python
from effgen.rag.chunking import (
    SemanticChunker,     # Split on semantic boundaries
    CodeChunker,         # Language-aware: functions, classes, blocks
    TableChunker,        # Preserve table structure
    HierarchicalChunker, # Maintain document hierarchy
)

# Code-aware chunking (supports py, js, ts, go, rust, java)
chunker = CodeChunker(language="python", max_chunk_size=1000)
chunks = chunker.chunk(document_text)
```

### 3. Hybrid Search

```python
from effgen.rag import HybridSearchEngine

engine = HybridSearchEngine(chunks)

# Combines dense retrieval, BM25, keyword matching, and metadata filtering
# Results fused via Reciprocal Rank Fusion
results = engine.search("scaling architecture", top_k=5)

# Custom weights
results = engine.search(
    "scaling architecture",
    top_k=5,
    weights={"dense": 0.4, "bm25": 0.3, "keyword": 0.2, "metadata": 0.1},
)
```

### 4. Reranking

```python
from effgen.rag.reranker import LLMReranker, RuleBasedReranker, CrossEncoderReranker

# LLM-based reranking (free, uses the agent's own model)
reranker = LLMReranker(model)
reranked = reranker.rerank(results, query="scaling architecture", top_k=3)

# Rule-based (boost by recency, source authority, keyword match, title match)
reranker = RuleBasedReranker(boost_keywords=["scaling", "performance"])
reranked = reranker.rerank(results, query="scaling architecture")

# Cross-encoder (optional, requires sentence-transformers)
reranker = CrossEncoderReranker()
reranked = reranker.rerank(results, query="scaling architecture")
```

### 5. Context Building with Citations

```python
from effgen.rag import ContextBuilder

builder = ContextBuilder(max_tokens=2048)
context, citations = builder.build(reranked)

# context: formatted text with inline [1], [2] citation markers
# citations: list of Citation objects with source, page, chunk_id, relevance_score, quote
```

### 6. Source Attribution

```python
from effgen.rag.attribution import CitationTracker

tracker = CitationTracker()
tracker.add_citations(citations)

# Verify citations against the response
verified = tracker.verify(response_text, citations)

# Extract which citations were actually used
used = tracker.extract_used_indices(response_text)
```

## AgentResponse Fields

When using the RAG preset or any agent with RAG tools:

```python
result = agent.run("What database does effGen use?")
print(result.output)       # "effGen uses SQLite for..."
print(result.citations)    # [Citation(source="architecture.md", quote="...")]
print(result.sources)      # ["architecture.md"] (deduplicated)
```
