# SemanticScholarTool

Query the [Semantic Scholar Graph API](https://api.semanticscholar.org/api-docs/graph)
for academic papers, citations, and references. Free; an optional
`SEMANTIC_SCHOLAR_API_KEY` env var lifts the rate-limit quota.

## Quick start

```python
import asyncio
from effgen.tools.builtin import SemanticScholarTool

async def main():
    tool = SemanticScholarTool()

    out = await tool.execute(query="transformer attention", max_results=5)
    for paper in out.output["results"]:
        print(paper["paperId"], paper["title"], paper["citationCount"])

    paper = await tool.execute(
        operation="paper", paper_id="ARXIV:1706.03762",
    )
    print(paper.output["result"]["title"])

asyncio.run(main())
```

## Operations

| operation     | required params | description |
|---------------|-----------------|-------------|
| `search`      | `query`         | Paper search across the Graph index |
| `paper`       | `paper_id`      | Single paper lookup |
| `citations`   | `paper_id`      | Papers that cite this paper |
| `references`  | `paper_id`      | Papers that this paper cites |

`paper_id` accepts the native Semantic Scholar id, or prefixed external ids:
`DOI:...`, `ARXIV:...`, `MAG:...`, `ACL:...`, `PMID:...`, `URL:...`.

## Output schema

Each paper record is normalised to:

```jsonc
{
  "paperId": "...",
  "externalIds": { "DOI": "...", "ArXiv": "..." },
  "title": "...",
  "abstract": "...",
  "venue": "...",
  "year": 2017,
  "authors": ["..."],
  "citationCount": 123,
  "referenceCount": 45,
  "influentialCitationCount": 12,
  "isOpenAccess": true,
  "fieldsOfStudy": ["Computer Science"],
  "url": "https://www.semanticscholar.org/paper/..."
}
```

## Rate limits

The public, unauthenticated endpoint is capped at **100 requests / 5 minutes
globally** (shared with every other caller). The tool implements:

- a process-local sliding-window limiter, and
- exponential backoff with up to 4 retries on HTTP 429.
- a `paper/search/match` fallback for exact-title search queries when the broad
  `paper/search` endpoint is temporarily throttled.
- an arXiv-title fallback for `paper("ARXIV:<id>")` when the direct paper lookup
  endpoint is throttled.

Set `SEMANTIC_SCHOLAR_API_KEY` (sent via the `x-api-key` header) to use a
partner-program quota — the local limiter still applies but never blocks since
it is a strict subset of the keyed quota.
