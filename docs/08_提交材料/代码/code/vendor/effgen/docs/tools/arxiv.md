# ArXivTool

Search arXiv, fetch paper metadata, or download a paper PDF via the free arXiv
Atom API. No authentication required.

The class is importable as both `ArXivTool` (new) and `ArxivTool` (legacy alias
kept for back-compat with v0.2.4).

## Quick start

```python
import asyncio
from effgen.tools.builtin import ArXivTool

async def main():
    tool = ArXivTool()

    search = await tool.execute(query="transformer attention", max_results=5)
    for paper in search.output["results"]:
        print(paper["arxiv_id"], paper["title"])

    paper = await tool.execute(operation="fetch", arxiv_id="1706.03762")
    print(paper.output["result"]["title"])

    pdf = await tool.execute(
        operation="download_pdf",
        arxiv_id="1706.03762",
        dest="/tmp/attention.pdf",
    )
    print("saved", pdf.output["bytes"], "bytes to", pdf.output["path"])

asyncio.run(main())
```

## Operations

| operation      | required params         | description |
|----------------|-------------------------|-------------|
| `search`       | `query`                 | Full-text search across arXiv |
| `fetch`        | `arxiv_id`              | Metadata for a single paper |
| `download_pdf` | `arxiv_id`, [`dest`]    | Stream the paper PDF to `dest` (or `cwd/<id>.pdf`) |

`arxiv_id` accepts the bare id (`1706.03762`), a URL (`https://arxiv.org/abs/1706.03762`),
or `pdf` variants — they are normalised internally.

## Output schema

Each paper record contains:

```jsonc
{
  "arxiv_id": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani", "..."],
  "summary": "...",
  "published": "2017-06-12T...",
  "updated": "...",
  "primary_category": "cs.CL",
  "categories": ["cs.CL", "cs.LG"],
  "comment": "15 pages, 5 figures",
  "doi": null,
  "url": "http://arxiv.org/abs/1706.03762v...",
  "pdf_url": "http://arxiv.org/pdf/1706.03762v...pdf"
}
```

## Notes

- arXiv's export endpoint occasionally returns 429s or read timeouts under
  load — the tool surfaces these as `ConnectionError`.
- `download_pdf` writes raw bytes to disk; pair it with a PDF parser to extract
  text in agent flows.
