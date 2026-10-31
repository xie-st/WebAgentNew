# PubMedTool

Search the biomedical literature index at PubMed via NCBI's free E-utilities API.

## Quick start

```python
import asyncio
from effgen.tools.builtin import PubMedTool

async def main():
    tool = PubMedTool()
    result = await tool.execute(
        operation="search",
        query="transformer attention",
        max_results=5,
    )
    for paper in result.output["results"]:
        print(paper["title"], "-", paper["url"])

asyncio.run(main())
```

## Operations

| operation  | required params | description |
|------------|-----------------|-------------|
| `search`   | `query`         | Full-text search; returns PMIDs + summaries |
| `fetch`    | `pmid`          | Full XML metadata (title, authors, abstract, DOI, journal, year) |
| `abstract` | `pmid`          | Abstract text only |

`pmid` may be a single id (`"26952870"`) or a comma-separated list.

## Output schema

`search`:

```jsonc
{
  "query": "transformer attention",
  "count": 5,
  "total_count": 12345,
  "results": [
    {
      "pmid": "...",
      "title": "...",
      "authors": ["..."],
      "journal": "...",
      "pubdate": "...",
      "doi": "...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/<pmid>/"
    }
  ],
  "source": "pubmed"
}
```

`fetch` / `abstract` return a similar envelope with a `results` list where each
record carries `pmid`, `title`, `authors`, `journal`, `year`, `abstract`,
`doi`, and `url`.

Chapters indexed from the NCBI Bookshelf — StatPearls, GeneReviews — are records
too. A chapter has no journal, so `journal` names the book that contains it
(`"StatPearls"`), and a record for a whole book rather than one of its chapters is
titled by that book. `year` is the publication year, or the date range PubMed
publishes in place of one (`"1998 Dec-1999 Jan"`).

A `search` that matches nothing succeeds with `count: 0` — that is an answer, and
so is a `warninglist` naming a phrase PubMed does not index. A search NCBI answers
with an `ERROR` field is reported as a failed call instead, with the message
quoted.

A `fetch` or `abstract` names the record it wants, so a response carrying none is
also a failed call: `ToolResult.success` is `False` and `error` names the id,
quotes whatever NCBI said about it, and points at the record's page so the id can
be checked. NCBI sheds load with an empty body as well as with a 429, so a fetch
of an id that does resolve on that page is worth retrying.

## Rate limits

NCBI E-utilities allow **3 req/sec without an API key**, or **10 req/sec** when
the `NCBI_API_KEY` environment variable is set. PubMedTool enforces both via a
process-local token bucket; no extra configuration needed.

## Notes

- No authentication required. The tool is registered automatically in the
  `research` preset.
- All HTTP calls go through `urllib.request`; no third-party deps.
