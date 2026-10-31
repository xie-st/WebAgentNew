# RSSFeedTool

Fetch and search any RSS or Atom feed from a URL.

**Module:** `effgen.tools.builtin.rss`  
**Tool name:** `rss_feed`  
**Requires:** `feedparser` (`pip install feedparser` or `pip install effgen[rss]`)  
**Auth:** None required

## Operations

| Operation | Description |
|-----------|-------------|
| `fetch` | Retrieve all entries from the feed |
| `latest` | Retrieve the N most recent entries (default: 10) |
| `search_in_feed` | Filter entries by keyword in title/summary/author |

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operation` | string | No | `fetch` | One of `fetch`, `latest`, `search_in_feed` |
| `url` | string | **Yes** | — | RSS/Atom feed URL |
| `n` | integer | No | `10` | Max entries (for `latest`) |
| `query` | string | No | — | Keyword to search (for `search_in_feed`) |

## Output Schema

```json
{
  "success": true,
  "operation": "latest",
  "url": "https://hnrss.org/frontpage",
  "feed": {
    "title": "Hacker News: Front Page",
    "link": "https://news.ycombinator.com/",
    "description": "..."
  },
  "entries": [
    {
      "title": "Show HN: My project",
      "link": "https://...",
      "published": "Sat, 16 May 2026 12:00:00 +0000",
      "summary": "...",
      "author": "",
      "id": "https://..."
    }
  ],
  "data": {
    "feed": {"title": "Hacker News: Front Page", "link": "https://news.ycombinator.com/"},
    "entries": [
      {
        "title": "Show HN: My project",
        "link": "https://..."
      }
    ]
  },
  "entry_count": 10,
  "malformed": false,
  "error": null
}
```

## Examples

```python
import asyncio
from effgen.tools.builtin import RSSFeedTool

tool = RSSFeedTool()

# Fetch latest 5 HN posts
result = asyncio.run(tool.execute(
    operation="latest",
    url="https://hnrss.org/frontpage",
    n=5
))

# Search for Python-related posts
result = asyncio.run(tool.execute(
    operation="search_in_feed",
    url="https://hnrss.org/frontpage",
    query="python"
))
```

## Error Handling

Malformed or unavailable feeds are handled without raising:
- Missing/unreachable feed: returns `success=False` with a descriptive `error` field
- HTML or non-feed URLs: logs a warning and returns `success=True` with empty `entries`
- Partially malformed feed (for example encoding issues): logs a warning and returns whatever entries could be parsed
- Individual unparseable entries: silently skipped

## Preset Integration

`rss_feed` is included in the **research** and **general** presets.
