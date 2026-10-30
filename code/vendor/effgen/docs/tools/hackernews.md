# HackerNewsTool

Fetch Hacker News stories, items, and user profiles via the official Firebase API.

**Module:** `effgen.tools.builtin.hackernews`  
**Tool name:** `hackernews`  
**Backend:** `https://hacker-news.firebaseio.com/v0/`  
**Auth:** None required

## Operations

| Operation | Description |
|-----------|-------------|
| `top_stories` | Current top-ranked stories (up to 500) |
| `new_stories` | Newest submitted stories |
| `best_stories` | Best-ranked stories |
| `ask_stories` | Current Ask HN posts |
| `show_stories` | Current Show HN posts |
| `job_stories` | Current job listings |
| `story` | Single item (story, comment, poll) by ID |
| `user` | HN user profile by username |

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operation` | string | **Yes** | — | One of the operations above |
| `n` | integer | No | `30` | Max stories to return for list operations (1–500) |
| `item_id` | integer | Cond. | — | HN item ID (for `story`) |
| `username` | string | Cond. | — | HN username (for `user`) |

## Output Schema

### List operations (top_stories, new_stories, …)

```json
{
  "success": true,
  "list": "topstories",
  "stories": [
    {
      "id": 39000001,
      "type": "story",
      "title": "Example HN Story",
      "url": "https://example.com",
      "hn_url": "https://news.ycombinator.com/item?id=39000001",
      "external_url": "https://example.com",
      "score": 350,
      "author": "username",
      "time": 1700000000,
      "descendants": 42,
      "text": ""
    }
  ],
  "story_count": 10,
  "error": null
}
```

### `story` operation

```json
{
  "success": true,
  "item": { ... },
  "error": null
}
```

### `user` operation

```json
{
  "success": true,
  "user": {
    "id": "pg",
    "karma": 155000,
    "created": 1160418000,
    "about": "...",
    "submitted_count": 1500,
    "submitted_ids": [...]
  },
  "error": null
}
```

## Examples

```python
from effgen.tools.builtin.hackernews import HackerNewsTool
import asyncio

tool = HackerNewsTool()

# Top 10 stories
result = asyncio.run(tool.execute(operation="top_stories", n=10))
for story in result.output["stories"]:
    print(story["title"], story["score"])

# Fetch a specific item
result = asyncio.run(tool.execute(operation="story", item_id=1))
print(result.output["item"])

# User profile
result = asyncio.run(tool.execute(operation="user", username="pg"))
print(result.output["user"]["karma"])
```

## Notes

- Stories are fetched in parallel (up to 10 concurrent requests) for speed.
- `url` is always populated. For Ask HN, jobs, polls, or stories without an external URL, it falls back to the HN discussion URL.
- `external_url` contains only the off-site URL when Hacker News provides one.
- `text` field (for Ask HN, jobs) is truncated to 500 characters.
- `submitted_ids` returns up to the 20 most recent submission IDs; use `story` to hydrate them.
- No documented rate limit, but the tool is polite — avoid very high `n` values in rapid loops.
