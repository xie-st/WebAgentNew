# RedditTool

Fetch posts and comments from Reddit using public JSON endpoints — no OAuth required.

**Module:** `effgen.tools.builtin.reddit`  
**Tool name:** `reddit`  
**Auth:** None (read-only public API)  
**Rate limiting:** Automatic exponential backoff on HTTP 429

## Operations

| Operation | Description |
|-----------|-------------|
| `subreddit_top` | Top posts from a subreddit, filtered by time |
| `subreddit_hot` | Hot (currently trending) posts from a subreddit |
| `user_submissions` | Recent posts submitted by a user |
| `thread_comments` | Comments on a specific thread (by ID or permalink) |

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operation` | string | **Yes** | — | One of the four operations above |
| `subreddit` | string | Cond. | — | Subreddit name without `r/` (for `subreddit_*`) |
| `username` | string | Cond. | — | Reddit username (for `user_submissions`) |
| `thread_id` | string | Cond. | — | Thread ID or full permalink URL (for `thread_comments`) |
| `time_filter` | string | No | `day` | One of `hour`, `day`, `week`, `month`, `year`, `all` (for `subreddit_top`) |
| `n` | integer | No | `25` | Max posts/comments to return (1–100) |

## Output Schema

```json
{
  "success": true,
  "subreddit": "python",
  "sort": "top",
  "time_filter": "day",
  "requested_count": 25,
  "posts": [
    {
      "id": "abc123",
      "title": "Example post",
      "author": "user",
      "subreddit": "python",
      "url": "https://example.com",
      "permalink": "https://reddit.com/r/python/comments/abc123/...",
      "score": 1234,
      "upvote_ratio": 0.97,
      "num_comments": 42,
      "created_utc": 1700000000,
      "selftext": "...",
      "is_self": false,
      "flair": ""
    }
  ],
  "post_count": 10,
  "source_exhausted": false,
  "error": null
}
```

## Examples

```python
from effgen.tools.builtin.reddit import RedditTool
import asyncio

tool = RedditTool()

# Top posts from r/python today
result = asyncio.run(tool.execute(
    operation="subreddit_top",
    subreddit="python",
    time_filter="day",
    n=10
))
print(result.output["posts"])

# Hot posts from r/MachineLearning
result = asyncio.run(tool.execute(
    operation="subreddit_hot",
    subreddit="MachineLearning",
    n=5
))

# Comments on a thread
result = asyncio.run(tool.execute(
    operation="thread_comments",
    thread_id="abc123",
    subreddit="python",
    n=50
))
```

## Notes

- Uses `https://old.reddit.com/.json` endpoints — no OAuth needed for public content.
- If `old.reddit.com` returns HTTP 403 from a deployment network, the same path is retried on `www.reddit.com`.
- User-Agent is set to `effGen/<version>` per Reddit's API Terms of Service.
- On HTTP 429, backs off exponentially (2s → 4s) across 3 total attempts.
- `source_exhausted` is true when Reddit returns fewer posts/comments than requested.
- `selftext` is truncated to 500 characters.
- Deleted/removed content is filtered from `thread_comments`.
