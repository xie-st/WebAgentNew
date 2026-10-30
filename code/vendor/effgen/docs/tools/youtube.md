# YouTube Tools

Two tools for working with YouTube content without a Google API key:

- **YouTubeTranscriptTool** — fetch video captions/transcripts
- **YouTubeMetadataTool** — fetch video/channel metadata via yt-dlp

## YouTubeTranscriptTool

**Module:** `effgen.tools.builtin.youtube_transcript`  
**Tool name:** `youtube_transcript`  
**Requires:** `youtube-transcript-api` and `yt-dlp` for fallback subtitle metadata (`pip install effgen[youtube]`)  
**Auth:** None required

### URL Formats Supported

All of the following are accepted as `video_id`:

| Format | Example |
|--------|---------|
| Plain video ID | `dQw4w9WgXcQ` |
| watch?v= URL | `https://www.youtube.com/watch?v=dQw4w9WgXcQ` |
| youtu.be/ short URL | `https://youtu.be/dQw4w9WgXcQ` |
| /shorts/ URL | `https://www.youtube.com/shorts/dQw4w9WgXcQ` |
| /embed/ URL | `https://www.youtube.com/embed/dQw4w9WgXcQ` |

### Operations

| Operation | Description |
|-----------|-------------|
| `get_transcript` | Fetch captions for a video in the specified language |
| `list_available_languages` | List all languages for which transcripts exist |
| `translated` | Fetch captions translated to a target language |

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operation` | string | No | `get_transcript` | One of the operations above |
| `video_id` | string | **Yes** | — | Video ID or URL |
| `lang` | string | No | `en` | Language code for `get_transcript` (e.g. `en`, `fr`, `ja`) |
| `target_lang` | string | No | — | Target language for `translated` |

### Output Schema

```json
{
  "success": true,
  "operation": "get_transcript",
  "video_id": "dQw4w9WgXcQ",
  "data": {
    "video_id": "dQw4w9WgXcQ",
    "language": "English",
    "language_code": "en",
    "is_generated": false,
    "snippet_count": 61,
    "full_text": "We're no strangers to love...",
    "snippets": [
      {"text": "We're no strangers to love", "start": 1.36, "duration": 2.0}
    ]
  },
  "error": null
}
```

### Examples

```python
import asyncio
from effgen.tools.builtin import YouTubeTranscriptTool

tool = YouTubeTranscriptTool()

# Get English transcript by video ID
result = asyncio.run(tool.execute(
    operation="get_transcript",
    video_id="dQw4w9WgXcQ",
    lang="en",
))

# Get transcript via URL
result = asyncio.run(tool.execute(
    operation="get_transcript",
    video_id="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
))

# List available languages
result = asyncio.run(tool.execute(
    operation="list_available_languages",
    video_id="dQw4w9WgXcQ",
))

# Get French translation (may be blocked on cloud IPs)
result = asyncio.run(tool.execute(
    operation="translated",
    video_id="dQw4w9WgXcQ",
    target_lang="fr",
))
```

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Video has no captions | Returns `success=False` with `NoTranscriptAvailableError` message |
| Private/deleted video | Returns `success=False` with descriptive error |
| IP blocked by YouTube | Returns `success=False` with `ConnectionError` message |
| yt-dlp subtitle fallback blocked/outdated | Returns `success=False` with upgrade/fallback details |
| Unsupported URL format | Returns `success=False` with parse error |

> **Note on IP blocking:** YouTube may block requests from cloud provider IPs (AWS, GCP, Azure, etc.). If you encounter `IpBlocked` errors, see the [youtube-transcript-api README](https://github.com/jdepoix/youtube-transcript-api#working-around-ip-bans-requestblocked-or-ipblocked-exception) for workarounds.
> effGen also attempts a yt-dlp subtitle fallback when transcript-api fetches are blocked, but YouTube can block timedtext subtitle downloads from the same host.

---

## YouTubeMetadataTool

**Module:** `effgen.tools.builtin.youtube_metadata`  
**Tool name:** `youtube_metadata`  
**Requires:** `yt-dlp` (`pip install yt-dlp` or `pip install effgen[youtube]`)  
**Auth:** None required for public content

> **Caveat:** yt-dlp requires occasional updates as YouTube changes its internal API. Keep yt-dlp up to date: `pip install -U yt-dlp`.

### Operations

| Operation | Description |
|-----------|-------------|
| `metadata` | Get metadata for a single video (title, description, views, duration, tags, etc.) |
| `channel` | List recent videos from a channel URL |

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operation` | string | No | `metadata` | One of `metadata`, `channel` |
| `video_id` | string | For `metadata` | — | Video ID or URL |
| `channel_url` | string | For `channel` | — | Channel URL (e.g. `https://www.youtube.com/@ChannelName`) |
| `max_videos` | integer | No | `10` | Max videos to return for `channel` (1–50) |

### Output Schema — `metadata`

```json
{
  "success": true,
  "operation": "metadata",
  "data": {
    "id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "description": "The official video...",
    "channel": "Rick Astley",
    "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
    "duration": 213,
    "view_count": 1773000000,
    "like_count": 19000000,
    "upload_date": "20091025",
    "categories": ["Music"],
    "tags": ["rick astley", "never gonna give you up"],
    "thumbnail": "https://i.ytimg.com/...",
    "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "age_limit": 0,
    "live_status": "not_live",
    "availability": "public",
    "subtitle_languages": ["en", "fr", "ja"],
    "auto_caption_languages": ["en"]
  },
  "error": null
}
```

### Output Schema — `channel`

```json
{
  "success": true,
  "operation": "channel",
  "data": {
    "channel_url": "https://www.youtube.com/@RickAstleyYT",
    "videos": [
      {"id": "...", "title": "...", "view_count": 1000000, ...}
    ],
    "count": 10
  },
  "error": null
}
```

### Examples

```python
import asyncio
from effgen.tools.builtin import YouTubeMetadataTool

tool = YouTubeMetadataTool()

# Get video metadata
result = asyncio.run(tool.execute(
    operation="metadata",
    video_id="dQw4w9WgXcQ",
))

# Get metadata via short URL
result = asyncio.run(tool.execute(
    operation="metadata",
    video_id="https://youtu.be/dQw4w9WgXcQ",
))

# Get channel's recent videos
result = asyncio.run(tool.execute(
    operation="channel",
    channel_url="https://www.youtube.com/@RickAstleyYT",
    max_videos=5,
))
```

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Private/deleted video | Returns `success=False` with descriptive error |
| Age-restricted content | Returns `success=False` with sign-in error |
| Channel requires auth | Returns `success=False` with sign-in error |
| yt-dlp not installed | Returns `success=False` with install instructions |
| yt-dlp extraction failure | Returns `success=False` with `python -m pip install -U yt-dlp` upgrade guidance |
| Timeout | Returns `success=False` with timeout message |

> **Note on channel access:** YouTube may require sign-in to browse channel pages from cloud provider IPs. The `metadata` operation for individual videos is more reliable.

---

## Preset Integration

Both tools are included in the **research** preset:

```python
from effgen.presets import create_agent

agent = create_agent("research", model)
# youtube_transcript and youtube_metadata are available automatically
```

## Installation

```bash
# YouTube tools only
pip install effgen[youtube]

# Or install packages directly
pip install youtube-transcript-api yt-dlp
```
