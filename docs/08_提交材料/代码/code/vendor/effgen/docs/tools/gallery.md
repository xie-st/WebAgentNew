# Tool Gallery

Quick-reference for every built-in tool in effGen. Each entry has a one-line description and a minimal runnable snippet.

> `execute()` is a coroutine that takes **keyword arguments** and returns a
> `ToolResult` with `.success`, `.output`, and `.error`. In a plain script, drive
> it with `asyncio.run(...)`; inside an `async` function, `await` it directly.
> Any tool can also be wired into an `Agent` for agentic use — see the end of
> this page.
>
> On failure `.output` is `None` and `.error` carries the reason, so every
> snippet whose call can fail for a reason outside the snippet — it reaches a
> network service, or it names a file you have to supply — checks `.success`
> before reading `.output`:
>
> ```python
> if not result.success:
>     raise SystemExit(result.error)
> ```
>
> An upstream 5xx, a rate limit, a missing credential, an unreachable host or a
> path that does not exist on your machine then prints the message the tool
> produced instead of raising `TypeError` on `None`. The snippets that build
> their own input leave the check out to stay short; the same two lines apply
> to any tool.

---

## Core Utilities

### Calculator
**Math expressions, unit conversions, and basic statistics.**

```python
import asyncio
from effgen.tools.builtin.calculator import Calculator

result = asyncio.run(Calculator().execute(expression="2 ** 10 + sqrt(144)"))
print(result.output["result"])  # 1036.0
```

### PythonREPL
**Interactive Python execution with persistent state across calls.**

```python
import asyncio
from effgen.tools.builtin.python_repl import PythonREPL

repl = PythonREPL()
asyncio.run(repl.execute(code="x = [i**2 for i in range(5)]"))
result = asyncio.run(repl.execute(code="print(sum(x))"))
print(result.output["stdout"])  # 30
```

Each `session_id` is backed by its own worker subprocess. At most
`max_sessions` of them stay live (default 8, or `EFFGEN_REPL_MAX_SESSIONS`);
past that, the least recently used idle session is stopped and its variables
are discarded, so using it again starts an empty session. Raise the limit if
you need more sessions warm at once:

```python
repl = PythonREPL(max_sessions=32)
```

### CodeExecutor
**Sandboxed multi-language code execution (Python, JavaScript, Bash).**

```python
import asyncio
from effgen.tools.builtin.code_executor import CodeExecutor

result = asyncio.run(CodeExecutor().execute(language="python", code="print('hello')"))
print(result.output["stdout"])  # hello
```

### BashTool
**Restricted shell command execution with allow/deny lists.**

```python
import asyncio
from effgen.tools.builtin.bash_tool import BashTool

result = asyncio.run(BashTool().execute(command="ls -lh /tmp"))
print(result.output["stdout"])
```

### FileOps
**Read, write, list, and search files on the local filesystem.**

```python
import asyncio
from effgen.tools.builtin.file_ops import FileOperations

result = asyncio.run(FileOperations().execute(operation="read", path="README.md"))
print(result.output["data"][:200])
```

---

## Web & Search

### WebSearch
**DuckDuckGo search with caching; no API key required.**

```python
import asyncio
from effgen.tools.builtin.web_search import WebSearch

result = asyncio.run(WebSearch().execute(query="effGen AI framework", num_results=5))
if not result.success:
    raise SystemExit(result.error)
for r in result.output:
    print(r["title"], r["url"])
```

### URLFetch
**Fetch and extract text content from any public URL.**

```python
import asyncio
from effgen.tools.builtin.url_fetch import URLFetchTool

result = asyncio.run(URLFetchTool().execute(url="https://example.com"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["title"])       # Example Domain
print(result.output["text"][:200])
```

### WikipediaTool
**Search Wikipedia and retrieve article summaries; free API, no key needed.**

```python
import asyncio
from effgen.tools.builtin.wikipedia_tool import WikipediaTool

result = asyncio.run(WikipediaTool().execute(
    operation="summary", query="transformer neural network", sentences=2
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["title"])          # Transformer (deep learning)
print(result.output["summary"][:300])
# operation="search" returns a result list instead: [{"title", "snippet", "url"}, ...]
```

### HTTPTool
**Generic HTTP GET/POST requests with headers and JSON body support.**

```python
import asyncio
from effgen.tools.builtin.devops import HTTPTool

result = asyncio.run(HTTPTool().execute(method="GET", url="https://httpbin.org/get"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["status"])  # 200
# result.output also carries "headers", "body", and parsed "json" when applicable
```

---

## Academic Research

### PubMedTool
**Search PubMed via NCBI E-utilities, fetch metadata, and retrieve abstracts. Built-in rate limiter (3 req/s; 10/s with `NCBI_API_KEY`).**

```python
import asyncio
from effgen.tools.builtin.pubmed import PubMedTool

result = asyncio.run(PubMedTool().execute(
    operation="search", query="CRISPR gene editing", max_results=5
))
if not result.success:
    raise SystemExit(result.error)
for article in result.output["results"]:
    print(article["pmid"], article["title"])
```

### ArXivTool
**Search arXiv papers, fetch metadata by ID, or download the PDF. Free, no auth.**

```python
import asyncio
from effgen.tools.builtin.arxiv import ArXivTool

result = asyncio.run(ArXivTool().execute(
    operation="search", query="attention is all you need", max_results=3
))
if not result.success:
    raise SystemExit(result.error)
for paper in result.output["results"]:
    print(paper["arxiv_id"], paper["title"])
```

### SemanticScholarTool
**Search papers, get details, retrieve citations and references from Semantic Scholar Graph API. Built-in backoff (100 req/5 min unauth).**

```python
import asyncio
from effgen.tools.builtin.semantic_scholar import SemanticScholarTool

result = asyncio.run(SemanticScholarTool().execute(
    operation="search", query="large language models survey", max_results=3
))
if not result.success:
    raise SystemExit(result.error)
for paper in result.output["results"]:
    print(paper["paperId"], paper["title"])
```

---

## News & RSS

### RSSFeedTool
**Fetch, browse, and full-text search any RSS/Atom feed by URL. Handles malformed feeds without raising.**

```python
import asyncio
from effgen.tools.builtin.rss import RSSFeedTool

result = asyncio.run(RSSFeedTool().execute(
    operation="latest", url="https://hnrss.org/frontpage", n=5
))
if not result.success:
    raise SystemExit(result.error)
for entry in result.output["entries"]:
    print(entry["title"])
```

### NewsTool
**Aggregate top headlines across curated reputable sources (BBC, Reuters, HN, NPR, Al Jazeera, etc.). Optional `NEWS_API_KEY` for NewsAPI.org.**

```python
import asyncio
from effgen.tools.builtin.news import NewsTool

result = asyncio.run(NewsTool().execute(operation="top_headlines", max_results=5))
if not result.success:
    raise SystemExit(result.error)
for article in result.output["articles"]:
    print(article["title"], "-", article["source"])
```

---

## YouTube

### YouTubeTranscriptTool
**Fetch YouTube video captions/transcripts without a Google API key. Supports watch?v=, youtu.be/, and shorts/ URL formats.**

```python
import asyncio
from effgen.tools.builtin.youtube_transcript import YouTubeTranscriptTool

result = asyncio.run(YouTubeTranscriptTool().execute(
    operation="get_transcript", video_id="dQw4w9WgXcQ", lang="en"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["data"]["full_text"][:200])
```

### YouTubeMetadataTool
**Fetch video or channel metadata using yt-dlp in metadata-only mode. No auth required for public content.**

```python
import asyncio
from effgen.tools.builtin.youtube_metadata import YouTubeMetadataTool

result = asyncio.run(YouTubeMetadataTool().execute(
    operation="metadata", video_id="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
))
if not result.success:
    raise SystemExit(result.error)
info = result.output["data"]
print(info["title"], "|", info["uploader"])
# Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster) | Rick Astley
```

---

## Social Media

### RedditTool
**Access Reddit top/hot posts, user submissions, and thread comments via public JSON endpoints. No OAuth required for reads. Sets `effGen/<version>` User-Agent; exponential backoff on 429.**

```python
import asyncio
from effgen.tools.builtin.reddit import RedditTool

result = asyncio.run(RedditTool().execute(
    operation="subreddit_top", subreddit="python", time_filter="day", n=5
))
if not result.success:
    raise SystemExit(result.error)
for post in result.output["data"]["posts"]:
    print(post["title"])
```

Reddit rejects unauthenticated API traffic from some networks (data centers,
cloud hosts). When that happens the result reports `success=False` with an
error naming the HTTP 403 and the retry options.

### HackerNewsTool
**Fetch top/new stories, story details, and user profiles from the Hacker News Firebase API. No auth required.**

```python
import asyncio
from effgen.tools.builtin.hackernews import HackerNewsTool

result = asyncio.run(HackerNewsTool().execute(operation="top_stories", n=5))
if not result.success:
    raise SystemExit(result.error)
for story in result.output["data"]["stories"]:
    print(story["title"], story.get("url", ""))
```

---

## Translation & Language Detection

### TranslateTool
**Translate text between languages. Primary backend: LibreTranslate (configurable via `LIBRE_TRANSLATE_URL`). Offline fallback: `argostranslate` with language packs cached in `~/.effgen/argos/`.**

```python
import asyncio
from effgen.tools.builtin.translate import TranslateTool

result = asyncio.run(TranslateTool().execute(
    operation="translate", text="Hello, world!", source="en", target="fr"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["translated_text"])  # Bonjour, le monde !
```

### LanguageDetectTool
**Detect the language of text or a batch of texts. Offline via `langdetect` — supports 55+ languages.**

```python
import asyncio
from effgen.tools.builtin.language_detect import LanguageDetectTool

result = asyncio.run(LanguageDetectTool().execute(operation="detect", text="Bonjour le monde"))
print(result.output["language"], result.output["confidence"])  # fr 1.0
```

---

## QR Codes

### QRGenerateTool
**Generate QR codes locally from any text or URL. Saves a PNG file or returns a base64 data URL. No network required.**

```python
import asyncio
from effgen.tools.builtin.qr_generate import QRGenerateTool

result = asyncio.run(QRGenerateTool().execute(
    operation="generate", data="https://effgen.org", output_path="/tmp/qr.png"
))
print(result.output["saved_path"])  # /tmp/qr.png
# data_url_return=True instead returns result.output["data_url"] ("data:image/png;base64,...")
```

### QRReadTool
**Decode QR codes and barcodes from an image file path or base64 PNG using `pyzbar` + Pillow, with OpenCV QR fallback when zbar is unavailable. Fully local.**

```python
import asyncio
from effgen.tools.builtin.qr_read import QRReadTool

result = asyncio.run(QRReadTool().execute(operation="read", image_path="/tmp/qr.png"))
for code in result.output["codes"]:
    print(code["data"])  # https://effgen.org
```

---

## Data Science

### DataFrameTool
**Load CSV/JSON, inspect with head/describe, filter rows, and aggregate with pandas.**

```python
import asyncio
from effgen.tools.builtin.data_analysis import DataFrameTool

result = asyncio.run(DataFrameTool().execute(operation="head", file_path="/tmp/data.csv"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["columns"])  # ['name', 'score']
print(result.output["data"])     # first rows as a list of dicts
```

### PlotTool
**Create line, bar, scatter, and histogram charts with matplotlib; returns a PNG file path.**

```python
import asyncio
from effgen.tools.builtin.data_analysis import PlotTool

result = asyncio.run(PlotTool().execute(
    chart_type="line", x=[1, 2, 3, 4], y=[1, 4, 9, 16], title="Squares"
))
print(result.output["file_path"])  # /tmp/effgen_plot_....png
```

### StatsTool
**Compute mean, median, std, correlation, and linear regression with NumPy.**

```python
import asyncio
from effgen.tools.builtin.data_analysis import StatsTool

result = asyncio.run(StatsTool().execute(operation="mean", data=[1, 2, 3, 4, 5]))
print(result.output["result"])  # 3.0
```

---

## Finance

### StockPriceTool
**Fetch current stock quotes from Yahoo Finance. Not financial advice.**

```python
import asyncio
from effgen.tools.builtin.finance import StockPriceTool

result = asyncio.run(StockPriceTool().execute(symbol="AAPL"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["price"], result.output["currency"])  # e.g. 325.89 USD
```

### CurrencyConverterTool
**Convert between 170+ currencies using frankfurter.app (ECB rates). No API key needed.**

```python
import asyncio
from effgen.tools.builtin.finance import CurrencyConverterTool

result = asyncio.run(CurrencyConverterTool().execute(
    amount=100, from_currency="USD", to_currency="EUR"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["converted"])  # e.g. 87.66
```

### CryptoTool
**Fetch cryptocurrency prices and market data from CoinGecko. No API key for basic use.**

```python
import asyncio
from effgen.tools.builtin.finance import CryptoTool

result = asyncio.run(CryptoTool().execute(coin="bitcoin", vs_currency="usd"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["price"])  # e.g. 65604.0
```

---

## DevOps

### GitTool
**Read-only Git operations: status, log, diff, branch list, show.**

```python
import asyncio
from effgen.tools.builtin.devops import GitTool

result = asyncio.run(GitTool().execute(operation="log", cwd=".", n=5))
print(result.output["stdout"])  # one "<hash> <subject>" line per commit
```

### DockerTool
**Read-only Docker introspection: list containers, images, and fetch logs. Requires access to the Docker daemon socket.**

```python
import asyncio
from effgen.tools.builtin.devops import DockerTool

result = asyncio.run(DockerTool().execute(operation="ps"))
print(result.output["stdout"])  # `docker ps` table of running containers
```

### SystemInfoTool
**CPU, memory, disk, and network usage via psutil.**

```python
import asyncio
from effgen.tools.builtin.devops import SystemInfoTool

result = asyncio.run(SystemInfoTool().execute(kind="cpu"))
print(result.output["cpu"]["percent"])  # e.g. 10.7
```

---

## Utilities

### JSONTool
**Parse, query (JSONPath), transform, and validate JSON data.**

```python
import asyncio
from effgen.tools.builtin.json_tool import JSONTool

result = asyncio.run(JSONTool().execute(
    operation="query", data='{"a": [1, 2, 3]}', query="$.a[*]"
))
print(result.output["result"])  # [1, 2, 3]
```

### DateTimeTool
**Current time, timezone conversion, and date arithmetic.**

```python
import asyncio
from effgen.tools.builtin.datetime_tool import DateTimeTool

result = asyncio.run(DateTimeTool().execute(operation="now", timezone="US/Eastern"))
print(result.output["datetime"])  # e.g. 2026-07-23 04:04:57
```

### TextProcessingTool
**Word count, regex find/replace, text comparison, and basic NLP operations.**

```python
import asyncio
from effgen.tools.builtin.text_processing import TextProcessingTool

result = asyncio.run(TextProcessingTool().execute(operation="word_count", text="Hello world!"))
print(result.output["word_count"])  # 2
```

### WeatherTool
**Current weather and forecasts from Open-Meteo (free, no API key).**

```python
import asyncio
from effgen.tools.builtin.weather import WeatherTool

result = asyncio.run(WeatherTool().execute(operation="current", location="San Francisco"))
if not result.success:
    raise SystemExit(result.error)
data = result.output["data"]
print(data["temperature"], data["conditions"])  # e.g. 16.6 Clear sky
```

---

## Knowledge

### StackOverflowTool
**Search Stack Overflow questions and answers via the Stack Exchange API.**

```python
import asyncio
from effgen.tools.builtin.knowledge import StackOverflowTool

result = asyncio.run(StackOverflowTool().execute(query="python async await", max_results=3))
if not result.success:
    raise SystemExit(result.error)
for q in result.output["results"]:
    print(q["title"])
```

### GitHubTool
**Search GitHub repositories, issues, and code via the public API.**

```python
import asyncio
from effgen.tools.builtin.knowledge import GitHubTool

result = asyncio.run(GitHubTool().execute(query="effGen", kind="repositories", max_results=3))
if not result.success:
    raise SystemExit(result.error)
for repo in result.output["results"]:
    print(repo["full_name"], repo["stars"])
```

### WolframAlphaTool
**Query Wolfram Alpha for computation and factual answers. Requires `WOLFRAM_ALPHA_APPID`.**

```python
import asyncio
from effgen.tools.builtin.knowledge import WolframAlphaTool

result = asyncio.run(WolframAlphaTool().execute(query="integrate x^2 from 0 to 1"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["answer"])
```

---

## Communication (Draft Only)

### EmailDraftTool
**Compose email drafts. Does NOT send — returns the draft for review.**

```python
import asyncio
from effgen.tools.builtin.communication import EmailDraftTool

result = asyncio.run(EmailDraftTool().execute(
    to=["alice@example.com"],
    subject="Meeting tomorrow",
    body="Hi Alice, can we meet at 10am?",
))
print(result.output["draft"])
# To: alice@example.com
# Subject: Meeting tomorrow
#
# Hi Alice, can we meet at 10am?
```

### SlackDraftTool
**Compose Slack message drafts. Does NOT send — returns the draft for review.**

```python
import asyncio
from effgen.tools.builtin.communication import SlackDraftTool

result = asyncio.run(SlackDraftTool().execute(channel="#general", text="Deployment complete!"))
print(result.output["draft"])  # [#general]: Deployment complete!
```

---

## Provider-Native Tools

### OpenAI Native Tools
**Activate OpenAI-hosted capabilities within an Agent: web_search, code_interpreter, file_search. Requires an OpenAI model.**

```python
from effgen.tools.builtin.openai_native import OpenAIWebSearchTool
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig

model = load_model("gpt-5-nano", provider="openai")
agent = Agent(config=AgentConfig(name="a", model=model, tools=[OpenAIWebSearchTool()]))
```

See [openai_native.md](openai_native.md) for full docs.

### Gemini Native Tools
**Activate Gemini server-side capabilities: GoogleSearchTool, GeminiUrlContextTool, GeminiCodeExecutionTool. Requires a Gemini model.**

```python
from effgen.tools.builtin.gemini_native import GoogleSearchTool
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig

model = load_model("gemini-3.1-flash-lite", provider="gemini")
agent = Agent(config=AgentConfig(name="a", model=model, tools=[GoogleSearchTool()]))
```

See [gemini_native.md](gemini_native.md) for full docs.

### Anthropic Native Tools
**Experimental computer-use tools (bash, text_editor, computer). Requires an Anthropic model.**

See [anthropic_native.md](anthropic_native.md) for full docs.

---

## OCR

### OCRTool
**Extract text from images using Tesseract (local) with OCR.space free API fallback.**

```python
import asyncio
from effgen.tools.builtin.ocr import OCRTool

result = asyncio.run(OCRTool().execute(
    operation="extract", image_path="/tmp/scan.png", lang="eng"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["text"])
```

System dep: `sudo apt-get install tesseract-ocr` / `brew install tesseract` / `choco install tesseract`.
Without a backend the result reports `success=False` and the error lists the install options.
See [ocr.md](ocr.md) for full docs.

---

## Audio Transcription

### AudioTranscribeTool
**Transcribe audio files locally via faster-whisper (CPU/GPU auto-detected); HuggingFace Inference fallback with HF_TOKEN.**

```python
import asyncio
from effgen.tools.builtin.audio_transcribe import AudioTranscribeTool

result = asyncio.run(AudioTranscribeTool().execute(
    operation="transcribe", audio_path="/tmp/clip.mp3", model_size="base"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["text"])
```

System dep (non-WAV formats): `sudo apt-get install ffmpeg` / `brew install ffmpeg`.
See [audio_transcribe.md](audio_transcribe.md) for full docs.

---

## Image Analysis

### ImageInfoTool
**Extract image metadata (size, format, mode, EXIF, color stats) and perform local resize/thumbnail operations. Zero network.**

```python
import asyncio
from effgen.tools.builtin.image_info import ImageInfoTool

result = asyncio.run(ImageInfoTool().execute(operation="info", image_path="/tmp/photo.jpg"))
if not result.success:
    raise SystemExit(result.error)
data = result.output["data"]
print(data["width"], data["height"], data["format"], data["mode"])  # e.g. 400 300 JPEG RGB
```

### ImageCaptionTool
**Generate natural-language image descriptions via the effGen vision model router (Gemini / OpenAI / MLX-VLM).**

```python
import asyncio
from effgen.tools.builtin.image_caption import ImageCaptionTool

result = asyncio.run(ImageCaptionTool().execute(operation="caption", image_path="/tmp/photo.jpg"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["caption"])  # one-sentence description of the image
```

See [image.md](image.md) for full docs.

---

## Document Parsing

### PDFTool
**Extract text, tables, and metadata from PDF files using pypdf (primary) + pdfplumber (table fallback).**

```python
import asyncio
from effgen.tools.builtin.pdf import PDFTool

result = asyncio.run(PDFTool().execute(operation="text", path="/tmp/paper.pdf"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["text"][:500])
# Also: metadata, tables, extract_images
```

### DOCXTool
**Parse Word documents (.docx) — text, paragraphs, tables, and metadata via python-docx.**

```python
import asyncio
from effgen.tools.builtin.docx import DOCXTool

result = asyncio.run(DOCXTool().execute(operation="text", path="/tmp/report.docx"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["text"])
# Also: paragraphs, tables, metadata
```

### ExcelTool
**Read Excel workbooks (.xlsx) — sheets, headers, and row data via openpyxl + pandas.**

```python
import asyncio
from effgen.tools.builtin.excel import ExcelTool

tool = ExcelTool()
sheets = asyncio.run(tool.execute(operation="sheets", path="/tmp/data.xlsx"))
if not sheets.success:
    raise SystemExit(sheets.error)
print(sheets.output["sheets"])  # ['Sheet1']

result = asyncio.run(tool.execute(
    operation="read_sheet", path="/tmp/data.xlsx", sheet_name="Sheet1"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["rows"][:3])  # header row first, e.g. [['name', 'score'], ...]
```

See [documents.md](documents.md) for full docs.

---

## Geo / Weather

### WeatherTool
**Fetch current conditions, 7-day forecasts, or historical weather from Open-Meteo (free, no auth).**

```python
import asyncio
from effgen.tools.builtin.weather import WeatherTool

result = asyncio.run(WeatherTool().execute(operation="current", lat=37.42, lon=-122.08))
if not result.success:
    raise SystemExit(result.error)
data = result.output["data"]
print(data["temperature"], data["conditions"])  # e.g. 18.0 Clear sky
# Also: forecast (days=7), historical (start_date, end_date)
```

### GeocodeTool
**Forward/reverse geocoding via Nominatim (OpenStreetMap). Honors 1 req/s rate limit; sets effGen/<version> User-Agent.**

```python
import asyncio
from effgen.tools.builtin.geocode import GeocodeTool

result = asyncio.run(GeocodeTool().execute(
    operation="geocode", address="1600 Amphitheatre Pkwy, Mountain View, CA"
))
if not result.success:
    raise SystemExit(result.error)
data = result.output["data"]
print(data["lat"], data["lon"])  # 37.4224858 -122.0855846
# Also: reverse (lat, lon) → address
```

### MapsTool
**Render static PNG maps from OpenStreetMap tiles using the staticmap library.**

```python
import asyncio
from effgen.tools.builtin.maps import MapsTool

result = asyncio.run(MapsTool().execute(
    operation="render", lat=37.42, lon=-122.08, zoom=13, dest="/tmp/map.png"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["data"]["file"])  # /tmp/map.png
# Also: bounding_box (south, west, north, east)
```

See [weather.md](weather.md), [geocode.md](geocode.md), [maps.md](maps.md) for full docs.

---

## Email

### EmailSMTPTool
**Send email via SMTP (stdlib smtplib, TLS on by default). Config: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM.**

```python
import asyncio
from effgen.tools.builtin.email_smtp import EmailSMTPTool

result = asyncio.run(EmailSMTPTool().execute(
    operation="send",
    to="alice@example.com",
    subject="Hello from effGen",
    body="This message was sent by an AI agent.",
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["accepted"])  # ['alice@example.com']
```

### EmailIMAPTool
**Read email via IMAP (stdlib imaplib). Config: IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASSWORD.**

```python
import asyncio
from effgen.tools.builtin.email_imap import EmailIMAPTool

result = asyncio.run(EmailIMAPTool().execute(operation="fetch_recent", folder="INBOX", n=5))
if not result.success:
    raise SystemExit(result.error)
for msg in result.output["data"]["messages"]:
    print(msg["subject"], msg["from"])
# Also: list_folders, search, get
```

See [email.md](email.md) for full docs.

---

## Webhooks

### SlackWebhookTool
**Post messages to Slack via incoming webhook URL (no OAuth). Config: SLACK_WEBHOOK_URL. URL is redacted in logs.**

```python
import asyncio
from effgen.tools.builtin.slack_webhook import SlackWebhookTool

result = asyncio.run(SlackWebhookTool().execute(operation="post", text="Deployment complete!"))
if not result.success:
    raise SystemExit(result.error)
print(result.output["data"]["ok"])  # True
```

### DiscordWebhookTool
**Post messages to Discord via webhook URL. Config: DISCORD_WEBHOOK_URL. URL is redacted in logs.**

```python
import asyncio
from effgen.tools.builtin.discord_webhook import DiscordWebhookTool

result = asyncio.run(DiscordWebhookTool().execute(
    operation="post", content="Build passed!", username="effGen Bot"
))
if not result.success:
    raise SystemExit(result.error)
print(result.output["data"]["ok"])  # True
```

See [webhooks.md](webhooks.md) for full docs.

---

## Using Tools in an Agent

Any tool above can be wired into an Agent for agentic use:

```python
from effgen import load_model, Agent
from effgen.core.agent import AgentConfig
from effgen.tools.builtin.arxiv import ArXivTool
from effgen.tools.builtin.translate import TranslateTool
from effgen.tools.builtin.hackernews import HackerNewsTool

model = load_model("gpt-oss-120b", provider="cerebras")
agent = Agent(config=AgentConfig(
    name="researcher",
    model=model,
    tools=[ArXivTool(), TranslateTool(), HackerNewsTool()],
    system_prompt="You are a research assistant.",
))
result = agent.run("Find the top Hacker News post right now and summarize it in one French sentence.")
print(result.text)  # one French sentence summarizing the current top story
```

## Using Presets

```python
from effgen import load_model
from effgen.presets import create_agent

model = load_model("gpt-oss-120b", provider="cerebras")

research_agent = create_agent("research", model)  # ArXiv, PubMed, SemanticScholar, RSS, News, YouTube, Reddit, HN, Wikipedia, WebSearch, PDF, DOCX, Excel
general_agent  = create_agent("general", model)   # All of the above + OCR, ImageInfo, Weather, Geocode, Maps, Email, Webhooks, Translate, QR, ...
media_agent    = create_agent("media", model)     # AudioTranscribeTool + ImageCaptionTool
notify_agent   = create_agent("notify", model)    # EmailSMTPTool + EmailIMAPTool + SlackWebhookTool + DiscordWebhookTool
```
