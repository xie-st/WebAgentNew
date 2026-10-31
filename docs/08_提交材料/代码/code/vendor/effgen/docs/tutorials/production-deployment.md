# Production Deployment

This guide covers deploying effGen as an API server, with authentication, rate limiting, and the OpenAI-compatible gateway endpoints.

## API Server Quick Start

```bash
# Start the server
effgen serve --host 0.0.0.0 --port 8000

# With authentication
EFFGEN_API_KEY=your-secret-key effgen serve --port 8000

# With rate limiting
EFFGEN_RATE_LIMIT=120 effgen serve --port 8000
```

## OpenAI-Compatible API

The API server exposes OpenAI-compatible endpoints:

```bash
# Chat completions (drop-in OpenAI replacement)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "tools": [{"type": "function", "function": {"name": "calculator"}}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-key" \
  -d '{"model": "gpt-3.5-turbo", "messages": [...], "stream": true}'

# Embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer your-key" \
  -d '{"input": ["hello", "world"], "model": "text-embedding-small"}'
```

### Model Aliases

| OpenAI Name | Maps To |
|-------------|---------|
| `gpt-4` | Qwen2.5-7B-Instruct |
| `gpt-3.5-turbo` | Qwen2.5-3B-Instruct |

## Docker Deployment

```bash
# Build
docker build -t effgen .

# Run
docker run -p 8000:8000 \
  -e EFFGEN_API_KEY=your-secret \
  -e EFFGEN_RATE_LIMIT=60 \
  --gpus all \
  effgen

# With Docker Compose (see examples/deployment/)
docker compose up -d
```

## Multi-Tenancy

```python
from effgen.api.tenancy import TenantManager

manager = TenantManager()

# Create tenants with rate limits and model restrictions
tenant = manager.create_tenant(
    name="team-alpha",
    rate_limit=100,  # requests per minute
    model_restrictions=["Qwen2.5-3B-Instruct"],
    tool_permissions=["calculator", "web_search"],
)

# Issue API keys
api_key, raw_key = manager.create_api_key(tenant.id)
# raw_key is shown once — give to the tenant
# api_key.key_hash is stored (constant-time verification)
```

## Production Middleware

The server automatically includes:

- **CORS** — configurable origins for cross-domain requests
- **Request ID** — `X-Request-ID` header injected on every response
- **GZip compression** — automatic for large responses
- **Graceful shutdown** — in-flight requests complete before server stops
- **Request queue** — priority-based with backpressure (returns 503 when full)
- **Agent pool** — pre-warmed agents with configurable min/max pool size

## Client SDKs

### Python

```python
from effgen.client import EffGenClient

client = EffGenClient(base_url="http://localhost:8000", api_key="your-key")

# Sync
response = client.chat("What is 2+2?")

# Streaming
for chunk in client.chat_stream_sync("Tell me a story"):
    print(chunk, end="")

# Async
import asyncio
response = asyncio.run(client.achat("What is 2+2?"))

# Health check
health = client.health()
print(health.status)  # "ok"
```

### TypeScript/JavaScript

```typescript
import { EffGenClient } from 'effgen-client';

const client = new EffGenClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-key',
});

const response = await client.chat('What is 2+2?');

// Streaming
for await (const chunk of client.chatStream('Tell me a story')) {
  process.stdout.write(chunk);
}
```

## Monitoring

```bash
# Health endpoint
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics

# effGen CLI health check
effgen health
```

Import the Grafana dashboard from `configs/grafana/effgen-dashboard.json` for:
- Response latency (p50/p95/p99)
- Throughput (requests/sec)
- Error rate
- Tool execution breakdown
- GPU memory usage

## Scaling

See the [Scaling Guide](../guides/scaling.md) for:
- Request queue tuning
- Agent pool sizing
- Model pool with LRU eviction
- Continuous batching
- Multi-GPU distribution
