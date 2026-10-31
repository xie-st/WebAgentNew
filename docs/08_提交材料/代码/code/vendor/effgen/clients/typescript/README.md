# effgen-client

Official TypeScript / JavaScript client for the [effGen](https://effgen.org) API server.

Works in Node 18+, Deno, Bun, and modern browsers. Uses the built-in
`fetch` and `ReadableStream` — no extra runtime dependencies.

## Install

```bash
npm install effgen-client
```

## Usage

```ts
import { EffGenClient } from "effgen-client";

const client = new EffGenClient({
  baseUrl: "http://localhost:8000",
  apiKey: process.env.EFFGEN_API_KEY,
});

// Simple chat
const res = await client.chat("What is 2+2?", { tools: ["calculator"] });
console.log(res.content);

// Streaming chat
for await (const chunk of client.chatStream("Tell me a story")) {
  process.stdout.write(chunk);
}

// Embeddings
const vecs = await client.embed(["Hello", "World"]);
console.log(vecs.length, vecs[0].length);

// Health
const health = await client.health();
console.log(health.ok);
```

## Error handling

All errors derive from `EffGenClientError`:

- `EffGenConnectionError` — network failure
- `EffGenTimeoutError` — request timed out
- `EffGenAPIError` — non-2xx response
  - `EffGenAuthError` — 401 / 403
  - `EffGenRateLimitError` — 429
  - `EffGenServerError` — 5xx

Connection, timeout, 429, and 5xx errors are retried automatically with
exponential backoff (configurable via `maxRetries` / `backoffBaseMs`).

## License

Apache-2.0
