/**
 * effgen-client — TypeScript/JavaScript client for the effGen API.
 *
 * Fetch-based with streaming support via ReadableStream. Works in Node 18+,
 * Deno, Bun, and modern browsers.
 */

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  name?: string;
  tool_call_id?: string;
}

export interface ChatResponse {
  content: string;
  model?: string;
  toolCalls: unknown[];
  usage: Record<string, unknown>;
  raw: unknown;
}

export interface HealthStatus {
  status: string;
  details: Record<string, unknown>;
  ok: boolean;
}

export interface EffGenClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  backoffBaseMs?: number;
  fetchImpl?: typeof fetch;
}

export class EffGenClientError extends Error {}
export class EffGenConnectionError extends EffGenClientError {}
export class EffGenTimeoutError extends EffGenClientError {}

export class EffGenAPIError extends EffGenClientError {
  statusCode?: number;
  payload?: unknown;
  constructor(msg: string, statusCode?: number, payload?: unknown) {
    super(msg);
    this.statusCode = statusCode;
    this.payload = payload;
  }
}
export class EffGenAuthError extends EffGenAPIError {}
export class EffGenRateLimitError extends EffGenAPIError {}
export class EffGenServerError extends EffGenAPIError {}

export class EffGenClient {
  readonly baseUrl: string;
  readonly apiKey?: string;
  readonly timeoutMs: number;
  readonly maxRetries: number;
  readonly backoffBaseMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: EffGenClientOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? "http://localhost:8000").replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 60_000;
    this.maxRetries = opts.maxRetries ?? 3;
    this.backoffBaseMs = opts.backoffBaseMs ?? 500;
    this.fetchImpl = opts.fetchImpl ?? fetch;
  }

  private headers(extra?: Record<string, string>): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return { ...h, ...(extra ?? {}) };
  }

  private url(path: string): string {
    if (!path.startsWith("/")) path = "/" + path;
    return `${this.baseUrl}${path}`;
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise((r) => setTimeout(r, ms));
  }

  private backoff(attempt: number): number {
    const base = this.backoffBaseMs * Math.pow(2, attempt);
    return base + Math.random() * base * 0.2;
  }

  private raiseForStatus(status: number, payload: unknown): void {
    if (status >= 200 && status < 300) return;
    let msg = `HTTP ${status}`;
    if (payload && typeof payload === "object") {
      const p = payload as Record<string, unknown>;
      if (typeof p.error === "string") msg = p.error;
      else if (typeof p.message === "string") msg = p.message;
    }
    if (status === 401 || status === 403) throw new EffGenAuthError(msg, status, payload);
    if (status === 429) throw new EffGenRateLimitError(msg, status, payload);
    if (status >= 500) throw new EffGenServerError(msg, status, payload);
    throw new EffGenAPIError(msg, status, payload);
  }

  private async request<T = unknown>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    let lastErr: unknown = null;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const resp = await this.fetchImpl(this.url(path), {
          method,
          headers: this.headers(),
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
        });
        clearTimeout(timer);
        let payload: unknown = null;
        const text = await resp.text();
        if (text) {
          try {
            payload = JSON.parse(text);
          } catch {
            payload = text;
          }
        }
        this.raiseForStatus(resp.status, payload);
        return payload as T;
      } catch (e: unknown) {
        clearTimeout(timer);
        if (e instanceof EffGenAPIError) {
          if (e instanceof EffGenServerError || e instanceof EffGenRateLimitError) {
            lastErr = e;
          } else {
            throw e;
          }
        } else if ((e as { name?: string })?.name === "AbortError") {
          lastErr = new EffGenTimeoutError("request timed out");
        } else {
          lastErr = new EffGenConnectionError(String((e as Error)?.message ?? e));
        }
        if (attempt < this.maxRetries) {
          await this.sleep(this.backoff(attempt));
          continue;
        }
      }
    }
    throw lastErr;
  }

  async chat(
    message: string,
    opts: { tools?: string[]; model?: string } = {},
  ): Promise<ChatResponse> {
    const body: Record<string, unknown> = {
      model: opts.model ?? "effgen-default",
      messages: [{ role: "user", content: message }],
    };
    if (opts.tools) body.tools = opts.tools;
    const payload = await this.request<Record<string, unknown>>(
      "POST",
      "/v1/chat/completions",
      body,
    );
    return parseChat(payload);
  }

  async *chatStream(
    message: string,
    opts: { model?: string } = {},
  ): AsyncIterableIterator<string> {
    const body = {
      model: opts.model ?? "effgen-default",
      messages: [{ role: "user", content: message }],
      stream: true,
    };
    const resp = await this.fetchImpl(this.url("/v1/chat/completions"), {
      method: "POST",
      headers: this.headers({ Accept: "text/event-stream" }),
      body: JSON.stringify(body),
    });
    if (resp.status >= 400 || !resp.body) {
      const txt = await resp.text();
      this.raiseForStatus(resp.status, txt);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const chunk = parseSSELine(line);
        if (chunk) yield chunk;
      }
    }
  }

  async embed(
    texts: string[],
    model: string = "text-embedding-small",
  ): Promise<number[][]> {
    const payload = await this.request<{ data: { embedding: number[] }[] }>(
      "POST",
      "/v1/embeddings",
      { model, input: texts },
    );
    if (!payload || !Array.isArray(payload.data)) {
      throw new EffGenAPIError("Malformed embeddings response", undefined, payload);
    }
    return payload.data.map((d) => d.embedding);
  }

  async health(): Promise<HealthStatus> {
    const payload = await this.request<Record<string, unknown>>("GET", "/health");
    const status = String(payload?.status ?? "unknown");
    return {
      status,
      details: payload ?? {},
      ok: ["ok", "healthy", "up", "ready"].includes(status.toLowerCase()),
    };
  }
}

function parseChat(payload: Record<string, unknown>): ChatResponse {
  let content = "";
  let toolCalls: unknown[] = [];
  const choices = (payload.choices as unknown[]) ?? [];
  if (choices.length > 0) {
    const msg = ((choices[0] as Record<string, unknown>).message ?? {}) as Record<
      string,
      unknown
    >;
    content = typeof msg.content === "string" ? msg.content : "";
    toolCalls = (msg.tool_calls as unknown[]) ?? [];
  }
  return {
    content,
    model: payload.model as string | undefined,
    toolCalls,
    usage: (payload.usage as Record<string, unknown>) ?? {},
    raw: payload,
  };
}

function parseSSELine(line: string): string | null {
  if (!line.startsWith("data:")) return null;
  const data = line.slice(5).trim();
  if (!data || data === "[DONE]") return null;
  try {
    const obj = JSON.parse(data);
    const choices = obj?.choices;
    if (Array.isArray(choices) && choices.length > 0) {
      const delta = choices[0].delta ?? {};
      if (typeof delta.content === "string" && delta.content.length > 0) {
        return delta.content;
      }
    }
    return null;
  } catch {
    return data;
  }
}
