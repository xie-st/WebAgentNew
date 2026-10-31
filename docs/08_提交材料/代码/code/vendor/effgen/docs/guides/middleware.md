# Middleware

effGen ships guardrails, observability, reliability and cost tracking as
subsystems. Middleware is the general form of the same idea: somewhere to put
behaviour effGen does not ship — an approval gate, a cache, a redaction pass, a
per-run spend cap, a custom trace exporter — without patching the loop.

There are three points, each with a "before" and an "after":

| Point | Fires | Hooks |
|---|---|---|
| the run | once per `run()` | `before_run`, `after_run` |
| each model call | every generation, retries included | `before_model_call`, `after_model_call` |
| each tool call | every dispatch | `before_tool_call`, `after_tool_call` |

Subclass `AgentMiddleware` and override only what you need. Every hook has a
default that does nothing.

```python
from effgen import Agent, AgentConfig
from effgen.core.middleware import AgentMiddleware

class SpendCap(AgentMiddleware):
    def __init__(self, max_searches: int = 3):
        self.max_searches = max_searches
        self.used = 0

    def before_tool_call(self, ctx):
        if ctx.tool_name != "web_search":
            return None
        if self.used >= self.max_searches:
            return "Skipped: this run has spent its search budget."
        self.used += 1
        return None

agent = Agent(AgentConfig(model="gpt-5-nano", middleware=[SpendCap()]))
```

## Modifying and short-circuiting

A `before_` hook receives a context it can edit in place, and whatever it
returns decides what happens next:

- **`None`** — carry on. The (possibly edited) request is used.
- **anything else** — short-circuit. The real call does not happen and the
  returned value is used as its result. The matching `after_` hook still runs.

```python
class AnswerFromCache(AgentMiddleware):
    def before_run(self, ctx):
        hit = my_cache.get(ctx.task)
        return AgentResponse(output=hit, success=True) if hit else None

    def after_run(self, ctx, response):
        my_cache[ctx.task] = response.output
        return response
```

An `after_` hook receives the result and returns the one to use, so it can
transform as well as observe.

## Ordering

`before_` hooks run in the order the middleware were given; `after_` hooks run
in reverse. Middleware therefore nest the way context managers do — the first
one listed is the outermost.

```
before_run:  outer -> inner -> [the run] -> after_run: inner -> outer
```

The first `before_` hook to short-circuit wins, and the ones after it do not
run.

## Per call

`run(..., middleware=[...])` **adds** to the configured middleware for that call
only. It does not replace them, and it does not affect the next call.

```python
agent.run(task, middleware=[Timing()])
```

## Passing values between your own hooks

`ctx.metadata` is free space effGen never reads. Use it to get something from
a `before_` hook to its `after_`:

```python
class Timed(AgentMiddleware):
    def before_run(self, ctx):
        ctx.metadata["started"] = time.monotonic()
        return None

    def after_run(self, ctx, response):
        elapsed = time.monotonic() - ctx.metadata["started"]
        response.metadata["wall_clock"] = elapsed
        return response
```

## Failure

A hook that raises is not caught. The exception reaches the caller like any
other error in the run — which is what lets an approval gate that refuses stop
the run outright.

## What ships

| Middleware | What it does |
|---|---|
| `LoggingMiddleware` | Logs the run, every model call and every tool call. Also a worked example: it touches four hooks and changes nothing. |
| `ToolApprovalMiddleware` | Asks a callback before named tools run; a refusal is reported to the model instead of the tool's output. |

```python
from effgen.core.middleware import ToolApprovalMiddleware

def ask(tool_name, tool_input):
    return input(f"run {tool_name}({tool_input})? [y/N] ").lower() == "y"

agent = Agent(AgentConfig(
    model="gpt-5-nano",
    middleware=[ToolApprovalMiddleware(approve=ask, tools=["shell", "file_write"])],
))
```

## Contexts

| Context | Fields |
|---|---|
| `RunContext` | `task` (editable), `agent_name`, `mode`, `metadata` |
| `ModelCallContext` | `prompt` (editable), `model_name`, `kwargs` (editable), `attempt`, `run` |
| `ToolCallContext` | `tool_name`, `tool_input` (editable), `run` |

## Cost

An agent with no middleware pays one boolean test at each of the three points.
The hooks are safe to leave in place on the hot path.

## Related

- [Guardrails](../tutorials/guardrails.md) — the shipped policy layer.
- [Reliability](../observability/reliability.md) — retries, circuit breakers,
  fallback chains.
- [Tool calls](../models/tool-calls.md) — what a run reports about its calls.
