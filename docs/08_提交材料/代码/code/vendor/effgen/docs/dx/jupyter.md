# effGen Jupyter Magics

The `effgen.jupyter` extension adds three IPython magics for interactive use of the effGen framework inside Jupyter notebooks and the IPython REPL.

## Installation

```bash
pip install "effgen[jupyter]"
```

Or install all extras:

```bash
pip install "effgen[all]"
```

## Loading the extension

Add this to a notebook cell (or your IPython startup file):

```python
%load_ext effgen.jupyter
```

---

## Magics

### `%effgen_chat` — one-shot chat

Send a single message to an effGen model and display the response inline.

**Syntax**

```
%effgen_chat [--model MODEL] [--server URL] <message...>
```

**Examples**

```python
# Simple chat
%effgen_chat Tell me a one-sentence joke about gradient descent

# Use a specific model
%effgen_chat --model cerebras:zai-glm-4.7 Explain attention in transformers

# Route through a running effGen server
%effgen_chat --server http://localhost:8080 What is 42 * 7?
```

**Output:** A short `effGen (model, Xs)` header followed by effGen's rich result
card — the markdown-rendered answer plus a compact metric strip (status · time ·
tokens · cost) and, when tools ran, a collapsible step trace (the
`AgentResponse._repr_html_` card). In a plain terminal/IPython it falls back to a
readable text representation.

---

### `%%effgen_agent` — agentic cell

Run the cell body as a task for an effGen agent.  The agent can use tools and produces a final answer plus a tool-use trace.

**Syntax**

```
%%effgen_agent [preset] [--model MODEL] [--tools TOOL [TOOL ...]]
<task description in cell body>
```

**Examples**

```python
%%effgen_agent coding
Write a Python function to compute the nth Fibonacci number iteratively.
Return type-annotated code with a docstring.
```

```python
%%effgen_agent --model cerebras:zai-glm-4.7
Search Wikipedia for "mixture of experts" and summarize the key ideas.
```

```python
%%effgen_agent --tools calculator web_search
What is the population of France in millions, divided by 3.14159?
```

When `[preset]` names a built-in preset (`math`, `research`, `coding`, `general`, `rag`, `minimal`), the agent is created via `effgen.presets.create_agent`, so it is wired with that preset's tools — e.g. `math` and `general` include a calculator, so `Compute 17*23.` is actually computed (391), not guessed.

**Output:** effGen's rich result card with the agent's final answer and, when tools were used, a collapsible step trace listing each tool call.

> **Note on context windows.** Large presets such as `general` enable many tools, whose descriptions can exceed the context window of small models (e.g. a local model with a 4K–8K window). When that happens the magic prints a clear hint suggesting a smaller preset (e.g. `math`) or a larger-context model instead of a raw provider error.

---

### `%effgen_metrics` — Prometheus snapshot

Snapshot and display the current Prometheus counters registered by the effGen framework.

**Syntax**

```
%effgen_metrics
```

**Example**

```python
# Run a few agent calls then inspect metrics
%effgen_metrics
```

**Output:** An HTML table showing all `effgen_*` (and other registered) metric names and current values.

---

## Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `EFFGEN_JUPYTER_MODEL` | *(unset)* | Jupyter-specific default model for all magics |
| `EFFGEN_DEFAULT_MODEL` | *(unset)* | Framework-wide default model (used when `EFFGEN_JUPYTER_MODEL` is unset) |
| `EFFGEN_JUPYTER_SERVER_URL` | *(unset)* | If set, `%effgen_chat` routes through this server instead of running in-process |

These can also be set per-cell using `--model` / `--server` flags.

> **No hidden default model.** The magics never silently pick a (potentially
> paid) cloud model. If you pass no `--model` and neither environment variable is
> set, the magic prints a short, actionable message telling you to pass `--model`
> or set `EFFGEN_DEFAULT_MODEL` — run `effgen models list` to see options or
> `effgen doctor` to check which providers are usable.

---

## Tips

- Use `%%effgen_agent coding` for code generation tasks — the "coding" preset adds helpful system context.
- The in-process mode (default) requires effGen to be installed with the correct provider SDK (e.g. `pip install "effgen[cerebras]"`).
- Set `EFFGEN_DEV_MODE=1` if routing through a local server without OIDC configured.
