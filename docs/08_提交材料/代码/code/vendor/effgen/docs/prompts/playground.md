# effGen Prompt Playground

The playground provides two ways to explore prompt templates:

1. **Interactive REPL** — a command-line session where you select a prompt, set variables, render it, run it against a model, and save your session.
2. **Non-interactive mode** — one-shot CLI commands for scripting and automation.

---

## Interactive REPL

Launch with:

```bash
effgen prompts playground
```

### REPL commands

| Command | Description |
|---------|-------------|
| `select <name>` | Pick a prompt from the registry (seeds variables with fixture defaults) |
| `set <key> <value>` | Bind a variable; value is JSON-decoded if possible, else treated as a plain string |
| `unset <key>` | Remove a variable binding |
| `render` | Print the rendered prompt using current variable bindings |
| `run [--model <id>]` | Render and send to a model; stores output in session |
| `save [<path>]` | Save the session to a JSON file (auto-names under `~/.effgen/playground/` if path omitted) |
| `load <path>` | Restore a previously saved session |
| `reload` | Hot-reload the selected prompt's module (picks up edits without restart) |
| `list [--domain <d>]` | List registered prompts, optionally filtered by domain |
| `show <name>` | Show a prompt's schema, fixture, and rendered preview |
| `help` | Show command reference |
| `exit` / `quit` / Ctrl-D | Exit the REPL |

### Example session

```
effGen Prompt Playground
Type 'help' for commands, 'exit' to quit.

[(none)]> list --domain research
research.citation_extract.v1         research   tool        ...
research.literature_review.v1.cot    research   cot         ...
research.literature_review.v1.zero_shot  research  zero_shot  ...
...

[(none)]> select research.literature_review.v1.zero_shot
Selected: research.literature_review.v1.zero_shot
  Domain: research  Variant: zero_shot
  Pre-loaded fixture inputs: ['topic', 'years_range', 'max_papers']

[research.literature_review.v1.zero_shot]> set topic "diffusion models for protein design"
Set topic = "diffusion models for protein design"

[research.literature_review.v1.zero_shot]> set years_range "2023-2026"
Set years_range = "2023-2026"

[research.literature_review.v1.zero_shot]> set max_papers 10
Set max_papers = 10

[research.literature_review.v1.zero_shot]> render
╭── Rendered Prompt ──────────────────────────────╮
│ You are an expert research assistant. Conduct   │
│ a literature review on "diffusion models for   │
│ ...                                             │
╰─────────────────────────────────────────────────╯

[research.literature_review.v1.zero_shot]> run --model gpt-oss-120b
Running with model: gpt-oss-120b ...
╭── Model Output (gpt-oss-120b) ──────────────────╮
│ **Overview**                                   │
│ Diffusion models have emerged as ...           │
╰─────────────────────────────────────────────────╯

[research.literature_review.v1.zero_shot]> save
Session saved to: /home/user/.effgen/playground/20260519T120000Z_research.literature_review.v1.zero_shot.json

[research.literature_review.v1.zero_shot]> exit
Bye!
```

### Restoring a session

```
[(none)]> load ~/.effgen/playground/20260519T120000Z_research.literature_review.v1.zero_shot.json
Session loaded: prompt='research.literature_review.v1.zero_shot'  vars=3  renders=1  runs=1
```

### Hot-reload

Edit a template file under `effgen/prompts/library/domains/` and run `reload` in the REPL — the module is re-imported and the registry is refreshed. Useful during prompt development.

---

## Non-interactive mode

### Render a prompt

Print the rendered prompt to stdout without calling a model:

```bash
effgen prompts render <name>
effgen prompts render <name> --input input.json
```

`input.json` is merged over the prompt's fixture defaults:

```json
{
  "topic": "transformer architectures",
  "years_range": "2021-2024",
  "max_papers": 5
}
```

Example:

```bash
effgen prompts render research.literature_review.v1.zero_shot \
  --input inputs/research.json
```

### Render + run

Render and send to a model in one step:

```bash
effgen prompts run <name> --model <id>
effgen prompts run <name> --model <id> --input input.json
```

Example:

```bash
effgen prompts run coding.docstring_fill.v1 \
  --model gpt-oss-120b \
  --input inputs/coding.json
```

---

## Session file format

Sessions are stored as plain JSON:

```json
{
  "prompt_name": "research.literature_review.v1.zero_shot",
  "variables": {
    "topic": "diffusion models for protein design",
    "years_range": "2023-2026",
    "max_papers": 10
  },
  "render_history": ["...rendered text..."],
  "run_history": [
    {
      "model": "gpt-oss-120b",
      "rendered": "...rendered text...",
      "output": "...model response...",
      "timestamp": "2026-05-19T12:00:00Z"
    }
  ],
  "created_at": "2026-05-19T12:00:00Z",
  "updated_at": "2026-05-19T12:05:00Z"
}
```

Sessions are loaded via `PlaygroundSession.load(path)` in Python:

```python
from effgen.prompts.library.session import PlaygroundSession

session = PlaygroundSession.load("my_session.json")
print(session.run_history[-1].output)
```

---

## Python API

You can drive the playground programmatically:

```python
from effgen.cli.playground import cmd_render, cmd_run, PlaygroundREPL

# Render to stdout
cmd_render("research.literature_review.v1.zero_shot", {
    "topic": "neural radiance fields",
    "years_range": "2021-2024",
    "max_papers": 5,
})

# Render + run
cmd_run("coding.docstring_fill.v1", {}, model="gpt-oss-120b")

# Embed REPL in your script
repl = PlaygroundREPL(default_model="gpt-oss-120b")
repl.run()
```

---

## Available prompts

Use `effgen prompts list` to see all registered prompts, or filter by domain:

```bash
effgen prompts list --domain research
effgen prompts list --domain coding
effgen prompts list --domain data
effgen prompts list --domain legal
effgen prompts list --domain medical
effgen prompts list --domain creative
effgen prompts list --domain business
```

See the full gallery in `docs/prompts/gallery.md` when it is generated for the release.
