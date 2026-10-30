# Terminal appearance

This page documents how the `effgen` command-line tool decides what to draw:
color, the logo and first-run welcome, live animation, and the branded landing
shown when you run `effgen` with no command. Every visual is TTY-only and opt-out;
piped, redirected, and `--json` output stays plain text and is unaffected by the
settings below.

## Color and themes

`effgen` ships four color themes. The default look is unchanged from earlier
releases; the others are opt-in.

| Theme | Use it for |
|---|---|
| `default` | the standard look |
| `high-contrast` | bright, bold hues and no dim text (low-vision readers) |
| `monochrome` | structure only — bold/underline carry meaning, no hue |
| `light` | tuned for a light terminal background |

Select a theme for one command with `--theme`, or for the session with the
`EFFGEN_THEME` environment variable:

```bash
effgen --theme high-contrast doctor
export EFFGEN_THEME=light
```

`--theme` may appear before or after the subcommand.

### Turning color off

`effgen` honors the [`NO_COLOR`](https://no-color.org/) convention. Set
`NO_COLOR` to any value and every command renders structure — tables, panels,
the logo — with no color, whatever theme is selected:

```bash
NO_COLOR=1 effgen models list
```

Color is also off automatically when output is not a terminal (a pipe, a file, a
CI log) and when `TERM=dumb`.

## The bare `effgen` landing

Running `effgen` with no command **on an interactive terminal** shows a short
landing: the logo, the version, and a handful of quick actions. Press **Enter**
to continue into the interactive setup wizard (the historical behavior), or pick
a quick action:

```
  [Enter]  set up and run an agent (interactive wizard)
  [c]      chat — open an interactive session
  [q]      quickstart — a 2-minute guided first run
  [d]      doctor — check which providers are ready
  [m]      models — browse the model catalog
  [h]      help — all commands            [x] exit
```

The landing appears only when both input and output are terminals. When output
is piped or redirected, when you pass `--quiet`, or in CI, `effgen` skips the
landing and behaves exactly as before. The logo uses a Unicode block banner on
terminals that can render it and a plain ASCII banner otherwise; it collapses to
a one-line wordmark on a narrow terminal.

## The first-run welcome

The first time you use `effgen` on an interactive terminal, it prints a one-time
welcome pointing at `effgen doctor` and a hello agent, then records a flag under
`~/.effgen` so it never shows again. It is silent under `--quiet`, in CI, and on
non-interactive output.

## Live animation

Long-running commands show a live status line and progress bars on an
interactive terminal. Turn the animation off — leaving plain, single-line status
text that is safe to pipe or log — in any of these ways:

```bash
effgen run "..." --no-animation
export EFFGEN_NO_ANIM=1
```

`NO_COLOR` and a non-interactive stdout also disable animation.

## Answers, streaming, and the run summary

`effgen run` and `effgen chat` share one presentation for a model's answer, so a
one-shot run and a conversational turn read as the same tool:

- **The answer.** `run` frames the finished answer in a bordered panel; `chat`
  shows it inline under an `assistant` label. Both render markdown — headings,
  lists, fenced code, and tables — through the same renderer, and both take
  their color from the selected theme. A run that stopped at its iteration cap
  has no answer: the panel says what stopped it and what to do, and the tool
  output and reasoning the run had reached follows below it, labelled as partial
  progress so retrieved passages are not read as a result — a second panel in
  `run`, a labelled block under the reply in `chat`.
- **Streaming.** With `--stream` (and in `chat`), the answer renders live as it
  arrives: a brief `Thinking…` spinner until the first token, then a markdown
  region that updates in place. On a pipe, a redirect, a non-terminal, or with
  `NO_COLOR`, streaming falls back to plain token-by-token text with no spinner
  and no cursor, so captured output is clean.
- **The summary.** After a run, a single line reports the outcome at a glance —
  for example `✓ Done in 3.2s · 2 tools · 1,204 tokens · $0.0006`. A run stopped
  at its iteration cap is marked as a partial result, and a failure names its
  reason. `chat` shows a compact per-turn footer with a running session total.

`--quiet` prints the answer alone (no header, spinner, or summary). Under
`--json`, standard output stays a single JSON document and all of the above is
suppressed. On a non-UTF-8 terminal (for example `PYTHONIOENCODING=ascii`) the
status glyphs and separators fall back to ASCII stand-ins instead of failing.

## Tips

`effgen` surfaces an occasional one-line tip after commands you watch
interactively. Silence them with:

```bash
export EFFGEN_TIPS=0
```

## Summary of environment variables

| Variable | Effect |
|---|---|
| `NO_COLOR` | disable all color (structure still renders) |
| `EFFGEN_THEME` | select a theme (`default`, `high-contrast`, `monochrome`, `light`) |
| `EFFGEN_NO_ANIM` | disable live spinners/progress animation |
| `EFFGEN_TIPS=0` | silence the rotating tips |
| `EFFGEN_HOME` | relocate the per-user state directory (default `~/.effgen`) |
