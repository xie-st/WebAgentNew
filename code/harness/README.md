# WebAgent harness (A1 + A2 + B1 + C1)

Phase-1 deliverable for `D:\Dwen\Alpha26fall\WebAgentNew`. A1 provides the
immutable protocol, single-task session lifecycle, the one canonical web-tool
schema, budget/context helpers, events/evidence with redaction, non-sensitive
configuration, and offline fake model/fixtures. A2 adds the vendored EffGen
interactive loop and `effgen_bridge`, which drives that loop from an A1 session.
B1 adds the real text-model adapter, the Playwright browser backend, a
browser-use browser-layer trial, the seeded local fixture site, and the
`doctor` / `run` CLI with file-driven pause, resume and cancel.

C1 adds the official-platform SDK adapter: `OfficialModel` calls only
`third_party_agent.model.generate(messages)`, `OfficialBackend` converts
official observations/actions/artifacts, and `OfficialAgent` provides the SDK's
`reset` / `step` / `close` entry. C1 is offline-tested only; it does **not**
connect to the official platform, read a connection file or produce an official
score. Those checks remain for the main thread (C04-C08). No credential values
are read by the test suite. Acceptance records:
[A1.md](../../docs/06_分阶段实施/阶段一_双环境跑通/验收记录/A1.md),
[A2.md](../../docs/06_分阶段实施/阶段一_双环境跑通/验收记录/A2.md),
[B1.md](../../docs/06_分阶段实施/阶段一_双环境跑通/验收记录/B1.md),
[C1.md](../../docs/06_分阶段实施/阶段一_双环境跑通/验收记录/C1.md).

## Install and test

Project Python: `.venv/Scripts/python.exe` (Python 3.12.9). Run from the project
root:

```powershell
uv pip install --python .venv/Scripts/python.exe -e code/harness
uv pip install --python .venv/Scripts/python.exe -e "code/harness[real]"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest code\harness\tests -q
```

The base editable install pulls declared dependencies (EffGen 1.3.0, PyYAML,
tiktoken) through uv; it does not use `--no-deps` or import shielding. The
vendored EffGen editable install at `code/vendor/effgen` satisfies
`effgen==1.3.0`. The `real` extra pins `playwright==1.64.1`; B1 installed
Chromium build `chromium-1248`.

`browser-use==0.11.0` is declared as a separate `browser-use` extra, not part of
`real`. It pins `aiohttp==3.12.15`, while vendored EffGen requires
`aiohttp>=3.14.3`, so the two packages cannot be resolved into one environment
without violating a declared constraint. The B03 trial installed browser-use
into the project environment for a local-only run; see
`docs/06_分阶段实施/阶段一_双环境跑通/验收记录/B1.md`. Playwright remains the
default backend.

Upstream tests changed by A2 (the vendored suite carries `--strict-config` and a
pytest-asyncio option this environment does not install, so the run overrides
`addopts`; UTF-8 mode avoids the Windows GBK locale on UTF-8 sources):

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest code\vendor\effgen\tests\unit\test_interactive_loop.py -q -o addopts=""
.\.venv\Scripts\python.exe -m pytest code\vendor\effgen\tests\unit\test_one_loop.py code\vendor\effgen\tests\unit\test_tool_loop_guards.py code\vendor\effgen\tests\unit\test_loop_on_thread.py code\vendor\effgen\tests\unit\test_react_loop_efficiency.py -q -o addopts=""
```

Offline example runs (fake site/backend, no model call, no network):

```powershell
cd code\harness
..\..\.venv\Scripts\python.exe examples\offline_demo.py examples\sample_run
..\..\.venv\Scripts\python.exe examples\interactive_demo.py examples\sample_run_a2
```

## Layout

```text
code/harness/
├─ pyproject.toml
├─ src/webagent/
│  ├─ protocol.py        immutable Observation/ElementRef/WebAction/Feedback/Finish
│  ├─ session.py         single-task lifecycle, one pending action, pause/resume/close
│  ├─ tools.py           canonical tool schema, request→action binding, batch rejection
│  ├─ effgen_bridge.py   (A2) EffGen assembly, interactive loop driver, internal tools
│  ├─ budget.py          deadline, token counting, context packing
│  ├─ events.py          events.jsonl with mandatory fields and redaction
│  ├─ evidence.py        runtime/runs/<run_id>/ artifacts and integrity
│  ├─ redaction.py       credential/key/token filtering
│  ├─ config.py          strict YAML config, env names only
│  ├─ backend.py         backend Protocol used by A2/B/C
│  ├─ control.py         (B1) file-driven pause/resume/cancel
│  ├─ cli.py             (B1) doctor and run commands
│  ├─ __main__.py        (B1) python -m webagent
│  ├─ clock.py           wall + monotonic clocks (fake clock for tests)
│  ├─ backends/
│  │  ├─ browser_js.py        shared rendered-DOM observation extractor
│  │  ├─ playwright.py        (B1) headed/headless Chromium backend
│  │  ├─ browser_use.py       (B1 trial) BrowserSession/Page/Element only
│  │  └─ official.py          (C1) official request/action/artifact converter
│  ├─ models/
│  │  ├─ fake.py         EffGen BaseModel scripted fake
│  │  ├─ real.py         (B1) OpenAI-compatible text model with call records
│  │  └─ official.py     (C1) platform model.generate(messages) adapter
│  ├─ official_agent.py  (C1) SDK reset/step/close entry
│  └─ testing/fake_backend.py  deterministic offline environment
├─ configs/              official.yaml, real.yaml, fake.yaml, example.env
├─ tests/                offline pytest suite, seeded local site and task files
│  └─ fixtures/site/     search, results, pagination, detail preview
└─ examples/             offline demos + sample events/result
```

## B1: real model, browser and CLI

### Doctor

`doctor` loads and validates the YAML, checks imports, the Chromium binary hint,
the runtime directory, and whether the configured credential environment names
are present. It prints names, never values; it does **not** call a model, launch
Chromium, or start an official task.

```powershell
.\.venv\Scripts\python.exe -m webagent doctor --config code\harness\configs\real.yaml
```

### Run one external task

```powershell
.\.venv\Scripts\python.exe -m webagent run `
  --config code\harness\configs\real.yaml `
  --backend playwright `
  --task-file code\harness\tests\fixtures\tasks\local_chain.json
```

`--backend` is `playwright` (default recommendation) or `browser-use`.
Playwright starts headed by default; add `--headless` for offline tests.
`--runtime-dir`, `--profile-dir`, `--run-id`, `--max-seconds`,
`--action-timeout-ms`, `--navigation-timeout-ms` and `--pause-timeout` are
available. The real model is built from the config's environment variable names:
`WEBAGENT_MODEL_BASE_URL` and `WEBAGENT_MODEL_API_KEY` in `real.yaml`. No value is
ever written to a log or displayed by `doctor`.

For a purely offline integration run, `--model-script` replaces the real model
with a deterministic EffGen `ScriptedModel`; this is not real-model acceptance:

```powershell
.\.venv\Scripts\python.exe -m webagent run `
  --config code\harness\configs\real.yaml `
  --backend playwright --headless `
  --model-script code\harness\tests\fixtures\fake_scripts\local_chain.json `
  --task-file code\harness\tests\fixtures\tasks\local_chain.json `
  --runtime-dir runtime\b1_local
```

### Task file

The executor contains no task-to-answer or task-to-action map. A JSON task file
passes only task conditions:

```json
{
  "task": "Search for 'quantum', open the first result, type exactly 'Draft for quantum note' into the draft box, preview it, and screenshot the confirmation. Do not submit.",
  "start_url": "http://127.0.0.1:8765/index.html?seed=alpha-17",
  "draft_text": "Draft for quantum note",
  "seed": "alpha-17",
  "headless": true,
  "max_seconds": 120,
  "run_id": "b1-local-chain",
  "description": "optional local note"
}
```

`task` is required. `start_url` may also be supplied with `--start-url`.
`draft_text` is appended as an exact-value condition (or substituted for
`{draft_text}`); it never selects an element or supplies an action. Unknown keys
are rejected.

### Local fixture

`tests/fixtures/site` contains a static seeded catalog with a search box,
results, pagination and a detail page with a draft textarea and preview button.
It has no submit control. Serve it locally:

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 `
  --directory code\harness\tests\fixtures\site
```

The checked-in task uses `seed=alpha-17` and that port. Tests start their own
ephemeral server, so parallel runs do not need a fixed port.

### Pause, resume, cancel

Every run creates `runtime/runs/<run_id>/control.jsonl`. Append one command per
line (JSON `{"command": "pause"}` is also accepted):

```text
pause
resume
cancel
```

The CLI reads commands between actions. `pause` is accepted only when no action
is in flight; it drops the unconsumed proposal and prints that the user may take
over. `resume` calls `backend.observe()` first and installs the fresh
observation; old element numbers and old action bindings are invalid, and the
model receives an explicit "NOT EXECUTED" page result instead of a fabricated
one. `cancel` records the cancellation, closes the generator, browser and
session. Ctrl+C maps to cancel. The pause wait has `--pause-timeout`.

### Evidence

Each observation captures a viewport PNG through the session evidence store.
The `Artifact` metadata and the `artifact_saved` event carry the same
`url`, `page_id` and `observation_id`; run directories contain
`events.jsonl`, `observations/`, `screenshots/`, `result.json`, and (for a real
model) `model_calls.json`. `close` releases only this backend's persistent
context and Playwright driver; the dedicated profile remains under
`runtime/browser_profiles/<run_id>/` and is never a daily browser profile.

### Real model adapter

`models/real.py` implements EffGen's `BaseModel` with a text-only
`/chat/completions` request. Every HTTP attempt, including retries, is recorded
with request id, latency, HTTP status, usage source and failure type. Missing
provider usage is `null`; local estimates are separate fields. Retryable
statuses are 408/409/425/429/5xx. The adapter sends no tools, tool choice,
vision or model-probing request, and forces `trust_env=False` so a host proxy
setting cannot silently add a proxy. `model_calls.json` and `model_call` events
are redacted.

### browser-use trial

`backends/browser_use.py` implements the same `Backend` interface through
`browser_use.Browser` (BrowserSession), `Page`, `Element` and `Mouse` only. It
never constructs `browser_use.Agent`, and the test monkeypatches that symbol to
fail if constructed. It runs the same seeded task chain and passes headless with
a scripted model.

Trial constraints are explicit:

- browser-use 0.11.0 and EffGen 1.3.0 cannot be resolved together because of
  `aiohttp` pins; the installed project environment is therefore not a clean
  joint resolution.
- default browser-use extensions are disabled because that build downloads them
  at startup.
- browser-use 0.11.0's `Browser(timeout=...)` argument unexpectedly selects the
  cloud browser; the trial omits it. A local `httpx` proxy-parsing workaround is
  scoped to browser creation/start.

Because of the dependency conflict, extra startup workaround and unneeded
planning agent, **Playwright is the B1 default backend**. browser-use remains a
trial backend pending a user decision.



## C1: official platform adapter

### Offline verification

The C1 suite uses only fake SDK gateways/callbacks and local evidence
directories:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest `
  code\harness\tests\test_official_model.py `
  code\harness\tests\test_official_backend.py `
  code\harness\tests\test_official_agent.py -q
```

The model test imports `third_party_agent` from the frozen
`code/official_client` directory through `sys.path`, exactly as A1 did. No
connection file, platform account, real model service or network is used.

### Official connection (main thread only)

Run this from `code/official_client` so `third_party_agent` is importable; the
project Python must have the harness installed:

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\code\official_client
..\..\.venv\Scripts\python.exe -m third_party_agent connect `
  --connection-file "<连接文件绝对路径>" `
  --agent webagent.official_agent:OfficialAgent
```

`OfficialAgent` needs no arguments. It loads `configs/official.yaml`, writes
local runs under the project `runtime/` directory, and exposes the official run
id as `platform_run_id` while using a path-safe `local_run_id` for evidence.
Never put the connection file or its token in a log, prompt or repository.

### Agent contract

- `reset` creates an empty task container. It ignores any task/observation in the
  reset request and closes a previous task.
- The first `step` reads the complete task, current observation and
  `execution_deadline`, then creates one Session, OfficialBackend, OfficialModel
  and EffGen driver.
- Later steps create a new observation from the current request, bind it to the
  previous action as `observed` feedback, and send it into the same generator.
  Exactly one action string is returned.
- A terminal generator state becomes `stop [说明]`. `AgentResponse.success` is a
  local completion claim, not an official score.
- `close` is idempotent and releases only this task's generator, evidence and
  SDK-facing state.

### Action and observation rules

Commands follow the frozen README exactly: `click [id]`, `type [id] [text] [0|1]`,
`press [keys]`, `scroll [down|up]`, `goto [url]`, `go_back`, `go_forward`,
`stop [说明]`. Free text escapes `\\`, `[`, `]`, newline, carriage return and tab
as `\\`, `\[`, `\]`, `\n`, `\r`, `\t`; unsupported control characters and lone
surrogates are refused. This keeps one command on one structural line. `goto`
accepts only an exact, same-origin URL already observed in this task; it never
opens a URL locally.

AX numbers come only from `observation.text`. DOM `nodeId`/`backendNodeId` are
never action ids. DOM is read directly from `observation.dom.root` and saved by
the harness. Screenshot metadata is downloaded and verified on each observation
by default; HTML is downloaded on demand (for example `evidence_save`) through
the frozen browser SDK during the same callback. Both are checked for byte size
and SHA-256; failures appear as an unavailable artifact plus an `artifact_error`
event, never as a reused previous frame.

Model messages contain only text `role`/`content`. The complete EffGen prompt is
one user message; an explicitly supplied system prompt stays a system message.
No model name, temperature, output limit, tools, images or tool role is passed to
`third_party_agent.model.generate`. Provider usage is unavailable and therefore
`null`; local estimates are separate fields, and every actual call records its
platform `request_id` and increments the model-call count.

### C batch main-thread operation

1. Confirm the same connection token has no old client running and the team has
   no in-flight evaluation task.
2. Start the connection command above from `code/official_client`; wait for
   `Agent connected` and for the platform to show the Agent online.
3. Select one currently available task (or the platform's batch, if required)
   and execute it in order; do not start a second task in parallel.
4. Watch the platform progress and `runtime/runs/<local_run_id>/events.jsonl`.
   A local `stop` is not an official pass; record score, `finish_reason` and the
   platform task/run/batch ids separately.
5. Download the score JSON/replay and map it to `platform_run_id`; preserve the
   local result and any verified screenshots/HTML/DOM needed for audit.
6. Use `Ctrl+C`/close after the batch; do not reuse the token in a second client.

C04-C08 (actual connection, valid score, audit package, clean delivery, GitHub)
remain main-thread items. C1 only supplies and offline-verifies the adapter.

## A2: the interactive loop


The vendored entry is `Agent.run_interactive(task, *, external_tools=...)`. It is
a generator: `next()` yields one `InteractiveToolRequest`, and
`generator.send(result_text)` records that call's result and continues from the
same conversation. Its return value is the same `AgentResponse` that `run()`
returns. Ordinary `Agent.run()` and in-process tools are unchanged.

The harness side is `webagent.effgen_bridge`:

```python
from webagent.effgen_bridge import LoopSettings, create_driver

driver = create_driver(session, model, settings=LoopSettings())
step = driver.start()                       # ACTION, COMPLETED, DEADLINE, ...
while step.needs_action:
    action = driver.take_action()           # session.consume(); one call only
    result = backend.execute(action)        # the environment executes exactly once
    step = driver.accept_observation(result.observation)
```

`BackendLoopRunner(driver, backend).run()` is the same loop spelled once. A later
official/real entry point uses `take_action` / `accept_observation` from its own
`step()` callbacks.

Rules the bridge enforces:

- `start()` appends the canonical current observation to the task string it
  hands EffGen, exactly once. The model can therefore choose a valid initial
  element number; later turns receive the real observation returned for each
  action.
- Every request goes through `guard_requests` against `session.observation`
  before it can become an action. A rejected request is **not** dispatched; its
  `ActionRejected.correction` is sent back into the generator as the call's
  result, so the model sees the current observation id and valid numbers. At
  most `max_format_corrections` (default 2) corrections are appended; the next
  invalid output ends the run with `FORMAT_ERROR` and no action.
- A model output that carries a native batch of page actions is refused as a
  batch: the caller's `batch_correction` (which calls the same guard) is put in
  the conversation, nothing is queued, and no page tool runs in-process.
- `take_action()` consumes the proposal once; a second call raises
  `BridgeError`. `accept_observation()` requires the session's binding rules
  (feedback for exactly that action, a new observation id).
- The EffGen agent is built once per task and kept across every send; the loop
  never rebuilds it. The run's scope is installed per advance and removed before
  the request reaches the caller, so an SDK may advance the generator from a
  later callback without inheriting the previous binding.
- Default `max_iterations=None` means "no framework step cap": the session
  deadline ends the run (`NO_ITERATION_CAP` is used internally). Passing an int
  still caps the run. `max_turns_without_progress=None` keeps the framework from
  ending a page walk because turns looked repetitive.
- Page tools are excluded from the framework's record replay, drift threshold,
  repeated-result stop, input-error withdrawal and "you already have results"
  nudge. Two identical `scroll` calls bound to two observations are two
  requests and two executions.
- Internal tools (`web_observe`, `task_note`, `evidence_save`) run through
  EffGen's ordinary tool path. They answer synchronously (no event loop or
  worker thread), `web_observe` returns the latest recorded observation without
  re-observing, `task_note` stores a session note, and `evidence_save` copies an
  artifact this observation already captured and verifies its hash.
- `before_step` and `before_model_call` call `session.check_deadline` and record
  the model call. Expiry stops the run before the next turn; a run closed while a
  request is pending records no result for it.

`format_correction` is an event type added by A2. Its `result` names the reason
(`unknown_element`, `multiple_actions`, ...), the scope (`action` or `batch`)
and the correction text. `format_correction` never carries a page result.

## A1 protocol (A2 uses it unchanged)

The interactive EffGen entry must use this sequence. It must not invent a
second tool schema or a second action executor.

```python
from webagent.session import Session
from webagent.tools import guard_requests
from webagent.protocol import Finish, ToolRequest

session.start_task(task_text)
session.ingest_observation(backend.observe())      # initial observation

# On each model tool call (all web calls are rejected as a batch if >1):
guarded = guard_requests(requests, session.observation)
if guarded.action is None:
    ...  # internal tools (web_observe/task_note/evidence_save) execute directly
else:
    session.propose(guarded.action)                 # logs proposed; no execution
    action = session.consume()                      # logs dispatched; one-shot
    result = backend.execute(action)                # environment executes exactly once
    session.ingest_observation(result.observation)  # logs feedback; validates binding
```

Rules the contract enforces:

- `session.observation` is the only valid binding source. `guard_requests` and
  `Session.propose` reject an action from an older observation, another page,
  another run, or an element number absent from the current table. The full
  `(run_id, page_id, observation_id, number)` reference matters, not whether the
  number appeared earlier.
- More than one web action in one model output raises `ActionRejected` with
  `reason=multiple_actions`. No action from that batch is queued. Send
  `ActionRejected.correction` back to the model as the format correction; it
  names the current observation and valid numbers.
- A proposed action can be consumed once. A second `consume` (or a repeated
  request id) raises `duplicate_consumption`. `dispatched` is not a result.
- After `consume`, the next `ingest_observation` must carry feedback bound to
  that exact action. If the environment has no execution evidence, use
  `FeedbackStatus.UNKNOWN`, not a fabricated success.
- Two `scroll` calls with identical arguments are still two actions when each
  is bound to a new observation; do not cache the first result.
- `web_observe` returns the latest real observation; it must not re-observe in
  the background. Internal tools never produce `WebAction`s.
- The generator may use `session.record_model_call(...)` around every model
  call; it checks the deadline/model budget and writes the model event.
- Call `session.check_deadline(stage)` before model calls, downloads and action
  work. Expiry logs one `deadline` event and raises `DeadlineExceeded`.
- `session.finish(Finish(...))` records the agent's claim and reason; it never
  invents an official score. `session.close()` is idempotent and releases only
  the session's own logger/evidence handles.

Pause/resume for the minimum takeover path:

```python
session.pause()                 # only with no in-flight action; drops an unconsumed proposal
new_observation = backend.observe()   # user may have changed the page
session.resume(new_observation)       # fresh observation id, old table/actions invalid
```

`pause` refuses while an action is awaiting feedback. `resume` requires a new
observation and rejects one carrying feedback. The old element table and old
`WebAction`s are stale after resume.

The backend interface (`webagent.backend.Backend`) is:

```python
backend.observe() -> Observation
backend.execute(action) -> ExecutionResult   # feedback + fresh observation
backend.close() -> None
```

`webagent.testing.FakeBackend` implements it for offline A2 tests.

## Budget and context

`BudgetLimits` defaults: 24K input tokens, 32K planned context, 256 messages,
1 MiB serialized request. There is no default 10-action/iteration cap: A2 uses
the task deadline, and `max_model_calls` is optional and null by default.

`ContextPacker.pack(task=..., observation=..., pending=..., history=...)` keeps
the task, current AX text and pending confirmation intact. It compresses old
history (recording `dropped_history_entries`) and raises `ContextOverflow` with
required/available counts when mandatory content cannot fit. It never truncates
the current observation silently. `token_count_source` states whether a count is
exact or estimated; a verified target-model tokenizer is not available in A1.

Official request limits are checked on the serialized message list
(`max_messages`, `max_request_bytes`) and raise `RequestLimitExceeded`.

## Events and evidence

Run directory: `runtime/runs/<run_id>/` with `events.jsonl`, `result.json`,
`observations/`, and optional `screenshots/`, `html/`, `dom/`. Run ids are
validated as path-safe; references are relative and cannot escape the run
directory.

Every event line contains:

```text
schema_version, event_id, run_id, time, elapsed_ms, event_type,
page_id, url, observation_id, action_id, action_type, result,
model_calls_total, model_request_id
```

Unused fields are explicit `null`. `proposed` and `dispatched` always have
`result=null`; only `feedback` carries `observed|error|unknown`. `agent_claim`
carries the claim/reason and never a score. `model_call` adds input/output token
counts, `token_count_source`, latency and failure type. Terminal states are
`cancelled`, `deadline`, `error`.

All event strings pass through `Redactor`: registered fake secrets, `sk-...`
keys, `Bearer ...`, `password=...`, `token=...`, API-key fields and private-key
blocks are replaced with `[REDACTED]`. The same redactor is applied to
`result.json` and exception messages. Credentials are never read by A1.

Evidence saves compute SHA-256 and byte size from the actual bytes and verify
the read-back; a failed save raises instead of reusing an older artifact.

## Configuration

`configs/*.yaml` contain no credential values, only uppercase environment
variable names for the real model. Credential-looking keys or values are
rejected at load time. `configs/example.env` contains fake placeholders only.

- `official.yaml`: profile official, backend official, no endpoint/key.
- `real.yaml`: Playwright profile, `WEBAGENT_MODEL_BASE_URL` /
  `WEBAGENT_MODEL_API_KEY` names only.
- `fake.yaml`: offline fake script/site paths.

Disabled-by-requirement settings are explicit: SINGLE agent, ReAct text tool
mode, no capability probe, sub-agents, fallback, speculative execution or
cross-task memory. `max_iterations: null` is the default; an explicit positive
value is accepted by validation.

## Not in B1

- The official SDK adapter and platform connection (C1).
- Real-model, public-web or official-platform acceptance. The main thread runs
  those; this directory's tests are offline.
- Automatic login/verification recovery and a complete takeover state machine
  (phase 2). B1 provides only the minimum pause/resume/cancel path.
- A clean joint dependency resolution for browser-use and EffGen (see above).
- Prompt token-budget enforcement inside EffGen: A1/A2 keep the packer on the
  harness side; B1 does not silently truncate the current observation.
