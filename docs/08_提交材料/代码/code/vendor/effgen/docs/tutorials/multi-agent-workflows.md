# Multi-Agent Workflows

effGen provides multi-agent orchestration with a message bus, DAG-based
workflows, shared state, and agent lifecycle management. The main entry points
are importable from the top level:

```python
from effgen import (
    MultiAgentOrchestrator, OrchestrationPattern, TeamConfig,
    WorkflowDAG, WorkflowNode, SubAgentRouter,
)
```

## Teams — MultiAgentOrchestrator

Coordinate several agents with a chosen pattern (sequential, parallel,
hierarchical, collaborative, competitive, pipeline):

```python
from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern, load_model

m = load_model("gpt-5-nano")
writer = Agent(AgentConfig(name="writer", model=m))
editor = Agent(AgentConfig(name="editor", model=m))

orch = MultiAgentOrchestrator()
orch.create_team("blog", [writer, editor], pattern=OrchestrationPattern.SEQUENTIAL)

# Pass the team object or just its name:
result = orch.assign_task("Write one sentence about the ocean.", "blog")
print(result.success, result.output)
```

`assign_task` returns a `TeamResponse` whose `success` is `False` (never a
silent success) if the team is empty or any agent fails; per-agent outputs and
errors are kept in `result.agent_responses`. This holds for **every** pattern:
each member's `success`/`error` is recorded, and one failing member makes the
team `success=False` with a `reason` and a redacted `error` in
`result.metadata` — no pattern hides a failed agent behind a `True`.

### Choosing a pattern

| Pattern | What it does |
|---|---|
| `SEQUENTIAL` | Run agents in order; each output feeds the next. Stops at the first failure. |
| `PIPELINE` | Currently identical to `SEQUENTIAL` (give each stage a role-specific `system_prompt`). |
| `PARALLEL` | Run every agent on the same task at once, then concatenate. |
| `HIERARCHICAL` | A **manager** splits the task into subtasks and routes each to the **named** worker (see below). |
| `COLLABORATIVE` | Agents discuss over rounds until they converge. |
| `COMPETITIVE` | Every agent solves the task; the best answer is selected. |

### Cost & token tab

For teams **and** workflows, the summed spend is folded onto the result so you
never have to re-sum by hand:

```python
result = orch.assign_task("…", team)
print(result.metadata["cost_usd"], result.metadata["tokens_used"])
```

A member on a local engine, or on a model the catalog publishes no rate for,
contributes no cost; when no member reported one, `cost_usd` is `None` rather
than `0.0`.

### Routing a ticket to one specialist (triage → handoff)

A support flow usually wants **one** specialist to answer, not every agent in
the team. `HIERARCHICAL` does exactly this: the manager labels each subtask with
the worker who should handle it (`"billing: …"`), and the subtask is dispatched
to the agent of that name — never by list position.

```python
from effgen import Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern, load_model

m = load_model("gpt-5-nano")
manager = Agent(AgentConfig(name="manager", model=m))
team = [
    Agent(AgentConfig(name="billing", model=m,
                      system_prompt="You handle refunds and billing only.")),
    Agent(AgentConfig(name="tech", model=m,
                      system_prompt="You handle login/app bugs only.")),
]

orch = MultiAgentOrchestrator()
orch.create_team("support", team, pattern=OrchestrationPattern.HIERARCHICAL,
                 manager_agent=manager)
result = orch.assign_task("I was charged twice for order ORD-7788.", "support")
# The 'billing' subtask is dispatched to the billing agent, not whoever is first.
for r in result.agent_responses:
    print(r["agent_name"], "<-", r["subtask"])
```

If you prefer to make the routing decision yourself, run a triage agent first
and pick the specialist from its answer:

```python
triage = Agent(AgentConfig(name="triage", model=m,
    system_prompt="Reply with exactly one word: 'billing' or 'tech'."))
choice = triage.run(ticket).output.strip().lower()
specialist = {"billing": team[0], "tech": team[1]}.get(choice, team[0])
answer = specialist.run(ticket)
```

## Seeing the shape of a team

`TeamConfig.to_dict()` serializes the team's topology — its members, their
models and tools, and the edges the pattern implies — before anything runs, so a
team can be reviewed or drawn without spending on a run:

```python
team = orch.create_team("desk", [researcher, writer],
                        pattern=OrchestrationPattern.HIERARCHICAL, manager_agent=lead)
shape = team.to_dict()
# {'name': 'desk', 'pattern': 'hierarchical', 'manager': {...},
#  'agents': [{'name': 'researcher', 'model': 'gpt-5-nano', 'tools': ['wikipedia'], ...}],
#  'edges': [{'source': 'lead', 'target': 'researcher', 'kind': 'delegation'}, ...]}

print(team.diagram())                       # the shape, in the terminal
response = orch.assign_task("Draft the brief.", team)
print(team.diagram(response))               # the same shape, annotated with the run
```

The same structure is on `response.metadata["topology"]`, next to
`response.metadata["execution_id"]`.

## Grouping one execution's runs

Each team or workflow run issues an execution id, and every member run carries
it along with the agent that delegated the work and the role it played. Spans
and stored run records both carry these fields, so N sub-agent traces regroup
into the one execution they came from:

```python
from effgen.observability import run_log

response = orch.assign_task("Draft the brief.", team)
members = run_log.read_runs(execution_id=response.metadata["execution_id"])
for row in members:
    print(row["agent"], row["role"], "←", row["parent_agent"], row["status"])

# Or read the whole execution as a graph:
from effgen.observability.topology import build_topology
graph = build_topology(limit=1)["executions"][0]
```

The dashboard's Agent Topology panel draws exactly this, and because the records
are durable it shows teams and workflows run from a script or the CLI, not only
work done inside the server process.

## Running a team from async code

`assign_task_async` awaits the same run and returns the same `TeamResponse`:

```python
response = await orch.assign_task_async("Draft the brief.", team)
```

Workflows pair the same way: `dag.run()` / `dag.execute()` synchronously,
`await dag.run_async()` / `await dag.execute_async()` from async code.

## MessageBus — Agent Communication

```python
from effgen.core.message_bus import MessageBus, AgentMessage, MessageType

bus = MessageBus()

# Subscribe to a topic pattern (supports * / ? wildcards)
bus.subscribe("results.*", lambda msg: print(f"Got result: {msg.payload}"))

# Publish a message to its topic
bus.publish(AgentMessage(
    sender="math_agent",
    recipient="coordinator",
    type=MessageType.RESULT,
    payload={"answer": 42},
    topic="results.math",
))

# Mailbox-based (direct agent-to-agent)
bus.send(AgentMessage(
    sender="coordinator",
    recipient="research_agent",
    type=MessageType.TASK_ASSIGNMENT,
    payload={"task": "Search for quantum computing papers"},
))
messages = bus.receive("research_agent")
```

## DAG-Based Workflows

Define complex workflows as directed acyclic graphs:

### Python API

```python
from effgen import WorkflowDAG, WorkflowNode

dag = WorkflowDAG("report_pipeline")
dag.add_node(WorkflowNode(id="research", agent=research_agent))
dag.add_node(WorkflowNode(id="summarize", agent=summary_agent))
dag.add_node(WorkflowNode(id="format", agent=format_agent))

dag.connect("research", "summarize")   # add an edge (validates against cycles)
dag.connect("summarize", "format")

# A single string is routed to the entry node(s); or pass {node_id: task}.
result = dag.run("Find recent papers on small language models and write a report.")
# Runs in topological order and parallelizes independent nodes.
print(result.success, result.outputs["format"])
```

### YAML Workflow Definitions

```yaml
# workflow.yaml
nodes:
  - id: fetch_data
    preset: research
    task: "Fetch latest data on {topic}"
  - id: analyze
    preset: coding
    task: "Analyze the data and compute statistics"
    depends_on: [fetch_data]
  - id: visualize
    preset: coding
    task: "Create charts from the analysis"
    depends_on: [analyze]
  - id: report
    preset: general
    task: "Write a summary report"
    depends_on: [analyze, visualize]
```

```bash
# CLI
effgen workflow validate workflow.yaml
effgen workflow run workflow.yaml
```

A node whose required upstream **failed** (or was itself skipped) is marked
`skipped` rather than run on the upstream's error text — so an internal failure
is never silently rewritten into a downstream (often customer-facing) answer.
The overall `result.success` stays `False` because the failed node is recorded.

### Conditional Branching

Edges can have conditions based on previous node outputs. The condition receives
the **source node's output** and the target is skipped when it returns `False`:

```python
dag.connect("classify", "handle_urgent", condition=lambda out: "urgent" in str(out))
dag.connect("classify", "handle_normal", condition=lambda out: "urgent" not in str(out))
```

## Shared State

Thread-safe key-value store shared across agents in a workflow:

```python
from effgen.core.shared_state import SharedState

state = SharedState()

# Namespaced access
state.set("research", "papers_found", 42)
state.set("analysis", "top_topic", "quantum computing")

count = state.get("research", "papers_found")  # 42

# Snapshots for rollback
snapshot = state.snapshot()
# ... do work ...
state.rollback(snapshot)  # Restore to the snapshot
```

## Agent Lifecycle Management

```python
from effgen.core.lifecycle import AgentRegistry, AgentPool

# Registry — track agents by id
registry = AgentRegistry()
registry.register("agent-1", agent, timeout=300)  # 5-minute timeout

# Pool — reuse pre-built agents for fast allocation
pool = AgentPool(max_size=10)
pool.add(create_agent("general", model))  # pre-warm
agent = pool.acquire()
try:
    result = agent.run("task")
finally:
    pool.release(agent)

# Timeout and cancellation
registry.check_timeouts()    # Cancel agents that exceeded their timeout
registry.cancel("agent-1")   # Cancel a specific agent
```

## CLI Commands

```bash
# Workflow operations
effgen workflow run pipeline.yaml --model "Qwen/Qwen2.5-3B-Instruct"
effgen workflow validate pipeline.yaml

# Batch execution
effgen batch --input queries.jsonl --output results.jsonl --concurrency 5 --preset research
```
