# Fuzz Harness

effGen ships a hypothesis-driven fuzz harness under `tests/fuzz/` that exercises
three critical subsystems with at least 500 randomly generated examples per test.

## Running

```bash
# All fuzz tests (≥500 examples each):
pytest -m fuzz

# Single module:
pytest tests/fuzz/test_tool_fuzz.py -m fuzz
pytest tests/fuzz/test_message_fuzz.py -m fuzz
pytest tests/fuzz/test_router_fuzz.py -m fuzz

# Increase example count (CI regression coverage):
pytest -m fuzz --hypothesis-seed=0 -x
```

The `fuzz` marker is declared in `pyproject.toml`. Tests run without any live
API calls or network access.

---

## Modules

### `test_tool_fuzz.py` — Tool input fuzzing

For every `BaseTool` subclass, hypothesis generates random inputs that conform
to (or deliberately violate) the declared parameter schema and asserts:

| Assertion | Detail |
|-----------|--------|
| No unhandled exception | `execute()` always returns a `ToolResult` |
| `success=False` → `error` is non-empty | Failure information is always propagated |
| No secret in error message | All eight built-in secret patterns checked |
| Injected secrets never leak | Synthetic secrets placed in inputs never appear in errors |
| `ParameterSpec.validate()` returns `(bool, str\|None)` | Never raises |
| `_coerce_parameters` never crashes | Arbitrary string inputs are safe |

**Every built-in tool is covered.** `test_every_tool_execute_fuzz` discovers
all `BaseTool` subclasses via `ToolRegistry.discover_builtin_tools()` and
fuzzes each one's `execute()` entry point with a mix of schema-valid and
hostile inputs. Tools that hit the network or a system binary have their
`_execute` patched, so the validation / coercion / error-handling envelope is
exercised without any live calls. A `test_tool_discovery_nonempty` guard fails
if discovery silently shrinks below the expected tool count.

In addition, hand-picked tests cover `Calculator`, `DateTimeTool`,
`TextProcessingTool`, `JSONTool`, `WikipediaTool` (mocked network), a generic
schema fuzzer over all six `ParameterType` variants, parameter coercion, and
`ParameterSpec.validate`.

**Secret-injection fuzz.** `test_secret_injection_never_leaks` injects
synthetic credentials (one per redactor pattern) into every tool's inputs and
forces the underlying `_execute` to raise an exception that echoes its
arguments — the worst case for a leak. `BaseTool.execute()` scrubs its error
string through the redactor, so the raw secret never survives into
`ToolResult.error`.

### `test_message_fuzz.py` — Message schema fuzzing

Generates random `ContentPart` variants (`TextPart`, `ImagePart`, `AudioPart`,
`VideoPart`, `ToolCallPart`, `ToolResultPart`) and `Message` objects with random
roles, content lists, and metadata. Asserts:

| Assertion | Detail |
|-----------|--------|
| Valid construction never raises | All six `ContentPart` types |
| Invalid MIME / negative duration / empty ID raises only `InvalidMultimodalContent` | No `ValueError`, `KeyError`, etc. escape |
| `Message.text` / `.has_image` / `.has_audio` / `.has_video` never crash | Valid messages |
| `to_dict()` never crashes | Serialisation is safe |
| Factory methods (`.user`, `.assistant`, `.system`) accept any string | No crash |
| No secret in any exception message | All eight patterns checked |

### `test_router_fuzz.py` — Router fuzzing

Generates random `ProviderModelPair` candidates, `RoutingContext` constraint
combinations, and `model_info` dicts and asserts:

| Assertion | Detail |
|-----------|--------|
| `route()` returns `RouterDecision` or raises `NoCandidateError` | Never `None` or silent fail |
| `RouterDecision.chosen` is always a valid `ProviderModelPair` | `.provider` and `.model_id` non-empty |
| `estimate_complexity()` returns `0.0 ≤ score ≤ 1.0` | Any string input |
| `candidate_unavailable_reason()` returns `str\|None` | Never raises |
| `_model_capability_reason()` returns `str\|None` | Any model_info dict |
| Negative `failover_hops` raises `ValueError` | Constructor guard |
| `ProviderModelPair` is hashable | Used in sets for blocklist logic |

---

## Configuration

The harness uses hypothesis's `@settings(max_examples=500)` decorator. The
hypothesis database is stored under `.hypothesis/` (git-ignored). To reproduce
a specific failure:

```bash
pytest tests/fuzz/test_tool_fuzz.py::test_calculator_fuzz \
    --hypothesis-seed=<printed-seed>
```

To disable the health-check for slow tests (large state machines):

```python
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
```

---

## Secret Leak Detection

Every error message produced during a fuzz run is scanned against the same
eight regex patterns used by `effgen.observability.redact`:

- `openai_key` — `sk-[a-zA-Z0-9_-]{20,}`
- `anthropic_key` — `sk-ant-[a-zA-Z0-9_-]{20,}`
- `cerebras_key` — `csk-[a-zA-Z0-9_-]{20,}`
- `google_key` — `AIza[0-9A-Za-z_-]{35}`
- `hf_key` — `hf_[a-zA-Z0-9]{20,}`
- `groq_key` — `gsk_[a-zA-Z0-9_-]{20,}`
- `bearer_token` — `Bearer [^\s]{6,}`
- (Slack / Discord webhook URLs)

A match causes the test to fail with a clear message identifying the tool or
component that leaked the secret.

---

## Adding New Tools

Any new `BaseTool` subclass that registers via
`ToolRegistry.discover_builtin_tools()` is **automatically** fuzzed by
`test_every_tool_execute_fuzz` and `test_secret_injection_never_leaks` — no
test changes are required. The harness patches `_execute`, so tools that hit
the network or a system binary are covered without live calls.

To add a *dedicated, deeper* test for a specific tool (e.g. exercising a
particular operation branch):

1. Import the class in `tests/fuzz/test_tool_fuzz.py`.
2. If the tool makes live API calls, mock `_execute` with `AsyncMock`.
3. Write a `@given(...)` test using `_strategy_for_spec` to generate parameter
   values from the tool's `ParameterSpec` list, or use the generic
   `_kwargs_strategy(tool.metadata.parameters)` helper.
4. Call `_assert_result_ok(result, tool.name)` on the returned `ToolResult`.
