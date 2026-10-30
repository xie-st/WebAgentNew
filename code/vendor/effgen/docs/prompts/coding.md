# Coding Prompt Templates

The `coding` domain provides five prompt templates for common software-engineering tasks: code review, bug diagnosis, refactoring planning, test generation, and docstring writing.

---

## Templates

### `coding.code_review.v1` — Structured Code Review

**Variant:** `structured`  
**Tags:** `coding`, `review`, `structured`, `json`, `security`

Reviews code and returns a JSON object with a list of issues.

**Inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | yes | Source code to review |
| `language` | string | yes | Programming language (e.g. `"Python"`) |
| `focus` | string | no | `"all"` \| `"security"` \| `"performance"` \| `"style"` (default `"all"`) |

**Output schema:**

```json
{
  "issues": [
    {
      "severity": "critical|high|medium|low|info",
      "location": "function name or line reference",
      "suggestion": "actionable fix description"
    }
  ]
}
```

**Example:**

```python
from effgen.prompts.library.registry import registry

prompt = registry.get("coding.code_review.v1")
rendered = prompt.render(
    code="def get_user(uid):\n    return f'SELECT * FROM users WHERE id={uid}'",
    language="Python",
    focus="security",
)
print(rendered)
```

---

### `coding.bug_diagnose.v1` — Chain-of-Thought Bug Diagnosis

**Variant:** `cot`  
**Tags:** `coding`, `debugging`, `cot`, `bug`

Walks through a 6-step reasoning process to identify the root cause and propose a minimal fix.

**Inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | yes | Code containing the bug |
| `error_message` | string | yes | Full error message or traceback |
| `repro_steps` | string | no | Steps to reproduce the error |

**Example:**

```python
prompt = registry.get("coding.bug_diagnose.v1")
rendered = prompt.render(
    code="def fibonacci(n):\n    memo = {0: 0, 1: 1}\n    for i in range(2, n):\n        memo[i] = memo[i-1] + memo[i-2]\n    return memo[n]",
    error_message="KeyError: 2 when calling fibonacci(2)",
    repro_steps="Call fibonacci(2)",
)
```

---

### `coding.refactor_plan.v1` — Tool-Augmented Refactoring Plan

**Variant:** `tool`  
**Tags:** `coding`, `refactor`, `tool`, `plan`

Instructs the model (or an agent) to read the source file with effGen's `file_operations` tool and produce a five-section refactoring plan including risk assessment and test strategy.

**Inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | yes | Path to the file to refactor |
| `goals` | string | yes | Free-text description of refactoring goals |
| `language` | string | no | Programming language (default `"Python"`) |
| `code_snippet` | string | no | Pre-fetched code (skips file-read tool call if provided) |

**Output sections:**

1. Summary of current issues
2. Proposed changes
3. Code sketch (before/after snippets)
4. Risk assessment
5. Suggested test plan

---

### `coding.test_generate.v1` — Few-Shot Test Generation

**Variant:** `few_shot`  
**Tags:** `coding`, `testing`, `few_shot`, `pytest`, `ast`

Generates a complete pytest (or unittest) test suite using two exemplar suites to guide output style. Live evaluation asserts that `ast.parse()` succeeds on the generated Python.

**Inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | yes | Source function(s) to test |
| `framework` | string | yes | `"pytest"` \| `"unittest"` |
| `function_name` | string | no | Focus on a specific function (optional) |

**Live eval assertion:** `ast.parse(output)` must not raise `SyntaxError`.

**Example:**

```python
prompt = registry.get("coding.test_generate.v1")
rendered = prompt.render(
    code="def clamp(value, lo, hi):\n    return max(lo, min(value, hi))",
    framework="pytest",
    function_name="clamp",
)
```

---

### `coding.docstring_fill.v1` — Zero-Shot Docstring Generator

**Variant:** `zero_shot`  
**Tags:** `coding`, `docstring`, `zero_shot`, `documentation`

Adds docstrings to all undocumented functions and classes. Supports Google, NumPy, and Sphinx formats.

**Inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | yes | Python source code |
| `style` | string | no | `"google"` \| `"numpy"` \| `"sphinx"` (default `"google"`) |

**Example:**

```python
prompt = registry.get("coding.docstring_fill.v1")
rendered = prompt.render(
    code="def merge_sorted(a, b):\n    ...",
    style="numpy",
)
```

---

## Running Evaluations

**Golden eval (no API required):**

```bash
effgen prompts eval --domain coding
```

**Live eval (requires `CEREBRAS_API_KEY`):**

```bash
effgen prompts eval --domain coding --live --model gpt-oss-120b
```

For providers with tight request-per-minute limits, increase or decrease the spacing between live calls:

```bash
effgen prompts eval --domain coding --live --model gpt-oss-120b --delay 35
```

**Key live assertions:**

- `coding.test_generate.v1`: `ast.parse()` passes on generated Python
- `coding.code_review.v1`: output parses as JSON matching `{issues: [{severity, location, suggestion}]}`

---

## CLI Quick Reference

```bash
# List all coding prompts
effgen prompts list --domain coding

# Show a prompt
effgen prompts show coding.code_review.v1

# Show metadata and the fixture rendering
effgen prompts show coding.test_generate.v1
```
