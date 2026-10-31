# Evaluation & Benchmarking

effGen v0.2.0 includes a built-in evaluation framework for benchmarking agents, tracking regressions, and comparing models.

## Quick Start

```bash
# Run the math evaluation suite
effgen eval --suite math --model "Qwen/Qwen2.5-3B-Instruct"

# Compare multiple models
effgen compare --models "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B" --suite math
```

## Built-in Test Suites

| Suite | Cases | Description |
|-------|-------|-------------|
| `math` | 77 | Arithmetic, algebra, word problems (easy/medium/hard) |
| `tool_use` | 93 | Tool selection, parameter extraction, multi-step tool chains |
| `reasoning` | 40 | Logic, deduction, common sense |
| `safety` | 40 | Prompt injection resistance, PII handling, refusal |
| `conversation` | 20 | Multi-turn context, memory recall |

## Python API

```python
from effgen.eval import AgentEvaluator, MathSuite, ToolUseSuite
from effgen.eval.evaluator import ScoringMode

evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)

# Run a single suite
results = evaluator.run_suite(MathSuite())
print(results.summary())
# Accuracy: 72.7% (56/77)
# Avg latency: 13.9s
# By difficulty: easy 100%, medium 69.4%, hard 54.5%

# Run with threshold
results = evaluator.run_suite(MathSuite(), threshold=0.5)
```

### Scoring Modes

| Mode | Description |
|------|-------------|
| `EXACT_MATCH` | Output must exactly equal expected |
| `CONTAINS` | Expected value must appear in output |
| `REGEX` | Output must match regex pattern |
| `SEMANTIC_SIMILARITY` | Cosine similarity (requires sentence-transformers) |
| `LLM_JUDGE` | Uses agent's own model to judge correctness |

## Regression Tracking

```python
from effgen.eval import RegressionTracker

tracker = RegressionTracker()

# Save a baseline
tracker.save_baseline("math", results, version="0.2.0")

# Later, compare against baseline
new_results = evaluator.run_suite(MathSuite())
report = tracker.compare("math", new_results, version="0.2.1")

print(report.to_markdown())
# Shows: accuracy delta, latency delta, regressions by severity
# Thresholds: >5% accuracy drop = critical, >20% latency increase = warning
```

Baselines are stored in `tests/benchmarks/eval_baseline_<suite>.json`.

## Model Comparison

```python
from effgen.eval import ModelComparison

comparison = ModelComparison()
matrix = comparison.run(
    agents={"small": small_agent, "medium": medium_agent, "large": large_agent},
    suites=["math", "tool_use", "reasoning"],
)

print(matrix.to_markdown())
# Accuracy table + Latency table + Recommendations (best model per suite)

matrix.to_json("comparison.json")
```

## CLI Options

```bash
# Evaluation
effgen eval --suite math --scoring contains --threshold 0.5 -o results.json
effgen eval --suite tool_use --difficulty hard
effgen eval --suite math --save-baseline    # Save as regression baseline
effgen eval --suite math --compare-baseline # Compare against saved baseline

# Comparison
effgen compare --models "model_a,model_b" --suite math,reasoning -o comparison.md
```

### Shareable reports

`-o` picks its format from the extension (`.html` renders a report, `.md`
writes Markdown, anything else JSON), and `--report out.html` writes the
shareable report alongside the usual output:

```bash
effgen eval --suite math -m gpt-5-nano --provider openai --report eval.html
effgen compare --models "model_a,model_b" --suite math --report bakeoff.html

# Or render a result captured earlier, without re-running the models
effgen eval --suite math --json > eval.json && effgen report eval.json
```

The HTML file is self-contained and opens offline. Details in
[CLI developer-experience surfaces](../dx/cli.md).

## CI/CD Integration

The nightly CI workflow automatically:
1. Runs the math eval suite against stored baselines
2. Compares accuracy and latency
3. Opens a GitHub issue with `[regression, automated]` labels if thresholds are exceeded

See `.github/workflows/nightly.yml` for the eval-regression job configuration.
