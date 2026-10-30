# Contributing to effGen

Thank you for your interest in contributing to effGen! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (for integration tests)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen

# Create a virtual environment
conda create -n effgen python=3.11 -y
conda activate effgen

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Unit tests (no GPU required)
pytest tests/unit/ -v --no-cov

# Integration tests (requires GPU)
CUDA_VISIBLE_DEVICES=0 pytest tests/integration/ -v -m gpu --no-cov

# Performance benchmarks
pytest tests/benchmarks/ -v --no-cov

# All tests with coverage
pytest tests/ -v
```

### Cleaning up artifacts

Test, build, and runtime runs leave gitignored artifacts behind (caches, coverage,
`./checkpoints`, `__pycache__`, etc.). Tidy the tree at any time with:

```bash
scripts/clean.sh            # remove caches, coverage, build artifacts, runtime state
scripts/clean.sh --dry-run  # preview what would be removed
```

It only deletes already-ignored artifacts — never tracked source, `.git/`, or `.env`.

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting (profile: black)
- **flake8** for linting
- **mypy** for type checking
- **bandit** for security linting

All these run automatically via pre-commit hooks. You can also run them manually:

```bash
black effgen/
isort effgen/
flake8 effgen/
mypy effgen/ --ignore-missing-imports
```

### Type hints and the static-check ratchet

effGen ships `effgen/py.typed`, so every user's type checker checks their code
against our annotations. Two rules follow from that:

1. **Every public signature is fully annotated.** A non-underscore `def` in a
   public module — dunders included, since the language calls them — annotates
   each parameter and its return type. An unannotated one silently becomes
   `Any` in a user's type check. `tests/unit/test_public_signature_hints.py`
   enforces it at zero tolerance, which is what stops a new module arriving
   with an untyped public surface: mypy is configured with
   `disallow_untyped_defs = false`, so it reports nothing at all for one.

2. **The recorded mypy result may improve, never regress.** This is the second,
   separate gate: it holds the errors mypy *does* report, so annotations that
   disagree with the code cannot accumulate. The configuration is permissive and
   the package does not type-check clean, so instead of a gate that would have
   to be switched off, the *recorded* result is gated:

   ```bash
   python scripts/mypy_ratchet.py            # gate: exit 1 if it got worse
   python scripts/mypy_ratchet.py --update   # re-record after an improvement
   ```

   `scripts/mypy_baseline.json` holds every error's identity — its file, error
   code and message, with the line and column dropped so moving code around does
   not read as a new problem — together with how often each occurs. A new
   identity, a recorded one occurring more often, or a higher total fails the
   build. Because a different mypy reports a different set, `requirements-dev.txt`
   pins the version exactly and the ratchet refuses to compare across versions;
   bumping the pin means re-recording the baseline in the same change.

   The run is deliberately environment-independent: it passes
   `--no-site-packages`, so every third-party import is `Any` whether or not you
   have the optional extras installed. Without it, an install with `[all]` and a
   base install disagree about the error set and the gate fails for reasons that
   have nothing to do with the change under review. The flags are recorded in the
   baseline and checked too, so changing them means re-recording. effGen's use of
   third-party APIs is checked by the public-surface lane instead, which runs
   against the packages that are actually installed.

### The linter's target version tracks the supported Python floor

`[tool.ruff] target-version` in `pyproject.toml` equals the `requires-python`
floor, currently 3.11. Several pyupgrade rules are gated behind the target and
rewrite code that the floor cannot run — `UP017` (`datetime.UTC`), `UP041` (the
bare `TimeoutError` alias) and `UP042` (`enum.StrEnum`) need 3.11, and `UP047`
(PEP 695 type parameters) needs 3.12. They are therefore off because of the
target, not because they were added to the `ignore` list; both facts are gated by
`tests/unit/test_static_check_ratchet.py`. When the floor moves, raise the
target, then apply those rules as their own change.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for your changes (unit tests at minimum, integration tests if GPU-dependent)
3. **Update CHANGELOG.md** with a summary of your changes
4. **Ensure all tests pass**: `pytest tests/unit/ -v --no-cov`
5. **Submit your PR** with a clear description of the changes

### PR Checklist

- [ ] Tests added/updated
- [ ] CHANGELOG.md updated
- [ ] Code passes `black --check` and `isort --check`
- [ ] No new `TODO`/`FIXME` without a tracking issue
- [ ] Documentation updated if public API changed

## Issue Reporting

When reporting bugs, please include:

- Python version (`python --version`)
- effGen version (`python -c "import effgen; print(effgen.__version__)"`)
- GPU info (if relevant): `nvidia-smi`
- Full error traceback
- Steps to reproduce

## Architecture Overview

```
effgen/
├── core/           # Agent, AgentConfig, ReAct loop
├── models/         # Model backends (vLLM, Transformers, API adapters)
├── tools/          # Built-in tools and protocols (MCP, A2A, ACP)
├── memory/         # Short-term, long-term, vector memory
├── prompts/        # Template management and optimization
├── config/         # Configuration loading and validation
├── execution/      # Code execution sandboxing
├── gpu/            # GPU allocation and monitoring
└── utils/          # Logging, metrics, validators, health checks
```

### Key Design Principles

- **Open Source First**: All features must work without paid APIs
- **SLM-Optimized**: Prompts and tools designed for 1B-7B parameter models
- **Tools extend `BaseTool`** with `async _execute()` method
- **Agent uses ReAct loop**: Thought → Action → Observation → repeat

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
