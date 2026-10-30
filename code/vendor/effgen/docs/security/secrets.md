# Secret Management in effGen

effGen never stores API keys in source code. This document describes how we prevent secrets from entering the repository.

---

## How Secrets Are Managed

### 1. Environment Variables / `.env` file

API keys are loaded exclusively from environment variables or a `.env` file in the project root:

```bash
# .env (gitignored — never committed)
CEREBRAS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
# ... etc
```

The `.env` file is listed in `.gitignore` and will never be committed. effGen loads these via `python-dotenv` at startup.

### 2. `~/.effgen/.env` (Recommended for Production)

For production use, store secrets in `~/.effgen/.env` rather than the project directory:

```bash
mkdir -p ~/.effgen
cat > ~/.effgen/.env << 'EOF'
CEREBRAS_API_KEY=...
OPENAI_API_KEY=...
EOF
chmod 600 ~/.effgen/.env
```

### 3. CI/CD — GitHub Actions Secrets

In CI, keys are injected via GitHub Actions secrets (`Settings → Secrets → Actions`). Never hardcode keys in workflow files.

---

## Secret Scanning

### Gitleaks (Pre-commit + CI)

We use [gitleaks](https://github.com/gitleaks/gitleaks) to detect accidentally committed secrets.

**Pre-commit hook** (runs on every `git commit`):

```bash
# Install pre-commit hooks once
pip install pre-commit
pre-commit install
```

The hook is configured in `.pre-commit-config.yaml` and uses `.gitleaks.toml` for effGen-specific rules.

**CI** (`.github/workflows/secret-scan.yml`):

- Runs on every push and PR.
- Full history scan (fetches all commits).
- Fails the build if any secrets are detected.
- Report uploaded as artifact on failure.

### Supported Provider Pattern Detection

`.gitleaks.toml` includes rules for:

| Provider | Pattern |
|----------|---------|
| Cerebras | `csk-...` / `CEREBRAS_API_KEY=...` |
| Groq | `gsk_...` |
| OpenAI | `sk-proj-...` / `sk-svcacct-...` |
| Anthropic | `sk-ant-api03-...` |
| Google Gemini | `AIza...` |
| HuggingFace | `hf_...` |
| Replicate | `r8_...` |
| Together AI | `TOGETHER_API_KEY=...` |
| Fireworks AI | `FIREWORKS_API_KEY=...` |
| AWS | `AKIA...` |

### Running Manually

```bash
# Scan entire repo (non-git mode)
gitleaks dir . --config .gitleaks.toml --verbose

# Scan git history
gitleaks git . --config .gitleaks.toml --verbose

# Scan staged files (pre-commit style)
gitleaks protect --staged --config .gitleaks.toml --verbose
```

### Allowlist

The following are explicitly allowlisted and will not trigger false positives:

- `tests/` directory — fake/planted keys for scanner tests
- `docs/` directory — documentation with placeholder values
- `scripts/` — install scripts with comment placeholders
- `node_modules/`, `vendor/`, `site-packages/`, `.venv/` — vendored third-party deps
- `*.d.ts`, `*.min.js` — type definitions and minified bundles
- `.env` file itself (gitignored, not scanned)
- `requirements-lock.txt` — dependency hashes

> The CI workflow scans the full git history (`fetch-depth: 0`). Vendored deps that
> were committed and later removed remain in history, so they must stay allowlisted.

---

## What To Do If a Secret Is Accidentally Committed

1. **Immediately rotate the key** at the provider dashboard.
2. Remove the commit from history:
   ```bash
   git filter-repo --replace-text <(echo "compromised_value==>REDACTED")
   ```
3. Force-push the cleaned history.
4. Notify the team.

The old key is **permanently compromised** once pushed to a remote — rotation is the only safe option.

---

## Test Fixtures

Tests in `tests/security/test_secret_patterns.py` plant fake secrets in temporary directories to validate that gitleaks detects them. These keys are:
- Non-sequential random-looking hex strings
- Located in temporary directories not in the repo
- Never committed to git

See `tests/security/test_secret_patterns.py` for the test harness.
