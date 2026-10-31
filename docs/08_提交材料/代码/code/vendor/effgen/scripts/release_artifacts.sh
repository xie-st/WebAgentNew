#!/usr/bin/env bash
################################################################################
# Release artifact collector.
#
# Captures the dependency + environment diagnostics that should accompany every
# release of effGen: pip freeze, pip check, pip-audit, nvidia-smi,
# torch CUDA diagnostics, and a provider-readiness summary. Everything is written
# under an output directory; nothing is uploaded and no secrets are printed.
#
#   scripts/release_artifacts.sh [OUTPUT_DIR]
#
# Default OUTPUT_DIR: dist/release-artifacts-<timestamp>
################################################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${EFFGEN_PYTHON:-python}"

OUT="${1:-${REPO_ROOT}/dist/release-artifacts-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT}"
echo "[artifacts] writing to ${OUT}"

# 1. Exact environment.
"${PY}" -m pip freeze > "${OUT}/pip-freeze.txt" 2>&1 || true
echo "[artifacts] pip freeze -> pip-freeze.txt"

# 2. Dependency consistency.
"${PY}" -m pip check > "${OUT}/pip-check.txt" 2>&1
PIPCHECK_RC=$?
echo "[artifacts] pip check (rc=${PIPCHECK_RC}) -> pip-check.txt"

# 3. Vulnerability audit (best-effort; tool may be absent).
if "${PY}" -m pip_audit --version >/dev/null 2>&1; then
    "${PY}" -m pip_audit --progress-spinner off > "${OUT}/pip-audit.txt" 2>&1 || true
    echo "[artifacts] pip-audit -> pip-audit.txt"
else
    echo "pip-audit not installed (pip install pip-audit)" > "${OUT}/pip-audit.txt"
    echo "[artifacts] pip-audit not installed — noted"
fi

# 4. GPU + driver snapshot.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${OUT}/nvidia-smi.txt" 2>&1 || true
    echo "[artifacts] nvidia-smi -> nvidia-smi.txt"
else
    echo "nvidia-smi not found (no NVIDIA driver on this host)" > "${OUT}/nvidia-smi.txt"
fi

# 5. torch CUDA diagnostics.
"${PY}" - > "${OUT}/torch-cuda.txt" 2>&1 <<'PY' || true
try:
    import torch
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("cuda.is_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
except Exception as e:  # noqa: BLE001
    print(f"torch unavailable: {e}")
PY
echo "[artifacts] torch CUDA diag -> torch-cuda.txt"

# 6. Provider readiness summary (key-present / SDK-import; NO live calls, NO secrets).
"${PY}" - > "${OUT}/provider-readiness.txt" 2>&1 <<'PY' || true
import warnings
warnings.filterwarnings("ignore")
import importlib, os
providers = {
    "openai": ("OPENAI_API_KEY", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
    "gemini": ("GOOGLE_API_KEY", "google.generativeai"),
    "cerebras": ("CEREBRAS_API_KEY", "cerebras.cloud.sdk"),
    "groq": ("GROQ_API_KEY", "groq"),
    "together": ("TOGETHER_API_KEY", "together"),
    "fireworks": ("FIREWORKS_API_KEY", "fireworks"),
    "replicate": ("REPLICATE_API_TOKEN", "replicate"),
    "hf": ("HF_TOKEN", "huggingface_hub"),
}
print(f"{'provider':12} {'key':6} {'sdk':6}")
for name, (env, mod) in providers.items():
    key = "yes" if os.environ.get(env) else "no"   # presence only, never the value
    try:
        importlib.import_module(mod); sdk = "yes"
    except Exception:
        sdk = "no"
    print(f"{name:12} {key:6} {sdk:6}")
PY
echo "[artifacts] provider readiness -> provider-readiness.txt"

# 7. effGen version + a tiny manifest.
{
    echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "effgen_version=$("${PY}" -c 'import effgen; print(effgen.__version__)' 2>/dev/null || echo unknown)"
    echo "python=$("${PY}" --version 2>&1)"
    echo "pip_check_rc=${PIPCHECK_RC}"
} > "${OUT}/manifest.txt"

echo "[artifacts] done. Files:"
ls -1 "${OUT}"
exit 0
