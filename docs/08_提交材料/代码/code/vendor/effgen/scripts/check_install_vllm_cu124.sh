#!/usr/bin/env bash
################################################################################
# Install-matrix check: vLLM on CUDA 12.4.
#
# vLLM pins an exact torch build, so this lane installs the [vllm] extra and lets
# vLLM resolve its own torch, then runs pip check (the vLLM<->torch ABI is the
# historical conflict point) and confirms the vllm module imports. Heavy vLLM
# engine startup is intentionally NOT run here (it needs a long model load); the
# goal is to prove the dependency set is internally consistent and importable.
#
#   bash scripts/check_install_vllm_cu124.sh
#
# If vLLM and a driver-compatible torch genuinely cannot co-exist on this host,
# the script reports the conflict instead of silently "passing".
################################################################################
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_install_matrix_common.sh"

print_header "vLLM CUDA 12.4"
require_python_311
build_wheel
create_venv
# vLLM ships its own pinned torch+cuda wheels; install the extra first and let it
# resolve torch, rather than pre-pinning a cu124 torch it would then replace.
install_wheel "[vllm]"
check_pip_check
check_import_smoke
check_cli_smoke
log "vLLM import smoke ..."
if "${VENV_PY}" -c "import vllm; print('vllm', vllm.__version__)" >"${MATRIX_WORKDIR}/vllm.log" 2>&1; then
    ok "vLLM imports: $(cat "${MATRIX_WORKDIR}/vllm.log")"
else
    fail "vLLM import failed (likely a torch/CUDA ABI mismatch):"
    tail -15 "${MATRIX_WORKDIR}/vllm.log"
    exit 1
fi
print_footer "vLLM CUDA 12.4"
