#!/usr/bin/env bash
################################################################################
# Install-matrix check: torch-preservation constraints flow (CUDA 12.4).
#
# Proves that the documented constraints flow stops an extras install from
# silently replacing a driver-compatible torch:
#
#   1. install a cu124 torch (driver-compatible on any CUDA 12.x host),
#   2. install effGen + an extra WITH `-c constraints-cu124.txt`,
#   3. assert the torch version is byte-for-byte unchanged and CUDA still usable,
#   4. negative control: the same constraint rejects a different torch version,
#      proving the pin is load-bearing (not a no-op).
#
#   bash scripts/check_install_constraints.sh
#
# Defaults to the cu124 line (this box's driver). Override:
#   CUDA_LINE=cu128 CONSTRAINTS=constraints-cu128.txt bash scripts/check_install_constraints.sh
################################################################################
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_install_matrix_common.sh"

CUDA_LINE="${CUDA_LINE:-cu124}"
CONSTRAINTS="${CONSTRAINTS:-constraints-${CUDA_LINE}.txt}"

print_header "torch-preservation constraints (${CUDA_LINE})"
require_python_311
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — CUDA assertions will be skipped, only version preservation is checked."
    HAVE_GPU=0
else
    HAVE_GPU=1
fi

build_wheel
create_venv

# Step 1: pin a driver-compatible torch.
install_torch "${CUDA_LINE}"
TORCH_BEFORE="$(torch_version)"
[ -n "${TORCH_BEFORE}" ] || { fail "torch not importable after install"; exit 1; }
log "pinned torch before extras install: ${TORCH_BEFORE}"

# Step 2: install effGen + a use-case extra WITH the constraints file.
install_wheel_constrained "[local]" "${CONSTRAINTS}"

# Step 3: torch must be unchanged.
TORCH_AFTER="$(torch_version)"
log "torch after constrained extras install: ${TORCH_AFTER}"
if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then
    fail "torch was replaced despite the constraint: ${TORCH_BEFORE} -> ${TORCH_AFTER}"
    exit 1
fi
ok "torch preserved across extras install: ${TORCH_AFTER}"

if [ "${HAVE_GPU}" = "1" ]; then
    if "${VENV_PY}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        ok "torch.cuda.is_available() == True (GPU still usable)"
    else
        fail "torch.cuda.is_available() == False after constrained install — GPU lost"
        exit 1
    fi
fi

# Step 4: negative control — the constraint must REJECT a conflicting torch
# version (proving the pin actually binds, rather than torch just happening not
# to move). 2.5.1+cu124 exists on the cu124 index but != the pinned 2.6.0+cu124.
log "negative control: a conflicting torch version must be rejected by ${CONSTRAINTS} ..."
NEG_PKG="torch==2.5.1+${CUDA_LINE}"
if "${VENV_PY}" -m pip install --dry-run "${NEG_PKG}" -c "${REPO_ROOT}/${CONSTRAINTS}" \
        >"${MATRIX_WORKDIR}/neg.log" 2>&1; then
    fail "constraint did NOT block ${NEG_PKG} — the pin is not load-bearing"
    cat "${MATRIX_WORKDIR}/neg.log"
    exit 1
else
    ok "constraint correctly rejected ${NEG_PKG} (the pin binds torch)"
fi

check_pip_check
check_import_smoke
check_cli_smoke
print_footer "torch-preservation constraints (${CUDA_LINE})"
