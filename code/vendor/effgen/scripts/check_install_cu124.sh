#!/usr/bin/env bash
################################################################################
# Install-matrix check: CUDA 12.4 local-inference install.
#
# Installs the cu124 torch wheel (driver-compatible on any CUDA 12.x host) plus
# the [local] extra, then proves torch.cuda.is_available() and runs a tiny model
# ON the GPU. Use this on the A40 host to confirm the documented GPU install
# does not silently fall back to CPU.
#
#   bash scripts/check_install_cu124.sh
################################################################################
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_install_matrix_common.sh"

print_header "CUDA 12.4 local"
require_python_311
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — this lane validates a GPU install; running anyway but inference may fail."
fi
build_wheel
create_venv
install_torch cu124
install_wheel "[local]"
check_pip_check
check_import_smoke
check_cli_smoke
check_minimal_inference 1        # 1 = require a working GPU (fail if CPU fallback)
print_footer "CUDA 12.4"
