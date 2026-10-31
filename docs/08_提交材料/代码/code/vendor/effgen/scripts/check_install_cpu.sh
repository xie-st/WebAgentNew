#!/usr/bin/env bash
################################################################################
# Install-matrix check: CPU-only base install.
#
# Builds the wheel, installs it (with the CPU torch wheel) into a fresh venv,
# and runs pip check / import / CLI / minimal CPU inference. This is the lane
# every contributor can run without a GPU.
#
#   bash scripts/check_install_cpu.sh
#
# Env knobs (see scripts/_install_matrix_common.sh):
#   MATRIX_PYTHON   interpreter used to build + create the venv (default python3)
#   MATRIX_WHEEL    reuse a pre-built wheel instead of building
#   MATRIX_KEEP=1   keep the scratch workdir for debugging
################################################################################
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_install_matrix_common.sh"

print_header "CPU base"
require_python_311
build_wheel
create_venv
install_torch cpu
install_wheel "[local]"          # local = CPU-capable transformers inference path
check_pip_check
check_import_smoke
check_cli_smoke
check_minimal_inference 0        # 0 = GPU not required; CPU inference is fine
print_footer "CPU"
