#!/bin/bash
################################################################################
# effGen setup_and_verify.sh — DEPRECATED thin forwarder.
#
# Installer reconciliation: there is now ONE canonical install engine,
# scripts/install_effgen.sh (it owns the driver-aware PyTorch selection), fronted
# by the repo-root ./install.sh. This orchestrator used to sit in between and
# added a third banner; it is kept only so existing callers keep working and now
# simply forwards to ./install.sh with the same flags.
#
# Use ./install.sh instead.
################################################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[note] scripts/setup_and_verify.sh is deprecated — forwarding to ./install.sh"
echo "       (the canonical install engine is scripts/install_effgen.sh)"
echo ""

exec bash "$REPO_ROOT/install.sh" "$@"
