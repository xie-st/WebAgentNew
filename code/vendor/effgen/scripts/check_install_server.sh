#!/usr/bin/env bash
################################################################################
# Install-matrix check: server install.
#
# Installs the [server] extra into a fresh venv, runs the shared check battery,
# then boots `effgen serve` on loopback and verifies:
#   * the K8s probes (/health /healthz /livez /readyz) respond 200, and
#   * /v1/chat/completions rejects an unauthenticated request (auth-on default).
# The server is always torn down, even on failure.
#
#   bash scripts/check_install_server.sh
################################################################################
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_install_matrix_common.sh"

print_header "server"
require_python_311
build_wheel
create_venv
install_wheel "[server]"
check_pip_check
check_import_smoke
check_cli_smoke

# Pick an ephemeral-ish port and boot the server on loopback. Run it in its own
# process group (setsid) so we can kill uvicorn AND its worker children on
# teardown — a plain `kill <pid>` only reaps the CLI parent and orphans the
# listener, leaving the port bound.
PORT="${MATRIX_SERVER_PORT:-8791}"
log "starting effgen serve on 127.0.0.1:${PORT} ..."
setsid "${VENV_DIR}/bin/effgen" serve -p "${PORT}" >"${MATRIX_WORKDIR}/serve.log" 2>&1 &
SERVER_PID=$!
stop_server() {
    # Kill the whole process group (negative PID); fall back to the single PID.
    kill -TERM -"${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap 'stop_server; cleanup_matrix' EXIT

# Wait for readiness (up to ~20s).
ready=0
for _ in $(seq 1 20); do
    if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/healthz" 2>/dev/null; then ready=1; break; fi
    sleep 1
done
[ "${ready}" = "1" ] || { fail "server did not become ready; log:"; tail -25 "${MATRIX_WORKDIR}/serve.log"; exit 1; }

log "probing health endpoints ..."
for ep in /health /healthz /livez /readyz; do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}${ep}")"
    if [ "${code}" = "200" ]; then ok "${ep} -> 200"; else fail "${ep} -> ${code} (expected 200)"; exit 1; fi
done

log "verifying auth-on-by-default ..."
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d '{"model":"x","messages":[]}')"
if [ "${code}" = "401" ] || [ "${code}" = "403" ]; then
    ok "/v1/chat/completions without credentials -> ${code} (rejected)"
else
    fail "/v1/chat/completions without credentials -> ${code} (expected 401/403 — auth must be on by default)"
    exit 1
fi

stop_server
print_footer "server"
