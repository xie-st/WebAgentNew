#!/usr/bin/env bash
################################################################################
# Coverage runner — collect coverage safely, including across parallel shards.
#
# Coverage is intentionally OFF in the default pytest config (so `--collect-only`
# and naive parallel runs don't touch / corrupt the coverage DB). Use this script
# when you actually want a coverage report.
#
# Single process (default):
#   scripts/run_coverage.sh tests/unit
#
# Sharded (give each shard its own COVERAGE_FILE, then combine):
#   scripts/run_coverage.sh --shards 4 tests/unit tests/integration
#
# Each shard writes to a unique data file, so concurrent shards never race the
# shared `.coverage` SQLite DB (concurrent-write corruption). Outputs coverage.xml
# + htmlcov + a terminal report at the repo root.
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTEST="${EFFGEN_PYTEST:-python -m pytest}"
SHARDS=1
if [ "${1:-}" = "--shards" ]; then SHARDS="$2"; shift 2; fi
TARGETS=("$@"); [ "${#TARGETS[@]}" -eq 0 ] && TARGETS=(tests/unit)

DATADIR="$(mktemp -d -t effgen-cov-XXXXXX)"
rm -f "${REPO_ROOT}/.coverage" "${REPO_ROOT}"/.coverage.* "${REPO_ROOT}/coverage.xml"
trap 'rm -rf "${DATADIR}"' EXIT

echo "[coverage] ${SHARDS} shard(s) over: ${TARGETS[*]}"

if [ "${SHARDS}" -le 1 ]; then
    COVERAGE_FILE="${DATADIR}/.coverage" ${PYTEST} "${TARGETS[@]}" \
        --cov=effgen --cov-report= -p no:cacheprovider
else
    # Split collected node ids round-robin into SHARDS groups.
    mapfile -t NODEIDS < <(${PYTEST} "${TARGETS[@]}" --collect-only -q -p no:cacheprovider 2>/dev/null | grep '::')
    [ "${#NODEIDS[@]}" -gt 0 ] || { echo "[coverage] no tests collected"; exit 1; }
    for ((s=0; s<SHARDS; s++)); do : > "${DATADIR}/shard_${s}.txt"; done
    for ((i=0; i<${#NODEIDS[@]}; i++)); do
        printf '%s\n' "${NODEIDS[$i]}" >> "${DATADIR}/shard_$((i % SHARDS)).txt"
    done
    pids=()
    for ((s=0; s<SHARDS; s++)); do
        ( COVERAGE_FILE="${DATADIR}/.coverage.shard_${s}" \
          ${PYTEST} $(cat "${DATADIR}/shard_${s}.txt") --cov=effgen --cov-report= \
          -p no:cacheprovider -q > "${DATADIR}/shard_${s}.log" 2>&1 ) &
        pids+=($!)
    done
    rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    [ "$rc" -eq 0 ] || echo "[coverage] WARNING: a shard reported test failures (see logs)"
fi

# Combine + report from the repo root so source paths resolve.
COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage combine "${DATADIR}"/.coverage* 2>/dev/null || true
COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage report --precision=2 | tail -25
COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage xml -o "${REPO_ROOT}/coverage.xml" >/dev/null 2>&1 || true
COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage html -d "${REPO_ROOT}/htmlcov" >/dev/null 2>&1 || true
echo "[coverage] wrote ${REPO_ROOT}/coverage.xml + htmlcov/"
