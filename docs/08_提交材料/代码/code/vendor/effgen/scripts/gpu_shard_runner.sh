#!/usr/bin/env bash
################################################################################
# GPU shard runner — fan a GPU test selection across the free GPUs on this host.
#
# It reads nvidia-smi to find idle GPUs, splits the selected tests into one shard
# per GPU, pins each shard with CUDA_VISIBLE_DEVICES, runs them in parallel with a
# UNIQUE per-shard COVERAGE_FILE (so coverage never corrupts),
# optionally combines coverage, and kills any orphan GPU processes it spawned.
#
# Usage:
#   scripts/gpu_shard_runner.sh [-m "gpu"] [pytest args...]
#
# Examples:
#   scripts/gpu_shard_runner.sh                      # runs `-m gpu` across free GPUs
#   scripts/gpu_shard_runner.sh tests/e2e            # shard a directory
#   EFFGEN_SHARD_COV=1 scripts/gpu_shard_runner.sh   # also collect+combine coverage
#
# Env knobs:
#   EFFGEN_MAX_SHARDS   cap the number of GPUs/shards used (default: all free)
#   EFFGEN_GPU_MEM_MB   a GPU counts as "free" below this used-MiB (default 1000)
#   EFFGEN_SHARD_COV=1  enable coverage (unique data file per shard + combine)
#   EFFGEN_PYTEST       pytest entrypoint (default: python -m pytest)
################################################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTEST="${EFFGEN_PYTEST:-python -m pytest}"
GPU_MEM_MB="${EFFGEN_GPU_MEM_MB:-1000}"
WORKDIR="$(mktemp -d -t effgen-gpu-shards-XXXXXX)"
PIDS=()
declare -A SHARD_GPU
declare -A SHARD_RC

log()  { printf "[gpu-shards] %s\n" "$*"; }
fail() { printf "[gpu-shards][fail] %s\n" "$*" >&2; }

# Default selection: the `gpu` marker. Anything passed on the CLI overrides it.
PYTEST_ARGS=("$@")
if [ "${#PYTEST_ARGS[@]}" -eq 0 ]; then
    PYTEST_ARGS=(-m gpu)
fi

# -- 1. find free GPUs ---------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found — no GPUs to shard across."
    exit 2
fi

mapfile -t FREE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v lim="${GPU_MEM_MB}" '{ gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < lim+0) print $1 }'
)
if [ "${#FREE_GPUS[@]}" -eq 0 ]; then
    fail "no GPU under ${GPU_MEM_MB} MiB used — refusing to overcommit busy GPUs."
    exit 3
fi
if [ -n "${EFFGEN_MAX_SHARDS:-}" ] && [ "${#FREE_GPUS[@]}" -gt "${EFFGEN_MAX_SHARDS}" ]; then
    FREE_GPUS=("${FREE_GPUS[@]:0:${EFFGEN_MAX_SHARDS}}")
fi
NSHARDS="${#FREE_GPUS[@]}"
log "free GPUs: ${FREE_GPUS[*]} -> ${NSHARDS} shard(s)"


# -- 2. collect node ids (coverage-free thanks to the addopts fix) -------------
log "collecting test node ids for: ${PYTEST_ARGS[*]}"
mapfile -t NODEIDS < <(${PYTEST} "${PYTEST_ARGS[@]}" --collect-only -q -p no:cacheprovider 2>/dev/null \
    | grep '::' | grep -v '^[[:space:]]')
if [ "${#NODEIDS[@]}" -eq 0 ]; then
    fail "no tests collected for selection: ${PYTEST_ARGS[*]}"
    exit 4
fi
log "collected ${#NODEIDS[@]} test(s)"
[ "${NSHARDS}" -gt "${#NODEIDS[@]}" ] && NSHARDS="${#NODEIDS[@]}"

# -- 3. round-robin tests into per-GPU shard files -----------------------------
for ((s=0; s<NSHARDS; s++)); do : > "${WORKDIR}/shard_${s}.txt"; done
for ((i=0; i<${#NODEIDS[@]}; i++)); do
    printf '%s\n' "${NODEIDS[$i]}" >> "${WORKDIR}/shard_$((i % NSHARDS)).txt"
done

# -- 4. launch one pinned shard per free GPU -----------------------------------
COV_ARGS=()
if [ "${EFFGEN_SHARD_COV:-0}" = "1" ]; then
    COV_ARGS=(--cov=effgen --cov-report=)   # data only; we combine + report after
    log "coverage enabled (per-shard COVERAGE_FILE + combine)"
fi

# Job control puts every background job in a process group of its own, whose id
# is the job's pid. That is what makes a leftover process attributable later: a
# GPU process belongs to this run when its process group is one of ours, and a
# job started elsewhere on this shared host never is. Parentage cannot answer
# the question, because a worker that outlives its shard is reparented away.
set -m

for ((s=0; s<NSHARDS; s++)); do
    gpu="${FREE_GPUS[$s]}"
    SHARD_GPU[$s]="${gpu}"
    log "shard ${s} -> GPU ${gpu} ($(wc -l < "${WORKDIR}/shard_${s}.txt") tests)"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export COVERAGE_FILE="${WORKDIR}/.coverage.shard_${s}"
        # shellcheck disable=SC2046
        ${PYTEST} $(cat "${WORKDIR}/shard_${s}.txt") "${COV_ARGS[@]}" \
            -p no:cacheprovider -q > "${WORKDIR}/shard_${s}.log" 2>&1
    ) &
    PIDS[$s]=$!
done

# -- 5. wait + collect exit codes ----------------------------------------------
overall_rc=0
for ((s=0; s<NSHARDS; s++)); do
    wait "${PIDS[$s]}"; rc=$?
    SHARD_RC[$s]=$rc
    if [ "$rc" -eq 0 ]; then
        log "shard ${s} (GPU ${SHARD_GPU[$s]}): PASS"
    else
        fail "shard ${s} (GPU ${SHARD_GPU[$s]}): FAIL (rc=${rc}) — tail:"
        tail -15 "${WORKDIR}/shard_${s}.log" >&2
        overall_rc=1
    fi
done

# -- 6. combine coverage -------------------------------------------------------
if [ "${EFFGEN_SHARD_COV:-0}" = "1" ]; then
    log "combining coverage from ${NSHARDS} shard(s) ..."
    COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage combine "${WORKDIR}"/.coverage.shard_* 2>/dev/null || true
    COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage report 2>/dev/null | tail -3 || true
    COVERAGE_FILE="${REPO_ROOT}/.coverage" coverage xml -o "${REPO_ROOT}/coverage.xml" 2>/dev/null || true
    log "wrote ${REPO_ROOT}/coverage.xml"
fi

# -- 7. clean up orphan GPU procs on the assigned GPUs -------------------------
# Each shard is its own process group, so the whole group goes at once — a
# worker the shard forked is caught even after being reparented.
for ((s=0; s<NSHARDS; s++)); do
    kill -- "-${PIDS[$s]}" 2>/dev/null || true
done

# Warn about anything of ours still holding a GPU. Membership of one of this
# run's process groups is the test: a process on the same GPU belonging to
# somebody else's job is a normal thing to find on a shared host, and naming it
# here would send the reader after a process they must not touch.
is_ours() {
    local pgid
    pgid="$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' ')"
    [ -n "${pgid}" ] || return 1
    for ((k=0; k<NSHARDS; k++)); do
        [ "${pgid}" = "${PIDS[$k]}" ] && return 0
    done
    return 1
}

for gpu in "${FREE_GPUS[@]}"; do
    leftover=""
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu}" 2>/dev/null | tr -d ' '); do
        is_ours "${pid}" && leftover="${leftover} ${pid}"
    done
    leftover="$(printf '%s' "${leftover}" | sed 's/^ *//')"
    [ -n "${leftover}" ] && fail "GPU ${gpu} still has compute procs this run started: ${leftover}"
done

rm -rf "${WORKDIR}"
if [ "${overall_rc}" -eq 0 ]; then
    log "ALL ${NSHARDS} SHARD(S) PASSED"
else
    fail "one or more shards failed"
fi
exit "${overall_rc}"
