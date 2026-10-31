#!/usr/bin/env bash
################################################################################
# Repository cleanup — remove generated/test/cache artifacts from the tree.
#
# Everything removed here is already gitignored; this script just makes a tidy
# tree a one-liner so build/test runs don't leave litter behind.
#
#   scripts/clean.sh              # remove caches, coverage, build artifacts, runtime state
#   scripts/clean.sh --dry-run    # show what would be removed without deleting
#   scripts/clean.sh --build-envs # also remove stale effgen-v0.3.0-p* conda build envs
#   scripts/clean.sh --kill-orphans  # also kill leftover effgen build/test orphan procs
#
# By default this only removes gitignored artifacts. The --build-envs and
# --kill-orphans sweeps are OPT-IN because they remove conda environments and
# terminate processes; without those flags, lingering build envs and orphan
# processes are only reported, never touched.
#
# Never touches tracked source, .git/, or the .env file.
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
SWEEP_ENVS=0
KILL_ORPHANS=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)       DRY_RUN=1 ;;
        --build-envs)    SWEEP_ENVS=1 ;;
        --kill-orphans)  KILL_ORPHANS=1 ;;
        *) echo "[clean] unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

run() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[clean] would run: $*"
    else
        eval "$@"
    fi
}

[ "${DRY_RUN}" -eq 1 ] && LABEL=" (dry-run)" || LABEL=""
echo "[clean] tidying ${REPO_ROOT}${LABEL}"

# Coverage + test/type/lint caches and build outputs.
run "rm -rf .coverage .coverage.* coverage.xml htmlcov .hypothesis \
    .pytest_cache .ruff_cache .mypy_cache .dmypy.json dist build *.egg-info \
    gitleaks-report.json"

# Runtime state written under the working directory.
run "rm -rf checkpoints .effgen"
# Lane logs from scripts/run_tests.sh, and React stored by the web view.
run "rm -rf .test-run scripts/_watch_assets"
# `effgen config init` writes ./config.yaml unless -o names another path.
run "rm -f config.yaml"

# __pycache__ / *.pyc anywhere in the tree (never under .git or site-packages).
# Use `-exec rm` rather than `-delete`: `-delete` implies `-depth`, which
# silently disables the `./.git` prune (GNU find only warns, but bfs errors out
# and then deletes nothing). The prune + `-exec` form is the portable one.
run "find . -path ./.git -prune -o -type d -name __pycache__ \
    -not -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null || true"
run "find . -path ./.git -prune -o -type f -name '*.pyc' \
    -not -path '*/site-packages/*' -exec rm -f {} + 2>/dev/null || true"

# ---------------------------------------------------------------------------
# Stale per-run conda environments. Throwaway dev/test envs follow the naming
# convention effgen-v0.3.0-p<N>; if a run is interrupted before its
# `conda env remove`, they pile up. Report by default; remove with --build-envs.
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    STALE_ENVS="$(conda env list 2>/dev/null | awk '{print $1}' \
        | grep -E '^effgen-v0\.3\.0-p[0-9.]+$' || true)"
    if [ -n "${STALE_ENVS}" ]; then
        if [ "${SWEEP_ENVS}" -eq 1 ]; then
            for env in ${STALE_ENVS}; do
                run "conda env remove -n '${env}' -y >/dev/null"
                echo "[clean] removed conda env: ${env}"
            done
        else
            echo "[clean] stale build conda envs found (use --build-envs to remove):"
            echo "${STALE_ENVS}" | sed 's/^/[clean]   /'
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Orphan build/test processes owned by the current user. Detached repro
# scripts and stray servers (e.g. repro_*.py, isolate_*.py,
# `uvicorn ... create_app`, a long-running `python -m effgen.server`) can linger
# for days when a `timeout` wrapper fails to reap a detached child. Report by
# default; kill with --kill-orphans. Scoped to THIS user and to these specific
# command patterns so it never touches a co-tenant's or a production process.
# ---------------------------------------------------------------------------
# Narrow patterns only: detached repro/isolate scripts and stray effgen servers
# from a create_app smoke. Kept precise so a legitimate build run is never hit.
ORPHAN_RE='python[0-9.]* +[^ ]*/repro_[a-z0-9_]*\.py|python[0-9.]* +[^ ]*/isolate_[a-z0-9_]*\.py|uvicorn .*effgen.*:create_app|python[0-9.]* +-m +effgen\.server'
ORPHANS="$(pgrep -u "$(id -u)" -af . 2>/dev/null \
    | grep -E "${ORPHAN_RE}" \
    | grep -v -E 'clean\.sh|grep -E|grep -v|pgrep ' || true)"
if [ -n "${ORPHANS}" ]; then
    if [ "${KILL_ORPHANS}" -eq 1 ]; then
        echo "[clean] killing orphan build/test processes:"
        echo "${ORPHANS}" | sed 's/^/[clean]   /'
        ORPHAN_PIDS="$(echo "${ORPHANS}" | awk '{print $1}')"
        for pid in ${ORPHAN_PIDS}; do
            run "kill -TERM '${pid}' 2>/dev/null || true"
        done
        sleep 1
        for pid in ${ORPHAN_PIDS}; do
            run "kill -KILL '${pid}' 2>/dev/null || true"
        done
    else
        echo "[clean] orphan build/test processes found (use --kill-orphans to terminate):"
        echo "${ORPHANS}" | sed 's/^/[clean]   /'
    fi
fi

echo "[clean] done"
