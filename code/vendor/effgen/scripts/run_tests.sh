#!/usr/bin/env bash
################################################################################
# effGen — run the test suite, choosing which lanes to include.
#
#   scripts/run_tests.sh                 pick lanes interactively, then run
#   scripts/run_tests.sh --all           every lane this machine can run
#   scripts/run_tests.sh --only offline,lint
#   scripts/run_tests.sh --skip live,gpu,stress
#   scripts/run_tests.sh --list          show the lanes and exit
#
# Watch it from a second terminal:
#
#   scripts/watch_tests.sh
#
# Lanes are grouped into streams. Streams run at the same time; lanes inside a
# stream run one after another. Nothing is selected twice, so no test runs
# twice, and the lanes that spend provider quota share one stream so they cannot
# throttle each other.
#
# Results land in .test-run/ — one log, one exit code and one duration per lane,
# plus the manifest the watcher reads. Read .test-run/summary.txt afterwards.
################################################################################
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
L="$REPO/.test-run"
SUMMARY="$L/summary.txt"
PYTEST="${EFFGEN_PYTEST:-python -m pytest}"

cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1
export PAGER=cat GIT_PAGER=cat DEBIAN_FRONTEND=noninteractive

# A blank endpoint override is not "no override" to every client library: some
# read it as an address and send the call there. Clear them for the whole run.
for _v in EFFGEN_BASE_URL OPENAI_BASE_URL OPENAI_API_BASE; do unset "$_v"; done

OFFLINE_MARKERS="not gpu and not api and not live and not docker and not expensive"

# ---------------------------------------------------------------------------
# The lanes.  id | stream | estimated minutes | label
# ---------------------------------------------------------------------------
LANES='offline|A|20|Offline suite — every test that needs no key, GPU or network
gates|E|6|Packaging, public surface and docstring gates
lint|E|2|Lint (ruff)
types|E|12|Type checks — public surface and the ratchet
order|E|25|Collection-order matrix — finds order-dependent tests
coverage|B|25|Coverage run
live|C|45|Live provider tests — needs API keys
stress|C|50|Soak and contention — long, spends provider quota
gpu|D|25|GPU tests — one shard per free GPU
docker|F|15|Container tests — needs a running Docker daemon
install|G|75|Install-path checks — builds throwaway environments
secrets|G|10|Secret scan over the tree and the history'

lane_field() {  # lane_field <id> <1=stream 2=minutes 3=label>
    printf '%s\n' "$LANES" | awk -F'|' -v id="$1" -v f="$2" '$1==id {print $(f+1)}'
}

# ---------------------------------------------------------------------------
# What this machine can actually do. A lane that cannot run is never offered as
# if it could; the reason is printed instead, and it is recorded so the watcher
# can show it too.
# ---------------------------------------------------------------------------
unavailable() {  # unavailable <id> -> prints a reason, or nothing if it can run
    case "$1" in
        gpu)
            command -v nvidia-smi > /dev/null 2>&1 || { echo "no nvidia-smi"; return; }
            [ "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)" -gt 0 ] \
                || echo "no GPU visible"
            ;;
        docker)
            command -v docker > /dev/null 2>&1 || { echo "docker not installed"; return; }
            docker info > /dev/null 2>&1 || echo "docker daemon not reachable"
            ;;
        secrets)
            command -v gitleaks > /dev/null 2>&1 || echo "gitleaks not installed"
            ;;
        live|stress)
            has_any_key || echo "no provider key in the environment or .env"
            ;;
        types)
            command -v mypy > /dev/null 2>&1 || echo "mypy not installed"
            ;;
        lint)
            command -v ruff > /dev/null 2>&1 || echo "ruff not installed"
            ;;
        order)
            [ -f "$REPO/scripts/run_order_matrix.py" ] || echo "scripts/run_order_matrix.py is absent"
            ;;
        coverage)
            [ -f "$REPO/scripts/run_coverage.sh" ] || echo "scripts/run_coverage.sh is absent"
            ;;
    esac
}

has_any_key() {
    local v
    for v in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY \
             GROQ_API_KEY CEREBRAS_API_KEY TOGETHER_API_KEY FIREWORKS_API_KEY \
             MISTRAL_API_KEY COHERE_API_KEY REPLICATE_API_TOKEN DEEPSEEK_API_KEY; do
        [ -n "${!v:-}" ] && return 0
    done
    # A key present only in .env still counts: the suite loads it.
    [ -f "$REPO/.env" ] && grep -qE '^[A-Z_]*API_(KEY|TOKEN)=.+' "$REPO/.env" && return 0
    return 1
}

# ---------------------------------------------------------------------------
# What each lane actually runs.
# ---------------------------------------------------------------------------
run_lane() {
    case "$1" in
        offline)  run_offline_shards ;;
        gates)    $PYTEST -q -p no:randomly \
                      tests/packaging tests/unit/compat \
                      tests/unit/test_no_internal_scaffolding.py \
                      tests/unit/test_public_docstrings.py ;;
        lint)     ruff check effgen/ tests/ scripts/ ;;
        types)    bash scripts/check_public_types.sh && python scripts/mypy_ratchet.py ;;
        order)    python scripts/run_order_matrix.py ;;
        coverage) bash scripts/run_coverage.sh ;;
        live)     $PYTEST -q -p no:randomly --timeout=900 \
                      -m "(live or api) and not gpu and not docker and not expensive" tests ;;
        # The stress files set `timeout(0)` because a soak legitimately runs
        # long, so pytest's own per-test timeout cannot end a hang there. The
        # outer cap this lane runs under is the only thing that can.
        stress)   $PYTEST -q -p no:randomly -m "expensive and not docker" tests ;;
        gpu)      bash scripts/gpu_shard_runner.sh -m gpu ;;
        docker)   $PYTEST -q -p no:randomly --timeout=900 -m docker tests ;;
        install)  for s in cpu server; do
                      [ -f "scripts/check_install_${s}.sh" ] && bash "scripts/check_install_${s}.sh"
                  done ;;
        secrets)  gitleaks dir . --config .gitleaks.toml --no-banner \
                  && gitleaks git . --config .gitleaks.toml --no-banner ;;
    esac
}

# The offline suite is split by directory and the parts run at the same time.
# Each part writes its own log so the watcher can show them separately.
OFFLINE_SHARDS='unit|tests/unit
cli|tests/cli
models-core|tests/models tests/core
tools|tests/tools tests/presets tests/prompts
server|tests/server tests/deploy tests/security
dx|tests/dx tests/observability tests/reliability tests/packaging
integration|tests/integration tests/cookbook tests/multimodal
misc|tests/fuzz tests/stress tests/benchmarks tests/e2e'

run_offline_shards() {
    local name paths rc=0
    while IFS='|' read -r name paths; do
        [ -z "$name" ] && continue
        ( lane_exec "offline-$name" 3600 $PYTEST -q -p no:randomly --timeout=900 \
              -m "$OFFLINE_MARKERS" $paths ) &
    done <<< "$OFFLINE_SHARDS"
    wait
    for name in $(printf '%s\n' "$OFFLINE_SHARDS" | cut -d'|' -f1); do
        [ "$(cat "$L/offline-$name.rc" 2>/dev/null || echo 1)" != "0" ] && rc=1
    done
    return $rc
}

# ---------------------------------------------------------------------------
# Lane bookkeeping. The watcher reads exactly these files, so it needs no
# knowledge of what a lane does.
# ---------------------------------------------------------------------------
lane_exec() {  # lane_exec <name> <timeout-seconds> <command...>
    local name="$1" secs="$2"; shift 2
    local began; began=$(date +%s)
    date +%s > "$L/$name.start"
    timeout --kill-after=120 "$secs" "$@" > "$L/$name.txt" 2>&1 < /dev/null
    local rc=$?
    echo "$rc" > "$L/$name.rc"
    echo "$(( ($(date +%s) - began) / 60 ))" > "$L/$name.min"
    return 0
}

lane_run_wrapped() {  # a whole lane, including ones that are shell functions
    local id="$1" secs="$2"
    local began; began=$(date +%s)
    date +%s > "$L/$id.start"
    ( timeout --kill-after=120 "$secs" bash -c \
        "cd '$REPO' && source '$HERE/run_tests.sh' --source-only && run_lane '$id'" \
        > "$L/$id.txt" 2>&1 < /dev/null )
    local rc=$?
    echo "$rc" > "$L/$id.rc"
    echo "$(( ($(date +%s) - began) / 60 ))" > "$L/$id.min"
}

lane_timeout() {  # generous per-lane caps, in seconds
    case "$1" in
        offline) echo 5400 ;; live) echo 7200 ;; stress) echo 9000 ;;
        gpu) echo 5400 ;; install) echo 9000 ;; order) echo 5400 ;;
        coverage) echo 5400 ;; docker) echo 3600 ;; secrets) echo 2400 ;;
        *) echo 2400 ;;
    esac
}

# Allow the lane subshell to re-enter this file for its function definitions
# without re-running the whole script.
if [ "${1:-}" = "--source-only" ]; then return 0 2>/dev/null || exit 0; fi

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
declare -A SELECTED=() REASON=()
MODE=interactive; ONLY=""; SKIP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --all)   MODE=all ;;
        --list)  MODE=list ;;
        --only)  MODE=explicit; ONLY="$2"; shift ;;
        --skip)  MODE=all; SKIP="$2"; shift ;;
        -y|--yes) MODE=all ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift
done

ids() { printf '%s\n' "$LANES" | cut -d'|' -f1; }

for id in $(ids); do
    r="$(unavailable "$id")"
    REASON[$id]="$r"
    if [ -n "$r" ]; then SELECTED[$id]=0; else SELECTED[$id]=1; fi
done

case "$MODE" in
    explicit)
        for id in $(ids); do SELECTED[$id]=0; done
        IFS=',' read -ra want <<< "$ONLY"
        for id in "${want[@]}"; do
            [ -z "$id" ] && continue
            if [ -z "${REASON[$id]+exists}" ]; then
                echo "no such lane: '$id'"
            elif [ -n "${REASON[$id]}" ]; then
                echo "cannot run '$id': ${REASON[$id]}"
            else
                SELECTED[$id]=1
            fi
        done
        ;;
    all)
        IFS=',' read -ra drop <<< "${SKIP:-}"
        for id in "${drop[@]}"; do [ -n "$id" ] && SELECTED[$id]=0; done
        ;;
esac

print_menu() {
    printf '\n  %-3s %-4s %-10s %-6s %s\n' "" "" "lane" "est" "what it runs"
    printf '  %s\n' "────────────────────────────────────────────────────────────────────────────"
    local n=0 id stream est label mark
    while IFS='|' read -r id stream est label; do
        [ -z "$id" ] && continue
        n=$((n + 1))
        if [ -n "${REASON[$id]}" ]; then
            printf '  \033[2m%-3s [-]  %-10s %-6s %s — %s\033[0m\n' \
                "$n)" "$id" "${est}m" "$label" "${REASON[$id]}"
        else
            [ "${SELECTED[$id]}" = "1" ] && mark="\033[32mx\033[0m" || mark=" "
            printf '  %-3s [%b]  %-10s %-6s %s\n' "$n)" "$mark" "$id" "${est}m" "$label"
        fi
    done <<< "$LANES"
    printf '  %s\n' "────────────────────────────────────────────────────────────────────────────"
    printf '  \033[2mlanes marked [-] cannot run on this machine; the reason is shown\033[0m\n'
}

if [ "$MODE" = "list" ]; then print_menu; exit 0; fi

if [ "$MODE" = "interactive" ]; then
    if [ ! -t 0 ]; then
        echo "not a terminal — use --all, --only or --skip when running unattended"
        exit 2
    fi
    while :; do
        print_menu
        printf '\n  toggle by number (e.g. "3" or "3 5 7"), "none", "all", or Enter to start: '
        read -r reply || reply=""
        case "$reply" in
            "") break ;;
            all)  for id in $(ids); do [ -z "${REASON[$id]}" ] && SELECTED[$id]=1; done ;;
            none) for id in $(ids); do SELECTED[$id]=0; done ;;
            q|quit) echo "nothing run."; exit 0 ;;
            *)
                for num in $reply; do
                    id="$(ids | sed -n "${num}p")"
                    if [ -z "$id" ]; then echo "  no lane $num"; continue; fi
                    if [ -n "${REASON[$id]}" ]; then
                        echo "  $id cannot run here: ${REASON[$id]}"; continue
                    fi
                    [ "${SELECTED[$id]}" = "1" ] && SELECTED[$id]=0 || SELECTED[$id]=1
                done
                ;;
        esac
    done
fi

chosen=0
for id in $(ids); do [ "${SELECTED[$id]}" = "1" ] && chosen=$((chosen + 1)); done
if [ "$chosen" -eq 0 ]; then echo "no lane selected — nothing to run."; exit 0; fi

# ---------------------------------------------------------------------------
# The manifest. This is what the watcher reads; it needs nothing else.
# ---------------------------------------------------------------------------
rm -rf "$L"; mkdir -p "$L"
date +%s > "$L/run.start"
: > "$L/manifest.txt"

while IFS='|' read -r id stream est label; do
    [ -z "$id" ] && continue
    if [ "${SELECTED[$id]}" = "1" ]; then
        state=run
    elif [ -n "${REASON[$id]}" ]; then
        state="unavailable:${REASON[$id]}"
    else
        state="skipped:not selected"
    fi
    printf '%s|%s|%s|%s|%s\n' "$id" "$stream" "$est" "$label" "$state" >> "$L/manifest.txt"
    # Offline is many parts; list them so the watcher can show each one.
    if [ "$id" = "offline" ] && [ "$state" = "run" ]; then
        while IFS='|' read -r sname spaths; do
            [ -z "$sname" ] && continue
            printf 'offline-%s|A|3|  %s|run\n' "$sname" "$spaths" >> "$L/manifest.txt"
        done <<< "$OFFLINE_SHARDS"
    fi
done <<< "$LANES"

STARTED_AT="$(date -Is)"
{
    echo "# effGen — test run"
    echo "started:  $STARTED_AT"
    echo "host:     $(hostname)   $(nproc 2>/dev/null || echo '?') cores"
    echo "python:   $(python -V 2>&1)"
    echo "head:     $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
    echo ""
    echo "lanes selected: $chosen"
    while IFS='|' read -r id stream est label state; do
        case "$state" in
            run) printf '  run          %s\n' "$id" ;;
            skipped:*)     printf '  skipped      %-10s (%s)\n' "$id" "${state#skipped:}" ;;
            unavailable:*) printf '  unavailable  %-10s (%s)\n' "$id" "${state#unavailable:}" ;;
        esac
    done < "$L/manifest.txt"
    echo ""
} > "$SUMMARY"
cat "$SUMMARY"
echo "watch it with:  scripts/watch_tests.sh"
echo ""

# ---------------------------------------------------------------------------
# Run. One background job per stream; lanes inside a stream run in order.
# ---------------------------------------------------------------------------
streams="$(printf '%s\n' "$LANES" | cut -d'|' -f2 | sort -u)"
pids=""
for s in $streams; do
    (
        while IFS='|' read -r id stream est label; do
            [ "$stream" = "$s" ] || continue
            [ "${SELECTED[$id]}" = "1" ] || continue
            lane_run_wrapped "$id" "$(lane_timeout "$id")"
        done <<< "$LANES"
    ) &
    pids="$pids $!"
done
wait $pids 2>/dev/null

# Anything a lane forked and did not reap is a descendant of this script.
descendants() { local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do descendants "$c"; echo "$c"; done; }
for p in $(descendants $$); do kill -9 "$p" 2>/dev/null; done

# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
{
    echo "=============================== RESULT ==============================="
    echo "finished: $(date -Is)"
    echo ""
    while IFS='|' read -r id stream est label state; do
        case "$state" in
            run)
                rc="$(cat "$L/$id.rc" 2>/dev/null || echo '-')"
                mins="$(cat "$L/$id.min" 2>/dev/null || echo '?')"
                tot="$(grep -aE '[0-9]+ (passed|failed)' "$L/$id.txt" 2>/dev/null | tail -1)"
                case "$rc" in
                    0)       word="passed " ;;
                    124|137) word="TIMEOUT" ;;
                    -)       word="not run" ;;
                    *)       word="FAILED " ;;
                esac
                printf '  %-16s %s  %4sm  %s\n' "$id" "$word" "$mins" "${tot:0:60}"
                ;;
            skipped:*)     printf '  %-16s skipped by you\n' "$id" ;;
            unavailable:*) printf '  %-16s not available — %s\n' "$id" "${state#unavailable:}" ;;
        esac
    done < "$L/manifest.txt"
    echo ""
    echo "## Failing tests"
    # Leading whitespace is allowed: a lane that drives pytest itself reports a
    # failure indented inside its own traceback, and anchoring at column 0
    # silently drops it.
    named=0
    if grep -haE '^[[:space:]]*(FAILED|ERROR) tests/' "$L"/*.txt 2>/dev/null \
        | sed -e 's/^[[:space:]]*//' -e 's/ - .*//' | sort -u | grep .; then
        named=1
    else
        echo "   (none)"
    fi

    # A lane can fail without printing a pytest node id — it may run a checker of
    # its own, or die before pytest starts. Its exit code is then the only
    # evidence there is, so the verdict reads it rather than the log text.
    red=""
    while IFS='|' read -r id stream est label state; do
        [ "$state" = "run" ] || continue
        rc="$(cat "$L/$id.rc" 2>/dev/null || echo 0)"
        [ "$rc" = "0" ] || red="$red  $id (exit $rc)"$'\n'
    done < "$L/manifest.txt"

    echo ""
    echo "## Lanes that ended non-zero"
    if [ -n "$red" ]; then printf '%s' "$red"; else echo "   (none)"; fi

    echo ""
    if [ "$named" = "1" ] || [ -n "$red" ]; then
        echo "RESULT: FAILURES — the lists above need a decision."
    else
        echo "RESULT: NO FAILURES in the lanes that ran."
    fi
    echo ""
    echo "logs: $L/<lane>.txt"
} >> "$SUMMARY"

echo "======================================================================"
sed -n '/^===* RESULT/,$p' "$SUMMARY"
