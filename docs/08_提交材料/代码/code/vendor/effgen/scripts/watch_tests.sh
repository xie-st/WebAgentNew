#!/usr/bin/env bash
################################################################################
# effGen — live status for a test run started by scripts/run_tests.sh.
#
#   scripts/watch_tests.sh              refresh every 10s
#   REFRESH=5 scripts/watch_tests.sh    refresh every 5s
#
# It only reads .test-run/, so starting it, stopping it and restarting it cannot
# affect the run. Ctrl-C ends the display, not the tests.
#
# Lanes you chose not to run are listed too, marked "skipped by you", so the
# picture is of the whole suite rather than only the part in flight.
#
# Percentages come from pytest's own progress markers. Lanes that print no
# progress — the linters, the installers, the scanners — show elapsed time
# against a fixed budget and are marked `est`, so an estimate is never mistaken
# for a measurement.
################################################################################
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
L="$REPO/.test-run"
REFRESH="${REFRESH:-10}"

bar() {
    local pct="$1" w="$2" i out=""
    [ "$pct" -lt 0 ] && pct=0; [ "$pct" -gt 100 ] && pct=100
    local filled=$(( pct * w / 100 ))
    for ((i = 0; i < w; i++)); do
        if [ "$i" -lt "$filled" ]; then out+="█"; else out+="░"; fi
    done
    printf '%s' "$out"
}

hms() {
    local s="$1"
    if   [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
    elif [ "$s" -ge 60 ];   then printf '%dm' $((s/60))
    else printf '%ds' "$s"; fi
}

# `$( )` strips trailing newlines, so a bare `OUT+=$(printf '...\n')` would
# collapse the whole display onto one line. Put the newline back.
add() { OUT+="$(printf "$@")"$'\n'; }

while :; do
    if [ ! -f "$L/manifest.txt" ]; then
        printf '\033[H\033[J\n  No test run found.\n\n  Start one with:  scripts/run_tests.sh\n'
        sleep "$REFRESH"; continue
    fi

    now=$(date +%s)
    run_start="$(cat "$L/run.start" 2>/dev/null || echo "$now")"
    elapsed=$(( now - run_start ))
    OUT=""
    add '\033[1meffGen — test run\033[0m    elapsed %s    %s' "$(hms $elapsed)" "$(date +%H:%M:%S)"
    add '══════════════════════════════════════════════════════════════════════════════'

    done_n=0; run_n=0; queue_n=0; skip_n=0; bad_n=0; last_stream=""
    declare -A LEFT=()
    order=""

    while IFS='|' read -r id stream est label state; do
        [ -z "$id" ] && continue
        log="$L/$id.txt"; rcf="$L/$id.rc"; est_s=$(( est * 60 ))
        [[ " $order " == *" $stream "* ]] || order="$order $stream"
        [ -z "${LEFT[$stream]:-}" ] && LEFT[$stream]=0

        if [ "$stream" != "$last_stream" ]; then
            add '\n\033[1;36m  stream %s\033[0m' "$stream"
            last_stream="$stream"
        fi

        case "$state" in
            skipped:*)
                skip_n=$((skip_n + 1))
                add '  \033[2m%-18s %s  skipped by you\033[0m' "$id" "$(bar 0 18)"
                continue ;;
            unavailable:*)
                skip_n=$((skip_n + 1))
                add '  \033[2m%-18s %s  not available — %s\033[0m' \
                    "$id" "$(bar 0 18)" "${state#unavailable:}"
                continue ;;
        esac

        if [ -f "$rcf" ]; then
            rc="$(cat "$rcf")"; mins="$(cat "$L/$id.min" 2>/dev/null || echo '?')"
            tot="$(grep -aoE '[0-9]+ (passed|failed)[a-z0-9, ]*' "$log" 2>/dev/null | tail -1)"
            [ -z "$tot" ] && tot="$(grep -av '^[[:space:]]*$' "$log" 2>/dev/null | tail -1)"
            done_n=$((done_n + 1))
            case "$rc" in
                0)       tag='\033[32mpassed \033[0m' ;;
                124|137) tag='\033[31mTIMEOUT\033[0m'; bad_n=$((bad_n + 1)) ;;
                *)       tag='\033[31mFAILED \033[0m'; bad_n=$((bad_n + 1)) ;;
            esac
            add "  %-18s %s $tag %4sm  %s" "$id" "$(bar 100 18)" "$mins" "${tot:0:38}"
            continue
        fi

        if [ ! -f "$L/$id.start" ]; then
            queue_n=$((queue_n + 1))
            LEFT[$stream]=$(( LEFT[$stream] + est_s ))
            add '  \033[2m%-18s %s  queued          ~%s est\033[0m' \
                "$id" "$(bar 0 18)" "$(hms $est_s)"
            continue
        fi

        run_n=$((run_n + 1))
        st="$(cat "$L/$id.start")"
        le=$(( now - st )); [ "$le" -lt 0 ] && le=0
        pct="$(grep -aoE '\[ *[0-9]+%\]' "$log" 2>/dev/null | tail -1 | tr -dc '0-9')"
        if [ -n "${pct:-}" ] && [ "$pct" -gt 0 ]; then
            left=$(( le * (100 - pct) / pct )); note="~$(hms $left) left"
        else
            pct=$(( est_s > 0 ? le * 100 / est_s : 0 )); [ "$pct" -gt 99 ] && pct=99
            left=$(( est_s - le )); [ "$left" -lt 60 ] && left=60
            note="~$(hms $left) est"
        fi
        LEFT[$stream]=$(( LEFT[$stream] + left ))
        prog="$(grep -aoE '^[.sFExX]+' "$log" 2>/dev/null | tr -d '\n')"
        info="${prog:+${#prog} run}"
        nred="$(printf '%s' "$prog" | tr -cd 'FE' | wc -c)"
        [ "$nred" -gt 0 ] && info="$info, ${nred}F"
        add '  %-18s %s \033[33m%3d%%\033[0m %4sm  %-14s %s' \
            "$id" "$(bar "$pct" 18)" "$pct" "$((le / 60))" "${info:-running}" "$note"
    done < "$L/manifest.txt"

    # Streams run at the same time, so what is left overall is the slowest of
    # them, not the sum. The offline parts inside stream A also run at the same
    # time, so that stream's figure is its slowest part.
    eta=0; slowest="-"
    for s in $order; do
        v="${LEFT[$s]:-0}"
        if [ "$s" = "A" ]; then
            v=0
            while IFS='|' read -r id stream est label state; do
                [ "$stream" = "A" ] && [ "$state" = "run" ] || continue
                [ -f "$L/$id.rc" ] && continue
                if [ -f "$L/$id.start" ]; then
                    le=$(( now - $(cat "$L/$id.start") ))
                    p="$(grep -aoE '\[ *[0-9]+%\]' "$L/$id.txt" 2>/dev/null | tail -1 | tr -dc '0-9')"
                    if [ -n "${p:-}" ] && [ "$p" -gt 0 ]; then r=$(( le * (100 - p) / p )); else r=$(( est * 60 )); fi
                else r=$(( est * 60 )); fi
                [ "$r" -gt "$v" ] && v="$r"
            done < "$L/manifest.txt"
        fi
        if [ "$v" -gt "$eta" ]; then eta="$v"; slowest="$s"; fi
    done

    total=$(( done_n + run_n + queue_n ))
    overall=$(( total > 0 ? done_n * 100 / total : 0 ))
    reds="$(grep -hac '^FAILED' "$L"/*.txt 2>/dev/null | paste -sd+ - | bc 2>/dev/null)"

    add '──────────────────────────────────────────────────────────────────────────────'
    add '  \033[32m%d done\033[0m · \033[33m%d running\033[0m · %d queued · %d skipped   (of %d selected)' \
        "$done_n" "$run_n" "$queue_n" "$skip_n" "$total"

    if [ "$run_n" -eq 0 ] && [ "$queue_n" -eq 0 ] && [ "$done_n" -gt 0 ]; then
        add '  \033[1;32mFINISHED\033[0m in %s — read %s' "$(hms $elapsed)" "$L/summary.txt"
        printf '\033[H\033[J%s' "$OUT"
        grep -aE '^RESULT:' "$L/summary.txt" 2>/dev/null
        break
    fi

    add '  [%s] \033[1m%d%%\033[0m of lanes done · \033[1m~%s remaining\033[0m (stream %s is the long pole)' \
        "$(bar "$overall" 30)" "$overall" "$(hms $eta)" "$slowest"
    add '  \033[2mfailing tests so far: %s · lanes with a non-zero exit: %s\033[0m' "${reds:-0}" "$bad_n"
    add '  \033[2mCtrl-C stops this display only — the run keeps going.\033[0m'
    printf '\033[H\033[J%s' "$OUT"
    sleep "$REFRESH"
done
