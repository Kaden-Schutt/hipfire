#!/usr/bin/env bash
# speed-gate.sh — MQ4 prefill/decode speed regression gate.
#
# Runs bench_qwen35_mq4 on each available MQ4 model and compares
# prefill_tok_s + gen_tok_s against the committed baselines in
# tests/speed-baselines/<arch>.txt. ANY regression below baseline × (1 - tolerance)
# is a PERFORMANCE BUG.
#
# The current MQ4 numbers are the permanent GROUND FLOOR — optimizations may
# RAISE the baseline (via --update-baselines) but nothing may ship that lowers
# it without explicit justification.
#
# Modes:
#   ./scripts/speed-gate.sh              # all available sizes
#   ./scripts/speed-gate.sh --fast       # 4B only (~25s, used by pre-commit)
#   ./scripts/speed-gate.sh --update-baselines  # record current speeds as new floor
#   ./scripts/speed-gate.sh --tolerance 0.1     # allow up to 10% regression (default 0.05)
#   ./scripts/speed-gate.sh --verbose    # show full bench output on fail
#
# Exit codes:
#   0   all metrics within tolerance
#   1   regression detected
#   2   build or environment error
#
# Each metric is best-of-2 runs to smooth JIT/thermal noise.

set -u
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
EXE="./target/release/examples/bench_qwen35_mq4"
DFLASH_EXE="./target/release/examples/dflash_spec_demo"
DFLASH_PROMPT_FILE="benchmarks/prompts/lru_cache_pep8_strict.txt"
DFLASH_PROMPT_MERGE_SORT="benchmarks/prompts/merge_sort_thinking_off.txt"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Arch detection ────────────────────────────────────────────────────────
BASELINE_ARCH=""
if [ -n "${HIPFIRE_BASELINE_ARCH:-}" ]; then
    BASELINE_ARCH="$HIPFIRE_BASELINE_ARCH"
else
    for probe in amdgpu-arch offload-arch \
                 /opt/rocm/bin/amdgpu-arch /opt/rocm/bin/offload-arch \
                 /opt/rocm/llvm/bin/amdgpu-arch; do
        if command -v "$probe" >/dev/null 2>&1 || [ -x "$probe" ]; then
            BASELINE_ARCH="$("$probe" 2>/dev/null | head -1)"
            if [ -n "$BASELINE_ARCH" ]; then break; fi
        fi
    done
    if [ -z "$BASELINE_ARCH" ]; then
        for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            [ -f "$node_props" ] || continue
            ver=$(awk '/gfx_target_version/ {print $2; exit}' "$node_props" 2>/dev/null || true)
            case "$ver" in
                90006)          BASELINE_ARCH="gfx906";  break ;;
                90008)          BASELINE_ARCH="gfx908";  break ;;
                100100)         BASELINE_ARCH="gfx1010"; break ;;
                100300|100302)  BASELINE_ARCH="gfx1030"; break ;;
                110000|110001)  BASELINE_ARCH="gfx1100"; break ;;
                110501)         BASELINE_ARCH="gfx1151"; break ;;
                120000)         BASELINE_ARCH="gfx1200"; break ;;
                120001)         BASELINE_ARCH="gfx1201"; break ;;
            esac
        done
    fi
    if [ -z "$BASELINE_ARCH" ] && command -v rocminfo >/dev/null 2>&1; then
        BASELINE_ARCH="$(rocminfo 2>/dev/null | awk '/^  Name:/ && $2 ~ /^gfx/ {print $2; exit}')"
    fi
fi
case "${HSA_OVERRIDE_GFX_VERSION:-}" in
    9.0.6|9.0) BASELINE_ARCH="gfx906" ;;
    10.1.0|10.1) BASELINE_ARCH="gfx1010" ;;
    10.3.0|10.3) BASELINE_ARCH="gfx1030" ;;
    11.0.0|11.0) BASELINE_ARCH="gfx1100" ;;
esac
if [ -z "$BASELINE_ARCH" ]; then
    echo "speed-gate: cannot detect GPU arch — set HIPFIRE_BASELINE_ARCH=gfxNNNN" >&2
    exit 2
fi

BASELINE_FILE="tests/speed-baselines/${BASELINE_ARCH}.txt"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-/home/kaden/ClaudeCode/autorocm/hipfire/models}"

FAST=0
UPDATE=0
VERBOSE=0
TOLERANCE="0.05"
while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1 ;;
        --update|--update-baselines) UPDATE=1 ;;
        --tolerance) TOLERANCE="$2"; shift ;;
        --verbose|-v) VERBOSE=1 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

color() {
    if [ -t 1 ]; then
        case "$1" in
            green)  printf '\033[32m%s\033[0m' "$2" ;;
            red)    printf '\033[31m%s\033[0m' "$2" ;;
            yellow) printf '\033[33m%s\033[0m' "$2" ;;
            bold)   printf '\033[1m%s\033[0m'  "$2" ;;
            *)      printf '%s' "$2" ;;
        esac
    else
        printf '%s' "$2"
    fi
}

# Model sizes to test. 4B is fastest reliable signal; 9B/27B catch BW
# regressions; 0.8B catches launch-overhead regressions.
if [ "$FAST" -eq 1 ]; then
    SIZES=("4b")
else
    SIZES=("0.8b" "4b" "9b" "27b")
fi

ensure_build() {
    if [ ! -x "$EXE" ]; then
        echo "Building bench_qwen35_mq4 (release)..."
        cargo build --release -p engine --example bench_qwen35_mq4 --features deltanet 2>&1 \
            | grep -E '^(error|   Compiling)' | tail -5
        if [ ! -x "$EXE" ]; then
            color red "BUILD FAILED"; echo
            exit 2
        fi
    fi
    # DFlash gate is opt-in by default — only requires dflash_spec_demo + the
    # 27B target + draft + prompt file. Skip silently if unavailable.
}

# DFlash LRU-code gate. Echoes "tok_s tau" or one of: MISSING_TARGET,
# MISSING_DRAFT, MISSING_PROMPT, MISSING_BIN, CRASH.
# Canonical bench: 27B-3.5 LRU code @ max=120, asym3 KV, no chatml, no
# adaptive-b. Default flags include prompt_normalize=true (engine default
# since 2026-04-26). Best-of-2 (within 5% jitter on hot-cache).
bench_dflash_27b_lru() {
    # Models may live at $MODELS_DIR (project-local) or ~/.hipfire/models/
    # (install). Drafts in particular are usually only at the install path.
    local target draft
    for dir in "$MODELS_DIR" "$HOME/.hipfire/models"; do
        [ -f "$dir/qwen3.5-27b.mq4" ] && [ -z "${target:-}" ] && target="$dir/qwen3.5-27b.mq4"
        # Registry filename is `qwen35-27b-dflash-mq4.hfq` (cli/index.ts:469).
        # The legacy `qwen35-27b-dflash.mq4` was renamed when the registry
        # standardized on `<base>-<quant>.hfq`. Accept both for back-compat
        # with older pulls; #61 reporter hit MISSING_DRAFT here.
        for cand in "$dir/qwen35-27b-dflash-mq4.hfq" "$dir/qwen35-27b-dflash.mq4"; do
            [ -f "$cand" ] && [ -z "${draft:-}" ] && draft="$cand" && break
        done
    done
    [ ! -x "$DFLASH_EXE" ] && { echo "MISSING_BIN"; return; }
    [ -z "${target:-}" ] && { echo "MISSING_TARGET"; return; }
    [ -z "${draft:-}" ] && { echo "MISSING_DRAFT"; return; }
    [ ! -f "$DFLASH_PROMPT_FILE" ] && { echo "MISSING_PROMPT"; return; }
    local best_t=0 best_tau=0
    local prompt
    prompt=$(cat "$DFLASH_PROMPT_FILE")
    for run in 1 2 3; do
        local out
        out=$(HIPFIRE_DPM_WARMUP_SECS=10 "$DFLASH_EXE" \
            --target "$target" --draft "$draft" \
            --prompt "$prompt" \
            --max 120 --no-chatml --kv-mode asym3 2>&1)
        local t tau
        t=$(echo "$out" | sed -nE 's/.*emitted: [0-9]+ tokens in [0-9.]+s\s+\(([0-9.]+) tok\/s\).*/\1/p' | tail -1)
        tau=$(echo "$out" | sed -nE 's/.*τ=([0-9.]+).*/\1/p' | tail -1)
        if [ -n "$t" ] && awk "BEGIN { exit !($t > $best_t) }"; then
            best_t="$t"
            best_tau="$tau"
        fi
    done
    if [ "$best_t" = "0" ]; then echo "CRASH"; else echo "$best_t $best_tau"; fi
}

# DFlash high-acceptance gate — merge_sort thinking-OFF, max=256.
# Anchors the high-τ ceiling (real production-shape bounded code) since
# the LRU max=120 anchor above is loop-edge. Echoes "tok_s tau ttft_ms"
# or one of MISSING_*/CRASH. Best-of-3.
#
# Generic core: takes target + draft basenames, runs the bench, returns
# best of 3. Per-size wrappers below pin the model paths.
_bench_dflash_merge_sort_core() {
    local target_name="$1" draft_name="$2"
    local target draft
    for dir in "$MODELS_DIR" "$HOME/.hipfire/models"; do
        [ -f "$dir/$target_name" ] && [ -z "${target:-}" ] && target="$dir/$target_name"
        [ -f "$dir/$draft_name" ]  && [ -z "${draft:-}" ]  && draft="$dir/$draft_name"
    done
    [ ! -x "$DFLASH_EXE" ] && { echo "MISSING_BIN"; return; }
    [ -z "${target:-}" ] && { echo "MISSING_TARGET"; return; }
    [ -z "${draft:-}" ] && { echo "MISSING_DRAFT"; return; }
    [ ! -f "$DFLASH_PROMPT_MERGE_SORT" ] && { echo "MISSING_PROMPT"; return; }
    local best_t=0 best_tau=0 best_ttft=0
    local prompt
    prompt=$(cat "$DFLASH_PROMPT_MERGE_SORT")
    for run in 1 2 3; do
        local out
        out=$(HIPFIRE_DPM_WARMUP_SECS=10 "$DFLASH_EXE" \
            --target "$target" --draft "$draft" \
            --prompt "$prompt" \
            --max 256 --no-chatml --kv-mode asym3 2>&1)
        local t tau ttft
        t=$(echo "$out" | sed -nE 's/^decode_tok_s: ([0-9.]+).*/\1/p' | tail -1)
        tau=$(echo "$out" | sed -nE 's/^decode_tau: ([0-9.]+).*/\1/p' | tail -1)
        ttft=$(echo "$out" | sed -nE 's/^ttft_ms: ([0-9.]+).*/\1/p' | tail -1)
        if [ -n "$t" ] && awk "BEGIN { exit !($t > $best_t) }"; then
            best_t="$t"; best_tau="$tau"; best_ttft="$ttft"
        fi
    done
    if [ "$best_t" = "0" ]; then echo "CRASH"; else echo "$best_t $best_tau $best_ttft"; fi
}

bench_dflash_27b_merge_sort() {
    # Registry filename is `qwen35-27b-dflash-mq4.hfq`; the legacy
    # `qwen35-27b-dflash.mq4` was renamed when the registry standardized.
    # Try the current name first, then the legacy. Reported in #61.
    if _draft_exists "qwen35-27b-dflash-mq4.hfq"; then
        _bench_dflash_merge_sort_core "qwen3.5-27b.mq4" "qwen35-27b-dflash-mq4.hfq"
    else
        _bench_dflash_merge_sort_core "qwen3.5-27b.mq4" "qwen35-27b-dflash.mq4"
    fi
}

# Helper: does a draft file with this basename exist in any known model dir?
_draft_exists() {
    local name="$1"
    for dir in "$MODELS_DIR" "$HOME/.hipfire/models"; do
        [ -f "$dir/$name" ] && return 0
    done
    return 1
}

bench_dflash_9b_merge_sort() {
    _bench_dflash_merge_sort_core "qwen3.5-9b.mq4" "qwen35-9b-dflash-mq4.hfq"
}

# Run bench_qwen35_mq4 once at a given prefill size.
# Echoes "prefill_tok_s decode_tok_s" or empty on failure.
bench_run() {
    local size="$1"
    local prefill="$2"
    local model_path="$MODELS_DIR/qwen3.5-${size}.mq4"
    # givens4 was removed in the asym migration; asym3 is the current default
    # (5.5× compression, same speed regime as the old givens4 baseline).
    local env_prefix="HIPFIRE_KV_MODE=asym3"
    # 0.8B has a known hipGraph panic; use plain path.
    if [ "$size" != "0.8b" ]; then
        env_prefix="$env_prefix HIPFIRE_GRAPH=1"
    fi
    local out
    out=$(eval "$env_prefix $EXE $model_path --prefill $prefill --warmup 5 --gen 50" 2>&1 | grep "^SUMMARY" | tail -1)
    local p d
    p=$(echo "$out" | sed -nE 's/.*prefill_tok_s=([0-9.]+).*/\1/p')
    d=$(echo "$out" | sed -nE 's/.*gen_tok_s=([0-9.]+).*/\1/p')
    if [ -n "$p" ] && [ -n "$d" ]; then
        echo "$p $d"
    fi
}

# Best-of-2 runs at a given prefill size.
bench_best_of_2() {
    local size="$1"
    local prefill="$2"
    local model_path="$MODELS_DIR/qwen3.5-${size}.mq4"
    if [ ! -f "$model_path" ]; then
        echo "MISSING"
        return
    fi
    local best_p=0 best_d=0
    for run in 1 2; do
        local r
        r=$(bench_run "$size" "$prefill")
        if [ -z "$r" ]; then continue; fi
        read -r p d <<< "$r"
        if awk "BEGIN { exit !($p > $best_p) }"; then best_p="$p"; fi
        if awk "BEGIN { exit !($d > $best_d) }"; then best_d="$d"; fi
    done
    if [ "$best_p" = "0" ] || [ "$best_d" = "0" ]; then
        echo "CRASH"
    else
        echo "$best_p $best_d"
    fi
}

check_metric() {
    local label="$1" baseline="$2" observed="$3"
    local min
    min=$(awk "BEGIN { printf \"%.1f\", $baseline * (1 - $TOLERANCE) }")
    local pct
    pct=$(awk "BEGIN { printf \"%+.1f\", ($observed - $baseline) / $baseline * 100 }")

    if awk "BEGIN { exit !($observed >= $min) }"; then
        if awk "BEGIN { exit !($observed > $baseline * 1.02) }"; then
            printf "  %-30s " "$label"
            color green "FAST"
            printf "  baseline=%-7.1f observed=%-7.1f (%s%%)\n" "$baseline" "$observed" "$pct"
        else
            printf "  %-30s " "$label"
            color green "OK  "
            printf "  baseline=%-7.1f observed=%-7.1f (%s%%)\n" "$baseline" "$observed" "$pct"
        fi
        return 0
    else
        printf "  %-30s " "$label"
        color red "FAIL"
        printf "  baseline=%-7.1f observed=%-7.1f (%s%% — below floor %.1f)\n" \
            "$baseline" "$observed" "$pct" "$min"
        return 1
    fi
}

# -------- main --------

if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "speed-gate" || { color red "could not acquire GPU lock"; echo; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

ensure_build

if [ "$UPDATE" -eq 1 ]; then
    color bold "=== Capturing MQ4 speed baselines (ground floor) ==="; echo
    color yellow "WARNING: This replaces the speed floor. The new numbers become"; echo
    color yellow "         the permanent minimum — only raise, never lower."; echo
    echo
    tmpfile=$(mktemp)
    {
        echo "# hipfire speed-gate baseline — $BASELINE_ARCH"
        echo "# Captured $(date -u +%Y-%m-%d) against commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "# Config: KV=asym3, HIPFIRE_GRAPH=1 for 4B/9B/27B (0.8B has known hipGraph bug)"
        echo "# Tolerance: 0.05 (fail if any metric drops below baseline × (1 - tolerance))"
        echo "#"
        echo "# Measurements: best-of-2 via bench_qwen35_mq4. Two prefill sizes per model:"
        echo "#   pp32  — short-context / launch-overhead regression detector"
        echo "#   pp128 — realistic prompt / GEMM-efficiency detector"
        echo "# Decode numbers are taken from the pp32 run (warmup+50 gen is enough to settle)."
        echo "# GROUND FLOOR — no commit may regress below these numbers."
        echo ""
    } > "$tmpfile"

    for size in "${SIZES[@]}"; do
        for pf in 32 128; do
            printf "  %-5s pp%-3s " "$size" "$pf"
            result=$(bench_best_of_2 "$size" "$pf")
            case "$result" in
                MISSING) color yellow "SKIP"; echo " (model not present)"; continue ;;
                CRASH)   color red "CRASH"; echo " (bench failed both runs)"; continue ;;
            esac
            read -r p d <<< "$result"
            printf "prefill=%7.1f  decode=%7.1f tok/s\n" "$p" "$d"
            {
                echo "${size}_mq4_pp${pf}_prefill_tok_s=${p}"
                # Only emit decode once per size (from the pp32 run, which is cheaper).
                [ "$pf" = "32" ] && echo "${size}_mq4_gen_tok_s=${d}"
            } >> "$tmpfile"
        done
        echo "" >> "$tmpfile"
    done

    # DFlash canonical gate (added 2026-04-26 per perf-regression-recovery).
    # 27B-3.5 LRU code @ max=120, default flags (prompt_normalize=true since
    # 2026-04-26). Catches the regression class that PR #32 dead-wmma-kernels
    # missed because it only affected DFlash verify, not AR prefill.
    printf "  27B-3.5 DFlash LRU code  "
    dflash_result=$(bench_dflash_27b_lru)
    case "$dflash_result" in
        MISSING_*|CRASH) color yellow "$dflash_result"; echo "" ;;
        *)
            read -r dt dtau <<< "$dflash_result"
            printf "tok/s=%-7.1f τ=%s\n" "$dt" "$dtau"
            {
                echo "27b_3.5_dflash_lru_code_tok_s=${dt}"
                echo "27b_3.5_dflash_lru_code_tau=${dtau}"
            } >> "$tmpfile"
            ;;
    esac

    printf "  27B-3.5 DFlash merge_sort  "
    ms_result=$(bench_dflash_27b_merge_sort)
    case "$ms_result" in
        MISSING_*|CRASH) color yellow "$ms_result"; echo "" ;;
        *)
            read -r mst mstau msttft <<< "$ms_result"
            printf "tok/s=%-7.1f τ=%s ttft_ms=%s\n" "$mst" "$mstau" "$msttft"
            {
                echo "27b_3.5_dflash_merge_sort_tok_s=${mst}"
                echo "27b_3.5_dflash_merge_sort_tau=${mstau}"
                echo "27b_3.5_dflash_merge_sort_ttft_ms=${msttft}"
            } >> "$tmpfile"
            ;;
    esac

    printf "  9B-3.5 DFlash merge_sort   "
    ms9_result=$(bench_dflash_9b_merge_sort)
    case "$ms9_result" in
        MISSING_*|CRASH) color yellow "$ms9_result"; echo "" ;;
        *)
            read -r mst9 mstau9 msttft9 <<< "$ms9_result"
            printf "tok/s=%-7.1f τ=%s ttft_ms=%s\n" "$mst9" "$mstau9" "$msttft9"
            {
                echo "9b_3.5_dflash_merge_sort_tok_s=${mst9}"
                echo "9b_3.5_dflash_merge_sort_tau=${mstau9}"
                echo "9b_3.5_dflash_merge_sort_ttft_ms=${msttft9}"
            } >> "$tmpfile"
            ;;
    esac
    echo "" >> "$tmpfile"

    mkdir -p "$(dirname "$BASELINE_FILE")"
    cp "$tmpfile" "$BASELINE_FILE"
    rm -f "$tmpfile"
    echo
    color bold "Baselines written to $BASELINE_FILE"; echo
    echo "Review with:  git diff $BASELINE_FILE"
    echo "Then commit them alongside your code change."
    exit 0
fi

if [ ! -f "$BASELINE_FILE" ]; then
    color yellow "speed-gate: no baseline file at $BASELINE_FILE"; echo
    echo "  generate with: HIPFIRE_BASELINE_ARCH=$BASELINE_ARCH ./scripts/speed-gate.sh --update-baselines"
    exit 2
fi

color bold "=== MQ4 Speed Gate (tolerance ${TOLERANCE}) ==="; echo
echo "baseline: $BASELINE_FILE"
echo

pass=0
fail=0
skip=0

for size in "${SIZES[@]}"; do
    # --fast only measures pp32 for speed; otherwise pp32 + pp128.
    if [ "$FAST" -eq 1 ]; then
        prefills=("32")
    else
        prefills=("32" "128")
    fi

    # Always measure decode from the pp32 run.
    decode_observed=""

    for pf in "${prefills[@]}"; do
        result=$(bench_best_of_2 "$size" "$pf")
        case "$result" in
            MISSING)
                printf "  %-5s pp%-3s " "$size" "$pf"
                color yellow "SKIP"; echo " (model not present)"
                skip=$((skip+1))
                continue
                ;;
            CRASH)
                printf "  %-5s pp%-3s " "$size" "$pf"
                color red "CRASH"; echo " (bench failed)"
                fail=$((fail+1))
                continue
                ;;
        esac
        read -r p d <<< "$result"
        [ "$pf" = "32" ] && decode_observed="$d"

        p_base=$(grep -oE "^${size}_mq4_pp${pf}_prefill_tok_s=[0-9.]+" "$BASELINE_FILE" | cut -d= -f2)

        if [ -z "$p_base" ]; then
            printf "  %-5s pp%-3s " "$size" "$pf"
            color yellow "NO BASELINE"; echo " (add with --update-baselines)"
            skip=$((skip+1))
            continue
        fi

        check_metric "${size} MQ4 pp${pf} prefill" "$p_base" "$p"
        case $? in 0) pass=$((pass+1)) ;; *) fail=$((fail+1)) ;; esac
    done

    # Decode check (once per size).
    if [ -n "$decode_observed" ]; then
        d_base=$(grep -oE "^${size}_mq4_gen_tok_s=[0-9.]+" "$BASELINE_FILE" | cut -d= -f2)
        if [ -z "$d_base" ]; then
            printf "  %-5s decode " "$size"
            color yellow "NO BASELINE"; echo " (add with --update-baselines)"
            skip=$((skip+1))
        else
            check_metric "${size} MQ4 decode" "$d_base" "$decode_observed"
            case $? in 0) pass=$((pass+1)) ;; *) fail=$((fail+1)) ;; esac
        fi
    fi
done

# DFlash canonical gate — 27B-3.5 LRU code, max=120, default flags.
# Added 2026-04-26 after PR #32 dead-wmma-kernels passed AR-only speed-gate
# while regressing 27B DFlash by 40%. AR-only gates are insufficient for
# perf protection. Skipped in --fast mode (DFlash bench takes ~5s × 3 runs).
if [ "$FAST" -eq 0 ]; then
    dflash_result=$(bench_dflash_27b_lru)
    case "$dflash_result" in
        MISSING_*)
            printf "  27B-3.5 DFlash LRU code  "
            color yellow "SKIP"; echo " ($dflash_result)"
            skip=$((skip+1))
            ;;
        CRASH)
            printf "  27B-3.5 DFlash LRU code  "
            color red "CRASH"; echo
            fail=$((fail+1))
            ;;
        *)
            read -r dt dtau <<< "$dflash_result"
            dflash_base=$(grep -oE "^27b_3.5_dflash_lru_code_tok_s=[0-9.]+" "$BASELINE_FILE" | cut -d= -f2)
            if [ -z "$dflash_base" ]; then
                printf "  27B-3.5 DFlash LRU code  "
                color yellow "NO BASELINE"; echo " (add with --update-baselines; observed=${dt} τ=${dtau})"
                skip=$((skip+1))
            else
                check_metric "27B-3.5 DFlash LRU code" "$dflash_base" "$dt"
                case $? in 0) pass=$((pass+1)) ;; *) fail=$((fail+1)) ;; esac
            fi
            ;;
    esac

    # Second DFlash anchor: merge_sort thinking-OFF (high-τ ceiling).
    ms_result=$(bench_dflash_27b_merge_sort)
    case "$ms_result" in
        MISSING_*)
            printf "  27B-3.5 DFlash merge_sort  "
            color yellow "SKIP"; echo " ($ms_result)"
            skip=$((skip+1))
            ;;
        CRASH)
            printf "  27B-3.5 DFlash merge_sort  "
            color red "CRASH"; echo
            fail=$((fail+1))
            ;;
        *)
            read -r mst mstau msttft <<< "$ms_result"
            ms_base=$(grep -oE "^27b_3.5_dflash_merge_sort_tok_s=[0-9.]+" "$BASELINE_FILE" | cut -d= -f2)
            if [ -z "$ms_base" ]; then
                printf "  27B-3.5 DFlash merge_sort  "
                color yellow "NO BASELINE"; echo " (add with --update-baselines; observed=${mst} τ=${mstau} ttft=${msttft}ms)"
                skip=$((skip+1))
            else
                check_metric "27B-3.5 DFlash merge_sort" "$ms_base" "$mst"
                case $? in 0) pass=$((pass+1)) ;; *) fail=$((fail+1)) ;; esac
            fi
            ;;
    esac

    # Third DFlash anchor: 9B merge_sort (small-model high-τ ceiling).
    ms9_result=$(bench_dflash_9b_merge_sort)
    case "$ms9_result" in
        MISSING_*)
            printf "  9B-3.5 DFlash merge_sort   "
            color yellow "SKIP"; echo " ($ms9_result)"
            skip=$((skip+1))
            ;;
        CRASH)
            printf "  9B-3.5 DFlash merge_sort   "
            color red "CRASH"; echo
            fail=$((fail+1))
            ;;
        *)
            read -r mst9 mstau9 msttft9 <<< "$ms9_result"
            ms9_base=$(grep -oE "^9b_3.5_dflash_merge_sort_tok_s=[0-9.]+" "$BASELINE_FILE" | cut -d= -f2)
            if [ -z "$ms9_base" ]; then
                printf "  9B-3.5 DFlash merge_sort   "
                color yellow "NO BASELINE"; echo " (add with --update-baselines; observed=${mst9} τ=${mstau9} ttft=${msttft9}ms)"
                skip=$((skip+1))
            else
                check_metric "9B-3.5 DFlash merge_sort" "$ms9_base" "$mst9"
                case $? in 0) pass=$((pass+1)) ;; *) fail=$((fail+1)) ;; esac
            fi
            ;;
    esac
fi

echo
if [ "$fail" -eq 0 ]; then
    if [ "$pass" -eq 0 ]; then
        color yellow "=== NO METRICS CHECKED (${skip} skipped) ==="; echo
        exit 0
    fi
    color green "=== ${pass} METRICS PASSED"
    [ "$skip" -gt 0 ] && printf ", %d SKIPPED" "$skip"
    color green " ==="; echo
    exit 0
fi

color red "=== ${fail} METRIC(S) REGRESSED, ${pass} PASSED, ${skip} SKIPPED ==="; echo
echo
echo "This is a PERFORMANCE REGRESSION. The current MQ4 stats are the ground"
echo "floor — nothing may ship below them."
echo
echo "Either:"
echo "  1. Fix the regression before committing."
echo "  2. If the change intentionally trades speed for something else (e.g.,"
echo "     correctness on a new codepath), justify the tradeoff AND re-baseline."
echo
echo "Re-bench with:"
echo "  HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \\"
echo "    ./target/release/examples/bench_qwen35_mq4 <model>"
echo
exit 1
