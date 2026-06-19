#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/hipfire-lds-dim-bisect}"
DIM="${DIM:-N}"
LOW="${LOW:-1024}"
HIGH="${HIGH:-3072}"
ARCH="${ARCH:-gfx1103}"
VARIANT="${VARIANT:-tile6_synth}"
MODE="${MODE:-full}"
LAUNCHES="${N_LAUNCH:-56}"
M="${M:-512}"
N="${N:-3072}"
K="${K:-3072}"
K_LIMIT="${K_LIMIT:-0}"

SRC="${ROOT}/scripts/lds_gemm_standalone_probe.hip"

if [[ ! -f "$SRC" ]]; then
    echo "missing source: $SRC" >&2
    exit 1
fi

if [[ "$DIM" != "N" && "$DIM" != "K" && "$DIM" != "M" ]]; then
    echo "DIM must be N, K, or M" >&2
    exit 1
fi

mkdir -p "$OUT"
bin="$OUT/lds_gemm_standalone_probe"
temps="$OUT/save-temps"
mkdir -p "$temps"

cp "$SRC" "$OUT/lds_gemm_standalone_probe.hip"

/opt/rocm/bin/hipcc -O3 --offload-arch="$ARCH" -save-temps=obj \
    "$SRC" -o "$bin" >"$OUT/build.log" 2>&1

run_case() {
    local value="$1"
    local label="$2"
    local case_dir="$OUT/$label"
    mkdir -p "$case_dir"

    local m="$M"
    local n="$N"
    local k="$K"
    case "$DIM" in
        M) m="$value" ;;
        N) n="$value" ;;
        K) k="$value" ;;
    esac

    echo "[bisect] label=$label ${DIM}=$value shape=${m}x${n}x${k}" >&2
    dmesg --ctime >"$case_dir/dmesg.before.txt" 2>&1 || true
    set +e
    "$bin" "$VARIANT" "$MODE" "$LAUNCHES" "$m" "$n" "$k" "$K_LIMIT" \
        >"$case_dir/run.log" 2>&1
    rc=$?
    set -e
    dmesg --ctime >"$case_dir/dmesg.after.txt" 2>&1 || true
    echo "$rc" >"$case_dir/exit_code.txt"
    if [[ "$rc" -ne 0 && -r /sys/class/drm/card0/device/devcoredump/data ]]; then
        timeout 10s sudo -n dd if=/sys/class/drm/card0/device/devcoredump/data \
            of="$case_dir/devcoredump.data" bs=1M count=16 status=none || true
    fi
    echo "$rc"
}

status_log="$OUT/status.tsv"
printf 'label\tvalue\texit\n' >"$status_log"
log_result() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$status_log"; }

low_rc="$(run_case "$LOW" "low")"
log_result low "$LOW" "$low_rc"
high_rc="$(run_case "$HIGH" "high")"
log_result high "$HIGH" "$high_rc"

if [[ "$low_rc" -ne 0 ]]; then
    echo "low_value_failed=$LOW" >"$OUT/summary.txt"
    cat "$OUT/summary.txt"
    exit 0
fi

if [[ "$high_rc" -eq 0 ]]; then
    echo "high_value_passed=$HIGH" >"$OUT/summary.txt"
    cat "$OUT/summary.txt"
    exit 0
fi

while [[ $((HIGH - LOW)) -gt 1 ]]; do
    mid=$(((LOW + HIGH) / 2))
    if [[ "$mid" -le "$LOW" ]]; then
        mid=$((LOW + 1))
    fi
    mid_rc="$(run_case "$mid" "v${mid}")"
    log_result "v${mid}" "$mid" "$mid_rc"
    if [[ "$mid_rc" -eq 0 ]]; then
        LOW="$mid"
        low_rc="$mid_rc"
    else
        HIGH="$mid"
        high_rc="$mid_rc"
    fi
done

{
    echo "last_pass_${DIM}=$LOW"
    echo "first_fail_${DIM}=$HIGH"
    echo "status_log=$status_log"
} >"$OUT/summary.txt"
cat "$OUT/summary.txt"
