#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/hipfire-lds-klimit-bisect}"
ARCH="${ARCH:-gfx1103}"
VARIANT="${VARIANT:-tile6_synth}"
MODE="${MODE:-full}"
LAUNCHES="${N_LAUNCH:-100}"
M="${M:-512}"
N="${N:-3072}"
K="${K:-3072}"

SRC="${ROOT}/scripts/lds_gemm_standalone_probe.hip"

if [[ ! -f "$SRC" ]]; then
    echo "missing source: $SRC" >&2
    exit 1
fi

mkdir -p "$OUT"
bin="$OUT/lds_gemm_standalone_probe"
temps="$OUT/save-temps"
mkdir -p "$temps"

{
    echo "variant=$VARIANT"
    echo "mode=$MODE"
    echo "launches=$LAUNCHES"
    echo "shape=$M x $N x $K"
    echo "arch=$ARCH"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$OUT/meta.txt"

cp "$SRC" "$OUT/lds_gemm_standalone_probe.hip"

/opt/rocm/bin/hipcc -O3 --offload-arch="$ARCH" -save-temps=obj \
    "$SRC" -o "$bin" >"$OUT/build.log" 2>&1

find "$OUT" -maxdepth 2 -type f \( -name '*.hsaco' -o -name '*.o' -o -name '*.s' -o -name '*.ll' \) \
    >"$OUT/generated-files.txt" 2>/dev/null || true

while IFS= read -r f; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    cp "$f" "$temps/$base" 2>/dev/null || true
    if file "$f" | grep -qi ELF; then
        /opt/rocm/llvm/bin/llvm-readobj --notes --sections --symbols "$f" \
            >"$temps/$base.readobj.txt" 2>&1 || true
        /opt/rocm/llvm/bin/llvm-objdump -d --mcpu="$ARCH" "$f" \
            >"$temps/$base.isa.txt" 2>&1 || true
    fi
done <"$OUT/generated-files.txt"

run_case() {
    local k_limit="$1"
    local label="$2"
    local case_dir="$OUT/$label"
    mkdir -p "$case_dir"

    echo "[bisect] label=$label k_limit=$k_limit" >&2
    dmesg --ctime >"$case_dir/dmesg.before.txt" 2>&1 || true
    set +e
    "$bin" "$VARIANT" "$MODE" "$LAUNCHES" "$M" "$N" "$K" "$k_limit" \
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
printf 'label\tk_limit\texit\n' >"$status_log"

log_result() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$status_log"
}

fail_at_full="$(run_case 0 full)"
log_result full 0 "$fail_at_full"

if [[ "$fail_at_full" -eq 0 ]]; then
    {
        echo "full_run_passed=1"
        echo "status_log=$status_log"
    } >"$OUT/summary.txt"
    cat "$OUT/summary.txt"
    exit 0
fi

low_k=1
low_rc="$(run_case "$low_k" "k${low_k}")"
log_result "k${low_k}" "$low_k" "$low_rc"

if [[ "$low_rc" -ne 0 ]]; then
    {
        echo "low_k_limit_failed=$low_k"
        echo "full_run_failed=1"
        echo "status_log=$status_log"
    } >"$OUT/summary.txt"
    cat "$OUT/summary.txt"
    exit 0
fi

high_k="$low_k"
next_rc=0
while [[ "$high_k" -lt $((K - 1)) ]]; do
    next_k=$((high_k * 2))
    if [[ "$next_k" -ge "$K" ]]; then
        next_k=$((K - 1))
    fi
    if [[ "$next_k" -le "$high_k" ]]; then
        break
    fi

    next_rc="$(run_case "$next_k" "k${next_k}")"
    log_result "k${next_k}" "$next_k" "$next_rc"
    if [[ "$next_rc" -ne 0 ]]; then
        break
    fi
    low_k="$next_k"
    low_rc="$next_rc"
    high_k="$next_k"
done

if [[ "$next_rc" -eq 0 ]]; then
    {
        echo "no_fail_found_within_k_limit_range=1"
        echo "status_log=$status_log"
    } >"$OUT/summary.txt"
    cat "$OUT/summary.txt"
    exit 0
fi

high_k="$next_k"
high_rc="$next_rc"

while [[ $((high_k - low_k)) -gt 1 ]]; do
    mid_k=$(((low_k + high_k) / 2))
    if [[ "$mid_k" -le "$low_k" ]]; then
        mid_k=$((low_k + 1))
    fi
    mid_rc="$(run_case "$mid_k" "k${mid_k}")"
    log_result "k${mid_k}" "$mid_k" "$mid_rc"
    if [[ "$mid_rc" -eq 0 ]]; then
        low_k="$mid_k"
        low_rc="$mid_rc"
    else
        high_k="$mid_k"
        high_rc="$mid_rc"
    fi
done

{
    echo "last_pass_k_limit=$low_k"
    echo "first_fail_k_limit=$high_k"
    echo "status_log=$status_log"
} >"$OUT/summary.txt"

cat "$OUT/summary.txt"
