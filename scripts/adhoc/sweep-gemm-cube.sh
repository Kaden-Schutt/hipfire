#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/hipfire-lds-cube}"
ARCH="${ARCH:-gfx1103}"
VARIANT="${VARIANT:-tile6_synth}"
MODE="${MODE:-full}"
LAUNCHES="${N_LAUNCH:-56}"
M_VALUES="${M_VALUES:-507 508}"
N_VALUES="${N_VALUES:-3041 3042}"
K_VALUES="${K_VALUES:-3020 3021}"
K_LIMIT="${K_LIMIT:-0}"

SRC="${ROOT}/scripts/lds_gemm_standalone_probe.hip"

if [[ ! -f "$SRC" ]]; then
    echo "missing source: $SRC" >&2
    exit 1
fi

mkdir -p "$OUT"
bin="$OUT/lds_gemm_standalone_probe"
temps="$OUT/save-temps"
mkdir -p "$temps"

cp "$SRC" "$OUT/lds_gemm_standalone_probe.hip"

/opt/rocm/bin/hipcc -O3 --offload-arch="$ARCH" -save-temps=obj \
    "$SRC" -o "$bin" >"$OUT/build.log" 2>&1

status_log="$OUT/status.tsv"
printf 'case\tm\tn\tk\tlaunches\texit\n' >"$status_log"

run_case() {
    local m="$1"
    local n="$2"
    local k="$3"
    local label="m${m}_n${n}_k${k}"
    local case_dir="$OUT/$label"
    mkdir -p "$case_dir"

    echo "[cube] $label launches=$LAUNCHES" >&2
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$m" "$n" "$k" "$LAUNCHES" "$rc" >>"$status_log"
}

read -r -a M_LIST <<<"$M_VALUES"
read -r -a N_LIST <<<"$N_VALUES"
read -r -a K_LIST <<<"$K_VALUES"

for m in "${M_LIST[@]}"; do
    for n in "${N_LIST[@]}"; do
        for k in "${K_LIST[@]}"; do
            run_case "$m" "$n" "$k"
        done
    done
done

{
    echo "status_log=$status_log"
} >"$OUT/summary.txt"
cat "$OUT/summary.txt"
