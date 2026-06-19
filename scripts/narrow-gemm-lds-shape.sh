#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/hipfire-lds-shape-narrow}"
ARCH="${ARCH:-gfx1103}"
LAUNCHES="${N_LAUNCH:-100}"
M="${M:-512}"
N="${N:-3072}"
K="${K:-3072}"
K_LIMIT="${K_LIMIT:-0}"
LIMIT_CASES="${LIMIT_CASES:-}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"

RUNNER="${ROOT}/scripts/lds_gemm_standalone_matrix.sh"

if [[ ! -x "$RUNNER" ]]; then
    echo "missing runner: $RUNNER" >&2
    exit 1
fi

mkdir -p "$OUT"

report="${OUT}/narrow-report.tsv"
printf 'case\tvariant\tmode\tlaunches\tshape\tk_limit\texit\tartifacts\n' >"$report"

run_case() {
    local case_name="$1"
    local variant="$2"
    local mode="$3"
    local launches="$4"
    local m="$5"
    local n="$6"
    local k="$7"
    local klim="$8"
    local case_out="$OUT/$case_name"

    mkdir -p "$case_out"
    echo "[narrow] case=$case_name variant=$variant mode=$mode launches=$launches shape=${m}x${n}x${k} klim=$klim" >&2

    set +e
    ARCH="$ARCH" \
    VARIANT="$variant" \
    MODE="$mode" \
    N_LAUNCH="$launches" \
    M="$m" \
    N="$n" \
    K="$k" \
    K_LIMIT="$klim" \
    "$RUNNER" "$case_out"
    rc=$?
    set -e

    printf '%s\t%s\t%s\t%s\t%sx%sx%s\t%s\t%s\t%s\n' \
        "$case_name" "$variant" "$mode" "$launches" "$m" "$n" "$k" "$klim" "$rc" \
        "$case_out" >>"$report"
    return "$rc"
}

cases=(
    "00-tile5-pass tile5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "01-tile6-synth tile6_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "02-tile6-synth-masked tile6_synth_masked full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "03-tile6-noglobal-nostore tile6 noglobal_nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "04-tile6-aonly-nostore tile6 aonly_nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "05-tile6-bonly-nostore tile6 bonly_nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "06-tile6-nostore tile6 nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "07-tile6-aonly tile6 aonly ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "08-tile6-bonly tile6 bonly ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "09-tile6-noglobal tile6 noglobal ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "10-tile6-full tile6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
)

seen=0
first_fail=""
last_pass=""

for entry in "${cases[@]}"; do
    read -r case_name variant mode launches m n k klim <<<"$entry"
    if [[ -n "$LIMIT_CASES" && "$seen" -ge "$LIMIT_CASES" ]]; then
        break
    fi
    seen=$((seen + 1))

    if run_case "$case_name" "$variant" "$mode" "$launches" "$m" "$n" "$k" "$klim"; then
        last_pass="$case_name"
        continue
    fi

    if [[ -z "$first_fail" ]]; then
        first_fail="$case_name"
    fi
    if [[ "$CONTINUE_ON_FAIL" != "1" ]]; then
        break
    fi
done

{
    echo "last_pass=${last_pass:-none}"
    echo "first_fail=${first_fail:-none}"
    echo "report=${report}"
} >"$OUT/summary.txt"

cat "$OUT/summary.txt"
