#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/scripts/lds_gemm_standalone_matrix.sh"
OUT="${OUT:-/tmp/hipfire-lds-tail-snop-repro}"
ARCH="${ARCH:-gfx1103}"
M="${M:-512}"
N="${N:-3072}"
K="${K:-3072}"
PROFILE="${PROFILE:-repro}"

if [[ ! -x "$RUNNER" ]]; then
    echo "missing runner: $RUNNER" >&2
    exit 1
fi

mkdir -p "$OUT"
report="$OUT/report.tsv"
printf 'case\tvariant\tmode\tlaunches\tshape\tk_limit\texpected\texit\tmatch\tartifacts\n' >"$report"

run_case() {
    local case_name="$1"
    local variant="$2"
    local launches="$3"
    local k_limit="$4"
    local expected="$5"
    local case_out="$OUT/$case_name"
    local rc match

    mkdir -p "$case_out"
    echo "[gfx1103-lds-tail-snop] case=$case_name variant=$variant launches=$launches shape=${M}x${N}x${K} k_limit=$k_limit expected=$expected" >&2

    set +e
    ARCH="$ARCH" \
    VARIANT="$variant" \
    MODE=full \
    N_LAUNCH="$launches" \
    M="$M" \
    N="$N" \
    K="$K" \
    K_LIMIT="$k_limit" \
    "$RUNNER" "$case_out"
    rc=$?
    set -e

    match=0
    if [[ "$expected" == "pass" && "$rc" -eq 0 ]]; then
        match=1
    elif [[ "$expected" == "fail" && "$rc" -ne 0 ]]; then
        match=1
    fi

    printf '%s\t%s\tfull\t%s\t%sx%sx%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$variant" "$launches" "$M" "$N" "$K" "$k_limit" \
        "$expected" "$rc" "$match" "$case_out" >>"$report"

    [[ "$match" -eq 1 ]]
}

baseline="tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned"
tail_snop="tile6_lds_tail_snop_noextra_load4_consume4_pinned"

declare -a cases
case "$PROFILE" in
    repro)
        cases=(
            "00-baseline-noextra-full-n100 $baseline 100 0 pass"
            "01-tail-snop-full-n100 $tail_snop 100 0 fail"
        )
        ;;
    kedge)
        cases=(
            "00-tail-snop-klimit3024 $tail_snop 100 3024 pass"
            "01-tail-snop-klimit3032 $tail_snop 100 3032 fail"
        )
        ;;
    full)
        cases=(
            "00-baseline-noextra-full-n100 $baseline 100 0 pass"
            "01-tail-snop-klimit3024 $tail_snop 100 3024 pass"
            "02-tail-snop-klimit3032 $tail_snop 100 3032 fail"
            "03-tail-snop-full-n100 $tail_snop 100 0 fail"
        )
        ;;
    *)
        echo "unknown PROFILE=$PROFILE; expected repro, kedge, or full" >&2
        exit 2
        ;;
esac

failures=0
for entry in "${cases[@]}"; do
    read -r case_name variant launches k_limit expected <<<"$entry"
    if ! run_case "$case_name" "$variant" "$launches" "$k_limit" "$expected"; then
        failures=$((failures + 1))
    fi
done

{
    echo "profile=$PROFILE"
    echo "arch=$ARCH"
    echo "shape=${M}x${N}x${K}"
    echo "report=$report"
    echo "mismatches=$failures"
} >"$OUT/summary.txt"

cat "$OUT/summary.txt"
exit "$failures"
