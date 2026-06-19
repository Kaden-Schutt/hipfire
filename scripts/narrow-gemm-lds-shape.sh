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
    "01-tile6-nolds-synth tile6_nolds_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "02-tile6-barrier-synth tile6_barrier_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "03-tile6-barrier3-synth tile6_barrier3_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "04-tile6-lds-one-synth tile6_lds_one_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "05-tile6-lds-padded-one-synth tile6_lds_padded_one_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "06-tile6-lds-two-store-once-one-read tile6_lds_two_store_once_one_read full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "07-tile6-lds-forced-same-load-only tile6_lds_forced_same_load_only full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "08-tile6-lds-same-phase-no-wide-read tile6_lds_same_phase_no_wide_read full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "09-tile6-lds-forced-same-read1 tile6_lds_forced_same_read1 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "10-tile6-lds-forced-same-read2 tile6_lds_forced_same_read2 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "11-tile6-lds-forced-same-read4 tile6_lds_forced_same_read4 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "12-tile6-lds-forced-same-read5 tile6_lds_forced_same_read5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "13-tile6-lds-second-store-only-read6 tile6_lds_second_store_only_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "14-tile6-lds-load-independent-store-read6 tile6_lds_load_independent_store_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "15-tile6-lds-load-next-store-same-read6 tile6_lds_load_next_store_same_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "16-tile6-lds-load-same-store-next-read6 tile6_lds_load_same_store_next_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "17-tile6-lds-store-then-load-read0 tile6_lds_store_then_load_read0 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "18-tile6-lds-store-then-load-read1 tile6_lds_store_then_load_read1 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "19-tile6-lds-store-then-load-read2 tile6_lds_store_then_load_read2 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "20-tile6-lds-store-then-load-read4 tile6_lds_store_then_load_read4 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "21-tile6-lds-store-then-load-read5 tile6_lds_store_then_load_read5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "22-tile6-lds-store-then-load-read6 tile6_lds_store_then_load_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "23-tile6-lds-store-then-load-dynamiccols-read6 tile6_lds_store_then_load_dynamiccols_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "24-tile6-lds-store-then-load-dynamiccols-load4-use4 tile6_lds_store_then_load_dynamiccols_load4_use4 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "25-tile6-lds-store-then-load-dynamiccols-load5-use5 tile6_lds_store_then_load_dynamiccols_load5_use5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "26-tile6-lds-store-then-load-dynamiccols-load5-serial-use5 tile6_lds_store_then_load_dynamiccols_load5_serial_use5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "27-tile6-lds-store-then-load-dynamiccols-load5-split4-use5 tile6_lds_store_then_load_dynamiccols_load5_split4_use5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "28-tile6-lds-store-then-load-dynamiccols-load5-split3-use5 tile6_lds_store_then_load_dynamiccols_load5_split3_use5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "29-tile6-lds-store-then-load-dynamiccols-load5-split2-3-use5 tile6_lds_store_then_load_dynamiccols_load5_split2_3_use5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "30-tile6-lds-store-then-load-dynamiccols-load5-split2-2-1-use5 tile6_lds_store_then_load_dynamiccols_load5_split2_2_1_use5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "31-tile6-lds-store-then-load-dynamiccols-load5-split1-keep5 tile6_lds_store_then_load_dynamiccols_load5_split1_keep5 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "32-tile6-lds-store-then-load-dynamiccols-load4-split1-keep4 tile6_lds_store_then_load_dynamiccols_load4_split1_keep4 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "33-tile6-lds-store-then-load-dynamiccols-load3-split1-keep3 tile6_lds_store_then_load_dynamiccols_load3_split1_keep3 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "34-tile6-lds-store-then-load-dynamiccols-load4-split1-consume4-pinned tile6_lds_store_then_load_dynamiccols_load4_split1_consume4_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "35-tile6-lds-store-then-load-dynamiccols-load3-split1-consume3-pinned tile6_lds_store_then_load_dynamiccols_load3_split1_consume3_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "36-tile6-lds-store-then-load-dynamiccols-load4-noextra-consume4-pinned tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "37-tile6-lds-store-then-load-dynamiccols-load3-noextra-consume3-pinned tile6_lds_store_then_load_dynamiccols_load3_noextra_consume3_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "38-tile6-lds-single-store-then-load-dynamiccols-load4-consume4-pinned tile6_lds_single_store_then_load_dynamiccols_load4_consume4_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "39-tile6-lds-store-then-load-dynamiccols-load4-nextrow-consume4-pinned tile6_lds_store_then_load_dynamiccols_load4_nextrow_consume4_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "40-tile6-lds-store-then-load-dynamiccols-load4-separate-tile-consume4-pinned tile6_lds_store_then_load_dynamiccols_load4_separate_tile_consume4_pinned full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "41-tile6-lds-store-then-load-nextrow-read6 tile6_lds_store_then_load_nextrow_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "42-tile6-lds-store-then-load-separate-tile-read6 tile6_lds_store_then_load_separate_tile_read6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "43-tile6-lds-store-then-load-separate-readtile tile6_lds_store_then_load_separate_readtile full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "44-tile6-lds-forced-same-second-store tile6_lds_forced_same_second_store full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "45-tile6-lds-two-store-one-read tile6_lds_two_store_one_read full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "46-tile6-synth tile6_synth full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "47-tile6-synth-masked tile6_synth_masked full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "48-tile6-noglobal-nostore tile6 noglobal_nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "49-tile6-aonly-nostore tile6 aonly_nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "50-tile6-bonly-nostore tile6 bonly_nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "51-tile6-nostore tile6 nostore ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "52-tile6-aonly tile6 aonly ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "53-tile6-bonly tile6 bonly ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "54-tile6-noglobal tile6 noglobal ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
    "55-tile6-full tile6 full ${LAUNCHES} ${M} ${N} ${K} ${K_LIMIT}"
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
