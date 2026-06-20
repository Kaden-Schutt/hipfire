#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/scripts/lds_direct_ab_multi_exec_matrix.sh"
COMPARE="$ROOT/scripts/lds_direct_ab_summary_compare.sh"

OUT="${OUT:-/tmp/hipfire-lds-direct-ab-780m}"
ARCH="${ARCH:-gfx1103}"
mode="build-only"
compare_left=""
compare_right=""
compare_out="/tmp/hipfire-lds-direct-ab-780m-compare.tsv"

usage() {
    cat <<EOF
usage: $0 [--build-only|--risky] [--out DIR] [--arch ARCH]
       $0 --compare local-direct-ab-summary.tsv other-direct-ab-summary.tsv [out.tsv]

Default mode is --build-only, which compiles the current direct-AB repro
shapes and captures codegen artifacts without launching the reset-prone probe.

Modes:
  --build-only  Safe compile/codegen capture under OUT-buildonly
  --risky       Run focused pass-side controls first, then the 6x6 fail-side
                direct-AB repro. This can trip HIP-719 and wedge/reset affected
                gfx1103/780M stacks.
  --compare     Compare two direct-ab-artifact-summary.tsv files

Environment:
  OUT=$OUT
  ARCH=$ARCH
  HIPCC=/opt/rocm/bin/hipcc
  WAIT_DEVCD_MS=12000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only)
            mode="build-only"
            ;;
        --risky)
            mode="risky"
            ;;
        --out)
            OUT="${2:?missing DIR after --out}"
            shift
            ;;
        --arch)
            ARCH="${2:?missing ARCH after --arch}"
            shift
            ;;
        --compare)
            mode="compare"
            compare_left="${2:?missing local summary after --compare}"
            compare_right="${3:?missing other summary after --compare}"
            if [[ $# -ge 4 && "$4" != --* ]]; then
                compare_out="$4"
                shift 3
            else
                shift
                shift
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

for tool in "$RUNNER" "$COMPARE"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing executable: $tool" >&2
        exit 1
    fi
done

print_reference() {
    cat <<'EOF'

# Local gfx1103/780M direct-AB reference signatures
# Split-child control observed locally:
#   active=6x6 block=6x6 reads=3 iters=448 chunks=96,5 grid=512x86 exit=0
# One-child fail-side observed locally:
#   active=6x6 block=6x6 reads=3 iters=448 chunks=101 grid=512x86 exit=4
#   sync_failure='phase1 sync 24 global 24 failed: unspecified launch failure (719)'
#   dmesg_remove_queue=3
#   devcore_sig=1/0x000074669d000000/0x841051/0x3f000007/0x0fc00113
#   gcvm_sig=MORE_FAULTS,PERMISSION_FAULTS,RW/cid=8/rw=1/vmid=8
# Active-lane controls observed locally:
#   active=4x4 chunks=101 exit=0
#   active=5x5 chunks=101 exit=0
#   active=5x5 chunks=200 exit=0
EOF
}

run_case() {
    local case_name="$1"
    local active_x="$2"
    local active_y="$3"
    local chunks="$4"
    local expected="$5"
    local build_only="$6"
    local clear_coredump rc match reported_expected

    reported_expected="$expected"
    if [[ "$build_only" == "1" ]]; then
        reported_expected="build-only"
        clear_coredump="${CLEAR_COREDUMP:-0}"
    else
        clear_coredump="${CLEAR_COREDUMP:-1}"
    fi

    echo "[direct-ab-780m] case=$case_name active=${active_x}x${active_y} chunks=$chunks build_only=$build_only expected=$reported_expected out=$OUT" >&2

    set +e
    ARCH="$ARCH" \
    BUILD_ONLY="$build_only" \
    CLEAR_COREDUMP="$clear_coredump" \
    WAIT_DEVCD_MS="${WAIT_DEVCD_MS:-12000}" \
    ACTIVE_X="$active_x" \
    ACTIVE_Y="$active_y" \
    BLOCK_X="$active_x" \
    BLOCK_Y="$active_y" \
    READS=3 \
    ITERS=448 \
    CHUNKS="$chunks" \
    GRID_X=512 \
    GRID_Y=86 \
    MODE=plain \
    "$RUNNER" "$OUT" >"$OUT/$case_name.wrapper.log" 2>&1
    rc=$?
    set -e

    match=0
    if [[ "$build_only" == "1" && "$rc" -eq 0 ]]; then
        match=1
    elif [[ "$expected" == "pass" && "$rc" -eq 0 ]]; then
        match=1
    elif [[ "$expected" == "fail" && "$rc" -ne 0 ]]; then
        match=1
    fi

    printf '%s\t%sx%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$active_x" "$active_y" "$chunks" "$build_only" \
        "$reported_expected" "$rc" "$match" "$OUT/$case_name.wrapper.log" >>"$report"

    [[ "$match" -eq 1 ]]
}

run_matrix() {
    local build_only="$1"
    local failures=0

    mkdir -p "$OUT"
    report="$OUT/report.tsv"
    printf 'case\tactive\tchunks\tbuild_only\texpected\texit\tmatch\twrapper_log\n' >"$report"

    if [[ "$build_only" == "1" ]]; then
        run_case "00-build-4x4-one-child-101" 4 4 101 pass 1 || failures=$((failures + 1))
        run_case "01-build-5x5-one-child-101" 5 5 101 pass 1 || failures=$((failures + 1))
        run_case "02-build-6x6-split-96-5" 6 6 96,5 pass 1 || failures=$((failures + 1))
        run_case "03-build-6x6-one-child-101" 6 6 101 fail 1 || failures=$((failures + 1))
    else
        run_case "00-control-4x4-one-child-101" 4 4 101 pass 0 || failures=$((failures + 1))
        run_case "01-control-5x5-one-child-101" 5 5 101 pass 0 || failures=$((failures + 1))
        run_case "02-control-5x5-one-child-200" 5 5 200 pass 0 || failures=$((failures + 1))
        run_case "03-control-6x6-split-96-5" 6 6 96,5 pass 0 || failures=$((failures + 1))
        run_case "04-fail-6x6-one-child-101" 6 6 101 fail 0 || failures=$((failures + 1))
    fi

    {
        echo "mode=$mode"
        echo "arch=$ARCH"
        echo "out=$OUT"
        echo "report=$report"
        echo "direct_ab_summary_tsv=$OUT/direct-ab-artifact-summary.tsv"
        echo "direct_ab_summary_md=$OUT/direct-ab-artifact-summary.md"
        echo "mismatches=$failures"
    } >"$OUT/summary.txt"

    cat "$OUT/summary.txt"
    print_reference
    exit "$failures"
}

case "$mode" in
    build-only)
        OUT="$OUT-buildonly"
        run_matrix 1
        ;;
    risky)
        run_matrix 0
        ;;
    compare)
        "$COMPARE" "$compare_left" "$compare_right" "$compare_out"
        ;;
esac
