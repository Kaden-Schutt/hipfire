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
  --risky       Run focused READS=2 pass-side controls first, then the
                33/34-lane direct-AB repros. This can trip HIP-719 and
                wedge/reset affected gfx1103/780M stacks.
  --compare     Compare two direct-ab-artifact-summary.tsv files

Environment:
  OUT=$OUT
  ARCH=$ARCH
  HIPCC=/opt/rocm/bin/hipcc
  WAIT_DEVCD_MS=12000

Artifacts:
  OUT/direct-ab-artifact-summary.tsv is the file to send back for comparison.
  Use --compare local.tsv other-780m.tsv to classify codegen/runtime drift.
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
# READS=2 one-wave pass-side observed locally:
#   active=8x4 block/layout=8x4 reads=2 iters=448 chunks=500/1000 grid=512x86 exit=0
# READS=2 first-two-wave fail-side observed locally:
#   active=9x4 block/layout=9x4 reads=2 iters=448 chunks=140 grid=512x86 exit=4
#   sync_failure='phase1 sync 133 global 133 failed: unspecified launch failure (719)'
#   dmesg_remove_queue=3
#   devcore_sig=1/0x000074669d000000/0x841051/0x3f000007/0x0fc00113
#   gcvm_sig=MORE_FAULTS,PERMISSION_FAULTS,RW/cid=8/rw=1/vmid=8
# READS=2 process-boundary control observed locally:
#   active=9x4 block/layout=9x4 reads=2 iters=448 chunks=130,10 grid=512x86 exit=0
# READS=2 one-wave controls observed locally:
#   active=32x1 block/layout=32x1 reads=2 iters=448 chunks=500 grid=512x86 exit=0
#   active=1x32 block/layout=1x32 reads=2 iters=448 chunks=500 grid=512x86 exit=0
# READS=2 33/34-lane edge checks observed locally:
#   active=33x1 block/layout=33x1 reads=2 iters=448 chunks=130 grid=512x86 exit=4
#   active=1x33 block/layout=1x33 reads=2 iters=448 chunks=140 grid=512x86 exit=0
#   active=1x33 block/layout=1x33 reads=2 iters=448 chunks=500 grid=512x86 exit=4
#   active=34x1 block/layout=34x1 reads=2 iters=448 chunks=130 grid=512x86 exit=4
#   active=1x34 block/layout=1x34 reads=2 iters=448 chunks=130 grid=512x86 exit=4
#   active=17x2 block/layout=17x2 reads=2 iters=448 chunks=130 grid=512x86 exit=4
#   active=2x17 block/layout=2x17 reads=2 iters=448 chunks=130 grid=512x86 exit=4
# READS=2 narrowed 34-layout controls observed locally:
#   active=30x1 start=0x0 block/layout=34x1 reads=2 iters=448 chunks=500 grid=512x86 exit=0
#   active=31x1 start=0x0 block/layout=34x1 reads=2 iters=448 chunks=500 grid=512x86 exit=4
#   active=31x1 start=1x0 block/layout=34x1 reads=2 iters=448 chunks=500 grid=512x86 exit=0
#   active=31x1 start=2x0 block/layout=34x1 reads=2 iters=448 chunks=500 grid=512x86 exit=4
#   active=1x32 start=0x0 block/layout=1x34 reads=2 iters=448 chunks=500 grid=512x86 force_wrap=0 exit=4
#   active=1x32 start=0x0 block/layout=1x34 reads=2 iters=448 chunks=500 grid=512x86 force_wrap=1 exit=0
EOF
}

run_case() {
    local case_name="$1"
    local active_x="$2"
    local active_y="$3"
    local block_x="$4"
    local block_y="$5"
    local layout_x="$6"
    local layout_y="$7"
    local reads="$8"
    local iters="$9"
    local grid_x="${10}"
    local grid_y="${11}"
    local chunks="${12}"
    local expected="${13}"
    local build_only="${14}"
    local active_x_start="${15:-0}"
    local active_y_start="${16:-0}"
    local force_wrap_cndmask="${17:-0}"
    local clear_coredump rc match reported_expected

    reported_expected="$expected"
    if [[ "$build_only" == "1" ]]; then
        reported_expected="build-only"
        clear_coredump="${CLEAR_COREDUMP:-0}"
    else
        clear_coredump="${CLEAR_COREDUMP:-1}"
    fi

    echo "[direct-ab-780m] case=$case_name active=${active_x}x${active_y} start=${active_x_start}x${active_y_start} block=${block_x}x${block_y} layout=${layout_x}x${layout_y} reads=$reads iters=$iters grid=${grid_x}x${grid_y} chunks=$chunks force_wrap=$force_wrap_cndmask build_only=$build_only expected=$reported_expected out=$OUT" >&2

    set +e
    ARCH="$ARCH" \
    BUILD_ONLY="$build_only" \
    CLEAR_COREDUMP="$clear_coredump" \
    WAIT_DEVCD_MS="${WAIT_DEVCD_MS:-12000}" \
    ACTIVE_X="$active_x" \
    ACTIVE_Y="$active_y" \
    ACTIVE_X_START="$active_x_start" \
    ACTIVE_Y_START="$active_y_start" \
    BLOCK_X="$block_x" \
    BLOCK_Y="$block_y" \
    LAYOUT_X="$layout_x" \
    LAYOUT_Y="$layout_y" \
    READS="$reads" \
    ITERS="$iters" \
    FORCE_WRAP_CNDMASK="$force_wrap_cndmask" \
    CHUNKS="$chunks" \
    GRID_X="$grid_x" \
    GRID_Y="$grid_y" \
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

    printf '%s\t%sx%s\t%sx%s\t%sx%s\t%sx%s\t%s\t%s\t%s\t%sx%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$active_x" "$active_y" "$block_x" "$block_y" \
        "$layout_x" "$layout_y" "$active_x_start" "$active_y_start" \
        "$force_wrap_cndmask" "$reads" "$iters" "$grid_x" "$grid_y" \
        "$chunks" "$build_only" "$reported_expected" "$rc" "$match" \
        "$OUT/$case_name.wrapper.log" >>"$report"

    [[ "$match" -eq 1 ]]
}

run_matrix() {
    local build_only="$1"
    local failures=0

    mkdir -p "$OUT"
    report="$OUT/report.tsv"
    printf 'case\tactive\tblock\tlayout\tactive_start\tforce_wrap_cndmask\treads\titers\tgrid\tchunks\tbuild_only\texpected\texit\tmatch\twrapper_log\n' >"$report"

    if [[ "$build_only" == "1" ]]; then
        run_case "00-build-8x4-r2-one-child-500" 8 4 8 4 8 4 2 448 512 86 500 pass 1 || failures=$((failures + 1))
        run_case "01-build-9x4-r2-one-child-130" 9 4 9 4 9 4 2 448 512 86 130 pass 1 || failures=$((failures + 1))
        run_case "02-build-9x4-r2-split-130-10" 9 4 9 4 9 4 2 448 512 86 130,10 pass 1 || failures=$((failures + 1))
        run_case "03-build-9x4-r2-one-child-140" 9 4 9 4 9 4 2 448 512 86 140 fail 1 || failures=$((failures + 1))
        run_case "04-build-32x1-r2-one-child-500" 32 1 32 1 32 1 2 448 512 86 500 pass 1 || failures=$((failures + 1))
        run_case "05-build-1x32-r2-one-child-500" 1 32 1 32 1 32 2 448 512 86 500 pass 1 || failures=$((failures + 1))
        run_case "06-build-33x1-r2-one-child-130" 33 1 33 1 33 1 2 448 512 86 130 fail 1 || failures=$((failures + 1))
        run_case "07-build-1x33-r2-one-child-140" 1 33 1 33 1 33 2 448 512 86 140 pass 1 || failures=$((failures + 1))
        run_case "08-build-1x33-r2-one-child-500" 1 33 1 33 1 33 2 448 512 86 500 fail 1 || failures=$((failures + 1))
        run_case "09-build-34x1-r2-one-child-130" 34 1 34 1 34 1 2 448 512 86 130 fail 1 || failures=$((failures + 1))
        run_case "10-build-1x34-r2-one-child-130" 1 34 1 34 1 34 2 448 512 86 130 fail 1 || failures=$((failures + 1))
        run_case "11-build-17x2-r2-one-child-130" 17 2 17 2 17 2 2 448 512 86 130 fail 1 || failures=$((failures + 1))
        run_case "12-build-2x17-r2-one-child-130" 2 17 2 17 2 17 2 448 512 86 130 fail 1 || failures=$((failures + 1))
        run_case "13-build-30x1-in-34x1-r2-one-child-500" 30 1 34 1 34 1 2 448 512 86 500 pass 1 || failures=$((failures + 1))
        run_case "14-build-31x1-in-34x1-r2-one-child-500" 31 1 34 1 34 1 2 448 512 86 500 fail 1 || failures=$((failures + 1))
        run_case "15-build-31x1-start1-in-34x1-r2-one-child-500" 31 1 34 1 34 1 2 448 512 86 500 pass 1 1 0 || failures=$((failures + 1))
        run_case "16-build-31x1-start2-in-34x1-r2-one-child-500" 31 1 34 1 34 1 2 448 512 86 500 fail 1 2 0 || failures=$((failures + 1))
        run_case "17-build-1x32-in-1x34-r2-one-child-500" 1 32 1 34 1 34 2 448 512 86 500 fail 1 || failures=$((failures + 1))
        run_case "18-build-1x32-in-1x34-r2-wrapcnd-one-child-500" 1 32 1 34 1 34 2 448 512 86 500 pass 1 0 0 1 || failures=$((failures + 1))
    else
        run_case "00-control-8x4-r2-one-child-500" 8 4 8 4 8 4 2 448 512 86 500 pass 0 || failures=$((failures + 1))
        run_case "01-control-32x1-r2-one-child-500" 32 1 32 1 32 1 2 448 512 86 500 pass 0 || failures=$((failures + 1))
        run_case "02-control-1x32-r2-one-child-500" 1 32 1 32 1 32 2 448 512 86 500 pass 0 || failures=$((failures + 1))
        run_case "03-control-9x4-r2-one-child-130" 9 4 9 4 9 4 2 448 512 86 130 pass 0 || failures=$((failures + 1))
        run_case "04-control-9x4-r2-split-130-10" 9 4 9 4 9 4 2 448 512 86 130,10 pass 0 || failures=$((failures + 1))
        run_case "05-control-1x33-r2-one-child-140" 1 33 1 33 1 33 2 448 512 86 140 pass 0 || failures=$((failures + 1))
        run_case "06-fail-9x4-r2-one-child-140" 9 4 9 4 9 4 2 448 512 86 140 fail 0 || failures=$((failures + 1))
        run_case "07-fail-33x1-r2-one-child-130" 33 1 33 1 33 1 2 448 512 86 130 fail 0 || failures=$((failures + 1))
        run_case "08-fail-1x33-r2-one-child-500" 1 33 1 33 1 33 2 448 512 86 500 fail 0 || failures=$((failures + 1))
        run_case "09-fail-34x1-r2-one-child-130" 34 1 34 1 34 1 2 448 512 86 130 fail 0 || failures=$((failures + 1))
        run_case "10-fail-1x34-r2-one-child-130" 1 34 1 34 1 34 2 448 512 86 130 fail 0 || failures=$((failures + 1))
        run_case "11-fail-17x2-r2-one-child-130" 17 2 17 2 17 2 2 448 512 86 130 fail 0 || failures=$((failures + 1))
        run_case "12-fail-2x17-r2-one-child-130" 2 17 2 17 2 17 2 448 512 86 130 fail 0 || failures=$((failures + 1))
        run_case "13-control-30x1-in-34x1-r2-one-child-500" 30 1 34 1 34 1 2 448 512 86 500 pass 0 || failures=$((failures + 1))
        run_case "14-fail-31x1-in-34x1-r2-one-child-500" 31 1 34 1 34 1 2 448 512 86 500 fail 0 || failures=$((failures + 1))
        run_case "15-control-31x1-start1-in-34x1-r2-one-child-500" 31 1 34 1 34 1 2 448 512 86 500 pass 0 1 0 || failures=$((failures + 1))
        run_case "16-fail-31x1-start2-in-34x1-r2-one-child-500" 31 1 34 1 34 1 2 448 512 86 500 fail 0 2 0 || failures=$((failures + 1))
        run_case "17-fail-1x32-in-1x34-r2-one-child-500" 1 32 1 34 1 34 2 448 512 86 500 fail 0 || failures=$((failures + 1))
        run_case "18-control-1x32-in-1x34-r2-wrapcnd-one-child-500" 1 32 1 34 1 34 2 448 512 86 500 pass 0 0 0 1 || failures=$((failures + 1))
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
