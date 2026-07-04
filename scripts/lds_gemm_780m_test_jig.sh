#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO="$ROOT/tests/gfx1103-lds-tail-snop-repro.sh"
COMPARE="$ROOT/scripts/lds_gemm_summary_compare.sh"
ISA_COMPARE="$ROOT/scripts/lds_gemm_isa_compare.sh"
ISA_SUMMARY_COMPARE="$ROOT/scripts/lds_gemm_isa_summary_compare.sh"

OUT="${OUT:-/tmp/hipfire-lds-tail-snop-780m}"
ARCH="${ARCH:-gfx1103}"
mode="build-only"
compare_left=""
compare_right=""

usage() {
    cat <<EOF
usage: $0 [--build-only|--risky|--kedge|--full] [--out DIR] [--arch ARCH]
       $0 --compare local-summary.tsv other-summary.tsv
       $0 --isa-compare local-isa-summary.tsv other-isa-summary.tsv

Default mode is --build-only, which compiles and captures codegen artifacts
without launching the reset-prone repro kernel.

Modes:
  --build-only  Safe compile/codegen capture under OUT-buildonly
  --risky      Run the focused baseline-pass / tail-snop-fail repro under OUT
  --kedge      Run the K_LIMIT 3024 pass-side / 3032 fail-side repro under OUT-kedge
  --full       Run baseline, K-edge, and full-depth tail-snop cases under OUT-full
  --compare    Compare two artifact-summary.tsv files
  --isa-compare
              Compare two placement-aware isa-summary.tsv files

Environment:
  OUT=$OUT
  ARCH=$ARCH
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
        --kedge)
            mode="kedge"
            ;;
        --full)
            mode="full"
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
            shift 2
            ;;
        --isa-compare)
            mode="isa-compare"
            compare_left="${2:?missing local ISA summary after --isa-compare}"
            compare_right="${3:?missing other ISA summary after --isa-compare}"
            shift 2
            ;;
        -h | --help)
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

print_reference() {
    cat <<'EOF'

# Local gfx1103/780M reference signatures
# Reference source hash:
#   source_sha256=4267f867c3901afc
# Clean build-only selected ISA hashes from this stack:
#   baseline tile6_lds_store_then_load_dynamiccols_load4_noextra_consume4_pinned:
#     selected_isa_norm_sha256=07a8198f82d17006
#   tail-snop tile6_lds_tail_snop_noextra_load4_consume4_pinned:
#     selected_isa_norm_sha256=abcf16851242d139
# Expected failing coredump signature when the reset-prone path trips:
#   devcore_sig=1/0x0000000000000000/0x0/0x3f000007/0x0fc00113
#   devcore_gds_flags=WRITE_DIS,FAULT_DETECTED,GRBM
#   devcore_gds_addr=0xfc0
#   devcore_gds_vm_vmid=1
#   devcore_gds_vm_addr=0xfc0
EOF
}

run_isa_capture() {
    local out_dir="$1"

    echo "[780m-jig] safe ISA placement capture out=$out_dir arch=$ARCH" >&2
    ARCH="$ARCH" SINGLE_INSTANTIATION=1 "$ISA_COMPARE" "$out_dir"
}

run_repro() {
    local profile="$1"
    local build_only="$2"
    local out_dir="$3"

    echo "[780m-jig] profile=$profile build_only=$build_only out=$out_dir arch=$ARCH" >&2
    ARCH="$ARCH" BUILD_ONLY="$build_only" PROFILE="$profile" OUT="$out_dir" "$REPRO"
    if [[ "$build_only" == "1" ]]; then
        run_isa_capture "$out_dir/isa-placement-single"
    fi
    print_reference
    cat <<EOF

# Outputs:
#   report: $out_dir/report.tsv
#   summary: $out_dir/summary.txt
#   artifact summary: $out_dir/artifact-summary.tsv
#   artifact summary markdown: $out_dir/artifact-summary.md
EOF
    if [[ "$build_only" == "1" ]]; then
        cat <<EOF
#   placement ISA summary: $out_dir/isa-placement-single/isa-summary.tsv
EOF
    fi
}

if [[ ! -x "$REPRO" ]]; then
    echo "missing repro wrapper: $REPRO" >&2
    exit 1
fi
if [[ ! -x "$COMPARE" ]]; then
    echo "missing summary comparator: $COMPARE" >&2
    exit 1
fi
if [[ ! -x "$ISA_COMPARE" ]]; then
    echo "missing ISA comparator: $ISA_COMPARE" >&2
    exit 1
fi
if [[ ! -x "$ISA_SUMMARY_COMPARE" ]]; then
    echo "missing ISA summary comparator: $ISA_SUMMARY_COMPARE" >&2
    exit 1
fi

case "$mode" in
    build-only)
        run_repro repro 1 "$OUT-buildonly"
        ;;
    risky)
        run_repro repro 0 "$OUT"
        ;;
    kedge)
        run_repro kedge 0 "$OUT-kedge"
        ;;
    full)
        run_repro full 0 "$OUT-full"
        ;;
    compare)
        "$COMPARE" "$compare_left" "$compare_right"
        ;;
    isa-compare)
        "$ISA_SUMMARY_COMPARE" "$compare_left" "$compare_right"
        ;;
esac
