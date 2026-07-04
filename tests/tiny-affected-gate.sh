#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — git-diff-driven tiny-model gate selector.
#
# This is the "make-like" selector for tiny tests. Git is the file timestamp /
# touched-file database:
#   - default: staged paths (`git diff --cached --name-only --diff-filter=ACMR`)
#   - --base <ref>: paths changed against a merge base, useful for CI
#   - --files-from <file>: explicit newline-delimited path list
#   - --require-coverage: no selected tiny family is inconclusive, not pass
#
# It maps touched paths to the smallest tiny family allowlist, then runs the
# relevant tiny gates with HIPFIRE_TINYQUANT_FAMILIES=<families>.
#
# Exit:
#   0 tiny coverage passed; caller may skip the large coherence battery
#   1 tiny coverage failed; caller should run/block with larger gates
#   2 infrastructure error
#   3 no tiny coverage or inconclusive tiny rows; caller should run large gates
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE=staged
BASE=""
FILES_FROM=""
DRY_RUN=0
REQUIRE_COVERAGE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --base)
            [ $# -ge 2 ] || { echo "tiny-affected-gate: --base needs a ref" >&2; exit 2; }
            MODE=base
            BASE="$2"
            shift 2
            ;;
        --files-from)
            [ $# -ge 2 ] || { echo "tiny-affected-gate: --files-from needs a file" >&2; exit 2; }
            MODE=files
            FILES_FROM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --require-coverage)
            REQUIRE_COVERAGE=1
            shift
            ;;
        -h|--help)
            sed -n '1,28p' "$0"
            exit 0
            ;;
        *)
            echo "tiny-affected-gate: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

changed_paths() {
    case "$MODE" in
        staged)
            git diff --cached --name-only --diff-filter=ACMR
            ;;
        base)
            local mb
            mb="$(git merge-base "$BASE" HEAD 2>/dev/null)" || {
                echo "tiny-affected-gate: could not compute merge-base with $BASE" >&2
                return 2
            }
            git diff --name-only --diff-filter=ACMR "$mb"...HEAD
            ;;
        files)
            [ -f "$FILES_FROM" ] || {
                echo "tiny-affected-gate: files list not found: $FILES_FROM" >&2
                return 2
            }
            cat "$FILES_FROM"
            ;;
    esac
}

add_family() {
    case " $FAMILIES " in
        *" $1 "*) ;;
        *) FAMILIES="$FAMILIES $1" ;;
    esac
}

add_all_supported() {
    add_family llama
    add_family qwen2
    add_family gemma3
    add_family minimax
    add_family qwen3_5
    add_family qwen3_5_moe
}

select_quant() {
    RUN_QUANT=1
}

select_state() {
    RUN_STATE=1
}

select_all_tiny() {
    select_quant
    select_state
}

select_spec() {
    RUN_SPEC=1
}

FAMILIES=""
UNCOVERED=0
UNCOVERED_REASONS=""
RUN_QUANT=0
RUN_STATE=0
RUN_SPEC=0
PATHS="$(changed_paths)" || exit 2

if [ -z "$PATHS" ]; then
    echo "tiny-affected-gate: no changed paths"
    exit 0
fi

while IFS= read -r path; do
    [ -n "$path" ] || continue
    case "$path" in
        *.md|docs/*|README.md|NEXT-STEPS.md|TODO.md)
            ;;

        tests/tiny-quant-gate.sh|tests/tiny-affected-gate.sh|tests/tiny-quant-baselines.txt|\
        tests/tiny-state-gate.sh|tests/tiny-state-baselines.txt|tests/tiny-spec-gate.sh|\
        crates/hipfire-eval/src/executor_tinyquant.rs|\
        crates/hipfire-serving-core/src/tiny_harness.rs|\
        crates/hipfire-quantize/src/fixture.rs)
            add_all_supported
            select_all_tiny
            ;;

        crates/hipfire-runtime/src/ddtree.rs|crates/hipfire-runtime/src/dflash.rs|\
        crates/hipfire-arch-qwen35/src/speculative.rs|crates/hipfire-arch-qwen35/src/mtp_spec.rs|\
        crates/hipfire-arch-qwen35/src/mtp_compose.rs|*dflash*|*ddtree*|*spec_step*|*tree_verify*|*spec_decode*)
            add_family qwen3_5
            add_family qwen3_5_moe
            select_spec
            ;;

        crates/hipfire-arch-llama/*|crates/hipfire-runtime/src/llama.rs|*llama*.rs)
            add_family llama
            select_all_tiny
            ;;
        crates/hipfire-arch-qwen2/*|*qwen2*.rs)
            add_family qwen2
            select_all_tiny
            ;;
        crates/hipfire-arch-gemma3/*|*gemma3*.rs)
            add_family gemma3
            select_all_tiny
            ;;
        crates/hipfire-arch-minimax/*|*minimax*.rs)
            add_family minimax
            select_all_tiny
            ;;
        crates/hipfire-arch-qwen35/*|crates/hipfire-serving-core/src/qwen35_*|*qwen35*.rs|*qwen3*.rs)
            add_family qwen3_5
            add_family qwen3_5_moe
            select_all_tiny
            ;;

        crates/hipfire-arch-deepseek4/*|*deepseek4*.rs|\
        crates/hipfire-arch-lfm2moe/*|*lfm2moe*.rs|\
        crates/hipfire-arch-dots-ocr/*|crates/hipfire-arch-gemma3-vl/*|\
        crates/hipfire-arch-qwen35-vl/*|*vl*.rs)
            UNCOVERED=1
            UNCOVERED_REASONS="${UNCOVERED_REASONS}${UNCOVERED_REASONS:+, }$path"
            ;;

        kernels/*|crates/hipfire-rdna/*|crates/hipfire-runtime/src/weights.rs|\
        crates/hipfire-runtime/src/hfq.rs|crates/hipfire-runtime/src/arch.rs|\
        crates/hipfire-runtime/src/calibration.rs|\
        *gemv*|*gemm*|*wmma*|*mq4*|*mq3*|*mq6*|*hfq*|*quant*|*rmsnorm*|*rotate*|*forward*)
            add_all_supported
            select_all_tiny
            ;;

        crates/hipfire-runtime/src/kv.rs|*kv_cache*|*kv_*|*cache*)
            add_all_supported
            select_state
            ;;

        crates/hipfire-daemon/*|crates/hipfire-serving-core/*|crates/hipfire-generate/*)
            add_all_supported
            select_all_tiny
            ;;
    esac
done <<<"$PATHS"

FAMILIES="$(printf '%s\n' $FAMILIES | sort -u | paste -sd, -)"

if [ "$UNCOVERED" -eq 1 ]; then
    echo "tiny-affected-gate: changed paths include tiny-uncovered families/features:"
    echo "  $UNCOVERED_REASONS"
    echo "tiny-affected-gate: INCONCLUSIVE -> run large gate suite"
    exit 3
fi

if [ -z "$FAMILIES" ]; then
    if [ "$REQUIRE_COVERAGE" -eq 1 ]; then
        echo "tiny-affected-gate: no tiny coverage selected for changed paths"
        echo "tiny-affected-gate: INCONCLUSIVE -> run large gate suite"
        exit 3
    else
        echo "tiny-affected-gate: no tiny-model-relevant changed paths"
        exit 0
    fi
fi

echo "tiny-affected-gate: selected families: $FAMILIES"
echo "tiny-affected-gate: selected gates: quant=$RUN_QUANT state=$RUN_STATE spec=$RUN_SPEC"

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

status=0
if [ "$RUN_QUANT" -eq 1 ]; then
    HIPFIRE_TINYQUANT_FAMILIES="$FAMILIES" ./tests/tiny-quant-gate.sh || status=$?
fi
if [ "$status" -eq 0 ] && [ "$RUN_STATE" -eq 1 ]; then
    HIPFIRE_TINYQUANT_FAMILIES="$FAMILIES" ./tests/tiny-state-gate.sh || status=$?
fi
if [ "$status" -eq 0 ] && [ "$RUN_SPEC" -eq 1 ]; then
    ./tests/tiny-spec-gate.sh || status=$?
fi
exit "$status"
