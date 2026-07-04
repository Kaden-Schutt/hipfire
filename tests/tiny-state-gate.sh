#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny autoregressive state/KV/logit-hash gate (GPU).
#
# Emits seeded tiny fixtures, quantizes to each family's anchor format, then
# free-runs greedy decode with growing KV/state via tiny_quant_probe ar-hash.
# This is a structural tripwire for forward/KV/position/long-tail drift; it is
# not a semantic quality test.
#
#   ./tests/tiny-state-gate.sh            # check vs committed baselines
#   ./tests/tiny-state-gate.sh --record   # rewrite baselines for THIS gpu
#
# Env:
#   HIPFIRE_TINYQUANT_FAMILIES=llama,qwen2,...
#   HIPFIRE_TINYSTATE_LEN=128
#   HIPFIRE_TINYSTATE_PROMPT_LEN=4
#
# Exit: 0 = all selected cells pass, 1 = drift/crash, 2 = infra, 3 = skipped/no baseline.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD=0
[ "${1:-}" = "--record" ] && RECORD=1

BASELINES="tests/tiny-state-baselines.txt"
LEN="${HIPFIRE_TINYSTATE_LEN:-128}"
PROMPT_LEN="${HIPFIRE_TINYSTATE_PROMPT_LEN:-4}"
SEED="${HIPFIRE_TINYSTATE_SEED:-42}"
HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

ALL_FAMILIES=(llama qwen2 gemma3 minimax qwen3_5 qwen3_5_moe)
families=("${ALL_FAMILIES[@]}")
if [ -n "${HIPFIRE_TINYQUANT_FAMILIES:-}" ]; then
    IFS=',' read -r -a families <<<"$HIPFIRE_TINYQUANT_FAMILIES"
fi

anchor_for() {
    case "$1" in
        llama) echo q8f16 ;;
        minimax) echo mq6 ;;
        qwen2 | gemma3 | qwen3_5 | qwen3_5_moe) echo fp16 ;;
        *) return 1 ;;
    esac
}

lookup_baseline() {
    [ -f "$BASELINES" ] || return 0
    awk -v g="$1" -v f="$2" -v q="$3" '$1==g && $2==f && $3==q {print $4" "$5}' "$BASELINES" | head -1
}

echo "tiny-state-gate: building..."
cargo build --release \
    -p hipfire-quantize --bin hipfire-quantize \
    -p hipfire-serving-core --example tiny_quant_probe >/dev/null || {
    echo "build failed" >&2
    exit 2
}

"$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "tiny-state-gate" --watch-pid "$$" || {
    echo "could not acquire GPU lock" >&2
    exit 2
}
TMP="$(mktemp -d)"
trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true; rm -rf "$TMP"' EXIT

Q="$ROOT/target/release/hipfire-quantize"
P="$ROOT/target/release/examples/tiny_quant_probe"
fail=0
skip=0
matched=0
declare -a RECORDED=()

for raw_family in "${families[@]}"; do
    family="$(echo "$raw_family" | xargs)"
    [ -n "$family" ] || continue
    fmt="$(anchor_for "$family")" || {
        echo "  SKIP $family: unknown family"
        skip=$((skip + 1))
        continue
    }
    echo "== tiny-state: $family/$fmt =="
    hf_dir="$TMP/$family-hf"
    hfq="$TMP/$family-$fmt.hfq"
    if ! "$Q" --emit-fixture "$family" --out "$hf_dir" --seed "$SEED" >/dev/null 2>&1; then
        echo "  FAIL emit fixture"
        fail=$((fail + 1))
        continue
    fi
    extra=()
    [ "$family" = "qwen2" ] && extra=(--arch-id 7)
    if ! "$Q" --input "$hf_dir" --output "$hfq" --format "$fmt" "${extra[@]}" >/dev/null 2>&1; then
        echo "  SKIP $fmt: quantize unsupported"
        skip=$((skip + 1))
        continue
    fi

    out="$TMP/$family-ar.out"
    err="$TMP/$family-ar.err"
    if ! LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}" \
        "$P" ar-hash --arch "$family" --model "$hfq" --len "$LEN" --prompt-len "$PROMPT_LEN" --seed "$SEED" \
        >"$out" 2>"$err"; then
        echo "  FAIL ar-hash: $(tail -2 "$err" | tr '\n' ' ')"
        fail=$((fail + 1))
        continue
    fi
    gpu_arch="$(grep -oE 'gfx[0-9a-f]+' "$err" | head -1)"
    logit_hash="$(awk '/^logit_hash:/ {print $2}' "$out" | head -1)"
    token_hash="$(awk '/^token_hash:/ {print $2}' "$out" | head -1)"
    if [ -z "$gpu_arch" ] || [ -z "$logit_hash" ] || [ -z "$token_hash" ]; then
        echo "  FAIL missing ar-hash output"
        fail=$((fail + 1))
        continue
    fi
    RECORDED+=("$gpu_arch $family $fmt $logit_hash $token_hash 0")
    if [ "$RECORD" = 1 ]; then
        echo "  $fmt: $logit_hash $token_hash ($gpu_arch) [record]"
        matched=$((matched + 1))
        continue
    fi
    read -r base_logit base_token <<<"$(lookup_baseline "$gpu_arch" "$family" "$fmt")"
    if [ -z "$base_logit" ] || [ -z "$base_token" ]; then
        echo "  NOTE no baseline for $gpu_arch $family/$fmt — observed $logit_hash $token_hash"
        skip=$((skip + 1))
    elif [ "$base_logit" != "$logit_hash" ] || [ "$base_token" != "$token_hash" ]; then
        echo "  FAIL drift: observed $logit_hash/$token_hash baseline $base_logit/$base_token"
        fail=$((fail + 1))
    else
        echo "  OK $fmt: matches baseline ($logit_hash/$token_hash)"
        matched=$((matched + 1))
    fi
done

if [ "$RECORD" = 1 ]; then
    {
        echo "# tiny-state AR hash baselines — gpu_arch family format logit_hash token_hash rel_tol"
        echo "# len=$LEN prompt_len=$PROMPT_LEN seed=$SEED"
        printf '%s\n' "${RECORDED[@]}"
    } >"$BASELINES"
    echo "tiny-state-gate: wrote ${#RECORDED[@]} baselines to $BASELINES"
    exit 0
fi
if [ "$fail" -gt 0 ]; then
    echo "tiny-state-gate: FAIL ($fail cell(s))"
    exit 1
fi
if [ "$skip" -gt 0 ]; then
    echo "tiny-state-gate: INCONCLUSIVE ($skip skipped/no-baseline cell(s), $matched matched)"
    exit 3
fi
if [ "$matched" -gt 0 ]; then
    echo "tiny-state-gate: PASS ($matched cell(s))"
    exit 0
fi
echo "tiny-state-gate: INCONCLUSIVE (no cells ran)"
exit 3
