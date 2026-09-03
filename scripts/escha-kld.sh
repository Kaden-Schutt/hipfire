#!/usr/bin/env bash
# G5 — Escha-W2 quality gate: KLD on a FIXED corpus slice, teacher-forced.
#
# COMPARISON 1 (this script): hipfire-escha production vs `escha_ref` semantics.
#
#   The reference is escha_ref, NOT any Escha runtime: escha-mlx is Metal, the
#   escha wheel is CUDA (sm_80-sm_120) and ZML needs an NVIDIA driver, so none
#   of the three execute on gfx1151. `ref.py` declares itself "the semantic
#   contract for every Metal kernel in this package" and is gated on the
#   goldens, so agreeing with escha_ref IS agreeing with their runtime, and it
#   is exact rather than cross-machine.
#
#   escha_ref is a BLOCK-level oracle (codec, H128, expert_linear, swiglu) —
#   there is no CPU transformer in this repo and writing one for a 40-layer
#   hybrid DeltaNet MoE would make the reference itself the least-trusted
#   component. So the reference arm is the SAME hipfire forward with the escha
#   experts stored weight-exactly, `HIPFIRE_ESCHA_EXPERT_STORE=f16`. That is
#   bit-identical to `escha_ref::reconstruct`'s output (the decode already
#   produces fp16; G2 gates it bit-exact against escha_ref, G3 gates the H128
#   pair bit-exact), so the ONLY thing that differs between the two arms is the
#   Q8_0 re-quantisation of the expert weights — which is precisely what the
#   design doc predicts will dominate this number.
#
#   The f16 arm costs no more resident memory than production: per-expert
#   buffers are rounded to 2 MiB granules and Q8_0's 2.125/1.0625 MiB
#   projections already occupy the 4/2 MiB that f16 needs outright.
#
# COMPARISON 2 (bf16 parent Qwen/Qwen3.6-35B-A3B): NOT RUN. The bf16 parent is
#   not on this box in any form — /data/hipfire-models has no safetensors copy,
#   and the only cached artifact of the parent is
#   `unsloth/Qwen3.6-35B-A3B-GGUF: Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, a 4-bit
#   quant, which is not a reference. Fetching the parent is ~70 GB. Skipped
#   deliberately rather than substituted; see the design doc's Phase 1 results.
#
# TEACHER FORCING is structural here, not a flag. `build_kld_ref_native` writes
# the token stream into the HFKLDR file and `eval_hipfire` reads the tokens
# FROM that file rather than from any generation, so both arms are scored on
# one identical committed token stream by construction. Nothing is ever scored
# on a model's own greedy output — on ds4 that scored 8x better on the median
# and was optimistic.
#
# Both arms use --scoring-mode per-token so the candidate walks the same
# `forward_scratch` path the reference builder walks; the prefill-batch body is
# not admissible for escha anyway.
#
# --kv-mode f32 IS LOAD-BEARING. `build_kld_ref_native` builds its reference
# with an unquantised F32 KV cache. eval_hipfire defaults to `asym3`, and
# leaving that default in place folds the KV-quantisation error into the
# number: measured 0.018357 nats with asym3 against 0.002829 on the identical
# reference with f32, i.e. 6.5x, almost all of it KV rather than codec.
#
# Stage 3 is a NEGATIVE CONTROL, not decoration. It scores the f16 arm against
# its own reference and must print exactly 0.000000; anything else means the
# harness is measuring the run-to-run noise floor rather than the codec, and
# the stage-2 number cannot be attributed.
set -euo pipefail
cd "$(dirname "$0")/.."

HFQ=${1:-/data/hipfire-models/escha-35b.hfq}
SLICE=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
OUT=${ESCHA_KLD_OUT:-/tmp/escha-kld}
# n_ctx 384 => scored_per_chunk = 384 - 1 - 192 = 191 positions per chunk, so
# CHUNKS=1 is the design doc's "~192 positions". 6 chunks (1146 positions) is
# the default because one chunk is one sequence and gives no CI at all.
NCTX=${NCTX:-384}
CHUNKS=${CHUNKS:-6}
TOPK=${TOPK:-256}
ARCH=${ARCH:-gfx1151}

mkdir -p "$OUT/per-seq" "$OUT/control"
REF="$OUT/escha-35b-f16-exact-${NCTX}x${CHUNKS}.kldref.bin"

echo "== 1/4  weight-exact reference (escha_ref semantics, f16 expert store) =="
if [ ! -s "$REF" ]; then
    HIPFIRE_ESCHA_EXPERT_STORE=f16 \
    ./target/release/examples/build_kld_ref_native \
        --model "$HFQ" --slice "$SLICE" --top-k "$TOPK" \
        --n-ctx "$NCTX" --max-chunks "$CHUNKS" --output "$REF"
else
    echo "   reusing $REF"
fi

echo "== 2/4  score the production Q8_0 arm on the SAME token stream =="
./target/release/examples/eval_hipfire \
    --model "$HFQ" --ref "$REF" \
    --scoring-mode per-token --kv-mode f32 \
    --output "$OUT/per-seq/escha-35b-q8_0__${ARCH}__per-token.kldseq"

echo "== 3/4  negative control: the reference arm against its own reference =="
echo "        (must be exactly 0.000000, or stage 2 is unattributable)"
HIPFIRE_ESCHA_EXPERT_STORE=f16 \
./target/release/examples/eval_hipfire \
    --model "$HFQ" --ref "$REF" \
    --scoring-mode per-token --kv-mode f32 \
    --output "$OUT/control/escha-35b-f16-selfcontrol__${ARCH}__per-token.kldseq"

echo "== 4/4  reduce =="
python3 benchmarks/quality-baselines/harness/kld_reduce.py \
    --result-dir "$OUT/per-seq" \
    --out-md "$OUT/result-table.md" \
    --out-json "$OUT/result-data.json"
cat "$OUT/result-table.md"
