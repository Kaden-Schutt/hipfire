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
# its own reference and must be exactly 0.000000; anything else means the
# harness is measuring the run-to-run noise floor rather than the codec, and
# the stage-2 number cannot be attributed.
#
# BOTH the control and the headline number are now ASSERTED by this script
# (stage 4). They used to be enforced only by a human reading stdout — the
# control's result directory was written and never read at all — which is not
# enforcement. A gate whose pass condition lives in a person's attention is a
# gate that passes.
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

# Upper bound on the headline KLD. Measured: 0.0027576 nats (95% CI
# 0.0019491-0.0038610) with PPL 7.6585.
#
# 0.010 is ~3.6x the measured mean and ~2.6x the upper CI bound. Chosen loose
# on purpose: the quantity being bounded is a Q8_0 re-quantisation error over
# a 6-chunk sample, so it has real sampling spread, and a bound tight enough
# to flake on a reseed would get raised rather than investigated the first
# time it fired. It is still far below anything a genuine defect produces —
# the failure modes this port has actually hit (a dropped H128 transform, a
# stale activation cache, an unrounded router) move the logits by ~1e-1 and
# the KLD by orders of magnitude, not by a factor of three.
KLD_MAX=${KLD_MAX:-0.010}

# The reference cache key includes everything the reference DEPENDS on, not
# just its shape. It used to be keyed on "${NCTX}x${CHUNKS}" alone and skipped
# whenever a file of that name was non-empty, so pointing the script at a
# different $HFQ — or rebuilding build_kld_ref_native — silently scored the
# new candidate against the OLD model's reference. That is a stale-oracle
# false negative: the number it prints is meaningless and looks fine.
#
# Keyed on: the model's identity and bytes-in-place (realpath, size, mtime),
# the slice contents, the reference builder binary, and the sampling
# parameters. Cheap: no 12 GB hash, and any of these changing changes the key.
ref_key() {
    local builder=./target/release/examples/build_kld_ref_native
    {
        printf '%s\n' "$(readlink -f "$HFQ")"
        stat -c '%s %Y' "$HFQ"
        sha256sum "$SLICE" | cut -d' ' -f1
        stat -c '%s %Y' "$builder"
        printf '%s %s %s\n' "$NCTX" "$CHUNKS" "$TOPK"
    } | sha256sum | cut -c1-16
}
KEY=$(ref_key)
REF="$OUT/escha-35b-f16-exact-${NCTX}x${CHUNKS}-${KEY}.kldref.bin"

# Each run gets its OWN result directories, named by the same key.
#
# `kld_reduce.py` reduces a whole DIRECTORY, so a single shared `per-seq/`
# means any `.kldseq` left behind by an earlier experiment silently joins this
# run's reduction — and the stage-4 assertions would then be scoring a mixture
# of runs, or asserting against a row this run did not produce. Per-key
# directories make "the reduction describes exactly this run" structural
# rather than something the operator has to remember to clean up.
RUNDIR="$OUT/run-${NCTX}x${CHUNKS}-${KEY}"
PER_SEQ="$RUNDIR/per-seq"
CONTROL="$RUNDIR/control"
mkdir -p "$PER_SEQ" "$CONTROL"

echo "== 1/4  weight-exact reference (escha_ref semantics, f16 expert store) =="
echo "        cache key: $(basename "$REF")"
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
    --output "$PER_SEQ/escha-35b-q8_0__${ARCH}__per-token.kldseq"

echo "== 3/4  negative control: the reference arm against its own reference =="
echo "        (must be exactly 0.000000, or stage 2 is unattributable)"
HIPFIRE_ESCHA_EXPERT_STORE=f16 \
./target/release/examples/eval_hipfire \
    --model "$HFQ" --ref "$REF" \
    --scoring-mode per-token --kv-mode f32 \
    --output "$CONTROL/escha-35b-f16-selfcontrol__${ARCH}__per-token.kldseq"

echo "== 4/4  reduce and ASSERT =="
python3 benchmarks/quality-baselines/harness/kld_reduce.py \
    --result-dir "$PER_SEQ" \
    --out-md "$RUNDIR/result-table.md" \
    --out-json "$RUNDIR/result-data.json"
# The control arm gets reduced too. Writing a result directory and never
# reading it is what let "must print exactly 0.000000" be a comment.
python3 benchmarks/quality-baselines/harness/kld_reduce.py \
    --result-dir "$CONTROL" \
    --out-md "$RUNDIR/control-table.md" \
    --out-json "$RUNDIR/control-data.json"
cat "$RUNDIR/result-table.md"
echo
cat "$RUNDIR/control-table.md"
echo

python3 - "$RUNDIR/result-data.json" "$RUNDIR/control-data.json" "$KLD_MAX" <<'PYEOF'
import json, sys

result, control, kld_max = sys.argv[1], sys.argv[2], float(sys.argv[3])
fail = []

def one(path, what):
    rows = json.load(open(path))
    if len(rows) != 1:
        fail.append(
            f"{what}: expected exactly one row in {path}, got {len(rows)} "
            f"({[r['variant'] for r in rows]}). A stale .kldseq from an earlier "
            "run is in the result directory; the reduction is not describing "
            "this run."
        )
        return None
    return rows[0]

# What "exactly 0.000000" means, precisely.
#
# The contract in the header is about the value eval_hipfire PRINTS, which is
# 6 decimal places. The underlying float is NOT bit-zero: measured
#
#     mean_kld = 2.1341004702939135e-10    p99_kld = 6.029504362788427e-09
#
# and — this is the part that matters — those two figures reproduce BIT-FOR-BIT
# across repeated runs against the same reference. So the residue is not a
# nondeterminism floor. It is the fixed difference between two programs
# computing the same forward: the reference is written by
# `build_kld_ref_native` and scored by `eval_hipfire`, which are separate
# binaries with their own scratch reuse and launch order. A constant ~1e-10
# between them is f32 last-bit noise, and it is ~1.3e7 times smaller than the
# 2.7576e-3 the production arm reports.
#
# The assertion therefore encodes what actually has to hold for stage 2 to be
# attributable, in two parts:
#   (1) the control rounds to 0.000000 at the printed precision — the literal
#       documented contract; and
#   (2) the control is at least 10 000x below the production number, so no
#       part of the headline figure can be the floor.
# A control that had drifted to genuine run-to-run noise would break (2) long
# before it broke (1), which is why (2) is here at all.
CONTROL_ABS_MAX = 5e-7   # rounds to 0.000000 at 6 dp
CONTROL_RATIO   = 1e4    # measured margin is ~1.3e7

ctl = one(control, "negative control")
if ctl is not None:
    if not (abs(ctl["mean_kld"]) < CONTROL_ABS_MAX and abs(ctl["p99_kld"]) < CONTROL_ABS_MAX):
        fail.append(
            f"negative control does not round to 0.000000: "
            f"mean_kld={ctl['mean_kld']!r} p99_kld={ctl['p99_kld']!r}. The "
            "reference arm scored against its own reference must agree with "
            "it to the printed precision; anything visible at 6 dp means the "
            "harness has a floor of its own and the stage-2 KLD cannot be "
            "attributed to the Q8_0 expert re-quantisation."
        )
    else:
        print(
            f"negative control: mean_kld={ctl['mean_kld']:.3e} "
            f"p99_kld={ctl['p99_kld']:.3e} (prints as 0.000000), OK"
        )

res = one(result, "production arm")
if res is not None:
    print(
        f"production arm: mean_kld={res['mean_kld']:.7f} nats "
        f"(95% CI {res['mean_kld_ci_lo']:.7f}-{res['mean_kld_ci_hi']:.7f}) "
        f"ppl={res['ppl']:.4f}"
    )
    if not (res["mean_kld"] <= kld_max):
        fail.append(
            f"KLD {res['mean_kld']:.7f} nats exceeds the bound {kld_max}. "
            "See the KLD_MAX rationale at the top of escha-kld.sh: this bound "
            "is ~3.6x the recorded 0.0027576, so exceeding it is not sampling "
            "spread."
        )
    if ctl is not None and not (
        abs(ctl["mean_kld"]) * CONTROL_RATIO < res["mean_kld"]
    ):
        fail.append(
            f"the negative control ({ctl['mean_kld']:.3e}) is within "
            f"{CONTROL_RATIO:.0e}x of the production KLD "
            f"({res['mean_kld']:.7f}). Whatever stage 2 is measuring, a "
            "material fraction of it is the harness floor rather than the "
            "codec, and the number must not be reported as a codec result."
        )

if fail:
    print()
    for f in fail:
        print("G5 FAIL:", f)
    sys.exit(1)
print("G5 PASS")
PYEOF
