#!/usr/bin/env bash
# mi300x_sub_0_10_attempt.sh — final lever toward sub-0.10 KLD on Q3.5-9B MQ4.
#
# Uses the --awq-raw-sumsq-npz override (commit 59d6be49) which bypasses the
# rotated-Hessian-diagonal mismatch diagnosed in iterate run-006: AWQ scales
# were being read from np.diagonal(R H R.T) where R is FWHT — channels mixed,
# not raw per-channel sum(x²). The override feeds unsloth-imatrix raw sumsq
# directly, decoupling AWQ stats from GPTQ stats.
#
# Pipeline:
#   1. Convert unsloth 9B GGUF imatrix → raw-sumsq .npz (scripts/convert_gguf_imatrix_to_npz.py)
#   2. Build F1-AWQ-eligible iterate mask from a v3 9B HFQ
#   3. Run iterate with --awq-raw-sumsq-npz for 4 KM-damped rounds
#   4. Eval KLD per round; report whether any round crosses 0.10
#
# Expected outcomes (per investigation results.md):
#   - Round 0 lands at ~0.13 (v3 anchor, perturbed by more-discriminating scales)
#   - Subsequent rounds drift up/down a few percent
#   - SUB-0.10 IS PLAUSIBLE BUT NOT GUARANTEED — this is the test, not a victory lap.

set -euo pipefail

WORK="${WORK:-/workspace}"
HIPFIRE="${HIPFIRE_DIR:-${WORK}/hipfire}"
IMATRIX="${WORK}/imatrix/Qwen3.5-9B-GGUF/imatrix_unsloth.gguf_file"
MODELS_OUT="${WORK}/models/v3"
TS="${TS:-$(date +%Y%m%dT%H%M%S)}"
RUN_DIR="${WORK}/results/sub-0-10/${TS}"
PYTHON="${PYTHON:-python3}"

# Tunable iteration params
AWQ_ALPHA="${AWQ_ALPHA:-0.5}"
DAMPING="${DAMPING:-0.5}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
EPSILON="${EPSILON:-0.01}"

mkdir -p "$RUN_DIR"
cd "$HIPFIRE"

phase() { echo; echo "═══ [$(date +%H:%M:%S)] $* ═══"; }
ok()    { printf "    \033[32m✓\033[0m %s\n" "$*"; }
die()   { printf "    \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ── 0. Prerequisites ───────────────────────────────────────────────────────
phase "0  Prerequisites"
[ -s "$IMATRIX" ] || die "missing unsloth 9B imatrix at $IMATRIX (run bootstrap)"
# v3 base must exist (produced by v3_matrix or PR #266 recipe). If missing,
# build it now using the v3 recipe.
V3_BASE="$MODELS_OUT/qwen3.5-9b.mq4-v3"
if [ ! -f "$V3_BASE" ]; then
    die "v3 9B HFQ missing at $V3_BASE — run scripts/mi300x_v3_matrix.sh first (or just the 9B leg)"
fi
ok "v3 base: $V3_BASE ($(stat -c%s "$V3_BASE") bytes)"
ok "imatrix: $IMATRIX ($(stat -c%s "$IMATRIX") bytes)"

# Tools
[ -x target/release/examples/eval_hipfire ] || die "eval_hipfire not built"
[ -x target/release/hipfire-quantize ]      || die "hipfire-quantize not built"

# ── 1. GGUF imatrix → raw-sumsq .npz ───────────────────────────────────────
phase "1  Convert unsloth GGUF imatrix → raw-sumsq npz"
RAW_NPZ="$RUN_DIR/unsloth-9b-raw-sumsq.npz"
$PYTHON scripts/convert_gguf_imatrix_to_npz.py \
    --in "$IMATRIX" \
    --out "$RAW_NPZ" 2>&1 | tail -5
[ -s "$RAW_NPZ" ] || die "convert produced no npz"
n_entries=$($PYTHON -c "import numpy as np; print(len(np.load('$RAW_NPZ').files))")
ok "raw-sumsq npz: $n_entries entries"

# ── 2. Build F1-AWQ iterate mask from the v3 HFQ ──────────────────────────
phase "2  Build F1-AWQ iterate mask"
F1_MASK="$RUN_DIR/mask-f1-184-v3.json"
$PYTHON - "$V3_BASE" "$F1_MASK" <<'PYMASK'
import sys, json, struct, hashlib
hfq_path, out_path = sys.argv[1], sys.argv[2]

# F1 suffix list mirrors _is_awq_eligible_f1 in mq4_masked_calib.py.
F1_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "qkv_proj.weight", "wqkv.weight",
    "gate_proj.weight", "up_proj.weight",
    "w_gate.weight", "w_up.weight",
    "gate_up_proj.weight",
    "mlp.gate.weight", "router.weight",
)
def is_f1(name):
    if any(name.endswith(s) for s in F1_SUFFIXES):
        return True
    if ".in_proj_" in name:
        return True
    return False

# Parse HFQ header (light) to enumerate tensors + their quant type.
with open(hfq_path, "rb") as f:
    magic = f.read(4)
    assert magic == b"HFQM", f"bad magic {magic}"
    version, header_len, metadata_len = struct.unpack("<III", f.read(12))
    data_offset = struct.unpack("<Q", f.read(8))[0]
    metadata = json.loads(f.read(metadata_len).decode("utf-8"))
    tensor_count = struct.unpack("<I", f.read(4))[0]
    tensors = []
    for _ in range(tensor_count):
        nlen = struct.unpack("<H", f.read(2))[0]
        name = f.read(nlen).decode("utf-8")
        qt = struct.unpack("<B", f.read(1))[0]
        _bpe, _gc, _gs = struct.unpack("<BII", f.read(9))
        out_features, in_features = struct.unpack("<II", f.read(8))
        payload_len = struct.unpack("<Q", f.read(8))[0]
        tensors.append({
            "hfq_name": name,
            "quant_type": qt,
            "packable_flat_mq4": qt == 13,  # MQ4G256
            "in_features": in_features,
            "out_features": out_features,
        })

f1_eligible = [t for t in tensors if is_f1(t["hfq_name"]) and t["packable_flat_mq4"]]
mask = {
    "schema": "hipfire.astrea.mq4_masked.mask.v0",
    "base": hfq_path,
    "tensors": f1_eligible,
}
with open(out_path, "w") as fp:
    json.dump(mask, fp, indent=2, sort_keys=True)
print(f"wrote {len(f1_eligible)} F1-eligible MQ4 tensors → {out_path}")
PYMASK
n=$(jq '.tensors | length' "$F1_MASK")
ok "F1 mask: $n tensors"

# ── 3. Iterate with raw-sumsq override ─────────────────────────────────────
phase "3  Iterate (4 rounds, KM damping, raw-sumsq AWQ override)"
ITER_OUT="$RUN_DIR/iterate"
mkdir -p "$ITER_OUT"

# Round 0 needs initial-stats for GPTQ Hessian. Easiest path: use the
# in-process collection from a clean run. iterate's first round will collect
# them; subsequent rounds use the previous round's model. The raw-sumsq npz
# overrides AWQ derivation regardless of Hessian source.
BF16_DIR=$($PYTHON -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='Qwen/Qwen3.5-9B',
    revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','tokenizer*']))
")

$PYTHON scripts/mq4_masked_calib.py iterate \
    --hf-model "$BF16_DIR" \
    --calib-text benchmarks/calib/calib-1m.txt \
    --imatrix-mask "$F1_MASK" \
    --base-output-dir "$ITER_OUT" \
    --awq-alpha "$AWQ_ALPHA" \
    --damping "$DAMPING" \
    --epsilon "$EPSILON" \
    --max-rounds "$MAX_ROUNDS" \
    --awq-raw-sumsq-npz "$RAW_NPZ" \
    --gpu 0 \
    --bench-each-round \
    2>&1 | tee "$RUN_DIR/iterate.log"

# ── 4. Per-round KLD eval ─────────────────────────────────────────────────
phase "4  Per-round KLD eval"
KLDREF="$WORK/kldref/qwen3.5-9b-bf16.kldref.bin"
[ -s "$KLDREF" ] || die "missing kldref at $KLDREF; run v3_matrix first"

for r in "$ITER_OUT"/round_*; do
    [ -d "$r" ] || continue
    model="$r/model.hfq"
    [ -f "$model" ] || { echo "    $r: no model.hfq"; continue; }
    out_json="$r/kld-c512-q8.json"
    if [ ! -s "$out_json" ]; then
        ./target/release/examples/eval_hipfire \
            --model "$model" \
            --kldref "$KLDREF" \
            --kv-mode q8 --ctx 512 --scoring-mode prefill \
            --emit-json "$out_json" 2>&1 | tail -5 >> "$RUN_DIR/iterate.log"
    fi
    kld=$(jq -r '.kld_mean' "$out_json" 2>/dev/null || echo "?")
    ppl=$(jq -r '.ppl' "$out_json" 2>/dev/null || echo "?")
    rid=$(basename "$r")
    echo "    $rid:  KLD=$kld  PPL=$ppl"
done

# ── 5. Verdict ─────────────────────────────────────────────────────────────
phase "5  Verdict"
$PYTHON - "$ITER_OUT" "$RUN_DIR" > "$RUN_DIR/verdict.md" <<'PY'
import json, sys
from pathlib import Path
iter_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

rounds = []
for r in sorted(iter_dir.glob("round_*")):
    j = r / "kld-c512-q8.json"
    if not j.exists():
        continue
    d = json.load(j.open())
    rounds.append({"round": r.name, "kld": d.get("kld_mean"), "ppl": d.get("ppl")})

best = min((r for r in rounds if r["kld"] is not None), key=lambda x: x["kld"], default=None)
print("# sub-0.10 KLD attempt — verdict")
print()
print("## Per-round results")
print()
print("| Round | KLD | PPL | Δ vs v3 (0.1257) |")
print("|---|---:|---:|---:|")
for r in rounds:
    delta = "—" if r["kld"] is None else f"{(r['kld']-0.1257)/0.1257*100:+.1f}%"
    kld = "—" if r["kld"] is None else f"{r['kld']:.4f}"
    ppl = "—" if r["ppl"] is None else f"{r['ppl']:.3f}"
    print(f"| {r['round']} | {kld} | {ppl} | {delta} |")
print()
if best:
    print(f"## Best round: {best['round']}  KLD={best['kld']:.4f}  PPL={best['ppl']:.3f}")
    if best["kld"] < 0.10:
        print()
        print(f"### 🎯 SUB-0.10 ACHIEVED — KLD={best['kld']:.4f} crosses the 0.10 threshold.")
    else:
        margin = (best["kld"] - 0.10) / 0.10 * 100
        print()
        print(f"### Sub-0.10 NOT achieved. Best round is {margin:+.1f}% above target (KLD={best['kld']:.4f} vs target 0.10).")
        print()
        print("Next plausible levers (untried by this script):")
        print("- Per-tensor α grid search (Tier 2 from investigation results.md)")
        print("- Iterate on top of v3's GPTQ-corrected model rather than re-running GPTQ from scratch")
        print("- Longer calibration corpus matching unsloth's published recipe shape")
else:
    print("## No usable round outputs — see iterate.log for failure mode")
PY
cat "$RUN_DIR/verdict.md"
echo
ok "Full results: $RUN_DIR"
