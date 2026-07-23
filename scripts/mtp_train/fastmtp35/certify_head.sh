#!/usr/bin/env bash
set -euo pipefail

CANDIDATE="${1:?usage: certify_head.sh <candidate.mtp> [output-dir]}"
OUT="${2:-$HOME/.hipfire/training/fastmtp-qwen36-a3b-v1/certification}"
TRUNK="${TRUNK:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
STOCK_MTP="${STOCK_MTP:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mtp}"
SESSION="${SESSION:-$HOME/mv/session_coding.json}"
TAG="${TAG:-qwen3.6:35b-a3b-mq4r}"
GPU="${GPU:-0}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-$HOME/.cache/hipfire}/hipfire-locks"

for path in "$CANDIDATE" "$TRUNK" "$STOCK_MTP" "$SESSION"; do
    [[ -s "$path" ]] || { echo "missing certification input: $path" >&2; exit 2; }
done
mkdir -p "$OUT/stock" "$OUT/candidate"
mkdir -p "$LOCK_ROOT"
export HIPFIRE_GPU_LOCKFILE="$LOCK_ROOT/gpu-${GPU}.lock"
source scripts/gpu-lock.sh
gpu_acquire "fastmtp-certify-gpu${GPU}"
trap gpu_release EXIT
ln -sfn "$TRUNK" "$OUT/stock/model.mq4r"
ln -sfn "$STOCK_MTP" "$OUT/stock/model.mtp"
ln -sfn "$TRUNK" "$OUT/candidate/model.mq4r"
ln -sfn "$CANDIDATE" "$OUT/candidate/model.mtp"

run_serve() {
    local label="$1"
    local model="$2"
    local mtp="$3"
    HIP_VISIBLE_DEVICES="$GPU" ROCR_VISIBLE_DEVICES="$GPU" \
        python3 scripts/serve_harness.py \
        --model "$model" \
        --tag "$TAG" \
        --kv q8 \
        --mtp "$mtp" \
        --thinking med \
        --max-tokens 4096 \
        --max-seq 32768 \
        --sampling registry \
        --mode session \
        --session "$SESSION" \
        --port 11520 \
        --home "$OUT/home-$label" \
        --serve-log "$OUT/$label.serve.log" \
        --out "$OUT/$label.json"
}

# Same eight sampled multi-turn requests, with AR as the quality/perf floor.
run_serve ar "$OUT/stock/model.mq4r" off
run_serve stock-mtp "$OUT/stock/model.mq4r" on
run_serve candidate-mtp "$OUT/candidate/model.mq4r" on

# The trained sidecar must not perturb the already-certified trunk PM4 route.
HIP_VISIBLE_DEVICES="$GPU" ROCR_VISIBLE_DEVICES="$GPU" \
HIPFIRE_REPLAY_MANUAL_CAPTURE=1 HIPFIRE_REPLAY_BACKEND=shadow \
    python3 scripts/redline_daemon_harness.py \
    --model "$OUT/candidate/model.mq4r" \
    --skip-prefill \
    --pm4 \
    --kv-mode q8 \
    --shadow-iterations 15 \
    --out "$OUT/redline-shadow.json" \
    --log "$OUT/redline-shadow.log"

python3 - "$OUT" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
summary = {}
for label in ("ar", "stock-mtp", "candidate-mtp"):
    rows = json.loads((root / f"{label}.json").read_text())
    rates = [row["decode_tok_s"] for row in rows if isinstance(row.get("decode_tok_s"), (int, float))]
    taus = [row["tau"] for row in rows if isinstance(row.get("tau"), (int, float))]
    summary[label] = {
        "turns": len(rows),
        "median_decode_tok_s": statistics.median(rates) if rates else None,
        "mean_decode_tok_s": statistics.mean(rates) if rates else None,
        "mean_tau": statistics.mean(taus) if taus else None,
        "runaway": sum(bool(row.get("runaway")) for row in rows),
        "empty": sum(bool(row.get("empty")) for row in rows),
        "attractor": sum(bool(row.get("attractor")) for row in rows),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
