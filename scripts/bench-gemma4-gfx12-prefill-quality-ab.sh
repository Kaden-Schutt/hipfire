#!/usr/bin/env bash
# Longer-output quality/performance A/B for the combined gfx1201 prefill routes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-$ROOT/target/validation/gemma4-eseries/gsm8k/test.jsonl}"
LIMIT="${LIMIT:-30}"
MAX_TOKENS="${MAX_TOKENS:-512}"
COOLDOWN="${COOLDOWN:-10}"
MAX_ACCURACY_DROP="${MAX_ACCURACY_DROP:-0.05}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-gfx12-prefill-quality/$RUN_ID}"

run_mode() {
    local mode="$1" enabled="$2"
    OUT_ROOT="$OUT_ROOT/$mode" RUN_ID="$mode" GPU_ID="$GPU_ID" ARCH_LABEL=gfx1201 \
    EXPECTED_ARCH=gfx1201 \
    DATASET="$DATASET" BATCHES=64 LIMIT="$LIMIT" REPEATS=1 \
    MAX_TOKENS="$MAX_TOKENS" COOLDOWN="$COOLDOWN" \
    FUSED_Q8_PREFILL=off PLE_BATCHED_PREFILL=off \
    BATCHED_EMBEDDING_PREFILL="$enabled" \
    PLE_BRANCH_BATCHED_PREFILL="$enabled" \
    PLE_ACTIVATION_FUSED_PREFILL="$enabled" \
        "$ROOT/scripts/bench-gemma4-gfx11-prefill-batch.sh"
}

mkdir -p "$OUT_ROOT"
run_mode baseline off
sleep "$COOLDOWN"
run_mode combined on

python3 - "$OUT_ROOT" "$MAX_ACCURACY_DROP" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
max_drop = float(sys.argv[2])
for model in ("e2b", "e4b"):
    baseline = json.loads((root / "baseline" / model / "b64" / "summary.json").read_text())
    combined = json.loads((root / "combined" / model / "b64" / "summary.json").read_text())
    if baseline["errors"] or combined["errors"]:
        raise SystemExit(f"{model}: execution errors in quality A/B")
    if baseline["completed"] != combined["completed"] or baseline["valid"] != combined["valid"]:
        raise SystemExit(f"{model}: baseline/combined completion mismatch")
    if combined["accuracy"] + max_drop < baseline["accuracy"]:
        raise SystemExit(
            f"{model}: accuracy regressed by more than {max_drop:.3f}: "
            f"{baseline['accuracy']:.4f} -> {combined['accuracy']:.4f}"
        )
    print(
        f"{model}: accuracy {baseline['accuracy']:.4f} -> {combined['accuracy']:.4f}; "
        f"prefill median {baseline['prefill_tok_s']['median']:.3f} -> "
        f"{combined['prefill_tok_s']['median']:.3f} tok/s"
    )
PY

echo "Gemma 4 gfx1201 quality A/B complete: $OUT_ROOT"
