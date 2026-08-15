#!/usr/bin/env bash
# LongBench-v2 hard30 quality/performance A/B for Gemma 4 E-series prefill routes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON="${DAEMON:-$ROOT/target/release/examples/daemon}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.hipfire/models/gemma4-eseries}"
E2B="${E2B:-$MODEL_ROOT/gemma4-e2b-it-pr439-q8.hfq}"
E4B="${E4B:-$MODEL_ROOT/gemma4-e4b-it-pr439-q8.hfq}"
DATA_ROOT="${DATA_ROOT:-$HOME/.hipfire/datasets/longbench-v2}"
DATASET="${DATASET:-$DATA_ROOT/longbench-hard30-pp32k.jsonl}"
MANIFEST="${MANIFEST:-$DATA_ROOT/longbench-hard30-pp32k.manifest.json}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_ARCH="${EXPECTED_ARCH:?set EXPECTED_ARCH to gfx1100 or gfx1201}"
PREFILL_BATCH="${PREFILL_BATCH:-64}"
MAX_SEQ="${MAX_SEQ:-65536}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
LIMIT="${LIMIT:-30}"
COOLDOWN="${COOLDOWN:-10}"
MAX_ACCURACY_DROP="${MAX_ACCURACY_DROP:-0.05}"
MAX_CORRECTNESS_REGRESSIONS="${MAX_CORRECTNESS_REGRESSIONS:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/gemma4-longbench-prefill/$EXPECTED_ARCH/$RUN_ID}"

for file in "$DAEMON" "$E2B" "$E4B" "$DATASET" "$MANIFEST"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done
if [[ -d "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to mix LongBench results in non-empty OUT_ROOT: $OUT_ROOT" >&2
    exit 2
fi

run_one() {
    local mode="$1" model="$2" artifact="$3"
    local feature_args=()
    if [[ "$mode" == off ]]; then
        feature_args+=(
            --no-q8-fused-prefill
            --no-batched-embedding-prefill
            --no-ple-batched-prefill
            --no-ple-branch-batched-prefill
            --no-ple-activation-fused-prefill
        )
    fi
    python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
        --daemon "$DAEMON" --model "$artifact" \
        --model-label "gemma4-${model}-${EXPECTED_ARCH}-longbench-${mode}" \
        --suite longbench --dataset "$DATASET" --manifest "$MANIFEST" \
        --out-dir "$OUT_ROOT/$mode/$model" --physical-gpu "$GPU_ID" \
        --expected-arch "$EXPECTED_ARCH" \
        --runtime-home "/tmp/hipfire-gemma4-${model}-${EXPECTED_ARCH}-longbench-${mode}" \
        --max-seq "$MAX_SEQ" --max-tokens "$MAX_TOKENS" --limit "$LIMIT" \
        --prefill-batch "$PREFILL_BATCH" --timeout 3600 \
        "${feature_args[@]}"
}

mkdir -p "$OUT_ROOT"
for mode in off auto; do
    run_one "$mode" e2b "$E2B"
    sleep "$COOLDOWN"
    run_one "$mode" e4b "$E4B"
    sleep "$COOLDOWN"
done

python3 - "$OUT_ROOT" "$MAX_ACCURACY_DROP" "$MAX_CORRECTNESS_REGRESSIONS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
max_drop = float(sys.argv[2])
max_regressions = int(sys.argv[3])
report = {
    "max_accuracy_drop": max_drop,
    "max_correctness_regressions": max_regressions,
    "models": {},
}
for model in ("e2b", "e4b"):
    summaries = {
        mode: json.loads((root / mode / model / "summary.json").read_text())
        for mode in ("off", "auto")
    }
    rows = {}
    for mode in ("off", "auto"):
        rows[mode] = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in (root / mode / model / "results.jsonl").read_text().splitlines()
                if line.strip()
            )
        }
    off, auto = summaries["off"], summaries["auto"]
    if off["errors"] or auto["errors"]:
        raise SystemExit(f"{model}: execution errors in LongBench A/B")
    if off["completed"] != auto["completed"] or off["valid"] != auto["valid"]:
        raise SystemExit(f"{model}: off/auto completion mismatch")
    if off["scored"] != off["completed"] or auto["scored"] != auto["completed"]:
        raise SystemExit(
            f"{model}: missing explicit final answers: "
            f"off={off['scored']}/{off['completed']}, "
            f"auto={auto['scored']}/{auto['completed']}"
        )
    if auto["accuracy"] + max_drop < off["accuracy"]:
        raise SystemExit(
            f"{model}: accuracy regressed by more than {max_drop:.3f}: "
            f"{off['accuracy']:.4f} -> {auto['accuracy']:.4f}"
        )
    common = sorted(set(rows["off"]) & set(rows["auto"]))
    if set(rows["off"]) != set(rows["auto"]) or len(common) != off["completed"]:
        raise SystemExit(
            f"{model}: incomplete paired sample set: off={len(rows['off'])}, "
            f"auto={len(rows['auto'])}, common={len(common)}, completed={off['completed']}"
        )
    same_prediction = sum(
        rows["off"][key].get("prediction_sha256")
        == rows["auto"][key].get("prediction_sha256")
        for key in common
    )
    regressions = sum(
        bool(rows["off"][key].get("correct")) and not bool(rows["auto"][key].get("correct"))
        for key in common
    )
    gains = sum(
        not bool(rows["off"][key].get("correct")) and bool(rows["auto"][key].get("correct"))
        for key in common
    )
    if regressions > max_regressions:
        raise SystemExit(
            f"{model}: {regressions} per-example correctness regressions "
            f"exceed allowed {max_regressions}"
        )
    result = {
        "completed": off["completed"],
        "off_accuracy": off["accuracy"],
        "auto_accuracy": auto["accuracy"],
        "off_prefill_tok_s_median": off["prefill_tok_s"]["median"],
        "auto_prefill_tok_s_median": auto["prefill_tok_s"]["median"],
        "off_ttft_ms_median": off["ttft_ms"]["median"],
        "auto_ttft_ms_median": auto["ttft_ms"]["median"],
        "same_prediction": same_prediction,
        "paired": len(common),
        "correctness_regressions": regressions,
        "correctness_gains": gains,
    }
    report["models"][model] = result
    print(
        f"{model}: accuracy {off['accuracy']:.4f} -> {auto['accuracy']:.4f}; "
        f"prefill {off['prefill_tok_s']['median']:.3f} -> "
        f"{auto['prefill_tok_s']['median']:.3f} tok/s; "
        f"same predictions {same_prediction}/{len(common)}"
    )
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
PY

echo "Gemma 4 LongBench prefill A/B complete: $OUT_ROOT"
