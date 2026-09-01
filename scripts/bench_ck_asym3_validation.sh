#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reproducible quantized-KV CK validation: LongBench hard30 plus long AR decode.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
SIDECAR="${SIDECAR:-$ROOT/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so}"
DAEMON="${DAEMON:-$ROOT/target/release/daemon}"
BENCH="${BENCH:-$ROOT/target/release/examples/bench_qwen35_mq4}"
DATA_ROOT="${DATA_ROOT:-$HOME/.hipfire/datasets/longbench-v2}"
DATASET="${DATASET:-$DATA_ROOT/longbench-hard30-pp32k.jsonl}"
MANIFEST="${MANIFEST:-$DATA_ROOT/longbench-hard30-pp32k.manifest.json}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_ARCH="${EXPECTED_ARCH:-gfx1100}"
KV_MODE="${KV_MODE:-asym3}"
NATIVE_GPU_ID="${NATIVE_GPU_ID:-$GPU_ID}"
CK_GPU_ID="${CK_GPU_ID:-$GPU_ID}"
PARALLEL_AB="${PARALLEL_AB:-0}"
MAX_SEQ="${MAX_SEQ:-65536}"
LONG_BENCH_MAX_TOKENS="${LONG_BENCH_MAX_TOKENS:-96}"
LONG_BENCH_LIMIT="${LONG_BENCH_LIMIT:-30}"
DECODE_PREFILL="${DECODE_PREFILL:-8192}"
DECODE_TOKENS="${DECODE_TOKENS:-4096}"
DECODE_RUNS="${DECODE_RUNS:-1}"
COOLDOWN="${COOLDOWN:-10}"
WORKSPACE_BYTES="${WORKSPACE_BYTES:-536870912}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/validation/ck-${KV_MODE}/$RUN_ID}"

for file in "$MODEL" "$SIDECAR" "$DATASET" "$MANIFEST"; do
    [[ -f "$file" ]] || { echo "missing required file: $file" >&2; exit 2; }
done
if [[ "${BUILD:-1}" == "1" ]]; then
    cargo build --release -p hipfire-daemon --features deltanet,flash-attn-ck
    cargo build --release -p hipfire-runtime --example bench_qwen35_mq4 \
        --features deltanet,flash-attn-ck
fi
for file in "$DAEMON" "$BENCH"; do
    [[ -x "$file" ]] || { echo "missing executable: $file" >&2; exit 2; }
done
if [[ -d "$OUT_ROOT" ]] && find "$OUT_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing non-empty OUT_ROOT: $OUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUT_ROOT"

{
    echo "git_head=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "kv_mode=$KV_MODE"
    sha256sum "$MODEL" "$SIDECAR" "$DATASET" "$MANIFEST" "$DAEMON" "$BENCH"
    rocm-smi --showproductname --showmeminfo vram
} >"$OUT_ROOT/meta.txt" 2>&1

run_longbench() {
    local mode="$1" gpu_id="$2"
    local ck_args=()
    if [[ "$mode" == "ck" ]]; then
        ck_args+=(
            --flash-attn-ck-lib "$SIDECAR"
            --flash-attn-ck-workspace-bytes "$WORKSPACE_BYTES"
        )
    fi
    python3 "$ROOT/scripts/eval_gemma4_eseries.py" \
        --daemon "$DAEMON" --model "$MODEL" \
        --model-label "qwen3.6-27b-${KV_MODE}-longbench-$mode" \
        --suite longbench --dataset "$DATASET" --manifest "$MANIFEST" \
        --out-dir "$OUT_ROOT/longbench/$mode" --physical-gpu "$gpu_id" \
        --expected-arch "$EXPECTED_ARCH" --runtime-home "/tmp/hipfire-ck-${KV_MODE}-$mode" \
        --max-seq "$MAX_SEQ" --max-tokens "$LONG_BENCH_MAX_TOKENS" \
        --limit "$LONG_BENCH_LIMIT" --prefill-batch 8 --kv-mode "$KV_MODE" --closed-think \
        --timeout 3600 "${ck_args[@]}"
}

run_decode() {
    local mode="$1" gpu_id="$2"
    local run log="$OUT_ROOT/decode/$mode.log"
    mkdir -p "$OUT_ROOT/decode"
    : >"$log"
    for ((run = 1; run <= DECODE_RUNS; run++)); do
        if [[ "$mode" == "ck" ]]; then
            env HIP_VISIBLE_DEVICES="$gpu_id" HIPFIRE_KV_MODE="$KV_MODE" \
                HIPFIRE_FLASH_ATTN_CK_LIB="$SIDECAR" \
                HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES="$WORKSPACE_BYTES" \
                "$BENCH" "$MODEL" --prefill "$DECODE_PREFILL" --prefill-runs 1 \
                --warmup 8 --gen "$DECODE_TOKENS" 2>&1 | tee -a "$log"
        else
            env -u HIPFIRE_FLASH_ATTN_CK_LIB -u HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES \
                -u HIPFIRE_FLASH_PREFILL HIP_VISIBLE_DEVICES="$gpu_id" HIPFIRE_KV_MODE="$KV_MODE" \
                "$BENCH" "$MODEL" --prefill "$DECODE_PREFILL" --prefill-runs 1 \
                --warmup 8 --gen "$DECODE_TOKENS" 2>&1 | tee -a "$log"
        fi
        ((run == DECODE_RUNS)) || sleep "$COOLDOWN"
    done
}

if [[ "$PARALLEL_AB" == "1" ]]; then
    run_longbench native "$NATIVE_GPU_ID" & native_pid=$!
    run_longbench ck "$CK_GPU_ID" & ck_pid=$!
    wait "$native_pid"
    wait "$ck_pid"
    run_decode native "$NATIVE_GPU_ID" & native_pid=$!
    run_decode ck "$CK_GPU_ID" & ck_pid=$!
    wait "$native_pid"
    wait "$ck_pid"
else
    run_longbench native "$NATIVE_GPU_ID"
    sleep "$COOLDOWN"
    run_longbench ck "$CK_GPU_ID"
    sleep "$COOLDOWN"
    run_decode native "$NATIVE_GPU_ID"
    sleep "$COOLDOWN"
    run_decode ck "$CK_GPU_ID"
fi

python3 - "$OUT_ROOT" <<'PY'
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
summaries = {
    mode: json.loads((root / "longbench" / mode / "summary.json").read_text())
    for mode in ("native", "ck")
}
rows = {
    mode: {
        row["id"]: row
        for row in map(json.loads, (root / "longbench" / mode / "results.jsonl").read_text().splitlines())
    }
    for mode in ("native", "ck")
}
common = sorted(set(rows["native"]) & set(rows["ck"]))
same = sum(
    rows["native"][key].get("prediction_sha256") == rows["ck"][key].get("prediction_sha256")
    for key in common
)
decode = {}
for mode in ("native", "ck"):
    text = (root / "decode" / f"{mode}.log").read_text()
    values = [float(value) for value in re.findall(r"tok/s \(gen\):\s+([0-9.]+)", text)]
    decode[mode] = {"samples": values, "median": statistics.median(values) if values else None}
report = {
    "longbench": {
        "native": summaries["native"],
        "ck": summaries["ck"],
        "paired": len(common),
        "same_prediction": same,
    },
    "long_decode": decode,
}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    f"LongBench prefill median: {summaries['native']['prefill_tok_s']['median']:.3f} -> "
    f"{summaries['ck']['prefill_tok_s']['median']:.3f} tok/s; identical={same}/{len(common)}"
)
print(f"Long decode medians: native={decode['native']['median']} ck={decode['ck']['median']} tok/s")
PY

echo "CK $KV_MODE validation complete: $OUT_ROOT"
