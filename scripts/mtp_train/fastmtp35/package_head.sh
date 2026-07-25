#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="${1:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
OUTPUT="${2:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
VOCAB_MAP="${3:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map.json}"
HF_MODEL="${HF_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B}"
STOCK_MTP="${STOCK_MTP:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mtp}"
MTP_QUANT="${MTP_QUANT:-mixed}"
OVERRIDE="${HIPFIRE_FASTMTP_CHECKPOINT:-$TRAINING_DIR/final.safetensors}"
ALLOW_PARTIAL="${HIPFIRE_FASTMTP_ALLOW_PARTIAL:-0}"

[[ -s "$OVERRIDE" && -s "$TRAINING_DIR/training-manifest.json" ]] || {
    echo "selected checkpoint or training manifest is missing: $OVERRIDE" >&2
    exit 2
}
case "$(basename "$OVERRIDE")" in
    final.safetensors|step-[0-9]*.safetensors) ;;
    *)
        echo "selected checkpoint must be final.safetensors or step-N.safetensors: $OVERRIDE" >&2
        exit 2
        ;;
esac
if [[ "$(realpath -m "$(dirname "$OVERRIDE")")" != "$(realpath -m "$TRAINING_DIR")" ]]; then
    echo "selected checkpoint must be inside the training directory: $OVERRIDE" >&2
    exit 2
fi
[[ -s "$VOCAB_MAP" ]] || {
    echo "compressed-vocabulary map is missing: $VOCAB_MAP" >&2
    exit 2
}
python3 - "$TRAINING_DIR/training-manifest.json" "$VOCAB_MAP" "$ALLOW_PARTIAL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
vocab_map = Path(sys.argv[2])
allow_partial = sys.argv[3] == "1"
manifest = json.loads(manifest_path.read_text())

required = {
    "steps",
    "planned_steps",
    "stop_step",
    "output",
    "world_size",
    "alignment",
    "recurrence_input",
    "feature_header",
    "vocab_map_sha256",
}
missing = sorted(required - manifest.keys())
if missing:
    raise SystemExit(f"training manifest is missing required fields: {missing}")

if manifest["output"] != "final.safetensors":
    raise SystemExit(
        f"training manifest output is {manifest['output']!r}, expected 'final.safetensors'"
    )
if not allow_partial and not (
    manifest["steps"] == manifest["planned_steps"] == manifest["stop_step"]
):
    raise SystemExit(
        "refusing to package a partial FastMTP run: "
        f"steps={manifest['steps']} planned_steps={manifest['planned_steps']} "
        f"stop_step={manifest['stop_step']}; set "
        "HIPFIRE_FASTMTP_ALLOW_PARTIAL=1 only for an explicit research artifact"
    )
if manifest["world_size"] != 4:
    raise SystemExit(
        f"production hiptrx manifest has world_size={manifest['world_size']}, expected 4"
    )
if manifest["alignment"] != "runtime-shifted-v1":
    raise SystemExit(
        f"unexpected training alignment: {manifest['alignment']!r}"
    )
if manifest["recurrence_input"] != "teacher-forced-v1":
    raise SystemExit(
        f"unexpected recurrence input: {manifest['recurrence_input']!r}"
    )

feature = manifest["feature_header"]
for key, expected in (("kv_mode", "q8"), ("state_quant", "q8")):
    if feature.get(key) != expected:
        raise SystemExit(
            f"feature manifest {key}={feature.get(key)!r}, expected {expected!r}"
        )

actual_vocab_hash = hashlib.sha256(vocab_map.read_bytes()).hexdigest()
if manifest["vocab_map_sha256"] != actual_vocab_hash:
    raise SystemExit(
        "vocabulary map hash does not match the training manifest: "
        f"{actual_vocab_hash} != {manifest['vocab_map_sha256']}"
    )
PY
case "$MTP_QUANT" in
    mixed|mq4) ;;
    q8)
        echo "full-Q8 sidecars are not runtime-compatible with routed-MoE MTP; use mixed" >&2
        exit 2
        ;;
    *)
        echo "unsupported MTP_QUANT=$MTP_QUANT (expected mixed or mq4)" >&2
        exit 2
        ;;
esac
if [[ "$(realpath -m "$OUTPUT")" == "$(realpath -m "$STOCK_MTP")" ]]; then
    echo "refusing to overwrite the deployed stock sidecar: $STOCK_MTP" >&2
    exit 2
fi
SNAPSHOT="$HF_MODEL"
if [[ -s "$HF_MODEL/refs/main" ]]; then
    candidate="$HF_MODEL/snapshots/$(<"$HF_MODEL/refs/main")"
    [[ -d "$candidate" ]] && SNAPSHOT="$candidate"
else
    shopt -s nullglob
    snapshots=("$HF_MODEL"/snapshots/*)
    shopt -u nullglob
    (( ${#snapshots[@]} == 0 )) || SNAPSHOT="${snapshots[0]}"
fi

cargo run --release -p hipfire-quantize --bin mtp_extract -- \
    --hf-dir "$SNAPSHOT" \
    --mtp-override "$OVERRIDE" \
    --vocab-sidecar "$VOCAB_MAP" \
    --quant "$MTP_QUANT" \
    --output "$OUTPUT" \
    --verbose

sha256sum \
    "$OVERRIDE" \
    "$TRAINING_DIR/training-manifest.json" \
    "$VOCAB_MAP" \
    "$OUTPUT" >"$OUTPUT.sha256"

{
    printf 'schema_version=1\n'
    printf 'quant=%s\n' "$MTP_QUANT"
    printf 'producer_git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'hf_snapshot=%s\n' "$(realpath "$SNAPSHOT")"
    printf 'training_checkpoint=%s\n' "$(realpath "$OVERRIDE")"
    printf 'training_checkpoint_sha256=%s\n' "$(sha256sum "$OVERRIDE" | cut -d " " -f 1)"
    printf 'training_manifest_sha256=%s\n' \
        "$(sha256sum "$TRAINING_DIR/training-manifest.json" | cut -d " " -f 1)"
    printf 'vocab_map_sha256=%s\n' "$(sha256sum "$VOCAB_MAP" | cut -d " " -f 1)"
    printf 'sidecar_sha256=%s\n' "$(sha256sum "$OUTPUT" | cut -d " " -f 1)"
} >"$OUTPUT.provenance"
