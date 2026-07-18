#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Compare image B after an image-A turn with fresh image B.  This is deliberately
# a daemon-level check: image preprocessing alone cannot prove that the decoder
# KV state was reset before the second vision-conditioned prefill.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXE="$ROOT/target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
MODEL="${HIPFIRE_DOTS_OCR_MODEL:-$MODELS_DIR/dots-ocr.q8.hfq}"
IMAGE_A="${HIPFIRE_DOTS_OCR_IMAGE_A:-$ROOT/benchmarks/images/dots_ocr_smoke_001.jpg}"
IMAGE_B="${HIPFIRE_DOTS_OCR_IMAGE_B:-$IMAGE_A}"

if [[ ! -x "$EXE" || ! -f "$MODEL" || ! -f "$IMAGE_A" || ! -f "$IMAGE_B" ]]; then
    echo "dots-ocr-image-reset-gate: missing daemon, model, or image fixture" >&2
    exit 2
fi
if [[ "$IMAGE_A" == "$IMAGE_B" ]]; then
    echo "dots-ocr-image-reset-gate: set distinct IMAGE_A and IMAGE_B fixtures" >&2
    exit 2
fi

if [[ -r "$ROOT/scripts/gpu-lock.sh" ]]; then
    # shellcheck disable=SC1090
    . "$ROOT/scripts/gpu-lock.sh"
    gpu_acquire "dots-ocr-image-reset-gate" || exit 2
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"; gpu_release 2>/dev/null || true' EXIT

python3 - "$tmp/reused.jsonl" "$MODEL" "$IMAGE_A" "$IMAGE_B" <<'PY'
import json, sys
out, model, image_a, image_b = sys.argv[1:]
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps({"type": "load", "model": model, "params": {"max_seq": 8192}}) + "\n")
    f.write(json.dumps({"type": "generate", "id": "a", "prompt": "Read the image.", "image": image_a, "temperature": 0.0, "max_tokens": 128}) + "\n")
    f.write(json.dumps({"type": "generate", "id": "b", "prompt": "Read the image.", "image": image_b, "temperature": 0.0, "max_tokens": 128}) + "\n")
    f.write(json.dumps({"type": "unload"}) + "\n")
PY

python3 - "$tmp/fresh.jsonl" "$MODEL" "$IMAGE_B" <<'PY'
import json, sys
out, model, image_b = sys.argv[1:]
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps({"type": "load", "model": model, "params": {"max_seq": 8192}}) + "\n")
    f.write(json.dumps({"type": "generate", "id": "b", "prompt": "Read the image.", "image": image_b, "temperature": 0.0, "max_tokens": 128}) + "\n")
    f.write(json.dumps({"type": "unload"}) + "\n")
PY

"$EXE" <"$tmp/reused.jsonl" >"$tmp/reused.log" 2>&1
"$EXE" <"$tmp/fresh.jsonl" >"$tmp/fresh.log" 2>&1

python3 - "$tmp/reused.log" "$tmp/fresh.log" <<'PY'
import json, sys

def output(path, request_id):
    chunks = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("type") == "token" and value.get("id") == request_id:
            chunks.append(value.get("text", ""))
    return "".join(chunks)

reused = output(sys.argv[1], "b")
fresh = output(sys.argv[2], "b")
if not reused or not fresh or reused != fresh:
    raise SystemExit("dots.ocr image-A→image-B output differs from fresh-B output")
print("dots.ocr image reset parity: PASS (image-A→image-B == fresh-B)")
PY
