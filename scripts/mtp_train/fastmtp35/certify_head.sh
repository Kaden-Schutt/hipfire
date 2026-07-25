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

if ! command -v hipcc >/dev/null 2>&1; then
    for rocm_bin in /opt/rocm/bin /opt/rocm/core-*/bin; do
        if [[ -x "$rocm_bin/hipcc" ]]; then
            export PATH="$rocm_bin:$PATH"
            break
        fi
    done
fi
command -v hipcc >/dev/null 2>&1 || {
    echo "hipcc was not found in PATH or an installed ROCm bin directory" >&2
    exit 2
}

for path in "$CANDIDATE" "$TRUNK" "$STOCK_MTP" "$SESSION"; do
    [[ -s "$path" ]] || { echo "missing certification input: $path" >&2; exit 2; }
done
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "refusing to certify from a checkout with tracked modifications" >&2
    git status --short >&2
    exit 2
fi

# Certification must execute the source commit recorded below, never whatever
# release binaries happened to be left in target/ by a prior experiment.
cargo build --release -p hipfire-cli
cargo build --release -p hipfire-runtime --example daemon
export HIPFIRE_CLI_BIN="$PWD/target/release/hipfire"
export HIPFIRE_DAEMON_BIN="$PWD/target/release/examples/daemon"

mkdir -p "$OUT/stock" "$OUT/candidate"
mkdir -p "$LOCK_ROOT"
export HIPFIRE_GPU_LOCKFILE="$LOCK_ROOT/gpu-${GPU}.lock"
source scripts/gpu-lock.sh
gpu_acquire "fastmtp-certify-gpu${GPU}"
trap gpu_release EXIT

# Do not symlink the trunk fixture. The serve control plane canonicalizes model
# paths before sidecar discovery; a symlink therefore resolves back into the
# global model directory and silently loads its stock `.mtp` for BOTH arms.
# Same-filesystem hard links preserve the exact trunk bytes while keeping the
# fixture path canonical, so sibling `model.mtp` selection is unambiguous and
# costs no data copy.
link_fixture() {
    local source="$1"
    local destination="$2"
    rm -f "$destination"
    if ! ln "$source" "$destination"; then
        echo "cannot hard-link certification fixture (source/output must share a filesystem): $source -> $destination" >&2
        exit 2
    fi
}
link_fixture "$TRUNK" "$OUT/stock/model.mq4r"
link_fixture "$STOCK_MTP" "$OUT/stock/model.mtp"
link_fixture "$TRUNK" "$OUT/candidate/model.mq4r"
link_fixture "$CANDIDATE" "$OUT/candidate/model.mtp"

python3 - "$TRUNK" "$STOCK_MTP" "$CANDIDATE" "$SESSION" \
    "$OUT/certification-manifest.json" "$(git rev-parse HEAD)" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

trunk, stock, candidate, session, output = map(Path, sys.argv[1:6])
commit = sys.argv[6]

def describe(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

manifest = {
    "schema_version": 1,
    "producer_git_commit": commit,
    "trunk": describe(trunk),
    "stock_mtp": describe(stock),
    "candidate_mtp": describe(candidate),
    "session": describe(session),
    "contract": {
        "turns": 8,
        "sampling": "registry",
        "thinking": "med",
        "max_tokens": 4096,
        "kv_mode": "q8",
        "mtp_k": 3,
        "redline_shadow_iterations": 15,
    },
}
partial = output.with_suffix(".json.partial")
partial.write_text(json.dumps(manifest, indent=2) + "\n")
os.replace(partial, output)
PY

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
    if [[ "$mtp" == "on" ]]; then
        local expected_sidecar="${model%.*}.mtp"
        if ! grep -Fq "MTP head loaded (sidecar $expected_sidecar)" "$OUT/$label.serve.log"; then
            echo "$label did not load its fixture sidecar: $expected_sidecar" >&2
            grep -F "MTP head loaded (sidecar " "$OUT/$label.serve.log" >&2 || true
            exit 2
        fi
    fi
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
    --daemon "$HIPFIRE_DAEMON_BIN" \
    --skip-prefill \
    --pm4 \
    --kv-mode q8 \
    --shadow-iterations 15 \
    --out "$OUT/redline-shadow.json" \
    --log "$OUT/redline-shadow.log"

python3 scripts/mtp_train/fastmtp35/evaluate_certification.py "$OUT"
