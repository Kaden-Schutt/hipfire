#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# fetch_hermes_corpus.sh — pull `lambda/hermes-agent-reasoning-traces` from
# HuggingFace and flatten it into a plain-text calibration corpus suitable
# for `triattn_validate --corpus`.
#
# Output format: concatenated ChatML-wrapped conversations, one turn per
# line-ish. Tokenizer-agnostic; the sidecar calibrator just needs a long
# string of tokens representative of the deployment distribution.
#
# Usage:
#   bash fetch_hermes_corpus.sh [OUT_PATH]
#
# Default OUT_PATH: /root/hermes_corpus.txt
# Default size: both configs (kimi + glm-5.1), ~14.7k traces, ~100-200 MB.
#
# Requires the HF stack installed (pip install huggingface_hub pyarrow).
# Set HF_TOKEN env var if the dataset is gated (this one isn't as of 2026-04-19).

set -euo pipefail

OUT="${1:-/root/hermes_corpus.txt}"
MAX_ROWS="${MAX_ROWS:-0}"   # 0 = all rows
CONFIGS="${CONFIGS:-kimi,glm-5.1}"

log() { printf '[hermes-corpus] %s\n' "$*"; }

log "output: $OUT"
log "configs: $CONFIGS"
log "max rows: ${MAX_ROWS:-all}"

# Locate a python with HF stack. Prefer pytorch venv if present.
PY=python3
if [ -f /root/pytorch_env/bin/python3 ]; then
    PY=/root/pytorch_env/bin/python3
fi

"$PY" - "$OUT" "$CONFIGS" "$MAX_ROWS" <<'PY'
import json, os, sys
from pathlib import Path
out_path, configs_str, max_rows_str = sys.argv[1], sys.argv[2], sys.argv[3]
max_rows = int(max_rows_str) if max_rows_str else 0
configs = [c.strip() for c in configs_str.split(',') if c.strip()]

try:
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"[hermes-corpus] ERROR: missing deps ({e}). Run: pip install huggingface_hub pyarrow")
    sys.exit(3)

print(f"[hermes-corpus] snapshot_download lambda/hermes-agent-reasoning-traces...")
root = snapshot_download('lambda/hermes-agent-reasoning-traces', repo_type='dataset')
print(f"[hermes-corpus] cached at {root}")

total_rows = 0
total_bytes = 0
with open(out_path, 'w', encoding='utf-8') as fout:
    for cfg in configs:
        pq_path = Path(root) / 'data' / cfg / 'train.parquet'
        if not pq_path.exists():
            print(f"[hermes-corpus] WARN: {pq_path} missing, skipping")
            continue
        t = pq.read_table(str(pq_path))
        n = t.num_rows
        print(f"[hermes-corpus] config {cfg}: {n} rows")
        rows_iter = t.to_pylist() if max_rows == 0 or max_rows >= n else t.slice(0, max_rows).to_pylist()
        for row in rows_iter:
            convs = row.get('conversations') or []
            # Emit a ChatML-style flattened conversation per row. This is
            # close to what the target sees at inference time; the sidecar
            # calibrator just needs representative token streams.
            lines = []
            for c in convs:
                frm = c.get('from', '')
                val = c.get('value', '')
                if frm == 'system':
                    lines.append(f"<|im_start|>system\n{val}<|im_end|>")
                elif frm == 'human':
                    lines.append(f"<|im_start|>user\n{val}<|im_end|>")
                elif frm in ('gpt', 'assistant'):
                    lines.append(f"<|im_start|>assistant\n{val}<|im_end|>")
                elif frm == 'tool':
                    lines.append(f"<|im_start|>tool\n{val}<|im_end|>")
            if lines:
                block = '\n'.join(lines) + '\n\n'
                fout.write(block)
                total_rows += 1
                total_bytes += len(block.encode('utf-8'))
            if max_rows > 0 and total_rows >= max_rows:
                break
print(f"[hermes-corpus] wrote {total_rows} conversations, {total_bytes/1e6:.1f} MB to {out_path}")
PY

log "done: $(wc -c < "$OUT" | awk '{printf "%.1f MB\n", $1/1024/1024}') at $OUT"
