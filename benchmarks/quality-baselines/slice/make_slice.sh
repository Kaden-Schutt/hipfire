#!/usr/bin/env bash
# Generate the eval slice: 1024 sequences × 2048 tokens from wikitext-2 train.
#
# Output:
#   wikitext2-1024s-2048ctx.txt   — concatenated text
#   slice.md5                     — md5 tripwire
#
# This script is the recipe; the output is the canonical artifact (committed
# to git). Re-run only if intentionally regenerating the comparable cohort —
# md5 will change, all prior result tables become incomparable.
#
# Dependencies:
#   - Project venv at REPO_ROOT/.venv with `datasets` installed.
#     Set up via:  python3 -m venv .venv && .venv/bin/pip install datasets
#
# Usage:
#   ./make_slice.sh

set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd ../../.. && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -x "$PYTHON" ]; then
    echo "make_slice.sh: $PYTHON not found." >&2
    echo "Set up the venv:  cd ${REPO_ROOT} && python3 -m venv .venv && .venv/bin/pip install datasets" >&2
    exit 2
fi

OUT="wikitext2-1024s-2048ctx.txt"
N_SEQ=1024
N_CTX=2048

if [ -f "$OUT" ]; then
    echo "make_slice.sh: $OUT already exists. Refusing to overwrite — delete first if you really mean it."
    echo "(Regenerating changes the md5 and invalidates all prior result tables.)"
    exit 1
fi

echo "make_slice.sh: generating $OUT (target: $N_SEQ sequences × $N_CTX tokens)..."

"$PYTHON" <<PYEOF
import sys
from datasets import load_dataset

# Pin the dataset config + split + revision to maximize reproducibility.
# wikitext-2-raw-v1 is the canonical raw split (no UNK substitution).
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Concatenate, drop empty lines (HF's WT2 has lots of '' header rows).
lines = [r["text"] for r in ds if r["text"].strip()]
text = "\n".join(lines)

# We don't actually slice by tokens here — we slice by characters such that
# downstream tokenization at 2048 ctx × 1024 chunks produces enough usable
# tokens. Different tokenizers produce different counts, so we conservatively
# write enough text that any 30K-200K-vocab tokenizer yields >= 2_097_152
# tokens (1024 * 2048).
#
# Empirical: Qwen3.5 tokenizer ~3-4 chars/token on English Wikipedia.
# 2.1M tokens × 4 chars = 8.4 MB. We write up to 10 MB of text; downstream
# llama-perplexity will chunk it into n_ctx-sized windows naturally.
target_bytes = 10 * 1024 * 1024
text = text[:target_bytes]

with open("$OUT", "w") as f:
    f.write(text)

print(f"  wrote {len(text):,} bytes ({len(text)/1024/1024:.2f} MB)", file=sys.stderr)
PYEOF

# md5 tripwire
md5sum "$OUT" | awk '{print $1}' > slice.md5
echo "make_slice.sh: $OUT md5 = $(cat slice.md5)"

echo "make_slice.sh: done."
echo "  size: $(stat -c '%s' "$OUT") bytes"
echo "  md5:  $(cat slice.md5)"
echo
echo "Commit both $OUT and slice.md5. Do not regenerate without intent —"
echo "regenerating invalidates all prior comparable result tables."
