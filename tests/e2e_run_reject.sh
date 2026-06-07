#!/usr/bin/env bash
# Verify `hipfire run` (local-spawn path) surfaces daemon KV-budget errors
# instead of exiting 0 with empty stdout. HIPFIRE_LOCAL=1 skips the serve-
# is-up probe and forces the fresh-daemon path through run()'s for-await
# loop, which is the consumer we patched to print and set exitCode=1 on
# {"type":"error"}.
#
# The sibling HTTP-proxy path (runViaHttp) has the same error-handling
# shape — it parses SSE chunks and treats top-level `chunk.error` the
# same way. That server→SSE contract is independently verified by
# tests/e2e_kv_reject.sh (streaming case). Together they cover both
# `hipfire run` consumption paths.

set -uo pipefail
MODEL=${MODEL:-qwen3.5:0.8b}
TMPCFG=$(mktemp -d)
trap "rm -rf $TMPCFG" EXIT

# Isolated HOME; only config.json differs, models/bin are symlinked.
mkdir -p "$TMPCFG/.hipfire"
ln -sfn "$HOME/.hipfire/models" "$TMPCFG/.hipfire/models"
ln -sfn "$HOME/.hipfire/bin"    "$TMPCFG/.hipfire/bin"
# Tight max_seq so max_tokens alone forces daemon rejection. Set it below
# the min-viable bump in buildLoadMessage (max_tokens+1024), so load still
# picks max_seq=max_tokens+1024, and a request with max_tokens >> that gets
# rejected by the daemon. We pass max_tokens directly on the CLI.
cat > "$TMPCFG/.hipfire/config.json" <<'JSON'
{"max_seq": 512, "max_tokens": 16, "default_model": "qwen3.5:0.8b"}
JSON

OUT=$(mktemp); ERR=$(mktemp)
HIPFIRE_LOCAL=1 HOME="$TMPCFG" \
  bun cli/index.ts run "$MODEL" --max-tokens 100000 "hi" > "$OUT" 2> "$ERR"
EC=$?
echo "exit=$EC"
echo "--- stdout (head) ---"
head -c 200 "$OUT" || true
echo
echo "--- stderr (tail) ---"
tail -c 400 "$ERR" || true
echo

if [[ $EC -ne 0 ]] && grep -q "KV budget" "$ERR"; then
  echo "PASS"
  rm -f "$OUT" "$ERR"
  exit 0
else
  echo "FAIL: expected non-zero exit + 'KV budget' in stderr"
  rm -f "$OUT" "$ERR"
  exit 1
fi
