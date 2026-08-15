#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Inject a real client disconnect into the DS4 heterogeneous direct-HIP route,
# then prove the cross-device join drained, rollback was attested, and the next
# request remained coherent. The service must already be running with
# dense-expert-split placement and the daemon from this checkout.
set -euo pipefail

PORT=${1:-11435}
MODEL=${HIPFIRE_TEST_MODEL:?set HIPFIRE_TEST_MODEL to the loaded DS4 artifact or tag}
LOG=${HIPFIRE_SERVE_LOG:?set HIPFIRE_SERVE_LOG to the active serve log}
OUTDIR=${HIPFIRE_ABORT_EVIDENCE_DIR:?set HIPFIRE_ABORT_EVIDENCE_DIR to a durable evidence directory}
EP="http://127.0.0.1:${PORT}/v1/chat/completions"

case "$MODEL" in
  *[\"\\]*) echo "model selector may not contain quote or backslash" >&2; exit 2 ;;
esac
command -v curl >/dev/null || { echo "need curl" >&2; exit 2; }
command -v rg >/dev/null || { echo "need rg" >&2; exit 2; }
curl -fsS -m 10 "http://127.0.0.1:${PORT}/v1/models" >/dev/null \
  || { echo "serve is not ready on port ${PORT}" >&2; exit 2; }
mkdir -p "$OUTDIR"

LINES0=$(wc -l < "$LOG" 2>/dev/null || echo 0)
REQ_A=$(printf '{"model":"%s","messages":[{"role":"user","content":"Write an exhaustive numbered explanation of B-trees, LSM-trees, hash indexes, and tries. Continue for at least 800 numbered points."}],"max_tokens":1024,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}' "$MODEL")

echo "=== Turn A: close the streaming socket during heterogeneous decode ==="
curl -sS -N -m 5 "$EP" -H 'content-type: application/json' -d "$REQ_A" \
  >"$OUTDIR/abort-turn-a.sse" 2>"$OUTDIR/abort-turn-a.stderr" || true
BYTES=$(wc -c < "$OUTDIR/abort-turn-a.sse")
echo "  socket closed after ${BYTES} streamed bytes"
test "$BYTES" -gt 0 || { echo "FAIL: disconnect happened before decode" >&2; exit 1; }
sleep 3

echo "=== Turn B: clean request after rollback ==="
REQ_B=$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply with exactly: the quick brown fox"}],"max_tokens":48,"temperature":0,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}' "$MODEL")
curl -fsS -m 180 "$EP" -H 'content-type: application/json' -d "$REQ_B" \
  >"$OUTDIR/follow-up.json"

tail -n +$((LINES0 + 1)) "$LOG" >"$OUTDIR/serve-log-slice.txt"
rg -q 'drafter=ar-heterogeneous abort=client rollback=attested post_join=true' \
  "$OUTDIR/serve-log-slice.txt" \
  || { echo "FAIL: no attested post-join rollback marker" >&2; exit 1; }
rg -q 'drafter=ar-heterogeneous tau=1.00' "$OUTDIR/serve-log-slice.txt" \
  || { echo "FAIL: follow-up did not use the heterogeneous route" >&2; exit 1; }
rg -q 'quick brown fox' "$OUTDIR/follow-up.json" \
  || { echo "FAIL: follow-up response was not coherent" >&2; exit 1; }

echo "PASS: heterogeneous disconnect drained, rolled back, and resumed coherently"
