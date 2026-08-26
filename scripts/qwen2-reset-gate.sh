#!/usr/bin/env bash
# qwen2-reset-gate.sh — regression guard for the qwen2 (arch_id=7) daemon
# per-request reset no-op (#462 bundle-migration class).
#
# WHY THIS EXISTS: Qwen2Carrier moved the live Qwen2State into
# ModelState::Qwen2(bundle), but the daemon's reset/checkpoint handlers kept
# resetting the now-None `m.qwen2_state` direct field (only dots-ocr arch_id=8
# still populates it). So `{"type":"reset"}` between requests SILENTLY no-op'd:
# `next_pos` accumulated across requests until the max_seq ceiling auto-reset
# fired, bleeding prior-turn KV into new turns. Outputs stay on-topic (each
# request re-sends the full ChatML scaffold, anchoring the model to the latest
# turn), so attractor detection — serve-multiturn-gate.sh — cannot see it. The
# precise, deterministic signal is the ceiling auto-reset firing AT ALL.
#
# This loads a small qwen2 model at a SMALL max_seq and sends N explicit
# reset+generate cycles. With a WORKING reset, next_pos rewinds to 0 each turn,
# so the ceiling (next_pos + prompt + max_tokens > max_seq) can NEVER fire.
# If the daemon logs "resetting Qwen2State.next_pos", the reset no-op'd → FAIL.
#
# Exit: 0 PASS, 1 FAIL (reset no-op), 2 infra error, 3 SKIPPED (no qwen2 model).
set -u

EXE="./target/release/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
MAX_SEQ=256
MAX_TOKENS=32
CYCLES=16
# A fixed multi-sentence prompt: its re-prefill alone (~40+ tokens/cycle)
# accumulates next_pos past the ceiling within a few cycles when reset no-ops,
# independent of how many tokens the model actually generates.
PROMPT="Explain, in two or three sentences, what a hash map is, how it stores keys and values, and why average-case lookups are fast."

MODEL=""
for c in qwen25-0.5b-instruct.mq4 qwen25-0.5b-q2.mq4 vibethinker-3b.mq4.hfq; do
    [ -f "$MODELS_DIR/$c" ] && { MODEL="$MODELS_DIR/$c"; break; }
done
[ -z "$MODEL" ] && { echo "qwen2-reset-gate: SKIPPED — no qwen2 model under $MODELS_DIR" >&2; exit 3; }
[ -x "$EXE" ] || { echo "qwen2-reset-gate: daemon not built at $EXE" >&2; exit 2; }

if [ -r ./scripts/gpu-lock.sh ]; then
    # shellcheck disable=SC1090
    . ./scripts/gpu-lock.sh
    gpu_acquire "qwen2-reset-gate" || { echo "qwen2-reset-gate: could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

in_file="$(mktemp)"; out_file="$(mktemp)"
{
    printf '{"type":"load","model":"%s","params":{"max_seq":%d}}\n' "$MODEL" "$MAX_SEQ"
    pj="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROMPT")"
    for i in $(seq 1 "$CYCLES"); do
        printf '{"type":"reset"}\n'
        printf '{"type":"generate","id":"r%d","prompt":%s,"temperature":0.0,"max_tokens":%d}\n' "$i" "$pj" "$MAX_TOKENS"
    done
    printf '{"type":"unload"}\n'
} > "$in_file"

timeout 600 "$EXE" < "$in_file" > "$out_file" 2>&1
ec=$?
ceil="$(grep -c 'resetting Qwen2State.next_pos' "$out_file")"
toks="$(grep -c '"type":"token"' "$out_file")"
panic="$(grep -cE 'panicked|"type":"error"' "$out_file")"
rm -f "$in_file"

echo "qwen2-reset-gate: model=$(basename "$MODEL") max_seq=$MAX_SEQ cycles=$CYCLES -> ceiling_resets=$ceil tokens=$toks panics=$panic daemon_exit=$ec"
echo "  (full daemon log: $out_file)"

if [ "$ec" -ne 0 ] && [ "$ec" -ne 124 ]; then echo "qwen2-reset-gate: FAIL — daemon exit=$ec" >&2; exit 2; fi
if [ "$panic" -ne 0 ]; then echo "qwen2-reset-gate: FAIL — daemon panic/error event" >&2; exit 1; fi
if [ "$toks" -eq 0 ]; then echo "qwen2-reset-gate: FAIL — zero tokens generated" >&2; exit 2; fi
if [ "$ceil" -ne 0 ]; then
    echo "qwen2-reset-gate: FAIL — per-request reset no-op'd: next_pos accumulated and the max_seq=$MAX_SEQ ceiling auto-reset fired $ceil time(s) across $CYCLES reset+generate cycles. The reset handler is not rewinding the live ModelState::Qwen2 bundle state." >&2
    exit 1
fi
echo "qwen2-reset-gate: PASS — $CYCLES reset+generate cycles at max_seq=$MAX_SEQ, ceiling never fired: reset rewinds next_pos." >&2
exit 0
