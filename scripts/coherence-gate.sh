#!/usr/bin/env bash
# Coherence battery — replaces the byte-exact MQ4 quality gate.
#
# Rationale: byte-exact comparison blocks legitimate numerical-correctness
# improvements (e.g., norm convention fixes that change output for the
# better). This gate instead runs a small fixed matrix of (model × prompt)
# through the daemon and writes a markdown report that a human/agent
# reviewer reads before committing. The gate itself only fails on hard
# daemon/error signals (panics, non-zero exit, zero tokens emitted);
# correctness is assessed qualitatively on the report.
#
# Exit codes:
#   0  battery ran clean — open the report and inspect coherence
#   1  a test hit a hard error (daemon panic / zero tokens / timeout)
#   2  build or environment error
#
# Report destination: /tmp/coherence-<timestamp>.md (or $HIPFIRE_COHERENCE_OUT)
#
# Modes:
#   ./scripts/coherence-gate.sh          # short battery (~2-4 min)
#   ./scripts/coherence-gate.sh --full   # add A3B tests (~6-10 min)

set -u
cd "$(dirname "$0")/.."

FULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        -h|--help)
            sed -n '3,21p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild daemon if any relevant source is newer than the binary ────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/engine/src/qwen35.rs crates/engine/src/llama.rs \
               crates/engine/src/hfq.rs crates/engine/examples/daemon.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate: rebuilding daemon..."
    if ! cargo build --release --example daemon --features deltanet >&2; then
        echo "coherence-gate: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Test matrix ───────────────────────────────────────────────────────────
# Format: "model_file|id|prompt|max_tokens[|system_prompt_file]"
# The optional 5th field names a file under benchmarks/prompts/ to be read
# verbatim and passed as the daemon's `system` field. Used for tool-call
# coverage (see #87 — auto-MMQ regression slipped through previous gates
# because none exercised tool-emission shapes).
# Short battery: the three dense sizes + a small multi-turn recall check
# + a tool-call shape (auto-MMQ regression detector for #87 redo).
# Full battery (--full): adds A3B MoE tests (loads large models, ~2-3 min each).
SHORT_TESTS=(
    "qwen3.5-0.8b.mq4|cap|What is the capital of France? Answer in one short sentence.|80"
    "qwen3.5-4b.mq4|code|Write a one-line Python function named square that returns n*n.|180"
    "qwen3.5-9b.mq4|reason|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    "qwen3.5-9b.mq4|tool-call|What does the file /tmp/fibonacci.c contain?|180|tool_call_system.txt"
    # MQ3 coverage (gfx11+gfx12 only — refused on other archs at load).
    # Verifies WMMA prefill family + K4-unroll decode + fused residual all
    # dispatch and stay coherent. Same prompts as the MQ4 rows so output
    # drift between bit-widths is comparable.
    "qwen3.5-9b.mq3|reason-mq3|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|300"
    "qwen3.5-27b.mq3|cap-mq3-27b|What is the capital of France? Answer in one short sentence.|80"
)
FULL_EXTRA=(
    "qwen3.5-35b-a3b.mq4|moe-sheep|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|500"
    "qwen3.6-35b-a3b.mq4|moe36-sheep|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|800"
    "qwen3.6-27b.mq4|tool-call-27b|What does the file /tmp/fibonacci.c contain?|220|tool_call_system.txt"
)
tests=("${SHORT_TESTS[@]}")
if [ "$FULL" -eq 1 ]; then
    tests+=("${FULL_EXTRA[@]}")
fi

# ── Run ───────────────────────────────────────────────────────────────────
hard_errors=0

{
    echo "# Coherence battery"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FULL" -eq 1 ] && echo full || echo short )"
    echo
    echo "Review each output for coherence (fluent English, on-topic, not stuck"
    echo "in verbatim loops). Hard errors fail the gate; soft output changes do not."
    echo
} > "$OUT"

for entry in "${tests[@]}"; do
    IFS='|' read -r model_file prompt_id prompt max_tok system_file <<< "$entry"
    model_path="$MODELS_DIR/$model_file"
    if [ ! -f "$model_path" ]; then
        echo "## $model_file — $prompt_id — SKIPPED (model not present)" >> "$OUT"
        echo >> "$OUT"
        continue
    fi

    # Optional system prompt: load from benchmarks/prompts/ if specified
    # in the test entry. Used for tool-call shape coverage (#87) where the
    # system block contains the tools <tools>...</tools> definition.
    system_json=""
    if [ -n "${system_file:-}" ]; then
        system_path="benchmarks/prompts/$system_file"
        if [ -f "$system_path" ]; then
            system_text=$(python3 -c "import sys,json; print(json.dumps(open(sys.argv[1]).read()))" "$system_path")
            system_json=",\"system\":${system_text}"
        else
            echo "## $model_file — $prompt_id — SKIPPED (system prompt $system_path not found)" >> "$OUT"
            continue
        fi
    fi

    echo "== $model_file / $prompt_id =="
    # JSONL input for daemon. Use python json.dumps for the user prompt so
    # special tokens / quotes / backslashes in the fixture survive intact
    # (the previous sed-based escape only handled `"`, missing tabs, JSON
    # control chars, and `\n` literals — a tool-emit fixture would need
    # those).
    in_file="/tmp/coh_in_$$.jsonl"
    out_file="/tmp/coh_out_$$.log"
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.05${system_json}}
{"type":"unload"}
JL
    t0=$(date +%s.%N)
    timeout 240 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    done_line=$(grep -aE '"type":"done"' "$out_file" | head -1)
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    status="OK"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    fi

    # Tool-call shape: hard-fail on ChatML special-token leakage in the
    # visible output. Healthy tool-call emit looks like:
    #   <tool_call>{"name":"read","arguments":{"path":"/tmp/foo.c"}}</tool_call><|im_end|>
    # The corruption signature from #87 (auto-MMQ regression on gfx1151) is
    # `<|im_start|>` *inside* the tool_call body, e.g.
    #   <tool_call>\n<|im_start|>box", "bash","command":"..."
    # Count `<|im_start|>` in the assembled token text; healthy = 0,
    # corrupted ≥ 1. Trailing `<|im_end|>` is fine and stripped by clients;
    # we only flag im_start as a hard failure since legitimate output
    # never embeds it.
    case "$prompt_id" in
        tool-call*)
            text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))')
            im_start_leaks=$(printf '%s' "$text" | grep -oE '<\|im_start\|>' | wc -l | tr -d ' ')
            if [ "${im_start_leaks:-0}" -gt 0 ]; then
                status="HARD_ERROR (tool-call corruption: ${im_start_leaks}× <|im_start|> leaked into visible output — see #87)"
                hard_errors=$((hard_errors + 1))
            elif ! printf '%s' "$text" | grep -qE '<tool_call>'; then
                # Soft warn — model didn't emit a tool_call at all. Could
                # be quantization noise (small model decided to answer
                # directly); not a corruption signal.
                status="OK (soft: no <tool_call> emitted; model answered inline)"
            fi
            ;;
    esac

    {
        echo "## $model_file — $prompt_id"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        if [ -n "$done_line" ]; then
            echo "- stats: \`$done_line\`"
        fi
        echo "- prompt: \"$prompt\""
        echo
        if [ -n "$panic" ]; then
            echo '**PANIC/ERROR DETECTED:**'
            echo
            echo '```'
            echo "$panic"
            echo '```'
            echo
        fi
        echo '**Output:**'
        echo
        echo '```'
        grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))'
        echo '```'
        echo
    } >> "$OUT"

    rm -f "$in_file" "$out_file"
done

echo
echo "coherence report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"

# ── PFlash regression stage ─────────────────────────────────────────────
# Optional follow-up stage that asserts PFlash bench wall-clock and
# verdicts haven't regressed against the committed baseline. Skipped
# when target/drafter aren't present or when HIPFIRE_SKIP_PFLASH_GATE=1.
# Release the daemon GPU lock first so pflash-gate.sh can acquire its own.
gpu_release 2>/dev/null || true
trap - EXIT

if [ "${HIPFIRE_SKIP_PFLASH_GATE:-0}" = "1" ]; then
    echo
    echo "pflash-gate: SKIPPED (HIPFIRE_SKIP_PFLASH_GATE=1)"
    exit 0
fi
PFLASH_TARGET="${HIPFIRE_PFLASH_TARGET:-$MODELS_DIR/qwen3.5-27b.mq3}"
PFLASH_DRAFTER="${HIPFIRE_PFLASH_DRAFTER:-$MODELS_DIR/qwen3.5-0.8b.mq4}"
if [ ! -f "$PFLASH_TARGET" ] || [ ! -f "$PFLASH_DRAFTER" ]; then
    echo
    echo "pflash-gate: SKIPPED (target or drafter not present)"
    exit 0
fi

echo
echo "── pflash regression stage ────────────────────────────────────────"
HIPFIRE_PFLASH_TARGET="$PFLASH_TARGET" HIPFIRE_PFLASH_DRAFTER="$PFLASH_DRAFTER" \
    ./scripts/pflash-gate.sh
pflash_rc=$?
if [ "$pflash_rc" -ne 0 ]; then
    echo "pflash-gate: FAILED (exit $pflash_rc) — combined gate FAILED"
    exit 1
fi
exit 0
