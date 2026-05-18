#!/usr/bin/env bash
# awq_coherence_check.sh — eyeball coherence of AWQ quants post-sweep.
#
# Uses target/release/examples/run (interactive REPL) with stdin-piped
# prompts to capture actual generated text. The probe-based variant
# (coherence_probe) only prints detector verdicts, not the text.
#
# Compares:
#   (1) 9B  α=0.5  — anchor (AWQ-paper default)
#   (2) 9B  α=0.6  — top KLD candidate from screening
#   (3) 0.8B α=0.0 — mq4-base baseline (AWQ identity)
#   (4) 0.8B α=0.55 — same alpha shipped
#
# (3) vs (4) discriminates size/bpw floor (both incoherent) from
# AWQ-specific corruption at small models (only (4) incoherent).

set -euo pipefail
cd "$(dirname "$0")/.."

# ── paths ─────────────────────────────────────────────────────────────
BF16_9B=/local/hipfire/Qwen3.5-9B-BF16-st
BF16_08B=/local/hipfire/Qwen3.5-0.8B-BF16-st
IMATRIX_9B=benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.imatrix.gguf
IMATRIX_08B=benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf
QUANT_BIN=target/release/hipfire-quantize
RUN_BIN=target/release/examples/run

QUANT_SLOT_9B=/local/hipfire/qwen3.5-9b.mq4-awq-current
QUANT_SLOT_08B=/local/hipfire/qwen3.5-0.8b.mq4-awq-current

OUT_DIR=benchmarks/quality-baselines/results/2026-05-14-awq-coherence-check
mkdir -p "$OUT_DIR"

# ── probe prompts (mixed domains; deliberately short to let thinking budget breathe) ─
# Tagged to make scanning the output dir easy.
PROMPT_short_qa='What is the capital of France? Answer in one sentence.'
PROMPT_code='Write a Python function that returns the sum of squares of a list of integers. Just the function, no explanation.'
PROMPT_reason='If a train leaves station A at 9am traveling 60 mph, and another leaves station B at 10am traveling 80 mph toward A, and the stations are 280 miles apart, at what time do they meet? Show your work briefly.'
PROMPT_NAMES=(short_qa code reason)

# ── pre-flight ────────────────────────────────────────────────────────
for f in "$BF16_9B" "$BF16_08B" "$IMATRIX_9B" "$IMATRIX_08B" "$QUANT_BIN" "$RUN_BIN"; do
    if [ ! -e "$f" ]; then
        echo "FATAL: missing $f" >&2
        exit 2
    fi
done

run_one_prompt() {
    local label="$1"
    local model="$2"
    local pname="$3"
    local pvar="PROMPT_$pname"
    local prompt="${!pvar}"
    local out="$OUT_DIR/${label}__${pname}.txt"

    echo "  -- prompt[$pname]: ${prompt:0:80}..."
    # Run REPL: pipe prompt + EOF. --max-seq 4096 gives ~800 token budget after prompt.
    # No --temp → defaults to 0.3 (slight sampling; deterministic-enough for eyeball).
    # 2>&1 captures loading logs + generated text + REPL prompt markers.
    echo "$prompt" | timeout 180 "$RUN_BIN" "$model" --max-seq 4096 > "$out" 2>&1 \
        || echo "  WARN: run exited with $? (may be timeout/normal-EOF)"
    # Show the part after first ">>> " marker (skips load logs, captures actual generation)
    echo "  --- generated text (post-prompt) ---"
    sed -n '/^>>> /,$ p' "$out" | head -50
    echo "  ---"
    echo ""
}

run_variant() {
    local label="$1"
    local model="$2"
    echo ""
    echo "==== variant: $label ($model) ===="
    for pname in "${PROMPT_NAMES[@]}"; do
        run_one_prompt "$label" "$model" "$pname"
    done
}

quantize_9b() {
    local alpha="$1"
    rm -f "$QUANT_SLOT_9B"
    "$QUANT_BIN" \
        --input "$BF16_9B" \
        --output "$QUANT_SLOT_9B" \
        --format mq4g256 \
        --imatrix "$IMATRIX_9B" \
        --awq-alpha "$alpha" \
        > "$OUT_DIR/9b-a${alpha//./_}.quantize.log" 2>&1
    if ! grep -q "^AWQ pre-scaling: ENABLED" "$OUT_DIR/9b-a${alpha//./_}.quantize.log"; then
        echo "FATAL: AWQ did not enable on 9B α=$alpha quantize" >&2
        exit 1
    fi
}

quantize_08b() {
    local alpha="$1"
    rm -f "$QUANT_SLOT_08B"
    # α=0 is a special case: with --awq-alpha 0 the binary still tries to load imatrix
    # and run AWQ math with s[j] = RMS_act^0 = 1 = identity. That's the intent here
    # (test AWQ wiring at identity).  For pure no-AWQ baseline, omit both flags.
    if [ "$alpha" = "0.0" ] || [ "$alpha" = "0" ]; then
        "$QUANT_BIN" \
            --input "$BF16_08B" \
            --output "$QUANT_SLOT_08B" \
            --format mq4g256 \
            > "$OUT_DIR/0.8b-a${alpha//./_}.quantize.log" 2>&1
    else
        "$QUANT_BIN" \
            --input "$BF16_08B" \
            --output "$QUANT_SLOT_08B" \
            --format mq4g256 \
            --imatrix "$IMATRIX_08B" \
            --awq-alpha "$alpha" \
            > "$OUT_DIR/0.8b-a${alpha//./_}.quantize.log" 2>&1
        if ! grep -q "^AWQ pre-scaling: ENABLED" "$OUT_DIR/0.8b-a${alpha//./_}.quantize.log"; then
            echo "FATAL: AWQ did not enable on 0.8B α=$alpha quantize" >&2
            exit 1
        fi
    fi
}

# ── sequence ──────────────────────────────────────────────────────────
echo "(1) 9B α=0.5 — anchor"
quantize_9b 0.5
run_variant "9b-a0_5-anchor" "$QUANT_SLOT_9B"

echo "(2) 9B α=0.6 — top candidate"
quantize_9b 0.6
run_variant "9b-a0_6-top" "$QUANT_SLOT_9B"

echo "(3) 0.8B α=0.0 — mq4-base baseline (no AWQ)"
quantize_08b 0.0
run_variant "0.8b-a0_0-mq4base" "$QUANT_SLOT_08B"

echo "(4) 0.8B α=0.55 — AWQ shipped default"
quantize_08b 0.55
run_variant "0.8b-a0_55-awq" "$QUANT_SLOT_08B"

echo ""
echo "==== DONE ===="
echo "Outputs in: $OUT_DIR"
ls -la "$OUT_DIR"/*.txt | head -20
