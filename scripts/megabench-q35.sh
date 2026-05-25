#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Qwen3.5 megabench — all models, all quants, KV modes, coherence check.
# Auto-detects the GPU; runs on whatever the host happens to be.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
INFER="$REPO/target/release/examples/infer"
MODELS_DIR="$REPO/models"
OUT_DIR="$REPO/dev/bench"
mkdir -p "$OUT_DIR"

# Detected GPU populates the report banner instead of a hardcoded
# "RX 5700 XT" line. Override per-var if rocminfo is unavailable.
. "$(dirname "$0")/_detect-gpu.sh"

RESULTS="$OUT_DIR/megabench-q35-$(date '+%Y%m%d-%H%M').md"
LONG_RESULTS="$OUT_DIR/longctx-q35-$(date '+%Y%m%d-%H%M').md"

# Prompts: simple (speed), reasoning (coherence), code (intelligence)
PROMPT_SIMPLE="Hello, how are you today?"
PROMPT_REASON="Explain why the sky is blue in exactly three sentences."
PROMPT_CODE="Write a Python function that checks if a string is a palindrome. Just the function, no explanation."
PROMPT_HARD="Compare and contrast TCP and UDP protocols. Be specific about use cases."

# Long context prompt for KV stress test
LONG_PROMPT="You are a helpful AI assistant. Here is a long passage for context:

The history of computing is a fascinating journey that spans centuries. Charles Babbage designed the Analytical Engine in 1837, which is considered the first general-purpose computer design. Ada Lovelace wrote what is recognized as the first computer program for this machine. The development of electronic computers began in the 1940s with machines like ENIAC and Colossus. The invention of the transistor in 1947 revolutionized electronics and led to smaller, faster computers. The integrated circuit, invented in 1958, further miniaturized computing. The 1970s saw the rise of personal computers with the Apple II and IBM PC. The internet, which began as ARPANET in 1969, transformed how computers communicate. Tim Berners-Lee invented the World Wide Web in 1989. The 2000s brought smartphones, cloud computing, and the rise of AI. Modern GPUs, originally designed for graphics, became essential for machine learning. AMD's RDNA architecture represents the latest evolution in GPU design, offering improved performance per watt. The development of large language models like GPT and Qwen has pushed the boundaries of what AI can accomplish. These models require massive computational resources for training but can run efficiently on consumer hardware with proper optimization techniques like quantization and efficient attention mechanisms.

The field of natural language processing has evolved dramatically. Early systems relied on rule-based approaches and simple statistical methods. The introduction of word embeddings like Word2Vec in 2013 was a major breakthrough. The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need', revolutionized the field. BERT demonstrated the power of pre-training in 2018. GPT-2 showed that scaling language models could produce remarkably coherent text. The scaling laws discovered by Kaplan et al. showed predictable relationships between model size, data, and performance.

Now, based on this context, answer the following question with specific references to the passage above: What were the key technological transitions that enabled modern AI, and how does GPU computing fit into this narrative? Be thorough and cite specific dates and inventions from the passage."

echo "=== Qwen3.5 Megabench ===" | tee "$RESULTS"
echo "GPU: $(hipfire_gpu_banner)" | tee -a "$RESULTS"
echo "Date: $(date '+%Y-%m-%d %H:%M')" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ─── Coherence check ──────────────────────────────────────
check_coherence() {
    local text="$1"
    local min_words=${2:-5}

    # Empty or very short
    local wc
    wc=$(echo "$text" | wc -w)
    if [ "$wc" -lt "$min_words" ]; then
        echo "SHORT($wc)"
        return
    fi

    # Repetition: most-repeated word (excluding common stopwords)
    local max_rep
    max_rep=$(echo "$text" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alpha:]' '\n' | \
        grep -vxE '(the|a|an|is|are|was|were|be|been|of|in|to|and|or|for|it|that|this|with|as|on|at|by|from|not|but|if|no|do|so|up|out|all|has|had|have|will|can|its|they|we|he|she|you|i|my|our|your|their|his|her)' | \
        sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
    max_rep=${max_rep:-0}

    if [ "$max_rep" -gt 15 ]; then
        echo "LOOP($max_rep)"
    elif [ "$max_rep" -gt 8 ]; then
        echo "REPET($max_rep)"
    else
        echo "OK"
    fi
}

# ─── Run a single benchmark ──────────────────────────────
run_bench() {
    local model="$1" label="$2" prompt="$3" flags="${4:-}" maxgen="${5:-256}"

    local cmd="$INFER $model $flags --no-think"
    local output
    output=$(timeout 120 bash -c "$cmd \"$prompt\" 2>&1") || output="TIMEOUT"

    local tok_s ntok text coherence
    tok_s=$(echo "$output" | grep '=== Done:' | grep -oP '[\d.]+(?= tok/s)' || echo "FAIL")
    ntok=$(echo "$output" | grep '=== Done:' | grep -oP '\d+(?= tokens in)' || echo "0")

    # Extract generated text (after "Prefill:" line, before "=== Done:" line)
    text=$(echo "$output" | sed -n '/^Prefill:/,/^=== Done:/p' | grep -v '^Prefill:\|^=== Done:\|^<think>' | head -20)
    # Fallback: if text empty, grab anything that's not stderr
    if [ -z "$text" ]; then
        text=$(echo "$output" | grep -v '^\[' | grep -v '^===' | grep -v '^Model:' | grep -v '^Text:' | grep -v '^Prompt:' | grep -v '^KV cache:' | grep -v '^GPU:' | grep -v 'pre-compiled' | grep -v '^Loading\|^Prefill\|^Vision' | head -10)
    fi

    coherence=$(check_coherence "$text" 3)
    echo "| $label | $tok_s | $ntok | $coherence |"

    # Save full output for review
    echo "--- $label ---" >> "$OUT_DIR/megabench-raw.log"
    echo "$output" >> "$OUT_DIR/megabench-raw.log"
    echo "" >> "$OUT_DIR/megabench-raw.log"
}

# ─── Phase 1: All models, default KV (Q8) ─────────────────

# ─── VRAM detection ──────────────────────────────────────
VRAM_MB=0
if command -v rocm-smi &>/dev/null; then
    VRAM_MB=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total" | grep -oP '\d+' | head -1 || echo "0")
    VRAM_MB=$((VRAM_MB / 1048576))  # bytes → MB
fi
if [ "$VRAM_MB" -eq 0 ] 2>/dev/null; then
    # Fallback: parse from kernel log or sysfs
    VRAM_MB=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1 || echo "0")
    VRAM_MB=$((VRAM_MB / 1048576))  # bytes → MB
fi
echo "Detected VRAM: ${VRAM_MB}MB" >&2

# Model list with minimum VRAM requirement in MB
# Format: filename:label:min_vram_mb
QWEN35_MODELS=(
    "qwen3.5-0.8b.q4.hfq:0.8B-Q4:1024"
    "qwen3.5-0.8b.hfq6.hfq:0.8B-HFQ6:1024"
    "qwen3.5-2b.q4.hfq:2B-Q4:2048"
    "qwen3.5-2b.hfq6.hfq:2B-HFQ6:2560"
    "qwen3.5-4b.q4.hfq:4B-Q4:3584"
    "qwen3.5-4b.hfq6.hfq:4B-HFQ6:4608"
    "qwen3.5-9b.q4.hfq:9B-Q4:6144"
    "qwen3.5-9b.hfq6.hfq:9B-HFQ6:8192"
    "qwen3.5-27b.q4.hfq:27B-Q4:15360"
    "qwen3.5-27b.hfq6.hfq:27B-HFQ6:22528"
)

echo "## Phase 1: Speed + Coherence (Q8 KV, --no-think)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
echo "| Model | tok/s | tokens | coherence |" | tee -a "$RESULTS"
echo "|-------|-------|--------|-----------|" | tee -a "$RESULTS"

echo "" > "$OUT_DIR/megabench-raw.log"

for entry in "${QWEN35_MODELS[@]}"; do
    IFS=':' read -r file label min_vram <<< "$entry"
    model="$MODELS_DIR/$file"
    if [ ! -f "$model" ]; then
        echo "| $label | MISSING | - | - |" | tee -a "$RESULTS"
        continue
    fi
    if [ "$VRAM_MB" -gt 0 ] && [ "$min_vram" -gt "$VRAM_MB" ]; then
        echo "| $label | SKIP (${min_vram}MB > ${VRAM_MB}MB VRAM) | - | - |" | tee -a "$RESULTS"
        echo ">>> Skipping $label (needs ${min_vram}MB, have ${VRAM_MB}MB)" >&2
        continue
    fi
    echo ">>> Running $label ..." >&2
    run_bench "$model" "$label (simple)" "$PROMPT_SIMPLE" "" 128 | tee -a "$RESULTS"
    run_bench "$model" "$label (reason)" "$PROMPT_REASON" "" 256 | tee -a "$RESULTS"
    run_bench "$model" "$label (code)" "$PROMPT_CODE" "" 256 | tee -a "$RESULTS"
    run_bench "$model" "$label (hard)" "$PROMPT_HARD" "" 256 | tee -a "$RESULTS"
done

echo "" | tee -a "$RESULTS"

# ─── Phase 2: KV mode comparison (4B + 9B) ────────────────

echo "## Phase 2: KV Mode Comparison" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
echo "| Model | KV Mode | tok/s | tokens | coherence |" | tee -a "$RESULTS"
echo "|-------|---------|-------|--------|-----------|" | tee -a "$RESULTS"

KV_TEST_MODELS=(
    "qwen3.5-4b.q4.hfq:4B-Q4"
    "qwen3.5-9b.q4.hfq:9B-Q4"
)
KV_MODES=(
    ":Q8 (default)"
    "--turbo4:Turbo4"
    "--turbo2:Turbo2"
    "--asym:Asym (Q8K+T4V)"
)

for entry in "${KV_TEST_MODELS[@]}"; do
    file="${entry%%:*}"
    label="${entry##*:}"
    model="$MODELS_DIR/$file"
    [ -f "$model" ] || continue
    for kv_entry in "${KV_MODES[@]}"; do
        flags="${kv_entry%%:*}"
        kv_label="${kv_entry##*:}"
        echo ">>> Running $label $kv_label ..." >&2
        run_bench "$model" "$label" "$PROMPT_HARD" "$flags" 256 | sed "s/| $label/| $label | $kv_label/" | tee -a "$RESULTS"
    done
done

echo "" | tee -a "$RESULTS"

# ─── Phase 3: Long Context KV Stress ──────────────────────

echo "## Phase 3: Long Context (Progressive KV Stress)" | tee "$LONG_RESULTS"
echo "" | tee -a "$LONG_RESULTS"
echo "| Model | KV Mode | tok/s | tokens | coherence |" | tee -a "$LONG_RESULTS"
echo "|-------|---------|-------|--------|-----------|" | tee -a "$LONG_RESULTS"

LONG_CTX_MODEL="$MODELS_DIR/qwen3.5-4b.q4.hfq"
if [ -f "$LONG_CTX_MODEL" ]; then
    for kv_entry in "${KV_MODES[@]}"; do
        flags="${kv_entry%%:*}"
        kv_label="${kv_entry##*:}"
        echo ">>> Long context: 4B-Q4 $kv_label ..." >&2
        run_bench "$LONG_CTX_MODEL" "4B-Q4" "$LONG_PROMPT" "$flags" 512 | sed "s/| 4B-Q4/| 4B-Q4 | $kv_label/" | tee -a "$LONG_RESULTS"
    done
else
    echo "4B-Q4 model not found, skipping long context test" >&2
fi

# Also test 9B with long context
LONG_CTX_9B="$MODELS_DIR/qwen3.5-9b.q4.hfq"
if [ -f "$LONG_CTX_9B" ]; then
    for kv_entry in "" "--turbo4:Turbo4" "--asym:Asym (Q8K+T4V)"; do
        [ -z "$kv_entry" ] && continue
        flags="${kv_entry%%:*}"
        kv_label="${kv_entry##*:}"
        echo ">>> Long context: 9B-Q4 $kv_label ..." >&2
        run_bench "$LONG_CTX_9B" "9B-Q4" "$LONG_PROMPT" "$flags" 512 | sed "s/| 9B-Q4/| 9B-Q4 | $kv_label/" | tee -a "$LONG_RESULTS"
    done
fi

echo "" | tee -a "$LONG_RESULTS"

# Append long context results to main results
cat "$LONG_RESULTS" >> "$RESULTS"

echo "" | tee -a "$RESULTS"
echo "=== Megabench complete ===" | tee -a "$RESULTS"
echo "Results: $RESULTS" >&2
echo "Raw output: $OUT_DIR/megabench-raw.log" >&2
