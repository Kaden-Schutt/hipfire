#!/bin/bash
# Full profiling benchmark: hipfire vs llama.cpp, 3 runs each, both models, short + long
# Outputs JSON per run, then compiles to summary
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="bench/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LLAMACPP="$HOME/llama.cpp/build/bin"
export LD_LIBRARY_PATH="${LLAMACPP}:${LD_LIBRARY_PATH:-}"

MODEL_06B="models/qwen3-0.6b-hfq4.hfq"
MODEL_8B="models/qwen3-8b.q4.hfq"
GGUF_06B="$HOME/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf"
GGUF_8B="$HOME/llama.cpp/models/Qwen3-8B-Q4_K_M.gguf"

RUNS=3
SHORT_TOKENS=20
LONG_TOKENS=128

echo "=== hipfire profiling benchmark ==="
echo "Timestamp: $TIMESTAMP"
echo "Runs per config: $RUNS"
echo ""

# Stop ollama to free GPU
echo "Stopping ollama..."
echo 150794 | sudo -S systemctl stop ollama 2>/dev/null || true
sleep 2

# ─── hipfire profiling ───
for model_name in "0.6b" "8b"; do
    if [ "$model_name" = "0.6b" ]; then
        MODEL="$MODEL_06B"
    else
        MODEL="$MODEL_8B"
    fi

    for n_tok in $SHORT_TOKENS $LONG_TOKENS; do
        for run in $(seq 1 $RUNS); do
            OUT="$RESULTS_DIR/hipfire_${model_name}_${n_tok}tok_run${run}_${TIMESTAMP}.json"
            echo "hipfire ${model_name} ${n_tok}tok run${run}..."
            cargo run --release --example profile_layers -- "$MODEL" "$n_tok" > "$OUT" 2>/dev/null
            sleep 2  # KFD cooldown
        done
    done
done

# ─── llama.cpp benchmarks ───
echo ""
echo "=== llama.cpp benchmarks ==="
for model_name in "0.6b" "8b"; do
    if [ "$model_name" = "0.6b" ]; then
        GGUF="$GGUF_06B"
    else
        GGUF="$GGUF_8B"
    fi

    for n_tok in $SHORT_TOKENS $LONG_TOKENS; do
        LLAMA_OUT="$RESULTS_DIR/llamacpp_${model_name}_${n_tok}tok_${TIMESTAMP}.txt"
        echo "llama.cpp ${model_name} pp26 tg${n_tok}..."
        "$LLAMACPP/llama-bench" -m "$GGUF" -p 26 -n "$n_tok" -r $RUNS -ngl 99 2>&1 | tee "$LLAMA_OUT"
        echo ""
    done
done

echo ""
echo "=== All results in $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"/*"$TIMESTAMP"*
echo ""
echo "Run: python3 bench/compile_results.py $RESULTS_DIR $TIMESTAMP"
