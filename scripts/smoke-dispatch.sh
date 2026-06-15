#!/usr/bin/env bash
# Smoke test for formats that must work before dispatch-unification ships.
# Covers: DS4 MQ2Lloyd, Qwen3.6 MQ4, MQ6 (9B proxy), PARO (35B A3B).
# Usage: ./scripts/smoke-dispatch.sh
# Exit code: 0 = all pass, 1 = one or more hard errors

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/daemon"
MODELS="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
PROMPT="A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."
MAX_TOKENS=300

pass=0
fail=0
skip=0

run_case() {
    local label="$1"
    local model_path="$2"
    shift 2
    local extra_env=("$@")

    if [ ! -e "$model_path" ]; then
        echo "SKIP  $label (not found: $model_path)"
        skip=$((skip + 1))
        return
    fi

    local in_file out_file
    in_file=$(mktemp /tmp/smoke_in_XXXXXX.jsonl)
    out_file=$(mktemp /tmp/smoke_out_XXXXXX.log)

    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$PROMPT")
    cat > "$in_file" <<JL
{"type":"load","model":"${model_path}","params":{"max_seq":2048}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":${MAX_TOKENS},"repeat_penalty":1.05}
{"type":"unload"}
JL

    local t0 t1 wall ec n_tokens panic status
    t0=$(date +%s.%N)
    env "${extra_env[@]}" timeout 300 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    n_tokens=$(grep -ac '"type":"token"' "$out_file" 2>/dev/null || echo 0)
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL' "$out_file" | head -1 || true)

    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        status="FAIL"
        fail=$((fail + 1))
        echo "FAIL  $label  (exit=$ec tokens=$n_tokens wall=${wall}s)"
        grep -aE 'panicked|error\[|Error|FATAL' "$out_file" | head -5 | sed 's/^/      /'
    else
        # Extract last meaningful token text for quick sanity
        last=$(grep '"type":"token"' "$out_file" | tail -3 | python3 -c "
import sys,json
lines=sys.stdin.readlines()
toks=[json.loads(l).get('text','') for l in lines if l.strip()]
print(''.join(toks).strip()[:60])
" 2>/dev/null || true)
        echo "PASS  $label  tokens=$n_tokens wall=${wall}s  ...${last}"
        pass=$((pass + 1))
    fi

    rm -f "$in_file" "$out_file"
}

# ── Build daemon if stale ─────────────────────────────────────────────────
if [ ! -x "$EXE" ]; then
    echo "Building daemon..."
    cargo build --release --example daemon --features deltanet 2>&1 | tail -3
fi

echo "=== smoke-dispatch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git rev-parse --short HEAD 2>/dev/null) ==="
echo

# 1. DS4 MQ2Lloyd
DS4_BASE="$MODELS/deepseek-v4-flash.mq2lloyd"
DS4_MTP="$MODELS/deepseek-v4-flash-mtp.mq2lloyd"
if [ -f "$DS4_MTP" ]; then
    run_case "DS4 MQ2Lloyd (+MTP)" "$DS4_BASE" "HIPFIRE_DEEPSEEK4_MTP_ADDON=$DS4_MTP"
else
    run_case "DS4 MQ2Lloyd" "$DS4_BASE"
fi

# 2. Qwen3.6 MQ4
run_case "Qwen3.6-27B MQ4" "$MODELS/qwen3.6-27b.mq4"

# 3. MQ6 — use 9B as format proxy (no qwen3.6-27b-mq6 in registry)
run_case "Qwen3.5-9B MQ6" "$MODELS/qwen3.5-9b.mq6"

# 4. PARO (Qwen3.6-35B-A3B shisa packed)
run_case "PARO Qwen3.6-35B-A3B" "$MODELS/shisa-Qwen3.6-35B-A3B-PARO-packed"

echo
echo "=== Results: ${pass} pass  ${fail} fail  ${skip} skip ==="
[ "$fail" -eq 0 ]
