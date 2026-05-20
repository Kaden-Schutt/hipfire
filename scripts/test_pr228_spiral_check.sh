#!/bin/bash
# A/B test: confirms the <think> spiral on Qwen3.6-35B-A3B (originally
# documented as PR #228's "known issue") is dissolved by the
# repeat_penalty 1.3 → 1.0 default fix in commit 9b4ab74a.
#
# Two runs of the same train-pursuit prompt at temp=0, max_tokens=800,
# both with the corrected GemmaRMSNorm:
#
#   Run A: repeat_penalty=1.0 (current default).  EXPECTATION: coherent.
#   Run B: repeat_penalty=1.3 (prior default).    EXPECTATION: spiral.
#
# Run B reproduces the self-doubt / number-hallucination pattern that
# the 9b4ab74a commit message describes ("Wait, re-reading prompt...",
# hallucinated gap value, repeated Step B/C/D). Run A produces a clean
# step-by-step trajectory to the correct answer (t = 120/30 = 4 hours).
#
# The HIPFIRE_QWEN_MOE_FINAL_NORM_RAW=1 fallback env var was removed
# (no longer load-bearing) in the same commit that this test was
# written for; if you're running this against an older daemon that
# still has the env var, the workaround behavior is documented in
# the commit log of the removal.
set -uo pipefail

export PATH=/opt/rocm-7.12/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.12/lib:${LD_LIBRARY_PATH:-}

DAEMON="${HIPFIRE_DAEMON:-$HOME/.hipfire/bin/daemon}"
MODEL="${HIPFIRE_MODEL:-/local/hipfire/qwen3.6-35b-a3b.mq4}"
PROMPT='A train leaves Station A traveling at 60 km/h. Two hours later, a second train leaves Station A on the same track traveling at 90 km/h. How long after the second train departs will it catch up to the first? Show your reasoning step by step.'
OUT_DIR="/tmp/pr228-spiral-check-$$"
mkdir -p "$OUT_DIR"

run_one() {
    local label="$1"  # filename suffix
    local rp="$2"     # repeat_penalty
    local env_pre="$3"  # extra env vars
    local in_file="$OUT_DIR/in_${label}.jsonl"
    local out_file="$OUT_DIR/out_${label}.log"

    local prompt_json
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$PROMPT")

    # JSONL invocation pattern matches scripts/coherence-gate.sh:201-202
    # (single-turn `prompt` field, manual ChatML scaffolding inside the
    # daemon, no Jinja env-gate). max_tokens=800 matches the gate's
    # moe36-sheep row (line 126) and the gate-run report on this PR that
    # observed a full reasoning trace + </think> close at that budget.
    cat > "$in_file" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":$prompt_json,"temperature":0.0,"max_tokens":800,"repeat_penalty":$rp}
{"type":"unload"}
JL

    echo "=== Run $label: repeat_penalty=$rp $env_pre ==="
    eval "$env_pre" timeout 360 "$DAEMON" < "$in_file" > "$out_file" 2>&1
    local ec=$?
    local n_tokens
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    local panic
    panic=$(grep -aE 'panicked|FATAL|thread.*panicked' "$out_file" | head -1)
    echo "exit=$ec tokens=$n_tokens panic=${panic:-none}"

    # Assemble token text in order; classify spiral if generation truncated
    # to 400 with empty content OR with extreme single-token repetition.
    python3 <<PY
import json,re,sys
toks=[]
text_parts=[]
with open("$out_file","rb") as f:
    for line in f:
        try:
            o=json.loads(line.decode("utf-8","replace"))
        except Exception:
            continue
        if o.get("type")=="token":
            t=o.get("text","")
            text_parts.append(t)
text="".join(text_parts)
think_open=text.find("<think>") if "<think>" in text else 0
think_close=text.find("</think>")
think_body= text[think_open:think_close] if think_close>think_open else text
visible= text[think_close+len("</think>"):] if think_close>-1 else ""
print(f"  total_chars={len(text)}  think_body={len(think_body)}  visible={len(visible)}")
print(f"  closed_think={'YES' if think_close>-1 else 'NO'}")
# 3-gram density inside think body
def density(s):
    words=s.split()
    if len(words)<6: return 0.0
    grams=[" ".join(words[i:i+3]) for i in range(len(words)-2)]
    return 1.0 - len(set(grams))/len(grams)
d=density(think_body)
print(f"  3gram_repeat_density={d:.2f}")
# Unique token-text ratio over first 128 emitted text chunks
toks_first=text_parts[:128]
if toks_first:
    uniq=len(set(toks_first))/len(toks_first)
    most=max(set(toks_first), key=lambda x: toks_first.count(x))
    most_freq=toks_first.count(most)/len(toks_first)
    print(f"  first128_uniq_token_ratio={uniq:.2f} max_freq={most_freq:.2f}")
print("  --- think body (first 600 chars) ---")
print(think_body[:600])
print("  --- visible answer (first 400 chars) ---")
print(visible[:400])
PY
    echo
}

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi
if [ ! -x "$DAEMON" ]; then
    echo "ERROR: daemon binary not found: $DAEMON" >&2
    exit 1
fi

md5sum "$DAEMON" "$MODEL"
echo

# Run A trips the n-gram loop guard on legitimate "    *   " bullet-list
# formatting after ~400 tokens — a false positive on coherent reasoning,
# unrelated to the spiral being tested. Disable the guard so the full
# trajectory (think → </think> → final answer) is visible.
NOGUARD="export HIPFIRE_NGRAM_LOOP_THRESHOLD=0;"

run_one "A_rp1.0" "1.0" "$NOGUARD"
run_one "B_rp1.3" "1.3" "$NOGUARD"

echo "Outputs: $OUT_DIR"
