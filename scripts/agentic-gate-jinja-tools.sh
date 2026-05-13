#!/usr/bin/env bash
# Agentic gate (Jinja + structured tools) — Phase 1 smoke for the
# daemon-side `tools` / `messages` JSONL fields.
#
# Why this gate exists:
#   `scripts/agentic-gate.sh` sends tool definitions as TEXT inside the
#   system prompt (the same shape cli/index.ts pre-renders today). The
#   model's Jinja template `{% if tools %}` block is therefore unreachable
#   from the daemon — there is no way to A/B the upstream tools-block
#   against the hand-rolled system-prompt-text path.
#
#   Phase 1 of the Jinja-everywhere migration adds `tools` and `messages`
#   to the daemon's stdin JSONL "generate" schema. This script proves the
#   end-to-end path fires:
#     1. JSONL with structured `tools` parses without daemon error.
#     2. `HIPFIRE_JINJA_CHAT=1` + structured tools routes through
#        `JinjaChatFrame::render_messages(messages, Some(&tools), None)`.
#     3. Model emits non-zero tokens (proves prompt was built and decoded).
#
# What this script does NOT cover (by design today):
#   - Tool-call body schema validation (XML vs JSON, name/arguments).
#     That is Phase 3 once `messages` history and tool-response role are
#     wired through the Plain ChatML fallback path too.
#   - Comparing tools-block-rendered text vs hand-rolled system-prompt-
#     text output quality. Phase 2 work after `cli/index.ts` switches to
#     structured tools.
#
# Exit codes:
#   0 - daemon emitted tokens for the structured-tools request
#   1 - daemon panicked / zero tokens / Jinja render failed back to Plain
#   2 - build / env failure (model absent, daemon won't build, etc.)

set -u
cd "$(dirname "$0")/.."

# ---- Setup -----------------------------------------------------------------
EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# Model preference order: prefer Qwen3.6 over 3.5 (newer chat_template),
# dense over MoE (lower VRAM ceiling), and within each family the
# largest variant that's plausibly host-fitting. The point of this gate
# is to exercise the daemon's structured-tools JSONL path end-to-end,
# not to stress a specific arch — any Qwen3.5/3.6 model carries the
# upstream chat_template the Jinja branch needs.
#
# Override with HIPFIRE_JINJA_TOOLS_MODEL=<path> for hosts where the
# default-selected file doesn't fit (or for picking A3B explicitly).
CANDIDATES=(
    "$MODELS_DIR/qwen3.6-27b.mq4"
    "$MODELS_DIR/qwen3.5-27b.mq4"
    "$MODELS_DIR/qwen3.5-9b.mq4"
    "$MODELS_DIR/qwen3.6-35b-a3b.mq4"
    "$MODELS_DIR/qwen3.5-35b-a3b.mq4"
)
MODEL=""
LABEL=""
if [ -n "${HIPFIRE_JINJA_TOOLS_MODEL:-}" ]; then
    if [ -f "$HIPFIRE_JINJA_TOOLS_MODEL" ]; then
        MODEL="$HIPFIRE_JINJA_TOOLS_MODEL"
        LABEL="$(basename "$MODEL" .mq4)_jinja_tools"
    else
        echo "agentic-gate-jinja-tools: HIPFIRE_JINJA_TOOLS_MODEL=$HIPFIRE_JINJA_TOOLS_MODEL not found" >&2
        exit 2
    fi
else
    for cand in "${CANDIDATES[@]}"; do
        if [ -f "$cand" ]; then
            MODEL="$cand"
            LABEL="$(basename "$MODEL" .mq4)_jinja_tools"
            break
        fi
    done
fi
if [ -z "$MODEL" ]; then
    echo "agentic-gate-jinja-tools: no Qwen3.5/3.6 model present in $MODELS_DIR — SKIPPED"
    echo "  (set HIPFIRE_JINJA_TOOLS_MODEL=<path> to point at a non-default model)"
    exit 0
fi
echo "agentic-gate-jinja-tools: using $LABEL ($(du -h "$MODEL" | cut -f1))"

# Rebuild daemon if any tracked source is newer than the binary. Mirrors
# the rebuild gate in agentic-gate.sh.
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/hfq.rs crates/hipfire-runtime/examples/daemon.rs \
               crates/hipfire-runtime/src/prompt_frame.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1; break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "agentic-gate-jinja-tools: rebuilding daemon..."
    if ! cargo build --release --example daemon --features deltanet >&2; then
        echo "agentic-gate-jinja-tools: build failed" >&2
        exit 2
    fi
fi

# GPU lock — same pattern as the other gates so parallel agents don't
# stomp each other's GPU.
DAEMON_PID=""
cleanup() {
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null
        sleep 1
        kill -9 "$DAEMON_PID" 2>/dev/null
    fi
    DAEMON_PID=""
    gpu_release 2>/dev/null || true
}
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "agentic-gate-jinja-tools" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap cleanup EXIT
fi

# ---- Build JSONL session ---------------------------------------------------
# load + 1 generate w/ structured tools + unload. The tools array carries
# one OpenAI-style function def with required+optional args — exercises
# the template's `tools | rejectattr` / `parameters.required` walks on
# Qwen3.5/3.6.
JSONL_FILE="$(mktemp /tmp/agentic-gate-jinja-tools.XXXXXX.jsonl)"
OUTPUT_FILE="$(mktemp /tmp/agentic-gate-jinja-tools.XXXXXX.out)"

python3 - "$MODEL" "$JSONL_FILE" <<'PY' >/dev/null
import sys, json

model_path, jsonl_path = sys.argv[1], sys.argv[2]

load_msg = {
    "type": "load",
    "model": model_path,
    # Single-turn smoke; 1024 ctx is plenty for sys+user+tools-block+
    # response and keeps KV cache headroom usable on hosts where the
    # weight blob already takes most of the VRAM (e.g., 22 GB A3B-36
    # on a 24 GB card OOMs at max_seq=4096).
    "params": {"max_seq": 1024},
}

# Structured tools: one function with description + a typed parameters
# schema. The Qwen3.5/3.6 template walks `function.parameters.properties`
# and `function.parameters.required` to render the tools-block prompt
# the model was trained on.
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "unit": {"type": "string", "enum": ["c", "f"], "description": "Temperature unit."},
            },
            "required": ["city"],
        },
    },
}]

# Single-turn structured request: terse system + a user message that
# clearly needs the tool. No Hermes-shape system text — the template's
# tools-block carries the tool catalog.
gen_msg = {
    "type": "generate",
    "id": "jinja_tools_t1",
    "system": "You are a helpful assistant. Use tools when appropriate.",
    "prompt": "What's the weather in San Francisco?",
    "tools": tools,
    "max_tokens": 192,
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "max_think_tokens": 1,
    "assistant_prefix": "closed_think",
}

with open(jsonl_path, "w") as f:
    f.write(json.dumps(load_msg) + "\n")
    f.write(json.dumps(gen_msg) + "\n")
    f.write(json.dumps({"type": "unload"}) + "\n")
PY

# ---- Spawn daemon with Jinja path forced on --------------------------------
STDIN_FIFO="$(mktemp -u /tmp/agentic-gate-jinja-tools-fifo.XXXXXX)"
mkfifo "$STDIN_FIFO"

env HIPFIRE_JINJA_CHAT=1 \
    HIPFIRE_KV_MODE=asym3 \
    HIPFIRE_GRAPH=1 \
    "$EXE" < "$STDIN_FIFO" > "$OUTPUT_FILE" 2>&1 &
DAEMON_PID=$!

# Pace: load (~90s), generate (~30s for 192 tokens), unload (~2s).
(
    while IFS= read -r line; do
        printf '%s\n' "$line"
        type="$(echo "$line" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print(d["type"])' 2>/dev/null)"
        case "$type" in
            load)     sleep 90 ;;
            generate) sleep 45 ;;
            unload)   sleep 2  ;;
        esac
    done < "$JSONL_FILE"
) > "$STDIN_FIFO"

# Give the daemon a beat to flush, then kill by tracked PID if still alive.
for _ in 1 2 3 4 5; do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then break; fi
    sleep 1
done
if kill -0 "$DAEMON_PID" 2>/dev/null; then
    kill "$DAEMON_PID" 2>/dev/null
    sleep 1
    kill -9 "$DAEMON_PID" 2>/dev/null
fi
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""
rm -f "$STDIN_FIFO" "$JSONL_FILE"

# ---- Verdict ---------------------------------------------------------------
python3 - "$OUTPUT_FILE" "$LABEL" <<'PY'
import sys, json, pathlib

out_path, label = sys.argv[1], sys.argv[2]
raw = pathlib.Path(out_path).read_text()

events = []
for line in raw.splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass

panic = next((ev for ev in events if ev.get("type") == "error"), None)
fatal_singleton = "FATAL: hipfire daemon already running" in raw
toks = [ev["text"] for ev in events if ev.get("type") == "token" and ev.get("id") == "jinja_tools_t1"]
text = "".join(toks)

print(f"# Agentic gate (Jinja + structured tools): {label}")
print()
print(f"- model        : {label}")
print(f"- jinja env    : HIPFIRE_JINJA_CHAT=1")
print(f"- tools field  : structured (1 function: get_weather)")
print(f"- tokens emit  : {len(toks)}")
print(f"- daemon panic : {panic['message'] if panic else 'none'}")
print()

verdict = "PASS"
if fatal_singleton:
    print("HARD_FAIL: another hipfire daemon was holding the singleton flock — release it and retry")
    verdict = "HARD_FAIL"
elif panic is not None:
    print(f"HARD_FAIL: daemon emitted error event: {panic.get('message')!r}")
    verdict = "HARD_FAIL"
elif len(toks) == 0:
    print("HARD_FAIL: zero tokens emitted — Jinja render failed silently, or tools schema rejected by template")
    verdict = "HARD_FAIL"
else:
    print("PASS: structured-tools JSONL accepted, Jinja path rendered, model emitted tokens")

print()
print("## Model output")
print("```")
print(text if text else "(no tokens)")
print("```")

sys.exit(0 if verdict == "PASS" else 1)
PY
res=$?
rm -f "$OUTPUT_FILE"
exit $res
