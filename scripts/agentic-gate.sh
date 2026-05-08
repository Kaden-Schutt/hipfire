#!/usr/bin/env bash
# Agentic gate - tool-call shape regression battery for A3B variants.
#
# Why this gate exists:
#   coherence-gate.sh runs a tool-call test row but only checks for `<tool_call>`
#   tag presence and `<|im_start|>` leakage in visible text. It uses a 50-token
#   system prompt. Issue #87's auto-MMQ regression produced clean output on
#   short prompts but corrupt tool-call JSON on long agent-shape prompts.
#   This gate guards that regression class with realistic 780-1300 token
#   system contexts (Pi-style + Hermes-style) and machine-evaluated JSON
#   structural validation.
#
#   Value-add over coherence-gate.sh --full (which already runs A3B sheep):
#     1. Prompt-length-sensitive MMQ regression cover (the #87 class)
#     2. Tool-call structural validation (JSON.parse + schema match)
#
# What this gate does NOT cover (by design, today):
#   - Jinja chat-template rendering. Master hardcodes AssistantPrefix::Plain
#     at 4 daemon sites; there is no HIPFIRE_JINJA_CHAT toggle on master.
#     PR #175 ships that toggle (gated 1=on, default off + Plain). When #175
#     lands, this gate should grow a `jinja=1` axis so we exercise both
#     framing paths. Until then, every cell uses the Plain scaffold.
#   - OpenAI HTTP path. Cells drive the daemon over stdin JSONL; the
#     `serve` HTTP path runs additional Node-side Jinja rendering that
#     this harness skips. OpenAI-shaped tool-call regressions need a
#     daemon-up + curl harness; defer.
#
# Modes:
#   ./scripts/agentic-gate.sh                # full: 2 models * 4 cells = 8 cells, ~5 min
#   ./scripts/agentic-gate.sh --fast         # 1 cell, ~2 min - used by pre-commit
#   ./scripts/agentic-gate.sh --self-check   # detector rot guard, <1s
#
# Exit codes:
#   0 - battery ran clean
#   1 - hard error (panic, zero tokens, JSON parse fail, special-token leak,
#       stacked openers, or > 1/N cells soft-warned)
#   2 - build / env / detector-self-check failure
#
# Skip semantics (CI-safe):
#   - Both A3B models absent           -> exit 0 with SKIPPED message
#   - One model absent                 -> run the present one's cells, log skip
#   - Model exceeds host VRAM          -> treat as absent (path-blank); skips
#                                         silently otherwise → zero-token hard-fail
#   - HIPFIRE_SKIP_AGENTIC_GATE=1      -> exit 0 immediately
#   - HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK=1 -> bypass VRAM-headroom skip
#                                            (e.g., registry minimums too conservative
#                                            for your host)
#
# Report destination: /tmp/agentic-gate-<timestamp>.md (or $HIPFIRE_AGENTIC_GATE_OUT)

set -u
cd "$(dirname "$0")/.."

MODE="full"
while [ $# -gt 0 ]; do
    case "$1" in
        --fast)        MODE="fast"; ;;
        --self-check)  MODE="self-check"; ;;
        -h|--help)
            sed -n '3,32p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "${HIPFIRE_SKIP_AGENTIC_GATE:-0}" = "1" ]; then
    echo "agentic-gate: HIPFIRE_SKIP_AGENTIC_GATE=1 - skipping"
    exit 0
fi

# ---- Self-check mode (detector rot guard) ----------------------------------
# Construct a payload that intentionally trips every detector, run them, and
# assert each fires. Cheap (<1s); guards the gate against silent rot if the
# detector regexes drift away from real failure shapes.
if [ "$MODE" = "self-check" ]; then
    python3 - <<'PY'
import sys, json, re

# Two synthetic payloads cover all 4 detectors. parse-fail and schema-violation
# are mutually exclusive on a single payload (one requires successful parse, the
# other forbids it), so we test them on separate inputs.

# Trips: stacked_openers + special_token_leak + json_parse_fail
PAYLOAD_CORRUPT = '''\
Here is my response:

<tool_call>
<tool_call>
{"arguments": {"path": "/tmp/x"<|im_start|>}}
</tool_call><|im_end|>
'''

# Trips: schema_violation only (JSON parses, but missing required `name` field)
PAYLOAD_SCHEMA_BAD = '''\
<tool_call>
{"arguments": {"path": "/tmp/x"}}
</tool_call><|im_end|>
'''

def detect_stacked_openers(text):
    return bool(re.search(r"<tool_call>\s*<tool_call>", text))

def detect_special_token_leak(body):
    return any(tok in body for tok in ("<|im_start|>", "<|endoftext|>"))

def detect_json_parse_fail(body):
    try:
        json.loads(body)
        return False
    except json.JSONDecodeError:
        return True

def detect_schema_violation(body):
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return False  # parse-fail is its own detector; don't double-count
    return not (isinstance(obj, dict) and "name" in obj and "arguments" in obj)

def body_of(payload):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", payload, re.S)
    body = m.group(1) if m else ""
    # Strip any nested opener before JSON tests so stacked_openers stays
    # independent.
    return re.sub(r"^<tool_call>\s*", "", body)

corrupt_body = body_of(PAYLOAD_CORRUPT)
schema_body = body_of(PAYLOAD_SCHEMA_BAD)

results = {
    "stacked_openers":     detect_stacked_openers(PAYLOAD_CORRUPT),
    "special_token_leak":  detect_special_token_leak(corrupt_body),
    "json_parse_fail":     detect_json_parse_fail(corrupt_body),
    "schema_violation":    detect_schema_violation(schema_body),
}

failed = [k for k, v in results.items() if not v]
print("self-check results:")
for k, v in results.items():
    print(f"  {k:20s} {'fired' if v else 'MISSED'}")
if failed:
    print(f"\nself-check FAILED: detectors did not fire on synthetic corrupt: {failed}", file=sys.stderr)
    print("detector rot suspected - review the regexes and update before merging.", file=sys.stderr)
    sys.exit(2)
print("\nself-check passed: all 4 detectors fired against synthetic corrupt payloads")
PY
    exit $?
fi

# ---- Setup -----------------------------------------------------------------
EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_AGENTIC_GATE_OUT:-/tmp/agentic-gate-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

A3B_35="$MODELS_DIR/qwen3.5-35b-a3b.mq4"
A3B_36="$MODELS_DIR/qwen3.6-35b-a3b.mq4"
PI_SYS="benchmarks/prompts/agentic_pi_system.txt"
HERMES_SYS="benchmarks/prompts/agentic_hermes_system.txt"
USER_READ="benchmarks/prompts/agentic_user_read.txt"

# ---- VRAM headroom check ---------------------------------------------------
# Models declare min_vram_gb in cli/registry.json. If the host can't meet
# that minimum, the model loads but silently OOMs during prefill and emits
# zero tokens (looks like a tool-call regression — it isn't). Treat such
# models as absent so the existing skip path covers this case too.
#
# Threshold semantics: min_vram_gb is interpreted as a strict floor expressed
# in GiB (1024^3). Daemon overhead (KV cache, graph capture, scratch) is
# assumed pre-included in the registry value — A3B-35B is an 18.7 GB blob
# but declares min_vram_gb=22, i.e. ~3.3 GiB margin. We do NOT add a
# secondary +1 GB headroom because (a) it'd compound, and (b) the registry
# is the single source of truth for "needs N GiB". Comparison is in bytes
# to avoid GiB-truncation false-positives on cards that report slightly
# under their advertised size (e.g. a 24 GB card reading 23.98 GiB).
#
# Disable with HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK=1 (e.g., on hosts where
# the registry minimums are conservative).
REGISTRY="cli/registry.json"
# Multi-card systems take the max VRAM across all DRM cards. iGPUs report
# zero / a sliver, so max correctly picks the dGPU. On dual-dGPU hosts this
# is optimistic — we don't know which card the daemon will use.
VRAM_BYTES=0
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v="$(cat "$f" 2>/dev/null)" || continue
    [ "$v" -gt "$VRAM_BYTES" ] && VRAM_BYTES="$v"
done
VRAM_GB=$(( VRAM_BYTES / 1073741824 ))
if [ "$VRAM_BYTES" -eq 0 ] && [ "${HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK:-0}" != "1" ]; then
    echo "agentic-gate: VRAM detection unavailable (no /sys/class/drm/card*/device/mem_info_vram_total) - skipping VRAM check"
fi

model_fits_vram() {
    local model_path="$1"
    [ "${HIPFIRE_AGENTIC_GATE_NO_VRAM_CHECK:-0}" = "1" ] && return 0
    [ -f "$REGISTRY" ] || return 0
    [ "$VRAM_BYTES" -gt 0 ] || return 0
    # python3 is a hard script-wide dep (self-check + JSONL builder). No guard.

    local fname min_gb min_bytes
    fname="$(basename "$model_path")"
    min_gb="$(python3 -c '
import json, sys
fname = sys.argv[1]
with open(sys.argv[2]) as fh:
    data = json.load(fh)
for entry in data.get("models", {}).values():
    if isinstance(entry, dict) and entry.get("file") == fname:
        v = entry.get("min_vram_gb")
        if v is not None:
            print(v)
        break
' "$fname" "$REGISTRY" 2>/dev/null)"
    [ -n "$min_gb" ] || return 0

    min_bytes=$(( min_gb * 1073741824 ))
    if [ "$VRAM_BYTES" -lt "$min_bytes" ]; then
        echo "agentic-gate: $fname needs ${min_gb} GB VRAM, host has ${VRAM_GB} GB - skipping cell"
        return 1
    fi
    return 0
}

# Path-blank when over budget so the existing `[ -f "$A3B_..." ]` checks
# downstream naturally treat these models as absent (`[ -f "" ]` is false).
A3B_35_VRAM_SKIP=0
A3B_36_VRAM_SKIP=0
if [ -f "$A3B_35" ] && ! model_fits_vram "$A3B_35"; then A3B_35=""; A3B_35_VRAM_SKIP=1; fi
if [ -f "$A3B_36" ] && ! model_fits_vram "$A3B_36"; then A3B_36=""; A3B_36_VRAM_SKIP=1; fi

# Skip-on-absence: both models missing -> exit 0. Distinguish absent-on-disk
# from skipped-for-VRAM so the operator can tell which case fired.
if [ ! -f "$A3B_35" ] && [ ! -f "$A3B_36" ]; then
    if [ "$A3B_35_VRAM_SKIP" = "1" ] || [ "$A3B_36_VRAM_SKIP" = "1" ]; then
        echo "agentic-gate: all A3B models absent or exceed host VRAM (${VRAM_GB} GB) - SKIPPED"
    else
        echo "agentic-gate: A3B models absent ($MODELS_DIR/qwen3.{5,6}-35b-a3b.mq4) - SKIPPED"
    fi
    exit 0
fi

# Required fixtures
for f in "$PI_SYS" "$HERMES_SYS" "$USER_READ"; do
    if [ ! -f "$f" ]; then
        echo "agentic-gate: required fixture missing: $f" >&2
        exit 2
    fi
done

# Rebuild daemon if any tracked source is newer than the binary.
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/hfq.rs crates/hipfire-runtime/examples/daemon.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1; break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "agentic-gate: rebuilding daemon..."
    if ! cargo build --release --example daemon --features deltanet >&2; then
        echo "agentic-gate: build failed" >&2
        exit 2
    fi
fi

# Concurrency policy: the gate uses the daemon's existing singleton flock
# at $HOME/.hipfire/daemon.pid (daemon.rs:171). If a user daemon is already
# running, the gate's daemon will exit with FATAL; we detect that in the
# per-cell parser and surface it as a hard fail. We do NOT override $HOME
# or rm the pid file — both would either bypass the singleton (allowing two
# 35B daemons on one GPU) or break a parallel user daemon's lock. When the
# gate's daemon dies via SIGKILL (cleanup path), the kernel auto-releases
# the flock; the stale pid-file content is harmless (next daemon truncates
# and overwrites at startup).
DAEMON_PID=""

cleanup() {
    # Kill ONLY the daemon we spawned, by tracked PID. Never pkill -f.
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null
        # Give it a beat, then SIGKILL if still alive.
        sleep 1
        kill -9 "$DAEMON_PID" 2>/dev/null
    fi
    DAEMON_PID=""
    gpu_release 2>/dev/null || true
}

# GPU lock
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "agentic-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap cleanup EXIT
fi

# ---- Build cell list -------------------------------------------------------
# Each cell: model | system_fixture | thinking_clamp_bool | multi_turn_bool | label
build_cells() {
    local model="$1"
    local prefix="$2"
    local fast_only="$3"  # 1 = emit only the fast cell for this model
    if [ "$fast_only" = "1" ]; then
        echo "$model|$HERMES_SYS|0|0|${prefix}_hermes_unclamped"
        return
    fi
    # Full mode: 4 cells per model
    echo "$model|$PI_SYS|0|0|${prefix}_pi_unclamped"
    echo "$model|$PI_SYS|1|0|${prefix}_pi_clamped"
    echo "$model|$HERMES_SYS|0|0|${prefix}_hermes_unclamped"
    echo "$model|$HERMES_SYS|0|1|${prefix}_hermes_multiturn"
}

CELLS=()
if [ "$MODE" = "fast" ]; then
    # Fast mode: prefer 3.6 (newer) if present; fall back to 3.5 if only it
    # is installed. Never silently skip if at least one A3B is available.
    if [ -f "$A3B_36" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_36" "3.6" 1)
    elif [ -f "$A3B_35" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_35" "3.5" 1)
    fi
else
    # Full mode: every available A3B contributes its 4 cells.
    if [ -f "$A3B_35" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_35" "3.5" 0)
    fi
    if [ -f "$A3B_36" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && CELLS+=("$line")
        done < <(build_cells "$A3B_36" "3.6" 0)
    fi
fi

if [ "${#CELLS[@]}" -eq 0 ]; then
    # Belt-and-suspenders: should never reach here, since the early skip-on-
    # absence at the top exited 0 when both models are missing. If we DO get
    # here, something is wrong with the cell builder; fail loud rather than
    # silently passing.
    echo "agentic-gate: no cells built despite at least one A3B model present" >&2
    echo "  A3B_35=$A3B_35  exists=$([ -f "$A3B_35" ] && echo yes || echo no)" >&2
    echo "  A3B_36=$A3B_36  exists=$([ -f "$A3B_36" ] && echo yes || echo no)" >&2
    exit 2
fi

# ---- Report header ---------------------------------------------------------
{
    echo "# Agentic gate"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $MODE"
    echo "- cells:  ${#CELLS[@]}"
    echo
    echo "Hard-fail predicates:"
    echo "- daemon panic / zero tokens / timeout"
    echo "- tool_call body fails JSON parse"
    echo "- parsed JSON missing required field (name or arguments)"
    echo "- special-token leak inside tool_call body (<|im_start|>, <|endoftext|>)"
    echo "- stacked openers (two consecutive <tool_call>)"
    echo "- soft-warn count > 1 across all cells"
    echo
    echo "Soft-warn signals (printed, included in collective threshold):"
    echo "- <tool_call> not emitted (model answered inline)"
    echo "- <tool_call> appearing inside <think>...</think>"
    echo
} > "$OUT"

# ---- Run cells -------------------------------------------------------------
HARD_FAIL=0
SOFT_WARN_TOTAL=0

# Group cells by model for single-load efficiency.
declare -A MODEL_CELLS
for cell in "${CELLS[@]}"; do
    model="$(echo "$cell" | cut -d'|' -f1)"
    MODEL_CELLS["$model"]+="${cell}"$'\n'
done

# Iterate models; per model, build a single JSONL session.
for model in "${!MODEL_CELLS[@]}"; do
    model_short="$(basename "$model" .mq4)"
    echo "## $model_short" >> "$OUT"
    echo >> "$OUT"
    echo "agentic-gate: model $model_short ($(echo "${MODEL_CELLS[$model]}" | grep -c '|') cells)..."

    # Build JSONL: load + (per cell: generate, optional second-turn generate) + unload
    # Each cell gets a unique id ${prefix}_${cellnum}_t1 (and _t2 for multi-turn).
    # NOTE: never touch $HOME/.hipfire/daemon.pid here. The daemon's lock at
    # daemon.rs:142 always opens that path under $HOME — we override $HOME
    # for the daemon child below so its pid lives at $GATE_HIPFIRE_DIR/.hipfire/
    # daemon.pid instead. Removing the user's pid file would unlink an active
    # user daemon's flock target.
    JSONL_FILE="$(mktemp /tmp/agentic-gate-jsonl.XXXXXX)"

    python3 - "$model" "$JSONL_FILE" <<'PY' >/dev/null
import sys, json, os

model_path = sys.argv[1]
jsonl_path = sys.argv[2]

# 4-second pacing between sends; daemon will buffer.
# load
print_load = json.dumps({"type": "load", "model": model_path,
                         "params": {"max_seq": 4096}})

with open(jsonl_path, "w") as f:
    f.write(print_load + "\n")
PY

    # Read cells back for this model and build the JSONL body in Python (cleanest
    # JSON escaping). Pass cells as repeated args.
    cell_args=()
    while IFS= read -r line; do
        [ -n "$line" ] && cell_args+=("$line")
    done <<< "${MODEL_CELLS[$model]}"

    python3 - "$JSONL_FILE" "$USER_READ" "${cell_args[@]}" <<'PY' >/dev/null
import sys, json

jsonl = sys.argv[1]
user_file = sys.argv[2]
cells = sys.argv[3:]

with open(user_file) as f:
    user_prompt = f.read()

with open(jsonl, "a") as out:
    for idx, cell in enumerate(cells):
        model, sys_path, clamp, multi, label = cell.split("|")
        with open(sys_path) as f:
            system = f.read()
        max_think = 1 if clamp == "1" else 0
        gen_msg = {
            "type": "generate",
            "id": f"c{idx}_t1",
            "prompt": user_prompt,
            "system": system,
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "thinking": False if clamp == "1" else True,
            "max_think_tokens": max_think,
        }
        out.write(json.dumps(gen_msg) + "\n")
        if multi == "1":
            # Synthesize a tool_response then ask the model to continue.
            tool_resp_text = (
                "<tool_response>\n"
                "{\"contents\": \"int main() { return 0; }\"}\n"
                "</tool_response>\n"
                "Now describe what the file does in one sentence."
            )
            out.write(json.dumps({
                "type": "generate", "id": f"c{idx}_t2",
                "prompt": tool_resp_text, "system": "",
                "max_tokens": 128, "temperature": 0.0, "top_p": 1.0,
                "repeat_penalty": 1.0, "thinking": False, "max_think_tokens": 1,
            }) + "\n")
    # unload at end
    out.write(json.dumps({"type": "unload"}) + "\n")
PY

    # Pace JSONL into the daemon. The daemon blocks on stdin readline, so we
    # need a slow producer to allow load + each generate to complete in order.
    # We use named-pipe + background daemon so we can capture the daemon's
    # exact PID and kill ONLY that PID at end-of-cell. Never pkill -f.
    OUTPUT_FILE="$(mktemp /tmp/agentic-gate-out.XXXXXX)"
    STDIN_FIFO="$(mktemp -u /tmp/agentic-gate-fifo.XXXXXX)"
    mkfifo "$STDIN_FIFO"

    # Spawn daemon in background, redirected stdin from FIFO, stdout to file.
    # The daemon's flock at $HOME/.hipfire/daemon.pid (daemon.rs:171) acts
    # as the singleton: if another daemon is already running, this child
    # will exit with "FATAL: hipfire daemon already running" and the
    # detector picks it up. No HOME override here — that bypasses the
    # singleton and risks two 35B daemons on one GPU.
    env HIPFIRE_KV_MODE=asym3 \
        HIPFIRE_GRAPH=1 \
        "$EXE" < "$STDIN_FIFO" > "$OUTPUT_FILE" 2>&1 &
    DAEMON_PID=$!

    # Producer: pace each JSONL line. The daemon reads-and-blocks per
    # newline, so we sleep after each send to let it complete.
    (
        while IFS= read -r line; do
            printf '%s\n' "$line"
            type="$(echo "$line" | python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print(d["type"])' 2>/dev/null)"
            case "$type" in
                load)     sleep 90 ;;
                generate) sleep 60 ;;
                unload)   sleep 2  ;;
            esac
        done < "$JSONL_FILE"
    ) > "$STDIN_FIFO"

    # Producer is done; daemon will see EOF on stdin after unload. Wait
    # briefly for graceful exit, then kill by tracked PID if still alive.
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

    # Parse output, run detectors per cell, append to report.
    python3 - "$OUTPUT_FILE" "$OUT" "${cell_args[@]}" <<'PY'
import sys, json, re, pathlib

out_path = sys.argv[1]
report_path = sys.argv[2]
cells = sys.argv[3:]

raw_text = pathlib.Path(out_path).read_text()
events = []
for line in raw_text.splitlines():
    line = line.strip()
    if not line.startswith("{"): continue
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass

# Group token events by id.
by_id = {}
for ev in events:
    if ev.get("type") == "token":
        by_id.setdefault(ev["id"], []).append(ev["text"])

# Detect daemon panic / error (JSON error events).
panic = any(ev.get("type") == "error" for ev in events)
panic_msg = next((ev["message"] for ev in events if ev.get("type") == "error"), "")

# Detect singleton-collision FATAL. The Rust daemon emits this as plain
# text on stderr (now merged into stdout) when another daemon already
# holds $HOME/.hipfire/daemon.pid (daemon.rs:178-182). It is NOT a JSON
# event, so it slips past the panic check above.
fatal_singleton = "FATAL: hipfire daemon already running" in raw_text

def extract_tool_call_body(text):
    # Use minimal regex; the body extraction is the same as coherence-gate.
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    return m.group(1).strip() if m else None

def detect_stacked_openers(text):
    return bool(re.search(r"<tool_call>\s*<tool_call>", text))

def detect_tool_in_think(text):
    # Tool call appearing inside a <think>...</think> block.
    for m in re.finditer(r"<think>(.*?)</think>", text, re.S):
        if "<tool_call>" in m.group(1):
            return True
    return False

def detect_special_token_leak(body):
    return any(tok in body for tok in ("<|im_start|>", "<|endoftext|>"))

def detect_json(body):
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ("parse_fail", str(e))
    if not isinstance(obj, dict):
        return ("schema_fail", "tool_call body is not a JSON object")
    if "name" not in obj or "arguments" not in obj:
        return ("schema_fail", "missing required field (name or arguments)")
    return ("ok", obj)

results = []
soft_warn_count = 0
hard_fail = False

with open(report_path, "a") as report:
    for idx, cell in enumerate(cells):
        model, sys_path, clamp, multi, label = cell.split("|")
        cid_t1 = f"c{idx}_t1"
        cid_t2 = f"c{idx}_t2"
        toks_t1 = by_id.get(cid_t1, [])
        text_t1 = "".join(toks_t1)
        verdict = "PASS"
        notes = []

        # Hard-fail: singleton collision (a user daemon was already running)
        if fatal_singleton:
            verdict = "HARD_FAIL"
            notes.append("daemon singleton: another hipfire daemon was already running on this host - release it before running the gate")
            hard_fail = True
        # Hard-fail: zero tokens
        if not toks_t1:
            verdict = "HARD_FAIL"
            notes.append("zero tokens emitted")
            hard_fail = True
        else:
            # Hard: stacked openers
            if detect_stacked_openers(text_t1):
                verdict = "HARD_FAIL"
                notes.append("stacked <tool_call> openers")
                hard_fail = True
            body = extract_tool_call_body(text_t1)
            if body is None:
                # Soft: tool_call not emitted
                soft_warn_count += 1
                notes.append("soft: <tool_call> not emitted (inline answer)")
                if verdict == "PASS": verdict = "SOFT_WARN"
            else:
                # Hard: special-token leak
                if detect_special_token_leak(body):
                    verdict = "HARD_FAIL"
                    notes.append("special-token leak inside tool_call body")
                    hard_fail = True
                # Hard: JSON parse / schema
                kind, info = detect_json(body)
                if kind != "ok":
                    verdict = "HARD_FAIL"
                    notes.append(f"json {kind}: {info}")
                    hard_fail = True
                else:
                    notes.append(f"json ok: name={info.get('name')!r}")
            # Soft: tool inside think
            if detect_tool_in_think(text_t1):
                soft_warn_count += 1
                notes.append("soft: <tool_call> inside <think>...</think>")
                if verdict == "PASS": verdict = "SOFT_WARN"

        # Multi-turn t2 evaluation if applicable
        if multi == "1":
            toks_t2 = by_id.get(cid_t2, [])
            text_t2 = "".join(toks_t2)
            if not toks_t2:
                verdict = "HARD_FAIL"
                notes.append("multi-turn t2: zero tokens")
                hard_fail = True
            else:
                # Either coherent text or another tool_call - both ok.
                # Just check for stacked openers and special-token leak in t2.
                if detect_stacked_openers(text_t2):
                    verdict = "HARD_FAIL"
                    notes.append("multi-turn t2: stacked openers")
                    hard_fail = True
                body_t2 = extract_tool_call_body(text_t2)
                if body_t2 is not None and detect_special_token_leak(body_t2):
                    verdict = "HARD_FAIL"
                    notes.append("multi-turn t2: special-token leak")
                    hard_fail = True

        report.write(f"### {label}    {verdict}\n\n")
        report.write(f"- system: `{sys_path}` ({pathlib.Path(sys_path).stat().st_size} bytes)\n")
        report.write(f"- thinking: {'clamped (max_think=1)' if clamp == '1' else 'unclamped'}\n")
        report.write(f"- multi-turn: {'yes' if multi == '1' else 'no'}\n")
        report.write(f"- notes: {'; '.join(notes) if notes else 'clean'}\n\n")
        report.write("```\n" + (text_t1 if text_t1 else "(no tokens)") + "\n```\n\n")
        if multi == "1":
            text_t2 = "".join(by_id.get(f"c{idx}_t2", []))
            report.write("**Turn 2 (after synthesized tool_response):**\n\n")
            report.write("```\n" + (text_t2 if text_t2 else "(no tokens)") + "\n```\n\n")

    # Append per-model summary line via stderr so caller can read it.
    print(json.dumps({"hard_fail": hard_fail, "soft_warn_count": soft_warn_count}), file=sys.stderr)
PY
    res=$?
    if [ "$res" -ne 0 ]; then
        echo "agentic-gate: detector script failed for $model_short" >&2
        HARD_FAIL=1
    fi
    # Read per-model summary written to stderr by the python detector.
    # (We didn't capture stderr above; instead recompute by running a thin checker.)
    # For simplicity, re-run a quick aggregator from the report file.
    rm -f "$OUTPUT_FILE"
done

# ---- Aggregate verdict -----------------------------------------------------
# Count HARD_FAIL and SOFT_WARN headers in the report.
hard_count=$(grep -cE '^### .* HARD_FAIL$' "$OUT" || true)
soft_count=$(grep -cE '^### .* SOFT_WARN$' "$OUT" || true)
total_cells=${#CELLS[@]}
{
    echo "## Summary"
    echo
    echo "- total cells: $total_cells"
    echo "- hard fails: $hard_count"
    echo "- soft warns: $soft_count"
} >> "$OUT"

if [ "$HARD_FAIL" -eq 1 ]; then
    {
        echo
        echo "## Detector failure"
        echo
        echo "One or more model passes had a detector script crash. Exit code 1"
        echo "regardless of report-level cell counts."
    } >> "$OUT"
    echo "agentic-gate: HARD FAIL (detector crashed during a model pass)"
    echo "report: $OUT"
    exit 1
fi
if [ "$hard_count" -gt 0 ]; then
    echo "agentic-gate: HARD FAIL ($hard_count cells)"
    echo "report: $OUT"
    exit 1
fi
# Soft-warn collective threshold: > 1 of N cells = hard fail.
if [ "$soft_count" -gt 1 ]; then
    echo "agentic-gate: HARD FAIL via soft-warn threshold ($soft_count > 1)"
    echo "report: $OUT"
    exit 1
fi
echo "agentic-gate: PASS ($total_cells cells, $soft_count soft warns)"
echo "report: $OUT"
exit 0
