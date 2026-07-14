#!/usr/bin/env bash
# Qwen35 DFlash eviction lifecycle regression gate.
#
# Required fixtures (there are deliberately no filename fallbacks):
#   HIPFIRE_EVICTION_TARGET=/path/to/qwen3.5-or-3.6-target.mq4
#   HIPFIRE_EVICTION_DRAFT=/path/to/paired-dflash.hfq
#   HIPFIRE_EVICTION_SIDECAR=/path/to/triattn.sidecar
#
# The baseline and eviction runs use separate daemon processes. Both load plain
# TriAttention (`cask:false`) with DFlash enabled. Request A must generate past
# budget+beta, then an explicit reset must make request B emit exactly the same
# JSONL token frames as clean request B. This checks the request-owned physical
# cursor / compact offset / recurrent target / DFlash mirror without rebuilding
# the model-owned eviction policy and its reusable GPU scratch.
#
# Also run the published-owner regression with the same model pair:
#   HIPFIRE_DFLASH_FREE_TARGET="$HIPFIRE_EVICTION_TARGET" \
#   HIPFIRE_DFLASH_FREE_DRAFT="$HIPFIRE_EVICTION_DRAFT" \
#   nix develop --command cargo test -p hipfire-loader --lib \
#     --locked unload_model_reclaims_published_qwen35_dflash_state -- --ignored
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "test-qwen35-eviction-lifecycle: run from a git worktree" >&2
    exit 2
}
cd "$ROOT"

require_file_env() {
    local name="$1" value="${!1:-}"
    if [ -z "$value" ]; then
        echo "test-qwen35-eviction-lifecycle: required env is unset: $name" >&2
        exit 2
    fi
    if [ ! -f "$value" ]; then
        echo "test-qwen35-eviction-lifecycle: fixture does not exist: $name=$value" >&2
        exit 2
    fi
}

validate_sessions() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import re
import sys

baseline_path, eviction_path, threshold = sys.argv[1:]
threshold = int(threshold)

def parse(path):
    requests = {}
    resets = []
    unloads = []
    failures = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if "panicked" in line.lower() or "fatal" in line.lower():
                failures.append(f"line {line_number}: daemon panic/fatal: {line}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "error":
                failures.append(f"line {line_number}: daemon error: {event.get('message', event)}")
            elif kind == "reset":
                resets.append(event)
            elif kind == "unloaded":
                unloads.append(event)
            elif kind in {"token", "done"}:
                request = requests.setdefault(event.get("id", "<missing-id>"), {"frames": [], "done": []})
                if kind == "token":
                    request["frames"].append(event.get("text", ""))
                else:
                    request["done"].append(event)
    return requests, resets, unloads, failures

def checked(requests, request_id, label):
    item = requests.get(request_id)
    if item is None:
        raise AssertionError(f"{label}: missing JSONL request id {request_id}")
    if len(item["done"]) != 1:
        raise AssertionError(f"{label}: expected one done event, got {len(item['done'])}")
    done = item["done"][0]
    if not isinstance(done.get("tokens"), int) or done["tokens"] <= 0:
        raise AssertionError(f"{label}: zero or invalid done token count: {done.get('tokens')!r}")
    if not item["frames"] or not "".join(item["frames"]).strip():
        raise AssertionError(f"{label}: zero visible token output")
    if done.get("dflash") is not True:
        raise AssertionError(f"{label}: request did not run through DFlash: {done}")
    return item, done

def coherent(frames, label):
    words = re.findall(r"\S+", "".join(frames))
    if len(words) < 3:
        raise AssertionError(f"{label}: too few words for a coherent response")
    unique_ratio = len(set(words)) / len(words)
    max_frequency = max(words.count(word) for word in set(words)) / len(words)
    if unique_ratio < 0.30 or max_frequency > 0.50:
        raise AssertionError(f"{label}: incoherent token output")

baseline, _, baseline_unloads, baseline_failures = parse(baseline_path)
eviction, resets, eviction_unloads, eviction_failures = parse(eviction_path)
if baseline_failures or eviction_failures:
    raise AssertionError("\n".join(baseline_failures + eviction_failures))
if len(baseline_unloads) != 1:
    raise AssertionError(f"baseline session expected exactly one unloaded event, got {len(baseline_unloads)}")
if len(eviction_unloads) != 1:
    raise AssertionError(f"eviction session expected exactly one unloaded event, got {len(eviction_unloads)}")

baseline_b, _ = checked(baseline, "baseline-b", "clean request B")
eviction_a, done_a = checked(eviction, "eviction-a", "eviction request A")
eviction_b, _ = checked(eviction, "eviction-b", "eviction request B")
if done_a["tokens"] <= threshold:
    raise AssertionError(f"request A did not exceed budget+beta={threshold}")
if not any(reset.get("seq_pos") == 0 for reset in resets):
    raise AssertionError("eviction session has no successful explicit reset event")
if baseline_b["frames"] != eviction_b["frames"]:
    raise AssertionError("request B token frames differ after eviction/reset")
coherent(baseline_b["frames"], "clean request B")
coherent(eviction_b["frames"], "eviction request B")
print(f"PASS: plain-TriAttention DFlash request A exceeded budget+beta ({done_a['tokens']} > {threshold}); reset request B matched clean token frames")
PY
}

run_self_check() {
    local directory baseline eviction missing_unload
    directory="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-eviction-self-check.XXXXXX")"
    trap 'rm -rf "$directory"' RETURN
    baseline="$directory/baseline.jsonl"
    eviction="$directory/eviction.jsonl"
    missing_unload="$directory/missing-unload.jsonl"
    python3 - "$baseline" "$eviction" "$missing_unload" <<'PY'
import json
import sys

baseline, eviction, missing_unload = sys.argv[1:]
clean = [
    {"type": "token", "id": "baseline-b", "text": "fresh reset starts now"},
    {"type": "done", "id": "baseline-b", "tokens": 4, "dflash": True},
    {"type": "unloaded"},
]
evicted = [
    {"type": "token", "id": "eviction-a", "text": "1 2 3"},
    {"type": "done", "id": "eviction-a", "tokens": 41, "dflash": True},
    {"type": "reset", "seq_pos": 0},
    {"type": "token", "id": "eviction-b", "text": "fresh reset starts now"},
    {"type": "done", "id": "eviction-b", "tokens": 4, "dflash": True},
    {"type": "unloaded"},
]
for path, events in ((baseline, clean), (eviction, evicted), (missing_unload, clean[:-1])):
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
PY
    validate_sessions "$baseline" "$eviction" 40 >/dev/null
    if validate_sessions "$missing_unload" "$eviction" 40 >/dev/null 2>&1; then
        echo "test-qwen35-eviction-lifecycle: self-check accepted a missing unload event" >&2
        return 1
    fi
    echo "test-qwen35-eviction-lifecycle: self-check PASS" >&2
}

if [ "${1:-}" = "--self-check" ]; then
    run_self_check
    exit 0
fi

require_file_env HIPFIRE_EVICTION_TARGET
require_file_env HIPFIRE_EVICTION_DRAFT
require_file_env HIPFIRE_EVICTION_SIDECAR

TARGET="$HIPFIRE_EVICTION_TARGET"
DRAFT="$HIPFIRE_EVICTION_DRAFT"
SIDECAR="$HIPFIRE_EVICTION_SIDECAR"
BUDGET="${HIPFIRE_EVICTION_BUDGET:-32}"
BETA="${HIPFIRE_EVICTION_BETA:-8}"
MAX_SEQ="${HIPFIRE_EVICTION_MAX_SEQ:-2048}"
TIMEOUT_SECONDS="${HIPFIRE_EVICTION_TIMEOUT_SECONDS:-900}"

for value_name in BUDGET BETA MAX_SEQ TIMEOUT_SECONDS; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]] || [ "$value" -eq 0 ]; then
        echo "test-qwen35-eviction-lifecycle: $value_name must be a positive integer, got $value" >&2
        exit 2
    fi
done

THRESHOLD=$((BUDGET + BETA))
MAX_A=$((THRESHOLD + 64))
EXE="./target/release/examples/daemon"
STALE_SOURCES=(
    crates/hipfire-runtime/examples/daemon.rs
    crates/hipfire-runtime/src/triattn.rs
    crates/hipfire-runtime/src/cask.rs
    crates/hipfire-runtime/src/dflash.rs
    crates/hipfire-loader/src/lib.rs
    crates/hipfire-loader/src/carriers.rs
    crates/hipfire-loader/src/spec_build.rs
    crates/hipfire-arch-qwen35/src/qwen35.rs
    crates/hipfire-arch-qwen35/src/speculative.rs
    crates/hipfire-arch-qwen35/src/dflash_spec.rs
)
stale=0
if [ ! -x "$EXE" ]; then
    stale=1
else
    for source in "${STALE_SOURCES[@]}"; do
        if [ "$source" -nt "$EXE" ]; then
            stale=1
            break
        fi
    done
fi
if [ "$stale" -ne 0 ]; then
    echo "test-qwen35-eviction-lifecycle: building daemon..." >&2
    cargo build --release --features deltanet --example daemon -p hipfire-runtime >&2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen35-eviction-lifecycle.XXXXXX")"
cleanup() {
    rm -f "$WORK_DIR"/*.input
    if declare -F gpu_release >/dev/null; then
        gpu_release 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ -r ./scripts/gpu-lock.sh ]; then
    # shellcheck disable=SC1091
    . ./scripts/gpu-lock.sh
    gpu_acquire "qwen35-eviction-lifecycle" || {
        echo "test-qwen35-eviction-lifecycle: could not acquire GPU lock" >&2
        exit 2
    }
fi

write_session() {
    local mode="$1" destination="$2"
    python3 - "$mode" "$destination" "$TARGET" "$DRAFT" "$SIDECAR" "$BUDGET" "$BETA" "$MAX_SEQ" "$MAX_A" <<'PY'
import json
import sys

mode, destination, target, draft, sidecar, budget, beta, max_seq, max_a = sys.argv[1:]
params = {
    "max_seq": int(max_seq),
    "draft": draft,
    "cask_sidecar": sidecar,
    "cask": False,
    "cask_budget": int(budget),
    "cask_beta": int(beta),
}
request_b = "Return exactly one clear sentence explaining that a reset starts a fresh lifecycle."
events = [{"type": "load", "model": target, "params": params}]
if mode == "baseline":
    events.append({
        "type": "generate", "id": "baseline-b", "prompt": request_b,
        "temperature": 0.0, "max_tokens": 48,
    })
elif mode == "eviction":
    events.extend([
        {
            "type": "generate", "id": "eviction-a",
            "prompt": "Write the integers from 1 through 200 in order, separated by commas. Do not stop early.",
            "temperature": 0.0, "max_tokens": int(max_a),
        },
        {"type": "reset"},
        {
            "type": "generate", "id": "eviction-b", "prompt": request_b,
            "temperature": 0.0, "max_tokens": 48,
        },
    ])
else:
    raise SystemExit(f"unknown session mode: {mode}")
events.append({"type": "unload"})
with open(destination, "w", encoding="utf-8") as handle:
    for event in events:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")
PY
}

run_session() {
    local label="$1" input="$2" output="$3"
    echo "test-qwen35-eviction-lifecycle: running $label" >&2
    # Capturing during baseline B but replaying during eviction B would compare
    # different graph states instead of request lifecycle state.
    local status=0
    HIPFIRE_VERIFY_GRAPH=0 timeout "$TIMEOUT_SECONDS" "$EXE" < "$input" > "$output" 2>&1 || status=$?
    if [ "$status" -eq 0 ]; then
        return 0
    fi
    echo "test-qwen35-eviction-lifecycle: $label daemon exit=$status (log: $output)" >&2
    return "$status"
}

BASELINE_INPUT="$WORK_DIR/baseline.input"
EVICTION_INPUT="$WORK_DIR/eviction.input"
BASELINE_LOG="$WORK_DIR/baseline.jsonl"
EVICTION_LOG="$WORK_DIR/eviction.jsonl"
write_session baseline "$BASELINE_INPUT"
write_session eviction "$EVICTION_INPUT"
run_session baseline "$BASELINE_INPUT" "$BASELINE_LOG" || exit 2
run_session eviction "$EVICTION_INPUT" "$EVICTION_LOG" || exit 2

validate_sessions "$BASELINE_LOG" "$EVICTION_LOG" "$THRESHOLD"

echo "test-qwen35-eviction-lifecycle: PASS (logs retained at $WORK_DIR)" >&2
