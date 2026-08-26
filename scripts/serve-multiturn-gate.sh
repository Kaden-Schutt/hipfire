#!/usr/bin/env bash
# serve-multiturn-gate.sh — multi-request serve-output regression gate.
#
# WHY THIS EXISTS (#462): PR #455's carrier registry moved qwen3.5's DeltaNet/KV
# recurrent state into ModelState::Qwen35(bundle) but did not migrate the
# daemon's reset/checkpoint/abort state machine — so recurrent state bled ACROSS
# requests into a </think> thinking-loop attractor on serve (catastrophic on
# DFlash, mild on AR). It passed unit + coherence gates because those drive the
# daemon SINGLE-request (coherence-gate.sh does load→ONE generate→unload per
# row). Only MULTI-request serve output exposes the bleed.
#
# This gate loads a qwen3.5 (DeltaNet) model ONCE and sends several distinct
# prompts in the SAME session (no unload between). It then runs attractor
# detection on EVERY request — the request-2..N checks are the regression
# guard: if reset no-ops and state bleeds, request 2+ collapses into a
# low-unique-token / high-max-frequency attractor and the gate hard-fails.
#
# Exit codes: 0 = all requests coherent; 1 = a request degraded (bleed/attractor);
# 2 = infra error (build/lock/daemon panic/zero tokens); 3 = SKIPPED (no models).
#
# Covers AR (always, small model) and DFlash (the catastrophic path, when a
# 27B target + draft are present).
set -u

EXE="./target/release/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_SERVE_GATE_OUT:-/tmp/serve-multiturn-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# Distinct prompts — different topics so a bled recurrent state can't be
# mistaken for legitimate continuation. Greedy (temp 0) for determinism.
PROMPTS=(
    "What is 2+2? Answer briefly."
    "Name three primary colors."
    "What is the capital of Japan? One sentence."
    "Write one short sentence about cats."
)
MAX_TOKENS=80

# ── Build daemon if stale ─────────────────────────────────────────────────
if [ ! -x "$EXE" ] || \
   [ crates/hipfire-daemon/src/main.rs -nt "$EXE" ] || \
   [ crates/hipfire-loader/src/lib.rs -nt "$EXE" ]; then
    echo "serve-multiturn-gate: building daemon..." >&2
 cargo build --release -p hipfire-daemon >&2 || { echo "build failed" >&2; exit 2; }
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "serve-multiturn-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

echo "# serve-multiturn gate — $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"
hard_fail=0
ran_any=0

# detect(out_file): per-request attractor detection over a daemon JSONL log.
# Hard-fails (exit 1) if any request's visible text is an attractor:
#   - unique word ratio < 0.30  (structural loop, e.g. "منذ منذ منذ…")
#   - max single-word frequency > 0.50
#   - a </think> with no opening <think> AND heavy repetition (the #462 signature)
# Prints a per-request PASS/FAIL table to the report.
detect() {
    python3 - "$1" "$OUT" <<'PY'
import sys, json, re
out_file, report = sys.argv[1], sys.argv[2]
reqs = {}      # id -> [token text...]
order = []
panic = False
with open(out_file, errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "panicked" in line or "FATAL" in line:
            panic = True
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        t = m.get("type")
        if t == "token":
            rid = m.get("id", "?")
            if rid not in reqs:
                reqs[rid] = []; order.append(rid)
            reqs[rid].append(m.get("text", ""))
        elif t == "error":
            panic = True

rows = []
failed = False
for i, rid in enumerate(order, 1):
    txt = "".join(reqs[rid])
    words = txt.split()
    n = len(words)
    uniq = len(set(words)) / n if n else 0.0
    maxf = (max((words.count(w) for w in set(words)), default=0) / n) if n else 1.0
    think_close = txt.count("</think>")
    think_open = txt.count("<think>")
    runaway_think = think_close > think_open and uniq < 0.4
    bad = (n == 0) or (uniq < 0.30) or (maxf > 0.50) or runaway_think
    if bad:
        failed = True
    rows.append((i, rid, n, uniq, maxf, "FAIL" if bad else "pass", repr(txt[:90])))

with open(report, "a") as r:
    r.write("\n| # | req | toks | uniq | maxfreq | status | sample |\n")
    r.write("|---|-----|------|------|---------|--------|--------|\n")
    for (i, rid, n, uniq, maxf, st, samp) in rows:
        r.write(f"| {i} | {rid} | {n} | {uniq:.2f} | {maxf:.2f} | {st} | {samp} |\n")
    if panic:
        r.write("\n**HARD: daemon panic / error event**\n")

sys.exit(1 if (failed or panic) else 0)
PY
}

run_session() {
    local label="$1"; shift
    local in_file out_file
    in_file="$(mktemp)"; out_file="$(mktemp)"
    printf '%s\n' "$@" > "$in_file"
    echo "== $label =="
    echo -e "\n## $label" >> "$OUT"
    timeout 900 "$EXE" < "$in_file" > "$out_file" 2>&1
    local ec=$?
    if [ "$ec" -ne 0 ] && [ "$ec" -ne 124 ]; then
        echo "**HARD: daemon exit=$ec**" >> "$OUT"; hard_fail=1
    fi
    detect "$out_file" || hard_fail=1
    rm -f "$in_file" "$out_file"
}

# Build a load+N-generate+unload JSONL session for a model (+optional draft).
build_session() {
    local model="$1" draft="${2:-}"
    local params="\"max_seq\":2048"
    [ -n "$draft" ] && params="$params,\"draft\":\"$draft\""
    printf '{"type":"load","model":"%s","params":{%s}}\n' "$model" "$params"
    local i=0
    for p in "${PROMPTS[@]}"; do
        i=$((i+1))
        local pj; pj=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$p")
        printf '{"type":"generate","id":"r%d","prompt":%s,"temperature":0.0,"max_tokens":%d}\n' "$i" "$pj" "$MAX_TOKENS"
    done
    printf '{"type":"unload"}\n'
}

# ── AR multi-request (always; small DeltaNet model) ───────────────────────
AR_MODEL=""
for c in qwen3.5-0.8b.mq4 qwen3.5-4b.mq4 qwen3.5-9b.mq4; do
    [ -f "$MODELS_DIR/$c" ] && { AR_MODEL="$MODELS_DIR/$c"; break; }
done
if [ -n "$AR_MODEL" ]; then
    ran_any=1
    mapfile -t lines < <(build_session "$AR_MODEL")
    run_session "AR multi-request — $(basename "$AR_MODEL")" "${lines[@]}"
fi

# ── DFlash multi-request (the *catastrophic* path) ────────────────────────
# DFlash is the worst case for the #462 bleed. It needs a 27B target + draft,
# so it auto-runs only when both are present under MODELS_DIR (skipped
# otherwise). Set HIPFIRE_SERVE_GATE_DFLASH=0 to force-skip even when present.
DF_TARGET=""; DF_DRAFT="${HIPFIRE_DFLASH_DRAFT:-}"
if [ "${HIPFIRE_SERVE_GATE_DFLASH:-1}" != "0" ]; then
    for c in qwen3.6-27b.mq4 qwen3.5-27b.mq4; do
        [ -f "$MODELS_DIR/$c" ] && { DF_TARGET="$MODELS_DIR/$c"; break; }
    done
    if [ -z "$DF_DRAFT" ]; then
        for c in qwen36-27b-dflash-mq4.hfq qwen35-27b-dflash.mq4; do
            [ -f "$MODELS_DIR/$c" ] && { DF_DRAFT="$MODELS_DIR/$c"; break; }
        done
    fi
fi
if [ -n "$DF_TARGET" ] && [ -n "$DF_DRAFT" ]; then
    ran_any=1
    mapfile -t lines < <(build_session "$DF_TARGET" "$DF_DRAFT")
    run_session "DFlash multi-request — $(basename "$DF_TARGET") + $(basename "$DF_DRAFT")" "${lines[@]}"
fi

echo "" >> "$OUT"
if [ "$ran_any" -eq 0 ]; then
    echo "## SKIPPED — no qwen3.5 model under $MODELS_DIR" >> "$OUT"
    echo "serve-multiturn-gate: SKIPPED (no models). Report: $OUT" >&2
    exit 3
fi
echo "serve-multiturn-gate: report at $OUT" >&2
if [ "$hard_fail" -ne 0 ]; then
    echo "serve-multiturn-gate: FAIL — a request degraded across the session (cross-request state bleed?)" >&2
    exit 1
fi
echo "serve-multiturn-gate: PASS — all requests coherent across the session" >&2
exit 0
