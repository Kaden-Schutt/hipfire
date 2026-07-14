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
# Exit codes: 0 = all requests coherent; 1 = quality failure (degraded,
# panic/error event, zero output, missing terminal frame, or daemon failure);
# 2 = setup error (invalid config, build, or lock); 3 = SKIPPED (no models).
#
# Covers AR (always, small model) and DFlash (the catastrophic path, when a
# 27B target + draft are present).
set -u

EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_SERVE_GATE_OUT:-/tmp/serve-multiturn-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
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

validate_cask_budget() {
    local value="$1"
    if [ -n "$value" ] && [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "serve-multiturn-gate: HIPFIRE_SERVE_GATE_CASK_BUDGET must be an integer, got $value" >&2
        return 2
    fi
    # Decimal-only input has already been checked above. Match all zero digits
    # instead of shell arithmetic so leading zeros cannot imply octal or overflow.
    if [[ "$value" =~ ^0+$ ]]; then
        echo "serve-multiturn-gate: HIPFIRE_SERVE_GATE_CASK_BUDGET must be greater than zero" >&2
        return 2
    fi
}

run_self_check() {
    local zero
    if [ "${HIPFIRE_SERVE_GATE_SELF_CHECK_FORCE_FAIL:-}" = 1 ]; then
        echo "serve-multiturn-gate: self-check forced failure" >&2
        return 1
    fi
    for zero in 0 00 000; do
        if validate_cask_budget "$zero" >/dev/null 2>&1; then
            echo "serve-multiturn-gate: self-check accepted zero budget $zero" >&2
            return 1
        fi
    done
    validate_cask_budget 32
    echo "serve-multiturn-gate: self-check PASS" >&2
}

if [ "${1:-}" = "--self-check" ]; then
    run_self_check
    exit $?
fi

# Opt-in only: without a sidecar, `build_session` emits the exact historical
# load JSONL. A DFlash draft always requests plain TriAttention (`cask:false`)
# because CASK m-folding plus DFlash is intentionally unsupported.
if [ -n "${HIPFIRE_SERVE_GATE_CASK_SIDECAR:-}" ]; then
    validate_cask_budget "${HIPFIRE_SERVE_GATE_CASK_BUDGET:-}" || exit $?
    value="${HIPFIRE_SERVE_GATE_CASK_BETA:-}"
    if [ -n "$value" ] && [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "serve-multiturn-gate: HIPFIRE_SERVE_GATE_CASK_BETA must be an integer, got $value" >&2
        exit 2
    fi
fi

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
    echo "serve-multiturn-gate: building daemon..." >&2
    cargo build --release --example daemon --features deltanet >&2 || { echo "build failed" >&2; exit 2; }
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

# detect(out_file, expected_request_ids...): per-request attractor detection
# over a daemon JSONL log. A session is valid only after every submitted request
# terminates with `done` and the daemon acknowledges `unloaded`.
# Hard-fails (exit 1) if any request's visible text is an attractor:
#   - unique word ratio < 0.30  (structural loop, e.g. "منذ منذ منذ…")
#   - max single-word frequency > 0.50
#   - a </think> with no opening <think> AND heavy repetition (the #462 signature)
# Prints a per-request PASS/FAIL table to the report.
detect() {
    python3 - "$1" "$OUT" "${@:2}" <<'PY'
import sys, json, re
out_file, report, *expected = sys.argv[1:]
reqs = {rid: [] for rid in expected}
dones = {rid: [] for rid in expected}
panic = False
unloaded = False
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
            if rid in reqs:
                reqs[rid].append(m.get("text", ""))
        elif t == "done":
            rid = m.get("id", "?")
            if rid in dones:
                dones[rid].append(m)
        elif t == "error":
            panic = True
        elif t == "unloaded":
            unloaded = True

rows = []
failed = False
for i, rid in enumerate(expected, 1):
    txt = "".join(reqs[rid])
    words = txt.split()
    n = len(words)
    uniq = len(set(words)) / n if n else 0.0
    maxf = (max((words.count(w) for w in set(words)), default=0) / n) if n else 1.0
    think_close = txt.count("</think>")
    think_open = txt.count("<think>")
    runaway_think = think_close > think_open and uniq < 0.4
    terminal = len(dones[rid]) == 1
    bad = (not terminal) or (n == 0) or (uniq < 0.30) or (maxf > 0.50) or runaway_think
    if bad:
        failed = True
    rows.append((i, rid, n, len(dones[rid]), uniq, maxf, "FAIL" if bad else "pass", repr(txt[:90])))
if not unloaded:
    failed = True

with open(report, "a") as r:
    r.write("\n| # | req | words | done | uniq | maxfreq | status | sample |\n")
    r.write("|---|-----|-------|------|------|---------|--------|--------|\n")
    for (i, rid, n, done, uniq, maxf, st, samp) in rows:
        r.write(f"| {i} | {rid} | {n} | {done} | {uniq:.2f} | {maxf:.2f} | {st} | {samp} |\n")
    if panic:
        r.write("\n**HARD: daemon panic / error event**\n")
    if not unloaded:
        r.write("\n**HARD: daemon did not acknowledge unload**\n")

sys.exit(1 if (failed or panic) else 0)
PY
}

run_session() {
    local label="$1"; shift
    local in_file out_file ec i
    local expected_ids=()
    in_file="$(mktemp)"; out_file="$(mktemp)"
    printf '%s\n' "$@" > "$in_file"
    echo "== $label =="
    echo -e "\n## $label" >> "$OUT"
    timeout 900 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    if [ "$ec" -ne 0 ]; then
        echo "**HARD: daemon exit=$ec**" >> "$OUT"; hard_fail=1
    fi
    for ((i = 1; i <= ${#PROMPTS[@]}; i++)); do
        expected_ids+=("r$i")
    done
    detect "$out_file" "${expected_ids[@]}" || hard_fail=1
    rm -f "$in_file" "$out_file"
}

# Build a load+N-generate+unload JSONL session for a model (+optional draft).
build_session() {
    local model="$1" draft="${2:-}"
    local params="\"max_seq\":2048"
    [ -n "$draft" ] && params="$params,\"draft\":\"$draft\""
    if [ -n "${HIPFIRE_SERVE_GATE_CASK_SIDECAR:-}" ]; then
        local sidecar_json budget beta
        sidecar_json="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$HIPFIRE_SERVE_GATE_CASK_SIDECAR")"
        budget="${HIPFIRE_SERVE_GATE_CASK_BUDGET:-32}"
        beta="${HIPFIRE_SERVE_GATE_CASK_BETA:-8}"
        params="$params,\"cask_sidecar\":$sidecar_json,\"cask\":false,\"cask_budget\":$budget,\"cask_beta\":$beta"
    fi
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
