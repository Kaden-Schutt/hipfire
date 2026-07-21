#!/usr/bin/env bash
# serve-loop-gate.sh — MULTI-REQUEST serve OUTPUT gate.
#
# Why this exists (issue #462): the coherence/dflash gates drive the daemon
# SINGLE-request (one prompt, fresh state), which is clean even when recurrent
# state leaks ACROSS requests. PR #455 moved qwen3.5's DeltaNet/KV state into a
# bundle and left the daemon's reset/checkpoint sites reading the now-dead
# direct fields → recurrent state bled across requests into a `</think>`
# thinking-loop attractor that ONLY a real, multi-request serve exposes. Every
# single-request gate passed while serve was broken.
#
# This gate starts a REAL native `hipfire serve` (Rust → daemon → HTTP), fires a batch
# of repeated + distinct greedy (temperature=0) requests at one warm daemon, and
# HARD-FAILS if any response is the thinking-loop attractor or a degenerate
# token loop. It exercises the DFlash path by default (the catastrophic case);
# pass --ar to also run with dflash_mode=off.
#
# Exit: 0 = clean, 1 = attractor/garbage detected (gate fails), 2 = setup error.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "serve-loop-gate: not a git repo" >&2; exit 2; }
cd "$ROOT"
PORT="${HIPFIRE_SERVE_GATE_PORT:-11455}"
MODELS_DIR="${HIPFIRE_DIR:-$HOME/.hipfire}/models"
RUN_AR=0
[ "${1:-}" = "--ar" ] && RUN_AR=1

# Pick the smallest available DeltaNet (qwen3.5/3.6, arch 5/6) model — those are
# the recurrent-state arch this gate guards. Prefer a small SKU for speed.
pick_model() {
  local tag f
  for tag in qwen3.5:0.8b qwen3.5:4b qwen3.5:9b qwen3.6:27b; do
    f=$(jq -r --arg t "$tag" '.models[$t].file // empty' "$ROOT/registry/v1.json" 2>/dev/null)
    [ -n "$f" ] && [ -e "$MODELS_DIR/$f" ] && { echo "$tag"; return 0; }
  done
  # fallback: any qwen3.5-*.mq4 file on disk
  ls "$MODELS_DIR"/qwen3.5-*.mq4 2>/dev/null | head -1 | xargs -r basename
}
MODEL="$(pick_model)"
[ -z "$MODEL" ] && { echo "serve-loop-gate: SKIP — no qwen3.5 DeltaNet model in $MODELS_DIR" >&2; exit 0; }

# GPU coordination (reuse the repo's flock-based mutex).
# shellcheck disable=SC1091
if [ -f "$ROOT/scripts/gpu-lock.sh" ]; then source "$ROOT/scripts/gpu-lock.sh" && gpu_acquire "serve-loop-gate" || true; fi
cleanup() {
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  pkill -f "examples/daemon" >/dev/null 2>&1 || true
  command -v gpu_release >/dev/null 2>&1 && gpu_release || true
}
trap cleanup EXIT

HIPFIRE="${HIPFIRE_CLI_BIN:-$ROOT/target/release/hipfire}"
[ -x "$HIPFIRE" ] || HIPFIRE="$(command -v hipfire || true)"
[ -x "$HIPFIRE" ] || { echo "serve-loop-gate: native hipfire CLI not found; build cargo build --release -p hipfire-cli" >&2; exit 2; }

SERVELOG="$(mktemp)"
start_serve() { # $1 = extra env (e.g. "HIPFIRE_DFLASH_OFF")
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true; pkill -f "examples/daemon" >/dev/null 2>&1 || true; sleep 1
  : > "$SERVELOG"
  local cfg=""; [ "${1:-}" = "ar" ] && cfg="HIPFIRE_DFLASH_DRAFT="  # disable draft auto-detect → AR path
  env $cfg HIPFIRE_MODEL="$MODEL" setsid "$HIPFIRE" serve 0.0.0.0 "$PORT" >"$SERVELOG" 2>&1 & disown
  local warm=0
  for _ in $(seq 1 150); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | jq -e '.model != null and .loading_model == null' >/dev/null 2>&1; then warm=1; break; fi
    grep -qiE 'panicked|FATAL|port in use|pre-warm failed' "$SERVELOG" && break
    sleep 2
  done
  if grep -qiE 'panicked|FATAL' "$SERVELOG"; then echo "serve-loop-gate: FAIL — daemon panicked at startup" >&2; tail -5 "$SERVELOG" >&2; return 1; fi
  [ "$warm" = 1 ] || { echo "serve-loop-gate: FAIL — serve never warmed up" >&2; tail -5 "$SERVELOG" >&2; return 1; }
  return 0
}

# 10 requests: repeated + distinct, all factual greedy. Contamination accumulates
# on the 2nd+ request, so the repeats are the trap.
PROMPTS=(
  "What is the capital of France?"
  "What is the capital of France?"
  "Name a primary color."
  "What is 7 times 8?"
  "What is the capital of France?"
  "What is the capital of Japan?"
  "What is the capital of France?"
  "Spell the word cat backwards."
  "What is the capital of France?"
  "What is 2 plus 2?"
)

run_batch() { # $1 = label
  local label="$1" fails=0 i=1
  for p in "${PROMPTS[@]}"; do
    local body resp; body=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"temperature":0,"max_tokens":400,"stream":false}' "$MODEL" "$p")
    resp=$(curl -s --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" 2>/dev/null)
    # Combine content + reasoning so an attractor in either is caught, then
    # detect the degenerate signatures (shell-native: jq + awk + grep -P).
    local verdict txt think maxrep loop
    if ! printf '%s' "$resp" | jq -e '.choices[0].message' >/dev/null 2>&1; then
      verdict="ERR: $(printf '%s' "$resp" | jq -r '.error.message // .message // "no choices"' 2>/dev/null | head -c 60)"
    else
      txt=$(printf '%s' "$resp" | jq -r '.choices[0].message | ((.content // "") + "\n" + (.reasoning_content // ""))' 2>/dev/null)
      think=$(printf '%s' "$txt" | grep -oF '</think>' | wc -l | tr -d ' ')
      # longest run of identical consecutive non-empty lines (e.g. </think> loop)
      maxrep=$(printf '%s\n' "$txt" | awk 'NF{if($0==prev)r++;else r=1; if(r>m)m=r; prev=$0} END{print m+0}')
      # any <=12-char chunk repeated >=4x back-to-back (token/substring loop)
      loop=$(printf '%s' "$txt" | tr -d '\n' | grep -Pc '(.{1,12})\1{3,}' 2>/dev/null); loop=${loop:-0}
      if [ "${think:-0}" -ge 3 ] || [ "${maxrep:-0}" -ge 3 ] || [ "${loop:-0}" -ge 1 ]; then
        verdict="ATTRACTOR think=$think maxrep=$maxrep loop=$loop :: $(printf '%s' "$txt" | tr '\n' ' ' | head -c 80)"
      else
        verdict="ok"
      fi
    fi
    if [[ "$verdict" != ok* ]]; then
      printf "  [%s #%d] FAIL %-28s -> %s\n" "$label" "$i" "${p:0:28}" "$verdict" >&2
      fails=$((fails+1))
    else
      printf "  [%s #%d] ok   %-28s\n" "$label" "$i" "${p:0:28}"
    fi
    i=$((i+1))
  done
  return $fails
}

echo "=== serve-loop-gate: model=$MODEL port=$PORT ==="
RC=0
echo "── DFlash path (default) ──"
if start_serve; then run_batch "dflash" || RC=1; else RC=1; fi
if [ "$RUN_AR" = 1 ]; then
  echo "── AR path (--ar) ──"
  if start_serve ar; then run_batch "ar" || RC=1; else RC=1; fi
fi

if [ "$RC" = 0 ]; then echo "serve-loop-gate: PASS — no cross-request attractor"; else echo "serve-loop-gate: FAIL — cross-request recurrent-state contamination (see #462)" >&2; fi
exit $RC
