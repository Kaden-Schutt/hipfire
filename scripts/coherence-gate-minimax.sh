#!/usr/bin/env bash
# coherence-gate-minimax.sh — MiniMax-M2 prefill coherence gate via the DAEMON
# chat template (the real serving path), not raw greedy prompts.
#
# Sends a diverse prompt matrix as chat `messages` so the daemon renders the
# MiniMax chat template → the model answers and emits EOS naturally (no greedy
# rambling). Runs the production path with no env levers: short prompts use the
# per-token indexed MoE GEMV, while the long passage (>=256 rows/chunk) drives
# the scatter-grouped MoE prefill (i8 on gfx1151 by arch). Both must stay
# coherent.
#
# Hard-fails (exit 1) if any request's visible chat output is degenerate:
#   - zero tokens, OR max single-word frequency > 0.50 (true attractor), OR
#     unique-word ratio < 0.30 (structural loop), OR a daemon panic/error.
# Chat-templated answers are clean, so these thresholds reflect real failures
# (unlike the raw-prompt example battery, where greedy repetition is expected).
#
# Exit: 0 PASS, 1 degenerate/regression, 2 infra (build/lock/panic), 3 skipped.
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/daemon"
MODEL="${HIPFIRE_MINIMAX_MODEL:-$HOME/.hipfire/models/MiniMax-M2.7.mq2}"
MAX_TOKENS="${MAX_TOKENS:-200}"
OUT="${HIPFIRE_MINIMAX_GATE_OUT:-/tmp/coherence-minimax-$(date +%Y%m%d-%H%M%S).md}"
hard_fail=0

[ -f "$MODEL" ] || { echo "coherence-gate-minimax: model $MODEL not found — SKIP" >&2; exit 3; }

PROMPTS=(
  "What is the capital of France? Answer in one sentence."
  "If a farmer has 17 sheep and all but 9 die, how many are left? Show your reasoning."
  "Write a Python function fib(n) that returns the nth Fibonacci number."
  "List the first ten prime numbers."
  "Explain in a short paragraph how binary search works."
  "Summarize how photosynthesis works in two sentences."
  # Long prompt (>=256 rendered tokens) so the prefill chunk hits the
  # scatter-grouped MoE path (b>=256), not just the indexed path.
  "Read the following passage and then summarize it in one sentence. \
Photosynthesis is the process by which green plants, algae, and some bacteria \
convert light energy into chemical energy stored in glucose. It takes place in \
the chloroplasts, which contain the pigment chlorophyll. The process has two \
main stages. In the light-dependent reactions, energy from sunlight is captured \
and used to split water molecules, producing oxygen as a byproduct and \
generating the energy carriers ATP and NADPH. In the light-independent \
reactions, also called the Calvin cycle, the ATP and NADPH are used to fix \
carbon dioxide from the air into three-carbon sugars, which are then assembled \
into glucose. Photosynthesis is fundamental to life on Earth because it produces \
the oxygen we breathe and forms the base of nearly every food chain. The overall \
balanced equation is six carbon dioxide plus six water, in the presence of light \
energy, yielding one glucose molecule plus six oxygen molecules."
)

if [ ! -x "$EXE" ] || [ crates/hipfire-daemon/src/main.rs -nt "$EXE" ]; then
  echo "coherence-gate-minimax: building daemon..." >&2
 cargo build --release -p hipfire-daemon >&2 || { echo "build failed" >&2; exit 2; }
fi

if [ -f scripts/gpu-lock.sh ]; then
  # shellcheck disable=SC1091
  source scripts/gpu-lock.sh
  gpu_acquire "coherence-gate-minimax" || { echo "could not acquire GPU lock" >&2; exit 2; }
  trap 'gpu_release 2>/dev/null || true' EXIT
fi

echo "# coherence-gate-minimax — $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"
echo "model: $MODEL  max_tokens: $MAX_TOKENS" >> "$OUT"

detect() {
  python3 - "$1" "$OUT" <<'PY'
import sys, json
out_file, report = sys.argv[1], sys.argv[2]
reqs, order, panic = {}, [], False
with open(out_file, errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "panicked" in line or "FATAL" in line or "Memory access fault" in line:
            panic = True; continue
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
rows, failed = [], False
for i, rid in enumerate(order, 1):
    txt = "".join(reqs[rid]); words = txt.split(); n = len(words)
    uniq = len(set(words)) / n if n else 0.0
    maxf = (max(words.count(w) for w in set(words)) / n) if n else 1.0
    bad = (n == 0) or (uniq < 0.30) or (maxf > 0.50)
    failed = failed or bad
    rows.append((i, rid, n, uniq, maxf, "FAIL" if bad else "pass", repr(txt[:100])))
with open(report, "a") as r:
    r.write("\n| # | req | toks | uniq | maxfreq | status | sample |\n|---|---|---|---|---|---|---|\n")
    for (i, rid, n, uniq, maxf, st, samp) in rows:
        r.write(f"| {i} | {rid} | {n} | {uniq:.2f} | {maxf:.2f} | {st} | {samp} |\n")
    if panic:
        r.write("\n**HARD: daemon panic / fault / error event**\n")
sys.exit(1 if (failed or panic) else 0)
PY
}

run_mode() {
  local label="$1"; shift  # remaining args = env assignments
  local in_file out_file; in_file="$(mktemp)"; out_file="$(mktemp)"
  {
    printf '{"type":"load","model":"%s","params":{"max_seq":2048}}\n' "$MODEL"
    local i=0
    for p in "${PROMPTS[@]}"; do
      i=$((i+1))
      local cj; cj=$(python3 -c 'import sys,json; print(json.dumps(sys.argv[1]))' "$p")
      printf '{"type":"generate","id":"r%d","messages":[{"role":"user","content":%s}],"temperature":0.0,"max_tokens":%d}\n' "$i" "$cj" "$MAX_TOKENS"
    done
    printf '{"type":"unload"}\n'
  } > "$in_file"
  echo "== $label ==" ; echo -e "\n## $label" >> "$OUT"
  env "$@" timeout 1200 "$EXE" < "$in_file" > "$out_file" 2>&1
  local ec=$?
  [ "$ec" -ne 0 ] && [ "$ec" -ne 124 ] && { echo "**HARD: daemon exit=$ec**" >> "$OUT"; hard_fail=1; }
  detect "$out_file" || hard_fail=1
  rm -f "$in_file" "$out_file"
}

# Production path (no env levers): short prompts use the indexed MoE GEMV,
# the long prompt (>=256 rows/chunk) exercises the scatter-grouped MoE (i8 on
# gfx1151). Both must stay coherent through the chat template.
run_mode "production (indexed short + grouped-i8 long)"

echo "" >> "$OUT"
if [ "$hard_fail" -ne 0 ]; then
  echo "coherence-gate-minimax: FAIL (see $OUT)" >&2
  echo "**GATE: FAIL**" >> "$OUT"
  exit 1
fi
echo "coherence-gate-minimax: PASS (see $OUT)" >&2
echo "**GATE: PASS**" >> "$OUT"
exit 0
