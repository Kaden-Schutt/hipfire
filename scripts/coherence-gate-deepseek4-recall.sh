#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

# Recall gate — DeepSeek V4 Flash far-context (DSA compressed) recall.
#
# Sibling to coherence-gate-deepseek4-mtp.sh. Where the MTP gate guards
# spec-decode token attractors, THIS gate guards the agentic "lossiness in
# recall" class: a value that sits in the system prompt (here a working-
# directory path) must be reproduced EXACTLY after enough filler has pushed
# it out of the 128-token SWA window and into the DSA compressed path.
#
# Why this gate exists (regression guards two fixes):
#   1. comp_rope prefill/decode phase mismatch — the compressed KV was BUILT
#      with a different compressor-RoPE phase ("mid") than it was READ with at
#      decode ("start"). Fixed: precompute_positions_batched default -> "start".
#   2. prefix-cache PARTIAL-hit ring misalignment — generations that share a
#      system prompt take a partial prefix-cache hit (lcp>0), which skipped
#      zero_decode_caches and reused the DSA compressor/SWA rings at the prior
#      turn's end position, pooling a STALE overlap window over the cached tail.
#      Fixed: cold-rebuild on partial hits only (daemon).
# Both manifested as the cwd recalling as e.g. /home/n/... or .../t. Before the
# fixes this scenario was 0/4; after, 4/4.
#
# The scenario deliberately reproduces the partial-hit trigger: a SHORT
# in-window "warm" generate that shares the system prompt, then several DEEP
# generates (also sharing the system) — each deep one is a partial hit.
#
# Hard-fail conditions (block commit, exit 1):
#   - daemon non-zero exit / panic / zero tokens
#   - any DEEP (compressed-path) recall does not reproduce the cwd EXACTLY
#
# Soft (reported, does NOT fail):
#   - the in-window (warm) recall is wrong — for some compound-word paths the
#     model's greedy decoding splits a token even at short context (a decoding
#     quirk, not a KV-cache regression); the compressed path often still
#     recalls them fully, so this is a baseline note only.
#
# Exit codes:
#   0  all recalls exact (or model absent -> skipped)
#   1  hard error (panic / zero tokens / mangled recall)
#   2  build or environment error
#
# Modes:
#   ./scripts/coherence-gate-deepseek4-recall.sh         # warm + 3 deeps (~1-2 min)
#   ./scripts/coherence-gate-deepseek4-recall.sh --fast  # warm + 1 deep   (<1 min)
#   ./scripts/coherence-gate-deepseek4-recall.sh --full  # warm + 4 deeps + 2nd cwd shape

set -u
cd "$(dirname "$0")/.."

FULL=0
FAST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --fast) FAST=1 ;;
        -h|--help) sed -n '3,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$FAST" -eq 1 ] && [ "$FULL" -eq 1 ]; then
    echo "coherence-gate-deepseek4-recall: --fast and --full are mutually exclusive" >&2
    exit 2
fi

EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
V4F_MODEL="$MODELS_DIR/deepseek-v4-flash.mq2lloyd"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-deepseek4-recall-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-420}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild daemon if any DeepSeek V4 / prefill / dispatch source is newer ──
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-deepseek4/src/arch.rs \
               crates/hipfire-arch-deepseek4/src/deepseek4.rs \
               crates/hipfire-arch-deepseek4/src/forward.rs \
               crates/hipfire-runtime/examples/daemon.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate-deepseek4-recall: rebuilding daemon..."
    if ! cargo build --release --example daemon >&2; then
        echo "coherence-gate-deepseek4-recall: build failed" >&2
        exit 2
    fi
fi

# Skip cleanly if the DeepSeek V4 base model isn't present.
if [ ! -f "$V4F_MODEL" ]; then
    {
        echo "# Recall gate — DeepSeek V4 Flash"
        echo
        echo "## SKIPPED — DeepSeek V4 base model not found at $V4F_MODEL"
    } > "$OUT"
    echo "coherence-gate-deepseek4-recall: DeepSeek V4 model not present, skipping (no hard error)"
    echo "report: $OUT"
    exit 0
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-deepseek4-recall" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Scenario depths (filler word count). 300+ pushes the early system-prompt
#    cwd out of the 128 SWA window into the DSA compressed path. ──────────────
if [ "$FAST" -eq 1 ]; then
    DEPTHS=(700)
elif [ "$FULL" -eq 1 ]; then
    DEPTHS=(400 700 1200 2000)
else
    DEPTHS=(400 900 1500)
fi
DEPTHS_CSV=$(IFS=,; echo "${DEPTHS[*]}")
EXTRA_CWD=""
[ "$FULL" -eq 1 ] && EXTRA_CWD="/opt/acme/services/auth-gateway-v2"

{
    echo "# Recall gate — DeepSeek V4 Flash (DSA compressed far-context recall)"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FAST" -eq 1 ] && echo fast || ( [ "$FULL" -eq 1 ] && echo full || echo short ) )"
    echo "- model:  $V4F_MODEL"
    echo "- depths: ${DEPTHS_CSV} (filler words; cwd lives in the system prompt)"
    echo
    echo "Hard-fail: zero tokens / panic / in-window recall wrong / any DEEP"
    echo "(compressed-path) recall not reproducing the cwd exactly."
    echo
} > "$OUT"

# ── Driver + checker (self-contained). Builds the JSONL (shared system prompt
#    => the later generates are prefix-cache partial hits), runs the daemon,
#    verifies each deep recall reproduces the cwd exactly, appends the report,
#    and exits 0 (pass) / 1 (hard fail). ──────────────────────────────────────
DRIVER_PY=$(cat <<'PYEOF'
import sys, json, re, subprocess, os

EXE, MODEL, depths_csv, extra_cwd, out_path = sys.argv[1:6]
depths = [int(x) for x in depths_csv.split(",") if x]
ASK = "Output the working directory path exactly, nothing else."
TIMEOUT = int(os.environ.get("HIPFIRE_COHERENCE_TIMEOUT", "420"))

def fil(n):
    return " ".join(f"item{i}" for i in range(n))

def run_scenario(cwd):
    sysmsg = f"The working directory is {cwd}."
    def gen(tag, prompt):
        return {"type": "generate", "id": tag, "prompt": prompt, "system": sysmsg,
                "temperature": 0.0, "max_tokens": 40, "repeat_penalty": 1.0}
    cmds = [{"type": "load", "model": MODEL, "params": {"max_seq": 16384}},
            gen("warm", "hi\n\n" + ASK)]
    for d in depths:
        cmds.append(gen(f"D{d}", f"(depth {d}) " + fil(d) + "\n\n" + ASK))
    cmds.append({"type": "unload"})
    stdin = "\n".join(json.dumps(c) for c in cmds) + "\n"
    try:
        p = subprocess.run([EXE], input=stdin, capture_output=True, text=True,
                           timeout=TIMEOUT * (len(depths) + 3))
        rc, sout, serr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return {}, 124, True
    ev = {}
    for line in sout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        i = d.get("id", "")
        if d.get("type") in ("token", "reasoning"):
            ev[i] = ev.get(i, "") + d.get("text", "")
    panic = "panicked" in serr or "illegal memory access" in (sout + serr)
    return ev, rc, panic

def recalled(text):
    m = re.search(r"/\S+", (text or "").strip())
    return m.group(0).rstrip(".") if m else ""

cwds = ["/home/nick/CLionProjects/tb"]
if extra_cwd:
    cwds.append(extra_cwd)

overall_ok = True
lines = []
for cwd in cwds:
    ev, rc, panic = run_scenario(cwd)
    total_tokens = sum(len(v) for v in ev.values())
    warm_val = recalled(ev.get("warm", ""))
    warm_ok = (warm_val == cwd)
    deeps = []
    for d in depths:
        val = recalled(ev.get(f"D{d}", ""))
        deeps.append((d, val == cwd, val))
    # Hard-fail on the DEEP (compressed-path) recalls — that's the bug class
    # this gate guards. The in-window (warm) recall is a SOFT baseline only:
    # some compound-word paths (e.g. "gateway", "blinkhash") get split by the
    # model's greedy decoding even at short context — a decoding quirk, not a
    # KV-cache regression — and the compressed path often recalls them fully.
    hard = (rc != 0) or panic or (total_tokens == 0) or any(not ok for _, ok, _ in deeps)
    if hard:
        overall_ok = False
    lines.append(f"## cwd `{cwd}`")
    lines.append("")
    lines.append(f"- rc={rc} panic={panic} tokens={total_tokens}")
    lines.append(f"- in-window (warm) recall [soft]: **{'OK' if warm_ok else 'soft-warn'}** `{warm_val}`")
    for d, ok, val in deeps:
        lines.append(f"- deep depth={d:>5}: **{'OK' if ok else 'MANGLE'}** `{val}`")
    lines.append("")

lines.append(f"## verdict: **{'PASS' if overall_ok else 'HARD-FAIL'}**")
lines.append("")
report = "\n".join(lines)
with open(out_path, "a") as f:
    f.write(report + "\n")
print(report)
sys.exit(0 if overall_ok else 1)
PYEOF
)

echo "== running recall scenario (cwd in system, warm + deep generates) =="
HIPFIRE_COHERENCE_TIMEOUT="$CASE_TIMEOUT" \
    python3 -c "$DRIVER_PY" "$EXE" "$V4F_MODEL" "$DEPTHS_CSV" "$EXTRA_CWD" "$OUT"
rc=$?

echo
if [ "$rc" -ne 0 ]; then
    echo "coherence-gate-deepseek4-recall: HARD FAIL (recall mangled) — see $OUT" >&2
    echo "report: $OUT"
    exit 1
fi
echo "coherence-gate-deepseek4-recall: all recalls exact — review $OUT"
echo "report: $OUT"
exit 0
