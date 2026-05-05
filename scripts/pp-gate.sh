#!/usr/bin/env bash
# Multi-GPU pipeline-parallel gate.
#
# Validates that pp>1 paths still match pp=1 byte-for-byte under the
# deterministic flag. Skips silently when fewer than 2 HIP devices are
# visible — that's the expected state in CI / single-GPU dev boxes, and
# we don't want pp-gate to block commits that don't touch PP code.
#
# Three barriers (all skipped on <2 GPU):
#
#   1. pp_parity_chatml example — per-token forward_scratch{,_multi}
#      bit-equivalence (50 decode tokens after ChatML prefill, asym3 KV).
#      This is the floor: if forward_scratch_multi diverges from
#      forward_scratch even with HIPFIRE_DETERMINISTIC=1, pp=2 is broken.
#
#   2. daemon pp=1 vs pp=2 byte-equivalence on dense 0.8B mq4 — the
#      end-to-end smoke. Catches regressions in the load/prefill/decode/
#      sample chain that the example doesn't exercise (top_p sampler,
#      repeat penalty, attractor block, ChatML wrap, etc.). Run with
#      HIPFIRE_DETERMINISTIC=1 to opt out of the inherited ksplit
#      atomicAdd reduction non-det.
#
#   3. Refusal sanity — DFlash + pp=2 and CASK + pp=2 must still
#      produce clean error messages at load. These are v1 contract
#      promises (see plan v2 stages 7 + 9).
#
# Exit codes:
#   0  passed, or skipped (<2 GPU)
#   1  hard failure (parity broken, refusal missing, daemon panic)
#   2  build / environment error
#
# Run by .githooks/pre-commit when staged diff touches multi_gpu.rs,
# pp_*, peer_access, or other pipeline-related code.
#
# Manual invocation:
#   ./scripts/pp-gate.sh            # full battery
#   ./scripts/pp-gate.sh --skip-end-to-end   # parity example only

set -u
cd "$(dirname "$0")/.."

SKIP_E2E=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-end-to-end) SKIP_E2E=1 ;;
        -h|--help)
            sed -n '3,33p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# ── ROCm env ────────────────────────────────────────────────────────────
# Auto-source rocm-env.sh so HIP libraries land on the loader path on
# NixOS-style hosts. No-op if already loaded.
if [ -r "./scripts/rocm-env.sh" ]; then
    # shellcheck disable=SC1091
    . ./scripts/rocm-env.sh
fi

# ── GPU count detection ──────────────────────────────────────────────────
# rocm-smi --showid prints multiple lines per GPU; count unique `GPU[N]`
# tags. Fall back to HIP_VISIBLE_DEVICES string parsing for environments
# without rocm-smi (rare).
gpu_count=0
if command -v rocm-smi >/dev/null 2>&1; then
    gpu_count=$(rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l || true)
fi
if [ "$gpu_count" -eq 0 ] && [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    gpu_count=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

if [ "$gpu_count" -lt 2 ]; then
    echo "pp-gate: only $gpu_count GPU(s) visible — skipping"
    exit 0
fi

# Export for the test runs. pp_parity_chatml + daemon pp=2 want both
# devices visible; without this they'd see only $HIP_VISIBLE_DEVICES set
# upstream (or all devices if unset). Pin to 0,1 — the dual-7900 XTX
# layout this gate was sized for. Override via PP_GATE_DEVICES if the
# host has more cards.
export HIP_VISIBLE_DEVICES="${PP_GATE_DEVICES:-0,1}"
export HIPFIRE_DETERMINISTIC=1

EXE="./target/release/examples/daemon"
EXAMPLES_DIR="./target/release/examples"
MODEL="${HIPFIRE_PP_GATE_MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

if [ ! -f "$MODEL" ]; then
    echo "pp-gate: model not found at $MODEL — skipping"
    echo "         set HIPFIRE_PP_GATE_MODEL or install qwen3.5-0.8b.mq4"
    exit 0
fi

# ── Rebuild ──────────────────────────────────────────────────────────────
rebuild=0
if [ ! -x "$EXE" ] || [ ! -x "$EXAMPLES_DIR/pp_parity_chatml" ]; then
    rebuild=1
else
    # Post-0.1.20 modular topology: forward-pass + multi-GPU sources are
    # split across hipfire-runtime (KvCache, Gpus, daemon) and
    # hipfire-arch-qwen35 (qwen35 forward, prefill batch). Watch both.
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs \
               crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/multi_gpu.rs \
               crates/hipfire-runtime/examples/daemon.rs \
               crates/hipfire-runtime/examples/pp_parity_chatml.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then rebuild=1; break; fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "pp-gate: rebuilding..."
    if ! cargo build --release --features deltanet -p hipfire-runtime \
            --example daemon --example pp_parity_chatml >&2; then
        echo "pp-gate: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ─────────────────────────────────────────────────────────────
# Only acquire if no caller has already taken it. Otherwise we'd
# deadlock on the parent's lock — gpu_acquire polls indefinitely and
# doesn't recognize a parent agent's reservation. Detection: lockfile
# present at script start.
if [ -r "$LOCK_SCRIPT" ] && [ ! -f /tmp/hipfire-gpu.lock ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "pp-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

fail=0
say() { printf '\n── %s ──\n' "$1"; }

# ── 1. pp_parity_chatml ─────────────────────────────────────────────────
say "pp_parity_chatml (per-token forward bit-equivalence, 50 decode tokens)"
if "$EXAMPLES_DIR/pp_parity_chatml" "$MODEL" 2>&1 | tee /tmp/pp-gate-parity.log | \
   grep -qE 'ALL [0-9]+ tokens identical'; then
    echo "PASS"
else
    echo "FAIL — see /tmp/pp-gate-parity.log"
    fail=1
fi

if [ "$SKIP_E2E" -eq 1 ]; then
    [ "$fail" -ne 0 ] && exit 1
    echo
    echo "pp-gate: parity-only mode passed."
    exit 0
fi

# ── 2. daemon pp=1 vs pp=2 byte-equivalence ─────────────────────────────
say "daemon pp=1 vs pp=2 byte-identical (greedy, ChatML, HIPFIRE_DETERMINISTIC=1)"
gen_sha () {
    local pp_arg="$1"
    local params='{"max_seq":2048}'
    [ "$pp_arg" = "2" ] && params='{"max_seq":2048,"pp":2}'
    (printf '%s\n' \
        '{"type":"load","model":"'"$MODEL"'","params":'"$params"'}' \
        '{"type":"generate","id":"r1","prompt":"Write a one-sentence greeting.","temperature":0.0,"max_tokens":40}' \
        '{"type":"unload"}'
    ) | "$EXE" 2>/dev/null \
      | grep '"text"' \
      | python3 -c '
import sys, json, hashlib
toks = []
for line in sys.stdin:
    obj = json.loads(line.strip())
    toks.append(obj["text"])
print(hashlib.sha256("".join(toks).encode()).hexdigest()[:16])
'
}
PP1_SHA=$(gen_sha 1)
PP2_SHA=$(gen_sha 2)
echo "pp=1: $PP1_SHA"
echo "pp=2: $PP2_SHA"
if [ -n "$PP1_SHA" ] && [ "$PP1_SHA" = "$PP2_SHA" ]; then
    echo "PASS"
else
    echo "FAIL"
    fail=1
fi

# ── 3. refusal contracts ─────────────────────────────────────────────────
say "refusal: DFlash + pp=2 must error at load"
DFLASH_REFUSAL=$( (printf '%s\n' \
    '{"type":"load","model":"'"$MODEL"'","params":{"max_seq":2048,"pp":2,"draft":"/nonexistent.hfq"}}'
) | "$EXE" 2>/dev/null | grep -c 'DFlash speculative decode requires pp=1' || true)
if [ "$DFLASH_REFUSAL" -ge 1 ]; then echo "PASS"; else echo "FAIL"; fail=1; fi

say "refusal: CASK + pp=2 must error at load"
CASK_REFUSAL=$( (printf '%s\n' \
    '{"type":"load","model":"'"$MODEL"'","params":{"max_seq":2048,"pp":2,"cask_sidecar":"/nonexistent.bin"}}'
) | "$EXE" 2>/dev/null | grep -c 'CASK / TriAttention eviction requires pp=1' || true)
if [ "$CASK_REFUSAL" -ge 1 ]; then echo "PASS"; else echo "FAIL"; fail=1; fi

echo
if [ "$fail" -ne 0 ]; then
    echo "pp-gate: FAIL"
    exit 1
fi
echo "pp-gate: PASS"
exit 0
