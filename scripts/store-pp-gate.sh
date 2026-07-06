#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Device-mesh store→PP bit-exact gate.
#
# Guards the manifest-driven pipeline-parallel LOAD + EXECUTE path against
# the bespoke single-GPU llama loader. The `llama_store_pp` example:
#
#   1. fulfills the WHOLE llama `weight_manifest` across a 2-stage pipeline
#      mesh (`HIPFIRE_EMULATE_GPUS=2`) via the generic `fulfill_manifest`,
#      asserting every tensor lands on its mesh-correct stage;
#   2. assembles `LlamaWeights` from the store and runs a REAL banded PP
#      forward — stage 0 embed + band(0..k) on dev0 → `boundary_copy` →
#      stage 1 band(k..n) + head on dev1;
#   3. asserts the PP logits are byte-for-byte identical to the bespoke
#      `Llama::load_weights` + `forward_scratch_compute` reference
#      (`max |Δ| == 0`, same argmax).
#
# The example self-asserts (3) and panics on any divergence, so this gate is
# a thin build+run+exit-code wrapper. It exists so the drop-in-loader claim
# ("the generic manifest path is bit-exact vs the bespoke llama loader") is
# a committed, re-runnable check — not a one-off manual run.
#
# Emulation note: this needs only ONE physical GPU. `HIPFIRE_EMULATE_GPUS=2`
# splits it into two logical ranks aliased onto device 0 (peer copies become
# same-device d2d). Real distinct-device PP is validated separately on
# hiptrx/hipx; that is NOT what this gate covers.
#
# Environment knobs:
#   HIPFIRE_STORE_PP_GATE_MODEL=<path>   # override the default model
#
# Default model: $HOME/.hipfire/models/qwen3-0.6b-llama.mq4 (a small dense
# GQA llama-family model with per-head q/k norm — exercises the full manifest).
#
# Exit codes:
#   0  passed, or skipped (no GPU / model absent / Windows host)
#   1  hard failure (PP logits diverged from bespoke, or non-finite / panic)
#   2  build / environment error
#
# Must run inside the dev shell (`nix develop`) so the linker + HIP libs are
# on PATH — same assumption as pp-gate.sh / coherence-gate.sh.
#
# Manual invocation:
#   ./scripts/store-pp-gate.sh
#   HIPFIRE_STORE_PP_GATE_MODEL=/path/to/model.mq4 ./scripts/store-pp-gate.sh

set -u
cd "$(dirname "$0")/.." || { echo "store-pp-gate: failed to cd to repo root" >&2; exit 2; }

# ── Platform gate ───────────────────────────────────────────────────────
# Emulated PP dispatch is Linux-ROCm-only (needs /dev/kfd).
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "store-pp-gate: Windows host ($(uname -s)) — skipping (Linux ROCm only)"
        exit 0
        ;;
esac

# ── ROCm env ────────────────────────────────────────────────────────────
# Put HIP libraries on the loader path on NixOS-style hosts. No-op if loaded.
if [ -r "./scripts/rocm-env.sh" ]; then
    # shellcheck disable=SC1091
    . ./scripts/rocm-env.sh
fi

# ── GPU presence (need ≥1; the example emulates the 2nd rank) ─────────────
# /dev/kfd is the ROCm compute node; its absence means no usable GPU here
# (CI container, non-ROCm host) → skip rather than fail, matching pp-gate.
if [ ! -e /dev/kfd ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    echo "store-pp-gate: no /dev/kfd and no HIP_VISIBLE_DEVICES — no GPU, skipping"
    exit 0
fi

# ── Model ───────────────────────────────────────────────────────────────
MODEL="${HIPFIRE_STORE_PP_GATE_MODEL:-$HOME/.hipfire/models/qwen3-0.6b-llama.mq4}"
if [ ! -f "$MODEL" ]; then
    echo "store-pp-gate: model not found at $MODEL — skipping"
    echo "              set HIPFIRE_STORE_PP_GATE_MODEL or install qwen3-0.6b-llama.mq4"
    exit 0
fi

EXE="./target/release/examples/llama_store_pp"

# ── Rebuild if the example or any source it exercises is newer ───────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-runtime/examples/llama_store_pp.rs \
               crates/hipfire-runtime/src/weight_store.rs \
               crates/hipfire-runtime/src/weight_manifest.rs \
               crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/multi_gpu.rs \
               crates/hipfire-hardware/src/mesh.rs \
               crates/hipfire-arch-llama/src/arch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then rebuild=1; break; fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "store-pp-gate: building llama_store_pp..."
    if ! cargo build --release -p hipfire-runtime --example llama_store_pp >&2; then
        echo "store-pp-gate: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock (only if no caller already holds it — else we'd deadlock) ───
LOCK_SCRIPT="./scripts/gpu-lock.sh"
if [ -r "$LOCK_SCRIPT" ] && [ ! -f /tmp/hipfire-gpu.lock ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "store-pp-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Run: the example self-asserts max|Δ|==0 and panics on divergence ─────
echo "── store→PP bit-exact (manifest-driven load + banded PP == bespoke) ──"
out=$("$EXE" "$MODEL" 2>&1)
status=$?
echo "$out"

if [ "$status" -ne 0 ]; then
    echo "store-pp-gate: FAIL — example exited $status (PP diverged from bespoke or panicked)" >&2
    exit 1
fi
if ! printf '%s\n' "$out" | grep -q "REAL banded PP forward OK"; then
    echo "store-pp-gate: FAIL — success sentinel missing (banded PP did not complete)" >&2
    exit 1
fi

echo "store-pp-gate: PASS — manifest-driven store→banded-PP forward is bit-exact vs bespoke"
exit 0
