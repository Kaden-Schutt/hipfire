#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.
#
# Kernel resource-usage gate for the SP1 batched-attention kernels.
#
# WHY THIS EXISTS
# ---------------
# `scripts/attn_legacy_baseline.sh` compares NUMERIC output, so it is
# structurally blind to a performance regression. During SP1 the flash-prefill
# kernel silently lost 25% of its occupancy ON THE NULL-DESCRIPTOR LEGACY PATH
# — 92 VGPR / 0 scratch / 16 waves became 115 / 64 / 12 — and sailed through all
# nine per-task numeric checks, on both gfx1151 and gfx1201. Two causes: a
# device `assert` that shipped in release (compiler.rs never passes -DNDEBUG),
# and two runtime-unknown 64-bit descriptor bases living across a staging loop.
#
# A numeric gate can never catch that. This one compiles each kernel exactly as
# the runtime does and fingerprints register pressure, scratch and occupancy.
#
# It is COMPILE-ONLY: no GPU, no model, negligible memory. Safe to run any time,
# including while something else holds the GPU.
#
# USAGE
#   scripts/kernel_resource_gate.sh                      # print current
#   scripts/kernel_resource_gate.sh > baseline.txt       # capture
#   diff scripts/kernel_resource_gate.beta.txt <(scripts/kernel_resource_gate.sh)
#
# Any diff is a signal, not necessarily a defect: a deliberate change that costs
# registers should update the baseline in the same commit, with the reason in
# the message. Silent drift is what this prevents.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

HIPCC="${HIPCC:-hipcc}"
command -v "$HIPCC" >/dev/null 2>&1 || { echo "kernel_resource_gate: hipcc not found" >&2; exit 2; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

K=kernels/src

# Assemble a kernel exactly as the runtime does: strip the #include directives
# and prepend the header bodies, because kernels compile at runtime in a cache
# dir with no -I to kernels/src.
assemble() {
  local out="$1" defines="$2" src="$3"; shift 3
  : > "$out"
  [ -n "$defines" ] && printf '%b' "$defines" >> "$out"
  # Skip headers that do not exist, so this script can also fingerprint a
  # pre-SP1 checkout (which has no kv_slot_desc.h) for before/after comparison.
  for hdr in "$@"; do [ -f "$K/$hdr" ] && { cat "$K/$hdr" >> "$out"; printf '\n' >> "$out"; }; done
  sed -e 's|#include "kv_slot_desc.h"||' \
      -e 's|#include "givens_common.h"||' \
      -e 's|#include "turbo_common.h"||' "$K/$src" >> "$out"
}

# name | defines | source | headers (in the runtime's prepend order)
assemble "$WORK/prefill.hip" '#define BR 8\n#define BC 32\n#define NTHREADS 256\n' \
    attention_q8_0_flash_prefill.hip kv_slot_desc.h
assemble "$WORK/q8_lds.hip"  '' attention_q8_0_kv_batched.hip        kv_slot_desc.h
assemble "$WORK/q8_tile.hip" '' attention_flash_q8_0_tile_batched.hip kv_slot_desc.h
assemble "$WORK/asym3.hip"   '' attention_flash_asym3_tile_batched.hip \
    turbo_common.h givens_common.h kv_slot_desc.h
# Sibling that SP1 did not port, but which now receives extra kernargs from the
# shared launcher — included so an accidental edit to it is visible here.
assemble "$WORK/asym2.hip"   '' attention_flash_asym2_tile_batched.hip \
    turbo_common.h givens_common.h

for arch in gfx1151 gfx1201; do
  for k in prefill q8_lds q8_tile asym3 asym2; do
    line=$("$HIPCC" --genco "--offload-arch=$arch" -O3 \
             -Rpass-analysis=kernel-resource-usage \
             -o "$WORK/$k.$arch.o" "$WORK/$k.hip" 2>&1 \
           | awk -F': ' '
               /VGPRs:/            && !v {v=$NF+0}
               /ScratchSize/       && s=="" {s=$NF+0}
               /Occupancy/         && !o {o=$NF+0}
               END{ if (v=="") print "COMPILE_FAILED"; else printf "vgpr=%-4d scratch=%-4d occupancy=%d", v, s, o }')
    printf '%-8s %-9s %s\n' "$arch" "$k" "$line"
  done
done
