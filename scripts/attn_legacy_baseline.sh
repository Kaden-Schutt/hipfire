#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.
#
# Legacy-path regression gate for the SP1 multi-slot batched-attention work.
#
# Tasks 4-6 add slot descriptors to four attention kernels. Every one of them
# must keep a null-descriptor path that is BITWISE IDENTICAL to the pre-SP1
# behaviour. This script produces the canonical fingerprint of that behaviour.
#
# Why filtered output: the raw example prints cargo compile chatter and
# kernel-cache messages that legitimately vary run to run (verified: two
# consecutive runs of the same binary differ in 39 leading lines). Diffing raw
# output produces false failures. Only the result lines are numerically
# meaningful, and those are what we fingerprint.
#
# Usage:
#   scripts/attn_legacy_baseline.sh > baseline.txt      # capture
#   diff baseline.txt <(scripts/attn_legacy_baseline.sh)  # compare
#
# Shapes cover both target models' full-attention layers:
#   qwen3.6-35b-a3b : n_heads=16 n_kv_heads=2  head_dim=256  (GQA 8:1)
#   qwen3.6-27b     : n_heads=24 n_kv_heads=4  head_dim=256  (GQA 6:1)
# plus small/awkward shapes that exercise tile remainders and short contexts.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

run() {
  local label="$1"; shift
  # Each pair is VAR=value; pass through to the example's env interface.
  local out
  out=$(env "$@" cargo run --release -p rdna-compute --features deltanet \
        --example test_q8_flash_prefill 2>&1)
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "$label :: HARNESS_FAILED rc=$rc"
    return
  fi
  # Keep only numerically meaningful lines.
  echo "$out" | grep -E '^(kernel=|max_abs=|PASS|FAIL)' | sed "s|^|$label :: |"
}

# --- 35B-A3B shape (GQA 8:1) ---
run "a3b/ctx32_n16"    NH=16 NKV=2 HD=256 N=16  CTX=32
run "a3b/ctx512_n16"   NH=16 NKV=2 HD=256 N=16  CTX=512
run "a3b/ctx4096_n64"  NH=16 NKV=2 HD=256 N=64  CTX=4096
run "a3b/ctx8192_n256" NH=16 NKV=2 HD=256 N=256 CTX=8192

# --- 27B shape (GQA 6:1) ---
run "27b/ctx32_n16"    NH=24 NKV=4 HD=256 N=16  CTX=32
run "27b/ctx512_n16"   NH=24 NKV=4 HD=256 N=16  CTX=512
run "27b/ctx4096_n64"  NH=24 NKV=4 HD=256 N=64  CTX=4096
run "27b/ctx8192_n256" NH=24 NKV=4 HD=256 N=256 CTX=8192

# --- awkward shapes: non-multiples, tiny context, single query row ---
run "odd/ctx7_n1"      NH=16 NKV=2 HD=256 N=1   CTX=7
run "odd/ctx129_n5"    NH=16 NKV=2 HD=256 N=5   CTX=129
run "odd/ctx131_n17"   NH=24 NKV=4 HD=256 N=17  CTX=131
