#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Cross-process determinism check for ds4_parent_forward_gate.
# Runs the identical invocation in two separate processes and asserts
# the printed logits_sha256 values match. Chosen over an in-binary
# --verify-determinism re-exec so the two processes share nothing.
#
# Usage:
#   ds4_parent_forward_gate_determinism.sh -- <gate-args...>
# Example:
#   ./ds4_parent_forward_gate_determinism.sh -- \
#       --model /mnt/scratch/models/DeepSeek-V4-Flash-0731 \
#       --token-ids /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin \
#       --plog /mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/run1.plog
#
# Environment:
#   GATE_BIN   path to ds4_parent_forward_gate (default: look next to this script
#              then in $CARGO_TARGET_DIR/release/examples/)
#   OUT_DIR    directory for the two run logs (default: same dir as --plog, else cwd)

set -euo pipefail

extract_sha() {
  # Prefer the in-process reference line; fall back to any logits_sha256=.
  local f="$1"
  local line
  line=$(grep -E 'logits_sha256 \(in-process reference\) = ' "$f" | tail -n1 || true)
  if [[ -z "$line" ]]; then
    line=$(grep -E 'logits_sha256=' "$f" | tail -n1 || true)
  fi
  if [[ -z "$line" ]]; then
    echo "deepseek4 parent: no logits_sha256 found in $f" >&2
    return 1
  fi
  # Last whitespace-separated field.
  awk '{print $NF}' <<<"$line" | tr -d '\r'
}

GATE_BIN="${GATE_BIN:-}"
if [[ -z "$GATE_BIN" ]]; then
  here=$(cd "$(dirname "$0")" && pwd)
  for cand in \
    "$here/../../../target/release/examples/ds4_parent_forward_gate" \
    "${CARGO_TARGET_DIR:-}/release/examples/ds4_parent_forward_gate" \
    "./target/release/examples/ds4_parent_forward_gate"
  do
    if [[ -n "$cand" && -x "$cand" ]]; then
      GATE_BIN="$cand"
      break
    fi
  done
fi
if [[ -z "${GATE_BIN}" || ! -x "$GATE_BIN" ]]; then
  echo "deepseek4 parent: GATE_BIN not found/executable (set GATE_BIN=...)" >&2
  exit 2
fi

# Strip a leading "--" if present (allows `script -- args...`).
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 -- <ds4_parent_forward_gate args...>" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-.}"
mkdir -p "$OUT_DIR"
ts=$(date -u +%Y%m%dT%H%M%SZ)
log1="$OUT_DIR/determinism_run1_${ts}.log"
log2="$OUT_DIR/determinism_run2_${ts}.log"

echo "=== ds4_parent_forward_gate_determinism ==="
echo "GATE_BIN=$GATE_BIN"
echo "run1 log: $log1"
echo "run2 log: $log2"
echo "args: $*"
echo

echo "--- process 1 ---"
set +e
"$GATE_BIN" "$@" >"$log1" 2>&1
rc1=$?
set -e
echo "exit=$rc1"
sha1=$(extract_sha "$log1") || {
  echo "FAIL: could not extract logits_sha256 from run1" >&2
  tail -n 40 "$log1" >&2 || true
  exit 1
}
echo "logits_sha256_run1=$sha1"

echo
echo "--- process 2 ---"
set +e
"$GATE_BIN" "$@" >"$log2" 2>&1
rc2=$?
set -e
echo "exit=$rc2"
sha2=$(extract_sha "$log2") || {
  echo "FAIL: could not extract logits_sha256 from run2" >&2
  tail -n 40 "$log2" >&2 || true
  exit 1
}
echo "logits_sha256_run2=$sha2"

echo
if [[ "$sha1" == "$sha2" && -n "$sha1" ]]; then
  echo "CROSS_PROCESS_DETERMINISM: PASS"
  echo "logits_sha256=$sha1"
  # Non-zero gate exit still surfaces; hash match is independent.
  if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
    echo "NOTE: gate exit codes were rc1=$rc1 rc2=$rc2 (hash still matched)"
    exit 1
  fi
  exit 0
else
  echo "CROSS_PROCESS_DETERMINISM: FAIL"
  echo "run1=$sha1"
  echo "run2=$sha2"
  echo "FINDING: parent teacher is not cross-process deterministic"
  exit 1
fi
