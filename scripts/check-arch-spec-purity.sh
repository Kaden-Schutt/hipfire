#!/usr/bin/env bash
# Arch capability-layer purity gate.
#
# The base capability crate (hipfire-arch-api) and every arch `-spec` core must
# express NEEDS, never format SOLUTIONS: no concrete on-disk quant/format token may
# appear in their source. An arch declares importance + requirements; the
# deployment maps that to a codec. This is what structurally prevents the
# `is_q8_tensor` / `is_deepseek4_keep_f16` smell from regrowing.
# See docs/plans/2026-07-03-arch-capability-layer.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Crates whose src must be format-token-free: hipfire-arch-api today; each arch
# `-spec` core is auto-included by the glob as it lands.
CRATES=(crates/hipfire-arch-api)
for d in crates/*-spec; do
  [ -d "$d" ] && CRATES+=("$d")
done

# Banned tokens: concrete on-disk quant formats. NOT bare `f16`/`i32` (legit
# dtypes) — the ban is on naming a FORMAT as a policy decision, e.g. Oq4, Qtip3, Q8.
PATTERN='\b(Oq[0-9]|Mq[0-9]|Qtip[0-9]|oq[0-9]l?|mq[0-9]l?|qtip[0-9])\b|\bDType::|\bQ8\b|\bQ4F16\b'

fail=0
scanned=0
for c in "${CRATES[@]}"; do
  [ -d "$c/src" ] || continue
  scanned=$((scanned + 1))
  if hits=$(grep -rnE "$PATTERN" "$c/src" 2>/dev/null); then
    echo "FAIL: format token in capability-layer crate '$c' (declare NEEDS, not formats):"
    echo "$hits"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "arch-spec purity OK ($scanned crate(s) format-token-free)"
fi
exit "$fail"
