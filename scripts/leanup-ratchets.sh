#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Ratchets for docs/governance/2026-08-15-hipfire-leanup-map.md § 4.
#
# Run from the repository root. Every number is measured, never asserted.
#
# On the compute:arch ratio — read this before quoting it
# -------------------------------------------------------
# The original ratchet counted `.rs` files in eight compute crates and
# compared the result against llama.cpp's `ggml/`. That is not a like-for-like
# comparison: `ggml/` is almost entirely kernel source (.c/.cpp/.cu/.metal/
# .cl/.comp), while hipfire's equivalent — `kernels/`, ~120k lines of HIP —
# was excluded from its own compute side. The leanup map's § 6 explicitly
# names "the kernel family" as part of the compute layer, so excluding it
# contradicted the very definition being measured.
#
# This script reports the ratio three ways so the rule is visible rather than
# buried:
#
#   crates-only   what the original ratchet measured. Kept for continuity
#                 with the historical figure; not a fair comparison.
#   all-kernels   every kernel line on the compute side, which is the rule
#                 `ggml/` is measured under.
#   strict        arch-named kernels (deepseek4_*, fused_gemma4_*, …) moved
#                 to the arch side. llama.cpp has zero model-named files in
#                 `ggml/` — verified — so this is the closest true analogue
#                 and is the number to quote.
set -uo pipefail
cd "${1:-.}" || exit 1

lines() { find $1 -type f \( ${2} \) 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1+0}'; }
RS='-name *.rs'
KS='-name *.hip -o -name *.h -o -name *.hpp -o -name *.cpp -o -name *.cl'
p() { printf '%-26s %s\n' "$1" "$2"; }

DAEMON=crates/hipfire-daemon/src/main.rs
p HEAD "$(git rev-parse --short HEAD 2>/dev/null)"
p daemon_lines "$(wc -l < $DAEMON)"
p daemon_arch_id "$(grep -cE 'arch_id *==' $DAEMON)"
p daemon_arch_refs "$(grep -coE 'hipfire_arch_[a-z0-9_]+' $DAEMON)"
p required_features "$(grep -c 'required-features' crates/hipfire-daemon/Cargo.toml)"
p runtime_examples "$(grep -c '^\[\[example\]\]' crates/hipfire-runtime/Cargo.toml)"
p grammar_copies "$(find crates/hipfire-arch-*/src -name grammar.rs 2>/dev/null | wc -l)"
p glossary "$([ -f docs/GLOSSARY.md ] && echo present || echo MISSING)"

big=0
for d in crates/hipfire-arch-*/; do
  n=$(lines "$d/src" "$RS")
  [ "${n:-0}" -gt 10000 ] && { p "OVER_10k" "$(basename $d) $n"; big=$((big+1)); }
done
p arch_crates_over_10k "$big"

c=0
for x in rdna-compute redline redline-dispatch redline-rocr radiowave \
         hip-bridge hsa-bridge hipfire-detect; do
  c=$((c + $(lines "crates/$x/src" "$RS")))
done
a=0
for d in crates/hipfire-arch-*/; do a=$((a + $(lines "$d/src" "$RS"))); done

k_all=$(lines kernels "$KS")
k_arch=$(find kernels -type f \( $KS \) 2>/dev/null \
         | grep -iE '/[^/]*(qwen|deepseek|llama|gemma|cohere|minimax|glimmer|lfm)[^/]*$' \
         | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1+0}')
k_gen=$((k_all - k_arch))

r() { awk -v c=$1 -v a=$2 'BEGIN{printf "%.3f : 1", c/a}'; }
p compute_crates_rs "$c"
p kernels_total "$k_all"
p kernels_arch_named "$k_arch"
p arch_crates_rs "$a"
p 'ratio (crates-only)' "$(r $c $a)   <- original ratchet; not like-for-like"
p 'ratio (all-kernels)' "$(r $((c+k_all)) $a)"
p 'ratio (strict)' "$(r $((c+k_gen)) $((a+k_arch)))   <- quote this one"
