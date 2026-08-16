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
#                 `ggml/` — verified.
#   +substrate    strict, plus the engine substrate. SECOND measurement defect:
#                 the compute list below was written before the saddle layering
#                 existed and was never updated, so `saddle-core`,
#                 `hipfire-engine` and `hipfire-dispatch` — which carry ZERO
#                 `hipfire_arch_*` references and ZERO arch Cargo deps, and so
#                 cannot be arch code under any reading — were counted on
#                 NEITHER side. llama.cpp's analogue (`src/` minus
#                 `src/models/`: llama-context, llama-kv-cache, llama-batch,
#                 llama-sampling) is 53,974 lines and is likewise not arch.
#                 Quote THIS one; it is the conservative figure.
#   +dispatchers  also counts hipfire-runtime/loader/generate, which reference
#                 arch only to dispatch into it, exactly as llama.cpp's
#                 `llama-model.cpp` switches over `LLM_ARCH_*`. Upper bound.
#
# Measured llama.cpp for calibration (see docs/governance): ggml:src/models is
# 16.20 : 1 and (ggml+substrate):src/models is 19.19 : 1. The 9.7 : 1 figure
# quoted in the original grounding doc could not be reproduced from the tree.
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
# `daemon_arch_refs` greps `hipfire_arch_*`, which `ModelState::Qwen35` does NOT match:
# ModelState is a LOADER-owned enum wrapping arch bundles. The daemon therefore reported
# 0 arch refs while still doing a 7-way architecture dispatch (main.rs:1732-1751). Count
# the laundered form too, or the gate certifies a decoupling that has not happened.
p daemon_modelstate "$(grep -co 'ModelState::' $DAEMON)"
p loader_modelstate "$(grep -rho 'ModelState::' crates/hipfire-loader/src | wc -l)"
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
# Engine substrate: the layers the saddle work created. Split by whether the
# crate names an architecture at all, so the conservative figure stands without
# argument.
sub_clean=0
for x in saddle-core hipfire-engine hipfire-dispatch; do
  sub_clean=$((sub_clean + $(lines "crates/$x/src" "$RS")))
done
sub_disp=0
for x in hipfire-runtime hipfire-loader hipfire-generate; do
  sub_disp=$((sub_disp + $(lines "crates/$x/src" "$RS")))
done
# Guard the conservative bucket: if any of those three ever gains an arch
# reference it stops being unambiguous substrate and this must be revisited.
leak=$(grep -roE 'hipfire_arch_[a-z0-9_]+' crates/saddle-core/src crates/hipfire-engine/src \
        crates/hipfire-dispatch/src 2>/dev/null | wc -l)
p substrate_clean "$sub_clean"
p substrate_dispatching "$sub_disp"
p substrate_clean_arch_refs "$leak$([ "$leak" -eq 0 ] && echo '' || echo '  <- NOT clean; conservative ratio invalid')"
p 'ratio (crates-only)' "$(r $c $a)   <- original ratchet; not like-for-like"
p 'ratio (all-kernels)' "$(r $((c+k_all)) $a)"
p 'ratio (strict)' "$(r $((c+k_gen)) $((a+k_arch)))   <- kernels fixed, substrate still omitted"
p 'ratio (+substrate)' "$(r $((c+k_gen+sub_clean)) $((a+k_arch)))   <- quote this one"
p 'ratio (+dispatchers)' "$(r $((c+k_gen+sub_clean+sub_disp)) $((a+k_arch)))   <- upper bound"
