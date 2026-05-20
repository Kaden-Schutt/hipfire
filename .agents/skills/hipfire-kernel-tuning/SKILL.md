---
name: hipfire-kernel-tuning
description: Optimize hipfire HIP/compute kernels — pick a tuning lever (multi-row, K-tile depth, prefetch, wave-size port, WMMA/MFMA, fused projections, ISA flags) and validate the win across the supported RDNA arch matrix. Use when you've identified a hot kernel, want to land a real perf win, and need to NOT regress on archs you don't have hardware for. Codifies the methodology from this repo's actual perf history — wave64 CDNA3 port (commit 4105035, 2× decode), nontemporal-load revert (34eb024, -13% caught only by clean-baseline bisect), gfx12 WMMA port (PR #56). Triggers on phrases like "tune kernel X", "optimize gemv on RDNA*", "make kernel Y faster on gfx*", "perf regression on <arch>", "should I add a multi-row variant", "kernel runs slow at low batch sizes".
---

# hipfire-kernel-tuning

Skill for landing real kernel perf wins in hipfire without breaking
cross-arch portability or shipping a regression that the speed-gate
catches but you talked yourself past. Codifies the empirical
methodology that's actually worked across the gfx1010 → gfx1201 +
gfx94x matrix.

## When to use

- A profiler / `crate::profile` timer flagged a hot kernel as the
  bottleneck and you want to pick the right lever.
- You have a candidate optimization (multi-row variant, deeper K
  unroll, wave64 port, ISA flag) and need to validate it doesn't
  regress on the cross-arch matrix.
- The speed-gate flagged a regression on a "should-be-no-op"
  refactor and you need the bisect / fresh-process recipe.
- You have R9700 (or any new arch hardware) and want to write
  arch-specific fast paths beyond the canonical port.

## Read these in order

1. **`playbook.md`** — measure → root-cause → pick lever → validate
   → ship. The 6-step workflow with the gates each step has to clear.
   Start here.
2. **`levers.md`** — catalog of optimization patterns hipfire actually
   uses (wave64 port, multi-row GEMV, K-tile depth, `s_prefetch_data`,
   WMMA/MFMA, fused projections, per-kernel hipcc flags). Each lever
   names the commits where it landed (or was reverted) so you can
   read the diff.
3. **`cross-arch.md`** — dispatch routing rules. How to add a new
   fast path that wins on gfx1100 without regressing gfx1010
   /gfx1030 /gfx1200 /gfx94x. The "no unreachable branches" rule and
   why predicate helpers (`has_wmma_f16`, `has_dot2_f32_f16`) exist.
4. **`case-studies.md`** — five worked examples from the git log:
   wave64 CDNA3 port (+2× decode), nontemporal-load fake-win revert
   (−13% caught), prompt-shape recovery (+24% τ), k2x32 null result,
   WMMA C-mapping silent-corruption fix. Read these to calibrate
   what a real win looks like vs a measurement artifact.

## Key rules

- **Trust the speed-gate, not your gut.** Within-session A/B noise
  on gfx1100 is ±10–15%. A "+8% win" measured by editing code +
  re-running in the same shell is inside the noise band. Use
  `scripts/probe_commits.sh <baseline> <candidate>` for cross-process
  measurement.
- **Bisect against the committed baseline, not against your last
  bench run.** This is how the nontemporal-load fake +2% got caught
  as an actual −13%. See `case-studies.md` §2.
- **Negative results ship too** — if a lever LOOKS like it should
  win and doesn't, document it in the commit message with the
  hypothesis why. Future contributors save the same hours.
- **Cross-arch verify before merging.** The speed-gate runs on the
  baseline arch (gfx1100 typically); your change still needs to
  not regress gfx1010/gfx1030 if the relevant codepath touches them.
  See `cross-arch.md`.

## What's not in this skill

- **New arch ports** — see `.agents/skills/hipfire-arch-port/` instead. That
  skill covers porting an EXISTING kernel to a new GPU family. This
  skill covers MAKING an existing kernel faster on the archs it
  already runs on.
- **DFlash spec-decode tuning** — those wins come from algorithm
  changes (n-gram cache, prompt shape, draft retraining), not kernel
  ISA work. See `crates/hipfire-runtime/src/dflash.rs` and the
  `coherence-gate-dflash.sh` battery.

## Cross-references

- [`docs/methodology/perf-benchmarking.md`](../../docs/methodology/perf-benchmarking.md) — the
  bench protocol (within-session noise band, stale-binary trap,
  prompt-md5 discipline).
- [`docs/QUANTIZATION.md`](../../docs/QUANTIZATION.md) — MQ4/HF4 design
  + asym KV math; required reading before touching quant kernels.
  If this doc conflicts with `AGENTS.md` on MQ3/DFlash release status, treat
  `AGENTS.md` and current code as the fresher source.
- [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — the dispatch
  layering + two-model-paths surface that constrains where new
  variants can plug in.
- `tests/speed-baselines/<arch>.txt` — the committed perf floor per
  arch. The speed-gate compares against these.
