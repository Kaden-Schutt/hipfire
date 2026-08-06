---
name: hipfire-kernel-tuning
description: Optimize hipfire HIP/compute kernels — pick one tuning lever (multi-row, K-tile depth, prefetch, wave-size port, WMMA/MFMA, fused projections, ISA flags) and validate it with profile → ISA → fresh-process measurement. Use when a hot kernel is identified, you want a real perf win, and you must not regress adjacent archs or promote unverified deltas. Codifies the methodology from this repo's perf history (wave64 CDNA3 port 4105035, nontemporal-load revert 34eb024, gfx12 WMMA PR #56, barrier-free nosync wins). Triggers on "tune kernel", "optimize gemv/gemm", "make kernel faster", "perf regression on <arch>", "multi-row variant", "wave64 port", "WMMA performance".
---

# hipfire-kernel-tuning

Land real kernel perf wins without inventing a universal gate, shipping
measurement noise, or silently regressing an adjacent arch.

This skill is **workflow only**. Mutable numbers, noise bands, admission
policy, and claim→route maps live in canonical owners under `docs/`. Do not
copy those tables into commits or skill prose.

## When to use

- A profile (`HIPFIRE_PROFILE`, Kernel Atlas, or rocprof) names a hot kernel
  and you need the right lever.
- You have one candidate change (multi-row, deeper K-tile, wave64, chip
  override, barrier-free path, hipcc flag) and must prove it on the target
  arch without breaking others.
- A "should-be-no-op" dispatch refactor needs clean-baseline bisect after a
  speed-floor warning.
- You have hardware for a chip (e.g. gfx1201) and want an arch-specific
  fast path beyond a family port.

## Read order

1. **`playbook.md`** — profile → root-cause → one lever → source/ISA inspect
   → adjacent-arch boundaries → correctness route → fresh-process measure →
   reject/log. Start here.
2. **`levers.md`** — catalog of patterns that shipped or failed in this tree,
   with kernel paths under `kernels/src/` and commits to read.
3. **`cross-arch.md`** — file tags, dispatch fall-through, "no unreachable
   branches", and when **not** to flip public dispatch without hardware.
4. **`case-studies.md`** — worked wins, fake wins, null results, and silent
   corruption. Calibrate what "real" looks like before claiming.

## Non-negotiable rules

1. **One hypothesis, one lever, one commit.** Bundle three speculative
   changes and you cannot bisect a fake win.
2. **Profile before editing.** No "this feels slow" optimizations.
3. **Inspect source + ISA before claiming the bottleneck class.** Occupancy,
   spills, BW%, launch count, and matrix-op mix decide the lever family.
4. **Capability predicates gate ISA correctness; they never select a perf
   sub-variant.** Perf picks are measured arch allowlists with a conservative
   default — see [`docs/methodology/perf-arch-discipline.md`](../../../docs/methodology/perf-arch-discipline.md).
5. **Correctness route is claim-scoped.** Pick the narrowest row in
   [`docs/VALIDATION.md`](../../../docs/VALIDATION.md). There is **no**
   universal GPU gate. Retired `coherence-gate-*.sh` batteries are not
   acceptance.
6. **Perf claims need the protocol**, not a one-shell A/B:
   [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md).
   Fresh process, warmup, binary/prompt md5, repeated/raw samples, and a
   surface-matched harness — not bare `probe_commits.sh` / `gates.sh --perf`
   alone (see Supporting tools).
7. **Negative results ship.** Log the rejection (commit message and/or
   Atlas task ledger) so the next pass does not re-burn the same hours.
8. **No unverified promotion.** A local delta is **measured**, not product
   default, floor, or admission. Admissions stay in
   [`docs/admissions.yml`](../../../docs/admissions.yml) (fail closed when empty).

## What's not in this skill

| Concern | Go here instead |
|---|---|
| New arch / new WMMA builtin port | `.agents/skills/hipfire-arch-port/` |
| ISA Fit View / Atlas collect-eval loop | `.agents/skills/hipfire-kernel-atlas/` + [`docs/methodology/kernel-atlas.md`](../../../docs/methodology/kernel-atlas.md) |
| Bring-up / smoke matrix | `.agents/skills/hipfire-tester/` |
| Runtime hang / missing kernel triage | `.agents/skills/hipfire-autoheal/` / `.agents/skills/hipfire-diag/` |
| Spec-decode *algorithm* tuning (n-gram, draft, prompt shape) | runtime DFlash sources — not ISA levers |
| Redline / retained-replay certification | [`docs/REDLINE.md`](../../../docs/REDLINE.md) |

## Canonical owners (link; do not duplicate)

| Concern | Owner |
|---|---|
| Claim → validation route | [`docs/VALIDATION.md`](../../../docs/VALIDATION.md) |
| Warmup, fresh-process, prompt md5, speed-floor use | [`docs/methodology/perf-benchmarking.md`](../../../docs/methodology/perf-benchmarking.md) |
| Capability vs perf-variant selection | [`docs/methodology/perf-arch-discipline.md`](../../../docs/methodology/perf-arch-discipline.md) |
| Channel + speed for *arch ports* | [`docs/methodology/arch-port-validation.md`](../../../docs/methodology/arch-port-validation.md) |
| Atlas rows, ISA manifests, suggest/task/eval | [`docs/methodology/kernel-atlas.md`](../../../docs/methodology/kernel-atlas.md) |
| Quant math before touching quant kernels | [`docs/QUANTIZATION.md`](../../../docs/QUANTIZATION.md) |
| Crate / dispatch overview | [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md) |
| Docs lifecycle / ownership map | [`docs/INDEX.md`](../../../docs/INDEX.md) |

## Supporting tools (claim-scoped)

Use only when the claim class needs them — full roles in `VALIDATION.md`:

- Profile: `HIPFIRE_PROFILE=1`, optional rocprof kernel-trace, Atlas
  `collect-ar` / `collect-dflash` with `--profile-*` and `--isa-*`.
- Compile matrix: `scripts/compile-kernels.sh` for touched chip/family tags.
- Numeric kernel check: `target/release/examples/test_kernels` (build via
  `hipfire-runtime` `test_kernels` example).
- Speed floor (when policy applies): `scripts/speed-gate.sh` vs
  `tests/speed-baselines/<arch>.txt`.
- Fresh-process A/B: prefer surface-matched harness with repeated refs and
  raw samples. `scripts/probe_commits.sh` is Qwen3.5/in-process only (one
  sample per ref) — not protocol-complete alone. Do not treat
  `scripts/gates.sh --perf` as complete evidence (single baseline/HEAD sample
  plus unrelated Redline/serve arms).
- Path-specific parity oracles: arch-owned `dump_*_hidden_states` /
  graph-parity examples when the change can break state — **blocked** if no
  oracle exists for that surface.
- Serve semantics only: `scripts/serve_harness.py` / LFM
  `scripts/serve_harness.py` — never numerical parity substitutes.

`.agents/skills/hipfire-kernel-tuning/` is the sole executable root for this
skill.
