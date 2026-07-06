---
title: DFlash spec-decode verify is at its kernel ceiling on gfx1100 — int8 / LDS-tiling / graphing all dead
date: 2026-06-23
tags: [spec-decode, dflash, perf, gfx1100, falsification]
---

The DFlash verify forward (the big mq4 GEMMs over a draft block of B≈16) is
**memory/occupancy-bound, not compute-bound**, and every obvious speedup is
falsified. Don't re-chase these.

- **int8 / dp4a: dead.** rocprof (on the *daemon*) shows verify GEMMs BW-saturated
  (lm_head at ≥peak 960 GB/s) to occupancy-bound (gate_up ~48%, LDS=0), never
  compute-bound. int8-COMPUTE can't shrink the traffic: weights stream at **4-bit
  (mq4 storage) regardless of compute dtype** — the bandwidth win is already banked
  by 4-bit storage. int8 only halves the negligible activation term (~0.04% of
  traffic), and its 2× accumulator VGPRs spill. RDNA also has no scalar `v_dot4_i32_i8`
  ("int8 dp4a" = int8-WMMA), already reverted (−11..−34.5% prefill).
- **LDS-tiling: dead.** gate_up at 48% BW *looked* like LDS-tiling headroom, but it
  was tried 3 ways (`ldscoop` −14% E2E, `2tile` noise, `k4` 0%) and the line was
  closed (`docs/plans/mtp-gate-up-wmma-ceiling-2026-05-21.md`) — the base kernel's
  per-row L1 reuse is already good. The `HIPFIRE_GATE_UP_VARIANT` variants are also
  broken cross-request. (A "2tile loses 76.7 vs 83.5" number was a JIT-cold artifact —
  retracted.)
- **Graphing: ~1%.** The verify-graph is already the *new* blob-based hipGraph arch;
  measured benefit is ~1% (graph_ON 160.8 vs OFF 159.1) because the verify is
  kernel-bound, not launch-bound like AR decode (where AR_graph gives +1.4–9.9%). A
  draft/cycle-graph would be sub-0.3%.

**Only real spec-decode levers left:** drafter τ (retrain — infra-blocked) and the
shipped durable mode selection (greedy DFlash + sampled DFlash + MTP).

**Profiling gotcha:** rocprofv3 segfaults on the eager DFlash *demo*
(`dflash_spec_demo`, high dispatch) — even kernel-trace. It survives on the
**daemon** (hipGraph collapses dispatch). Profile DFlash via the daemon, stdin-piped.

Related: see `measure-spec-decode-on-the-daemon`.
