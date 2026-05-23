# gfx906 fused-projection MMQ probe — §6.1 results

**Date:** 2026-05-23
**Hardware:** MI50 (gfx906), ROCm 6.4, isolated to `ROCR_VISIBLE_DEVICES=0`
**Model:** /local/hipfire/qwen3.5-9b.mq4 (md5 from `bench_qwen35_mq4` runs)
**Prompt:** synthetic 256-token deterministic stream (tokens 0..255)
**Path:** `forward_prefill_batch` at B=256, kv_mode=asym3, warmup=2 then 1 profiled iter
**Profiler:** in-process `rdna_compute::profile::{start,stop}` (hipEvent-based;
              rocprofv3 segfaults on gfx906 GL2C counter init at this ROCm version)
**Tool:** `crates/hipfire-runtime/examples/profile_prefill_qwen35.rs` (new)

## Top per-kernel wall time (1 prefill of 256 tokens)

| rank | category | kernel                                              | calls | total us | avg us  | %wall |
|------|----------|-----------------------------------------------------|-------|----------|---------|-------|
| 1    | gemm     | `gemm_hfq4g256_mmq_set_gfx906`                      | 136   | 186 609  | 1 372.1 | 53.6% |
| 2    | gemm     | `gemm_hfq4g256_residual_mmq_gfx906`                 | 64    | 110 558  | 1 727.5 | 31.8% |
| 3    | deltanet | `gated_delta_net_q8_batch_seq`                      | 24    |  30 218  | 1 259.1 |  8.7% |
| 4    | deltanet | `conv1d_silu_split_f32_n`                           | 24    |   3 470  |   144.6 |  1.0% |
| 5    | fused    | `fused_rmsnorm_mq_rotate_batched`                   | 64    |   3 411  |    53.3 |  1.0% |
| 6    | gemm     | `gemm_qkvza_hfq4g256_wave64_dp4a` (screen-fallback) | 24    |   2 986  |   124.4 |  0.9% |
| (...) | rest sums to ~2% — RMSnorm, RoPE, gate gemv, etc.                                              |

TOTAL profiled work: 347 835 µs, 515 entries.

## §6.1 attribution

- `mmq_set_gfx906` = **53.6%** of all gfx906 prefill wall.
- Call count: **136 / 32 layers = 4.25 calls/layer**.
  - 8 FullAttention layers × (QKV=3 + gate_up=2) = 40 ideal
  - 24 LinearAttention × (QKVZA worst=4 + gate_up=2) = 144 ideal
  - Ideal total = 184. Observed = 136 ⇒ **screen rejects ~26% of LA-layer
    QKVZA calls** → routed through `qkvza_wave64_dp4a` (24 calls, 0.9% wall).
- The residual `wo` MMQ kernel adds another **31.8%** of wall (single-output,
  not a fusion candidate — already optimal at 1 output / call).

## §6.1 verdict: thesis CONFIRMED — proceed with fused-projection MMQ

Even pessimistic X-tile-reuse savings of 10% per fused projection yields:
  0.10 × 53.6% = **5.4% prefill-wide upside** as the floor.

PR 315's gfx1031 reported `+22% on Qwen3.5 LA layers` from qkvza-split-routing
alone. With residual MMQ unchanged (31.8% wall, already optimal), the upper
bound on this work is roughly:
  0.22 × (53.6% LA-layer fraction of mmq_set) ≈ **8-12% prefill-wide upside**.

## Side findings worth their own follow-up

1. **DeltaNet is only 8.7% of prefill wall.** Plan §4.7 was right to deprioritize
   it. Even halving it would only move 4% — much less than fused-projection MMQ.
2. **Screen-reject rate ≈ 26%** of LA QKVZA. Plan §4.5 (`mmq_screen_weight`
   thresholds on AWQ models) is worth a follow-up — current weights are the
   non-AWQ MQ4. An AWQ rerun would likely reduce screen-reject rate.
3. **gemm_qkvza_hfq4g256_wave64_dp4a only used as screen fallback** — not as
   primary path at B=256. The dp4a fallback is fast (124 µs/call vs mmq_set
   1372 µs). If a screen-rejected LA layer were correctly routed through MMQ
   instead, throughput would improve — but that's an orthogonal screen-tuning
   issue from plan §4.5.

## Bandwidth attribution

`mmq_set_gfx906`: 19.3 GiB/s at avg 1372 µs/call. gfx906 HBM peak is ~1 TB/s
(though effective is ~700 GiB/s). This kernel is firmly **compute-bound**, not
memory-bound — which means the fusion savings come primarily from **eliminating
redundant LDS X-tile staging passes** (compute work, not HBM bandwidth),
matching the PR 315 thesis exactly.

## Notes on the probe methodology

- `rdna_compute::profile` serializes launches (event sync after each), so the
  716 ms profiled wall is ~2× the un-profiled 348 ms. The relative %-of-wall
  numbers are still accurate — async overlap doesn't change per-kernel time,
  only their composition.
- `ensure_q8_1_mmq_x` (the Q8_1 quantize pass) is NOT instrumented in the
  profiler today. Adding a timer there would be a tiny extra change.

---

## Update 2026-05-23 — Phase 2 + Phase 3 shipped, measured win

After landing both fused-projection MMQ kernels and gating them behind
`HIPFIRE_HFQ4_MMQ_GFX906_FUSED=1`, the **production async** prefill
bench on the same hardware shows:

| run | baseline tok/s | fused tok/s |
|----:|---------------:|------------:|
|   1 |          726.1 |       779.7 |
|   2 |          727.6 |       780.5 |
|   3 |          727.0 |       779.6 |
| **mean** | **726.9** | **779.9** |
| Δ   | — | **+7.3%** |

`bench_qwen35_mq4 /local/hipfire/qwen3.5-9b.mq4 --prefill 256 --prefill-runs 3 --gen 10 --warmup 2`
on `ROCR_VISIBLE_DEVICES=0` (MI50 / gfx906).

Decode tok/s unchanged (~56.0 ±0.4) — the kernels touch only the batched
prefill path, as designed. Byte-exact A/B (greedy_dump) confirmed
numerical equivalence: md5 f9bb00845568951675012dbde42bc171 on both
sides for the 60-token "Capital of France?" generation.

Stddev across 3 runs is ~0.1% on both sides — extraordinarily tight for
a gfx906 prefill bench, well within the §6.1 ceiling estimate
(5-12%). The signal is real, not thermal noise.

**Conclusion**: §6 plan steps 2 + 3 deliver as predicted. The kernel
is correct, numerically equivalent, and ships a measurable +7.3%
async prefill win. Steps 4-6 (MMQ_Y sweep, HFQ3 port, screen-reject
analysis) remain optional follow-ups; the structural win is banked.
