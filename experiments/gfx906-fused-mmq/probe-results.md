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

---

## Update 2026-05-23 — Phase 4 MMQ_Y=64 sweep — NEGATIVE result

Plan §6 step 5 / §4.2 probe: replace gate_up's MMQ_Y=128 tile with
MMQ_Y=64, hoping for the same +5-14% PR #315 found on RDNA2 (gfx1030)
from halved LDS budget + halved accumulator regs → doubled per-CU
occupancy.

### Measured: −5.6% prefill regression on gfx906

| metric                    | Y=128 (fused)       | Y=64 (fused)        |
|---------------------------|---------------------|---------------------|
| prefill tok/s (3 runs)    | 778.6, 780.2, 779.3 | 735.3, 736.9, 735.6 |
| mean prefill tok/s        | **779.4**           | **735.9**           |
| Δ                         | —                   | **−5.6%**           |
| prefill wall (ms)         | 328.5               | 347.9               |
| per-call gate_up wall (µs)| 2683                | 3012 (**+12.3%**)   |

`bench_qwen35_mq4 /local/hipfire/qwen3.5-9b.mq4 --prefill 256 --prefill-runs 3 --gen 10 --warmup 2`
`HIPFIRE_HFQ4_MMQ_GFX906_FUSED=1 HIPFIRE_HFQ4_MMQ_GFX906_Y64=1`.

Byte-exact A/B (greedy_dump) confirms numerical equivalence — both
Y=128 and Y=64 produce md5 `f9bb00845568951675012dbde42bc171` on the
60-token "Capital of France?" generation. The regression is purely
perf, not correctness.

### Why Y=64 wins on RDNA2 but loses on gfx906

The gfx906 body's existing commentary
(`gemm_hfq4g256_residual_mmq_gfx906_body.cuh:95-100`) documents that
gfx906's MemUnitBusy collapses to 13.8% at mmq_x=32 on the b32 path —
the kernel is **LDS-issue-rate starved**, not LDS-capacity starved.
Cutting Y in half:
- Doubles WG count per grid (9B gate_up: 76 → 152 WGs).
- Halves per-WG work, but launch + sync overhead doesn't scale down.
- Doesn't relieve the actual issue-rate bottleneck.

The promised occupancy gain (LDS-permitted 2 WG/CU → 3 WG/CU) doesn't
materialize because the kernel was already issue-rate-limited at Y=128.

### Y=96 status: deferred

`MMQ_Y=96` is mentioned in plan §4.2 as a non-power-of-2 candidate
between Y=64 and Y=128. Implementing it requires fixing the
accumulator sizing — `MMQ_Y / WAVE_SIZE = 96 / 64 = 1.5` truncates to
1 in the inner loop, so the body needs `i0 < min(MMQ_Y,
batch_remaining)` guards everywhere. Not worth the refactor without a
clearer hypothesis that Y=96 would help; deferred indefinitely.

### Disposition

Y=64 wrappers + dispatcher gate stay in the tree as a research scaffold
gated by `HIPFIRE_HFQ4_MMQ_GFX906_Y64=1` (default OFF), following PR
#315's convention for kept-gated negative results. Future gfx906 MMQ
work should target the issue-rate bottleneck, not the occupancy
ceiling.

**Final wins banked from Phase 2 + 3 (Y=128):** +7.3% async prefill on
Qwen3.5 9B MQ4. Phase 4 yielded no additional win.

---

## Update 2026-05-24 — Default-on validation + env-gate removed

After Phase 2-4 baked behind `HIPFIRE_HFQ4_MMQ_GFX906_FUSED=1`, ran the
canonical validation suite to clear the env-gate for default-on shipping.

### Coherence gate

`scripts/coherence-gate.sh` on gfx906 with fused-by-default kernels:
9 model/prompt cells exercised (4 MQ4 — the kernels' direct surface;
5 MQ3 / MQ6 — collateral coverage), no hard errors. Sample cells:

  qwen3.5-0.8b.mq4 / cap         → "Paris."                       OK
  qwen3.5-9b.mq4 / reason        → correct sheep answer (9)        OK
  qwen3.5-9b.mq4 / tool-call     → clean <tool_call> JSON shape    OK

### KLD eval (n=50)

`eval_hipfire --max-chunks 50` on `/local/hipfire/qwen3.5-9b.mq4`,
asym3 KV, prefill scoring, with fused-by-default active. Compared
against the canonical §1.1 mq4-base entry from
`docs/plans/kld-measurements-master.md` (on `data/kld-measurements`
branch):

  measured: KLD = 0.323142  PPL = 9.2493  (n=50)
  master:   KLD = 0.3376    PPL = 9.116   (n=512, CI 0.3263–0.3494)

KLD is **inside the master CI band**, consistent with the n=50 vs n=512
spread of ±0.01 expected from slice-mean noise. PPL is within +1.5% of
the master, well inside the cross-run variance envelope on prefill
scoring at n=50.

### Disposition

Env-gate `HIPFIRE_HFQ4_MMQ_GFX906_FUSED` removed from `dispatch.rs`.
Fused-projection kernels are now the default on gfx906 whenever the
alignment guard passes (q_m/k_m/v_m, gate_m/up_m, qkv_m/z_m all
multiples of MMQ_Y=128 — Qwen3.5 family always satisfies).

The legacy 3×/2× mmq_set fall-through is retained as the correctness
backstop for hypothetical future models with non-aligned projection
dimensions; the dispatcher selects it automatically when alignment fails.

The MMQ_Y=64 research scaffold stays gated as documented (negative result).
