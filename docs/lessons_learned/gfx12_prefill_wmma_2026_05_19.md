# gfx12 prefill WMMA breakthrough — lessons learned (2026-05-19)

## TL;DR

A3B 256-token prefill on R9700/gfx1201 jumped **1016 → 2966 tok/s (+192%)**
by ripping out a scalar Q8 GEMM that was invisible to our internal profiler.
Fix is one new kernel + one auto-route gate: commit **218a88df**
("WMMA Q8_0 GEMM auto-routes from `gemm_q8_0_batched` on gfx12"). The
operator-selectable Q8 DeltaNet state that makes the win visible at the
daemon layer landed alongside it as **da653e61** (`params.dn_quant=q8`,
production default for MoE stays FP32 — operator opts in for short-output
workloads or ablations).

## The methodology lesson (most important)

**Internal profilers can lie by omission.** `HIPFIRE_PROFILE` only times
kernels whose dispatchers call `crate::profile::begin_timer`. The
fallback path `gemm_q8_0_batched` had no such call. Result: it
consumed **65% of GPU time per prefill (87ms of 135ms)** but registered
**0% in the Atlas profile**. The "167ms of overhead" we'd been chasing
was misattribution — the kernel was running, just invisible.

`rocprofv3 --kernel-trace` is the authoritative source for GPU-side
time. **Always run rocprof alongside HIPFIRE_PROFILE before making
kernel-level optimization decisions.** Reconciliation rule: any kernel
that shows up in rocprof but is missing from HIPFIRE_PROFILE — retrofit
`begin_timer` before you trust the Atlas numbers for that area.

How the lesson surfaced — two prior levers, both blessed by the wrong
profile, both fell on their face:

- **M2 2×1 M-block reg-blocking grouped-GEMM** (commit **db672aa8**):
  byte-exact correct, **−4% prefill regression**. Inspired by
  glovepost/wmma_ops's 2×2 reg-block pattern (+56% on plain WMMA GEMM).
  The pattern doesn't transfer to mq4-on-the-fly-dequant grouped-GEMM
  where dequant cost serializes the K-substep loop. Kept opt-in via
  `HIPFIRE_MOE_GROUPED_M2=1`, default off.
- **hipGraph capture/replay for prefill MoE FFN** (commit **42ca533d**,
  capture-safe lm_head fix **d86fafa3**): captured 1423 kernarg blobs,
  replayed identically. **−1.6% prefill**. ROCm 7.2.2 hipGraph on
  gfx1201 does *not* amortize launch overhead the way the decode
  hipGraph rule predicts — because the 167ms gap was kernel time, not
  launch overhead. Kept opt-in via `HIPFIRE_GRAPH_PREFILL=1`, default
  off.

Both kept as research artifacts. Neither offender was visible until
rocprof was pointed at the same workload.

## The win

Once rocprof named the actual hotspot, the fix was small:

- **NEW kernel** `kernels/src/gemm_q8_0_wmma.gfx12.hip` — non-residual
  WMMA Q8_0 GEMM, mirrors `kernels/src/gemm_q8_0_residual_wmma.gfx12.hip`
  with `=` instead of `+=` on Y. Same 16×16 WMMA tile, K4 unroll,
  gfx12 wave32 intrinsic. ~140 LOC.
- **Dispatcher auto-route** at
  `crates/rdna-compute/src/dispatch.rs:13618` (`gemm_q8_0_batched_chunked`)
  — on gfx12 + `K % 32 == 0`, skip the `MAX_BATCH=64` chunked-scalar
  path and delegate straight to `gemm_q8_0_wmma`. Opt-out via
  `HIPFIRE_Q8_BATCHED_LEGACY=1`.
- **Call sites that benefit** are all in
  `crates/hipfire-arch-qwen35/src/qwen35.rs` prefill body: MoE router
  (≈4225), shared-expert gate (≈4237), LA QKVZA Q8 fallback (4705–4708),
  LA wo (4908), dense FFN gate/up/down (4973–5045). ~40 calls per
  prefill, 4 sub-chunks each, every one now a single WMMA launch.

Bench (R9700/gfx1201, `HIPFIRE_MOE_GROUPED_GEMM=1`,
`HIPFIRE_DPM_WARMUP_SECS=5`):

| seq | prefill before | prefill after |
| --- | -------------- | ------------- |
| 256 | 1017.6 tok/s   | **2966 tok/s** (+192%) |
| 512 | 1022 tok/s     | **3115 tok/s**         |
| 1024| 1023 tok/s     | **3147 tok/s**         |

Reference: hipEngine's published 2500 prefill / 111 decode at 512/128
on gfx1100 (W7900). We land **+24% over that** on a similar-class
config at gfx1201.

## The "unsafe" gate (production lever you must know)

The bench example uses `DeltaNetState::new`, which defaults to
`StateQuant::Q8` — fast but drifts after ~200 generated tokens on
MoE. The daemon hardcodes `StateQuant::FP32` for MoE models to keep
output coherent. That made the 2966 tok/s number invisible to
production callers.

Commit **da653e61** exposes `params.dn_quant: "q8" | "fp32"` so an
operator can flip it explicitly. The daemon default stays FP32 for
MoE. Short-output workloads and ablations that can accept the drift
risk now have a documented way to get the +192%. We do not silently
flip defaults; the choice is explicit at the request boundary.

## External references that informed the work

The user pointed at three repos as inspiration; documenting which
patterns transferred and which didn't:

- **https://github.com/shisa-ai/hipEngine**
  (`hipengine/kernels/hip_gfx1100/moe/group_scatter.hip`,
  `prefill.py:qwen35_moe_prefill_grouped_compact`). Their +172–213%
  prefill claim used a compact-WMMA-grouped-GEMM at 512–4K tokens.
  Their best published is 2500 prefill on gfx1100. **Pattern
  transferred** — grouped-WMMA + sentinel + sync-free path landed
  pre-session in Path 2 (**6a7e4936**, **7f8e3d6e**, **ca7eaad5**).
  Their stack does not seem to swap the Q8 fallback to WMMA; that's
  the lift on top.
- **https://github.com/glovepost/wmma_ops**
  (`wmma_kernels_optimized.hpp` 2×2 reg-block;
  `wmma_device_helpers.hpp` half8 loads, LDS pad +8, Hilbert tile
  rasterization). +56% on pure FP16 WMMA GEMM at 4K³. **Pattern that
  didn't transfer** — 2×1 M-direction reg-blocking on mq4
  on-the-fly-dequant grouped-GEMM regressed −4% (commit **db672aa8**).
  Dequant cost serializes the K-substep loop differently from pure
  FP16 WMMA. Full 2×2 with N-direction amortization might still pay
  off but needs scatter pad=32 — deferred.
- **https://github.com/lhl/fsr4-rdna3-optimization**
  (`OPTIMIZATION_RESULTS.md` O02–O20). Findings: LDS staging *lost*
  on Strix Halo iGPU, scalar I/O beat packed `v_pk_*` by 31.8%,
  compile-time unroll +12%, fused single-pass +40–60% vs split
  kernels. **Pattern that informed** — the fused-single-pass-wins
  finding reinforces the existing fusion strategy in hipfire
  (`scatter_fused`, `fused_qk_l2_norm`, etc.). The LDS-staging-lost
  finding is iGPU-specific (no L2 of consequence); doesn't apply to
  our dGPU path.

## What's next

Three parallel agents are porting the grouped-WMMA pattern to HFQ6/MQ6,
HFQ3/MQ3, and HFP4/MFP4 quants (branches `feat/hfq6-moe-grouped-wmma`,
`feat/hfq3-moe-grouped-wmma`, `feat/hfp4-moe-grouped-wmma`) to unblock
AWQ-style mixed-precision MoE prefill — e.g.
`/mnt/nas/kaden/hipfire/mi300x-v3/qwen3-35b-a3b.mq4-awq` whose experts
are ≈50/50 MQ4/MQ6. After integration plus an admit-predicate update
(`moe_ffn_all_mq4` → `moe_ffn_supported`), AWQ A3B should benefit too.
Bench numbers pending.

## Negative-results log

One line each, file paths included so future-you doesn't re-investigate:

- **M2 2×1 reg-block grouped-GEMM (gfx12)** —
  `kernels/src/gemm_hfq4g256_moe_grouped_wmma_m2.gfx12.hip` (commit
  **db672aa8**) — −4% prefill, byte-exact. Env-gate
  `HIPFIRE_MOE_GROUPED_M2=1`, default off. Do not retry without
  N-direction amortization and scatter pad=32.
- **hipGraph capture/replay for prefill MoE FFN** —
  `crates/hipfire-runtime/examples/bench_qwen35_mq4.rs`
  `HIPFIRE_GRAPH_PREFILL` branch (commit **42ca533d**; capture-safe
  lm_head fix **d86fafa3**) — −1.6% prefill. ROCm 7.2.2 hipGraph on
  gfx1201 doesn't amortize launch overhead the way decode hipGraph
  does. Do not retry on this ROCm/arch pair.
