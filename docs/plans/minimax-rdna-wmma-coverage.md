# MiniMax-M2.7 — RDNA3+ WMMA kernel coverage & the MQ3-Lloyd gap

**Status (2026-05-30):** MiniMax-M2.7 is *correctness-portable* to RDNA3+ today,
but GEMV-bound in prefill. Fast RDNA prefill needs two things: (1) a batched
forward, and (2) grouped-WMMA MoE GEMM. There is exactly **one** new-kernel gap
(MQ3-Lloyd grouped). RDNA e2e validation is partly reachable today — hipx
(Strix Halo gfx1151, 96 GB) fits the mq2 / mq2-lloyd tiers — but the larger
tiers (mq3-lloyd, mq4) exceed it; that slice is **deferred**. This doc flags both.

## Why this matters

`generate_minimax` uses **per-token GEMV for everything** (q/k/v/o via
`weight_gemv`, MoE via the A3B/deepseek4 `gemv_*_indexed` decode kernels,
generic rmsnorm/rope/qk-norm, `attention_q8_0_kv`). MiniMax is batch-1 even in
prefill. That path runs correctly on **any** arch — kernels are JIT-compiled per
GPU (`ensure_kernel`/hiprtc), no MFMA, no WMMA, no arch gate — but it never
batches, so prefill is GEMV-bound. Fast RDNA prefill = a **batched forward**
(the DFlash batched-verify keystone, see `docs/plans/dflash-trainer.md`) feeding
**grouped-WMMA MoE GEMM**.

## Coverage map (what already exists on gfx11/gfx12)

| forward path | dtype | grouped/batched WMMA kernel | status |
|---|---|---|---|
| dense q/k/v/o, router, lm_head | Q8 | `gemm_gate_up_q8_0_wmma` + Q8 batched + `attention_q8_0_kv_batched` | ✅ exists |
| MoE gate/up + down | MQ4 (HFQ4) | `gemm_hfq4g256_moe_grouped_mmq.{gfx1151,gfx11_dgpu,gfx12}` | ✅ exists |
| MoE gate/up + down | MQ2-Lloyd | `gemm_mq2g256_lloyd_moe_grouped_wmma_{k2,4w_k2,4w_k2_cnd,4w_k2_n32,8w_k2}` | ✅ exists (5 variants, wired) |
| MoE gate/up + down | **MQ3-Lloyd** | — only per-token `gemv_mq3g256_lloyd_moe_{gate_up,down}_indexed` | ❌ **GAP** |

So once the batched forward exists, the **mq4** and **mq2-lloyd** tiers get fast
RDNA prefill with **zero new kernels** — just route their batched MoE through the
existing grouped kernels. (Earlier in this thread I mis-scoped this as "the Lloyd
formats are the gap" — wrong; MQ2-Lloyd grouped already exists. Caught by the
grep-existing-variants-first discipline.)

## The one gap: MQ3-Lloyd grouped-WMMA MoE GEMM

Tiers that need it: **mq3** (uniform MQ3-Lloyd), **mq3-lloyd** (gate/up
MQ3-Lloyd), **mq2** (down MQ3-Lloyd). One kernel covers all three.

It is a **graft of two existing kernels**, not from-scratch:
- grouped scatter / per-expert tile structure: `gemm_mq2g256_lloyd_moe_grouped_wmma_4w_k2.hip`
- 3-bit / 8-entry codebook dequant WMMA inner loop: `gemm_gate_up_mq3g256_lloyd_wmma.gfx12.hip`

## Probe-based dev loop (fast inner loop, slow gate)

The repo already has the tiny-probe harness convention; use it:

1. **Occupancy** — `gfx-kernel-metadata` skill: compile for gfx1151/gfx12, read
   VGPR/SGPR/LDS/spills from the `.hsaco`. **No GPU run** — doable even on the
   mi300/CDNA box via cross-compile. Catches register blowup instantly.
2. **Correctness** — fork `test_gemm_fused_mq3g256_lloyd_wmma.rs` (CPU **fp64**
   reference, f16-roundtripped inputs, max-abs/rel error). Exact, seconds. A bug
   reads as `0.4` rel-error, not as subtle token drift hidden in quant noise.
3. **Perf** — fork `bench_mq2g256_lloyd_moe_4w.rs`: isolated µs A/B vs the MQ3
   per-token indexed baseline (WARMUP/TRIALS already wired).
4. **GATE (survivors only)** — e2e fresh-process `scripts/probe_commits.sh` +
   `coherence-gate.sh`, on RDNA hardware.

**Hard rule:** a microbench win is a *hypothesis, not a result.* This codebase's
memory has multiple WMMA-Lloyd/MMQ kernels that won the microbench then failed
e2e or coherence (synth-win → prod-falsify: `fp8_wmma_hfp4g32`, `sgpr_lut`,
`dot2_trickle_down`, the i8-MMQ-MoE-grouped family). The probes *kill bad
variants fast and find geometry*; the **gate** is always e2e + coherence.

## Prerequisite

The **batched MiniMax forward** (DFlash batched-verify keystone). Nothing batched
dispatches today, so a WMMA kernel written before it has no caller. Build the
batched forward first — it independently fixes slow prefill — then wire the
existing grouped kernels (mq4, mq2-lloyd) and finally add the MQ3-Lloyd grouped.

## Validation reality (the flagged gap)

WMMA **executes only on RDNA3/4** — mi300 (CDNA/gfx942) cannot run it. RDNA
validation target = Strix Halo via `ssh hipx`.

**hipx specifics (verified 2026-05-30 from kfd topology):**
- The **8060S iGPU = gfx1151** (WMMA-capable, wave32) is **HIP device 1**
  (`HIP_VISIBLE_DEVICES=1` / `ROCR_VISIBLE_DEVICES=1`), with a **96 GB dedicated
  VRAM carveout** (kfd node 2) plus GTT spill → **~103 GB addressable** (a real
  localmaxxing run hit 103 GB there). Device 0 is an RX 5700 XT (gfx1010, RDNA1,
  7 GB, **no WMMA**). NB: `free -g` shows ~30 GB — that is the *host/CPU*
  partition (kfd node 0), **not** the iGPU pool; don't size MiniMax against it.
- **Tier fit on device 1 (~96 GB carveout, ~103 GB with GTT):**
  | tier | size | fits hipx? |
  |---|---|---|
  | mq2 | 79 GB | ✅ comfortable |
  | mq2-lloyd | 86 GB | ✅ comfortable |
  | mq3 | 102 GB | ⚠️ GTT-edge, tight with KV/scratch |
  | mq3-lloyd | 109 GB | ❌ exceeds |
  | mq4 | 124 GB | ❌ exceeds |
- **What hipx CAN do:** full e2e RDNA coherence + perf for **mq2 + mq2-lloyd**.
  Since mq2's **down** projection is MQ3-Lloyd, the mq2 tier exercises the
  **MQ3-Lloyd grouped *down* path e2e** — so the new kernel's down side gets a
  real end-to-end RDNA gate, not just a microbench.
- **The deferred slice:** the **MQ3-Lloyd grouped *gate/up* path** lives in the
  mq3 / mq3-lloyd tiers (102–109 GB), which don't fit (mq3 borderline). On hipx
  it is **microbench-only** (steps 1–3 above use tiny synthetic E=1 tensors —
  VRAM is irrelevant). Full gate/up e2e + the mq4 tier need a larger RDNA VRAM
  pool or multi-GPU PP — no such RDNA config in the fleet today. Flagged.

## Bottom line

- New-kernel surface = **1** (MQ3-Lloyd grouped-WMMA MoE), graftable from 2
  existing kernels.
- **hipx (gfx1151 / device 1, 96 GB) validates a lot:** full e2e for mq2 +
  mq2-lloyd, and mq2 covers the MQ3-Lloyd grouped *down* path e2e; the new
  kernel's *gate/up* path is microbench-validatable there always.
- **Only the MQ3-Lloyd gate/up *full-e2e* (mq3 / mq3-lloyd tiers) and the mq4
  tier are blocked** on RDNA memory capacity (>96 GB) — flagged, deferred to a
  larger RDNA pool / multi-GPU PP.
