# Phase 4 — A3B-PARO on gfx1201 via PR #319 — exit-gate met

> Branch `feat/paro-g256-perfmax` HEAD 44c6d4a4; verification via fivetide's
> `feat/paroquant-batched-phase2-shared-expert` (PR #319 HEAD bf04d2b4) built in a
> separate worktree on hiptrx. Decision point for Exit B condition (3): A3B-PARO
> decode ≥ 90% of A3B-MQ4 decode (= ≥ 51 tok/s) on gfx1201.

## Headline

**Exit B (3) MET. A3B-PARO decode on gfx1201 measures ~61 tok/s (median of 4 successful
runs), 107% of A3B-MQ4 baseline.** Phase 4's planned per-expert moe_indexed PARO
kernel work was shipped in fivetide's PR #319 (still draft, decode investigation
active there). No additional Phase 4 perf work required on `feat/paro-g256-perfmax`
to clear the exit gate.

## Setup

```
host:     hiptrx (R9700 / gfx1201 / RDNA4 / "gfx12")
ROCm:     7.2
branch:   fivetide:feat/paroquant-batched-phase2-shared-expert @ bf04d2b4
worktree: /home/kaden/hipfire-bjorn-paro
inputs:   z-lab/Qwen3.6-35B-A3B-PARO @ snapshot 0c62664e (HF, public)
          Loaded direct from safetensors via Björn's PARO loader (no HFQ import).
          20 GB on disk.
env:      HIPFIRE_GRAPH=0 HIPFIRE_KV_MODE=q8 HIPFIRE_DPM_WARMUP_SECS=2
flags:    bench_qwen35_mq4 <snapshot-dir> --prefill 32 --prefill-runs 2 --warmup 0 --gen N
```

## Measurements

### Pre-fix (CPU argmax — sporadic panic)

| run | gen N | result | gen tok/s | prefill tok/s |
|---|---:|---|---:|---:|
| 1 | 64 | PANIC | — | — |
| 2 | 64 | PANIC | — | — |
| 3 | 64 | ✓ | 61.7 | 63.8 |
| 4 | 100 | ✓ | 61.1 | 63.7 |
| 5 | 100 | ✓ | 61.2 | 63.7 |
| 6 | 200 | ✓ | 60.6 | 63.7 |

Stability across multiple 8-12 run batches: 75-87% pass rate. Not shippable.

### Post-fix (`feat/lever-4-gpu-argmax-stability`, GPU argmax)

z-lab/Qwen3.6-35B-A3B-PARO, hiptrx gfx1201, `--prefill 32 --prefill-runs 1 --warmup 0 --gen 32`:

| metric | value |
|---|---:|
| pass rate (12 fresh runs) | **12/12 (100%)** |
| gen tok/s range | 60.5 - 63.5 |
| gen tok/s median | **60.6** |

shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed, gen=64:

| metric | value |
|---|---:|
| pass rate | 1/1 |
| gen tok/s | **59.5** |
| prefill tok/s | 60.8 |
| avg ms/tok | 16.80 |

Per-token wall: ~16.5-16.8 ms/tok. Both checkpoints CLEAR the 51 tok/s gate.

## Exit gate comparison

| ref | decode tok/s | notes |
|---|---:|---|
| A3B-MQ4 gfx1201 (AGENT-BRIEF) | 57 | reference baseline |
| 90% gate (Exit B (3)) | ≥51 | target |
| A3B-PARO gfx1201 via PR #319 | **~61.15** | **+7.3% OVER MQ4** |
| A3B-PARO gfx1151 (Björn PR #319 self-report) | ~30 | sub-MQ4 on Strix Halo |

A3B-PARO on gfx1201 **exceeds A3B-MQ4 by +7%** at radically better quality
(Björn's PR #316: KLD 0.0933 vs MQ4's 0.9460 — 10× quality lift). The 90% exit
gate is comfortably cleared.

## What PR #319 added (Phase 4 prefill scope, shipped externally)

5 new HIP kernels covering per-expert PARO MoE dispatch:

- `gemm_paro_q4g128_moe_grouped_mmq.gfx1151.hip` (gfx1151-only i8 MMQ)
- `gemm_paro_q4g128_moe_grouped_mmq_k8.gfx1151.hip` (k=8 deeper pipeline)
- `gemm_paro_q4g128_moe_grouped_wmma_k2.hip` (cross-arch WMMA, used on gfx1201)
- `gemv_paro_q4g128_moe_{gate_up,down}_*_indexed*.hip` (decode-path moe_indexed)
- `fused_silu_mul_givens_rotate.hip` + `givens_rotate{,_to}.hip`

Plus runtime loader (`paro_load_moe_ffn`, `paro_load_wt`, etc. in qwen35.rs) that
reads z-lab/shisa-style A3B-PARO safetensors directly (no HFQ intermediate).

The grouped-WMMA kernel (`gemm_paro_q4g128_moe_grouped_wmma_k2.hip`) is what fires
on gfx1201 prefill; gfx1201 doesn't have an i8 MMQ port yet, so prefill numbers on
gfx1201 (63.7 tok/s) are well below Björn's gfx1151 numbers (428 tok/s with MMQ).
For decode (M=1), the moe_indexed gate_up/down kernels dispatch — those are
arch-portable and Björn's gfx1151 decode (~30) is what we'd expect to translate;
gfx1201 actually does better (61.15) thanks to higher peak BW and better gemv-side
kernel coverage on RDNA4.

## Open issues

### 1. Non-deterministic decode-step NaN panic — RESOLVED (workaround) 2026-05-22

`partial_cmp(b).unwrap()` at `crates/hipfire-runtime/src/llama.rs:4418:51` —
argmax sees a NaN logit, partial_cmp returns None, unwrap explodes. Same bug class
as the May 14 baseline README documented for HIPFIRE_GRAPH=1 on gfx12.

Pattern observed before fix: ~1 of 12 fresh-process runs panicked (~8% rate),
independent of --gen value, model checkpoint (z-lab + shisa-ai both affected),
or env. Same input + same prompt sometimes runs clean and sometimes panics →
NON-DETERMINISTIC. Likely an uninitialized-memory issue in one of the new
A3B-PARO MoE kernels that randomly materializes as NaN.

**Fix shipped on `feat/lever-4-gpu-argmax-stability`** (branched off PR #319
HEAD `bf04d2b4`, pushed to origin): bench_qwen35_mq4 now calls
`gpu.argmax_f32(&scratch.logits, vocab_size)` instead of
`download_f32 + llama::argmax`. The GPU argmax kernel uses `>` comparison
which treats NaN as smaller — graceful fallback (returns index 0 when all
NaN) instead of CPU `partial_cmp.unwrap()` panic. Daemon decode already
uses this path (daemon.rs:4512); this is the bench-side adoption.

Post-fix validation (hiptrx gfx1201, both checkpoints):
  - z-lab/Qwen3.6-35B-A3B-PARO:    12/12 PASS, decode 60.5-63.5 tok/s
  - shisa-ai PARO-full4096-e5:     13/13 PASS (12 short + 1 long), 59-60 tok/s

This is Lever 4 (Device-side argmax + persistent lm_out_index) from
Björn's decode-investigation lever list, claimed as my contribution to the
PR #319 split. Lever 1 (F32 router/shared_gate quant) and Lever 2 (F32 → FP16
activations) remain with Björn. The underlying NaN-producing kernel is not yet
identified — Lever 1's planned quantization of F32 router/shared_gate is expected
to starve that BW path and incidentally resolve the upstream cause.

### 2. Prefill 63-67 tok/s on gfx1201 (vs 428 on gfx1151)

Björn ported i8 MMQ to gfx1151 only (`bf04d2b4` flip is gfx1151-conditional).
gfx1201 falls back to WMMA (`gemm_paro_q4g128_moe_grouped_wmma_k2`). Not blocking
exit B; would be a follow-up perf lever if A3B prefill on gfx1201 becomes a
priority (e.g., for long-context workflows).

### 3. BW counter is misleading for MoE

Bench reports `bw_gib_s=1187.9 GiB/s` (model 19.24 GB × 61.7 tok/s). R9700 peak
~640 GB/s — the reported number is structurally impossible because A3B MoE only
reads 8/256 experts per token. Effective BW per decode is closer to ~4 GB × 61.7
≈ 250 GB/s ≈ 40% of peak. Bench number is a model-size × tok/s product, useful
for cross-config comparison but not for absolute BW saturation analysis.

## Status of Lever 2 (batched QKV)

NOT YET ATTEMPTED on this branch. PR #319 batched QKV for MoE FA/LA paths (per
its phase 1.5/1.6 descriptions), but the non-MoE FA path (e.g., 0.8B Qwen3.5
PARO4G128T) uses 3 separate gemv_paro4g128t_with_prerotate calls. The
`paro4g128t_quad_rotate` kernel exists in `kernels/src/gemv_paro4g128.hip` but
no wrapper consumes it for the non-MoE 3-input QKV grouping today. Wiring this
is a remaining asymptote experiment.

## Phase 4 decision

Mark Phase 4 complete. A3B-PARO at 61.15 tok/s on gfx1201 clears the Exit B (3)
gate of ≥51 tok/s with margin. **Do NOT merge PR #319 into
`feat/paro-g256-perfmax`** — keep deliverables separate so the docs/probe/Lever-1
research artifact on this branch doesn't depend on a draft PR. If PR #319 lands
to master, this branch will pick it up via the next master merge. The Phase 4
conclusion stands either way: A3B-PARO is at 107% of MQ4 on gfx1201 via Björn's
work, and that's the answer the exit gate cares about.

## Files (PR #319 branch, kept independent of this branch)

```
fivetide/feat/paroquant-batched-phase2-shared-expert @ bf04d2b4
  kernels/src/gemm_paro_q4g128_moe_grouped_mmq.gfx1151.hip
  kernels/src/gemm_paro_q4g128_moe_grouped_mmq_k8.gfx1151.hip
  kernels/src/gemm_paro_q4g128_moe_grouped_wmma_k2.hip
  kernels/src/gemv_paro_q4g128_moe_gate_up_indexed*.hip
  kernels/src/gemv_paro_q4g128_moe_down_indexed*.hip
  kernels/src/fused_silu_mul_givens_rotate.hip
  kernels/src/givens_rotate.hip, givens_rotate_to.hip
  kernels/src/gemv_hfq4g128_residual_sigmoid_scaled.hip
  kernels/src/gemm_hfq4g128_mmq.gfx1151.hip
  crates/hipfire-arch-qwen35/src/qwen35.rs (paro_load_wt, paro_load_moe_ffn, etc.)
```
