---
title: "Reproducible MoE down kernel — int64 accumulation complete"
tags: [device-mesh, moe, tp, ep, reproducibility, int64, minimax, deepseek4]
date: 2026-07-09
commit: 08097913
branch: feature/device-mesh
---

# Reproducible MoE down kernel — int64 accumulation complete

## Summary

MoE down projection now accumulates in fixed-point int64 (E=24, S=2^24)
instead of FP32, making the result partition-invariant under any K-split
where groups (256 elements) are wholly on one rank. This enables TP-of-experts
to be bit-exact across any tensor-parallel split of expert weights.

**Merged across 5 tasks, commits `bc44b748..08097913` on `feature/device-mesh`.**

## Fixed-point scheme (Task 1)

- **E = 24**, S = 2^24 = 16777216
- Per-group approach: each 256-element codebook group's FP warp-reduced partial
  is multiplied by `topk_w * S` and rounded to int64 **before** cross-group
  accumulation. This is the partition-invariance key: since each group lives
  wholly on one rank, `round(w*partial_g*S)` is computed the same way
  regardless of which rank processes it, and integer addition is commutative
  and associative.
- `atomicAdd(residual_i64, acc_i64)` uses unsigned long long arithmetic, which
  is bitwise equivalent to signed two's-complement addition — order-independent.
- Final conversion via `moe_i64_residual_to_f32`: `(float)((double)v / (double)S)`.
- **Why per-group, not per-expert:** folding `topk_w` after summing all groups
  (`round(topk_w * total_acc_i64)`) is NOT partition-invariant because
  `round(w*(a+b)) != round(w*a) + round(w*b)`.

## Kernels changed

### MQ3G256Lloyd (MiniMax M2.7 down projection format)

File: `kernels/src/gemv_mq3g256_lloyd_moe_down_indexed.hip`

- `gemv_mq3g256_lloyd_moe_down_residual_i64_k8_indexed` — single-token residual
  indexed (atomic into `residual_i64[M]`). Used by TP forward.
- `moe_i64_residual_to_f32` — element-wise int64→f32 converter, shared by all
  MQ3/MQ2 residual paths.

### MQ2G256Lloyd (DeepSeek V4 Flash down projection format)

Files:
- `kernels/src/gemv_mq2g256_lloyd_moe_down_indexed.hip` →
  `gemv_mq2g256_lloyd_moe_down_residual_i64_k8_indexed`
- `kernels/src/gemv_mq2g256_lloyd_moe_down_indexed_batched_k4.hip` →
  `gemv_mq2g256_lloyd_moe_down_residual_i64_k8_indexed_batched_k4`
- `kernels/src/gemv_mq2g256_lloyd_moe_down_expanded_k4.hip` →
  `gemv_mq2g256_lloyd_moe_down_expanded_i64_k4`

Helper kernel:
- `kernels/src/add_inplace.hip` → `add_inplace_i64` (used by
  `all_reduce_sum_i64_peer` to sum int64 buffers across emulated/real ranks).

## int64 TP collective (Task 3 — spec §8)

`crates/rdna-compute/src/norm.rs`: `all_reduce_sum_i64_peer`

Mirrors `all_reduce_sum_f32_peer` but operates on raw int64 buffers:
- D2H copy each rank's `residual_i64` partial to a pinned staging buffer
- H2D copy peer rank's partial onto the peer GPU
- `add_inplace_i64` kernel to add in-place
- No H2D conversion needed before MoE combine (stays int64 until `ConvertI64ToF32` step)

**Collective scheme:** Only used on the TP path (TP-of-experts, `forward_tp`).
The EP path uses `ZeroI64Only{dim}` → `DownResidualI64` → `ConvertI64ToF32` →
`AllReduce{Ep}` (FP32 collective, matching the existing EP allreduce).

## Parity results

### tp_minimax (Task 3) — argmax-exact TP-of-experts

`HIPFIRE_DETERMINISTIC=1 HIPFIRE_EMULATE_GPUS=2 ./target/release/examples/tp_minimax \
  --model ~/.hipfire/models/MiniMax-M2.7.mq2 --max 128`

Result (Task 3 commit `24d0f310`):
- argmax-exact: **true** across all 128 generated tokens
- logit max|Δ|: **0.00e0**
- tp=1 output == tp=2 output (byte-identical generation)

NOTE: MiniMax-M2.7.mq2 uses MQ2G256Lloyd for its gate/up projections and MQ3G256Lloyd
for its down projection. The i64 path triggers only for `MQ3G256Lloyd` on the TP path
(see `forward.rs:1298`: `let use_i64 = use_i64_down && matches!(ddt, DType::MQ3G256Lloyd)`).

### Kernel parity tests (Tasks 1+2+4)

`cargo test -p rdna-compute test_moe_down_repro_parity -- --nocapture`

Result: **3/3 PASS** (bit-exact K-split partition invariance)
- `test_moe_down_repro_parity_k_split_bit_exact` (MQ3L residual)
- `test_moe_down_repro_parity_mq2l_residual_k_split_bit_exact` (MQ2L residual)
- `test_moe_down_repro_parity_mq2l_expanded_k_split_bit_exact` (MQ2L expanded)

All verify: `full_i64[row] == half_a_i64[row] + half_b_i64[row]` for all rows (bit-exact).

## EP unification (Task 5 — Option A)

All four live EP sites in `daemon.rs` and the two EP example harnesses updated to use
the i64 path for the MoE down projection:

- `ep_minimax.rs`: `MINIMAX_EP2_FNV = 0x887c2e7717e9c3bf` — **unchanged** (live-confirmed)
- `ep_deepseek4.rs`: `DS4_EP2_FNV = 0x6c0f2f000f1d398f` — **unchanged** (live-confirmed)

FNVs held because: the i64 per-group rounding to the 2^-24 grid produces f32 values
that differ from the FP32 path by ~1e-7 relative error (not byte-identical); however,
these perturbations did not flip any argmax over the tested 32-token generation on the
test prompt. The EP-i64 path is argmax-stable on the test prompt, not byte-identical.

## Occupancy (gfx1151 — RDNA3 / Strix, wave32, 1024 VGPRs/SIMD)

| Kernel function | VGPR | SGPR | spill (priv bytes) | LDS | waves/SIMD |
|---|---:|---:|---:|---:|---:|
| `gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed` (FP, baseline) | 74 | 21 | 0 | 128 | 13 |
| `gemv_mq3g256_lloyd_moe_down_residual_i64_k8_indexed` (new i64) | **29** | 24 | **0** | 32 | **16** |
| `gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed` (FP, baseline) | 74 | 21 | 0 | 64 | 13 |
| `gemv_mq2g256_lloyd_moe_down_residual_i64_k8_indexed` (new i64) | **29** | 24 | **0** | 16 | **16** |
| `moe_i64_residual_to_f32` | 7 | 8 | 0 | 0 | 16 |
| `add_inplace_i64` | 6 | 7 | 0 | 0 | 16 |

No register spills on any new kernel (`private_segment_fixed_size = 0`).
The i64 kernels use FEWER VGPRs (29 vs 74) than the FP baseline because the
int64 accumulator (`long long acc_i64 = 0LL`) is scalar and lives in SGPRs/LDS;
the per-lane FP shuffle tree is eliminated by using `__shfl_down` for the group
partial and then converting only on lane 0.

## Perf (decode tok/s)

Model: MiniMax-M2.7.mq2 (single-GPU infer_minimax, no TP active)
Prompt: "What is the capital of France? Name the river that runs through it and
         one famous landmark there, then briefly explain why the city is historically
         important." md5: `1d32df5f12c414d3e34c7b35b6611e6c`
Tokens decoded: 64 per run, fresh process, DPM-warmed (HIPFIRE_DPM_WARMUP_SECS=10 first run)

| Run | Baseline 7421b998 (FP32) | HEAD 08097913 (i64) |
|-----|-------------------------:|--------------------:|
| warm 1 | 28.3 tok/s | 28.1 tok/s |
| warm 2 | 28.4 tok/s | 28.1 tok/s |
| median (warm) | **28.35 tok/s** | **28.1 tok/s** |

**Delta: −0.9%** — within ±3% noise band, no investigation required.

The single-GPU infer_minimax path does NOT use the i64 kernel (which only fires
on `forward_tp` when `use_i64_down=true`). The perf measurement therefore
validates that the kernel source changes (adding new i64 variant functions to
the same HIP files) did not regress the existing FP dispatch path.

## Coherence gate

`./scripts/coherence-gate.sh` at HEAD `08097913`:
- Battery: **11/11 OK, 0 hard errors** (all models fluent, on-topic, no verbatim loops)
- pflash-stage: FAIL — known gfx1100-baseline vs gfx1151 artifact (uniform ~2× slowdown
  on all rows including untouched AR baseline rows; pre-existing, unrelated to this change)

Report: `/tmp/coherence-20260709-112018.md`

## Status and next steps

The int64 reproducible MoE down kernel is complete and sign-off is done:
- **minimax TP-of-experts** already lands on true parity (argmax-exact, max|Δ|=0)
- **deepseek4 TP** (`tp_deepseek4`) is the next planned step — same scheme,
  MQ2L format, i64 kernels already available (`gemv_mq2g256_lloyd_moe_down_residual_i64*`)
- **D2b (TP-of-experts) can now resume** on true bit-parity; no more structural
  numerical divergence from FP32 non-associativity under K-split

Links:
- [[pd-decompose-d2a-decode-complete]] — parent decompose task
- Sketch/plan: `.superpowers/sdd/task-1-brief.md` through `task-6-brief.md`
- Full task reports: `.superpowers/sdd/task-{1..6}-report.md`
