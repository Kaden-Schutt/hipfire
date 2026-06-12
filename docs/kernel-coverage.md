# Kernel Coverage by Architecture

Tracks which GPU architectures have arch-specific kernel variants, and how
mature those variants are. Use this to identify gaps before writing new
arch-specific kernels or prioritising optimisation work.

Kernels live in `kernels/src/<arch>/`. Generic fallbacks (F32-only, no
arch-specific tuning) remain in `kernels/src/`.

---

## Architecture Summary

| Arch | Hardware | Wave | Native dtype | Memory | Notes |
|------|----------|------|-------------|--------|-------|
| gfx906 | CDNA1 / GCN5 (MI50, MI60) | 64 | F16, F32 | ~1 TB/s HBM2 | DP4A; no WMMA; no BF16 |
| gfx942 | CDNA3 (MI300X, MI300A) | 64 | BF16, F16, F32 | ~5.3 TB/s HBM3 | MFMA; BF16 accumulation |
| gfx1030 | RDNA2 dGPU (RX 6000) | 32 | F16, F32 | ~512 GB/s GDDR6 | Limited WMMA; no BF16 |
| gfx1100 | RDNA3 dGPU (RX 7000) | 32 | BF16, F16, F32 | ~432–576 GB/s GDDR6 | WMMA; native BF16 |
| gfx1151 | RDNA3.5 APU / Strix Halo | 32 | BF16, F16, F32 | ~89 GB/s LPDDR5X (UMA) | No dedicated VRAM; CPU shares bandwidth |
| gfx1200 | RDNA4 (RX 9070 XT) | 32 | BF16, F16, FP8, F32 | ~576 GB/s GDDR6 | WMMA; native FP8 |
| gfx1201 | RDNA4 (RX 9070) | 32 | BF16, F16, FP8, F32 | ~576 GB/s GDDR6 | WMMA; native FP8 |

gfx1200 and gfx1201 share the `gfx12` family dir for common kernels.
gfx1100 and gfx1151 share the `gfx11` family dir for cross-RDNA3 kernels.

---

## Maturity Tiers

| Tier | Meaning |
|------|---------|
| **—** | No arch-specific variant. Falls back to generic F32. May widen weights at load time. |
| **basic** | Single arch-specific implementation. Correct ISA usage; minimal tile/occupancy tuning. |
| **standard** | Multiple variants or format coverage; correct for the arch's memory hierarchy. |
| **tuned** | Systematic tile-shape benchmarking (x8/x16/.../x64 or v1–v5 suffixes); near bandwidth-optimal. |

---

## Generic Op Coverage

These kernels are shared by all model families.

### GEMV — decode weight projection (memory-bandwidth-bound)

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| gemv hfq4g256 | — | standard | **tuned** (v1–v5) | standard | standard | — |
| gemv hfq4g256 multirow | — | — | — | standard | standard (opt-in) | — |
| gemv hfq3g256 | — | — | — | standard | — | — |
| gemv mq4g256 lloyd | — | — | — | **tuned** (multiacc_diag) | — | — |
| gemv mq3g256 lloyd | — | — | — | standard | — | — |
| gemv hfp4g32 | — | — | — | basic | — | basic (fp8 path) |
| gemv hfq6g256 | — | — | — | — | — | basic |

gfx1151 now routes HFQ4/MQ4 decode GEMV through the RDNA3 single-row and multi-row sources rather than the generic fallback. A 2026-06-12 rows sweep on Qwen3.5-4B, Qwen3.5-9B, and Qwen3.6-35B-A3B found the single-row path fastest for decode (`4B gen50: R1 66.8 tok/s, R2 66.2, R4 66.1, R8 65.8`; `9B gen50: R1 44.2, R2 44.0, R4 42.6, R8 42.9`; `A3B gen20: R1 80.6, R2 80.3, R4 79.5, R8 79.9`), so gfx115x defaults back to R=1 while `HIPFIRE_GEMV_ROWS=2/4/8` remains available for larger-shape experiments.

### GEMM — prefill projection (compute-bound)

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| gemm residual hfq4g256 | **tuned** (x8–x64) | **tuned** (mfma v1–v4) | **tuned** (mmq x8–x32) | — | standard (mmq x16/x32_y64) | **tuned** (wmma) |
| gemm residual hfq3g256 | — | — | **tuned** (mmq x8–x32) | — | — | standard (wmma) |
| gemm residual hfq6g256 | — | — | — | — | — | standard (wmma) |
| gemm residual mq4g256 | — | — | — | — | standard (wmma) | standard (wmma) |
| gemm residual mq3g256 | — | — | — | — | — | standard (wmma) |
| gemm residual hfp4g32 | — | — | — | — | — | standard (wmma) |
| gemm residual q8_0 | — | — | — | — | standard (wmma 4w) | standard (wmma) |
| gemm gate_up hfq4g256 | **tuned** (x8–x64) | standard | **tuned** (mmq x8–x32) | — | standard (wmma) | **tuned** (wmma) |
| gemm gate_up hfq3g256 | — | — | **tuned** (mmq+dp4a) | — | — | standard (wmma) |
| gemm gate_up mq4g256 | — | — | — | standard | **tuned** (wmma, mb4) | **tuned** (wmma) |
| gemm gate_up mq3g256 | — | — | — | standard | — | standard (wmma) |
| gemm gate_up hfp4g32 | — | — | — | — | — | standard (wmma) |
| gemm gate_up q8_0 | — | — | — | — | standard (wmma 4w) | standard (wmma) |
| gemm qkv hfq4g256 | **tuned** (x8–x64) | standard | **tuned** (mmq x8–x32) | — | standard (wmma) | **tuned** (wmma) |
| gemm qkv hfq3g256 | — | — | **tuned** (mmq+dp4a) | — | — | standard (wmma) |
| gemm qkv mq4g256 | — | — | — | — | standard (wmma, mb4) | **tuned** (wmma) |
| gemm qkv mq3g256 | — | — | — | — | — | standard (wmma) |
| gemm qkv hfp4g32 | — | — | — | — | — | standard (wmma+fp8) |
| gemm lmhead hfq4g256 | — | — | — | — | — | basic |
| gemm bf16 | — | standard (mfma) | — | — | standard (wmma) | — |

gfx1151 HFQ4-G256 one-wave i8 WMMA now auto-routes only the aligned
K=2048 projection shapes used by Qwen3.5/3.6 A3B/shared paths. On
Qwen3.6-35B-A3B MQ4 pp128 profiling, this reduced `gemm_hfq4g256_mmq_set`
from 31.7 ms to 20.0 ms and total profiled prefill from 108.7 ms to
96.0 ms. Larger dense Qwen3.5-9B K=4096 pp256 still regresses badly
(`set` 164.0 ms vs 134.3 ms, `add` 88.4 ms vs 61.9 ms), so those shapes
remain on the existing MMQ x16/x32_y64 path. `HIPFIRE_HFQ4G256_MMQ_GFX1151=0`
forces the fallback; `=1` forces the probe for aligned experiments.

gfx1151 IU4 WMMA is available and now has a signed-Q4 tile channel test
(`test_gfx1151_s4_wmma_tile`) that validates packed S4xS4 -> I32 accumulator
layout against a CPU dot reference. A follow-on HFQ4-G256 correction-path
probe (`test_gfx1151_hfq4_s4_mmq`) validates existing HFQ4 blocks plus
signed-Q4 activation scratch against a CPU affine reference. Neither path is
routed into Qwen yet; the next production step is runtime symmetric-Q4
activation quantization plus quality/perf evaluation against the Q8_1 MMQ
path.

### MoE grouped GEMM

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| moe grouped hfq4g256 | — | — | — | basic (gfx11_dgpu, 2 variants) | standard (mmq k4/k8) | **tuned** (wmma+mmq, m2 variant) |
| moe grouped hfq3g256 | — | — | — | — | basic | standard |
| moe grouped hfq6g256 | — | — | — | — | **tuned** (4w default, v2 opt-in) | standard (v2) |
| moe grouped mq4g256 | — | — | — | — | — | — |
| moe grouped paro q4g128 | — | — | — | — | standard (k8 variant) | — |
| moe grouped bf16/f16 | — | — | — | — | basic | — |
| moe scalar path | — | — | — | — | basic | — |
| moe hfp4g32 | — | — | — | — | — | standard |

gfx1151 HFQ4 grouped MMQ: the `HIPFIRE_MOE_GROUPED_I8_4W=1` experiment stages
routed Q8_1 blocks in LDS across four row-warps and is bit-identical to k8, but
regressed the default k8 path on A3B pp256, 122B pp128, and 122B pp64. The
122B pp64 sweep measured default k8 at 71.2 ms inside a 294.2 ms profiled
prefill, versus 85.8 ms for k4 and 78.7 ms for k8-4w. A fresh
Qwen3.6-35B-A3B pp128 probe also kept k8 ahead (`44.0 ms` grouped total,
`663.9 tok/s`) versus k8-4w (`50.1 ms`, `642.6 tok/s`). Keep k8 as the
default. See
`docs/perf-checkpoints/2026-06-12-gfx1151-qwen35-profile-refresh.md`.

gfx1151 HFQ6 grouped MoE: the 4-warp WMMA path is default-on and remains the
best measured 122B route. On Qwen3.5-122B-A10B MQ4 pp64, HFQ6 grouped MoE was
86.8 ms with 4w enabled, versus 144.1 ms for v1 and 177.3 ms for v2 when 4w was
disabled. See `docs/perf-checkpoints/2026-06-12-gfx1151-moe-variant-sweep.md`
for the full variant table.

gfx1151 BF16/F16 routed MoE: native grouped WMMA is the default. A compact
indexed gate/up kernel has channel-test coverage but is not routed: it avoids
grouped padding but loses on Qwen3.6-35B-A3B BF16 pp128 (`904.0ms` grouped
gate/up vs `938.9ms` indexed gate/up across 40 calls).

### Fused QKV (prefill, quant decode into Q/K/V in one pass)

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| fused qkv hfq4g256 | — | standard (v2) | — | — | standard (wmma) | **tuned** |
| fused qkv mq4g256 | — | — | — | standard | standard (wmma, mb4) | **tuned** |
| fused qkv mq3g256 | — | — | — | standard | — | standard |
| fused qkv hfp4g32 | — | — | — | — | — | standard |
| fused qkv q8_0 | — | — | — | — | standard (wmma 4w) | standard |

gfx1151 Q8_0 fused prefill uses the four-warp 64x64 WMMA family for large
Qwen3.5/3.6 Q8 shapes. The conservative default remains `B>=128` for broad
Q8_0 fused projections, but Qwen3.5-122B-A10B `B=64` now auto-routes the
large `K=3072` QKVZA/QKV projections and `M=3072,K>=8192` residuals through
the gfx1151 4w kernels. On Qwen3.5-122B-A10B MQ4 pp64 profiling this moved
total profiled prefill from 441.3 ms to 294.9 ms; the main Q8 buckets moved
`gemm_qkvza_q8_0_wmma` 173.8 ms to `gemm_qkvza_q8_0_wmma_4w_gfx1151`
47.9 ms, `gemm_qkv_q8_0_wmma` 33.5 ms to
`gemm_qkv_q8_0_wmma_4w_gfx1151` 16.3 ms, and
`gemm_q8_0_residual_wmma` 25.4 ms to
`gemm_q8_0_residual_wmma_4w_gfx1151` 16.3 ms. The per-kernel env vars
`HIPFIRE_Q8_QKVZA_4W`, `HIPFIRE_Q8_QKV_4W`,
`HIPFIRE_Q8_RESIDUAL_4W`, and `HIPFIRE_Q8_GATE_UP_4W` are tri-state:
unset uses the shape gate, `0` forces the single-wave path, and `1` forces
the gfx1151 4w path for aligned `B%64==0` experiments.

### Attention

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| dflash wmma v3 causal f16kv | — | — | — | — | — | standard |
| flash q8_0 dp4a | basic (2 variants) | — | — | — | — | — |

### Norm and Rotate

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| rmsnorm | — | standard (reduce) | — | — | basic (wave-reduced) | — |
| rotate_with_rms (fused) | — | standard | — | — | **—** ⚠️ | — |
| fused_rmsnorm_mq_rotate | generic only | generic only | generic only | generic only | basic (wave-reduced RMS) | generic only |

All norm kernels are F32-only. gfx1151 now has wave-reduced `rmsnorm` and a
fused RMSNorm+MQ-rotate variant that keeps the generic FWHT phase but replaces
the 256-float LDS reduction ladder with wave reductions plus 8 wave sums. On
Qwen3.6-35B-A3B MQ4 pp256, fused RMSNorm+MQ rotate moved from 867.3us to
821.4us across 40 calls. On Qwen3.5-9B MQ4 pp256, plain batched RMSNorm moved
from 444.9us to 356.4us across 16 calls; standalone CPU-reference correctness
stays within 3.10e-6 max_abs at batch=3, K=12288. A 2026-06-12 profile refresh
put the remaining norm/scalar rows below the matrix kernels: on Qwen3.5-9B MQ4
decode, `gated_norm_f32` was 1.2% and the gfx1151 fused gate+conv row was 1.2%;
on Qwen3.6-35B-A3B MQ4 decode, they were 1.7% each. The remaining norm/rotate
gap is therefore correctness/precision-policy cleanup rather than the next
first-order gfx1151 performance lever.

### Misc compute

| Kernel | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| mq_rotate_x_dual | — | — | — | — | — | basic |
| pack_f32_to_fp8 | — | — | — | — | — | basic |
| gated_delta_net_q8 | — | — | — | — | basic (register-state probe, opt-in) | basic (gfx1200 only) |

gfx1151 GDN Q8 register-state probe: `HIPFIRE_GDN_Q8_REG_GFX1151=1` routes
the recurrent state through one 128-thread block per head with one S row per
thread. It is default-off after Qwen3.6-35B-A3B MQ4 pp256 profiling regressed
GDN time from 11.8 ms on the generic LDS-backed path to 168.5 ms on the
register-state path.

---

## Model-Specific Kernel Coverage

Model-specific kernel families have **no arch-specific variants on any arch** — all
fall back to generic F32.

### Qwen3.5 / DeltaNet (hipfire-arch-qwen35)

Unique kernels: `conv1d_*`, `fused_sigmoid_alpha_gate`, `alpha_gate`

| Kernel | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| conv1d_decode | — | — | — | — | **—** ⚠️ | — |
| conv1d_gated_decode | — | — | — | — | **—** ⚠️ | — |
| conv1d_silu / split / routed / tree | — | — | — | — | basic (`split` prefill, `tree`, decode fusion) | — |
| fused_sigmoid_alpha_gate | — | — | — | — | basic (decode fusion) | — |
| alpha_gate | — | — | — | — | — | — |

gfx1151 is the primary deployment target for Qwen3.5. The linear
`conv1d_silu_split_f32_n` large-prefill path now has a gfx1151 parallel-token
variant plus a state-commit kernel (admitted at `n_tokens >= 64` after pp32
speed-gate rejected the extra launch). On Qwen3.5-9B MQ4 pp256, the profiled
conv compute row moved from 3274.2us to 1494.3us across 24 calls, plus
164.7us for final state commits. Decode now has a gfx1151 single-token fusion
that combines `fused_sigmoid_alpha_gate_f32` with `conv1d_silu_split_f32`,
removing one launch per linear-attention decode layer while preserving the
generic math. On Qwen3.6-35B-A3B MQ4, an 8-token profiled decode changed from
6424 launches / 48.6 tok/s to 6184 launches / 49.5 tok/s. Routed-session conv
still uses the generic F32 kernel because it mutates per-session state
sequentially.

gfx1151 tree conv now has a token-parallel variant for DFlash/tree verify. The
generic kernel owns one channel per thread and loops over tree tokens; the
gfx1151 kernel maps one block row per token because tree conv leaves state
unchanged. On a synthetic DeltaNet tree-conv microbench (`k_dim=128`,
`v_dim=256`, 1000 trials), `n_tokens=64` moved from 15.760us generic to
2.729us gfx1151, with smaller trees also winning (`n=4`: 3.112us to 2.261us).
`HIPFIRE_CONV1D_TREE_GFX1151=0` forces the generic route for A/B checks.

### DeepSeek V4 Flash + mHC (hipfire-arch-deepseek4)

Unique kernels: `deepseek4_attn_swa_*`, `deepseek4_moe_topk_*`, `hc_*`

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| deepseek4_attn_swa (SWA) | — | — | — | — | — | — |
| deepseek4_moe_topk_bias_aware | — | — | — | — | — | — |
| deepseek4_fused_silu_mul_clamp_mq_rotate | — | — | — | — | — | — |
| hc_sinkhorn_4x4 (mHC manifold) | — | — | — | — | — | — |
| hc_mix_4stream | — | — | — | — | — | — |
| hc_compute_control | — | — | — | — | — | — |
| hc_apply_alpha | — | — | — | — | — | — |

hc_sinkhorn_4x4 runs every layer, every token, in F32. On gfx942 (primary large-model target) this is a significant MFMA opportunity.

### MiniMax / Plaid Flash (hipfire-arch-minimax)

Unique kernels: `hash_router_*`, `indexer_*`, `pflash_score_q8_kv`, `triattn_*`

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| hash_router_normalize | — | — | — | — | — | — |
| indexer_relu_score | — | — | — | — | — | — |
| pflash_score_q8_kv | — | — | — | — | — | — |
| triattn_score_* | — | — | — | — | — | — |
| triattn_accumulate | — | — | — | — | — | — |

### Qwen3.5-VL / Vision (hipfire-arch-qwen35-vl)

Unique kernels: `vit_attention*`, `apply_rope_2d_vision`, `rope_2d_*`

All generic on all arches.

### MTP / Spec-decode (hipfire-arch-qwen35, compressor path)

Unique kernels: `compressor_*`, `greedy_accept`, `kld_tile_topk_lse`

All generic on all arches.

---

## Priority Gaps

Ordered by estimated decode-path impact on active hardware.

| Priority | Gap | Arch | Reason |
|---|---|---|---|
| 1 | HFQ4/MQ4 decode GEMV family | gfx1151 | Fresh Qwen3.5-9B and Qwen3.6-35B-A3B profiles put standalone/residual/fused-QKV/fused-QKVZA/indexed-MoE GEMV at the top of decode time |
| 2 | HFQ4 grouped MMQ prefill | gfx1151 | Qwen3.6-35B-A3B pp128 spends 45.6% of profiled prefill in `gemm_hfq4g256_moe_grouped_mmq_k8_gfx1151`; existing 4w variant still loses |
| 3 | non-grouped HFQ4 MMQ prefill shapes | gfx1151 | K=2048 A3B/shared path routes to gfx1151 i8 WMMA, but dense K=4096 9B/122B still falls back because the gfx1151 route regresses those shapes |
| 4 | hc_sinkhorn_4x4 + hc_mix_4stream | gfx942 | Every forward pass layer in DeepSeek4; MFMA available but unused |
| 5 | GEMV decode (mq4g256, hfq4g256) | gfx12 | Only FP8 GEMV exists; MQ4 decode falls back to generic |
| 6 | deepseek4_attn_swa causal | gfx942, gfx12 | Hot path on MI300X / RX 9070; currently generic F32 |
| 7 | Qwen3.5/3.6 scalar cleanup | gfx1151 | `gated_norm`, `fused_silu_mul_mq_rotate`, and routed DeltaNet conv/gate BF16 policy remain open but are low single-digit rows in the current profile |
| 8 | fused_rmsnorm_mq_rotate BF16 | gfx1100, gfx12 | BF16 native on both; norm widening may help non-gfx1151 paths |
| 9 | indexer_relu_score wmma | gfx1151, gfx12 | MiniMax plaid-flash token routing; WMMA available |
