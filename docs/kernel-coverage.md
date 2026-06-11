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
| gemv hfq4g256 | — | standard | **tuned** (v1–v5) | standard | **—** ⚠️ | — |
| gemv hfq4g256 multirow | — | — | — | standard | — | — |
| gemv hfq3g256 | — | — | — | standard | — | — |
| gemv mq4g256 lloyd | — | — | — | **tuned** (multiacc_diag) | — | — |
| gemv mq3g256 lloyd | — | — | — | standard | — | — |
| gemv hfp4g32 | — | — | — | basic | — | basic (fp8 path) |
| gemv hfq6g256 | — | — | — | — | — | basic |

**gfx1151 has no decode GEMV variants at all.** The UMA bandwidth budget (~89 GB/s) is far tighter than a discrete GPU; this is the single largest performance gap on the dev machine.

### GEMM — prefill projection (compute-bound)

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| gemm residual hfq4g256 | **tuned** (x8–x64) | **tuned** (mfma v1–v4) | **tuned** (mmq x8–x32) | — | standard (wmma mb4) | **tuned** (wmma) |
| gemm residual hfq3g256 | — | — | **tuned** (mmq x8–x32) | — | — | standard (wmma) |
| gemm residual hfq6g256 | — | — | — | — | — | standard (wmma) |
| gemm residual mq4g256 | — | — | — | — | standard (wmma) | standard (wmma) |
| gemm residual mq3g256 | — | — | — | — | — | standard (wmma) |
| gemm residual hfp4g32 | — | — | — | — | — | standard (wmma) |
| gemm residual q8_0 | — | — | — | — | — | standard (wmma) |
| gemm gate_up hfq4g256 | **tuned** (x8–x64) | standard | **tuned** (mmq x8–x32) | — | standard (wmma) | **tuned** (wmma) |
| gemm gate_up hfq3g256 | — | — | **tuned** (mmq+dp4a) | — | — | standard (wmma) |
| gemm gate_up mq4g256 | — | — | — | standard | **tuned** (wmma, mb4) | **tuned** (wmma) |
| gemm gate_up mq3g256 | — | — | — | standard | — | standard (wmma) |
| gemm gate_up hfp4g32 | — | — | — | — | — | standard (wmma) |
| gemm gate_up q8_0 | — | — | — | — | — | standard (wmma) |
| gemm qkv hfq4g256 | **tuned** (x8–x64) | standard | **tuned** (mmq x8–x32) | — | standard (wmma) | **tuned** (wmma) |
| gemm qkv hfq3g256 | — | — | **tuned** (mmq+dp4a) | — | — | standard (wmma) |
| gemm qkv mq4g256 | — | — | — | — | standard (wmma, mb4) | **tuned** (wmma) |
| gemm qkv mq3g256 | — | — | — | — | — | standard (wmma) |
| gemm qkv hfp4g32 | — | — | — | — | — | standard (wmma+fp8) |
| gemm lmhead hfq4g256 | — | — | — | — | — | basic |
| gemm bf16 | — | standard (mfma) | — | — | standard (wmma) | — |

### MoE grouped GEMM

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| moe grouped hfq4g256 | — | — | — | basic (gfx11_dgpu, 2 variants) | standard (mmq k4/k8) | **tuned** (wmma+mmq, m2 variant) |
| moe grouped hfq3g256 | — | — | — | — | basic | standard |
| moe grouped hfq6g256 | — | — | — | — | basic | standard (v2) |
| moe grouped mq4g256 | — | — | — | — | — | — |
| moe grouped paro q4g128 | — | — | — | — | standard (k8 variant) | — |
| moe grouped bf16/f16 | — | — | — | — | basic | — |
| moe scalar path | — | — | — | — | basic | — |
| moe hfp4g32 | — | — | — | — | — | standard |

### Fused QKV (prefill, quant decode into Q/K/V in one pass)

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| fused qkv hfq4g256 | — | standard (v2) | — | — | standard (wmma) | **tuned** |
| fused qkv mq4g256 | — | — | — | standard | standard (wmma, mb4) | **tuned** |
| fused qkv mq3g256 | — | — | — | standard | — | standard |
| fused qkv hfp4g32 | — | — | — | — | — | standard |
| fused qkv q8_0 | — | — | — | — | — | standard |

### Attention

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| dflash wmma v3 causal f16kv | — | — | — | — | — | standard |
| flash q8_0 dp4a | basic (2 variants) | — | — | — | — | — |

### Norm and Rotate

| Kernel group | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| rmsnorm | — | standard (reduce) | — | — | **—** ⚠️ | — |
| rotate_with_rms (fused) | — | standard | — | — | **—** ⚠️ | — |
| fused_rmsnorm_mq_rotate | generic only | generic only | generic only | generic only | **generic only** ⚠️ | generic only |

All norm kernels are F32-only. The `fused_rmsnorm_mq_rotate` family (called before every GEMV) has no arch-specific variants anywhere. This is highest priority for gfx1151 where BF16 widening overhead is not hidden by discrete VRAM bandwidth.

### Misc compute

| Kernel | gfx906 | gfx942 | gfx1030 | gfx1100 | gfx1151 | gfx12 |
|---|---|---|---|---|---|---|
| mq_rotate_x_dual | — | — | — | — | — | basic |
| pack_f32_to_fp8 | — | — | — | — | — | basic |
| gated_delta_net_q8 | — | — | — | — | — | basic (gfx1200 only) |

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
| conv1d_silu / split / routed / tree | — | — | — | — | **—** ⚠️ | — |
| fused_sigmoid_alpha_gate | — | — | — | — | **—** ⚠️ | — |
| alpha_gate | — | — | — | — | — | — |

gfx1151 is the primary deployment target for Qwen3.5. Every DeltaNet layer runs through conv1d + fused_sigmoid_alpha_gate using F32-only generic kernels.

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
| 1 | GEMV decode (mq4g256, hfq4g256) | gfx1151 | No GEMV at all on dev machine; falls back to generic F32 widen+compute |
| 2 | fused_rmsnorm_mq_rotate BF16 | gfx1151 | Called before every GEMV; widening weight overhead not hidden by UMA bandwidth |
| 3 | conv1d + fused_sigmoid_alpha_gate BF16 | gfx1151 | Every DeltaNet layer; Qwen3.5 is the primary gfx1151 model |
| 4 | hc_sinkhorn_4x4 + hc_mix_4stream | gfx942 | Every forward pass layer in DeepSeek4; MFMA available but unused |
| 5 | GEMV decode (mq4g256, hfq4g256) | gfx12 | Only FP8 GEMV exists; MQ4 decode falls back to generic |
| 6 | deepseek4_attn_swa causal | gfx942, gfx12 | Hot path on MI300X / RX 9070; currently generic F32 |
| 7 | fused_rmsnorm_mq_rotate BF16 | gfx1100, gfx12 | BF16 native on both; norm widening is free performance |
| 8 | indexer_relu_score wmma | gfx1151, gfx12 | MiniMax plaid-flash token routing; WMMA available |
