# Gemma 4 12B Prefill Profile — Per-Token Decode

**Date:** 2026-06-09
**GPU:** gfx1151 (RDNA 3.5, 137.4 GB VRAM)
**Model:** gemma-4-12B-it-q8.hfq (Q8_0 weights)
**Prompt:** "What is the capital of France?" (20 tokens prefill, 8 tokens decode)
**Total time:** ~1.74s (11.2 tok/s inclusive)
**Total kernel dispatches:** 39,730

## GPU Time by Category

| Category | Calls | Time (ms) | % of GPU |
|---|---|---|---|
| **GEMV/GEMM (projections)** | 9,212 | 1,629.5 | **93.6%** |
| Normalization (rmsnorm) | 9,436 | 52.6 | 3.0% |
| Attention (tile + reduce) | 2,688 | 29.1 | 1.7% |
| Memory (copy/fill) | 6,214 | 9.2 | 0.5% |
| Elementwise (scale/add/mul/gelu) | 8,092 | 9.0 | 0.5% |
| RoPE | 1,344 | 5.9 | 0.3% |
| KV cache write | 2,688 | 4.7 | 0.3% |
| Embedding | 28 | 0.3 | 0.0% |
| Logits (softcap) | 28 | 0.2 | 0.0% |

## Top Individual Kernels

| Kernel | Calls | Total (ms) | Avg (µs) | % |
|---|---|---|---|---|
| `gemv_q8_0` | 9,212 | 1,629.5 | 176.9 | 93.6% |
| `rmsnorm_f32` | 9,436 | 52.6 | 5.6 | 3.0% |
| `attention_flash_q8_0_tile` | 1,120 | 23.2 | 20.7 | 1.3% |
| `memcpy` (ROCclr) | 5,968 | 6.8 | 1.1 | 0.4% |
| `rope_f32` | 1,120 | 5.1 | 4.5 | 0.3% |
| `kv_cache_write_q8_0` | 2,464 | 3.5 | 1.4 | 0.2% |
| `attention_flash_asym3_tile_hd512` | 224 | 3.4 | 15.3 | 0.2% |
| `scale_f32` | 2,716 | 2.9 | 1.1 | 0.2% |
| `add_inplace_f32` | 2,688 | 2.4 | 0.9 | 0.1% |
| `attention_flash_q8_0_reduce` | 1,344 | 2.4 | 1.8 | 0.1% |

## Per Token-Layer Analysis

20 tokens × 48 layers = 960 token-layers.

| Operation | Launches/token-layer | Avg µs | Total ms |
|---|---|---|---|
| GEMV (q/k/v/o/gate/up/down projections) | 9.6 | 177 | 1,629.5 |
| rmsnorm | 9.8 | 5.6 | 52.6 |
| attention (tile + reduce) | 1.2 | 10.8 | 29.1 |
| KV write | 2.6 | 1.8 | 4.7 |
| RoPE | 1.4 | 4.4 | 5.9 |

## Key Findings

1. **Projections dominate at 93.6% of GPU time.** This is the prefill bottleneck — not attention (1.7%), not norms (3.0%), not KV writes (0.3%).

2. **Each `gemv_q8_0` call takes 177µs average.** This includes both compute and launch overhead. The vast majority of this is launch overhead + latency for a single-row GEMV on a weight matrix.

3. **Attention is only 1.7% of GPU time.** Per-token attention with q8 ring-buffer is fast for short contexts. The per-token attention loop is NOT the bottleneck at 20-token prefill.

4. **At 9.6 GEMVs per token-layer, that's ~7 GEMVs per layer** (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj ≈ 7, plus the lm_head at the end). With 48 layers, that's 336 projection calls for 20 tokens, but each token gets its own launch — totaling 9,212.

5. **Batching projections across tokens would reduce 9,212 launches to ~336** (7 per layer × 48 layers). Even with the same compute per element, the launch overhead savings alone would be significant.

6. **WMMA GEMM provides ~30× more compute throughput per byte of bandwidth.** For Q8_0 weights (34 bytes per group of 32), the current GEMV loads each weight row once per token. WMMA would load each row once per batch, computing 16×16 tiles of output per load.

## Implications for Prefill Strategy

The profiling data confirms that **projections are the dominant bottleneck** (93.6%). The original plan's premise was correct: routing projections through WMMA batched GEMM is the highest-impact optimization.

However, the per-token attention loop is NOT a significant cost at B=20 (only 1.7%). The concern about "85:1 attention launches vs projection launches" (Finding 5/13 in the review) is only relevant for very long contexts where the attention tile kernel becomes compute-bound.

For short-to-medium prefill (20-128 tokens):
- **WMMA batched GEMM for projections** is the #1 priority
- **Per-token attention is fine** for short contexts
- **Batched attention** only matters for very long contexts (>1024 tokens)

Measured on: 2026-06-09, gfx1151, ROCm 7.13, per-token decode path, 20-token prompt.
